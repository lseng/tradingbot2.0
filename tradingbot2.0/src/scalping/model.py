"""
LightGBM Model for 5-Minute Scalping System

Binary classification model for predicting price direction (UP vs DOWN).
Uses gradient boosted trees which are:
- More robust to overfitting than neural networks
- Faster for inference (<10ms requirement)
- More interpretable via feature importance

Why LightGBM over XGBoost:
- Faster training with large datasets
- Native support for categorical features
- Lower memory usage
- Comparable accuracy

The spec explicitly recommends gradient boosted trees over neural networks
after the previous neural network approach failed to achieve profitability.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LightGBM model."""

    # Core parameters
    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"

    # Tree parameters
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 100

    # Learning parameters
    learning_rate: float = 0.05
    n_estimators: int = 1000
    early_stopping_rounds: int = 50

    # Regularization
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0

    # Other
    verbose: int = -1
    random_state: int = 42
    n_jobs: int = -1

    # Confidence threshold for trading
    min_confidence: float = 0.60

    def to_lgb_params(self) -> dict:
        """Convert to LightGBM parameters dict."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass
class TrainingResult:
    """Result of model training."""

    best_iteration: int
    train_auc: float
    val_auc: float
    train_accuracy: float
    val_accuracy: float
    feature_importance: Dict[str, float]
    training_history: Dict[str, List[float]]


class ScalpingModel:
    """
    LightGBM classifier for 5-minute scalping.

    Predicts probability of price going UP (above threshold) in next N bars.
    Uses confidence filtering to only trade when model is confident.

    Example usage:
        model = ScalpingModel()
        model.train(X_train, y_train, X_val, y_val)

        # Get probability and confidence
        prob = model.predict_proba(X_new)
        confidence = max(prob, 1 - prob)

        # Only trade if confident
        if confidence >= 0.60:
            direction = 1 if prob > 0.5 else -1
            execute_trade(direction)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize model.

        Args:
            config: Model configuration (uses defaults if None)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm>=4.0.0"
            )

        self.config = config or ModelConfig()
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self._training_result: Optional[TrainingResult] = None

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """
        Train LightGBM model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (binary: 0 or 1)
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            feature_names: Names of features (for importance tracking)

        Returns:
            TrainingResult with metrics and history
        """
        logger.info(f"Training LightGBM model on {len(X_train):,} samples")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Training history
        evals_result: Dict[str, Dict[str, List[float]]] = {}

        # Train model
        params = self.config.to_lgb_params()

        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(evals_result),
        ]

        if X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=True,
                )
            )

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Get best iteration
        best_iteration = self.model.best_iteration or self.model.num_trees()

        # Calculate metrics
        train_pred = self.predict_proba(X_train)
        train_pred_binary = (train_pred > 0.5).astype(int)
        train_accuracy = (train_pred_binary == y_train).mean()

        # Get AUC from history
        train_auc = evals_result.get("train", {}).get("auc", [0])[-1]

        val_auc = 0.0
        val_accuracy = 0.0
        if X_val is not None:
            val_pred = self.predict_proba(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)
            val_accuracy = (val_pred_binary == y_val).mean()
            val_auc = evals_result.get("valid", {}).get("auc", [0])[-1]

        # Get feature importance
        importance = self.feature_importance()

        # Create training result
        self._training_result = TrainingResult(
            best_iteration=best_iteration,
            train_auc=train_auc,
            val_auc=val_auc,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            feature_importance=importance,
            training_history=evals_result,
        )

        logger.info(f"Training complete. Best iteration: {best_iteration}")
        logger.info(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return self._training_result

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict probability of class 1 (UP).

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Array of probabilities (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # LightGBM returns probability of positive class for binary classification
        proba = self.model.predict(X)
        return proba

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict binary class.

        Args:
            X: Features (n_samples, n_features)
            threshold: Probability threshold for class 1

        Returns:
            Array of predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def predict_with_confidence(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Tuple of (predictions, confidences)
            - predictions: 1 for UP, 0 for DOWN
            - confidences: Distance from 0.5 (0.5 to 1.0 range)
        """
        proba = self.predict_proba(X)

        # Prediction is 1 if proba > 0.5
        predictions = (proba > 0.5).astype(int)

        # Confidence is how far from 0.5 we are
        # If proba=0.7, confidence=0.7. If proba=0.3, confidence=0.7 (for DOWN)
        confidences = np.where(proba > 0.5, proba, 1 - proba)

        return predictions, confidences

    def get_trading_signals(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        min_confidence: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get trading signals with confidence filtering.

        Args:
            X: Features (n_samples, n_features)
            min_confidence: Minimum confidence to generate signal (default: config value)

        Returns:
            Tuple of (signals, confidences, should_trade)
            - signals: 1 for LONG, -1 for SHORT, 0 for FLAT
            - confidences: Confidence scores (0.5 to 1.0)
            - should_trade: Boolean mask where confidence >= min_confidence
        """
        if min_confidence is None:
            min_confidence = self.config.min_confidence

        predictions, confidences = self.predict_with_confidence(X)

        # Convert to trading signals: 1=LONG, -1=SHORT
        signals = np.where(predictions == 1, 1, -1)

        # Should trade only if confident enough
        should_trade = confidences >= min_confidence

        # Set signals to 0 where not confident
        signals = np.where(should_trade, signals, 0)

        return signals, confidences, should_trade

    def feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'split', or 'shap')

        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if importance_type == "shap":
            raise NotImplementedError("SHAP importance not implemented")

        importance = self.model.feature_importance(importance_type=importance_type)

        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(len(importance))]
        else:
            names = self.feature_names

        # Normalize to percentages
        total = sum(importance)
        if total > 0:
            importance = [i / total * 100 for i in importance]

        return dict(sorted(zip(names, importance), key=lambda x: -x[1]))

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model (without extension)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_path = path.with_suffix(".txt")
        self.model.save_model(str(model_path))

        # Save config and metadata
        metadata = {
            "config": {
                "objective": self.config.objective,
                "metric": self.config.metric,
                "num_leaves": self.config.num_leaves,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "min_confidence": self.config.min_confidence,
            },
            "feature_names": self.feature_names,
            "training_result": {
                "best_iteration": self._training_result.best_iteration if self._training_result else None,
                "train_auc": self._training_result.train_auc if self._training_result else None,
                "val_auc": self._training_result.val_auc if self._training_result else None,
            } if self._training_result else None,
        }

        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScalpingModel":
        """
        Load model from file.

        Args:
            path: Path to model file (without extension)

        Returns:
            Loaded ScalpingModel instance
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        path = Path(path)

        # Load LightGBM model
        model_path = path.with_suffix(".txt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        booster = lgb.Booster(model_file=str(model_path))

        # Load metadata
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            config_dict = metadata.get("config", {})
            feature_names = metadata.get("feature_names")
        else:
            config_dict = {}
            feature_names = None

        # Create config
        config = ModelConfig(
            num_leaves=config_dict.get("num_leaves", 31),
            max_depth=config_dict.get("max_depth", 6),
            learning_rate=config_dict.get("learning_rate", 0.05),
            min_confidence=config_dict.get("min_confidence", 0.60),
        )

        # Create instance
        instance = cls(config=config)
        instance.model = booster
        instance.feature_names = feature_names

        logger.info(f"Model loaded from {model_path}")

        return instance

    def get_training_result(self) -> Optional[TrainingResult]:
        """Get training result from last training run."""
        return self._training_result


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_trials: int = 20,
) -> Tuple[ModelConfig, Dict[str, float]]:
    """
    Simple grid search for hyperparameter tuning.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: Feature names
        n_trials: Not used (grid search is exhaustive)

    Returns:
        Tuple of (best_config, best_metrics)
    """
    logger.info("Starting hyperparameter search")

    # Parameter grid (from spec)
    param_grid = {
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_data_in_leaf": [50, 100, 200],
    }

    best_auc = 0.0
    best_config = None
    best_metrics = {}

    # Grid search
    from itertools import product

    param_combinations = list(product(
        param_grid["num_leaves"],
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["min_data_in_leaf"],
    ))

    logger.info(f"Testing {len(param_combinations)} parameter combinations")

    for i, (num_leaves, max_depth, lr, min_data) in enumerate(param_combinations):
        config = ModelConfig(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=lr,
            min_data_in_leaf=min_data,
            n_estimators=500,  # Reduced for speed
            early_stopping_rounds=20,
        )

        model = ScalpingModel(config=config)
        result = model.train(X_train, y_train, X_val, y_val, feature_names)

        if result.val_auc > best_auc:
            best_auc = result.val_auc
            best_config = config
            best_metrics = {
                "val_auc": result.val_auc,
                "val_accuracy": result.val_accuracy,
                "train_auc": result.train_auc,
                "best_iteration": result.best_iteration,
            }
            logger.info(
                f"New best: AUC={best_auc:.4f} "
                f"(leaves={num_leaves}, depth={max_depth}, lr={lr}, min_data={min_data})"
            )

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{len(param_combinations)} trials")

    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info(f"Best config: {best_config}")

    return best_config, best_metrics
