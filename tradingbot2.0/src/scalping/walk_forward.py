"""
Walk-Forward Cross-Validation for 5-Minute Scalping System

Implements expanding window walk-forward validation on the training set (2019-2022)
to tune hyperparameters and validate model stability without touching the test set.

Why walk-forward instead of k-fold:
- Time series data has temporal dependencies
- Traditional k-fold would leak future information into training
- Walk-forward respects chronological ordering
- Better simulates real-world deployment (always train on past, predict future)

From spec: "Use 5 folds: train on months 1-N, validate on month N+1"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .model import ScalpingModel, ModelConfig, TrainingResult

logger = logging.getLogger(__name__)

# Constants
NY_TZ = ZoneInfo("America/New_York")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward cross-validation."""

    # Number of folds (validation windows)
    n_folds: int = 5

    # Minimum training size (in months) before first validation
    min_train_months: int = 12

    # Size of each validation window (in months)
    val_months: int = 6

    # Whether to use expanding window (True) or rolling window (False)
    expanding: bool = True

    # If rolling, size of training window in months
    rolling_train_months: int = 24

    # Model configuration to use (None = default)
    model_config: Optional[ModelConfig] = None

    # Verbose logging
    verbose: int = 1


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_idx: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime

    # Sample counts
    n_train_samples: int
    n_val_samples: int

    # Performance metrics
    train_auc: float
    val_auc: float
    train_accuracy: float
    val_accuracy: float

    # Calibration metrics
    val_brier_score: float  # Lower is better (0 = perfect)
    val_expected_calibration_error: float  # ECE

    # Model info
    best_iteration: int
    feature_importance: Dict[str, float]

    # Overfitting indicator: train_auc - val_auc (large = overfitting)
    @property
    def overfit_score(self) -> float:
        return self.train_auc - self.val_auc


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""

    fold_results: List[FoldResult]
    config: WalkForwardConfig

    # Best model config (if hyperparameter search was done)
    best_config: Optional[ModelConfig] = None

    # Aggregated metrics (mean across folds)
    @property
    def mean_val_auc(self) -> float:
        return np.mean([f.val_auc for f in self.fold_results])

    @property
    def std_val_auc(self) -> float:
        return np.std([f.val_auc for f in self.fold_results])

    @property
    def mean_val_accuracy(self) -> float:
        return np.mean([f.val_accuracy for f in self.fold_results])

    @property
    def std_val_accuracy(self) -> float:
        return np.std([f.val_accuracy for f in self.fold_results])

    @property
    def mean_val_brier(self) -> float:
        return np.mean([f.val_brier_score for f in self.fold_results])

    @property
    def mean_val_ece(self) -> float:
        return np.mean([f.val_expected_calibration_error for f in self.fold_results])

    @property
    def mean_overfit_score(self) -> float:
        return np.mean([f.overfit_score for f in self.fold_results])

    @property
    def is_stable(self) -> bool:
        """Check if model performance is stable across folds.

        Criteria:
        - Std of val AUC < 0.03 (3% variation)
        - Mean overfit score < 0.05 (not too much overfitting)
        - All folds have val AUC > 0.52 (better than random)
        """
        return (
            self.std_val_auc < 0.03 and
            self.mean_overfit_score < 0.05 and
            all(f.val_auc > 0.52 for f in self.fold_results)
        )

    def summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward results."""
        return {
            "n_folds": len(self.fold_results),
            "mean_val_auc": self.mean_val_auc,
            "std_val_auc": self.std_val_auc,
            "mean_val_accuracy": self.mean_val_accuracy,
            "std_val_accuracy": self.std_val_accuracy,
            "mean_val_brier": self.mean_val_brier,
            "mean_val_ece": self.mean_val_ece,
            "mean_overfit_score": self.mean_overfit_score,
            "is_stable": self.is_stable,
            "per_fold_auc": [f.val_auc for f in self.fold_results],
        }


class WalkForwardCV:
    """
    Walk-forward cross-validation for LightGBM scalping model.

    Implements expanding window validation on the training period (2019-2022)
    to tune hyperparameters and assess model stability.

    Example usage:
        cv = WalkForwardCV(config=WalkForwardConfig(n_folds=5))
        result = cv.run(X_train, y_train, feature_names=feature_cols)

        if result.is_stable:
            # Train final model on full training set
            final_model = ScalpingModel(config=result.best_config)
            final_model.train(X_train, y_train, X_val, y_val, feature_names)
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize walk-forward CV.

        Args:
            config: Walk-forward configuration (uses defaults if None)
        """
        self.config = config or WalkForwardConfig()
        self.fold_results: List[FoldResult] = []

    def generate_folds(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, datetime, datetime, datetime, datetime]]:
        """
        Generate expanding window folds from dataframe with datetime index.

        Args:
            df: DataFrame with datetime index (features + target)

        Returns:
            List of (train_df, val_df, train_start, train_end, val_start, val_end) tuples
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Get date range
        start_date = df.index.min()
        end_date = df.index.max()
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        logger.info(f"Generating {self.config.n_folds} folds from {total_months} months of data")
        logger.info(f"Data range: {start_date.date()} to {end_date.date()}")

        folds = []

        # Calculate validation window start points
        # Leave min_train_months for initial training, then space out val windows
        available_months = total_months - self.config.min_train_months
        if available_months < self.config.n_folds * self.config.val_months:
            raise ValueError(
                f"Not enough data for {self.config.n_folds} folds with "
                f"{self.config.val_months} months validation each. "
                f"Have {available_months} months, need {self.config.n_folds * self.config.val_months}"
            )

        # Calculate step size between fold starts
        step_months = (available_months - self.config.val_months) // max(self.config.n_folds - 1, 1)

        for i in range(self.config.n_folds):
            # Validation period starts after min_train_months + (i * step_months)
            val_start_offset = self.config.min_train_months + (i * step_months)
            val_start = start_date + pd.DateOffset(months=val_start_offset)
            val_end = val_start + pd.DateOffset(months=self.config.val_months) - pd.Timedelta(seconds=1)

            # Training period
            if self.config.expanding:
                # Expanding window: train from start to val_start
                train_start = start_date
            else:
                # Rolling window: train from (val_start - rolling_train_months) to val_start
                train_start = val_start - pd.DateOffset(months=self.config.rolling_train_months)
                if train_start < start_date:
                    train_start = start_date

            train_end = val_start - pd.Timedelta(seconds=1)

            # Filter data
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            val_mask = (df.index >= val_start) & (df.index <= val_end)

            train_df = df[train_mask]
            val_df = df[val_mask]

            if len(train_df) == 0 or len(val_df) == 0:
                logger.warning(f"Fold {i}: Empty train or val set, skipping")
                continue

            folds.append((
                train_df, val_df,
                train_start, train_end,
                val_start, val_end
            ))

            logger.info(
                f"Fold {i}: Train {train_start.date()} to {train_end.date()} "
                f"({len(train_df):,} samples), "
                f"Val {val_start.date()} to {val_end.date()} "
                f"({len(val_df):,} samples)"
            )

        return folds

    def generate_folds_from_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        lazy: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, datetime, datetime, datetime, datetime]]:
        """
        Generate folds from numpy arrays with timestamps.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            timestamps: DatetimeIndex aligned with X and y
            lazy: If True, return a generator to reduce memory usage (Bug #13 fix).
                  When lazy=True, folds are generated on-the-fly and only one fold's
                  data is in memory at a time. This reduces peak memory from
                  O(n_folds * data_size) to O(data_size).

        Returns:
            If lazy=False: List of (X_train, y_train, X_val, y_val, train_start, train_end, val_start, val_end) tuples
            If lazy=True: Generator yielding the same tuples one at a time
        """
        if lazy:
            return self._generate_folds_from_arrays_lazy(X, y, timestamps)

        # Create temporary DataFrame for fold generation
        temp_df = pd.DataFrame(index=timestamps)
        temp_df['_target'] = y

        # Generate folds using DataFrame method
        df_folds = self.generate_folds(temp_df)

        # Convert to array folds
        array_folds = []
        for train_df, val_df, train_start, train_end, val_start, val_end in df_folds:
            # Get indices in original arrays
            train_idx = np.isin(timestamps, train_df.index)
            val_idx = np.isin(timestamps, val_df.index)

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            array_folds.append((
                X_train, y_train, X_val, y_val,
                train_start, train_end, val_start, val_end
            ))

        return array_folds

    def _generate_folds_from_arrays_lazy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ):
        """
        Memory-efficient generator that yields folds one at a time.

        This is the Bug #13 fix: instead of materializing all folds upfront
        (which uses O(n_folds * data_size) memory), this generator creates
        each fold on demand, using only O(data_size) memory at any time.

        The generator yields:
            Tuple of (X_train, y_train, X_val, y_val, train_start, train_end, val_start, val_end)

        Memory savings example:
            - 5 folds, 1M samples each, 56 features, float32
            - Eager: 5 * 1M * 56 * 4 bytes * 2 (train+val) ≈ 2.2 GB
            - Lazy: 1M * 56 * 4 bytes * 2 ≈ 0.45 GB (per fold, then released)
        """
        # Generate fold boundaries without materializing data
        fold_boundaries = self._generate_fold_boundaries(timestamps)

        for train_start, train_end, val_start, val_end in fold_boundaries:
            # Create masks for this fold only
            train_mask = (timestamps >= train_start) & (timestamps <= train_end)
            val_mask = (timestamps >= val_start) & (timestamps <= val_end)

            # Slice data for this fold
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning(
                    f"Empty train or val set for period {train_start.date()} - {val_end.date()}, skipping"
                )
                continue

            yield (
                X_train, y_train, X_val, y_val,
                train_start, train_end, val_start, val_end
            )

    def _generate_fold_boundaries(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate fold boundaries (timestamps only) without materializing any data.

        This is a lightweight operation that only calculates the time boundaries
        for each fold, without copying any actual data arrays.

        Args:
            timestamps: DatetimeIndex to determine data range

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        # Get date range
        start_date = timestamps.min()
        end_date = timestamps.max()
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        logger.debug(
            f"Generating {self.config.n_folds} fold boundaries from {total_months} months of data"
        )

        boundaries = []

        # Calculate validation window start points
        available_months = total_months - self.config.min_train_months
        if available_months < self.config.n_folds * self.config.val_months:
            raise ValueError(
                f"Not enough data for {self.config.n_folds} folds with "
                f"{self.config.val_months} months validation each. "
                f"Have {available_months} months, need {self.config.n_folds * self.config.val_months}"
            )

        # Calculate step size between fold starts
        step_months = (available_months - self.config.val_months) // max(self.config.n_folds - 1, 1)

        for i in range(self.config.n_folds):
            # Validation period starts after min_train_months + (i * step_months)
            val_start_offset = self.config.min_train_months + (i * step_months)
            val_start = start_date + pd.DateOffset(months=val_start_offset)
            val_end = val_start + pd.DateOffset(months=self.config.val_months) - pd.Timedelta(seconds=1)

            # Training period
            if self.config.expanding:
                train_start = start_date
            else:
                train_start = val_start - pd.DateOffset(months=self.config.rolling_train_months)
                if train_start < start_date:
                    train_start = start_date

            train_end = val_start - pd.Timedelta(seconds=1)

            boundaries.append((train_start, train_end, val_start, val_end))

            logger.debug(
                f"Fold {i} boundaries: Train {train_start.date()} to {train_end.date()}, "
                f"Val {val_start.date()} to {val_end.date()}"
            )

        return boundaries

    def calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, float]:
        """
        Calculate calibration metrics: Brier score and Expected Calibration Error.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for ECE calculation

        Returns:
            Tuple of (brier_score, ece)

        Why calibration matters:
        - A well-calibrated model's 70% confidence predictions should be correct 70% of the time
        - Poor calibration means confidence filtering (60% threshold) won't work as expected
        - Brier score measures overall probability accuracy
        - ECE measures calibration quality across confidence levels
        """
        # Brier score: mean squared error of probabilities
        brier = np.mean((y_prob - y_true) ** 2)

        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(y_true)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                # Average predicted probability in bin
                avg_confidence = np.mean(y_prob[in_bin])
                # Actual accuracy in bin
                avg_accuracy = np.mean(y_true[in_bin])
                # Contribution to ECE
                ece += (n_in_bin / total_samples) * abs(avg_accuracy - avg_confidence)

        return brier, ece

    def train_fold(
        self,
        fold_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_start: datetime,
        train_end: datetime,
        val_start: datetime,
        val_end: datetime,
        feature_names: Optional[List[str]] = None,
    ) -> FoldResult:
        """
        Train model on a single fold and compute metrics.

        Args:
            fold_idx: Fold index (0-based)
            X_train, y_train: Training data
            X_val, y_val: Validation data
            train_start/end: Training period boundaries
            val_start/end: Validation period boundaries
            feature_names: Feature names for importance tracking

        Returns:
            FoldResult with metrics
        """
        logger.info(f"Training fold {fold_idx} ({len(X_train):,} train, {len(X_val):,} val samples)")

        # Create and train model
        model_config = self.config.model_config or ModelConfig()
        model = ScalpingModel(config=model_config)

        training_result = model.train(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names,
        )

        # Get predictions on validation set
        val_probs = model.predict_proba(X_val)
        val_preds = (val_probs > 0.5).astype(int)
        val_accuracy = (val_preds == y_val).mean()

        # Calculate calibration metrics
        brier, ece = self.calculate_calibration_metrics(y_val, val_probs)

        fold_result = FoldResult(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val),
            train_auc=training_result.train_auc,
            val_auc=training_result.val_auc,
            train_accuracy=training_result.train_accuracy,
            val_accuracy=val_accuracy,
            val_brier_score=brier,
            val_expected_calibration_error=ece,
            best_iteration=training_result.best_iteration,
            feature_importance=training_result.feature_importance,
        )

        if self.config.verbose >= 1:
            logger.info(
                f"Fold {fold_idx} complete: "
                f"Val AUC={fold_result.val_auc:.4f}, "
                f"Val Acc={fold_result.val_accuracy:.4f}, "
                f"Brier={fold_result.val_brier_score:.4f}, "
                f"ECE={fold_result.val_expected_calibration_error:.4f}, "
                f"Overfit={fold_result.overfit_score:.4f}"
            )

        return fold_result

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        feature_names: Optional[List[str]] = None,
        use_lazy_folds: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward cross-validation.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            timestamps: DatetimeIndex aligned with X and y
            feature_names: Feature names for importance tracking
            use_lazy_folds: If True (default), use memory-efficient lazy fold generation.
                           This reduces peak memory from O(n_folds * data_size) to O(data_size)
                           by generating each fold on-the-fly instead of materializing all folds
                           upfront. Set to False only for debugging or when memory is not a concern.
                           (Bug #13 fix)

        Returns:
            WalkForwardResult with all fold results and aggregated metrics
        """
        logger.info(
            f"Starting walk-forward CV: {self.config.n_folds} folds, "
            f"{'expanding' if self.config.expanding else 'rolling'} window"
            f"{', lazy fold generation' if use_lazy_folds else ''}"
        )

        # Generate folds - use lazy generation by default to avoid OOM (Bug #13)
        folds = self.generate_folds_from_arrays(X, y, timestamps, lazy=use_lazy_folds)

        # For lazy folds, we need to count them after iteration
        # For eager folds, we can check length immediately
        if not use_lazy_folds:
            if len(folds) == 0:
                raise ValueError("No valid folds generated. Check data size and config.")
            logger.info(f"Generated {len(folds)} folds")

        # Train each fold
        # Note: With lazy=True, folds is a generator. Memory for each fold is allocated
        # and released during iteration, reducing peak memory usage (Bug #13 fix).
        self.fold_results = []
        fold_count = 0
        for i, (X_train, y_train, X_val, y_val, train_start, train_end, val_start, val_end) in enumerate(folds):
            fold_count += 1
            fold_result = self.train_fold(
                fold_idx=i,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                feature_names=feature_names,
            )
            self.fold_results.append(fold_result)
            # With lazy folds, data from previous fold is released after each iteration
            # This is the key memory optimization from Bug #13 fix

        # Validate that we got at least one fold (important for lazy case)
        if fold_count == 0:
            raise ValueError("No valid folds generated. Check data size and config.")

        if use_lazy_folds:
            logger.info(f"Processed {fold_count} folds (lazy generation)")

        result = WalkForwardResult(
            fold_results=self.fold_results,
            config=self.config,
            best_config=self.config.model_config,
        )

        # Log summary
        logger.info("Walk-forward CV complete!")
        logger.info(f"Mean Val AUC: {result.mean_val_auc:.4f} (+/- {result.std_val_auc:.4f})")
        logger.info(f"Mean Val Accuracy: {result.mean_val_accuracy:.4f} (+/- {result.std_val_accuracy:.4f})")
        logger.info(f"Mean Brier Score: {result.mean_val_brier:.4f}")
        logger.info(f"Mean ECE: {result.mean_val_ece:.4f}")
        logger.info(f"Mean Overfit Score: {result.mean_overfit_score:.4f}")
        logger.info(f"Is Stable: {result.is_stable}")

        return result

    def run_with_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        use_lazy_folds: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward CV directly from DataFrame.

        Args:
            df: DataFrame with datetime index, feature columns, and target column
            feature_cols: List of feature column names
            target_col: Target column name
            use_lazy_folds: If True (default), use memory-efficient lazy fold generation.
                           (Bug #13 fix)

        Returns:
            WalkForwardResult
        """
        X = df[feature_cols].values
        y = df[target_col].values
        timestamps = df.index

        return self.run(X, y, timestamps, feature_names=feature_cols, use_lazy_folds=use_lazy_folds)


def run_walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    feature_names: Optional[List[str]] = None,
    n_folds: int = 5,
    min_train_months: int = 12,
    val_months: int = 6,
    expanding: bool = True,
    model_config: Optional[ModelConfig] = None,
    verbose: int = 1,
    use_lazy_folds: bool = True,
) -> WalkForwardResult:
    """
    Convenience function for walk-forward validation.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        timestamps: DatetimeIndex aligned with X and y
        feature_names: Feature names for importance tracking
        n_folds: Number of folds
        min_train_months: Minimum training months before first validation
        val_months: Months in each validation window
        expanding: Use expanding (True) or rolling (False) window
        model_config: Model configuration
        verbose: Verbosity level
        use_lazy_folds: If True (default), use memory-efficient lazy fold generation.
                       This reduces peak memory from O(n_folds * data_size) to O(data_size).
                       (Bug #13 fix)

    Returns:
        WalkForwardResult

    Example:
        result = run_walk_forward_validation(
            X_train, y_train, train_timestamps,
            feature_names=feature_cols,
            n_folds=5,
        )
        print(f"Mean AUC: {result.mean_val_auc:.4f}")
    """
    config = WalkForwardConfig(
        n_folds=n_folds,
        min_train_months=min_train_months,
        val_months=val_months,
        expanding=expanding,
        model_config=model_config,
        verbose=verbose,
    )

    cv = WalkForwardCV(config=config)
    return cv.run(X, y, timestamps, feature_names=feature_names, use_lazy_folds=use_lazy_folds)
