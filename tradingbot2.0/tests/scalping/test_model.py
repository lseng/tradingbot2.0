"""
Tests for the LightGBM scalping model.

Tests cover:
1. Model initialization and configuration
2. Training with validation
3. Prediction and confidence scores
4. Trading signal generation
5. Feature importance
6. Model save/load
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from scalping.model import (
        ScalpingModel,
        ModelConfig,
        TrainingResult,
        hyperparameter_search,
        LIGHTGBM_AVAILABLE,
    )
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Skip all tests if LightGBM not available
pytestmark = pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE,
    reason="LightGBM not installed"
)


@pytest.fixture
def sample_training_data():
    """Create sample training data for model tests."""
    np.random.seed(42)

    n_samples = 1000
    n_features = 24

    # Create features with some predictive power
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create target with some correlation to features
    # y = 1 if weighted sum of first 5 features > 0
    weights = np.array([0.3, 0.2, 0.15, 0.1, 0.05] + [0] * (n_features - 5))
    score = X @ weights + np.random.randn(n_samples) * 0.3
    y = (score > 0).astype(int)

    return X, y


@pytest.fixture
def sample_feature_names():
    """Create sample feature names."""
    return [
        "return_1bar", "return_3bar", "return_6bar", "return_12bar", "return_24bar",
        "close_vs_ema8", "close_vs_ema21", "close_vs_ema50", "close_vs_ema200",
        "rsi_7", "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_14", "bb_width", "bar_range",
        "volume_ratio_20", "volume_trend", "vwap_deviation",
        "time_of_day", "minutes_since_open", "is_first_hour", "is_last_hour",
    ]


@pytest.fixture
def trained_model(sample_training_data, sample_feature_names):
    """Create a trained model for testing."""
    X, y = sample_training_data

    # Split into train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = ScalpingModel()
    model.train(X_train, y_train, X_val, y_val, sample_feature_names)

    return model


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.objective == "binary"
        assert config.num_leaves == 31
        assert config.max_depth == 6
        assert config.learning_rate == 0.05
        assert config.min_confidence == 0.60

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            num_leaves=15,
            max_depth=4,
            learning_rate=0.1,
            min_confidence=0.70,
        )

        assert config.num_leaves == 15
        assert config.max_depth == 4
        assert config.learning_rate == 0.1
        assert config.min_confidence == 0.70

    def test_to_lgb_params(self):
        """Test conversion to LightGBM params dict."""
        config = ModelConfig()
        params = config.to_lgb_params()

        assert "objective" in params
        assert "num_leaves" in params
        assert "learning_rate" in params
        assert params["objective"] == "binary"


class TestScalpingModel:
    """Tests for ScalpingModel class."""

    def test_init_default(self):
        """Test model initialization with defaults."""
        model = ScalpingModel()

        assert model.config is not None
        assert model.model is None
        assert model.feature_names is None

    def test_init_custom_config(self):
        """Test model initialization with custom config."""
        config = ModelConfig(num_leaves=15)
        model = ScalpingModel(config=config)

        assert model.config.num_leaves == 15

    def test_train_basic(self, sample_training_data, sample_feature_names):
        """Test basic training."""
        X, y = sample_training_data
        model = ScalpingModel()

        result = model.train(X, y, feature_names=sample_feature_names)

        assert isinstance(result, TrainingResult)
        assert result.train_auc > 0.5  # Better than random
        assert model.model is not None

    def test_train_with_validation(self, sample_training_data, sample_feature_names):
        """Test training with validation set."""
        X, y = sample_training_data

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = ScalpingModel()
        result = model.train(X_train, y_train, X_val, y_val, sample_feature_names)

        assert result.val_auc > 0
        assert result.val_accuracy > 0
        assert result.best_iteration > 0

    def test_predict_proba(self, trained_model, sample_training_data):
        """Test probability prediction."""
        X, _ = sample_training_data
        proba = trained_model.predict_proba(X[:10])

        assert proba.shape == (10,)
        assert all(0 <= p <= 1 for p in proba)

    def test_predict_binary(self, trained_model, sample_training_data):
        """Test binary prediction."""
        X, _ = sample_training_data
        predictions = trained_model.predict(X[:10])

        assert predictions.shape == (10,)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_with_confidence(self, trained_model, sample_training_data):
        """Test prediction with confidence scores."""
        X, _ = sample_training_data
        predictions, confidences = trained_model.predict_with_confidence(X[:10])

        assert predictions.shape == (10,)
        assert confidences.shape == (10,)
        assert all(p in [0, 1] for p in predictions)
        assert all(0.5 <= c <= 1.0 for c in confidences)

    def test_get_trading_signals(self, trained_model, sample_training_data):
        """Test trading signal generation."""
        X, _ = sample_training_data
        signals, confidences, should_trade = trained_model.get_trading_signals(X[:10])

        assert signals.shape == (10,)
        assert confidences.shape == (10,)
        assert should_trade.shape == (10,)

        # Signals should be -1, 0, or 1
        assert all(s in [-1, 0, 1] for s in signals)

        # Should trade mask should be boolean
        assert should_trade.dtype == bool

        # Signals should be 0 where not confident enough
        for i, (s, st) in enumerate(zip(signals, should_trade)):
            if not st:
                assert s == 0

    def test_trading_signals_confidence_filter(self, trained_model, sample_training_data):
        """Test that confidence filter reduces trading signals."""
        X, _ = sample_training_data

        # With low threshold, should trade more
        _, _, should_trade_low = trained_model.get_trading_signals(X, min_confidence=0.5)

        # With high threshold, should trade less
        _, _, should_trade_high = trained_model.get_trading_signals(X, min_confidence=0.9)

        assert should_trade_low.sum() >= should_trade_high.sum()

    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        importance = trained_model.feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 24  # All features

        # Values should be percentages (sum to ~100)
        total = sum(importance.values())
        assert 99 < total < 101  # Allow small floating point error

    def test_untrained_model_predict_raises(self):
        """Test that prediction without training raises error."""
        model = ScalpingModel()

        with pytest.raises(ValueError, match="not trained"):
            model.predict_proba(np.random.randn(10, 24))

    def test_get_training_result(self, trained_model):
        """Test getting training result after training."""
        result = trained_model.get_training_result()

        assert result is not None
        assert isinstance(result, TrainingResult)
        assert result.train_auc > 0


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_and_load(self, trained_model, tmp_path, sample_training_data):
        """Test saving and loading model."""
        model_path = tmp_path / "test_model"
        trained_model.save(model_path)

        # Check files exist
        assert (tmp_path / "test_model.txt").exists()
        assert (tmp_path / "test_model.json").exists()

        # Load model
        loaded_model = ScalpingModel.load(model_path)

        # Test predictions match
        X, _ = sample_training_data
        original_pred = trained_model.predict_proba(X[:10])
        loaded_pred = loaded_model.predict_proba(X[:10])

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_load_preserves_config(self, trained_model, tmp_path):
        """Test that loaded model preserves configuration."""
        model_path = tmp_path / "test_model"
        trained_model.save(model_path)

        loaded_model = ScalpingModel.load(model_path)

        assert loaded_model.config.num_leaves == trained_model.config.num_leaves
        assert loaded_model.config.min_confidence == trained_model.config.min_confidence

    def test_load_preserves_feature_names(self, trained_model, tmp_path, sample_feature_names):
        """Test that loaded model preserves feature names."""
        model_path = tmp_path / "test_model"
        trained_model.save(model_path)

        loaded_model = ScalpingModel.load(model_path)

        assert loaded_model.feature_names == sample_feature_names

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            ScalpingModel.load(tmp_path / "nonexistent")


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_training_result_fields(self, trained_model):
        """Test TrainingResult has all expected fields."""
        result = trained_model.get_training_result()

        assert hasattr(result, "best_iteration")
        assert hasattr(result, "train_auc")
        assert hasattr(result, "val_auc")
        assert hasattr(result, "train_accuracy")
        assert hasattr(result, "val_accuracy")
        assert hasattr(result, "feature_importance")
        assert hasattr(result, "training_history")

    def test_training_history_has_metrics(self, sample_training_data, sample_feature_names):
        """Test training history tracks metrics."""
        X, y = sample_training_data

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = ScalpingModel()
        result = model.train(X_train, y_train, X_val, y_val, sample_feature_names)

        assert "train" in result.training_history
        assert "auc" in result.training_history["train"]


class TestHyperparameterSearch:
    """Tests for hyperparameter search."""

    def test_hyperparameter_search_basic(self, sample_training_data, sample_feature_names):
        """Test basic hyperparameter search (limited for speed)."""
        X, y = sample_training_data

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # This is slow, so we just verify it runs
        # In real use, reduce n_trials or use subset of param grid
        best_config, best_metrics = hyperparameter_search(
            X_train, y_train, X_val, y_val, sample_feature_names
        )

        assert best_config is not None
        assert isinstance(best_config, ModelConfig)
        assert "val_auc" in best_metrics
        assert best_metrics["val_auc"] > 0.5


class TestModelPerformance:
    """Tests for model performance characteristics."""

    def test_model_better_than_random(self, sample_training_data, sample_feature_names):
        """Test that model performs better than random guessing."""
        X, y = sample_training_data

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = ScalpingModel()
        result = model.train(X_train, y_train, X_val, y_val, sample_feature_names)

        # AUC > 0.5 means better than random
        assert result.val_auc > 0.5

    def test_inference_speed(self, trained_model, sample_training_data):
        """Test that inference is fast (<10ms per sample)."""
        import time

        X, _ = sample_training_data

        # Warmup
        trained_model.predict_proba(X[:1])

        # Time inference
        n_samples = 100
        start = time.time()
        for _ in range(n_samples):
            trained_model.predict_proba(X[:1])
        elapsed = time.time() - start

        avg_ms = (elapsed / n_samples) * 1000
        assert avg_ms < 10, f"Inference too slow: {avg_ms:.2f}ms per sample"

    def test_batch_inference_efficiency(self, trained_model, sample_training_data):
        """Test that batch inference is more efficient than individual."""
        import time

        X, _ = sample_training_data

        # Individual inference
        start = time.time()
        for i in range(100):
            trained_model.predict_proba(X[i:i+1])
        individual_time = time.time() - start

        # Batch inference
        start = time.time()
        trained_model.predict_proba(X[:100])
        batch_time = time.time() - start

        # Batch should be faster
        assert batch_time < individual_time


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_train_with_pandas_dataframe(self, sample_training_data, sample_feature_names):
        """Test training with pandas DataFrame input."""
        X, y = sample_training_data

        X_df = pd.DataFrame(X, columns=sample_feature_names)
        y_series = pd.Series(y)

        model = ScalpingModel()
        result = model.train(X_df, y_series)

        assert result.train_auc > 0.5

    def test_predict_with_single_sample(self, trained_model):
        """Test prediction with single sample."""
        X_single = np.random.randn(1, 24).astype(np.float32)
        proba = trained_model.predict_proba(X_single)

        assert proba.shape == (1,)

    def test_all_same_class_training(self, sample_feature_names):
        """Test training when all samples have same class."""
        X = np.random.randn(100, 24).astype(np.float32)
        y = np.ones(100, dtype=int)  # All class 1

        model = ScalpingModel()
        # Should still train without error
        result = model.train(X, y, feature_names=sample_feature_names)

        assert result is not None

    def test_extreme_confidence_threshold(self, trained_model, sample_training_data):
        """Test trading signals with extreme confidence thresholds."""
        X, _ = sample_training_data

        # Very low threshold - almost all should trade
        _, _, should_trade_low = trained_model.get_trading_signals(X, min_confidence=0.50001)

        # Very high threshold - almost none should trade
        _, _, should_trade_high = trained_model.get_trading_signals(X, min_confidence=0.9999)

        assert should_trade_high.sum() <= should_trade_low.sum()
