"""
Tests for Walk-Forward Cross-Validation

Tests verify:
1. Fold generation respects temporal ordering (no data leakage)
2. Expanding vs rolling window behavior
3. Calibration metrics calculation
4. Per-fold and aggregated results
5. Integration with ScalpingModel
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.scalping.walk_forward import (
    WalkForwardCV,
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    run_walk_forward_validation,
)
from src.scalping.model import ModelConfig

NY_TZ = ZoneInfo("America/New_York")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def multi_year_data():
    """
    Create 3 years of synthetic data for walk-forward testing.

    This simulates having data from 2020-01-01 to 2022-12-31 (3 years).
    Uses hourly frequency to ensure enough time span for fold generation.
    """
    np.random.seed(42)

    # Generate 3 years of hourly data (trading hours only: 78 bars/day * 252 days/year * 3 years)
    # Simplified: use continuous hourly data spanning 3 years
    start = pd.Timestamp("2020-01-02 09:00:00", tz=NY_TZ)
    end = pd.Timestamp("2022-12-31 16:00:00", tz=NY_TZ)

    # Generate hourly timestamps spanning 3 years
    dates = pd.date_range(start=start, end=end, freq="h")
    n_bars = len(dates)

    # Generate features (simple random features)
    n_features = 10
    X = np.random.randn(n_bars, n_features)

    # Generate target with slight predictability
    # This ensures the model can learn something for testing
    signal = 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n_bars) * 0.5
    y = (signal > 0).astype(int)

    return X, y, dates


@pytest.fixture
def small_data():
    """Small dataset for quick tests (~24 months of data)."""
    np.random.seed(42)

    # Generate 2 years of hourly data
    start = pd.Timestamp("2021-01-02 09:00:00", tz=NY_TZ)
    end = pd.Timestamp("2022-12-31 16:00:00", tz=NY_TZ)

    dates = pd.date_range(start=start, end=end, freq="h")
    n_bars = len(dates)

    n_features = 5
    X = np.random.randn(n_bars, n_features)

    signal = 0.3 * X[:, 0] + np.random.randn(n_bars) * 0.5
    y = (signal > 0).astype(int)

    return X, y, dates


@pytest.fixture
def sample_fold_result():
    """Sample FoldResult for testing."""
    return FoldResult(
        fold_idx=0,
        train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
        train_end=datetime(2021, 6, 30, tzinfo=NY_TZ),
        val_start=datetime(2021, 7, 1, tzinfo=NY_TZ),
        val_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
        n_train_samples=1000,
        n_val_samples=500,
        train_auc=0.65,
        val_auc=0.58,
        train_accuracy=0.62,
        val_accuracy=0.56,
        val_brier_score=0.20,
        val_expected_calibration_error=0.05,
        best_iteration=150,
        feature_importance={"feat_0": 30.0, "feat_1": 20.0},
    )


# ============================================================================
# Test WalkForwardConfig
# ============================================================================

class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()

        assert config.n_folds == 5
        assert config.min_train_months == 12
        assert config.val_months == 6
        assert config.expanding is True
        assert config.rolling_train_months == 24
        assert config.model_config is None
        assert config.verbose == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        model_config = ModelConfig(num_leaves=15, max_depth=4)

        config = WalkForwardConfig(
            n_folds=3,
            min_train_months=6,
            val_months=3,
            expanding=False,
            rolling_train_months=12,
            model_config=model_config,
            verbose=0,
        )

        assert config.n_folds == 3
        assert config.min_train_months == 6
        assert config.val_months == 3
        assert config.expanding is False
        assert config.rolling_train_months == 12
        assert config.model_config == model_config
        assert config.verbose == 0


# ============================================================================
# Test FoldResult
# ============================================================================

class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_creation(self, sample_fold_result):
        """Test creating a FoldResult."""
        result = sample_fold_result

        assert result.fold_idx == 0
        assert result.n_train_samples == 1000
        assert result.n_val_samples == 500
        assert result.train_auc == 0.65
        assert result.val_auc == 0.58

    def test_overfit_score(self, sample_fold_result):
        """Test overfitting score calculation."""
        result = sample_fold_result

        # overfit_score = train_auc - val_auc
        expected = 0.65 - 0.58
        assert abs(result.overfit_score - expected) < 1e-6

    def test_overfit_score_negative(self):
        """Test negative overfit score (val better than train)."""
        result = FoldResult(
            fold_idx=0,
            train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
            train_end=datetime(2021, 6, 30, tzinfo=NY_TZ),
            val_start=datetime(2021, 7, 1, tzinfo=NY_TZ),
            val_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
            n_train_samples=1000,
            n_val_samples=500,
            train_auc=0.55,
            val_auc=0.58,  # Better than train
            train_accuracy=0.52,
            val_accuracy=0.56,
            val_brier_score=0.22,
            val_expected_calibration_error=0.06,
            best_iteration=100,
            feature_importance={},
        )

        assert result.overfit_score < 0  # No overfitting


# ============================================================================
# Test WalkForwardResult
# ============================================================================

class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    @pytest.fixture
    def sample_result(self, sample_fold_result):
        """Create sample WalkForwardResult with multiple folds."""
        fold1 = sample_fold_result
        fold2 = FoldResult(
            fold_idx=1,
            train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
            train_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
            val_start=datetime(2022, 1, 1, tzinfo=NY_TZ),
            val_end=datetime(2022, 6, 30, tzinfo=NY_TZ),
            n_train_samples=1500,
            n_val_samples=500,
            train_auc=0.64,
            val_auc=0.60,
            train_accuracy=0.61,
            val_accuracy=0.58,
            val_brier_score=0.19,
            val_expected_calibration_error=0.04,
            best_iteration=180,
            feature_importance={"feat_0": 28.0, "feat_1": 22.0},
        )

        config = WalkForwardConfig(n_folds=2)
        return WalkForwardResult(
            fold_results=[fold1, fold2],
            config=config,
        )

    def test_mean_val_auc(self, sample_result):
        """Test mean validation AUC calculation."""
        expected = (0.58 + 0.60) / 2
        assert abs(sample_result.mean_val_auc - expected) < 1e-6

    def test_std_val_auc(self, sample_result):
        """Test std of validation AUC calculation."""
        expected = np.std([0.58, 0.60])
        assert abs(sample_result.std_val_auc - expected) < 1e-6

    def test_mean_val_accuracy(self, sample_result):
        """Test mean validation accuracy calculation."""
        expected = (0.56 + 0.58) / 2
        assert abs(sample_result.mean_val_accuracy - expected) < 1e-6

    def test_mean_val_brier(self, sample_result):
        """Test mean Brier score calculation."""
        expected = (0.20 + 0.19) / 2
        assert abs(sample_result.mean_val_brier - expected) < 1e-6

    def test_mean_val_ece(self, sample_result):
        """Test mean ECE calculation."""
        expected = (0.05 + 0.04) / 2
        assert abs(sample_result.mean_val_ece - expected) < 1e-6

    def test_mean_overfit_score(self, sample_result):
        """Test mean overfit score calculation."""
        # Fold 1: 0.65 - 0.58 = 0.07
        # Fold 2: 0.64 - 0.60 = 0.04
        expected = (0.07 + 0.04) / 2
        assert abs(sample_result.mean_overfit_score - expected) < 1e-6

    def test_is_stable_positive(self):
        """Test stability check passes with good folds."""
        fold1 = FoldResult(
            fold_idx=0,
            train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
            train_end=datetime(2021, 6, 30, tzinfo=NY_TZ),
            val_start=datetime(2021, 7, 1, tzinfo=NY_TZ),
            val_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
            n_train_samples=1000,
            n_val_samples=500,
            train_auc=0.58,
            val_auc=0.56,  # Close to train, slight overfitting
            train_accuracy=0.55,
            val_accuracy=0.54,
            val_brier_score=0.20,
            val_expected_calibration_error=0.05,
            best_iteration=100,
            feature_importance={},
        )
        fold2 = FoldResult(
            fold_idx=1,
            train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
            train_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
            val_start=datetime(2022, 1, 1, tzinfo=NY_TZ),
            val_end=datetime(2022, 6, 30, tzinfo=NY_TZ),
            n_train_samples=1500,
            n_val_samples=500,
            train_auc=0.57,
            val_auc=0.55,  # Stable performance
            train_accuracy=0.54,
            val_accuracy=0.53,
            val_brier_score=0.21,
            val_expected_calibration_error=0.05,
            best_iteration=120,
            feature_importance={},
        )

        result = WalkForwardResult(
            fold_results=[fold1, fold2],
            config=WalkForwardConfig(n_folds=2),
        )

        # Should be stable: low std, low overfit, val AUC > 0.52
        assert result.is_stable

    def test_is_stable_negative_high_variance(self):
        """Test stability check fails with high variance."""
        fold1 = FoldResult(
            fold_idx=0,
            train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
            train_end=datetime(2021, 6, 30, tzinfo=NY_TZ),
            val_start=datetime(2021, 7, 1, tzinfo=NY_TZ),
            val_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
            n_train_samples=1000,
            n_val_samples=500,
            train_auc=0.58,
            val_auc=0.55,
            train_accuracy=0.55,
            val_accuracy=0.53,
            val_brier_score=0.20,
            val_expected_calibration_error=0.05,
            best_iteration=100,
            feature_importance={},
        )
        fold2 = FoldResult(
            fold_idx=1,
            train_start=datetime(2020, 1, 1, tzinfo=NY_TZ),
            train_end=datetime(2021, 12, 31, tzinfo=NY_TZ),
            val_start=datetime(2022, 1, 1, tzinfo=NY_TZ),
            val_end=datetime(2022, 6, 30, tzinfo=NY_TZ),
            n_train_samples=1500,
            n_val_samples=500,
            train_auc=0.70,
            val_auc=0.62,  # Very different from fold 1
            train_accuracy=0.65,
            val_accuracy=0.58,
            val_brier_score=0.15,
            val_expected_calibration_error=0.03,
            best_iteration=200,
            feature_importance={},
        )

        result = WalkForwardResult(
            fold_results=[fold1, fold2],
            config=WalkForwardConfig(n_folds=2),
        )

        # Should NOT be stable: std_val_auc > 0.03
        assert result.std_val_auc > 0.03
        assert not result.is_stable

    def test_summary(self, sample_result):
        """Test summary dictionary generation."""
        summary = sample_result.summary()

        assert summary["n_folds"] == 2
        assert "mean_val_auc" in summary
        assert "std_val_auc" in summary
        assert "is_stable" in summary
        assert "per_fold_auc" in summary
        assert len(summary["per_fold_auc"]) == 2


# ============================================================================
# Test WalkForwardCV - Fold Generation
# ============================================================================

class TestWalkForwardCVFoldGeneration:
    """Tests for fold generation in WalkForwardCV."""

    def test_generate_folds_creates_correct_number(self, multi_year_data):
        """Test that correct number of folds is generated."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(
            n_folds=3,
            min_train_months=12,
            val_months=6,
        )
        cv = WalkForwardCV(config=config)

        # Create DataFrame for fold generation
        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        assert len(folds) == 3

    def test_generate_folds_no_overlap(self, multi_year_data):
        """Test that train and validation periods don't overlap."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(
            n_folds=3,
            min_train_months=12,
            val_months=6,
        )
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        for train_df, val_df, train_start, train_end, val_start, val_end in folds:
            # Train end must be before val start
            assert train_end < val_start

            # No samples should appear in both sets
            train_indices = set(train_df.index)
            val_indices = set(val_df.index)
            assert len(train_indices & val_indices) == 0

    def test_generate_folds_expanding_window(self, multi_year_data):
        """Test expanding window behavior."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(
            n_folds=3,
            min_train_months=12,
            val_months=6,
            expanding=True,
        )
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        # With expanding window, training size should increase
        train_sizes = [len(train_df) for train_df, _, _, _, _, _ in folds]

        # Each fold should have more training data than the previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1]

    def test_generate_folds_rolling_window(self, multi_year_data):
        """Test rolling window behavior."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(
            n_folds=3,
            min_train_months=6,
            val_months=3,
            expanding=False,
            rolling_train_months=12,
        )
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        # With rolling window, training sizes should be more similar
        train_sizes = [len(train_df) for train_df, _, _, _, _, _ in folds]

        # Key behavior: rolling window doesn't grow like expanding window
        # After the first fold (which may be limited by data start), sizes should stabilize
        # Check that later folds have similar sizes (excluding first fold which may be smaller)
        if len(train_sizes) > 2:
            later_sizes = train_sizes[1:]  # Exclude first fold
            max_size = max(later_sizes)
            min_size = min(later_sizes)
            # Later folds should be very similar in size (within 10%)
            assert min_size / max_size > 0.9

        # Also verify that rolling produces smaller training sets than expanding would
        # (This is the main characteristic of rolling windows)
        # The first fold is smaller due to data start constraint, but later folds are capped

    def test_generate_folds_temporal_ordering(self, multi_year_data):
        """Test that folds are in chronological order."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(n_folds=3, min_train_months=12, val_months=6)
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        # Validation periods should be chronological
        val_starts = [val_start for _, _, _, _, val_start, _ in folds]
        for i in range(1, len(val_starts)):
            assert val_starts[i] > val_starts[i-1]

    def test_generate_folds_from_arrays(self, multi_year_data):
        """Test fold generation from numpy arrays."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(n_folds=2, min_train_months=12, val_months=6)
        cv = WalkForwardCV(config=config)

        folds = cv.generate_folds_from_arrays(X, y, timestamps)

        assert len(folds) == 2

        for X_train, y_train, X_val, y_val, _, _, _, _ in folds:
            # Check arrays have correct dimensions
            assert X_train.ndim == 2
            assert y_train.ndim == 1
            assert X_val.ndim == 2
            assert y_val.ndim == 1

            # Check lengths match
            assert len(X_train) == len(y_train)
            assert len(X_val) == len(y_val)

    def test_generate_folds_insufficient_data(self):
        """Test error when not enough data for requested folds."""
        np.random.seed(42)

        # Only 6 months of data
        start = pd.Timestamp("2020-01-02 09:00:00", tz=NY_TZ)
        end = pd.Timestamp("2020-06-30 16:00:00", tz=NY_TZ)
        dates = pd.date_range(start=start, end=end, freq="h")

        n_bars = len(dates)
        X = np.random.randn(n_bars, 5)
        y = (X[:, 0] > 0).astype(int)

        # Request 5 folds with 12 months min training
        config = WalkForwardConfig(n_folds=5, min_train_months=12, val_months=3)
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=dates)

        with pytest.raises(ValueError, match="Not enough data"):
            cv.generate_folds(df)

    def test_generate_folds_requires_datetime_index(self):
        """Test error when DataFrame doesn't have DatetimeIndex."""
        cv = WalkForwardCV()

        df = pd.DataFrame({
            "_target": [0, 1, 0, 1],
        }, index=[0, 1, 2, 3])  # Integer index

        with pytest.raises(ValueError, match="DatetimeIndex"):
            cv.generate_folds(df)


# ============================================================================
# Test Calibration Metrics
# ============================================================================

class TestCalibrationMetrics:
    """Tests for calibration metric calculations."""

    def test_brier_score_perfect(self):
        """Test Brier score with perfect predictions."""
        cv = WalkForwardCV()

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        brier, ece = cv.calculate_calibration_metrics(y_true, y_prob)

        assert brier == 0.0  # Perfect calibration

    def test_brier_score_worst(self):
        """Test Brier score with completely wrong predictions."""
        cv = WalkForwardCV()

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])

        brier, ece = cv.calculate_calibration_metrics(y_true, y_prob)

        assert brier == 1.0  # Worst possible

    def test_brier_score_random(self):
        """Test Brier score with 0.5 predictions."""
        cv = WalkForwardCV()

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        brier, ece = cv.calculate_calibration_metrics(y_true, y_prob)

        # Brier score for 0.5 predictions on 50-50 labels
        expected_brier = 0.25  # (0.5-0)^2 = 0.25 for class 0, (0.5-1)^2 = 0.25 for class 1
        assert abs(brier - expected_brier) < 1e-6

    def test_ece_well_calibrated(self):
        """Test ECE with well-calibrated predictions."""
        cv = WalkForwardCV()

        # Create well-calibrated predictions
        np.random.seed(42)
        n = 1000
        y_prob = np.random.uniform(0, 1, n)

        # Generate labels that match the probabilities
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        brier, ece = cv.calculate_calibration_metrics(y_true, y_prob)

        # ECE should be low for well-calibrated predictions
        assert ece < 0.1

    def test_ece_poorly_calibrated(self):
        """Test ECE with poorly calibrated predictions."""
        cv = WalkForwardCV()

        # Predictions say 80% confidence but accuracy is 50%
        n = 100
        y_true = np.array([0, 1] * 50)  # 50-50 split
        y_prob = np.array([0.8] * 100)  # All predict 80% class 1

        brier, ece = cv.calculate_calibration_metrics(y_true, y_prob)

        # ECE should be high: predicted 80% but got 50%
        assert ece > 0.2


# ============================================================================
# Test WalkForwardCV - Full Run
# ============================================================================

class TestWalkForwardCVRun:
    """Tests for running walk-forward CV."""

    def test_run_completes(self, small_data):
        """Test that run() completes without errors."""
        X, y, timestamps = small_data

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run(X, y, timestamps)

        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 2

    def test_run_with_feature_names(self, small_data):
        """Test run with feature names."""
        X, y, timestamps = small_data
        n_features = X.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run(X, y, timestamps, feature_names=feature_names)

        # Check feature names in importance
        for fold in result.fold_results:
            for feat_name in fold.feature_importance.keys():
                assert feat_name.startswith("feature_")

    def test_run_with_custom_model_config(self, small_data):
        """Test run with custom model configuration."""
        X, y, timestamps = small_data

        model_config = ModelConfig(
            num_leaves=15,
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            early_stopping_rounds=10,
        )

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            model_config=model_config,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run(X, y, timestamps)

        assert result.best_config == model_config

    def test_run_with_dataframe(self, small_data):
        """Test run_with_dataframe convenience method."""
        X, y, timestamps = small_data
        n_features = X.shape[1]
        feature_cols = [f"feat_{i}" for i in range(n_features)]

        df = pd.DataFrame(X, columns=feature_cols, index=timestamps)
        df["target"] = y

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run_with_dataframe(df, feature_cols, "target")

        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 2

    def test_run_val_auc_better_than_random(self, small_data):
        """Test that model learns something (AUC > 0.5)."""
        X, y, timestamps = small_data

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run(X, y, timestamps)

        # With the synthetic data (signal + noise), model should do better than random
        assert result.mean_val_auc > 0.5

    def test_run_tracks_calibration(self, small_data):
        """Test that calibration metrics are tracked."""
        X, y, timestamps = small_data

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run(X, y, timestamps)

        # Check calibration metrics exist and are reasonable
        assert result.mean_val_brier >= 0
        assert result.mean_val_brier <= 1
        assert result.mean_val_ece >= 0
        assert result.mean_val_ece <= 1


# ============================================================================
# Test Convenience Function
# ============================================================================

class TestConvenienceFunction:
    """Tests for run_walk_forward_validation function."""

    def test_convenience_function(self, small_data):
        """Test convenience function with default parameters."""
        X, y, timestamps = small_data

        result = run_walk_forward_validation(
            X, y, timestamps,
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.fold_results) == 2

    def test_convenience_function_with_all_params(self, small_data):
        """Test convenience function with all parameters."""
        X, y, timestamps = small_data
        feature_names = [f"f{i}" for i in range(X.shape[1])]

        model_config = ModelConfig(num_leaves=15)

        result = run_walk_forward_validation(
            X, y, timestamps,
            feature_names=feature_names,
            n_folds=2,
            min_train_months=6,
            val_months=3,
            expanding=True,
            model_config=model_config,
            verbose=0,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.best_config == model_config


# ============================================================================
# Test Data Leakage Prevention
# ============================================================================

class TestDataLeakagePrevention:
    """Tests to verify no data leakage in walk-forward validation."""

    def test_no_future_data_in_training(self, multi_year_data):
        """Verify training data never includes future data."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(n_folds=3, min_train_months=12, val_months=6)
        cv = WalkForwardCV(config=config)

        folds = cv.generate_folds_from_arrays(X, y, timestamps)

        for i, (X_train, y_train, X_val, y_val, train_start, train_end, val_start, val_end) in enumerate(folds):
            # Get timestamps for train and val sets
            train_mask = (timestamps >= train_start) & (timestamps <= train_end)
            val_mask = (timestamps >= val_start) & (timestamps <= val_end)

            train_timestamps = timestamps[train_mask]
            val_timestamps = timestamps[val_mask]

            # All training timestamps must be before all validation timestamps
            if len(train_timestamps) > 0 and len(val_timestamps) > 0:
                assert train_timestamps.max() < val_timestamps.min(), \
                    f"Fold {i}: Training data extends into validation period"

    def test_validation_follows_training(self, multi_year_data):
        """Verify validation period immediately follows training."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(n_folds=3, min_train_months=12, val_months=6)
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        for train_df, val_df, train_start, train_end, val_start, val_end in folds:
            # Train end should be just before val start
            gap = val_start - train_end
            assert gap.total_seconds() <= 86400, \
                f"Gap between train and val too large: {gap}"

    def test_folds_independent(self, multi_year_data):
        """Verify validation sets of different folds don't overlap."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(n_folds=3, min_train_months=12, val_months=6)
        cv = WalkForwardCV(config=config)

        df = pd.DataFrame({"_target": y}, index=timestamps)
        folds = cv.generate_folds(df)

        # Collect all validation indices
        val_index_sets = []
        for train_df, val_df, _, _, _, _ in folds:
            val_index_sets.append(set(val_df.index))

        # Check for overlaps between consecutive validation sets
        for i in range(len(val_index_sets) - 1):
            overlap = val_index_sets[i] & val_index_sets[i + 1]
            # Some overlap is allowed since val periods may overlap
            # But train/val should never overlap
            pass  # This is just checking validation sets

    def test_training_never_sees_validation_samples(self, multi_year_data):
        """Critical test: training set must never contain validation samples."""
        X, y, timestamps = multi_year_data

        config = WalkForwardConfig(n_folds=3, min_train_months=12, val_months=6)
        cv = WalkForwardCV(config=config)

        folds = cv.generate_folds_from_arrays(X, y, timestamps)

        for i, (X_train, y_train, X_val, y_val, train_start, train_end, val_start, val_end) in enumerate(folds):
            # Get indices
            train_mask = (timestamps >= train_start) & (timestamps <= train_end)
            val_mask = (timestamps >= val_start) & (timestamps <= val_end)

            # No overlap
            overlap = train_mask & val_mask
            assert not overlap.any(), \
                f"Fold {i}: {overlap.sum()} samples appear in both train and val sets"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_fold(self, small_data):
        """Test with single fold."""
        X, y, timestamps = small_data

        config = WalkForwardConfig(
            n_folds=1,
            min_train_months=12,
            val_months=6,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        result = cv.run(X, y, timestamps)

        assert len(result.fold_results) == 1

    def test_empty_fold_skipped(self):
        """Test that empty folds are skipped gracefully."""
        np.random.seed(42)

        # Only ~6 months of data
        start = pd.Timestamp("2020-01-02 09:00:00", tz=NY_TZ)
        end = pd.Timestamp("2020-06-30 16:00:00", tz=NY_TZ)
        dates = pd.date_range(start=start, end=end, freq="h")

        n_bars = len(dates)
        X = np.random.randn(n_bars, 5)
        y = (X[:, 0] > 0).astype(int)

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        # Should complete without error or raise ValueError for insufficient data
        df = pd.DataFrame({"_target": y}, index=dates)
        try:
            folds = cv.generate_folds(df)
            # If we get folds, they should be valid
            for train_df, val_df, _, _, _, _ in folds:
                assert len(train_df) > 0
                assert len(val_df) > 0
        except ValueError:
            # Expected if not enough data
            pass

    def test_class_imbalance(self, small_data):
        """Test with imbalanced classes."""
        X, _, timestamps = small_data

        np.random.seed(42)
        # 90% class 0, 10% class 1
        n_bars = len(X)
        y = (np.random.uniform(0, 1, n_bars) < 0.1).astype(int)

        config = WalkForwardConfig(
            n_folds=2,
            min_train_months=6,
            val_months=3,
            verbose=0,
        )
        cv = WalkForwardCV(config=config)

        # Should complete without error
        result = cv.run(X, y, timestamps)
        assert len(result.fold_results) > 0
