"""
Tests for Walk-Forward Cross-Validation Optimizer.

These tests verify:
- Correct fold generation with temporal ordering
- No future data leakage across folds
- Proper aggregation of results across folds
- Overfitting detection via validation vs test comparison
- Robustness scoring across multiple folds

Why Walk-Forward Testing Matters:
- Time-series data requires special handling to prevent temporal leakage
- Standard k-fold CV would allow future data to inform past predictions
- Walk-forward maintains temporal ordering: train -> validate -> test
- Tests ensure the optimizer correctly prevents lookahead bias
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.optimization.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    WalkForwardFold,
    FoldResult,
    WalkForwardResult,
    run_walk_forward_optimization,
)
from src.optimization.parameter_space import ParameterConfig, ParameterSpace
from src.optimization.random_search import RandomSearchOptimizer, RandomSearchConfig

# Create a default inner optimizer config for tests
def get_test_inner_kwargs():
    """Get inner optimizer kwargs with config for testing."""
    return {
        "config": RandomSearchConfig(
            n_iterations=5,
            metric_name="sharpe_ratio",
            higher_is_better=True,
            verbose=0,
        )
    }


@pytest.fixture
def sample_data():
    """Create sample data spanning 3 years for walk-forward testing."""
    # Create 3 years of daily data (enough for multiple folds)
    dates = pd.date_range(
        start='2022-01-01',
        end='2024-12-31',
        freq='D',
        tz='America/New_York'
    )

    np.random.seed(42)
    n = len(dates)

    df = pd.DataFrame({
        'open': 5000 + np.random.randn(n).cumsum() * 0.1,
        'high': 5000 + np.random.randn(n).cumsum() * 0.1 + 1,
        'low': 5000 + np.random.randn(n).cumsum() * 0.1 - 1,
        'close': 5000 + np.random.randn(n).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, n),
    }, index=dates)

    return df


@pytest.fixture
def simple_space():
    """Create simple parameter space for testing."""
    return ParameterSpace(parameters=[
        ParameterConfig("threshold", 1.0, 10.0, param_type="float"),
        ParameterConfig("window", 5, 50, param_type="int"),
    ])


def create_mock_objective_factory(metric_base: float = 1.0):
    """
    Create an objective function factory for testing.

    The factory returns objectives that produce deterministic metrics
    based on parameter values, making tests reproducible.
    """
    def objective_factory(data: pd.DataFrame):
        """Create objective function for specific data slice."""
        data_len = len(data)

        def objective(params: Dict[str, Any]) -> Dict[str, float]:
            # Produce deterministic metrics based on params and data size
            threshold = params.get("threshold", 5.0)
            window = params.get("window", 20)

            # Metric varies with params to ensure optimizer finds different values
            sharpe = metric_base * (1.0 - abs(threshold - 5.0) / 10.0) * (1.0 - abs(window - 20) / 50.0)

            # Add small variation based on data length to simulate different periods
            sharpe += (data_len % 100) / 1000.0

            return {
                "sharpe_ratio": sharpe,
                "n_trades": max(10, data_len // 10),
            }

        return objective

    return objective_factory


class TestWalkForwardFold:
    """Tests for WalkForwardFold dataclass."""

    def test_fold_creation(self):
        """Test creating a fold."""
        fold = WalkForwardFold(
            fold_id=0,
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2022, 7, 1),
            val_start=datetime(2022, 7, 1),
            val_end=datetime(2022, 8, 1),
            test_start=datetime(2022, 8, 1),
            test_end=datetime(2022, 9, 1),
        )

        assert fold.fold_id == 0
        assert fold.train_range == (datetime(2022, 1, 1), datetime(2022, 7, 1))
        assert fold.val_range == (datetime(2022, 7, 1), datetime(2022, 8, 1))
        assert fold.test_range == (datetime(2022, 8, 1), datetime(2022, 9, 1))

    def test_fold_temporal_ordering(self):
        """Test that fold periods are temporally ordered."""
        fold = WalkForwardFold(
            fold_id=0,
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2022, 7, 1),
            val_start=datetime(2022, 7, 1),
            val_end=datetime(2022, 8, 1),
            test_start=datetime(2022, 8, 1),
            test_end=datetime(2022, 9, 1),
        )

        # Train < Val < Test (no overlap, proper ordering)
        assert fold.train_end <= fold.val_start
        assert fold.val_end <= fold.test_start


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()

        assert config.training_months == 6
        assert config.validation_months == 1
        assert config.test_months == 1
        assert config.step_months == 1
        assert config.min_trades_per_fold == 100
        assert config.max_overfitting_score == 0.3
        assert config.require_minimum_folds == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = WalkForwardConfig(
            training_months=3,
            validation_months=2,
            test_months=1,
            min_trades_per_fold=50,
        )

        assert config.training_months == 3
        assert config.validation_months == 2
        assert config.min_trades_per_fold == 50


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    def test_generate_folds(self, sample_data, simple_space):
        """Test fold generation."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            step_months=1,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
        )

        folds = optimizer.generate_folds(sample_data)

        # Should generate multiple folds (3 years of data)
        assert len(folds) >= 10

        # Check all folds have proper temporal ordering
        for fold in folds:
            assert fold.train_start < fold.train_end
            assert fold.train_end == fold.val_start
            assert fold.val_end == fold.test_start
            assert fold.test_start < fold.test_end

    def test_folds_no_overlap(self, sample_data, simple_space):
        """Test that train/val/test periods within a fold don't overlap."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
        )

        folds = optimizer.generate_folds(sample_data)

        for fold in folds:
            # Train period ends where validation starts
            assert fold.train_end == fold.val_start
            # Validation ends where test starts
            assert fold.val_end == fold.test_start

    def test_extract_fold_data(self, sample_data, simple_space):
        """Test extracting data for a fold."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
        )

        folds = optimizer.generate_folds(sample_data)
        fold = folds[0]

        train_data, val_data, test_data = optimizer._extract_fold_data(sample_data, fold)

        # All datasets should have data
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert len(test_data) > 0

        # No temporal overlap
        assert train_data.index.max() <= val_data.index.min()
        assert val_data.index.max() <= test_data.index.min()

    def test_optimize_basic(self, sample_data, simple_space):
        """Test basic walk-forward optimization."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            min_trades_per_fold=5,  # Lower for test
            require_minimum_folds=3,
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
            inner_optimizer_kwargs=get_test_inner_kwargs(),  # Quick for test
        )

        result = optimizer.optimize(sample_data)

        # Should have results
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds >= 3
        assert result.best_params is not None
        assert "threshold" in result.best_params
        assert "window" in result.best_params

    def test_overfitting_detection(self, sample_data, simple_space):
        """Test overfitting detection via validation vs test comparison."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            min_trades_per_fold=5,
            require_minimum_folds=3,
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
            inner_optimizer_kwargs=get_test_inner_kwargs(),
        )

        result = optimizer.optimize(sample_data)

        # Overfitting score should be calculated
        assert result.overfitting_score is not None
        assert isinstance(result.overfitting_score, float)

    def test_consistency_score(self, sample_data, simple_space):
        """Test consistency score calculation across folds."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            min_trades_per_fold=5,
            require_minimum_folds=3,
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
            inner_optimizer_kwargs=get_test_inner_kwargs(),
        )

        result = optimizer.optimize(sample_data)

        # Consistency score should be calculated
        assert result.consistency_score is not None
        assert isinstance(result.consistency_score, float)
        assert result.consistency_score >= 0  # Should be non-negative

    def test_insufficient_data_raises(self, simple_space):
        """Test that insufficient data raises an error."""
        # Create short data (less than required for 3 folds)
        short_data = pd.DataFrame({
            'open': [5000] * 30,
            'close': [5000] * 30,
        }, index=pd.date_range(start='2022-01-01', periods=30, freq='D', tz='America/New_York'))

        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            require_minimum_folds=3,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
        )

        with pytest.raises(ValueError, match="Not enough data"):
            optimizer.optimize(short_data)


class TestWalkForwardResult:
    """Tests for WalkForwardResult."""

    def test_result_summary(self, sample_data, simple_space):
        """Test result summary generation."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            min_trades_per_fold=5,
            require_minimum_folds=3,
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
            inner_optimizer_kwargs=get_test_inner_kwargs(),
        )

        result = optimizer.optimize(sample_data)

        summary = result.summary()

        # Summary should contain key information
        assert "WALK-FORWARD" in summary
        assert "Folds:" in summary
        assert "PERFORMANCE" in summary
        assert "BEST PARAMETERS" in summary

    def test_is_robust_property(self, sample_data, simple_space):
        """Test robustness check."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            min_trades_per_fold=5,
            require_minimum_folds=3,
            max_overfitting_score=0.5,  # Lenient for test
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
            inner_optimizer_kwargs=get_test_inner_kwargs(),
        )

        result = optimizer.optimize(sample_data)

        # is_robust should be a boolean
        assert isinstance(result.is_robust, bool)


class TestRunWalkForwardOptimization:
    """Tests for convenience function."""

    def test_run_walk_forward_convenience(self, sample_data, simple_space):
        """Test convenience function works."""
        result = run_walk_forward_optimization(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            data=sample_data,
            training_months=6,
            validation_months=1,
            test_months=1,
            min_trades_per_fold=5,
            require_minimum_folds=3,
            verbose=0,
            inner_optimizer_kwargs=get_test_inner_kwargs(),
        )

        assert isinstance(result, WalkForwardResult)
        assert result.n_folds >= 3


class TestTemporalLeakagePrevention:
    """
    Tests specifically for temporal leakage prevention.

    These tests are CRITICAL for ensuring the walk-forward optimizer
    maintains proper time boundaries and prevents future information
    from leaking into training/validation.
    """

    def test_no_future_data_in_training(self, sample_data, simple_space):
        """Test that training data never includes future data."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
        )

        folds = optimizer.generate_folds(sample_data)

        for fold in folds:
            train_data, val_data, test_data = optimizer._extract_fold_data(sample_data, fold)

            # Training data must end before validation data starts
            if len(train_data) > 0 and len(val_data) > 0:
                assert train_data.index.max() < val_data.index.min(), \
                    f"Training data overlaps with validation data in fold {fold.fold_id}"

            # Validation data must end before test data starts
            if len(val_data) > 0 and len(test_data) > 0:
                assert val_data.index.max() < test_data.index.min(), \
                    f"Validation data overlaps with test data in fold {fold.fold_id}"

    def test_test_data_always_in_future(self, sample_data, simple_space):
        """Test that test data is always in the future relative to validation."""
        config = WalkForwardConfig(
            training_months=6,
            validation_months=1,
            test_months=1,
            verbose=0,
        )

        optimizer = WalkForwardOptimizer(
            parameter_space=simple_space,
            objective_fn_factory=create_mock_objective_factory(),
            config=config,
        )

        folds = optimizer.generate_folds(sample_data)

        for fold in folds:
            _, val_data, test_data = optimizer._extract_fold_data(sample_data, fold)

            if len(val_data) > 0 and len(test_data) > 0:
                # All test timestamps must be after all validation timestamps
                assert val_data.index.max() < test_data.index.min(), \
                    f"Test data is not strictly in the future in fold {fold.fold_id}"
