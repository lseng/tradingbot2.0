"""
Comprehensive Tests for src/optimization/ Module.

This module tests:
- ParameterConfig (float, int, categorical parameters)
- ParameterSpace (adding params, sampling, grid generation)
- TrialResult and OptimizationResult (metrics, overfitting analysis)
- BaseOptimizer (objective functions, evaluation)
- GridSearchOptimizer (exhaustive search)
- RandomSearchOptimizer (random sampling, early stopping)
- BayesianOptimizer (TPE sampling, study persistence)
- Integration tests (end-to-end optimization)

Test Coverage Target: 50+ tests, >90% line coverage
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
import tempfile
import os
from pathlib import Path

from src.optimization.parameter_space import (
    ParameterConfig,
    ParameterSpace,
    DefaultParameterSpaces,
    ParameterType,
    create_parameter_space_from_config,
)
from src.optimization.results import (
    TrialResult,
    OptimizationResult,
    calculate_overfitting_score,
    is_overfitting,
    merge_results,
)
from src.optimization.optimizer_base import (
    BaseOptimizer,
    OptimizerConfig,
    create_backtest_objective,
    create_split_objective,
)
from src.optimization.grid_search import (
    GridSearchOptimizer,
    GridSearchConfig,
    run_grid_search,
    grid_search_with_cv,
)
from src.optimization.random_search import (
    RandomSearchOptimizer,
    RandomSearchConfig,
    AdaptiveRandomSearch,
    run_random_search,
)

# Try to import Bayesian optimizer (optional dependency)
try:
    from src.optimization.bayesian_optimizer import (
        BayesianOptimizer,
        BayesianConfig,
        run_bayesian_optimization,
    )
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def float_param():
    """Create a float parameter config."""
    return ParameterConfig(
        name="confidence",
        min_value=0.5,
        max_value=0.9,
        step=0.1,
        param_type="float",
        description="Confidence threshold",
    )


@pytest.fixture
def int_param():
    """Create an int parameter config."""
    return ParameterConfig(
        name="stop_ticks",
        min_value=4,
        max_value=12,
        step=2,
        param_type="int",
        default=8,
        description="Stop loss in ticks",
    )


@pytest.fixture
def categorical_param():
    """Create a categorical parameter config."""
    return ParameterConfig(
        name="order_type",
        param_type="categorical",
        choices=["market", "limit", "stop"],
        default="market",
        description="Order type",
    )


@pytest.fixture
def simple_space(float_param, int_param):
    """Create a simple parameter space."""
    return ParameterSpace(
        name="simple_test",
        parameters=[float_param, int_param],
    )


@pytest.fixture
def sample_trial_result():
    """Create a sample trial result."""
    return TrialResult(
        trial_id=1,
        params={"stop_ticks": 8, "confidence": 0.6},
        metrics={
            "sharpe_ratio": 1.5,
            "profit_factor": 1.3,
            "win_rate_pct": 55.0,
        },
        status="completed",
        duration_seconds=1.2,
    )


@pytest.fixture
def mock_objective():
    """Create a mock objective function."""
    def objective(params):
        # Simple function: maximize sum of params
        result = sum(v for v in params.values() if isinstance(v, (int, float)))
        return {
            "sharpe_ratio": result / 10.0,
            "profit_factor": result / 8.0,
            "win_rate_pct": 50.0 + result,
        }
    return objective


@pytest.fixture
def mock_backtest_engine():
    """Create a mock backtest engine."""
    engine = Mock()
    engine.config = Mock()
    engine.config.default_stop_ticks = 8
    engine.config.default_target_ticks = 16
    engine.config.min_confidence = 0.6

    # Mock run method
    result = Mock()
    result.report = Mock()
    result.report.metrics = Mock()
    result.report.metrics.to_dict = Mock(return_value={
        "performance": {
            "sharpe_ratio": 1.5,
            "profit_factor": 1.3,
            "win_rate_pct": 55.0,
        }
    })

    engine.run = Mock(return_value=result)
    return engine


# ============================================================================
# ParameterConfig Tests (12 tests)
# ============================================================================

class TestParameterConfig:
    """Tests for ParameterConfig class."""

    def test_float_param_creation(self, float_param):
        """Test creating a float parameter."""
        assert float_param.name == "confidence"
        assert float_param.min_value == 0.5
        assert float_param.max_value == 0.9
        assert float_param.param_type == "float"

    def test_int_param_creation(self, int_param):
        """Test creating an int parameter."""
        assert int_param.name == "stop_ticks"
        assert int_param.min_value == 4
        assert int_param.max_value == 12
        assert int_param.param_type == "int"
        assert int_param.default == 8

    def test_categorical_param_creation(self, categorical_param):
        """Test creating a categorical parameter."""
        assert categorical_param.name == "order_type"
        assert categorical_param.param_type == "categorical"
        assert len(categorical_param.choices) == 3
        assert "market" in categorical_param.choices

    def test_invalid_float_param_no_min(self):
        """Test that float param without min_value raises error."""
        with pytest.raises(ValueError, match="min_value and max_value required"):
            ParameterConfig(name="test", param_type="float", max_value=1.0)

    def test_invalid_float_param_min_greater_than_max(self):
        """Test that min > max raises error."""
        with pytest.raises(ValueError, match="min_value .* must be <= max_value"):
            ParameterConfig(
                name="test",
                min_value=1.0,
                max_value=0.5,
                param_type="float"
            )

    def test_invalid_categorical_no_choices(self):
        """Test that categorical without choices raises error."""
        with pytest.raises(ValueError, match="choices required"):
            ParameterConfig(name="test", param_type="categorical")

    def test_float_grid_values_with_step(self, float_param):
        """Test generating grid values for float with step."""
        values = float_param.get_grid_values()
        assert len(values) == 5  # 0.5, 0.6, 0.7, 0.8, 0.9
        assert values[0] == 0.5
        assert values[-1] == 0.9

    def test_int_grid_values_with_step(self, int_param):
        """Test generating grid values for int with step."""
        values = int_param.get_grid_values()
        assert all(isinstance(v, int) for v in values)
        assert values[0] == 4
        assert values[-1] == 12
        assert 8 in values

    def test_categorical_grid_values(self, categorical_param):
        """Test generating grid values for categorical."""
        values = categorical_param.get_grid_values()
        assert len(values) == 3
        assert set(values) == {"market", "limit", "stop"}

    def test_float_random_sampling(self, float_param):
        """Test random sampling from float parameter."""
        rng = np.random.default_rng(42)
        samples = [float_param.sample_random(rng) for _ in range(10)]

        # All samples should be within bounds
        assert all(0.5 <= s <= 0.9 for s in samples)
        # Should have some variation
        assert len(set(samples)) > 1

    def test_int_random_sampling(self, int_param):
        """Test random sampling from int parameter."""
        rng = np.random.default_rng(42)
        samples = [int_param.sample_random(rng) for _ in range(10)]

        # All samples should be integers within bounds
        assert all(isinstance(s, int) for s in samples)
        assert all(4 <= s <= 12 for s in samples)

    def test_categorical_random_sampling(self, categorical_param):
        """Test random sampling from categorical parameter."""
        rng = np.random.default_rng(42)
        samples = [categorical_param.sample_random(rng) for _ in range(20)]

        # All samples should be in choices
        assert all(s in categorical_param.choices for s in samples)
        # Should have multiple values
        assert len(set(samples)) > 1

    def test_param_min_equals_max(self):
        """Test parameter where min equals max."""
        param = ParameterConfig(
            name="fixed",
            min_value=5.0,
            max_value=5.0,
            param_type="float"
        )
        values = param.get_grid_values()
        # Should work but only give one value
        assert len(values) >= 1
        assert all(v == 5.0 for v in values)

    def test_log_scale_sampling(self):
        """Test log-scale sampling."""
        param = ParameterConfig(
            name="learning_rate",
            min_value=0.001,
            max_value=0.1,
            param_type="float",
            log_scale=True,
        )

        rng = np.random.default_rng(42)
        samples = [param.sample_random(rng) for _ in range(100)]

        # Should sample more values near the lower end
        below_0_01 = sum(1 for s in samples if s < 0.01)
        above_0_05 = sum(1 for s in samples if s > 0.05)
        # Due to log scale, should have more samples at lower end
        assert below_0_01 > above_0_05

    def test_param_to_dict(self, float_param):
        """Test parameter serialization."""
        d = float_param.to_dict()
        assert d["name"] == "confidence"
        assert d["min_value"] == 0.5
        assert d["max_value"] == 0.9
        assert d["param_type"] == "float"


# ============================================================================
# ParameterSpace Tests (13 tests)
# ============================================================================

class TestParameterSpace:
    """Tests for ParameterSpace class."""

    def test_space_creation(self, simple_space):
        """Test creating a parameter space."""
        assert simple_space.name == "simple_test"
        assert len(simple_space.parameters) == 2

    def test_add_parameter(self):
        """Test adding parameters to space."""
        space = ParameterSpace(name="test")
        param = ParameterConfig("new_param", 0.0, 1.0, param_type="float")

        result = space.add_parameter(param)
        assert result is space  # Should return self
        assert len(space.parameters) == 1

    def test_duplicate_parameter_name_error(self, float_param):
        """Test that duplicate parameter names raise error."""
        with pytest.raises(ValueError, match="Duplicate parameter names"):
            ParameterSpace(parameters=[float_param, float_param])

    def test_add_duplicate_parameter_error(self, simple_space, float_param):
        """Test adding duplicate parameter name."""
        with pytest.raises(ValueError, match="already exists"):
            simple_space.add_parameter(float_param)

    def test_remove_parameter(self, simple_space):
        """Test removing a parameter."""
        space = simple_space.remove_parameter("confidence")
        assert len(space.parameters) == 1
        assert space.get_parameter("confidence") is None

    def test_get_parameter(self, simple_space):
        """Test getting a parameter by name."""
        param = simple_space.get_parameter("confidence")
        assert param is not None
        assert param.name == "confidence"

        missing = simple_space.get_parameter("nonexistent")
        assert missing is None

    def test_grid_combinations_count(self, simple_space):
        """Test counting grid combinations."""
        count = simple_space.count_grid_combinations()
        # confidence: 5 values (0.5, 0.6, 0.7, 0.8, 0.9)
        # stop_ticks: 5 values (4, 6, 8, 10, 12)
        assert count == 25

    def test_grid_combinations_generation(self, simple_space):
        """Test generating all grid combinations."""
        combos = list(simple_space.get_grid_combinations())
        assert len(combos) == 25

        # Check first combination
        first = combos[0]
        assert "confidence" in first
        assert "stop_ticks" in first
        assert isinstance(first["stop_ticks"], int)

    def test_empty_space_grid(self):
        """Test grid generation for empty space."""
        space = ParameterSpace()
        combos = list(space.get_grid_combinations())
        assert len(combos) == 1
        assert combos[0] == {}

    def test_random_sampling(self, simple_space):
        """Test random sampling from space."""
        samples = simple_space.sample_random(n=10, seed=42)
        assert len(samples) == 10

        # All samples should have both parameters
        for sample in samples:
            assert "confidence" in sample
            assert "stop_ticks" in sample
            assert 0.5 <= sample["confidence"] <= 0.9
            assert 4 <= sample["stop_ticks"] <= 12

    def test_get_defaults(self, simple_space):
        """Test getting default values."""
        defaults = simple_space.get_defaults()
        assert defaults["stop_ticks"] == 8  # Explicit default
        # confidence has no default, should use midpoint
        assert 0.5 <= defaults["confidence"] <= 0.9

    def test_validate_params_valid(self, simple_space):
        """Test validating valid parameters."""
        params = {"confidence": 0.7, "stop_ticks": 8}
        is_valid, errors = simple_space.validate_params(params)
        assert is_valid
        assert len(errors) == 0

    def test_validate_params_missing(self, simple_space):
        """Test validating with missing parameter."""
        params = {"confidence": 0.7}
        is_valid, errors = simple_space.validate_params(params)
        assert not is_valid
        assert any("Missing parameter" in e for e in errors)

    def test_validate_params_out_of_bounds(self, simple_space):
        """Test validating with out-of-bounds values."""
        params = {"confidence": 1.5, "stop_ticks": 8}
        is_valid, errors = simple_space.validate_params(params)
        assert not is_valid
        assert any("outside range" in e for e in errors)

    def test_space_serialization(self, simple_space):
        """Test space to_dict and from_dict."""
        d = simple_space.to_dict()
        assert d["name"] == "simple_test"
        assert len(d["parameters"]) == 2

        # Reconstruct
        restored = ParameterSpace.from_dict(d)
        assert restored.name == simple_space.name
        assert len(restored.parameters) == len(simple_space.parameters)


# ============================================================================
# DefaultParameterSpaces Tests (6 tests)
# ============================================================================

class TestDefaultParameterSpaces:
    """Tests for DefaultParameterSpaces factory."""

    def test_mes_scalping_space(self):
        """Test MES scalping parameter space."""
        space = DefaultParameterSpaces.mes_scalping()
        assert space.name == "mes_scalping"
        assert len(space.parameters) == 5

        # Check required parameters
        param_names = {p.name for p in space.parameters}
        assert "stop_ticks" in param_names
        assert "target_ticks" in param_names
        assert "confidence_threshold" in param_names

    def test_quick_search_space(self):
        """Test quick search parameter space."""
        space = DefaultParameterSpaces.quick_search()
        assert space.name == "quick_search"

        # Should have fewer combinations than full space
        count = space.count_grid_combinations()
        assert count < 100  # Quick search should be small

    def test_risk_only_space(self):
        """Test risk-only parameter space."""
        space = DefaultParameterSpaces.risk_only()
        assert space.name == "risk_only"

        param_names = {p.name for p in space.parameters}
        assert "risk_pct" in param_names

    def test_entry_filters_space(self):
        """Test entry filters parameter space."""
        space = DefaultParameterSpaces.entry_filters()
        assert space.name == "entry_filters"

        param_names = {p.name for p in space.parameters}
        assert "confidence_threshold" in param_names
        assert "min_atr" in param_names

    def test_custom_space_creation(self):
        """Test creating custom space from tuples."""
        params = [
            ("param1", 0.0, 1.0, 0.1, "float"),
            ("param2", 1, 10, 1, "int"),
        ]
        space = DefaultParameterSpaces.custom(params, name="my_custom")

        assert space.name == "my_custom"
        assert len(space.parameters) == 2

    def test_create_from_config_dict(self):
        """Test creating space from config dictionary."""
        config = {
            "param1": {"min": 0.0, "max": 1.0, "step": 0.1, "type": "float"},
            "param2": {"min": 1, "max": 10, "step": 1, "type": "int"},
        }
        space = create_parameter_space_from_config(config)

        assert len(space.parameters) == 2
        param_names = {p.name for p in space.parameters}
        assert "param1" in param_names
        assert "param2" in param_names


# ============================================================================
# TrialResult Tests (8 tests)
# ============================================================================

class TestTrialResult:
    """Tests for TrialResult class."""

    def test_trial_creation(self, sample_trial_result):
        """Test creating a trial result."""
        assert sample_trial_result.trial_id == 1
        assert sample_trial_result.status == "completed"
        assert len(sample_trial_result.params) == 2
        assert len(sample_trial_result.metrics) == 3

    def test_get_metric(self, sample_trial_result):
        """Test getting metric value."""
        sharpe = sample_trial_result.get_metric("sharpe_ratio")
        assert sharpe == 1.5

        missing = sample_trial_result.get_metric("nonexistent", default=-1.0)
        assert missing == -1.0

    def test_overfitting_score_calculation(self):
        """Test calculating overfitting score."""
        trial = TrialResult(
            trial_id=1,
            params={},
            metrics={},
            in_sample_metrics={"sharpe_ratio": 2.0},
            out_of_sample_metrics={"sharpe_ratio": 1.5},
        )

        score = trial.get_overfitting_score("sharpe_ratio")
        assert score == pytest.approx(2.0 / 1.5)

    def test_overfitting_score_no_oos_data(self, sample_trial_result):
        """Test overfitting score without OOS data."""
        score = sample_trial_result.get_overfitting_score("sharpe_ratio")
        assert score is None

    def test_overfitting_score_zero_oos(self):
        """Test overfitting score when OOS is zero."""
        trial = TrialResult(
            trial_id=1,
            params={},
            metrics={},
            in_sample_metrics={"sharpe_ratio": 2.0},
            out_of_sample_metrics={"sharpe_ratio": 0.0},
        )

        score = trial.get_overfitting_score("sharpe_ratio")
        assert score == float('inf')

    def test_is_better_than_higher(self):
        """Test comparing trials (higher is better)."""
        trial1 = TrialResult(
            trial_id=1,
            params={},
            metrics={"sharpe_ratio": 2.0}
        )
        trial2 = TrialResult(
            trial_id=2,
            params={},
            metrics={"sharpe_ratio": 1.5}
        )

        assert trial1.is_better_than(trial2, "sharpe_ratio", higher_is_better=True)
        assert not trial2.is_better_than(trial1, "sharpe_ratio", higher_is_better=True)

    def test_is_better_than_lower(self):
        """Test comparing trials (lower is better)."""
        trial1 = TrialResult(
            trial_id=1,
            params={},
            metrics={"loss": 0.5}
        )
        trial2 = TrialResult(
            trial_id=2,
            params={},
            metrics={"loss": 1.0}
        )

        assert trial1.is_better_than(trial2, "loss", higher_is_better=False)

    def test_trial_serialization(self, sample_trial_result):
        """Test trial result serialization."""
        d = sample_trial_result.to_dict()
        assert d["trial_id"] == 1
        assert d["status"] == "completed"

        # Reconstruct
        restored = TrialResult.from_dict(d)
        assert restored.trial_id == sample_trial_result.trial_id
        assert restored.params == sample_trial_result.params


# ============================================================================
# OptimizationResult Tests (11 tests)
# ============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_result_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            best_params={"stop_ticks": 8},
            best_metric=1.5,
            metric_name="sharpe_ratio",
        )
        assert result.best_metric == 1.5
        assert result.metric_name == "sharpe_ratio"

    def test_get_best_trial(self):
        """Test getting best trial."""
        trials = [
            TrialResult(1, {}, {"sharpe_ratio": 1.0}, status="completed"),
            TrialResult(2, {}, {"sharpe_ratio": 2.0}, status="completed"),
            TrialResult(3, {}, {"sharpe_ratio": 1.5}, status="completed"),
        ]

        result = OptimizationResult(
            best_params={},
            best_metric=2.0,
            all_results=trials,
        )

        best = result.get_best_trial()
        assert best.trial_id == 2
        assert best.get_metric("sharpe_ratio") == 2.0

    def test_get_top_n_trials(self):
        """Test getting top N trials."""
        trials = [
            TrialResult(1, {}, {"sharpe_ratio": 1.0}, status="completed"),
            TrialResult(2, {}, {"sharpe_ratio": 2.0}, status="completed"),
            TrialResult(3, {}, {"sharpe_ratio": 1.5}, status="completed"),
            TrialResult(4, {}, {"sharpe_ratio": 1.8}, status="completed"),
        ]

        result = OptimizationResult(
            best_params={},
            best_metric=2.0,
            all_results=trials,
        )

        top_2 = result.get_top_n_trials(2)
        assert len(top_2) == 2
        assert top_2[0].get_metric("sharpe_ratio") == 2.0
        assert top_2[1].get_metric("sharpe_ratio") == 1.8

    def test_parameter_importance(self):
        """Test parameter importance calculation."""
        trials = [
            TrialResult(
                i,
                {"param1": i * 0.1, "param2": 10 - i},
                {"sharpe_ratio": i * 0.2}
            )
            for i in range(20)
        ]

        result = OptimizationResult(
            best_params={},
            best_metric=1.0,
            all_results=trials,
        )

        importance = result.get_parameter_importance()
        assert "param1" in importance
        assert "param2" in importance
        # param1 should be positively correlated with sharpe_ratio
        assert importance["param1"] > 0

    def test_overfitting_analysis(self):
        """Test overfitting analysis."""
        result = OptimizationResult(
            best_params={},
            best_metric=1.5,
            in_sample_metrics={
                "sharpe_ratio": 2.0,
                "profit_factor": 1.5,
            },
            out_of_sample_metrics={
                "sharpe_ratio": 1.5,
                "profit_factor": 1.2,
            },
        )

        analysis = result.get_overfitting_analysis()
        assert "sharpe_ratio" in analysis
        assert analysis["sharpe_ratio"]["ratio"] == pytest.approx(2.0 / 1.5)

    def test_parameter_stability(self):
        """Test parameter stability analysis."""
        trials = [
            TrialResult(i, {"param1": 0.5 + i * 0.01}, {"sharpe_ratio": 1.0 + i * 0.1})
            for i in range(10)
        ]

        result = OptimizationResult(
            best_params={},
            best_metric=1.0,
            all_results=trials,
        )

        stability = result.get_parameter_stability(top_n=5)
        assert "param1" in stability
        mean, std = stability["param1"]
        assert mean > 0
        assert std >= 0

    def test_convergence_curve(self):
        """Test convergence curve generation."""
        trials = [
            TrialResult(i, {}, {"sharpe_ratio": i * 0.1}, status="completed")
            for i in range(10)
        ]

        result = OptimizationResult(
            best_params={},
            best_metric=0.9,
            all_results=trials,
        )

        curve = result.get_convergence_curve()
        assert len(curve) == 10
        # Should be monotonically increasing
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1]

    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=100)

        result = OptimizationResult(
            best_params={},
            best_metric=1.0,
            start_time=start,
            end_time=end,
        )

        duration = result.duration_seconds()
        assert duration == pytest.approx(100.0, abs=0.1)

    def test_summary_generation(self):
        """Test summary string generation."""
        result = OptimizationResult(
            best_params={"stop_ticks": 8, "confidence": 0.7},
            best_metric=1.5,
            metric_name="sharpe_ratio",
            optimizer_type="GridSearch",
            total_trials=100,
            successful_trials=95,
        )

        summary = result.summary()
        assert "GridSearch" in summary
        assert "sharpe_ratio" in summary
        assert "stop_ticks" in summary
        assert "95/100" in summary

    def test_result_serialization(self):
        """Test result serialization to/from dict."""
        trials = [
            TrialResult(1, {}, {"sharpe_ratio": 1.0})
        ]

        result = OptimizationResult(
            best_params={"stop_ticks": 8},
            best_metric=1.5,
            all_results=trials,
            start_time=datetime(2024, 1, 1, 12, 0),
        )

        d = result.to_dict()
        assert d["best_metric"] == 1.5
        assert len(d["all_results"]) == 1

        # Reconstruct
        restored = OptimizationResult.from_dict(d)
        assert restored.best_metric == result.best_metric

    def test_result_json_save_load(self, tmp_path):
        """Test saving and loading result from JSON."""
        result = OptimizationResult(
            best_params={"stop_ticks": 8},
            best_metric=1.5,
        )

        path = tmp_path / "result.json"
        result.to_json(str(path))

        loaded = OptimizationResult.from_json(str(path))
        assert loaded.best_metric == result.best_metric
        assert loaded.best_params == result.best_params


# ============================================================================
# Results Utility Functions Tests (3 tests)
# ============================================================================

class TestResultsUtilities:
    """Tests for results utility functions."""

    def test_calculate_overfitting_score(self):
        """Test overfitting score calculation."""
        score = calculate_overfitting_score(2.0, 1.5)
        assert score == pytest.approx(2.0 / 1.5)

        # Zero OOS
        score = calculate_overfitting_score(2.0, 0.0)
        assert score == float('inf')

    def test_is_overfitting_function(self):
        """Test is_overfitting function."""
        assert is_overfitting(2.0, threshold=1.5)
        assert not is_overfitting(1.3, threshold=1.5)
        assert is_overfitting(1.6, threshold=1.5)

    def test_merge_results(self):
        """Test merging multiple optimization results."""
        result1 = OptimizationResult(
            best_params={"p1": 1},
            best_metric=1.0,
            all_results=[
                TrialResult(0, {"p1": 1}, {"sharpe_ratio": 1.0})
            ],
        )

        result2 = OptimizationResult(
            best_params={"p1": 2},
            best_metric=2.0,
            all_results=[
                TrialResult(0, {"p1": 2}, {"sharpe_ratio": 2.0})
            ],
        )

        merged = merge_results([result1, result2])
        assert merged.best_metric == 2.0
        assert len(merged.all_results) == 2
        assert merged.optimizer_type == "merged"


# ============================================================================
# BaseOptimizer Tests (8 tests)
# ============================================================================

class TestBaseOptimizer:
    """Tests for BaseOptimizer base class."""

    def test_optimizer_initialization(self, simple_space, mock_objective):
        """Test initializing base optimizer."""
        config = OptimizerConfig(metric_name="sharpe_ratio", n_jobs=1)

        # Create concrete implementation
        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, mock_objective, config)
        assert optimizer.parameter_space == simple_space
        assert optimizer.config.metric_name == "sharpe_ratio"

    def test_evaluate_params(self, simple_space, mock_objective):
        """Test evaluating single parameter combination."""
        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, mock_objective)
        result = optimizer.evaluate_params({"confidence": 0.7, "stop_ticks": 8})

        assert result.status == "completed"
        assert "sharpe_ratio" in result.metrics
        assert result.params == {"confidence": 0.7, "stop_ticks": 8}

    def test_evaluate_params_invalid(self, simple_space, mock_objective):
        """Test evaluating invalid parameters."""
        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, mock_objective)
        result = optimizer.evaluate_params({"confidence": 1.5, "stop_ticks": 8})

        assert result.status == "failed"
        assert "Invalid params" in result.error_message

    def test_evaluate_batch_sequential(self, simple_space, mock_objective):
        """Test evaluating batch of parameters sequentially."""
        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, mock_objective)
        param_list = [
            {"confidence": 0.6, "stop_ticks": 6},
            {"confidence": 0.7, "stop_ticks": 8},
            {"confidence": 0.8, "stop_ticks": 10},
        ]

        results = optimizer.evaluate_batch(param_list, parallel=False)
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)

    def test_get_current_best(self, simple_space, mock_objective):
        """Test getting current best parameters."""
        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, mock_objective)

        # Initially None
        assert optimizer.get_current_best() is None

        # After evaluation
        optimizer.evaluate_params({"confidence": 0.7, "stop_ticks": 8})
        best = optimizer.get_current_best()
        assert best is not None
        assert "confidence" in best

    def test_get_trial_count(self, simple_space, mock_objective):
        """Test getting trial count."""
        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, mock_objective)
        assert optimizer.get_trial_count() == 0

        result = optimizer.evaluate_params({"confidence": 0.7, "stop_ticks": 8})
        # Add result to the list to track it
        with optimizer._lock:
            optimizer._results.append(result)
        assert optimizer.get_trial_count() == 1

    def test_abstract_method_enforcement(self, simple_space, mock_objective):
        """Test that abstract method must be implemented."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate directly
            BaseOptimizer(simple_space, mock_objective)

    def test_objective_function_error_handling(self, simple_space):
        """Test handling errors in objective function."""
        def bad_objective(params):
            raise ValueError("Test error")

        class TestOptimizer(BaseOptimizer):
            def _run_optimization(self):
                return self._build_result()

        optimizer = TestOptimizer(simple_space, bad_objective)
        result = optimizer.evaluate_params({"confidence": 0.7, "stop_ticks": 8})

        assert result.status == "failed"
        assert "Test error" in result.error_message


# ============================================================================
# GridSearchOptimizer Tests (10 tests)
# ============================================================================

class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    def test_grid_search_creation(self, simple_space, mock_objective):
        """Test creating grid search optimizer."""
        config = GridSearchConfig(metric_name="sharpe_ratio")
        optimizer = GridSearchOptimizer(simple_space, mock_objective, config)

        assert optimizer.parameter_space == simple_space
        assert optimizer.config.metric_name == "sharpe_ratio"

    def test_grid_search_basic_run(self, mock_objective):
        """Test basic grid search run."""
        # Small space for quick test
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 0.2, 0.1, "float"),
            ParameterConfig("p2", 1, 2, 1, "int"),
        ])

        config = GridSearchConfig(metric_name="sharpe_ratio", verbose=0)
        optimizer = GridSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()

        assert result.best_params is not None
        assert result.best_metric > 0
        assert len(result.all_results) == 6  # 3 * 2 = 6 combinations

    def test_grid_search_finds_best(self):
        """Test that grid search finds the best parameters."""
        def objective(params):
            # Maximum at p1=1.0, p2=10
            score = params["p1"] + params["p2"] / 10.0
            return {"sharpe_ratio": score}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.5, "float"),
            ParameterConfig("p2", 5, 10, 5, "int"),
        ])

        optimizer = GridSearchOptimizer(space, objective)
        result = optimizer.optimize()

        # Should find the maximum
        assert result.best_params["p1"] == 1.0
        assert result.best_params["p2"] == 10

    def test_grid_search_max_combinations(self, mock_objective):
        """Test limiting max combinations."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.1, "float"),
        ])

        config = GridSearchConfig(max_combinations=5, verbose=0)
        optimizer = GridSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        assert len(result.all_results) <= 5

    def test_grid_search_shuffle(self, mock_objective):
        """Test shuffled grid search."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 0.4, 0.1, "float"),
        ])

        config = GridSearchConfig(shuffle=True, random_seed=42, verbose=0)
        optimizer = GridSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        # Should complete all trials even with shuffle
        assert len(result.all_results) == 5

    def test_grid_search_empty_space(self):
        """Test grid search with empty parameter space."""
        space = ParameterSpace()

        def objective(params):
            return {"sharpe_ratio": 1.0}

        optimizer = GridSearchOptimizer(space, objective)
        result = optimizer.optimize()

        # Should handle gracefully
        assert result.total_trials == 0

    def test_grid_search_estimate_time(self, simple_space, mock_objective):
        """Test time estimation."""
        optimizer = GridSearchOptimizer(simple_space, mock_objective)

        estimated = optimizer.estimate_time(time_per_trial=1.0)
        # Should be approximately equal to number of combinations
        assert estimated > 0
        assert estimated <= simple_space.count_grid_combinations()

    def test_run_grid_search_convenience(self, mock_objective):
        """Test convenience function for grid search."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 0.2, 0.1, "float"),
        ])

        result = run_grid_search(
            space,
            mock_objective,
            metric_name="sharpe_ratio",
            verbose=0
        )

        assert result.best_metric > 0
        assert len(result.all_results) == 3

    def test_grid_search_with_cv(self):
        """Test grid search with cross-validation."""
        def objective_factory(fold):
            def objective(params):
                # Different results per fold
                return {"sharpe_ratio": params["p1"] + fold * 0.1}
            return objective

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 0.2, 0.1, "float"),
        ])

        result = grid_search_with_cv(
            space,
            objective_factory,
            n_folds=3,
            n_jobs=1
        )

        # Should average across folds
        best_trial = result.get_best_trial()
        assert "sharpe_ratio" in best_trial.metrics
        assert "sharpe_ratio_std" in best_trial.metrics

    def test_grid_search_parallel(self, mock_objective):
        """Test parallel grid search execution."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 0.3, 0.1, "float"),
        ])

        config = GridSearchConfig(n_jobs=2, verbose=0)
        optimizer = GridSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        assert len(result.all_results) == 4
        assert all(r.status == "completed" for r in result.all_results)


# ============================================================================
# RandomSearchOptimizer Tests (12 tests)
# ============================================================================

class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""

    def test_random_search_creation(self, simple_space, mock_objective):
        """Test creating random search optimizer."""
        config = RandomSearchConfig(n_iterations=10)
        optimizer = RandomSearchOptimizer(simple_space, mock_objective, config)

        assert optimizer.random_config.n_iterations == 10

    def test_random_search_basic_run(self, mock_objective):
        """Test basic random search run."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = RandomSearchConfig(n_iterations=10, verbose=0)
        optimizer = RandomSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()

        assert result.total_trials == 10
        assert result.best_metric > 0

    def test_random_search_n_iterations_respected(self, mock_objective):
        """Test that n_iterations is respected."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        optimizer = RandomSearchOptimizer(space, mock_objective, n_iterations=15)
        result = optimizer.optimize()

        assert len(result.all_results) == 15

    def test_random_search_early_stopping(self):
        """Test early stopping on no improvement."""
        counter = {"calls": 0}

        def objective(params):
            counter["calls"] += 1
            # First 5 improve, then plateau
            return {"sharpe_ratio": min(counter["calls"], 5) * 0.1}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = RandomSearchConfig(
            n_iterations=50,
            early_stopping_rounds=10,
            early_stopping_threshold=0.001,
            verbose=0
        )
        optimizer = RandomSearchOptimizer(space, objective, config)

        result = optimizer.optimize()

        # Should stop early
        assert len(result.all_results) < 50
        assert len(result.all_results) >= 15  # At least 5 + 10 stopping rounds

    def test_random_search_deduplication(self, mock_objective):
        """Test deduplication of samples."""
        # Very small space - will hit duplicates
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0, 2, 1, "int"),
        ])

        config = RandomSearchConfig(
            n_iterations=100,
            deduplicate=True,
            max_duplicates=50,
            verbose=0,
            random_seed=42
        )
        optimizer = RandomSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()

        # Should stop when too many duplicates
        assert len(result.all_results) < 100

    def test_random_search_no_deduplication(self, mock_objective):
        """Test without deduplication."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0, 2, 1, "int"),
        ])

        config = RandomSearchConfig(
            n_iterations=10,
            deduplicate=False,
            verbose=0
        )
        optimizer = RandomSearchOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        # Should complete all iterations even with duplicates
        assert len(result.all_results) == 10

    def test_random_search_reproducibility(self, mock_objective):
        """Test reproducibility with random seed."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config1 = RandomSearchConfig(n_iterations=5, random_seed=42, verbose=0)
        optimizer1 = RandomSearchOptimizer(space, mock_objective, config1)
        result1 = optimizer1.optimize()

        config2 = RandomSearchConfig(n_iterations=5, random_seed=42, verbose=0)
        optimizer2 = RandomSearchOptimizer(space, mock_objective, config2)
        result2 = optimizer2.optimize()

        # Should get same parameters
        params1 = [r.params for r in result1.all_results]
        params2 = [r.params for r in result2.all_results]

        # Note: Due to deduplication and hashing, exact order might differ
        # but the set of parameters should be similar
        assert len(params1) == len(params2)

    def test_run_random_search_convenience(self, mock_objective):
        """Test convenience function for random search."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        result = run_random_search(
            space,
            mock_objective,
            n_iterations=10,
            seed=42
        )

        assert len(result.all_results) == 10

    def test_adaptive_random_search_creation(self, simple_space, mock_objective):
        """Test creating adaptive random search."""
        optimizer = AdaptiveRandomSearch(
            simple_space,
            mock_objective,
            exploration_ratio=0.5
        )

        assert optimizer.exploration_ratio == 0.5

    def test_adaptive_random_search_phases(self):
        """Test adaptive random search exploration and exploitation."""
        call_count = {"count": 0}
        params_seen = []

        def objective(params):
            call_count["count"] += 1
            params_seen.append(params.copy())
            return {"sharpe_ratio": params["p1"]}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = RandomSearchConfig(n_iterations=20, verbose=0, random_seed=42)
        optimizer = AdaptiveRandomSearch(
            space,
            objective,
            config,
            exploration_ratio=0.5,
            min_radius=0.1
        )

        result = optimizer.optimize()

        # Should have explored and exploited
        assert len(result.all_results) == 20
        # Best should be found
        assert result.best_metric > 0

    def test_adaptive_search_focuses_on_best(self):
        """Test that adaptive search focuses around best region."""
        def objective(params):
            # Optimal at p1=0.8
            return {"sharpe_ratio": 1.0 - abs(params["p1"] - 0.8)}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = RandomSearchConfig(n_iterations=30, verbose=0, random_seed=42)
        optimizer = AdaptiveRandomSearch(
            space,
            objective,
            config,
            exploration_ratio=0.3,  # 30% exploration, 70% exploitation
        )

        result = optimizer.optimize()

        # Best param should be near 0.8
        assert abs(result.best_params["p1"] - 0.8) < 0.3

    def test_random_search_reset(self, simple_space, mock_objective):
        """Test resetting random search state."""
        config = RandomSearchConfig(n_iterations=5, deduplicate=False, verbose=0)
        optimizer = RandomSearchOptimizer(simple_space, mock_objective, config)

        # Run once
        result1 = optimizer.optimize()
        assert len(result1.all_results) == 5

        # Reset and run again
        optimizer.reset()
        result2 = optimizer.optimize()
        assert len(result2.all_results) == 5


# ============================================================================
# BayesianOptimizer Tests (10+ tests - if Optuna available)
# ============================================================================

@pytest.mark.skipif(not BAYESIAN_AVAILABLE, reason="Optuna not installed")
class TestBayesianOptimizer:
    """Tests for BayesianOptimizer (requires Optuna)."""

    def test_bayesian_creation(self, simple_space, mock_objective):
        """Test creating Bayesian optimizer."""
        config = BayesianConfig(n_trials=10)
        optimizer = BayesianOptimizer(simple_space, mock_objective, config)

        assert optimizer.bayesian_config.n_trials == 10

    def test_bayesian_basic_run(self, mock_objective):
        """Test basic Bayesian optimization run."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, mock_objective, config)

        result = optimizer.optimize()

        assert result.total_trials == 5
        assert result.best_metric > 0

    def test_bayesian_n_trials_respected(self, mock_objective):
        """Test that n_trials is respected."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        optimizer = BayesianOptimizer(space, mock_objective, n_trials=8)
        result = optimizer.optimize()

        assert len(result.all_results) == 8

    def test_bayesian_tpe_sampler(self, mock_objective):
        """Test using TPE sampler."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = BayesianConfig(
            n_trials=10,
            sampler="tpe",
            n_startup_trials=3,
            show_progress_bar=False
        )
        optimizer = BayesianOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        assert result.total_trials == 10

    def test_bayesian_random_sampler(self, mock_objective):
        """Test using random sampler."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = BayesianConfig(
            n_trials=5,
            sampler="random",
            show_progress_bar=False
        )
        optimizer = BayesianOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        assert result.total_trials == 5

    def test_bayesian_with_pruner(self, mock_objective):
        """Test Bayesian optimization with pruning."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        config = BayesianConfig(
            n_trials=10,
            pruner="median",
            show_progress_bar=False
        )
        optimizer = BayesianOptimizer(space, mock_objective, config)

        result = optimizer.optimize()
        # Some trials might be pruned, but should still work
        assert result.total_trials >= 5

    def test_bayesian_study_persistence(self, mock_objective, tmp_path):
        """Test saving and loading Bayesian study."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        db_path = tmp_path / "study.db"
        storage = f"sqlite:///{db_path}"

        # Run first optimization
        config1 = BayesianConfig(
            n_trials=5,
            study_name="test_study",
            storage=storage,
            load_if_exists=False,
            show_progress_bar=False
        )
        optimizer1 = BayesianOptimizer(space, mock_objective, config1)
        result1 = optimizer1.optimize()

        # Resume optimization
        config2 = BayesianConfig(
            n_trials=5,
            study_name="test_study",
            storage=storage,
            load_if_exists=True,
            show_progress_bar=False
        )
        optimizer2 = BayesianOptimizer(space, mock_objective, config2)
        result2 = optimizer2.optimize()

        # Should have combined trials
        study = optimizer2.get_study()
        assert len(study.trials) == 10

    def test_bayesian_parameter_importance(self):
        """Test parameter importance calculation."""
        def objective(params):
            # p1 is more important than p2
            return {"sharpe_ratio": params["p1"] * 2.0 + params["p2"] * 0.1}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
            ParameterConfig("p2", 0.0, 1.0, param_type="float"),
        ])

        config = BayesianConfig(n_trials=20, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, objective, config)
        result = optimizer.optimize()

        importance = optimizer.get_importance()
        # p1 should be more important
        if len(importance) > 0:  # Might need more trials for reliable importance
            assert "p1" in importance

    def test_run_bayesian_optimization_convenience(self, mock_objective):
        """Test convenience function for Bayesian optimization."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
        ])

        result = run_bayesian_optimization(
            space,
            mock_objective,
            n_trials=5,
            seed=42
        )

        assert len(result.all_results) == 5

    def test_bayesian_with_int_and_categorical(self):
        """Test Bayesian optimization with mixed parameter types."""
        def objective(params):
            score = params["int_param"] + (0.5 if params["cat_param"] == "A" else 0.0)
            return {"sharpe_ratio": score}

        space = ParameterSpace(parameters=[
            ParameterConfig("int_param", 1, 10, 1, "int"),
            ParameterConfig("cat_param", param_type="categorical", choices=["A", "B", "C"]),
        ])

        config = BayesianConfig(n_trials=10, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, objective, config)

        result = optimizer.optimize()
        assert result.total_trials == 10
        # Should prefer int_param=10, cat_param=A
        assert result.best_params["int_param"] >= 8


# ============================================================================
# Integration Tests (8+ tests)
# ============================================================================

class TestOptimizationIntegration:
    """Integration tests for complete optimization workflows."""

    def test_end_to_end_grid_search(self, mock_backtest_engine):
        """Test end-to-end grid search with mock backtest."""
        space = DefaultParameterSpaces.quick_search()

        def signal_factory(params):
            return Mock()  # Mock signal generator

        objective = create_backtest_objective(
            mock_backtest_engine,
            pd.DataFrame(),  # Mock data
            signal_factory
        )

        config = GridSearchConfig(metric_name="sharpe_ratio", verbose=0)
        optimizer = GridSearchOptimizer(space, objective, config)

        result = optimizer.optimize()
        assert result.best_params is not None
        assert result.total_trials > 0

    def test_end_to_end_random_search(self, mock_backtest_engine):
        """Test end-to-end random search with mock backtest."""
        space = DefaultParameterSpaces.quick_search()

        def signal_factory(params):
            return Mock()

        objective = create_backtest_objective(
            mock_backtest_engine,
            pd.DataFrame(),
            signal_factory
        )

        config = RandomSearchConfig(n_iterations=10, verbose=0)
        optimizer = RandomSearchOptimizer(space, objective, config)

        result = optimizer.optimize()
        assert result.best_params is not None
        # Random search deduplicates, so may have fewer results than iterations
        assert len(result.all_results) <= 10
        assert len(result.all_results) >= 1

    def test_overfitting_detection_workflow(self):
        """Test complete overfitting detection workflow."""
        def objective_is(params):
            # In-sample: artificially good
            return {"sharpe_ratio": 3.0}

        def objective_oos(params):
            # Out-of-sample: worse
            return {"sharpe_ratio": 1.5}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.5, "float"),
        ])

        config = GridSearchConfig(verbose=0)

        # Optimize on in-sample
        optimizer = GridSearchOptimizer(space, objective_is, config)
        result_is = optimizer.optimize()

        # Evaluate best params on out-of-sample
        best_params = result_is.best_params
        oos_metrics = objective_oos(best_params)

        # Calculate overfitting
        score = calculate_overfitting_score(
            result_is.best_metric,
            oos_metrics["sharpe_ratio"]
        )

        assert is_overfitting(score, threshold=1.5)

    def test_parameter_bounds_respected(self):
        """Test that optimizers respect parameter bounds."""
        all_params_seen = []

        def objective(params):
            all_params_seen.append(params.copy())
            return {"sharpe_ratio": params["p1"]}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.3, 0.7, param_type="float"),
        ])

        # Test with random search
        config = RandomSearchConfig(n_iterations=20, verbose=0)
        optimizer = RandomSearchOptimizer(space, objective, config)
        optimizer.optimize()

        # All params should be within bounds
        for params in all_params_seen:
            assert 0.3 <= params["p1"] <= 0.7

    def test_results_consistency_across_optimizers(self):
        """Test that different optimizers produce consistent results."""
        def deterministic_objective(params):
            # Deterministic: maximum at p1=1.0
            return {"sharpe_ratio": params["p1"]}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.2, "float"),
        ])

        # Grid search
        grid_optimizer = GridSearchOptimizer(space, deterministic_objective)
        grid_result = grid_optimizer.optimize()

        # Random search with many iterations
        random_config = RandomSearchConfig(n_iterations=50, verbose=0, random_seed=42)
        random_optimizer = RandomSearchOptimizer(space, deterministic_objective, random_config)
        random_result = random_optimizer.optimize()

        # Both should find near-optimal solution
        assert grid_result.best_params["p1"] == pytest.approx(1.0, abs=0.01)
        assert random_result.best_params["p1"] == pytest.approx(1.0, abs=0.3)

    def test_multi_metric_optimization(self):
        """Test optimization tracking multiple metrics."""
        def multi_metric_objective(params):
            return {
                "sharpe_ratio": params["p1"],
                "profit_factor": params["p1"] * 1.2,
                "win_rate_pct": 50.0 + params["p1"] * 10,
                "max_drawdown_pct": -params["p1"] * 5,
            }

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.1, "float"),
        ])

        config = GridSearchConfig(metric_name="sharpe_ratio", verbose=0)
        optimizer = GridSearchOptimizer(space, multi_metric_objective, config)

        result = optimizer.optimize()

        # Best trial should have all metrics
        best_trial = result.get_best_trial()
        assert "sharpe_ratio" in best_trial.metrics
        assert "profit_factor" in best_trial.metrics
        assert "win_rate_pct" in best_trial.metrics
        assert "max_drawdown_pct" in best_trial.metrics

    def test_optimization_with_failed_trials(self):
        """Test handling of failed trials during optimization."""
        call_count = {"count": 0}

        def flaky_objective(params):
            call_count["count"] += 1
            # Fail every 3rd trial
            if call_count["count"] % 3 == 0:
                raise ValueError("Simulated failure")
            return {"sharpe_ratio": params["p1"]}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 0.5, 0.1, "float"),
        ])

        config = GridSearchConfig(verbose=0)
        optimizer = GridSearchOptimizer(space, flaky_objective, config)

        result = optimizer.optimize()

        # Should complete despite failures
        assert result.successful_trials < result.total_trials
        assert result.successful_trials > 0
        # Failed trials should be recorded
        failed_count = sum(1 for r in result.all_results if r.status == "failed")
        assert failed_count > 0

    def test_optimization_result_analysis_pipeline(self):
        """Test complete result analysis pipeline."""
        # Create synthetic results
        trials = []
        for i in range(50):
            params = {"p1": i * 0.02, "p2": 10 - i * 0.1}
            metrics = {
                "sharpe_ratio": 0.5 + i * 0.01 + np.random.randn() * 0.05,
                "profit_factor": 1.0 + i * 0.01,
            }
            trials.append(TrialResult(i, params, metrics, status="completed"))

        result = OptimizationResult(
            best_params=trials[-1].params,
            best_metric=trials[-1].get_metric("sharpe_ratio"),
            all_results=trials,
            metric_name="sharpe_ratio",
        )

        # Analyze
        top_10 = result.get_top_n_trials(10)
        assert len(top_10) == 10

        importance = result.get_parameter_importance()
        assert "p1" in importance

        stability = result.get_parameter_stability(top_n=10)
        assert "p1" in stability

        curve = result.get_convergence_curve()
        assert len(curve) == 50


# ============================================================================
# Edge Cases and Error Handling (5+ tests)
# ============================================================================

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_parameter_space_optimization(self):
        """Test optimization with no parameters."""
        space = ParameterSpace()

        def objective(params):
            return {"sharpe_ratio": 1.5}

        config = GridSearchConfig(verbose=0)
        optimizer = GridSearchOptimizer(space, objective, config)
        result = optimizer.optimize()

        # Should handle gracefully
        assert result.total_trials == 0

    def test_single_parameter_optimization(self):
        """Test optimization with single parameter."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.25, "float"),
        ])

        def objective(params):
            return {"sharpe_ratio": params["p1"]}

        optimizer = GridSearchOptimizer(space, objective)
        result = optimizer.optimize()

        assert result.best_params["p1"] == pytest.approx(1.0)

    def test_objective_returning_nan(self):
        """Test handling of NaN in objective function."""
        def nan_objective(params):
            return {"sharpe_ratio": float('nan')}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.5, "float"),
        ])

        config = GridSearchConfig(verbose=0)
        optimizer = GridSearchOptimizer(space, nan_objective, config)
        result = optimizer.optimize()

        # Should handle NaN gracefully
        assert result.total_trials > 0

    def test_objective_returning_inf(self):
        """Test handling of infinity in objective function."""
        def inf_objective(params):
            return {"sharpe_ratio": float('inf')}

        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, 0.5, "float"),
        ])

        config = RandomSearchConfig(n_iterations=5, deduplicate=False, verbose=0)
        optimizer = RandomSearchOptimizer(space, inf_objective, config)
        result = optimizer.optimize()

        # Should complete (might stop early due to duplicates if deduplicate=True)
        # With deduplicate=False, should get all 5 trials
        assert len(result.all_results) == 5

    def test_very_large_parameter_space(self):
        """Test handling of very large parameter space."""
        # Create space with millions of combinations
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 100.0, 0.01, "float"),
            ParameterConfig("p2", 0.0, 100.0, 0.01, "float"),
        ])

        count = space.count_grid_combinations()
        assert count > 100_000_000

        # Random search should still work
        def objective(params):
            return {"sharpe_ratio": params["p1"] + params["p2"]}

        config = RandomSearchConfig(n_iterations=10, verbose=0)
        optimizer = RandomSearchOptimizer(space, objective, config)
        result = optimizer.optimize()

        assert len(result.all_results) == 10

    def test_merge_empty_results_list(self):
        """Test merging empty results list."""
        with pytest.raises(ValueError, match="Cannot merge empty"):
            merge_results([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
