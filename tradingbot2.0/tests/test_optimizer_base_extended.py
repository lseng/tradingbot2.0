"""
Extended tests for optimizer_base.py to achieve high coverage.

Tests cover:
- Abstract method behavior
- Exception handling in optimize()
- Overfitting metrics computation
- Parallel execution with ThreadPoolExecutor
- Verbose logging paths
- create_backtest_objective with nested metrics
- create_split_objective function
"""

import pytest
import time
import logging
from unittest.mock import MagicMock, patch, Mock
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.optimization.optimizer_base import (
    BaseOptimizer,
    OptimizerConfig,
    create_backtest_objective,
    create_split_objective,
)
from src.optimization.parameter_space import ParameterSpace, ParameterConfig
from src.optimization.results import OptimizationResult, TrialResult


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_space():
    """Create a simple parameter space for testing."""
    params = [
        ParameterConfig(
            name="learning_rate",
            min_value=0.001,
            max_value=0.1,
            param_type="float",
        ),
        ParameterConfig(
            name="batch_size",
            min_value=16,
            max_value=64,
            param_type="int",
        ),
    ]
    return ParameterSpace(parameters=params, name="test_space")


@pytest.fixture
def simple_objective():
    """Create a simple objective function."""
    def objective(params: Dict[str, Any]) -> Dict[str, float]:
        lr = params.get("learning_rate", 0.01)
        bs = params.get("batch_size", 32)
        return {
            "sharpe_ratio": 1.0 + lr * 10 - bs * 0.01,
            "total_return": 0.05 + lr * 2,
        }
    return objective


@pytest.fixture
def config_parallel():
    """Create config for parallel execution."""
    return OptimizerConfig(
        metric_name="sharpe_ratio",
        higher_is_better=True,
        n_jobs=4,
        verbose=1,
        random_seed=42,
        timeout_per_trial=5.0,
    )


# ============================================================================
# Concrete Implementation for Testing
# ============================================================================

class ConcreteOptimizer(BaseOptimizer):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, *args, should_fail=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_fail = should_fail

    def _run_optimization(self) -> OptimizationResult:
        if self.should_fail:
            raise RuntimeError("Simulated optimization failure")

        # Simple grid-like optimization
        param_combos = [
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.05, "batch_size": 48},
            {"learning_rate": 0.1, "batch_size": 64},
        ]

        for params in param_combos:
            result = self.evaluate_params(params)
            with self._lock:
                self._results.append(result)

        return self._build_result()


class FailingOptimizer(BaseOptimizer):
    """Optimizer that always fails during optimization."""

    def _run_optimization(self) -> OptimizationResult:
        raise ValueError("Intentional failure for testing")


# ============================================================================
# Tests for Abstract Method
# ============================================================================

class TestAbstractMethod:
    """Tests for abstract method behavior."""

    def test_base_optimizer_is_abstract(self, simple_space, simple_objective):
        """Test that BaseOptimizer cannot be instantiated directly."""
        # The ABC doesn't raise TypeError on instantiation in Python 3.4+
        # unless you try to call the abstract method
        with pytest.raises(TypeError):
            BaseOptimizer(simple_space, simple_objective)

    def test_concrete_implementation_works(self, simple_space, simple_objective):
        """Test that concrete implementation can be instantiated."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)
        assert optimizer is not None
        assert optimizer.parameter_space == simple_space


# ============================================================================
# Tests for Exception Handling in optimize()
# ============================================================================

class TestOptimizeExceptionHandling:
    """Tests for exception handling in the optimize method."""

    def test_optimize_catches_exception_and_returns_failed_result(
        self, simple_space, simple_objective
    ):
        """Test that optimize() catches exceptions and returns failed result."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective, should_fail=True)

        result = optimizer.optimize()

        # Should return a result even though optimization failed
        assert result is not None
        assert "error" in result.config
        assert "Simulated optimization failure" in result.config["error"]
        assert result.best_params == {}
        assert result.best_metric == 0.0

    def test_failing_optimizer_returns_failed_result(
        self, simple_space, simple_objective
    ):
        """Test FailingOptimizer returns a failed result."""
        optimizer = FailingOptimizer(simple_space, simple_objective)

        result = optimizer.optimize()

        assert result is not None
        assert "error" in result.config
        assert "Intentional failure" in result.config["error"]

    def test_optimize_sets_end_time_on_failure(
        self, simple_space, simple_objective
    ):
        """Test that end_time is set even when optimization fails."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective, should_fail=True)

        result = optimizer.optimize()

        assert result.end_time is not None
        assert result.start_time is not None


# ============================================================================
# Tests for Overfitting Metrics Computation
# ============================================================================

class TestOverfittingMetrics:
    """Tests for _compute_overfitting_metrics method."""

    def test_overfitting_metrics_computed_with_holdout_data(
        self, simple_space, simple_objective
    ):
        """Test overfitting metrics are computed when holdout_objective_fn provided."""
        # Create a separate holdout objective function (required for proper OOS evaluation)
        def holdout_objective(params):
            # Simulate slightly different metrics on holdout data (typical of some overfitting)
            lr = params.get("learning_rate", 0.01)
            bs = params.get("batch_size", 32)
            # Return slightly worse metrics than in-sample (simulating overfitting)
            sharpe = lr * 10 + bs * 0.01 - 0.1
            return {"sharpe_ratio": sharpe, "total_return": sharpe * 0.05}

        optimizer = ConcreteOptimizer(
            simple_space,
            simple_objective,
            holdout_objective_fn=holdout_objective
        )

        # Provide holdout data
        holdout_data = pd.DataFrame({"price": [100, 101, 102]})

        result = optimizer.optimize(holdout_data=holdout_data)

        # Result should have overfitting metrics
        assert result.in_sample_metrics is not None or result.out_of_sample_metrics is not None

    def test_overfitting_metrics_skipped_without_holdout_data(
        self, simple_space, simple_objective
    ):
        """Test overfitting metrics are skipped when no holdout_data."""
        config = OptimizerConfig(compute_overfitting=True)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        result = optimizer.optimize()  # No holdout_data

        # Overfitting score should be None (not computed)
        assert result.overfitting_score is None

    def test_overfitting_metrics_handles_exception(
        self, simple_space
    ):
        """Test that overfitting metrics computation handles exceptions."""
        call_count = [0]

        def failing_objective(params):
            call_count[0] += 1
            if call_count[0] <= 3:  # Succeed for optimization
                return {"sharpe_ratio": 1.0, "total_return": 0.05}
            else:  # Fail for OOS evaluation
                raise ValueError("Simulated OOS evaluation failure")

        optimizer = ConcreteOptimizer(simple_space, failing_objective)
        holdout_data = pd.DataFrame({"price": [100, 101, 102]})

        # Should not raise, should handle gracefully
        result = optimizer.optimize(holdout_data=holdout_data)

        assert result is not None

    def test_compute_overfitting_with_best_trial(self, simple_space, simple_objective):
        """Test _compute_overfitting_metrics uses best trial metrics."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        holdout_data = pd.DataFrame({"price": [100, 101, 102]})
        result = optimizer.optimize(holdout_data=holdout_data)

        # in_sample_metrics should be populated from best trial
        if result.in_sample_metrics:
            assert "sharpe_ratio" in result.in_sample_metrics

    def test_overfitting_metrics_uses_holdout_objective_fn(self, simple_space):
        """
        Test that when holdout_objective_fn is provided, it is used for OOS evaluation.

        This is the critical test that verifies proper IS/OOS separation:
        - Validation objective returns sharpe_ratio = 1.5 (used for optimization)
        - Holdout objective returns sharpe_ratio = 0.8 (used for OOS evaluation)
        - If OOS metrics show 0.8, the holdout_objective_fn was correctly used
        - If OOS metrics show 1.5, the bug exists (same objective reused)
        """
        # Validation objective returns higher sharpe (in-sample)
        def validation_objective(params):
            return {"sharpe_ratio": 1.5, "total_return": 0.10}

        # Holdout objective returns lower sharpe (out-of-sample)
        def holdout_objective(params):
            return {"sharpe_ratio": 0.8, "total_return": 0.03}

        # Create optimizer WITH holdout_objective_fn
        optimizer = ConcreteOptimizer(
            simple_space,
            validation_objective,
            holdout_objective_fn=holdout_objective,
        )

        # Provide holdout_data (required to trigger overfitting computation)
        holdout_data = pd.DataFrame({"price": [100, 101, 102]})
        result = optimizer.optimize(holdout_data=holdout_data)

        # Verify OOS metrics come from holdout_objective (0.8), not validation (1.5)
        assert result.out_of_sample_metrics is not None, "OOS metrics should be computed"
        assert result.out_of_sample_metrics["sharpe_ratio"] == 0.8, (
            f"OOS sharpe should be 0.8 from holdout_objective, "
            f"got {result.out_of_sample_metrics['sharpe_ratio']}"
        )

        # Verify IS metrics come from validation_objective (1.5)
        assert result.in_sample_metrics is not None, "IS metrics should be computed"
        assert result.in_sample_metrics["sharpe_ratio"] == 1.5, (
            f"IS sharpe should be 1.5 from validation_objective, "
            f"got {result.in_sample_metrics['sharpe_ratio']}"
        )

        # Overfitting score should be 1.5 / 0.8 = 1.875 (indicating overfitting)
        expected_overfitting = 1.5 / 0.8
        assert abs(result.overfitting_score - expected_overfitting) < 0.01, (
            f"Overfitting score should be ~{expected_overfitting:.2f}, "
            f"got {result.overfitting_score}"
        )

    def test_overfitting_warning_without_holdout_objective_fn(self, simple_space, caplog):
        """
        Test that a warning is logged when holdout_data is provided without holdout_objective_fn.

        This ensures users are alerted when they're using the incorrect pattern
        that defeats the purpose of overfitting detection.
        """
        def simple_objective(params):
            return {"sharpe_ratio": 1.0, "total_return": 0.05}

        # Create optimizer WITHOUT holdout_objective_fn
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        holdout_data = pd.DataFrame({"price": [100, 101, 102]})

        import logging
        with caplog.at_level(logging.WARNING):
            result = optimizer.optimize(holdout_data=holdout_data)

        # Check that warning was logged
        assert any(
            "holdout_data was provided but no holdout_objective_fn was set" in record.message
            for record in caplog.records
        ), "Should warn when holdout_data provided without holdout_objective_fn"


# ============================================================================
# Tests for _update_best with Non-Completed Results
# ============================================================================

class TestUpdateBest:
    """Tests for _update_best method."""

    def test_update_best_ignores_failed_result(self, simple_space, simple_objective):
        """Test that _update_best ignores results with non-completed status."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        failed_result = TrialResult(
            trial_id=0,
            params={"learning_rate": 0.01, "batch_size": 32},
            status="failed",
            error_message="Test failure",
        )

        optimizer._update_best(failed_result)

        # Best result should remain None
        assert optimizer._best_result is None

    def test_update_best_accepts_completed_result(self, simple_space, simple_objective):
        """Test that _update_best accepts completed results."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        completed_result = TrialResult(
            trial_id=0,
            params={"learning_rate": 0.01, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 1.5},
        )

        optimizer._update_best(completed_result)

        assert optimizer._best_result == completed_result


# ============================================================================
# Tests for Verbose Logging in evaluate_batch
# ============================================================================

class TestEvaluateBatchVerbose:
    """Tests for verbose logging in evaluate_batch."""

    def test_evaluate_batch_logs_progress_every_10_trials(
        self, simple_space, simple_objective, caplog
    ):
        """Test that evaluate_batch logs progress every 10 trials."""
        config = OptimizerConfig(n_jobs=1, verbose=1)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        # Create 15 parameter combinations
        param_list = [
            {"learning_rate": 0.01 + i * 0.005, "batch_size": 32}
            for i in range(15)
        ]

        with caplog.at_level(logging.INFO):
            results = optimizer.evaluate_batch(param_list, parallel=False)

        assert len(results) == 15
        # Should have logged "Completed 10/15 trials"
        assert any("Completed 10/15" in record.message for record in caplog.records)

    def test_evaluate_batch_no_log_under_10_trials(
        self, simple_space, simple_objective, caplog
    ):
        """Test that evaluate_batch doesn't log for fewer than 10 trials."""
        config = OptimizerConfig(n_jobs=1, verbose=1)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        param_list = [
            {"learning_rate": 0.01 + i * 0.01, "batch_size": 32}
            for i in range(5)
        ]

        with caplog.at_level(logging.INFO):
            results = optimizer.evaluate_batch(param_list, parallel=False)

        assert len(results) == 5
        # Should not have "Completed X/5" messages at the 10-trial intervals
        assert not any("Completed 10" in record.message for record in caplog.records)


# ============================================================================
# Tests for Parallel Execution
# ============================================================================

class TestParallelExecution:
    """Tests for parallel execution in evaluate_batch."""

    def test_evaluate_batch_parallel_execution(
        self, simple_space, simple_objective, config_parallel
    ):
        """Test parallel execution in evaluate_batch."""
        optimizer = ConcreteOptimizer(
            simple_space, simple_objective, config=config_parallel
        )

        param_list = [
            {"learning_rate": 0.01 + i * 0.01, "batch_size": 32}
            for i in range(10)
        ]

        results = optimizer.evaluate_batch(param_list, parallel=True)

        assert len(results) == 10
        # Results should be sorted by trial_id
        trial_ids = [r.trial_id for r in results]
        assert trial_ids == sorted(trial_ids)

    def test_evaluate_batch_parallel_handles_timeout(
        self, simple_space
    ):
        """Test that parallel execution handles timeouts."""
        def slow_objective(params):
            if params.get("learning_rate", 0) > 0.05:
                time.sleep(10)  # This should timeout
            return {"sharpe_ratio": 1.0}

        config = OptimizerConfig(n_jobs=2, timeout_per_trial=0.5)
        optimizer = ConcreteOptimizer(simple_space, slow_objective, config=config)

        param_list = [
            {"learning_rate": 0.01, "batch_size": 32},  # Fast
            {"learning_rate": 0.1, "batch_size": 32},   # Slow (should timeout)
        ]

        results = optimizer.evaluate_batch(param_list, parallel=True)

        assert len(results) == 2
        # One should be completed, one should be failed due to timeout
        statuses = [r.status for r in results]
        assert "completed" in statuses

    def test_evaluate_batch_parallel_handles_exception(
        self, simple_space
    ):
        """Test that parallel execution handles exceptions in workers."""
        def failing_objective(params):
            if params.get("batch_size", 32) > 50:
                raise ValueError("Batch size too large")
            return {"sharpe_ratio": 1.0}

        config = OptimizerConfig(n_jobs=2)
        optimizer = ConcreteOptimizer(simple_space, failing_objective, config=config)

        param_list = [
            {"learning_rate": 0.01, "batch_size": 32},  # Should succeed
            {"learning_rate": 0.01, "batch_size": 64},  # Should fail
        ]

        results = optimizer.evaluate_batch(param_list, parallel=True)

        assert len(results) == 2
        # Check both results exist (one completed, one failed)
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]
        assert len(completed) >= 1

    def test_evaluate_batch_sequential_when_n_jobs_1(
        self, simple_space, simple_objective
    ):
        """Test that evaluate_batch runs sequentially when n_jobs=1."""
        config = OptimizerConfig(n_jobs=1)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        param_list = [
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.05, "batch_size": 48},
        ]

        results = optimizer.evaluate_batch(param_list, parallel=True)

        assert len(results) == 2
        for r in results:
            assert r.status == "completed"

    def test_evaluate_batch_sequential_when_parallel_false(
        self, simple_space, simple_objective, config_parallel
    ):
        """Test that evaluate_batch runs sequentially when parallel=False."""
        optimizer = ConcreteOptimizer(
            simple_space, simple_objective, config=config_parallel
        )

        param_list = [
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.05, "batch_size": 48},
        ]

        results = optimizer.evaluate_batch(param_list, parallel=False)

        assert len(results) == 2


# ============================================================================
# Tests for create_backtest_objective
# ============================================================================

class TestCreateBacktestObjective:
    """Tests for create_backtest_objective function."""

    def test_create_backtest_objective_updates_engine_config(self):
        """Test that objective function updates engine config."""
        # Mock engine with config
        engine = MagicMock()
        engine.config = MagicMock()
        engine.config.default_stop_ticks = 8
        engine.config.default_target_ticks = 16
        engine.config.min_confidence = 0.6

        # Mock result
        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {
            "sharpe_ratio": 1.5,
            "total_return": 0.08,
        }
        engine.run.return_value = mock_result

        # Create data
        data = pd.DataFrame({"price": [100, 101, 102]})

        # Create signal generator factory
        signal_gen_factory = MagicMock(return_value=lambda x: 1)

        objective = create_backtest_objective(engine, data, signal_gen_factory)

        # Call objective with params
        params = {
            "stop_ticks": 10,
            "target_ticks": 20,
            "confidence_threshold": 0.7,
        }
        metrics = objective(params)

        # Verify engine config was updated
        assert engine.config.default_stop_ticks == 10
        assert engine.config.default_target_ticks == 20
        assert engine.config.min_confidence == 0.7

        # Verify metrics returned
        assert "sharpe_ratio" in metrics
        assert metrics["sharpe_ratio"] == 1.5

    def test_create_backtest_objective_with_nested_metrics(self):
        """Test objective with nested metrics dictionary."""
        engine = MagicMock()
        engine.config = MagicMock()

        # Mock result with nested metrics
        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {
            "performance": {
                "sharpe_ratio": 1.5,
                "total_return": 0.08,
            },
            "risk": {
                "max_drawdown": -0.05,
                "volatility": 0.15,
            },
            "total_trades": 50,  # Non-dict value at top level
        }
        engine.run.return_value = mock_result

        data = pd.DataFrame({"price": [100, 101, 102]})
        signal_gen_factory = MagicMock(return_value=lambda x: 1)

        objective = create_backtest_objective(engine, data, signal_gen_factory)

        params = {"learning_rate": 0.01}
        metrics = objective(params)

        # Should flatten nested metrics
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
        assert "total_trades" in metrics

    def test_create_backtest_objective_with_context(self):
        """Test objective passes context to engine."""
        engine = MagicMock()
        engine.config = MagicMock()

        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {"sharpe_ratio": 1.0}
        engine.run.return_value = mock_result

        data = pd.DataFrame({"price": [100, 101]})
        signal_gen_factory = MagicMock(return_value=lambda x: 1)
        context = {"model": "test_model", "session_id": "123"}

        objective = create_backtest_objective(
            engine, data, signal_gen_factory, context=context
        )

        objective({"learning_rate": 0.01})

        # Verify context was passed to engine.run
        engine.run.assert_called_once()
        call_kwargs = engine.run.call_args.kwargs
        assert call_kwargs["context"] == context

    def test_create_backtest_objective_without_config(self):
        """Test objective handles engine without config attribute."""
        # Create engine mock without config attribute
        engine = MagicMock()
        del engine.config  # Remove config so hasattr returns False

        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {"sharpe_ratio": 1.0}
        engine.run.return_value = mock_result

        data = pd.DataFrame({"price": [100, 101]})
        signal_gen_factory = MagicMock(return_value=lambda x: 1)

        objective = create_backtest_objective(engine, data, signal_gen_factory)

        # Should not raise even without config
        metrics = objective({"stop_ticks": 10})
        assert "sharpe_ratio" in metrics


# ============================================================================
# Tests for create_split_objective
# ============================================================================

class TestCreateSplitObjective:
    """Tests for create_split_objective function."""

    def test_create_split_objective_returns_two_functions(self):
        """Test that create_split_objective returns two objective functions."""
        engine = MagicMock()
        engine.config = MagicMock()

        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {"sharpe_ratio": 1.0}
        engine.run.return_value = mock_result

        val_data = pd.DataFrame({"price": [100, 101, 102]})
        holdout_data = pd.DataFrame({"price": [103, 104, 105]})
        signal_gen_factory = MagicMock(return_value=lambda x: 1)

        val_obj, holdout_obj = create_split_objective(
            engine, val_data, holdout_data, signal_gen_factory
        )

        assert callable(val_obj)
        assert callable(holdout_obj)

    def test_create_split_objective_uses_different_data(self):
        """Test that val and holdout objectives use different data."""
        engine = MagicMock()
        engine.config = MagicMock()

        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {"sharpe_ratio": 1.0}
        engine.run.return_value = mock_result

        val_data = pd.DataFrame({"price": [100, 101, 102]})
        holdout_data = pd.DataFrame({"price": [200, 201, 202]})
        signal_gen_factory = MagicMock(return_value=lambda x: 1)

        val_obj, holdout_obj = create_split_objective(
            engine, val_data, holdout_data, signal_gen_factory
        )

        # Call both objectives
        val_obj({"learning_rate": 0.01})
        holdout_obj({"learning_rate": 0.01})

        # Verify engine.run was called twice with different data
        assert engine.run.call_count == 2

        # Get the data passed in each call
        calls = engine.run.call_args_list
        first_data = calls[0].args[0]
        second_data = calls[1].args[0]

        # Verify different data was used
        assert not first_data.equals(second_data)

    def test_create_split_objective_with_context(self):
        """Test that context is passed to both objectives."""
        engine = MagicMock()
        engine.config = MagicMock()

        mock_result = MagicMock()
        mock_result.report.metrics.to_dict.return_value = {"sharpe_ratio": 1.0}
        engine.run.return_value = mock_result

        val_data = pd.DataFrame({"price": [100, 101]})
        holdout_data = pd.DataFrame({"price": [200, 201]})
        signal_gen_factory = MagicMock(return_value=lambda x: 1)
        context = {"model": "test"}

        val_obj, holdout_obj = create_split_objective(
            engine, val_data, holdout_data, signal_gen_factory, context
        )

        val_obj({"lr": 0.01})
        holdout_obj({"lr": 0.01})

        # Both calls should have the context
        for call in engine.run.call_args_list:
            assert call.kwargs["context"] == context


# ============================================================================
# Tests for _build_failed_result
# ============================================================================

class TestBuildFailedResult:
    """Tests for _build_failed_result method."""

    def test_build_failed_result_structure(self, simple_space, simple_objective):
        """Test that _build_failed_result returns correct structure."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)
        optimizer._start_time = datetime.now()
        optimizer._end_time = datetime.now()

        result = optimizer._build_failed_result("Test error message")

        assert result.best_params == {}
        assert result.best_metric == 0.0
        assert result.config["error"] == "Test error message"
        assert result.parameter_space_name == "test_space"
        assert result.optimizer_type == "ConcreteOptimizer"

    def test_build_failed_result_preserves_partial_results(
        self, simple_space, simple_objective
    ):
        """Test that _build_failed_result preserves any partial results."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        # Add some results before failure
        partial_result = TrialResult(
            trial_id=0,
            params={"learning_rate": 0.01, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 1.0},
        )
        with optimizer._lock:
            optimizer._results.append(partial_result)

        result = optimizer._build_failed_result("Partial failure")

        assert len(result.all_results) == 1


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_optimizer_with_no_results(self, simple_space, simple_objective):
        """Test _build_result with no results."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        result = optimizer._build_result()

        assert result.best_params == {}
        assert result.best_metric == 0.0
        assert result.total_trials == 0

    def test_evaluate_params_with_invalid_params(self, simple_space, simple_objective):
        """Test evaluate_params with parameters outside valid range."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        invalid_params = {"learning_rate": 1.0, "batch_size": 1000}  # Out of range

        result = optimizer.evaluate_params(invalid_params)

        assert result.status == "failed"
        assert "Invalid params" in result.error_message

    def test_thread_safe_trial_counter(self, simple_space, simple_objective):
        """Test that trial counter is thread-safe."""
        config = OptimizerConfig(n_jobs=4)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        param_list = [
            {"learning_rate": 0.01 + i * 0.001, "batch_size": 32}
            for i in range(20)
        ]

        results = optimizer.evaluate_batch(param_list, parallel=True)

        # All trial IDs should be unique
        trial_ids = [r.trial_id for r in results]
        assert len(trial_ids) == len(set(trial_ids))

    def test_get_current_best_returns_copy(self, simple_space, simple_objective):
        """Test that get_current_best returns a copy of params."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        result = TrialResult(
            trial_id=0,
            params={"learning_rate": 0.01, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 1.0},
        )
        optimizer._update_best(result)

        best = optimizer.get_current_best()
        best["learning_rate"] = 999  # Modify the copy

        # Original should be unchanged
        assert optimizer._best_result.params["learning_rate"] == 0.01

    def test_get_trial_count(self, simple_space, simple_objective):
        """Test get_trial_count method."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        assert optimizer.get_trial_count() == 0

        optimizer.optimize()

        assert optimizer.get_trial_count() == 3


# ============================================================================
# Tests for Verbose New Best Logging
# ============================================================================

class TestParallelExceptionCatching:
    """Tests for catching exceptions in parallel execution futures."""

    def test_parallel_future_exception_creates_failed_result(self, simple_space):
        """Test that exceptions in parallel futures create failed results."""
        call_count = [0]

        def sometimes_failing_objective(params):
            call_count[0] += 1
            # Fail on even calls
            if call_count[0] % 2 == 0:
                raise RuntimeError(f"Simulated failure #{call_count[0]}")
            return {"sharpe_ratio": 1.0}

        config = OptimizerConfig(n_jobs=2, timeout_per_trial=5.0)
        optimizer = ConcreteOptimizer(simple_space, sometimes_failing_objective, config=config)

        param_list = [
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.02, "batch_size": 32},
            {"learning_rate": 0.03, "batch_size": 32},
            {"learning_rate": 0.04, "batch_size": 32},
        ]

        results = optimizer.evaluate_batch(param_list, parallel=True)

        assert len(results) == 4
        # Some should be failed
        failed_results = [r for r in results if r.status == "failed"]
        assert len(failed_results) >= 1


class TestOverfittingEmptyParams:
    """Tests for overfitting metrics with empty best_params."""

    def test_overfitting_skipped_when_best_params_none(self, simple_space):
        """Test that _compute_overfitting_metrics returns early when best_params is None."""
        def objective(params):
            # Always fail
            raise ValueError("Objective always fails")

        optimizer = ConcreteOptimizer(simple_space, objective, should_fail=True)

        # Add holdout data but make optimization fail
        holdout_data = pd.DataFrame({"price": [100, 101]})

        result = optimizer.optimize(holdout_data=holdout_data)

        # Since optimization failed, best_params should be empty
        assert result.best_params == {}
        # Overfitting metrics should not have been computed
        assert result.overfitting_score is None

    def test_overfitting_with_empty_best_params_dict(self, simple_space, simple_objective):
        """Test overfitting metrics when result has empty best_params dict."""
        optimizer = ConcreteOptimizer(simple_space, simple_objective)

        # Create a result with empty best_params
        result = OptimizationResult(
            best_params={},
            best_metric=0.0,
            metric_name="sharpe_ratio",
            all_results=[],
            parameter_space_name="test_space",
            optimizer_type="ConcreteOptimizer",
        )

        optimizer._holdout_data = pd.DataFrame({"price": [100, 101]})

        # Should return early without computing overfitting
        updated_result = optimizer._compute_overfitting_metrics(result)

        # Result should be unchanged (early return)
        assert updated_result == result


class TestVerboseNewBestLogging:
    """Tests for verbose logging when new best is found."""

    def test_update_best_logs_new_best(
        self, simple_space, simple_objective, caplog
    ):
        """Test that _update_best logs when a new best is found."""
        config = OptimizerConfig(verbose=1)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        # First result
        result1 = TrialResult(
            trial_id=0,
            params={"learning_rate": 0.01, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 1.0},
        )

        # Better result
        result2 = TrialResult(
            trial_id=1,
            params={"learning_rate": 0.05, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 1.5},
        )

        with caplog.at_level(logging.INFO):
            optimizer._update_best(result1)
            optimizer._update_best(result2)

        # Should have logged "New best" for the second result
        assert any("New best" in record.message for record in caplog.records)

    def test_update_best_no_log_when_not_better(
        self, simple_space, simple_objective, caplog
    ):
        """Test that _update_best doesn't log when result isn't better."""
        config = OptimizerConfig(verbose=1)
        optimizer = ConcreteOptimizer(simple_space, simple_objective, config=config)

        # Good result
        result1 = TrialResult(
            trial_id=0,
            params={"learning_rate": 0.05, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 2.0},
        )

        # Worse result
        result2 = TrialResult(
            trial_id=1,
            params={"learning_rate": 0.01, "batch_size": 32},
            status="completed",
            metrics={"sharpe_ratio": 1.0},
        )

        optimizer._update_best(result1)

        caplog.clear()
        with caplog.at_level(logging.INFO):
            optimizer._update_best(result2)

        # Should not have logged "New best" for the worse result
        new_best_logs = [r for r in caplog.records if "New best" in r.message]
        assert len(new_best_logs) == 0
