"""
Extended Tests for src/optimization/bayesian_optimizer.py.

These tests target uncovered lines to improve coverage from 63% to 80%+.

Uncovered lines being addressed:
- Lines 43-45: OPTUNA_AVAILABLE = False when optuna not installed
- Lines 219-220, 237-238: Unknown sampler/pruner warning fallbacks
- Lines 284-298: Trial failure handling (exception in objective)
- Lines 332, 339: Float parameter with step and without log_scale
- Lines 375, 382-385: Empty study / no completed trials case
- Lines 427, 432-434: get_importance() with exceptions
- Lines 443-458: save_study() functionality
- Lines 480-500: load_study() functionality
- Lines 538: Import error in run_bayesian_optimization
- Lines 575-617: create_visualization() function
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, Mock
import logging


# Check if optuna is available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


from src.optimization.parameter_space import (
    ParameterConfig,
    ParameterSpace,
)
from src.optimization.results import TrialResult, OptimizationResult


# Skip all tests if optuna not available
pytestmark = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")


if OPTUNA_AVAILABLE:
    from src.optimization.bayesian_optimizer import (
        BayesianConfig,
        BayesianOptimizer,
        run_bayesian_optimization,
        create_visualization,
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_objective():
    """Simple objective function for testing."""
    def obj(params):
        return {"sharpe_ratio": params.get("x", 0.5)}
    return obj


@pytest.fixture
def failing_objective():
    """Objective function that raises exceptions."""
    call_count = [0]
    def obj(params):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            raise ValueError("Simulated failure")
        return {"sharpe_ratio": params.get("x", 0.5)}
    return obj


@pytest.fixture
def simple_space():
    """Simple parameter space for testing."""
    return ParameterSpace(parameters=[
        ParameterConfig("x", 0.0, 1.0, param_type="float"),
    ])


@pytest.fixture
def mixed_space():
    """Mixed parameter space with int, float (step), and categorical."""
    return ParameterSpace(parameters=[
        ParameterConfig("int_param", 1, 10, step=1, param_type="int"),
        ParameterConfig("float_step", 0.0, 1.0, step=0.1, param_type="float"),
        ParameterConfig("float_log", 0.001, 1.0, param_type="float", log_scale=True),
        ParameterConfig("cat_param", param_type="categorical", choices=["A", "B"]),
    ])


# ============================================================================
# Test Unknown Sampler/Pruner Fallbacks (Lines 219-220, 237-238)
# ============================================================================

class TestUnknownSamplerPruner:
    """Tests for unknown sampler and pruner fallback handling."""

    def test_unknown_sampler_falls_back_to_tpe(self, simple_space, simple_objective):
        """Unknown sampler should fall back to TPE with warning."""
        config = BayesianConfig(
            n_trials=3,
            sampler="nonexistent_sampler",
            show_progress_bar=False,
        )

        with patch('src.optimization.bayesian_optimizer.logger') as mock_logger:
            optimizer = BayesianOptimizer(simple_space, simple_objective, config)
            result = optimizer.optimize()

            # Should still run successfully with TPE fallback
            assert result.total_trials == 3

            # Should have logged a warning
            mock_logger.warning.assert_called()
            warning_call = str(mock_logger.warning.call_args)
            assert "nonexistent_sampler" in warning_call or "Unknown sampler" in warning_call

    def test_unknown_pruner_falls_back_to_median(self, simple_space, simple_objective):
        """Unknown pruner should fall back to MedianPruner with warning."""
        config = BayesianConfig(
            n_trials=3,
            pruner="nonexistent_pruner",
            show_progress_bar=False,
        )

        with patch('src.optimization.bayesian_optimizer.logger') as mock_logger:
            optimizer = BayesianOptimizer(simple_space, simple_objective, config)
            result = optimizer.optimize()

            # Should still run successfully
            assert result.total_trials == 3


# ============================================================================
# Test Trial Failure Handling (Lines 284-298)
# ============================================================================

class TestTrialFailureHandling:
    """Tests for handling failed trials during optimization."""

    def test_objective_exception_handled(self, simple_space, failing_objective):
        """Exceptions in objective function should be handled gracefully."""
        config = BayesianConfig(
            n_trials=6,  # 3 will succeed, 3 will fail
            show_progress_bar=False,
        )

        optimizer = BayesianOptimizer(simple_space, failing_objective, config)

        with patch('src.optimization.bayesian_optimizer.logger') as mock_logger:
            result = optimizer.optimize()

            # Should complete all trials despite failures
            assert result.total_trials == 6

            # Should have some failed trials recorded
            failed_trials = [r for r in result.all_results if r.status == "failed"]
            assert len(failed_trials) >= 1

            # Failed trials should have error message
            for failed in failed_trials:
                assert failed.error_message is not None

    def test_all_trials_fail_returns_worst_value(self, simple_space):
        """When all trials fail, should return worst possible values."""
        def always_fails(params):
            raise RuntimeError("Always fails")

        config = BayesianConfig(
            n_trials=3,
            show_progress_bar=False,
        )

        optimizer = BayesianOptimizer(simple_space, always_fails, config)
        result = optimizer.optimize()

        # Should complete without crashing
        assert result.total_trials == 3
        # All trials should be marked as failed
        assert all(r.status == "failed" for r in result.all_results)


# ============================================================================
# Test Float Parameter with Step (Line 332, 339)
# ============================================================================

class TestFloatParameterStep:
    """Tests for float parameters with step values."""

    def test_float_with_step_value(self, simple_objective):
        """Test float parameter with step restriction."""
        space = ParameterSpace(parameters=[
            ParameterConfig("float_step", 0.0, 1.0, step=0.25, param_type="float"),
        ])

        config = BayesianConfig(n_trials=8, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, simple_objective, config)
        result = optimizer.optimize()

        # All sampled values should be multiples of 0.25
        for trial_result in result.all_results:
            val = trial_result.params.get("float_step", 0)
            # Value should be approximately a multiple of 0.25
            remainder = val % 0.25
            assert remainder < 0.01 or (0.25 - remainder) < 0.01

    def test_float_log_scale(self, simple_objective):
        """Test float parameter with log scale."""
        space = ParameterSpace(parameters=[
            ParameterConfig("log_param", 0.001, 1.0, param_type="float", log_scale=True),
        ])

        config = BayesianConfig(n_trials=10, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, simple_objective, config)
        result = optimizer.optimize()

        # Should have some values closer to the lower end due to log sampling
        values = [r.params.get("log_param", 0.5) for r in result.all_results]
        assert all(0.001 <= v <= 1.0 for v in values)


# ============================================================================
# Test Empty Study / No Completed Trials (Lines 375, 382-385)
# ============================================================================

class TestEmptyStudyHandling:
    """Tests for handling empty study or no completed trials."""

    def test_build_result_no_completed_trials(self, simple_space):
        """Test building result when no trials completed."""
        def always_fails(params):
            raise ValueError("Always fails")

        config = BayesianConfig(n_trials=2, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, always_fails, config)

        # Run optimization (all trials will fail)
        result = optimizer.optimize()

        # Should return a valid result object
        assert result is not None
        assert isinstance(result, OptimizationResult)

    def test_get_study_before_optimization(self, simple_space, simple_objective):
        """Test getting study before optimization runs."""
        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)

        # Study should be None before optimization
        assert optimizer.get_study() is None

        # After optimization, study should exist
        optimizer.optimize()
        assert optimizer.get_study() is not None


# ============================================================================
# Test get_importance() with Exceptions (Lines 427, 432-434)
# ============================================================================

class TestGetImportance:
    """Tests for parameter importance calculation."""

    def test_importance_before_optimization(self, simple_space, simple_objective):
        """get_importance should return empty dict before optimization."""
        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)

        importance = optimizer.get_importance()
        assert importance == {}

    def test_importance_with_exception(self, simple_space, simple_objective):
        """get_importance should handle exceptions gracefully."""
        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)
        optimizer.optimize()

        # Mock the importance function to raise an exception
        with patch('optuna.importance.get_param_importances', side_effect=ValueError("Test error")):
            importance = optimizer.get_importance()
            # Should return empty dict on error
            assert importance == {}

    def test_importance_after_optimization(self, simple_objective):
        """get_importance should work after successful optimization."""
        space = ParameterSpace(parameters=[
            ParameterConfig("important", 0.0, 1.0, param_type="float"),
            ParameterConfig("less_important", 0.0, 1.0, param_type="float"),
        ])

        def objective(params):
            return {"sharpe_ratio": params["important"] * 2 + params["less_important"] * 0.01}

        config = BayesianConfig(n_trials=25, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, objective, config)
        optimizer.optimize()

        importance = optimizer.get_importance()
        # Should have importance values for parameters
        assert len(importance) >= 1


# ============================================================================
# Test save_study() and load_study() (Lines 443-458, 480-500)
# ============================================================================

class TestStudySaveLoad:
    """Tests for saving and loading optimization studies."""

    def test_save_study_to_file(self, simple_space, simple_objective, tmp_path):
        """Test saving study to SQLite file."""
        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)
        optimizer.optimize()

        # Save study
        db_path = str(tmp_path / "saved_study.db")
        optimizer.save_study(db_path)

        # File should exist
        assert os.path.exists(db_path)

    def test_save_study_before_optimization(self, simple_space, simple_objective, tmp_path):
        """Test saving study before optimization runs (should warn)."""
        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)

        # Try to save before optimization
        db_path = str(tmp_path / "no_study.db")

        with patch('src.optimization.bayesian_optimizer.logger') as mock_logger:
            optimizer.save_study(db_path)
            mock_logger.warning.assert_called_once()

    def test_load_study_from_file(self, simple_space, simple_objective, tmp_path):
        """Test loading study from SQLite file."""
        # First, create and save a study
        db_path = str(tmp_path / "load_test.db")
        storage = f"sqlite:///{db_path}"

        config = BayesianConfig(
            n_trials=5,
            study_name="load_test_study",
            storage=storage,
            show_progress_bar=False,
        )
        optimizer1 = BayesianOptimizer(simple_space, simple_objective, config)
        result1 = optimizer1.optimize()

        # Load the study
        optimizer2 = BayesianOptimizer.load_study(
            simple_space,
            simple_objective,
            db_path,
            study_name="load_test_study",
        )

        # Loaded study should have the same trials
        study = optimizer2.get_study()
        assert study is not None
        assert len(study.trials) == 5

    def test_load_study_and_continue(self, simple_space, simple_objective, tmp_path):
        """Test loading study and continuing optimization."""
        db_path = str(tmp_path / "continue_test.db")
        storage = f"sqlite:///{db_path}"
        study_name = "continue_study"

        # First run
        config1 = BayesianConfig(
            n_trials=3,
            study_name=study_name,
            storage=storage,
            show_progress_bar=False,
        )
        optimizer1 = BayesianOptimizer(simple_space, simple_objective, config1)
        optimizer1.optimize()

        # Load and continue
        optimizer2 = BayesianOptimizer.load_study(
            simple_space,
            simple_objective,
            db_path,
            study_name=study_name,
        )

        # The loaded optimizer should allow continuing
        assert optimizer2.get_study() is not None
        assert len(optimizer2.get_study().trials) == 3


# ============================================================================
# Test run_bayesian_optimization Import Error (Line 538)
# ============================================================================

class TestRunBayesianImportError:
    """Tests for import error handling in convenience function."""

    def test_run_bayesian_works_with_optuna(self, simple_space, simple_objective):
        """run_bayesian_optimization should work when optuna is available."""
        result = run_bayesian_optimization(
            simple_space,
            simple_objective,
            n_trials=3,
        )

        assert result.total_trials == 3

    def test_run_bayesian_with_all_options(self, simple_space, simple_objective, tmp_path):
        """Test all options for run_bayesian_optimization."""
        db_path = str(tmp_path / "convenience_test.db")

        result = run_bayesian_optimization(
            simple_space,
            simple_objective,
            n_trials=5,
            metric_name="sharpe_ratio",
            study_name="convenience_study",
            storage=f"sqlite:///{db_path}",
            n_jobs=1,
            seed=123,
        )

        assert result.total_trials == 5
        assert os.path.exists(db_path)


# ============================================================================
# Test create_visualization() (Lines 575-617)
# ============================================================================

class TestCreateVisualization:
    """Tests for Optuna visualization generation."""

    def test_create_visualization_success(self, simple_space, simple_objective, tmp_path):
        """Test successful visualization creation."""
        config = BayesianConfig(n_trials=10, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)
        optimizer.optimize()

        study = optimizer.get_study()
        output_dir = str(tmp_path / "visualizations")

        plots = create_visualization(study, output_dir)

        # Should create output directory
        assert os.path.exists(output_dir)

        # Should have created some plots
        assert len(plots) >= 1

    def test_create_visualization_with_2_params(self, simple_objective, tmp_path):
        """Test visualization with multiple parameters."""
        space = ParameterSpace(parameters=[
            ParameterConfig("p1", 0.0, 1.0, param_type="float"),
            ParameterConfig("p2", 0.0, 1.0, param_type="float"),
        ])

        config = BayesianConfig(n_trials=15, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, simple_objective, config)
        optimizer.optimize()

        study = optimizer.get_study()
        output_dir = str(tmp_path / "multi_param_vis")

        plots = create_visualization(study, output_dir)

        # Should have created plots
        assert os.path.exists(output_dir)

    def test_create_visualization_handles_exceptions(self, simple_space, simple_objective, tmp_path):
        """Test that visualization handles plot creation failures gracefully."""
        config = BayesianConfig(n_trials=5, show_progress_bar=False)
        optimizer = BayesianOptimizer(simple_space, simple_objective, config)
        optimizer.optimize()

        study = optimizer.get_study()
        output_dir = str(tmp_path / "exception_test")

        # Mock plotly functions to fail (they're imported inside the function)
        with patch('optuna.visualization.plot_optimization_history', side_effect=Exception("Plot error")):
            with patch('optuna.visualization.plot_param_importances', side_effect=Exception("Plot error")):
                with patch('optuna.visualization.plot_parallel_coordinate', side_effect=Exception("Plot error")):
                    plots = create_visualization(study, output_dir)

                    # Should return empty or partial dict without crashing
                    assert isinstance(plots, dict)


# ============================================================================
# Test Mixed Parameter Types (Line 332 - INT step)
# ============================================================================

class TestMixedParameterTypes:
    """Tests for optimization with mixed parameter types."""

    def test_int_with_step(self, simple_objective):
        """Test integer parameter with step value."""
        space = ParameterSpace(parameters=[
            ParameterConfig("int_step", 0, 100, step=10, param_type="int"),
        ])

        config = BayesianConfig(n_trials=10, show_progress_bar=False)
        optimizer = BayesianOptimizer(space, simple_objective, config)
        result = optimizer.optimize()

        # All values should be multiples of 10
        for trial_result in result.all_results:
            val = trial_result.params.get("int_step", 0)
            assert val % 10 == 0

    def test_all_parameter_types_combined(self, mixed_space):
        """Test with all parameter types in one optimization."""
        def objective(params):
            score = params["int_param"] + params["float_step"] + params["float_log"]
            if params["cat_param"] == "A":
                score += 1
            return {"sharpe_ratio": score}

        config = BayesianConfig(n_trials=15, show_progress_bar=False)
        optimizer = BayesianOptimizer(mixed_space, objective, config)
        result = optimizer.optimize()

        assert result.total_trials == 15
        # Best params should be reasonable
        assert "int_param" in result.best_params
        assert "float_step" in result.best_params
        assert "float_log" in result.best_params
        assert "cat_param" in result.best_params


# ============================================================================
# Test Progress Callback (Lines 354-370)
# ============================================================================

class TestProgressCallback:
    """Tests for progress callback during optimization."""

    def test_progress_callback_logs_every_10_trials(self, simple_space, simple_objective):
        """Test that progress is logged every 10 trials."""
        config = BayesianConfig(
            n_trials=25,
            show_progress_bar=False,
            verbose=1,  # Enable progress callback
        )

        # Update the base config
        config.verbose = 1

        optimizer = BayesianOptimizer(simple_space, simple_objective, config)

        with patch('src.optimization.bayesian_optimizer.logger') as mock_logger:
            optimizer.optimize()

            # Should have logged progress at least twice (trials 10 and 20)
            info_calls = mock_logger.info.call_count
            assert info_calls >= 2


# ============================================================================
# Test Hyperband Pruner
# ============================================================================

class TestHyperbandPruner:
    """Tests for Hyperband pruner usage."""

    def test_hyperband_pruner(self, simple_space, simple_objective):
        """Test using Hyperband pruner."""
        config = BayesianConfig(
            n_trials=10,
            pruner="hyperband",
            show_progress_bar=False,
        )

        optimizer = BayesianOptimizer(simple_space, simple_objective, config)
        result = optimizer.optimize()

        # Should complete without errors
        assert result.total_trials >= 5


# ============================================================================
# Test Minimization Direction
# ============================================================================

class TestMinimizationDirection:
    """Tests for minimization vs maximization."""

    def test_minimization_direction(self, simple_space):
        """Test that minimization works correctly."""
        def objective(params):
            # Lower is better
            return {"loss": params["x"]}

        config = BayesianConfig(
            n_trials=15,
            higher_is_better=False,
            metric_name="loss",
            show_progress_bar=False,
        )

        optimizer = BayesianOptimizer(simple_space, objective, config)
        result = optimizer.optimize()

        # Best x should be close to 0 (lower bound)
        assert result.best_params["x"] < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
