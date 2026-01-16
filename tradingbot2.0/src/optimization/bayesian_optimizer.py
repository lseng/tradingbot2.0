"""
Bayesian Optimizer using Optuna for Parameter Optimization.

This module implements Bayesian optimization using the Optuna library.
Bayesian optimization is more sample-efficient than grid or random search,
making it ideal for expensive objective functions like backtesting.

Key Features:
- TPE (Tree-structured Parzen Estimator) sampler for smart exploration
- Pruning of poor-performing trials for efficiency
- Study persistence for save/resume capability
- Integration with Optuna's visualization tools

When to use:
- Expensive objective functions (>1 second per evaluation)
- Medium-sized parameter spaces
- When sample efficiency matters

Usage:
    from src.optimization.bayesian_optimizer import BayesianOptimizer

    optimizer = BayesianOptimizer(
        parameter_space=space,
        objective_fn=my_objective,
        n_trials=100,
    )
    result = optimizer.optimize()
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from src.optimization.optimizer_base import BaseOptimizer, OptimizerConfig
from src.optimization.parameter_space import ParameterSpace, ParameterType
from src.optimization.results import OptimizationResult, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class BayesianConfig(OptimizerConfig):
    """
    Configuration specific to Bayesian optimization.

    Attributes:
        n_trials: Number of trials to run
        sampler: Sampler type ("tpe", "random", "cmaes")
        pruner: Pruner type ("median", "hyperband", None)
        study_name: Name for the Optuna study
        storage: Database URL for study persistence (None = in-memory)
        load_if_exists: Whether to load existing study with same name
        n_startup_trials: Random trials before using TPE
        multivariate: Whether to use multivariate TPE
        show_progress_bar: Whether to show Optuna progress bar
    """
    n_trials: int = 100
    sampler: str = "tpe"
    pruner: Optional[str] = "median"
    study_name: str = "trading_optimization"
    storage: Optional[str] = None
    load_if_exists: bool = True
    n_startup_trials: int = 10
    multivariate: bool = True
    show_progress_bar: bool = False


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimizer using Optuna's TPE sampler.

    Uses Bayesian optimization to efficiently explore the parameter space,
    learning from previous evaluations to focus on promising regions.

    Features:
    - TPE (Tree-structured Parzen Estimator) for smart sampling
    - Automatic pruning of poor trials
    - Study persistence for checkpointing
    - Visualization support through Optuna

    Example:
        optimizer = BayesianOptimizer(
            parameter_space=space,
            objective_fn=my_objective,
            config=BayesianConfig(
                n_trials=100,
                study_name="mes_optimization",
                storage="sqlite:///optimization.db",
            ),
        )
        result = optimizer.optimize()

        # Resume later
        optimizer2 = BayesianOptimizer(
            parameter_space=space,
            objective_fn=my_objective,
            config=BayesianConfig(
                n_trials=50,  # Run 50 more
                study_name="mes_optimization",
                storage="sqlite:///optimization.db",
                load_if_exists=True,
            ),
        )
        result = optimizer2.optimize()
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        config: Optional[BayesianConfig] = None,
        n_trials: Optional[int] = None,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            parameter_space: Parameter space to search
            objective_fn: Function that takes params and returns metrics
            config: Bayesian optimization configuration
            n_trials: Shortcut for config.n_trials

        Raises:
            ImportError: If optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install with: pip install optuna>=3.3.0"
            )

        config = config or BayesianConfig()
        if n_trials is not None:
            config.n_trials = n_trials

        super().__init__(parameter_space, objective_fn, config)
        self.bayesian_config = config
        self._study: Optional["optuna.Study"] = None

    def _run_optimization(self) -> OptimizationResult:
        """
        Run Bayesian optimization using Optuna.

        Returns:
            OptimizationResult with best parameters found
        """
        logger.info(
            f"Starting Bayesian optimization with {self.bayesian_config.n_trials} trials"
        )

        # Create sampler
        sampler = self._create_sampler()

        # Create pruner
        pruner = self._create_pruner()

        # Determine direction
        direction = "maximize" if self.config.higher_is_better else "minimize"

        # Create or load study
        self._study = optuna.create_study(
            study_name=self.bayesian_config.study_name,
            storage=self.bayesian_config.storage,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            load_if_exists=self.bayesian_config.load_if_exists,
        )

        # Configure logging
        if self.config.verbose < 2:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        self._study.optimize(
            self._optuna_objective,
            n_trials=self.bayesian_config.n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.bayesian_config.show_progress_bar,
            callbacks=[self._progress_callback] if self.config.verbose >= 1 else None,
        )

        # Build result
        result = self._build_optuna_result()

        logger.info(
            f"Bayesian optimization complete: {len(self._study.trials)} total trials, "
            f"best {self.config.metric_name}={result.best_metric:.4f}"
        )

        return result

    def _create_sampler(self) -> "optuna.samplers.BaseSampler":
        """Create the Optuna sampler based on config."""
        sampler_type = self.bayesian_config.sampler.lower()

        if sampler_type == "tpe":
            return TPESampler(
                n_startup_trials=self.bayesian_config.n_startup_trials,
                multivariate=self.bayesian_config.multivariate,
                seed=self.config.random_seed,
            )
        elif sampler_type == "random":
            return RandomSampler(seed=self.config.random_seed)
        else:
            logger.warning(f"Unknown sampler '{sampler_type}', using TPE")
            return TPESampler(seed=self.config.random_seed)

    def _create_pruner(self) -> Optional["optuna.pruners.BasePruner"]:
        """Create the Optuna pruner based on config."""
        if self.bayesian_config.pruner is None:
            return None

        pruner_type = self.bayesian_config.pruner.lower()

        if pruner_type == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
            )
        elif pruner_type == "hyperband":
            return HyperbandPruner()
        else:
            logger.warning(f"Unknown pruner '{pruner_type}', using median")
            return MedianPruner()

    def _optuna_objective(self, trial: "optuna.Trial") -> float:
        """
        Optuna objective function that wraps our objective.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value to optimize
        """
        # Sample parameters using Optuna's suggestion API
        params = self._suggest_params(trial)

        # Evaluate with our objective function
        try:
            metrics = self.objective_fn(params)
            value = metrics.get(self.config.metric_name, 0.0)

            # Store all metrics as user attributes
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and not np.isnan(val):
                    trial.set_user_attr(key, float(val))

            # Create and store trial result
            trial_result = TrialResult(
                trial_id=trial.number,
                params=params,
                metrics=metrics,
                status="completed",
            )

            with self._lock:
                self._results.append(trial_result)
                if self._best_result is None or (
                    trial_result.is_better_than(
                        self._best_result,
                        self.config.metric_name,
                        self.config.higher_is_better
                    )
                ):
                    self._best_result = trial_result

            return value

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")

            trial_result = TrialResult(
                trial_id=trial.number,
                params=params,
                status="failed",
                error_message=str(e),
            )

            with self._lock:
                self._results.append(trial_result)

            # Return worst possible value
            return float('-inf') if self.config.higher_is_better else float('inf')

    def _suggest_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Suggest parameter values using Optuna's API.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameter values
        """
        params = {}

        for param in self.parameter_space.parameters:
            ptype = ParameterType(param.param_type)

            if ptype == ParameterType.CATEGORICAL:
                params[param.name] = trial.suggest_categorical(
                    param.name,
                    param.choices,
                )

            elif ptype == ParameterType.INT:
                step = int(param.step) if param.step else 1
                params[param.name] = trial.suggest_int(
                    param.name,
                    int(param.min_value),
                    int(param.max_value),
                    step=step,
                )

            else:  # FLOAT
                if param.log_scale:
                    params[param.name] = trial.suggest_float(
                        param.name,
                        param.min_value,
                        param.max_value,
                        log=True,
                    )
                elif param.step:
                    params[param.name] = trial.suggest_float(
                        param.name,
                        param.min_value,
                        param.max_value,
                        step=param.step,
                    )
                else:
                    params[param.name] = trial.suggest_float(
                        param.name,
                        param.min_value,
                        param.max_value,
                    )

        return params

    def _progress_callback(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial"
    ) -> None:
        """Callback for progress logging."""
        if trial.number % 10 == 0:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            rate = (trial.number + 1) / elapsed if elapsed > 0 else 0

            best_value = study.best_value if study.best_trial else 0

            logger.info(
                f"Trial {trial.number + 1}/{self.bayesian_config.n_trials}: "
                f"value={trial.value:.4f}, best={best_value:.4f}, "
                f"rate={rate:.1f}/s"
            )

    def _build_optuna_result(self) -> OptimizationResult:
        """Build OptimizationResult from Optuna study."""
        if self._study is None or not self._study.trials:
            return self._build_result()

        # Get best trial
        try:
            best_trial = self._study.best_trial
            best_params = best_trial.params
            best_metric = best_trial.value
        except ValueError:
            # No completed trials
            best_params = {}
            best_metric = 0.0

        # Get all metrics from best trial
        in_sample_metrics = {}
        if best_trial:
            for key, value in best_trial.user_attrs.items():
                in_sample_metrics[key] = value

        return OptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            metric_name=self.config.metric_name,
            all_results=list(self._results),
            in_sample_metrics=in_sample_metrics,
            parameter_space_name=self.parameter_space.name,
            optimizer_type="BayesianOptimizer",
            start_time=self._start_time,
            end_time=datetime.now(),
            total_trials=len(self._study.trials),
            successful_trials=len([
                t for t in self._study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]),
            config={
                "n_trials": self.bayesian_config.n_trials,
                "sampler": self.bayesian_config.sampler,
                "study_name": self.bayesian_config.study_name,
            },
        )

    def get_study(self) -> Optional["optuna.Study"]:
        """Get the underlying Optuna study."""
        return self._study

    def get_importance(self) -> Dict[str, float]:
        """
        Get parameter importance using Optuna's importance analyzer.

        Returns:
            Dict of parameter name -> importance score
        """
        if self._study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Failed to compute importance: {e}")
            return {}

    def save_study(self, path: str) -> None:
        """
        Save study to SQLite database.

        Args:
            path: Path to SQLite database file
        """
        if self._study is None:
            logger.warning("No study to save")
            return

        # Copy study to new storage
        storage = f"sqlite:///{path}"
        new_study = optuna.create_study(
            study_name=self._study.study_name,
            storage=storage,
            direction=self._study.direction,
        )

        for trial in self._study.trials:
            new_study.add_trial(trial)

        logger.info(f"Saved study to {path}")

    @classmethod
    def load_study(
        cls,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        path: str,
        study_name: str = "trading_optimization",
    ) -> "BayesianOptimizer":
        """
        Load a study from SQLite database.

        Args:
            parameter_space: Parameter space
            objective_fn: Objective function
            path: Path to SQLite database file
            study_name: Name of the study to load

        Returns:
            BayesianOptimizer with loaded study
        """
        storage = f"sqlite:///{path}"

        config = BayesianConfig(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        optimizer = cls(parameter_space, objective_fn, config)

        # Load the study
        optimizer._study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        logger.info(
            f"Loaded study with {len(optimizer._study.trials)} trials"
        )

        return optimizer


def run_bayesian_optimization(
    parameter_space: ParameterSpace,
    objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    n_trials: int = 100,
    metric_name: str = "sharpe_ratio",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    n_jobs: int = 1,
    seed: Optional[int] = 42,
) -> OptimizationResult:
    """
    Convenience function to run Bayesian optimization.

    Args:
        parameter_space: Space to search
        objective_fn: Objective function
        n_trials: Number of trials
        metric_name: Metric to optimize
        study_name: Name for the study
        storage: SQLite URL for persistence
        n_jobs: Number of parallel workers
        seed: Random seed

    Returns:
        OptimizationResult

    Example:
        result = run_bayesian_optimization(
            parameter_space=space,
            objective_fn=my_objective,
            n_trials=100,
            storage="sqlite:///my_study.db",
        )
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for Bayesian optimization. "
            "Install with: pip install optuna>=3.3.0"
        )

    config = BayesianConfig(
        n_trials=n_trials,
        metric_name=metric_name,
        study_name=study_name or "optimization",
        storage=storage,
        n_jobs=n_jobs,
        random_seed=seed,
    )

    optimizer = BayesianOptimizer(
        parameter_space=parameter_space,
        objective_fn=objective_fn,
        config=config,
    )

    return optimizer.optimize()


def create_visualization(
    study: "optuna.Study",
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate Optuna visualizations for a study.

    Args:
        study: Optuna study object
        output_dir: Directory to save plots

    Returns:
        Dict of plot name -> file path
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required for visualization")

    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_contour,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    try:
        # Optimization history
        fig = plot_optimization_history(study)
        path = output_path / "optimization_history.html"
        fig.write_html(str(path))
        plots["optimization_history"] = str(path)
    except Exception as e:
        logger.warning(f"Failed to create optimization history: {e}")

    try:
        # Parameter importances
        fig = plot_param_importances(study)
        path = output_path / "param_importances.html"
        fig.write_html(str(path))
        plots["param_importances"] = str(path)
    except Exception as e:
        logger.warning(f"Failed to create param importances: {e}")

    try:
        # Parallel coordinate
        fig = plot_parallel_coordinate(study)
        path = output_path / "parallel_coordinate.html"
        fig.write_html(str(path))
        plots["parallel_coordinate"] = str(path)
    except Exception as e:
        logger.warning(f"Failed to create parallel coordinate: {e}")

    return plots
