"""
Base Optimizer Class for Parameter Optimization.

This module defines the abstract base class for all optimization strategies.
It provides:
- Common interface for optimization
- Objective function wrapper for backtesting
- Results aggregation
- Overfitting metrics computation
- Thread-safe state management

All optimizer implementations (grid search, random search, Bayesian)
inherit from BaseOptimizer.

Usage:
    class MyOptimizer(BaseOptimizer):
        def optimize(self, **kwargs) -> OptimizationResult:
            # Implementation here
            pass
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.optimization.parameter_space import ParameterSpace
from src.optimization.results import (
    OptimizationResult,
    TrialResult,
    calculate_overfitting_score,
)

# Get logger
logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer behavior.

    Attributes:
        metric_name: Target metric to optimize (sharpe_ratio, calmar_ratio, etc.)
        higher_is_better: Whether higher metric values are better
        n_jobs: Number of parallel workers (1 = sequential)
        verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
        random_seed: Random seed for reproducibility
        timeout_per_trial: Max seconds per trial (None = no limit)
        min_trials: Minimum trials before early stopping
        compute_overfitting: Whether to compute overfitting metrics
    """
    metric_name: str = "sharpe_ratio"
    higher_is_better: bool = True
    n_jobs: int = 1
    verbose: int = 1
    random_seed: Optional[int] = 42
    timeout_per_trial: Optional[float] = None
    min_trials: int = 10
    compute_overfitting: bool = True


class BaseOptimizer(ABC):
    """
    Abstract base class for parameter optimizers.

    Provides common functionality for:
    - Objective function evaluation
    - Results tracking
    - Parallel execution
    - Overfitting analysis

    Subclasses must implement the `_run_optimization` method.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        config: Optional[OptimizerConfig] = None,
        holdout_objective_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            parameter_space: Parameter space to optimize over
            objective_fn: Function that takes params and returns metrics dict
            config: Optimizer configuration
            holdout_objective_fn: Optional separate objective function for OOS evaluation.
                                  If provided, this will be used for out-of-sample testing
                                  instead of reusing objective_fn (which would be incorrect).
                                  Use create_split_objective() to generate both functions.
        """
        self.parameter_space = parameter_space
        self.objective_fn = objective_fn
        self.config = config or OptimizerConfig()
        self.holdout_objective_fn = holdout_objective_fn

        # State tracking
        self._results: List[TrialResult] = []
        self._best_result: Optional[TrialResult] = None
        self._trial_counter = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._lock = threading.Lock()

        # Random state
        self._rng = np.random.default_rng(self.config.random_seed)

    @abstractmethod
    def _run_optimization(self) -> OptimizationResult:
        """
        Run the optimization algorithm.

        This method should be implemented by subclasses to define
        the specific optimization strategy.

        Returns:
            OptimizationResult with best parameters and all trials
        """
        pass

    def optimize(
        self,
        validation_data: Optional[pd.DataFrame] = None,
        holdout_data: Optional[pd.DataFrame] = None,
    ) -> OptimizationResult:
        """
        Run optimization and return results.

        Args:
            validation_data: Data for in-sample validation
            holdout_data: Held-out data for out-of-sample testing

        Returns:
            OptimizationResult with best parameters and analysis
        """
        self._reset_state()
        self._start_time = datetime.now()

        try:
            # Store validation/holdout data for overfitting analysis
            self._validation_data = validation_data
            self._holdout_data = holdout_data

            # Run the specific optimization algorithm
            result = self._run_optimization()

            # Compute overfitting metrics if data available
            if self.config.compute_overfitting and holdout_data is not None:
                result = self._compute_overfitting_metrics(result)

            self._end_time = datetime.now()
            result.start_time = self._start_time
            result.end_time = self._end_time

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            self._end_time = datetime.now()
            return self._build_failed_result(str(e))

    def _reset_state(self) -> None:
        """Reset internal state for a new optimization run."""
        with self._lock:
            self._results = []
            self._best_result = None
            self._trial_counter = 0
            self._start_time = None
            self._end_time = None

    def evaluate_params(
        self,
        params: Dict[str, Any],
        trial_id: Optional[int] = None,
    ) -> TrialResult:
        """
        Evaluate a single parameter combination.

        Args:
            params: Parameter values to evaluate
            trial_id: Optional trial ID (auto-assigned if None)

        Returns:
            TrialResult with metrics
        """
        # Assign trial ID
        with self._lock:
            if trial_id is None:
                trial_id = self._trial_counter
                self._trial_counter += 1

        start_time = time.time()

        try:
            # Validate parameters
            valid, errors = self.parameter_space.validate_params(params)
            if not valid:
                return TrialResult(
                    trial_id=trial_id,
                    params=params,
                    status="failed",
                    error_message=f"Invalid params: {errors}",
                )

            # Run objective function
            metrics = self.objective_fn(params)

            duration = time.time() - start_time

            result = TrialResult(
                trial_id=trial_id,
                params=params,
                metrics=metrics,
                status="completed",
                duration_seconds=duration,
            )

            # Update best result
            self._update_best(result)

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.warning(f"Trial {trial_id} failed: {e}")

            return TrialResult(
                trial_id=trial_id,
                params=params,
                status="failed",
                error_message=str(e),
                duration_seconds=duration,
            )

    def _update_best(self, result: TrialResult) -> None:
        """Update best result if this one is better."""
        if result.status != "completed":
            return

        with self._lock:
            if self._best_result is None:
                self._best_result = result
                return

            is_better = result.is_better_than(
                self._best_result,
                self.config.metric_name,
                self.config.higher_is_better,
            )

            if is_better:
                self._best_result = result
                if self.config.verbose >= 1:
                    logger.info(
                        f"New best: {self.config.metric_name}="
                        f"{result.get_metric(self.config.metric_name):.4f} "
                        f"params={result.params}"
                    )

    def evaluate_batch(
        self,
        param_list: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[TrialResult]:
        """
        Evaluate multiple parameter combinations.

        Args:
            param_list: List of parameter dicts to evaluate
            parallel: Whether to use parallel execution

        Returns:
            List of TrialResult
        """
        if not parallel or self.config.n_jobs == 1:
            results = []
            for i, params in enumerate(param_list):
                result = self.evaluate_params(params)
                results.append(result)
                if self.config.verbose >= 1 and (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(param_list)} trials")
            return results

        # Parallel execution
        results = []
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = {
                executor.submit(self.evaluate_params, params): i
                for i, params in enumerate(param_list)
            }

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout_per_trial)
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"Trial {idx} timed out or failed: {e}")
                    results.append(TrialResult(
                        trial_id=idx,
                        params=param_list[idx],
                        status="failed",
                        error_message=str(e),
                    ))

        return sorted(results, key=lambda r: r.trial_id)

    def _compute_overfitting_metrics(
        self,
        result: OptimizationResult,
    ) -> OptimizationResult:
        """
        Compute overfitting metrics by evaluating on holdout data.

        IMPORTANT: For accurate overfitting detection, the holdout evaluation
        MUST use a separate objective function that operates on holdout data,
        not the same objective_fn used for optimization (which uses validation data).

        Use create_split_objective() to generate separate validation and holdout
        objective functions, then pass holdout_objective_fn to the constructor.

        Args:
            result: Optimization result with best params

        Returns:
            Updated result with overfitting analysis
        """
        if result.best_params is None:
            return result

        # Check if we have a proper holdout objective function
        # OOS evaluation REQUIRES a separate holdout_objective_fn to be meaningful
        if self.holdout_objective_fn is None:
            if self._holdout_data is not None:
                logger.warning(
                    "holdout_data was provided but no holdout_objective_fn was set. "
                    "Skipping OOS evaluation - using the same objective function for both "
                    "in-sample and out-of-sample would defeat the purpose of overfitting detection. "
                    "Use create_split_objective() to create separate validation and holdout objectives."
                )
            # Cannot do proper OOS evaluation without separate holdout objective
            # Returning without OOS metrics is better than returning incorrect metrics
            return result
        else:
            # Proper setup - use the holdout objective function
            oos_objective = self.holdout_objective_fn

        try:
            # Get in-sample metrics from best trial
            best_trial = result.get_best_trial()
            if best_trial:
                result.in_sample_metrics = best_trial.metrics.copy()

            # Evaluate on holdout data using the SEPARATE holdout objective
            # This is critical for detecting overfitting - the holdout objective
            # must use different data than what was used for optimization
            oos_metrics = oos_objective(result.best_params)
            result.out_of_sample_metrics = oos_metrics

            # Calculate overfitting score
            is_value = result.in_sample_metrics.get(self.config.metric_name, 0.0)
            oos_value = oos_metrics.get(self.config.metric_name, 0.0)
            result.overfitting_score = calculate_overfitting_score(is_value, oos_value)

            logger.info(
                f"Overfitting analysis: IS={is_value:.4f}, "
                f"OOS={oos_value:.4f}, ratio={result.overfitting_score:.2f}"
            )

        except Exception as e:
            logger.warning(f"Failed to compute overfitting metrics: {e}")

        return result

    def _build_result(self) -> OptimizationResult:
        """Build optimization result from current state."""
        with self._lock:
            if self._best_result is None:
                best_params = {}
                best_metric = 0.0
            else:
                best_params = self._best_result.params
                best_metric = self._best_result.get_metric(self.config.metric_name)

            return OptimizationResult(
                best_params=best_params,
                best_metric=best_metric,
                metric_name=self.config.metric_name,
                all_results=list(self._results),
                parameter_space_name=self.parameter_space.name,
                optimizer_type=self.__class__.__name__,
                total_trials=len(self._results),
                successful_trials=sum(
                    1 for r in self._results if r.status == "completed"
                ),
                config={
                    "metric_name": self.config.metric_name,
                    "higher_is_better": self.config.higher_is_better,
                    "n_jobs": self.config.n_jobs,
                    "random_seed": self.config.random_seed,
                },
            )

    def _build_failed_result(self, error_message: str) -> OptimizationResult:
        """Build a failed optimization result."""
        return OptimizationResult(
            best_params={},
            best_metric=0.0,
            metric_name=self.config.metric_name,
            all_results=list(self._results),
            parameter_space_name=self.parameter_space.name,
            optimizer_type=self.__class__.__name__,
            start_time=self._start_time,
            end_time=self._end_time,
            config={"error": error_message},
        )

    def get_current_best(self) -> Optional[Dict[str, Any]]:
        """Get current best parameters (thread-safe)."""
        with self._lock:
            if self._best_result is None:
                return None
            return self._best_result.params.copy()

    def get_trial_count(self) -> int:
        """Get current trial count (thread-safe)."""
        with self._lock:
            return len(self._results)


def create_backtest_objective(
    engine,  # BacktestEngine
    data: pd.DataFrame,
    signal_generator_factory: Callable[[Dict[str, Any]], Callable],
    context: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """
    Create an objective function for backtesting optimization.

    This factory creates a function that:
    1. Takes parameters as input
    2. Creates a signal generator with those parameters
    3. Runs a backtest
    4. Returns performance metrics

    Args:
        engine: BacktestEngine instance
        data: Market data for backtesting
        signal_generator_factory: Function that creates signal generator from params
        context: Optional context for signal generator

    Returns:
        Objective function for optimization

    Example:
        def make_signal_gen(params):
            return create_simple_signal_generator(
                min_confidence=params["confidence_threshold"],
                stop_ticks=params["stop_ticks"],
                target_ticks=params["target_ticks"],
            )

        objective = create_backtest_objective(
            engine=engine,
            data=market_data,
            signal_generator_factory=make_signal_gen,
        )

        # Use with optimizer
        optimizer = GridSearchOptimizer(space, objective)
    """
    def objective(params: Dict[str, Any]) -> Dict[str, float]:
        # Update engine config with parameters
        if hasattr(engine, 'config'):
            if "stop_ticks" in params:
                engine.config.default_stop_ticks = params["stop_ticks"]
            if "target_ticks" in params:
                engine.config.default_target_ticks = params["target_ticks"]
            if "confidence_threshold" in params:
                engine.config.min_confidence = params["confidence_threshold"]

        # Create signal generator
        signal_gen = signal_generator_factory(params)

        # Run backtest
        result = engine.run(data, signal_gen, context=context, verbose=False)

        # Extract metrics
        metrics = result.report.metrics.to_dict()

        # Flatten nested metrics for optimization
        flat_metrics = {}
        for category, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[key] = float(value)
            elif isinstance(values, (int, float)):
                flat_metrics[category] = float(values)

        return flat_metrics

    return objective


def create_split_objective(
    engine,  # BacktestEngine
    validation_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    signal_generator_factory: Callable[[Dict[str, Any]], Callable],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Callable[[Dict[str, Any]], Dict[str, float]],
    Callable[[Dict[str, Any]], Dict[str, float]],
]:
    """
    Create separate objective functions for validation and holdout data.

    Used for proper overfitting prevention:
    - Optimize on validation_data
    - Test best parameters on holdout_data

    Args:
        engine: BacktestEngine instance
        validation_data: Data for optimization
        holdout_data: Held-out data for final testing
        signal_generator_factory: Creates signal generator from params
        context: Optional context

    Returns:
        Tuple of (validation_objective, holdout_objective)
    """
    val_objective = create_backtest_objective(
        engine, validation_data, signal_generator_factory, context
    )

    holdout_objective = create_backtest_objective(
        engine, holdout_data, signal_generator_factory, context
    )

    return val_objective, holdout_objective
