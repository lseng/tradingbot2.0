"""
Grid Search Optimizer for Parameter Optimization.

This module implements exhaustive grid search over parameter combinations.
It evaluates every combination in the parameter space, making it suitable
for small search spaces or when complete coverage is required.

Features:
- Exhaustive search over all parameter combinations
- Parallel execution support for faster optimization
- Progress tracking and logging
- Best parameter tracking by multiple metrics

Complexity:
- Time: O(n1 * n2 * ... * nk) where ni is the number of values for parameter i
- For large spaces, consider random search or Bayesian optimization

Usage:
    from src.optimization.grid_search import GridSearchOptimizer
    from src.optimization.parameter_space import DefaultParameterSpaces

    space = DefaultParameterSpaces.mes_scalping()
    optimizer = GridSearchOptimizer(
        parameter_space=space,
        objective_fn=my_objective,
        config=OptimizerConfig(metric_name="sharpe_ratio", n_jobs=4),
    )
    result = optimizer.optimize()
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional
import numpy as np

from src.optimization.optimizer_base import BaseOptimizer, OptimizerConfig
from src.optimization.parameter_space import ParameterSpace
from src.optimization.results import OptimizationResult, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class GridSearchConfig(OptimizerConfig):
    """
    Configuration specific to grid search.

    Inherits from OptimizerConfig and adds:
        batch_size: Number of trials to run in each batch (for memory efficiency)
        shuffle: Whether to randomize evaluation order
        max_combinations: Maximum combinations to evaluate (0 = no limit)
    """
    batch_size: int = 100
    shuffle: bool = False
    max_combinations: int = 0


class GridSearchOptimizer(BaseOptimizer):
    """
    Exhaustive grid search optimizer.

    Evaluates all combinations of parameter values in the search space.
    Best used when:
    - Search space is small (< 10,000 combinations)
    - Complete coverage is required
    - Parallel resources are available

    Example:
        # Define parameter space
        space = ParameterSpace(parameters=[
            ParameterConfig("stop_ticks", 4, 12, 2, "int"),
            ParameterConfig("confidence", 0.5, 0.8, 0.1, "float"),
        ])

        # Create optimizer
        optimizer = GridSearchOptimizer(
            parameter_space=space,
            objective_fn=my_backtest_function,
            config=GridSearchConfig(
                metric_name="sharpe_ratio",
                n_jobs=4,
            ),
        )

        # Run optimization
        result = optimizer.optimize()
        print(f"Best params: {result.best_params}")
        print(f"Best Sharpe: {result.best_metric:.3f}")
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        config: Optional[GridSearchConfig] = None,
        holdout_objective_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    ):
        """
        Initialize grid search optimizer.

        Args:
            parameter_space: Parameter space to search
            objective_fn: Function that takes params and returns metrics
            config: Grid search configuration
            holdout_objective_fn: Optional separate objective for OOS evaluation.
                                  Use create_split_objective() to generate both functions.
        """
        config = config or GridSearchConfig()
        super().__init__(parameter_space, objective_fn, config, holdout_objective_fn)
        self.grid_config = config

    def _run_optimization(self) -> OptimizationResult:
        """
        Run exhaustive grid search.

        Returns:
            OptimizationResult with best parameters found
        """
        # Count total combinations
        total_combinations = self.parameter_space.count_grid_combinations()

        if total_combinations == 0:
            logger.warning("Empty parameter space - no combinations to evaluate")
            return self._build_result()

        # Apply max combinations limit
        if self.grid_config.max_combinations > 0:
            if total_combinations > self.grid_config.max_combinations:
                logger.warning(
                    f"Grid has {total_combinations} combinations, "
                    f"limiting to {self.grid_config.max_combinations}"
                )

        logger.info(
            f"Starting grid search with {total_combinations} combinations, "
            f"using {self.config.n_jobs} workers"
        )

        # Generate all parameter combinations
        combinations = list(self.parameter_space.get_grid_combinations())

        # Apply max combinations limit
        if self.grid_config.max_combinations > 0:
            combinations = combinations[:self.grid_config.max_combinations]

        # Shuffle if requested
        if self.grid_config.shuffle:
            self._rng.shuffle(combinations)

        # Run evaluations
        if self.config.n_jobs == 1:
            self._run_sequential(combinations)
        else:
            self._run_parallel(combinations)

        # Build and return result
        result = self._build_result()

        logger.info(
            f"Grid search complete: {len(self._results)} trials, "
            f"best {self.config.metric_name}={result.best_metric:.4f}"
        )

        return result

    def _run_sequential(self, combinations: List[Dict[str, Any]]) -> None:
        """Run evaluations sequentially."""
        total = len(combinations)

        for i, params in enumerate(combinations):
            result = self.evaluate_params(params)

            with self._lock:
                self._results.append(result)

            # Progress logging
            if self.config.verbose >= 1 and (i + 1) % 10 == 0:
                elapsed = (datetime.now() - self._start_time).total_seconds()
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0

                best_val = (
                    self._best_result.get_metric(self.config.metric_name)
                    if self._best_result else 0
                )

                logger.info(
                    f"Progress: {i + 1}/{total} ({100 * (i + 1) / total:.1f}%), "
                    f"best={best_val:.4f}, "
                    f"rate={rate:.1f}/s, ETA={eta:.0f}s"
                )

    def _run_parallel(self, combinations: List[Dict[str, Any]]) -> None:
        """Run evaluations in parallel batches."""
        total = len(combinations)
        batch_size = self.grid_config.batch_size
        completed = 0

        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # Process in batches to manage memory
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch = combinations[batch_start:batch_end]

                # Submit batch
                futures = {}
                for i, params in enumerate(batch):
                    trial_id = batch_start + i
                    future = executor.submit(self.evaluate_params, params, trial_id)
                    futures[future] = trial_id

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_per_trial)
                        with self._lock:
                            self._results.append(result)
                        completed += 1
                    except Exception as e:
                        trial_id = futures[future]
                        logger.warning(f"Trial {trial_id} failed: {e}")
                        with self._lock:
                            self._results.append(TrialResult(
                                trial_id=trial_id,
                                params=combinations[trial_id],
                                status="failed",
                                error_message=str(e),
                            ))
                        completed += 1

                # Progress logging
                if self.config.verbose >= 1:
                    elapsed = (datetime.now() - self._start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0

                    best_val = (
                        self._best_result.get_metric(self.config.metric_name)
                        if self._best_result else 0
                    )

                    logger.info(
                        f"Progress: {completed}/{total} ({100 * completed / total:.1f}%), "
                        f"best={best_val:.4f}, "
                        f"rate={rate:.1f}/s, ETA={eta:.0f}s"
                    )

    def get_search_space_size(self) -> int:
        """Get total number of combinations in search space."""
        return self.parameter_space.count_grid_combinations()

    def estimate_time(self, time_per_trial: float = 1.0) -> float:
        """
        Estimate total optimization time.

        Args:
            time_per_trial: Estimated seconds per trial

        Returns:
            Estimated total seconds
        """
        combinations = self.get_search_space_size()
        n_jobs = max(1, self.config.n_jobs)
        return (combinations * time_per_trial) / n_jobs


def run_grid_search(
    parameter_space: ParameterSpace,
    objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    metric_name: str = "sharpe_ratio",
    n_jobs: int = 1,
    verbose: int = 1,
) -> OptimizationResult:
    """
    Convenience function to run grid search with default settings.

    Args:
        parameter_space: Space to search
        objective_fn: Objective function
        metric_name: Metric to optimize
        n_jobs: Number of parallel workers
        verbose: Verbosity level

    Returns:
        OptimizationResult

    Example:
        result = run_grid_search(
            parameter_space=DefaultParameterSpaces.quick_search(),
            objective_fn=my_objective,
            metric_name="sharpe_ratio",
            n_jobs=4,
        )
    """
    config = GridSearchConfig(
        metric_name=metric_name,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    optimizer = GridSearchOptimizer(
        parameter_space=parameter_space,
        objective_fn=objective_fn,
        config=config,
    )

    return optimizer.optimize()


def grid_search_with_cv(
    parameter_space: ParameterSpace,
    objective_fn_factory: Callable[[int], Callable[[Dict[str, Any]], Dict[str, float]]],
    n_folds: int = 5,
    metric_name: str = "sharpe_ratio",
    n_jobs: int = 1,
) -> OptimizationResult:
    """
    Run grid search with cross-validation.

    Evaluates each parameter combination across multiple folds
    and averages the results.

    Args:
        parameter_space: Parameter space to search
        objective_fn_factory: Function that takes fold index and returns objective
        n_folds: Number of cross-validation folds
        metric_name: Metric to optimize
        n_jobs: Number of parallel workers

    Returns:
        OptimizationResult with averaged metrics
    """

    def cv_objective(params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate params across all folds and average."""
        fold_metrics = []

        for fold in range(n_folds):
            try:
                objective_fn = objective_fn_factory(fold)
                metrics = objective_fn(params)
                fold_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Fold {fold} failed for params {params}: {e}")

        if not fold_metrics:
            return {metric_name: float('-inf')}

        # Average metrics across folds
        avg_metrics = {}
        all_keys = set()
        for m in fold_metrics:
            all_keys.update(m.keys())

        for key in all_keys:
            values = [m.get(key, 0.0) for m in fold_metrics if key in m]
            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f"{key}_std"] = float(np.std(values))

        return avg_metrics

    config = GridSearchConfig(
        metric_name=metric_name,
        n_jobs=n_jobs,
        verbose=1,
    )

    optimizer = GridSearchOptimizer(
        parameter_space=parameter_space,
        objective_fn=cv_objective,
        config=config,
    )

    result = optimizer.optimize()
    result.config["n_folds"] = n_folds
    result.config["cv_mode"] = True

    return result
