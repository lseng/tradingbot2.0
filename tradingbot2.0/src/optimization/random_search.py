"""
Random Search Optimizer for Parameter Optimization.

This module implements random sampling of the parameter space.
It's more efficient than grid search for high-dimensional spaces
and provides probabilistic coverage of the search space.

Key Benefits:
- Better exploration than grid search for high dimensions
- Configurable number of iterations
- Early stopping when no improvement
- Same API as GridSearchOptimizer

When to use:
- Large parameter spaces (> 1000 combinations)
- Initial exploration before focused search
- Limited computational budget

Usage:
    from src.optimization.random_search import RandomSearchOptimizer
    from src.optimization.parameter_space import DefaultParameterSpaces

    space = DefaultParameterSpaces.mes_scalping()
    optimizer = RandomSearchOptimizer(
        parameter_space=space,
        objective_fn=my_objective,
        n_iterations=100,
    )
    result = optimizer.optimize()
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np

from src.optimization.optimizer_base import BaseOptimizer, OptimizerConfig
from src.optimization.parameter_space import ParameterSpace
from src.optimization.results import OptimizationResult, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class RandomSearchConfig(OptimizerConfig):
    """
    Configuration specific to random search.

    Attributes:
        n_iterations: Number of random samples to evaluate
        early_stopping_rounds: Stop if no improvement for N rounds
        early_stopping_threshold: Minimum improvement to reset counter
        deduplicate: Whether to skip duplicate parameter combinations
        max_duplicates: Max duplicates to allow before stopping
    """
    n_iterations: int = 100
    early_stopping_rounds: int = 0  # 0 = disabled
    early_stopping_threshold: float = 0.001
    deduplicate: bool = True
    max_duplicates: int = 100


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random search optimizer with optional early stopping.

    Randomly samples parameter combinations from the search space.
    More efficient than grid search for high-dimensional spaces.

    Features:
    - Configurable number of iterations
    - Early stopping if no improvement
    - Deduplication of sampled combinations
    - Parallel execution support

    Example:
        optimizer = RandomSearchOptimizer(
            parameter_space=space,
            objective_fn=my_objective,
            config=RandomSearchConfig(
                n_iterations=200,
                early_stopping_rounds=50,
                n_jobs=4,
            ),
        )
        result = optimizer.optimize()
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        config: Optional[RandomSearchConfig] = None,
        n_iterations: Optional[int] = None,
        holdout_objective_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    ):
        """
        Initialize random search optimizer.

        Args:
            parameter_space: Parameter space to search
            objective_fn: Function that takes params and returns metrics
            config: Random search configuration
            n_iterations: Shortcut for config.n_iterations
            holdout_objective_fn: Optional separate objective for OOS evaluation.
                                  Use create_split_objective() to generate both functions.
        """
        config = config or RandomSearchConfig()
        if n_iterations is not None:
            config.n_iterations = n_iterations

        super().__init__(parameter_space, objective_fn, config, holdout_objective_fn)
        self.random_config = config

        # Track sampled combinations for deduplication
        self._sampled_hashes: Set[int] = set()

    def _run_optimization(self) -> OptimizationResult:
        """
        Run random search optimization.

        Returns:
            OptimizationResult with best parameters found
        """
        n_iterations = self.random_config.n_iterations

        logger.info(
            f"Starting random search with {n_iterations} iterations, "
            f"using {self.config.n_jobs} workers"
        )

        # Track early stopping
        rounds_without_improvement = 0
        best_metric_value = float('-inf') if self.config.higher_is_better else float('inf')
        duplicate_count = 0

        for i in range(n_iterations):
            # Sample random parameters
            params = self._sample_unique_params()

            if params is None:
                duplicate_count += 1
                if duplicate_count >= self.random_config.max_duplicates:
                    logger.info(
                        f"Stopping: too many duplicate samples ({duplicate_count})"
                    )
                    break
                continue

            # Evaluate
            result = self.evaluate_params(params, trial_id=i)

            with self._lock:
                self._results.append(result)

            # Check for improvement
            if result.status == "completed":
                current_metric = result.get_metric(self.config.metric_name)

                improved = False
                if self.config.higher_is_better:
                    if current_metric > best_metric_value + self.random_config.early_stopping_threshold:
                        improved = True
                        best_metric_value = current_metric
                else:
                    if current_metric < best_metric_value - self.random_config.early_stopping_threshold:
                        improved = True
                        best_metric_value = current_metric

                if improved:
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

            # Check early stopping
            if (self.random_config.early_stopping_rounds > 0 and
                    rounds_without_improvement >= self.random_config.early_stopping_rounds):
                logger.info(
                    f"Early stopping: no improvement for "
                    f"{self.random_config.early_stopping_rounds} rounds"
                )
                break

            # Progress logging
            if self.config.verbose >= 1 and (i + 1) % 10 == 0:
                self._log_progress(i + 1, n_iterations)

        # Build result
        result = self._build_result()

        logger.info(
            f"Random search complete: {len(self._results)} trials, "
            f"best {self.config.metric_name}={result.best_metric:.4f}"
        )

        return result

    def _sample_unique_params(self) -> Optional[Dict[str, Any]]:
        """
        Sample a unique parameter combination.

        Returns:
            Parameter dict or None if duplicate
        """
        # Sample from parameter space
        samples = self.parameter_space.sample_random(n=1, seed=None)
        if not samples:
            return None

        params = samples[0]

        # Check for duplicates
        if self.random_config.deduplicate:
            param_hash = self._hash_params(params)
            if param_hash in self._sampled_hashes:
                return None
            self._sampled_hashes.add(param_hash)

        return params

    def _hash_params(self, params: Dict[str, Any]) -> int:
        """Create a hash of parameter values for deduplication."""
        # Sort keys for consistent hashing
        items = sorted(params.items())
        # Round floats to avoid floating point issues
        rounded_items = []
        for k, v in items:
            if isinstance(v, float):
                rounded_items.append((k, round(v, 6)))
            else:
                rounded_items.append((k, v))
        return hash(tuple(rounded_items))

    def _log_progress(self, current: int, total: int) -> None:
        """Log progress information."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0

        best_val = (
            self._best_result.get_metric(self.config.metric_name)
            if self._best_result else 0
        )

        logger.info(
            f"Progress: {current}/{total} ({100 * current / total:.1f}%), "
            f"best={best_val:.4f}, "
            f"rate={rate:.1f}/s, ETA={eta:.0f}s"
        )

    def reset(self) -> None:
        """Reset optimizer state including sampled hashes."""
        self._reset_state()
        self._sampled_hashes.clear()


class AdaptiveRandomSearch(RandomSearchOptimizer):
    """
    Adaptive random search that focuses sampling around good regions.

    After initial exploration, concentrates sampling near the best
    parameters found so far, gradually reducing the search radius.

    Features:
    - Two-phase search: exploration then exploitation
    - Adaptive sampling radius
    - Combines random search with local refinement
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        config: Optional[RandomSearchConfig] = None,
        exploration_ratio: float = 0.3,
        min_radius: float = 0.1,
        holdout_objective_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    ):
        """
        Initialize adaptive random search.

        Args:
            parameter_space: Parameter space to search
            objective_fn: Function that takes params and returns metrics
            config: Random search configuration
            exploration_ratio: Fraction of iterations for pure exploration
            min_radius: Minimum sampling radius (as fraction of range)
            holdout_objective_fn: Optional separate objective for OOS evaluation.
                                  Use create_split_objective() to generate both functions.
        """
        super().__init__(parameter_space, objective_fn, config, holdout_objective_fn=holdout_objective_fn)
        self.exploration_ratio = exploration_ratio
        self.min_radius = min_radius

    def _run_optimization(self) -> OptimizationResult:
        """
        Run adaptive random search.

        Phase 1: Pure random exploration
        Phase 2: Focused sampling around best parameters
        """
        n_iterations = self.random_config.n_iterations
        n_exploration = int(n_iterations * self.exploration_ratio)
        n_exploitation = n_iterations - n_exploration

        logger.info(
            f"Starting adaptive random search: "
            f"{n_exploration} exploration + {n_exploitation} exploitation"
        )

        # Phase 1: Exploration
        logger.info("Phase 1: Exploration")
        for i in range(n_exploration):
            params = self._sample_unique_params()
            if params is None:
                continue

            result = self.evaluate_params(params, trial_id=i)
            with self._lock:
                self._results.append(result)

            if self.config.verbose >= 1 and (i + 1) % 10 == 0:
                self._log_progress(i + 1, n_iterations)

        # Get best parameters from exploration
        if self._best_result is None:
            logger.warning("No successful trials in exploration phase")
            return self._build_result()

        best_params = self._best_result.params.copy()

        # Phase 2: Exploitation (focused sampling around best)
        logger.info(f"Phase 2: Exploitation around best params: {best_params}")

        for i in range(n_exploitation):
            # Sample near best parameters with decreasing radius
            progress = i / max(1, n_exploitation - 1)
            radius = 1.0 - progress * (1.0 - self.min_radius)

            params = self._sample_near_best(best_params, radius)

            result = self.evaluate_params(
                params,
                trial_id=n_exploration + i
            )
            with self._lock:
                self._results.append(result)

            # Update best_params if we found better
            if result.status == "completed":
                if result.is_better_than(
                    self._best_result,
                    self.config.metric_name,
                    self.config.higher_is_better
                ):
                    best_params = result.params.copy()

            if self.config.verbose >= 1 and (i + 1) % 10 == 0:
                current = n_exploration + i + 1
                self._log_progress(current, n_iterations)

        return self._build_result()

    def _sample_near_best(
        self,
        best_params: Dict[str, Any],
        radius: float
    ) -> Dict[str, Any]:
        """
        Sample parameters near the best known values.

        Args:
            best_params: Best parameters found so far
            radius: Sampling radius as fraction of parameter range

        Returns:
            New parameter combination
        """
        new_params = {}

        for param in self.parameter_space.parameters:
            best_val = best_params.get(param.name)

            if param.param_type == "categorical":
                # For categorical, random choice with higher probability for current
                if self._rng.random() < 0.5:
                    new_params[param.name] = best_val
                else:
                    new_params[param.name] = param.sample_random(self._rng)

            else:
                # For numeric, sample in range around best value
                range_size = param.max_value - param.min_value
                delta = range_size * radius

                low = max(param.min_value, best_val - delta)
                high = min(param.max_value, best_val + delta)

                value = self._rng.uniform(low, high)

                if param.param_type == "int":
                    value = int(round(value))
                elif param.step is not None:
                    value = round(value / param.step) * param.step
                    value = round(value, 10)

                new_params[param.name] = value

        return new_params


def run_random_search(
    parameter_space: ParameterSpace,
    objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    n_iterations: int = 100,
    metric_name: str = "sharpe_ratio",
    early_stopping_rounds: int = 0,
    n_jobs: int = 1,
    seed: Optional[int] = 42,
) -> OptimizationResult:
    """
    Convenience function to run random search with default settings.

    Args:
        parameter_space: Space to search
        objective_fn: Objective function
        n_iterations: Number of random samples
        metric_name: Metric to optimize
        early_stopping_rounds: Stop after N rounds without improvement
        n_jobs: Number of parallel workers
        seed: Random seed

    Returns:
        OptimizationResult

    Example:
        result = run_random_search(
            parameter_space=space,
            objective_fn=my_objective,
            n_iterations=200,
            early_stopping_rounds=50,
        )
    """
    config = RandomSearchConfig(
        n_iterations=n_iterations,
        metric_name=metric_name,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        random_seed=seed,
    )

    optimizer = RandomSearchOptimizer(
        parameter_space=parameter_space,
        objective_fn=objective_fn,
        config=config,
    )

    return optimizer.optimize()
