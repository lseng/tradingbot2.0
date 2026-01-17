"""
Walk-Forward Cross-Validation Optimizer for Time-Series Data.

This module implements walk-forward optimization, which is CRITICAL for
time-series trading strategy development:

Why Walk-Forward Matters:
- Standard k-fold CV causes temporal leakage (future data informs past)
- Walk-forward maintains temporal ordering: train -> validate -> test
- Each fold simulates real-world deployment: only historical data used
- Detects overfitting by measuring performance degradation across folds

Walk-Forward Process:
    Fold 1: [====TRAIN====][=VAL=][TEST]
    Fold 2:   [====TRAIN====][=VAL=][TEST]
    Fold 3:     [====TRAIN====][=VAL=][TEST]
    ...

Usage:
    from src.optimization.walk_forward import WalkForwardOptimizer, WalkForwardConfig

    config = WalkForwardConfig(
        training_months=6,
        validation_months=1,
        test_months=1,
        step_months=1,
        min_trades_per_fold=100,
    )

    optimizer = WalkForwardOptimizer(
        parameter_space=space,
        objective_fn_factory=make_objective_for_data,  # Creates objective for specific data
        config=config,
        inner_optimizer_class=RandomSearchOptimizer,
        inner_optimizer_kwargs={"n_iterations": 50},
    )

    result = optimizer.optimize(data)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
import pandas as pd
import numpy as np

from src.optimization.optimizer_base import BaseOptimizer, OptimizerConfig
from src.optimization.parameter_space import ParameterSpace
from src.optimization.results import (
    OptimizationResult,
    TrialResult,
    calculate_overfitting_score,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """
    Represents a single fold in walk-forward validation.

    Attributes:
        fold_id: Unique identifier for this fold
        train_start: Start date of training window
        train_end: End date of training window
        val_start: Start date of validation window (same as train_end)
        val_end: End date of validation window
        test_start: Start date of test window (same as val_end)
        test_end: End date of test window
    """
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def train_range(self) -> Tuple[datetime, datetime]:
        return (self.train_start, self.train_end)

    @property
    def val_range(self) -> Tuple[datetime, datetime]:
        return (self.val_start, self.val_end)

    @property
    def test_range(self) -> Tuple[datetime, datetime]:
        return (self.test_start, self.test_end)


@dataclass
class FoldResult:
    """
    Results from optimizing and testing a single fold.

    Attributes:
        fold: The fold configuration
        optimization_result: Result from inner optimizer on validation data
        test_metrics: Metrics from testing best params on holdout test data
        train_metrics: Metrics from best params on training data (for overfitting analysis)
    """
    fold: WalkForwardFold
    optimization_result: OptimizationResult
    test_metrics: Dict[str, float]
    train_metrics: Optional[Dict[str, float]] = None
    n_trades_test: int = 0
    overfitting_score: Optional[float] = None


@dataclass
class WalkForwardConfig(OptimizerConfig):
    """
    Configuration for walk-forward optimization.

    Time Window Configuration:
        training_months: 6 months of historical data for training
        validation_months: 1 month for parameter optimization
        test_months: 1 month for out-of-sample testing
        step_months: Roll forward by 1 month per fold

    Validation Thresholds:
        min_trades_per_fold: Minimum 100 trades required per fold for statistical validity
        max_overfitting_score: Maximum acceptable overfitting (validation vs test degradation)
    """
    # Time window configuration (in months)
    training_months: int = 6
    validation_months: int = 1
    test_months: int = 1
    step_months: int = 1

    # Validation thresholds
    min_trades_per_fold: int = 100
    max_overfitting_score: float = 0.3  # 30% degradation threshold

    # Fold handling
    skip_folds_below_min_trades: bool = True
    require_minimum_folds: int = 3  # At least 3 folds for robustness


@dataclass
class WalkForwardResult:
    """
    Aggregated results from walk-forward optimization.

    Provides comprehensive analysis across all folds including:
    - Best parameters (most robust across folds)
    - Average and per-fold metrics
    - Overfitting analysis
    - Consistency score
    """
    fold_results: List[FoldResult]
    best_params: Dict[str, Any]
    avg_test_metric: float
    avg_val_metric: float
    consistency_score: float  # Std dev of test metrics across folds
    overfitting_score: float  # Avg degradation from val to test
    metric_name: str
    config: WalkForwardConfig
    start_time: datetime
    end_time: datetime

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    @property
    def successful_folds(self) -> int:
        return sum(1 for fr in self.fold_results if fr.test_metrics)

    @property
    def is_robust(self) -> bool:
        """Check if strategy is robust based on consistency and overfitting."""
        return (
            self.consistency_score < 0.5 and  # Low variance across folds
            self.overfitting_score < self.config.max_overfitting_score and
            self.n_folds >= self.config.require_minimum_folds
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "WALK-FORWARD OPTIMIZATION RESULTS",
            "=" * 60,
            f"Folds: {self.successful_folds}/{self.n_folds} successful",
            f"Metric: {self.metric_name}",
            "",
            "PERFORMANCE:",
            f"  Avg Validation {self.metric_name}: {self.avg_val_metric:.4f}",
            f"  Avg Test {self.metric_name}: {self.avg_test_metric:.4f}",
            f"  Consistency Score: {self.consistency_score:.4f} (lower is better)",
            f"  Overfitting Score: {self.overfitting_score:.4f} (lower is better)",
            "",
            "BEST PARAMETERS:",
        ]

        for param, value in self.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {param}: {value:.6f}")
            else:
                lines.append(f"  {param}: {value}")

        lines.extend([
            "",
            f"ROBUSTNESS CHECK: {'PASS' if self.is_robust else 'FAIL'}",
            "=" * 60,
        ])

        return "\n".join(lines)


class WalkForwardOptimizer:
    """
    Walk-forward optimization for time-series trading strategies.

    This optimizer:
    1. Generates overlapping train/val/test folds respecting temporal order
    2. Runs inner optimizer on each fold's validation data
    3. Tests best params on each fold's holdout test data
    4. Aggregates results and measures overfitting/robustness

    Prevents temporal leakage by ensuring:
    - Training data is always before validation data
    - Validation data is always before test data
    - No future information leaks into historical periods
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_fn_factory: Callable[[pd.DataFrame], Callable[[Dict[str, Any]], Dict[str, float]]],
        config: Optional[WalkForwardConfig] = None,
        inner_optimizer_class: Optional[Type[BaseOptimizer]] = None,
        inner_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            parameter_space: Parameter space to optimize over
            objective_fn_factory: Function that takes a DataFrame and returns an
                                  objective function for that data. This allows
                                  creating separate objectives for each fold's data.
            config: Walk-forward configuration
            inner_optimizer_class: Optimizer class to use for each fold (e.g., RandomSearchOptimizer)
            inner_optimizer_kwargs: Additional kwargs for inner optimizer
        """
        self.parameter_space = parameter_space
        self.objective_fn_factory = objective_fn_factory
        self.config = config or WalkForwardConfig()

        # Default to RandomSearchOptimizer if not specified
        if inner_optimizer_class is None:
            from src.optimization.random_search import RandomSearchOptimizer
            inner_optimizer_class = RandomSearchOptimizer

        self.inner_optimizer_class = inner_optimizer_class
        self.inner_optimizer_kwargs = inner_optimizer_kwargs or {}

        # Results tracking
        self._fold_results: List[FoldResult] = []
        self._best_params: Optional[Dict[str, Any]] = None

    def generate_folds(self, data: pd.DataFrame) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds from data.

        Args:
            data: DataFrame with datetime index

        Returns:
            List of WalkForwardFold objects
        """
        if len(data) == 0:
            return []

        start_date = data.index[0]
        end_date = data.index[-1]

        folds = []
        fold_id = 0
        fold_start = start_date

        while True:
            # Calculate window boundaries
            train_end = fold_start + pd.DateOffset(months=self.config.training_months)
            val_end = train_end + pd.DateOffset(months=self.config.validation_months)
            test_end = val_end + pd.DateOffset(months=self.config.test_months)

            # Check if we have enough data for this fold
            if test_end > end_date:
                break

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=fold_start,
                train_end=train_end,
                val_start=train_end,
                val_end=val_end,
                test_start=val_end,
                test_end=test_end,
            )
            folds.append(fold)
            fold_id += 1

            # Roll forward
            fold_start = fold_start + pd.DateOffset(months=self.config.step_months)

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds

    def _extract_fold_data(
        self,
        data: pd.DataFrame,
        fold: WalkForwardFold
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract train/val/test data for a fold.

        Args:
            data: Full dataset
            fold: Fold configuration

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        train_data = data[
            (data.index >= fold.train_start) & (data.index < fold.train_end)
        ]
        val_data = data[
            (data.index >= fold.val_start) & (data.index < fold.val_end)
        ]
        test_data = data[
            (data.index >= fold.test_start) & (data.index < fold.test_end)
        ]

        return train_data, val_data, test_data

    def _optimize_fold(
        self,
        fold: WalkForwardFold,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> FoldResult:
        """
        Optimize a single fold.

        Args:
            fold: Fold configuration
            train_data: Training data (for overfitting analysis)
            val_data: Validation data (for optimization)
            test_data: Test data (for out-of-sample evaluation)

        Returns:
            FoldResult with optimization and test results
        """
        logger.info(f"Optimizing fold {fold.fold_id + 1}: "
                   f"Train {fold.train_start.date()} to {fold.train_end.date()}, "
                   f"Val {fold.val_start.date()} to {fold.val_end.date()}, "
                   f"Test {fold.test_start.date()} to {fold.test_end.date()}")

        # Create objective functions for validation and test data
        val_objective = self.objective_fn_factory(val_data)
        test_objective = self.objective_fn_factory(test_data)

        # Create and run inner optimizer
        # The inner optimizer uses its own kwargs which should include config parameters
        # We don't override them - let the user specify everything in inner_optimizer_kwargs
        inner_optimizer = self.inner_optimizer_class(
            parameter_space=self.parameter_space,
            objective_fn=val_objective,
            **self.inner_optimizer_kwargs,
        )

        optimization_result = inner_optimizer.optimize()

        # Test best params on holdout test data
        best_params = optimization_result.best_params
        test_metrics = {}
        train_metrics = None
        n_trades_test = 0

        if best_params:
            # Evaluate on test data
            try:
                test_metrics = test_objective(best_params)
                n_trades_test = test_metrics.get('n_trades', 0)
            except Exception as e:
                logger.warning(f"Test evaluation failed for fold {fold.fold_id}: {e}")
                test_metrics = {self.config.metric_name: 0.0}

            # Optionally evaluate on training data for overfitting analysis
            if self.config.compute_overfitting and len(train_data) > 0:
                try:
                    train_objective = self.objective_fn_factory(train_data)
                    train_metrics = train_objective(best_params)
                except Exception as e:
                    logger.debug(f"Train evaluation failed for fold {fold.fold_id}: {e}")

        # Calculate overfitting score for this fold
        val_metric = optimization_result.best_metric or 0.0
        test_metric = test_metrics.get(self.config.metric_name, 0.0)

        if val_metric > 0:
            overfitting_score = (val_metric - test_metric) / val_metric
        else:
            overfitting_score = 0.0

        return FoldResult(
            fold=fold,
            optimization_result=optimization_result,
            test_metrics=test_metrics,
            train_metrics=train_metrics,
            n_trades_test=n_trades_test,
            overfitting_score=overfitting_score,
        )

    def optimize(self, data: pd.DataFrame) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            data: Full dataset with datetime index

        Returns:
            WalkForwardResult with aggregated results
        """
        start_time = datetime.now()

        # Generate folds
        folds = self.generate_folds(data)

        if len(folds) < self.config.require_minimum_folds:
            raise ValueError(
                f"Not enough data for walk-forward: got {len(folds)} folds, "
                f"need at least {self.config.require_minimum_folds}"
            )

        # Process each fold
        self._fold_results = []
        param_scores: Dict[str, List[float]] = {}  # Track param performance

        for fold in folds:
            train_data, val_data, test_data = self._extract_fold_data(data, fold)

            # Check minimum trades
            if self.config.skip_folds_below_min_trades:
                # We'll check after getting results
                pass

            fold_result = self._optimize_fold(fold, train_data, val_data, test_data)

            # Skip folds with too few trades
            if (self.config.skip_folds_below_min_trades and
                fold_result.n_trades_test < self.config.min_trades_per_fold):
                logger.warning(
                    f"Skipping fold {fold.fold_id}: only {fold_result.n_trades_test} trades "
                    f"(min: {self.config.min_trades_per_fold})"
                )
                continue

            self._fold_results.append(fold_result)

            # Track param performance for best param selection
            if fold_result.optimization_result.best_params:
                for param, value in fold_result.optimization_result.best_params.items():
                    if param not in param_scores:
                        param_scores[param] = []
                    param_scores[param].append(value)

            if self.config.verbose >= 1:
                test_metric = fold_result.test_metrics.get(self.config.metric_name, 0.0)
                val_metric = fold_result.optimization_result.best_metric or 0.0
                logger.info(
                    f"Fold {fold.fold_id + 1}: Val {self.config.metric_name}={val_metric:.4f}, "
                    f"Test {self.config.metric_name}={test_metric:.4f}, "
                    f"Overfitting={fold_result.overfitting_score:.2%}"
                )

        # Calculate aggregated metrics
        if not self._fold_results:
            raise ValueError("No valid folds after filtering")

        # Average best params (or most common for categorical)
        best_params = self._aggregate_best_params(param_scores)

        # Calculate metrics
        val_metrics = [
            fr.optimization_result.best_metric or 0.0
            for fr in self._fold_results
        ]
        test_metrics = [
            fr.test_metrics.get(self.config.metric_name, 0.0)
            for fr in self._fold_results
        ]

        avg_val_metric = np.mean(val_metrics) if val_metrics else 0.0
        avg_test_metric = np.mean(test_metrics) if test_metrics else 0.0

        # Consistency = coefficient of variation of test metrics
        consistency_score = (
            np.std(test_metrics) / abs(np.mean(test_metrics))
            if test_metrics and np.mean(test_metrics) != 0
            else float('inf')
        )

        # Overfitting = average degradation from val to test
        overfitting_scores = [
            fr.overfitting_score for fr in self._fold_results
            if fr.overfitting_score is not None
        ]
        overfitting_score = np.mean(overfitting_scores) if overfitting_scores else 0.0

        end_time = datetime.now()

        result = WalkForwardResult(
            fold_results=self._fold_results,
            best_params=best_params,
            avg_test_metric=avg_test_metric,
            avg_val_metric=avg_val_metric,
            consistency_score=consistency_score,
            overfitting_score=overfitting_score,
            metric_name=self.config.metric_name,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
        )

        if self.config.verbose >= 1:
            print(result.summary())

        return result

    def _aggregate_best_params(
        self,
        param_scores: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate best params across folds.

        For numeric params: use median (robust to outliers)
        For categorical params: use mode (most common)
        """
        best_params = {}

        for param in self.parameter_space.parameters:
            values = param_scores.get(param.name, [])
            if not values:
                # Use default or midpoint
                if param.param_type == "categorical":
                    best_params[param.name] = param.choices[0] if param.choices else None
                else:
                    best_params[param.name] = (param.min_value + param.max_value) / 2
            elif param.param_type == "categorical":
                # Mode for categorical
                from collections import Counter
                best_params[param.name] = Counter(values).most_common(1)[0][0]
            else:
                # Median for numeric (robust to outliers)
                best_params[param.name] = float(np.median(values))

        return best_params


def run_walk_forward_optimization(
    parameter_space: ParameterSpace,
    objective_fn_factory: Callable[[pd.DataFrame], Callable[[Dict[str, Any]], Dict[str, float]]],
    data: pd.DataFrame,
    training_months: int = 6,
    validation_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
    inner_optimizer_class: Optional[Type[BaseOptimizer]] = None,
    inner_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> WalkForwardResult:
    """
    Convenience function for walk-forward optimization.

    Args:
        parameter_space: Parameter space to optimize
        objective_fn_factory: Factory function that creates objectives for data slices
        data: Full dataset with datetime index
        training_months: Months in training window
        validation_months: Months in validation window
        test_months: Months in test window
        step_months: Months to roll forward per fold
        inner_optimizer_class: Optimizer to use for each fold
        inner_optimizer_kwargs: Additional args for inner optimizer
        **kwargs: Additional config options

    Returns:
        WalkForwardResult with aggregated results
    """
    config = WalkForwardConfig(
        training_months=training_months,
        validation_months=validation_months,
        test_months=test_months,
        step_months=step_months,
        **kwargs,
    )

    optimizer = WalkForwardOptimizer(
        parameter_space=parameter_space,
        objective_fn_factory=objective_fn_factory,
        config=config,
        inner_optimizer_class=inner_optimizer_class,
        inner_optimizer_kwargs=inner_optimizer_kwargs,
    )

    return optimizer.optimize(data)
