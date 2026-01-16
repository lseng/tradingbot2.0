"""
Parameter Optimization Framework for Trading Strategy Backtesting.

This module provides a comprehensive framework for optimizing trading
strategy parameters using various search methods:

- Grid Search: Exhaustive search over all parameter combinations
- Random Search: Random sampling with optional early stopping
- Bayesian Optimization: Smart sampling using Optuna's TPE sampler

Key Features:
- Overfitting prevention: Optimize on validation, test on holdout
- Multiple metrics support: Sharpe, Calmar, profit_factor, etc.
- Parallel execution for faster optimization
- Study persistence for save/resume capability
- Comprehensive results analysis

Typical Workflow:
    1. Define parameter space with ParameterConfig
    2. Create objective function from backtest engine
    3. Choose optimizer (Grid, Random, or Bayesian)
    4. Run optimization with validation/holdout split
    5. Analyze results for overfitting

Example:
    from src.optimization import (
        BayesianOptimizer,
        DefaultParameterSpaces,
        create_backtest_objective,
    )

    # Define parameter space
    space = DefaultParameterSpaces.mes_scalping()

    # Create objective function
    objective = create_backtest_objective(
        engine=backtest_engine,
        data=validation_data,
        signal_generator_factory=make_signal_gen,
    )

    # Run optimization
    optimizer = BayesianOptimizer(
        parameter_space=space,
        objective_fn=objective,
        n_trials=100,
    )
    result = optimizer.optimize(
        validation_data=val_data,
        holdout_data=test_data,
    )

    # Analyze results
    print(result.summary())
    print(f"Overfitting score: {result.overfitting_score}")
"""

from src.optimization.parameter_space import (
    ParameterConfig,
    ParameterSpace,
    ParameterType,
    DefaultParameterSpaces,
    create_parameter_space_from_config,
)

from src.optimization.results import (
    TrialResult,
    OptimizationResult,
    OptimizationStatus,
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

# Bayesian optimizer requires optuna - import conditionally
try:
    from src.optimization.bayesian_optimizer import (
        BayesianOptimizer,
        BayesianConfig,
        run_bayesian_optimization,
        create_visualization,
        OPTUNA_AVAILABLE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False
    BayesianOptimizer = None  # type: ignore
    BayesianConfig = None  # type: ignore
    run_bayesian_optimization = None  # type: ignore
    create_visualization = None  # type: ignore


__all__ = [
    # Parameter Space
    "ParameterConfig",
    "ParameterSpace",
    "ParameterType",
    "DefaultParameterSpaces",
    "create_parameter_space_from_config",
    # Results
    "TrialResult",
    "OptimizationResult",
    "OptimizationStatus",
    "calculate_overfitting_score",
    "is_overfitting",
    "merge_results",
    # Base Optimizer
    "BaseOptimizer",
    "OptimizerConfig",
    "create_backtest_objective",
    "create_split_objective",
    # Grid Search
    "GridSearchOptimizer",
    "GridSearchConfig",
    "run_grid_search",
    "grid_search_with_cv",
    # Random Search
    "RandomSearchOptimizer",
    "RandomSearchConfig",
    "AdaptiveRandomSearch",
    "run_random_search",
    # Bayesian Optimization (optional)
    "BayesianOptimizer",
    "BayesianConfig",
    "run_bayesian_optimization",
    "create_visualization",
    "OPTUNA_AVAILABLE",
]
