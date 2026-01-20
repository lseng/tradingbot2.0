"""
5-Minute Scalping System

LightGBM-based scalping system for MES futures using 5-minute bars.
This approach replaces the previous neural network approach that did not achieve profitability.

Key components:
- data_pipeline: Data loading, 5-min aggregation, RTH filtering, train/val/test splits
- features: 24-feature generator for 5-minute bars
- model: LightGBM classifier with confidence-based filtering
- backtest: Simplified backtest engine with 30-min time stops
- breakout: Breakout detection strategy using volatility prediction
"""

from .data_pipeline import (
    ScalpingDataPipeline,
    DataConfig,
    load_1min_data,
    aggregate_to_5min,
    filter_rth,
    create_temporal_splits,
)
from .features import (
    ScalpingFeatureGenerator,
    FeatureConfig,
    create_target_variable,
    create_volatility_target,
)
from .model import (
    ScalpingModel,
    ModelConfig,
    TrainingResult,
    hyperparameter_search,
)
from .walk_forward import (
    WalkForwardCV,
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    run_walk_forward_validation,
)
from .backtest import (
    ScalpingBacktest,
    BacktestConfig,
    BacktestResult,
    Trade,
    ExitReason,
    run_backtest,
    analyze_results,
    export_trades_csv,
    export_summary_json,
)
from .breakout import (
    BreakoutConfig,
    BreakoutFeatureGenerator,
    BreakoutTrader,
    ConsolidationType,
    create_breakout_target,
    identify_breakout_setups,
    run_breakout_backtest,
)

__all__ = [
    # Data pipeline
    "ScalpingDataPipeline",
    "DataConfig",
    "load_1min_data",
    "aggregate_to_5min",
    "filter_rth",
    "create_temporal_splits",
    # Features
    "ScalpingFeatureGenerator",
    "FeatureConfig",
    "create_target_variable",
    "create_volatility_target",
    # Model
    "ScalpingModel",
    "ModelConfig",
    "TrainingResult",
    "hyperparameter_search",
    # Walk-forward validation
    "WalkForwardCV",
    "WalkForwardConfig",
    "WalkForwardResult",
    "FoldResult",
    "run_walk_forward_validation",
    # Backtesting
    "ScalpingBacktest",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "ExitReason",
    "run_backtest",
    "analyze_results",
    "export_trades_csv",
    "export_summary_json",
    # Breakout detection
    "BreakoutConfig",
    "BreakoutFeatureGenerator",
    "BreakoutTrader",
    "ConsolidationType",
    "create_breakout_target",
    "identify_breakout_setups",
    "run_breakout_backtest",
]
