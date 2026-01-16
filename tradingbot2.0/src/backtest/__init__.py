"""
Backtesting Engine Module for MES Futures Scalping

This module provides event-driven backtesting capabilities with:
- Realistic transaction cost modeling (MES-specific: $0.84 round-trip)
- Tick-based slippage simulation
- Full integration with risk management module
- Walk-forward optimization support
- Comprehensive performance metrics

The backtesting engine is critical for validating ML model predictions
before risking real capital. It enforces all risk limits and EOD rules
in simulation to ensure the live trading system behaves identically.
"""

from .costs import TransactionCostModel, MESCostConfig
from .slippage import SlippageModel, SlippageConfig, MarketCondition
from .metrics import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
)
from .trade_logger import (
    TradeLog,
    TradeRecord,
    EquityCurve,
    EquityPoint,
    BacktestReport,
)
from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    OrderFillMode,
    Signal,
    SignalType,
)

__all__ = [
    # Cost model
    "TransactionCostModel",
    "MESCostConfig",
    # Slippage model
    "SlippageModel",
    "SlippageConfig",
    "MarketCondition",
    # Metrics
    "PerformanceMetrics",
    "calculate_metrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    # Logging
    "TradeLog",
    "TradeRecord",
    "EquityCurve",
    "EquityPoint",
    "BacktestReport",
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "OrderFillMode",
    "Signal",
    "SignalType",
]
