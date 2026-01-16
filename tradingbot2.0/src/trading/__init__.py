"""
Live Trading Module for MES Futures Scalping Bot.

This module provides the complete live trading system including:
- Position tracking and management
- Signal generation from ML predictions
- Order execution via TopstepX API
- Real-time feature calculation
- Error handling and recovery
- Main trading loop orchestration

Components:
- PositionManager: Tracks position state, calculates P&L
- SignalGenerator: Generates trading signals from model predictions
- OrderExecutor: Executes orders via TopstepX API
- RealTimeFeatureEngine: Calculates features from tick data
- RecoveryHandler: Handles errors and recovery
- LiveTrader: Main trading loop orchestrator

Usage:
    from src.trading import LiveTrader, TradingConfig

    config = TradingConfig(
        contract_id="CON.F.US.MES.H26",
        starting_capital=1000.0,
    )

    trader = LiveTrader(config, api_key="...")
    await trader.start()

Reference: specs/live-trading-execution.md
"""

# Position Management
from src.trading.position_manager import (
    Position,
    PositionDirection,
    PositionManager,
    PositionChange,
    Fill,
)

# Signal Generation
from src.trading.signal_generator import (
    Signal,
    SignalType,
    SignalConfig,
    SignalGenerator,
    ModelPrediction,
    is_entry_signal,
    is_exit_signal,
    is_reversal_signal,
    signal_to_direction,
)

# Order Execution
from src.trading.order_executor import (
    OrderExecutor,
    ExecutorConfig,
    EntryResult,
    ExecutionStatus,
)

# Real-Time Features
from src.trading.rt_features import (
    RealTimeFeatureEngine,
    RTFeaturesConfig,
    BarAggregator,
    FeatureVector,
    OHLCV,
)

# Error Recovery
from src.trading.recovery import (
    RecoveryHandler,
    RecoveryConfig,
    ErrorEvent,
    ErrorSeverity,
    ErrorCategory,
    with_retry,
    with_timeout,
)

# Main Trading Loop
from src.trading.live_trader import (
    LiveTrader,
    TradingConfig,
    SessionMetrics,
    run_live_trading,
)

__all__ = [
    # Position Management
    "Position",
    "PositionDirection",
    "PositionManager",
    "PositionChange",
    "Fill",
    # Signal Generation
    "Signal",
    "SignalType",
    "SignalConfig",
    "SignalGenerator",
    "ModelPrediction",
    "is_entry_signal",
    "is_exit_signal",
    "is_reversal_signal",
    "signal_to_direction",
    # Order Execution
    "OrderExecutor",
    "ExecutorConfig",
    "EntryResult",
    "ExecutionStatus",
    # Real-Time Features
    "RealTimeFeatureEngine",
    "RTFeaturesConfig",
    "BarAggregator",
    "FeatureVector",
    "OHLCV",
    # Error Recovery
    "RecoveryHandler",
    "RecoveryConfig",
    "ErrorEvent",
    "ErrorSeverity",
    "ErrorCategory",
    "with_retry",
    "with_timeout",
    # Main Trading Loop
    "LiveTrader",
    "TradingConfig",
    "SessionMetrics",
    "run_live_trading",
]
