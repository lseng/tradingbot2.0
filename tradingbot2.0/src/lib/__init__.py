"""
Shared utilities library for the trading bot.

This module provides common utilities used across the codebase:
- constants: Contract specifications, trading parameters, session times
- time_utils: Timezone handling, market calendar, session detection
- config: Unified configuration loading from YAML and environment variables
- logging: Structured logging with rotation and formatting
"""

from src.lib.constants import (
    MES_TICK_SIZE,
    MES_TICK_VALUE,
    MES_POINT_VALUE,
    MES_COMMISSION_PER_SIDE,
    MES_EXCHANGE_FEE_PER_SIDE,
    MES_ROUND_TRIP_COST,
    RTH_START,
    RTH_END,
    ETH_START,
    ETH_END,
    EOD_FLATTEN_TIME,
    NY_TIMEZONE,
    ContractSpec,
    MES_SPEC,
    ES_SPEC,
    MNQ_SPEC,
    NQ_SPEC,
)

from src.lib.time_utils import (
    get_ny_now,
    to_ny_time,
    is_rth,
    is_eth,
    is_market_open,
    get_session_start,
    get_session_end,
    minutes_to_close,
    is_trading_day,
    get_next_trading_day,
    get_eod_phase,
)

# EODPhase is imported lazily via __getattr__ to avoid circular dependency
# The canonical version is in src/risk/eod_manager.py
# Users can import directly: from src.risk.eod_manager import EODPhase

from src.lib.config import (
    TradingConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    RiskConfig,
    load_config,
    load_config_from_env,
    validate_config,
)

from src.lib.logging_utils import (
    setup_logging,
    get_logger,
    TradingLogger,
    TradingFormatter,
    LogLevel,
    log_trade,
    log_latency,
)

from src.lib.performance_monitor import (
    PerformanceMonitor,
    MetricType,
    LatencyStats,
    LatencySample,
    MemorySnapshot,
    Timer,
    AsyncTimer,
    measure_time,
    get_global_monitor,
    set_global_monitor,
    PERFORMANCE_THRESHOLDS,
)

from src.lib.alerts import (
    AlertChannel,
    AlertPriority,
    Alert,
    AlertConfig,
    AlertSender,
    ConsoleAlertSender,
    EmailAlertSender,
    SlackAlertSender,
    WebhookAlertSender,
    DiscordAlertSender,
    AlertManager,
    create_error_event_handler,
    create_alert_manager_from_env,
)

def __getattr__(name):
    """Lazy import for EODPhase to avoid circular dependency."""
    if name == "EODPhase":
        from src.risk.eod_manager import EODPhase
        return EODPhase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Constants
    "MES_TICK_SIZE",
    "MES_TICK_VALUE",
    "MES_POINT_VALUE",
    "MES_COMMISSION_PER_SIDE",
    "MES_EXCHANGE_FEE_PER_SIDE",
    "MES_ROUND_TRIP_COST",
    "RTH_START",
    "RTH_END",
    "ETH_START",
    "ETH_END",
    "EOD_FLATTEN_TIME",
    "NY_TIMEZONE",
    "ContractSpec",
    "MES_SPEC",
    "ES_SPEC",
    "MNQ_SPEC",
    "NQ_SPEC",
    # Time utilities
    "get_ny_now",
    "to_ny_time",
    "is_rth",
    "is_eth",
    "is_market_open",
    "get_session_start",
    "get_session_end",
    "minutes_to_close",
    "is_trading_day",
    "get_next_trading_day",
    "get_eod_phase",
    "EODPhase",
    # Config
    "TradingConfig",
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "TrainingConfig",
    "RiskConfig",
    "load_config",
    "load_config_from_env",
    "validate_config",
    # Logging
    "setup_logging",
    "get_logger",
    "TradingLogger",
    "TradingFormatter",
    "LogLevel",
    "log_trade",
    "log_latency",
    # Performance monitoring
    "PerformanceMonitor",
    "MetricType",
    "LatencyStats",
    "LatencySample",
    "MemorySnapshot",
    "Timer",
    "AsyncTimer",
    "measure_time",
    "get_global_monitor",
    "set_global_monitor",
    "PERFORMANCE_THRESHOLDS",
    # Alerts
    "AlertChannel",
    "AlertPriority",
    "Alert",
    "AlertConfig",
    "AlertSender",
    "ConsoleAlertSender",
    "EmailAlertSender",
    "SlackAlertSender",
    "WebhookAlertSender",
    "DiscordAlertSender",
    "AlertManager",
    "create_error_event_handler",
    "create_alert_manager_from_env",
]
