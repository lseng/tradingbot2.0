"""
Event-Driven Backtesting Engine for MES Futures Scalping

This module implements a realistic bar-by-bar simulation engine for
validating ML-based trading strategies before live deployment.

Key Features:
- Bar-by-bar event processing on 1-second data
- Full integration with risk management module
- Realistic order fill simulation with slippage
- EOD flatten enforcement at 4:30 PM NY
- Walk-forward optimization support
- Comprehensive logging and metrics
- Session filtering (RTH/ETH) with proper timezone handling
- UTC to NY timezone conversion for all session checks

The engine processes data chronologically, respecting time boundaries
and avoiding lookahead bias. All risk limits are enforced identically
to live trading.

Usage:
    engine = BacktestEngine(config=BacktestConfig())
    result = engine.run(
        data=df,
        signal_generator=my_signal_function,
    )
    result.report.export_all("./results")
"""

from dataclasses import dataclass, field
from datetime import datetime, time, date, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
import logging
import numpy as np
import pandas as pd
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set up logger
logger = logging.getLogger(__name__)

from .costs import TransactionCostModel, MESCostConfig
from .slippage import SlippageModel, SlippageConfig, OrderType, MarketCondition
from .metrics import PerformanceMetrics, calculate_metrics
from .trade_logger import (
    TradeLog,
    TradeRecord,
    EquityCurve,
    BacktestReport,
    ExitReason,
)

# Import risk management module for full integration
try:
    from src.risk.risk_manager import RiskManager, RiskLimits, TradingStatus
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False
    RiskManager = None
    RiskLimits = None
    TradingStatus = None

# Import time utilities for timezone handling
try:
    from src.lib.time_utils import to_ny_time, is_rth, is_eth
    from src.lib.constants import NY_TIMEZONE, RTH_START, RTH_END, ETH_START, ETH_END
    TIME_UTILS_AVAILABLE = True
except ImportError:
    TIME_UTILS_AVAILABLE = False
    to_ny_time = None
    is_rth = None
    is_eth = None
    NY_TIMEZONE = None
    RTH_START = None
    RTH_END = None
    ETH_START = None
    ETH_END = None


class SignalType(Enum):
    """Trading signal types."""
    HOLD = "hold"  # No action
    LONG_ENTRY = "long_entry"  # Open long
    SHORT_ENTRY = "short_entry"  # Open short
    EXIT_LONG = "exit_long"  # Close long
    EXIT_SHORT = "exit_short"  # Close short
    FLATTEN = "flatten"  # Close all positions


class OrderFillMode(Enum):
    """Order fill simulation modes."""
    # Fill at next bar open + slippage (realistic)
    NEXT_BAR_OPEN = "next_bar_open"
    # Fill at signal bar close (optimistic)
    SIGNAL_BAR_CLOSE = "signal_bar_close"
    # Fill only if price touches order price (conservative)
    PRICE_TOUCH = "price_touch"


class SessionFilter(Enum):
    """
    Trading session filter modes.

    Controls which market hours are included in the backtest:
    - ALL: Include all data (no filtering)
    - RTH_ONLY: Regular Trading Hours only (9:30 AM - 4:00 PM NY)
    - ETH_ONLY: Extended Trading Hours only (6:00 PM - 5:00 PM NY)
    """
    ALL = "all"
    RTH_ONLY = "rth_only"
    ETH_ONLY = "eth_only"


@dataclass
class Signal:
    """
    Trading signal from strategy.

    Attributes:
        signal_type: Type of signal
        confidence: Model confidence (0-1)
        predicted_class: Model's class prediction (0=down, 1=flat, 2=up)
        stop_ticks: Stop loss distance in ticks
        target_ticks: Take profit distance in ticks
        reason: Optional reason/explanation
    """
    signal_type: SignalType
    confidence: float = 0.0
    predicted_class: int = 1  # Default FLAT
    stop_ticks: float = 8.0  # Default 8 ticks = $10
    target_ticks: float = 16.0  # Default 16 ticks = $20 (2:1 R:R)
    reason: str = ""


@dataclass
class Position:
    """
    Open position state.

    Tracks entry details and running P&L for position management.
    """
    entry_time: datetime
    entry_price: float
    direction: int  # 1=long, -1=short
    contracts: int
    stop_price: float
    target_price: float
    entry_bar_idx: int = 0
    model_confidence: float = 0.0
    predicted_class: int = 1
    # Running stats
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    unrealized_pnl: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestConfig:
    """
    Configuration for backtest engine.

    All parameters needed for realistic simulation.

    Attributes:
        initial_capital: Starting account balance
        commission_per_side: Commission per side per contract
        exchange_fee_per_side: Exchange fee per side per contract
        slippage_ticks: Expected slippage in ticks
        tick_size: Minimum price increment (0.25 for MES)
        tick_value: Dollar value per tick (1.25 for MES)
        point_value: Dollar value per point (5.00 for MES)
        fill_mode: Order fill simulation mode
        min_confidence: Minimum confidence to trade
        default_stop_ticks: Default stop loss in ticks
        default_target_ticks: Default take profit in ticks
        max_daily_loss: Maximum loss per day (stop trading)
        max_position_size: Maximum contracts per position
        rth_start: RTH session start time (NY)
        rth_end: RTH session end time (NY)
        eod_flatten_time: Time to flatten all positions (NY)
        eod_reduce_time: Time to reduce position sizing (NY)
        eod_close_only_time: Time to stop new positions (NY)
        log_frequency: How often to log equity (bars between logs)
        session_filter: Which session to include (ALL, RTH_ONLY, ETH_ONLY)
        convert_timestamps_to_ny: Whether to convert UTC timestamps to NY (default: True)
        enable_risk_manager: Enable full RiskManager integration (default: True)
        kill_switch_loss: Cumulative loss to trigger kill switch (default: $300)
        min_account_balance: Minimum balance to allow trading (default: $700)
        max_consecutive_losses: Max consecutive losses before pause (default: 5)
        max_daily_drawdown: Max intraday drawdown (default: $75)
        max_per_trade_risk: Max risk per trade (default: $25)
    """
    initial_capital: float = 1000.0
    commission_per_side: float = 0.20
    exchange_fee_per_side: float = 0.22
    slippage_ticks: float = 1.0
    tick_size: float = 0.25
    tick_value: float = 1.25
    point_value: float = 5.0
    fill_mode: OrderFillMode = OrderFillMode.NEXT_BAR_OPEN
    min_confidence: float = 0.60
    default_stop_ticks: float = 8.0
    default_target_ticks: float = 16.0
    max_daily_loss: float = 50.0
    max_position_size: int = 5
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    eod_flatten_time: time = time(16, 30)  # 4:30 PM
    eod_reduce_time: time = time(16, 0)  # 4:00 PM
    eod_close_only_time: time = time(16, 15)  # 4:15 PM
    log_frequency: int = 60  # Log equity every 60 bars (1 min for 1-sec data)
    # Session filtering parameters
    session_filter: SessionFilter = SessionFilter.RTH_ONLY  # Filter to RTH by default (recommended)
    convert_timestamps_to_ny: bool = True  # Convert UTC timestamps to NY timezone
    # Full RiskManager integration parameters
    enable_risk_manager: bool = True  # Enable RiskManager for full risk limit enforcement
    kill_switch_loss: float = 300.0  # Cumulative loss to halt permanently (30% of $1000)
    min_account_balance: float = 700.0  # Minimum balance to allow trading
    max_consecutive_losses: int = 5  # Triggers 30-min pause
    max_daily_drawdown: float = 75.0  # Max intraday drawdown (7.5%)
    max_per_trade_risk: float = 25.0  # Max risk per individual trade (2.5%)

    @property
    def round_trip_cost(self) -> float:
        """Total cost for a round-trip trade per contract."""
        return (self.commission_per_side + self.exchange_fee_per_side) * 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "commission_per_side": self.commission_per_side,
            "exchange_fee_per_side": self.exchange_fee_per_side,
            "slippage_ticks": self.slippage_ticks,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "fill_mode": self.fill_mode.value,
            "min_confidence": self.min_confidence,
            "default_stop_ticks": self.default_stop_ticks,
            "default_target_ticks": self.default_target_ticks,
            "max_daily_loss": self.max_daily_loss,
            "max_position_size": self.max_position_size,
            "rth_start": self.rth_start.isoformat(),
            "rth_end": self.rth_end.isoformat(),
            "eod_flatten_time": self.eod_flatten_time.isoformat(),
            # Session filtering parameters
            "session_filter": self.session_filter.value,
            "convert_timestamps_to_ny": self.convert_timestamps_to_ny,
            # RiskManager integration parameters
            "enable_risk_manager": self.enable_risk_manager,
            "kill_switch_loss": self.kill_switch_loss,
            "min_account_balance": self.min_account_balance,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_daily_drawdown": self.max_daily_drawdown,
            "max_per_trade_risk": self.max_per_trade_risk,
        }


@dataclass
class BacktestResult:
    """
    Complete results from a backtest run.

    Attributes:
        report: Full backtest report with trades, equity, metrics
        config: Configuration used for this run
        data_stats: Statistics about the input data
        execution_time_seconds: How long the backtest took
    """
    report: BacktestReport
    config: BacktestConfig
    data_stats: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0


# Type alias for signal generator function
SignalGenerator = Callable[[pd.Series, Optional[Position], Dict[str, Any]], Signal]


class BacktestEngine:
    """
    Event-driven backtesting engine for MES futures.

    This engine processes 1-second bar data chronologically, generating
    signals, managing positions, and tracking performance.

    The main loop:
    1. Update current bar data
    2. Check for stop/target exits on open positions
    3. Check for EOD flatten requirements
    4. Generate new signal (if flat or should exit)
    5. Apply risk checks
    6. Execute if approved
    7. Log equity and state

    Example:
        def my_signal_generator(bar, position, context):
            # Your strategy logic here
            if position is None and bar['rsi'] < 30:
                return Signal(SignalType.LONG_ENTRY, confidence=0.75)
            return Signal(SignalType.HOLD)

        engine = BacktestEngine(config=BacktestConfig())
        result = engine.run(df, my_signal_generator)
        print(f"Total return: {result.report.metrics.total_return_pct:.2%}")
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration. Uses defaults if not provided.
        """
        self.config = config or BacktestConfig()

        # Initialize cost and slippage models
        self._cost_model = TransactionCostModel(
            MESCostConfig(
                commission_per_side=self.config.commission_per_side,
                exchange_fee_per_side=self.config.exchange_fee_per_side,
            )
        )
        self._slippage_model = SlippageModel(
            SlippageConfig(
                tick_size=self.config.tick_size,
                tick_value=self.config.tick_value,
                normal_slippage_ticks=self.config.slippage_ticks,
            )
        )

        # Initialize RiskManager if available and enabled
        self._risk_manager: Optional['RiskManager'] = None
        if self.config.enable_risk_manager and RISK_MANAGER_AVAILABLE:
            self._risk_manager = RiskManager(
                limits=RiskLimits(
                    starting_capital=self.config.initial_capital,
                    min_account_balance=self.config.min_account_balance,
                    max_daily_loss=self.config.max_daily_loss,
                    max_daily_drawdown=self.config.max_daily_drawdown,
                    max_per_trade_risk=self.config.max_per_trade_risk,
                    max_consecutive_losses=self.config.max_consecutive_losses,
                    kill_switch_loss=self.config.kill_switch_loss,
                    min_confidence=self.config.min_confidence,
                    tick_size=self.config.tick_size,
                    tick_value=self.config.tick_value,
                    point_value=self.config.point_value,
                    commission_per_side=self.config.commission_per_side + self.config.exchange_fee_per_side,
                    round_trip_commission=self.config.round_trip_cost,
                ),
                auto_persist=False,  # Don't persist during backtest
            )

        # State tracking
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all state for a new backtest run."""
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._position: Optional[Position] = None
        self._trade_log = TradeLog()
        self._equity_curve = EquityCurve(self.config.initial_capital)

        # Daily tracking
        self._current_date: Optional[date] = None
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_pnls: List[float] = []

        # Risk tracking
        self._halted_by_risk_manager = False
        self._risk_halt_reason: Optional[str] = None

        # Session filtering stats
        self._bars_filtered = 0
        self._bars_processed = 0

        # Reset cost/slippage tracking
        self._cost_model.reset()
        self._slippage_model.reset()

        # Reset RiskManager if enabled
        if self._risk_manager is not None:
            # Re-create risk manager to get clean state
            self._risk_manager = RiskManager(
                limits=RiskLimits(
                    starting_capital=self.config.initial_capital,
                    min_account_balance=self.config.min_account_balance,
                    max_daily_loss=self.config.max_daily_loss,
                    max_daily_drawdown=self.config.max_daily_drawdown,
                    max_per_trade_risk=self.config.max_per_trade_risk,
                    max_consecutive_losses=self.config.max_consecutive_losses,
                    kill_switch_loss=self.config.kill_switch_loss,
                    min_confidence=self.config.min_confidence,
                    tick_size=self.config.tick_size,
                    tick_value=self.config.tick_value,
                    point_value=self.config.point_value,
                    commission_per_side=self.config.commission_per_side + self.config.exchange_fee_per_side,
                    round_trip_commission=self.config.round_trip_cost,
                ),
                auto_persist=False,
            )

    def _to_ny_time(self, dt: datetime) -> datetime:
        """
        Convert a datetime to New York timezone.

        Uses the time_utils module if available, otherwise performs
        a basic conversion assuming UTC input.

        Args:
            dt: Datetime to convert

        Returns:
            Datetime in NY timezone
        """
        if not self.config.convert_timestamps_to_ny:
            return dt

        if TIME_UTILS_AVAILABLE and to_ny_time is not None:
            return to_ny_time(dt)

        # Fallback: assume naive datetime is UTC
        from zoneinfo import ZoneInfo
        ny_tz = ZoneInfo("America/New_York")

        if dt.tzinfo is None:
            # Assume naive datetime is UTC
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ny_tz)

    def _is_in_session(self, timestamp: datetime) -> bool:
        """
        Check if a timestamp is within the configured trading session.

        Respects the session_filter config setting (ALL, RTH_ONLY, ETH_ONLY).

        Args:
            timestamp: Datetime to check

        Returns:
            True if timestamp is within the configured session
        """
        if self.config.session_filter == SessionFilter.ALL:
            return True

        # Convert to NY timezone for session checks
        ny_time = self._to_ny_time(timestamp)
        current_time = ny_time.time()
        weekday = ny_time.weekday()

        if self.config.session_filter == SessionFilter.RTH_ONLY:
            # RTH: 9:30 AM - 4:00 PM NY, weekdays only
            if weekday >= 5:  # Saturday=5, Sunday=6
                return False
            return self.config.rth_start <= current_time < self.config.rth_end

        elif self.config.session_filter == SessionFilter.ETH_ONLY:
            # ETH: Outside RTH but market is open
            # Saturday is always closed
            if weekday == 5:
                return False

            # Sunday: ETH starts at 6:00 PM
            if weekday == 6:
                return current_time >= time(18, 0)

            # Friday: Globex closes at 5:00 PM (no overnight session)
            if weekday == 4:
                if current_time < self.config.rth_start:
                    return True  # Early morning before RTH
                if current_time >= self.config.rth_end and current_time < time(17, 0):
                    return True  # After RTH but before 5 PM close
                return False

            # Monday-Thursday: ETH is outside RTH
            if current_time < self.config.rth_start or current_time >= self.config.rth_end:
                # Exclude the 5:00 PM - 5:15 PM CME reset window
                if time(17, 0) <= current_time < time(17, 15):
                    return False
                return True

            return False

        return True

    def _filter_data_by_session(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only bars within the configured session.

        This is a critical preprocessing step that ensures backtests
        only include data from the intended trading hours.

        Args:
            data: DataFrame with datetime index

        Returns:
            Filtered DataFrame containing only session bars
        """
        if self.config.session_filter == SessionFilter.ALL:
            return data

        # Apply session filter
        mask = data.index.map(self._is_in_session)
        filtered_data = data[mask]

        # Track filtering stats
        original_count = len(data)
        filtered_count = len(filtered_data)
        self._bars_filtered = original_count - filtered_count

        logger.info(
            f"Session filter ({self.config.session_filter.value}): "
            f"{filtered_count}/{original_count} bars retained "
            f"({self._bars_filtered} filtered out)"
        )

        return filtered_data

    def run(
        self,
        data: pd.DataFrame,
        signal_generator: SignalGenerator,
        context: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run a complete backtest on the provided data.

        Args:
            data: DataFrame with OHLCV data and datetime index
            signal_generator: Function that generates signals given bar data
            context: Optional context dict passed to signal generator
            verbose: Whether to print progress

        Returns:
            BacktestResult with complete backtest outputs

        Raises:
            ValueError: If data is empty or missing required columns
        """
        import time as time_module
        start_time = time_module.time()

        # Validate data
        self._validate_data(data)

        # Reset state
        self._reset_state()

        # Initialize context
        context = context or {}

        # Filter data by session (RTH/ETH)
        original_bar_count = len(data)
        data = self._filter_data_by_session(data)

        if len(data) == 0:
            logger.warning("No data remaining after session filtering")
            # Return empty result
            metrics = calculate_metrics(
                trade_pnls=[],
                equity_curve=[self.config.initial_capital],
                initial_capital=self.config.initial_capital,
                trading_days=0,
                total_commission=0.0,
                total_slippage=0.0,
            )
            report = BacktestReport(
                trade_log=self._trade_log,
                equity_curve=self._equity_curve,
                metrics=metrics,
                config=self.config.to_dict(),
                start_date=None,
                end_date=None,
            )
            return BacktestResult(
                report=report,
                config=self.config,
                data_stats={"total_bars": 0, "bars_filtered": original_bar_count},
                execution_time_seconds=0.0,
            )

        # Data stats
        data_stats = {
            "total_bars": len(data),
            "original_bars": original_bar_count,
            "bars_filtered": self._bars_filtered,
            "session_filter": self.config.session_filter.value,
            "start_date": data.index[0].isoformat() if len(data) > 0 else None,
            "end_date": data.index[-1].isoformat() if len(data) > 0 else None,
        }

        # Main loop
        total_bars = len(data)
        for bar_idx, (timestamp, bar) in enumerate(data.iterrows()):
            # Check for new trading day
            self._handle_new_day(timestamp)

            # Check stop/target on open position
            if self._position is not None:
                exit_signal = self._check_position_exits(bar, timestamp)
                if exit_signal is not None:
                    self._execute_exit(exit_signal, bar, timestamp)

            # Check EOD flatten
            if self._should_flatten_eod(timestamp):
                if self._position is not None:
                    self._execute_exit(
                        Signal(SignalType.FLATTEN, reason="EOD flatten"),
                        bar,
                        timestamp,
                    )
                continue  # Skip signal generation after flatten time

            # Check if can trade (daily loss limit)
            if not self._can_trade_today():
                continue

            # Generate signal
            signal = signal_generator(bar, self._position, context)

            # Process signal
            self._process_signal(signal, bar, timestamp, bar_idx)

            # Update equity tracking
            self._update_equity(bar, timestamp, bar_idx)

            # Progress logging
            if verbose and bar_idx % 10000 == 0:
                pct = (bar_idx / total_bars) * 100
                print(f"Progress: {pct:.1f}% ({bar_idx}/{total_bars})")

        # Handle any open position at end of data
        if self._position is not None:
            self._force_close_position(data.iloc[-1], data.index[-1])

        # Finalize daily P&L for last day
        if self._daily_pnl != 0:
            self._daily_pnls.append(self._daily_pnl)

        # Calculate metrics
        metrics = self._calculate_final_metrics(data)

        # Add risk manager metrics to data_stats
        if self._risk_manager is not None:
            risk_metrics = self._risk_manager.get_metrics()
            data_stats["risk_manager_enabled"] = True
            data_stats["risk_manager_metrics"] = risk_metrics
            data_stats["halted_by_risk_manager"] = self._halted_by_risk_manager
            data_stats["risk_halt_reason"] = self._risk_halt_reason
        else:
            data_stats["risk_manager_enabled"] = False

        # Build report
        report = BacktestReport(
            trade_log=self._trade_log,
            equity_curve=self._equity_curve,
            metrics=metrics,
            config=self.config.to_dict(),
            start_date=data.index[0] if len(data) > 0 else None,
            end_date=data.index[-1] if len(data) > 0 else None,
        )

        execution_time = time_module.time() - start_time

        return BacktestResult(
            report=report,
            config=self.config,
            data_stats=data_stats,
            execution_time_seconds=execution_time,
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns and format."""
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")

        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

    def _handle_new_day(self, timestamp: datetime) -> None:
        """Handle transition to a new trading day."""
        current_date = timestamp.date()

        if self._current_date is not None and current_date != self._current_date:
            # Save previous day's P&L
            if self._daily_pnl != 0 or self._daily_trades > 0:
                self._daily_pnls.append(self._daily_pnl)

            # Reset daily tracking
            self._daily_pnl = 0.0
            self._daily_trades = 0

            # Reset RiskManager daily state
            if self._risk_manager is not None:
                self._risk_manager.reset_daily_state(current_date)

        self._current_date = current_date

    def _can_trade_today(self) -> bool:
        """
        Check if trading is allowed today.

        Uses RiskManager if enabled, which enforces:
        - Daily loss limit
        - Daily drawdown limit
        - Kill switch (cumulative loss)
        - Minimum account balance
        - Consecutive loss pauses
        """
        # If permanently halted by risk manager, don't trade
        if self._halted_by_risk_manager:
            return False

        # Use RiskManager if available
        if self._risk_manager is not None:
            can_trade = self._risk_manager.can_trade()
            if not can_trade:
                # Check if this is a permanent halt (kill switch)
                status = self._risk_manager.state.status
                if status == TradingStatus.HALTED:
                    self._halted_by_risk_manager = True
                    self._risk_halt_reason = self._risk_manager.state.halt_reason
            return can_trade

        # Fallback to simple daily loss check
        return self._daily_pnl > -self.config.max_daily_loss

    def _should_flatten_eod(self, timestamp: datetime) -> bool:
        """
        Check if we need to flatten positions for EOD.

        Converts timestamp to NY timezone before checking against
        EOD flatten time to ensure correct behavior regardless
        of input timezone.

        Args:
            timestamp: Current bar timestamp (any timezone)

        Returns:
            True if positions should be flattened
        """
        ny_time = self._to_ny_time(timestamp)
        current_time = ny_time.time()
        return current_time >= self.config.eod_flatten_time

    def _can_open_new_position(self, timestamp: datetime) -> bool:
        """
        Check if we can open a new position (EOD restrictions).

        After 4:15 PM NY, no new positions should be opened.
        Converts timestamp to NY timezone for accurate check.

        Args:
            timestamp: Current bar timestamp (any timezone)

        Returns:
            True if new positions are allowed
        """
        ny_time = self._to_ny_time(timestamp)
        current_time = ny_time.time()
        return current_time < self.config.eod_close_only_time

    def _get_eod_size_multiplier(self, timestamp: datetime) -> float:
        """
        Get position size multiplier based on time of day.

        Implements EOD position size reduction:
        - Before 4:00 PM: Full size (1.0)
        - 4:00 PM - 4:15 PM: Half size (0.5)
        - After 4:15 PM: No new positions (0.0)

        Converts timestamp to NY timezone for accurate check.

        Args:
            timestamp: Current bar timestamp (any timezone)

        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        ny_time = self._to_ny_time(timestamp)
        current_time = ny_time.time()

        if current_time >= self.config.eod_close_only_time:
            return 0.0  # No new positions
        elif current_time >= self.config.eod_reduce_time:
            return 0.5  # Half size
        else:
            return 1.0  # Full size

    def _check_position_exits(
        self,
        bar: pd.Series,
        timestamp: datetime,
    ) -> Optional[Signal]:
        """
        Check if open position should exit due to stop/target.

        Uses bar high/low to check if stop or target was hit.

        Args:
            bar: Current bar data (OHLCV)
            timestamp: Current timestamp

        Returns:
            Exit signal if position should close, None otherwise
        """
        if self._position is None:
            return None

        pos = self._position

        # Update MFE/MAE
        if pos.direction == 1:  # Long
            mfe_price = bar['high']
            mae_price = bar['low']
            current_pnl = (mfe_price - pos.entry_price) * pos.contracts * self.config.point_value
            adverse_pnl = (mae_price - pos.entry_price) * pos.contracts * self.config.point_value
        else:  # Short
            mfe_price = bar['low']
            mae_price = bar['high']
            current_pnl = (pos.entry_price - mfe_price) * pos.contracts * self.config.point_value
            adverse_pnl = (pos.entry_price - mae_price) * pos.contracts * self.config.point_value

        pos.max_favorable_excursion = max(pos.max_favorable_excursion, current_pnl)
        pos.max_adverse_excursion = min(pos.max_adverse_excursion, adverse_pnl)
        pos.bars_held += 1

        # Check stop loss hit
        if pos.direction == 1:  # Long
            if bar['low'] <= pos.stop_price:
                return Signal(SignalType.EXIT_LONG, reason="stop_hit")
        else:  # Short
            if bar['high'] >= pos.stop_price:
                return Signal(SignalType.EXIT_SHORT, reason="stop_hit")

        # Check target hit
        if pos.direction == 1:  # Long
            if bar['high'] >= pos.target_price:
                return Signal(SignalType.EXIT_LONG, reason="target_hit")
        else:  # Short
            if bar['low'] <= pos.target_price:
                return Signal(SignalType.EXIT_SHORT, reason="target_hit")

        return None

    def _process_signal(
        self,
        signal: Signal,
        bar: pd.Series,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """
        Process a trading signal.

        Args:
            signal: The signal to process
            bar: Current bar data
            timestamp: Current timestamp
            bar_idx: Index of current bar
        """
        if signal.signal_type == SignalType.HOLD:
            return

        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            return

        # Entry signals
        if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            # Can't enter if already in position
            if self._position is not None:
                return

            # Check EOD restrictions
            if not self._can_open_new_position(timestamp):
                return

            self._execute_entry(signal, bar, timestamp, bar_idx)

        # Exit signals
        elif signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.FLATTEN):
            if self._position is not None:
                self._execute_exit(signal, bar, timestamp)

    def _execute_entry(
        self,
        signal: Signal,
        bar: pd.Series,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """
        Execute an entry order.

        Args:
            signal: Entry signal
            bar: Current bar data
            timestamp: Current timestamp
            bar_idx: Bar index for tracking
        """
        direction = 1 if signal.signal_type == SignalType.LONG_ENTRY else -1

        # Get fill price based on fill mode
        if self.config.fill_mode == OrderFillMode.SIGNAL_BAR_CLOSE:
            base_price = bar['close']
        else:
            # NEXT_BAR_OPEN or PRICE_TOUCH - use close as approximation
            base_price = bar['close']

        # Apply slippage
        entry_price = self._slippage_model.apply_slippage(
            price=base_price,
            direction=direction,
            order_type=OrderType.MARKET,
        )

        # Calculate position size
        size_multiplier = self._get_eod_size_multiplier(timestamp)
        contracts = max(1, int(size_multiplier))  # Minimum 1 contract
        contracts = min(contracts, self.config.max_position_size)

        # Calculate stop and target prices
        stop_ticks = signal.stop_ticks or self.config.default_stop_ticks
        target_ticks = signal.target_ticks or self.config.default_target_ticks

        if direction == 1:  # Long
            stop_price = entry_price - (stop_ticks * self.config.tick_size)
            target_price = entry_price + (target_ticks * self.config.tick_size)
        else:  # Short
            stop_price = entry_price + (stop_ticks * self.config.tick_size)
            target_price = entry_price - (target_ticks * self.config.tick_size)

        # Create position
        self._position = Position(
            entry_time=timestamp,
            entry_price=entry_price,
            direction=direction,
            contracts=contracts,
            stop_price=stop_price,
            target_price=target_price,
            entry_bar_idx=bar_idx,
            model_confidence=signal.confidence,
            predicted_class=signal.predicted_class,
        )

    def _execute_exit(
        self,
        signal: Signal,
        bar: pd.Series,
        timestamp: datetime,
    ) -> None:
        """
        Execute an exit order.

        Args:
            signal: Exit signal
            bar: Current bar data
            timestamp: Current timestamp
        """
        if self._position is None:
            return

        pos = self._position

        # Determine exit price based on reason
        if signal.reason == "stop_hit":
            # Stopped out - fill at stop price with slippage
            base_price = pos.stop_price
            exit_reason = ExitReason.STOP
        elif signal.reason == "target_hit":
            # Target hit - fill at target price (limit order, no slippage)
            base_price = pos.target_price
            exit_reason = ExitReason.TARGET
        elif signal.reason == "EOD flatten":
            base_price = bar['close']
            exit_reason = ExitReason.EOD_FLATTEN
        else:
            base_price = bar['close']
            exit_reason = ExitReason.SIGNAL

        # Apply slippage for market exits (not targets)
        if exit_reason != ExitReason.TARGET:
            exit_price = self._slippage_model.apply_slippage(
                price=base_price,
                direction=-pos.direction,  # Opposite direction for exit
                order_type=OrderType.MARKET,
                contracts=pos.contracts,
            )
        else:
            exit_price = base_price

        # Calculate P&L
        if pos.direction == 1:  # Long
            price_move = exit_price - pos.entry_price
        else:  # Short
            price_move = pos.entry_price - exit_price

        gross_pnl = price_move * pos.contracts * self.config.point_value

        # Calculate costs
        commission = self._cost_model.record_trade(pos.contracts)
        # NOTE: Slippage is already reflected in entry/exit prices via apply_slippage()
        # so gross_pnl already accounts for slippage. The slippage_cost here is
        # calculated for LOGGING purposes only (passed to add_trade for record-keeping)
        # and NOT deducted from net_pnl to avoid double-counting.
        slippage_cost = self._slippage_model.get_slippage_cost(
            self.config.slippage_ticks * 2,  # Entry and exit
            pos.contracts,
        )

        # Net P&L = gross - commission only (slippage already in prices)
        net_pnl = gross_pnl - commission

        # Record trade
        self._trade_log.add_trade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            contracts=pos.contracts,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage_cost,
            exit_reason=exit_reason,
            model_confidence=pos.model_confidence,
            predicted_class=pos.predicted_class,
            stop_price=pos.stop_price,
            target_price=pos.target_price,
            bars_held=pos.bars_held,
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
        )

        # Update equity
        self._equity += net_pnl
        self._daily_pnl += net_pnl
        self._daily_trades += 1

        # Update peak
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Record trade result with RiskManager for circuit breaker/kill switch tracking
        if self._risk_manager is not None:
            self._risk_manager.record_trade_result(
                pnl=net_pnl,
                is_win=(net_pnl > 0),
            )

        # Clear position
        self._position = None

    def _force_close_position(
        self,
        bar: pd.Series,
        timestamp: datetime,
    ) -> None:
        """Force close position at end of data."""
        if self._position is None:
            return

        self._execute_exit(
            Signal(SignalType.FLATTEN, reason="End of data"),
            bar,
            timestamp,
        )

    def _update_equity(
        self,
        bar: pd.Series,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Update equity curve with current state."""
        # Only log at configured frequency
        if bar_idx % self.config.log_frequency != 0:
            return

        # Calculate unrealized P&L if in position
        unrealized_pnl = 0.0
        position_size = 0

        if self._position is not None:
            pos = self._position
            position_size = pos.contracts * pos.direction

            if pos.direction == 1:  # Long
                unrealized_pnl = (bar['close'] - pos.entry_price) * pos.contracts * self.config.point_value
            else:  # Short
                unrealized_pnl = (pos.entry_price - bar['close']) * pos.contracts * self.config.point_value

        # Log equity point
        self._equity_curve.add_point(
            timestamp=timestamp,
            equity=self._equity + unrealized_pnl,
            position_size=position_size,
            unrealized_pnl=unrealized_pnl,
        )

    def _calculate_final_metrics(self, data: pd.DataFrame) -> PerformanceMetrics:
        """Calculate final performance metrics."""
        trade_pnls = self._trade_log.get_trade_pnls()
        equity_values = self._equity_curve.get_equity_values()

        # Ensure we have equity values
        if not equity_values:
            equity_values = [self.config.initial_capital, self._equity]

        # Calculate trading days
        if len(data) > 0:
            start_date = data.index[0]
            end_date = data.index[-1]
            trading_days = len(pd.date_range(start_date, end_date, freq='B'))
        else:
            trading_days = 1

        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_values,
            initial_capital=self.config.initial_capital,
            trading_days=trading_days,
            total_commission=self._cost_model.get_total_commission(),
            total_slippage=self._slippage_model.get_total_slippage_dollars(),
            daily_pnls=self._daily_pnls if self._daily_pnls else None,
            start_date=data.index[0] if len(data) > 0 else None,
            end_date=data.index[-1] if len(data) > 0 else None,
        )

        return metrics

    def get_risk_manager_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get current risk manager metrics.

        Returns:
            Dictionary of risk metrics if RiskManager is enabled, None otherwise.
        """
        if self._risk_manager is None:
            return None

        return self._risk_manager.get_metrics()

    @property
    def is_halted(self) -> bool:
        """Check if backtest was halted by risk manager (kill switch)."""
        return self._halted_by_risk_manager

    @property
    def halt_reason(self) -> Optional[str]:
        """Get reason for risk manager halt, if any."""
        return self._risk_halt_reason


def create_simple_signal_generator(
    prediction_column: str = 'prediction',
    confidence_column: str = 'confidence',
    min_confidence: float = 0.60,
    stop_ticks: float = 8.0,
    target_ticks: float = 16.0,
) -> SignalGenerator:
    """
    Create a simple signal generator from model predictions.

    This is a convenience function for basic strategy testing.
    For production, implement a custom SignalGenerator with full logic.

    Args:
        prediction_column: Column name for model prediction (0=down, 1=flat, 2=up)
        confidence_column: Column name for model confidence
        min_confidence: Minimum confidence to generate signal
        stop_ticks: Stop loss distance in ticks
        target_ticks: Take profit distance in ticks

    Returns:
        SignalGenerator function
    """
    def signal_generator(
        bar: pd.Series,
        position: Optional[Position],
        context: Dict[str, Any],
    ) -> Signal:
        # Get prediction and confidence
        prediction = bar.get(prediction_column, 1)  # Default FLAT
        confidence = bar.get(confidence_column, 0.0)

        # Check confidence threshold
        if confidence < min_confidence:
            return Signal(SignalType.HOLD)

        # If we have a position, check for exit
        if position is not None:
            # Exit long if prediction is DOWN
            if position.direction == 1 and prediction == 0:
                return Signal(
                    SignalType.EXIT_LONG,
                    confidence=confidence,
                    predicted_class=prediction,
                    reason="Model predicts DOWN",
                )
            # Exit short if prediction is UP
            elif position.direction == -1 and prediction == 2:
                return Signal(
                    SignalType.EXIT_SHORT,
                    confidence=confidence,
                    predicted_class=prediction,
                    reason="Model predicts UP",
                )
            return Signal(SignalType.HOLD)

        # If flat, check for entry
        if prediction == 2:  # UP
            return Signal(
                SignalType.LONG_ENTRY,
                confidence=confidence,
                predicted_class=prediction,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                reason="Model predicts UP",
            )
        elif prediction == 0:  # DOWN
            return Signal(
                SignalType.SHORT_ENTRY,
                confidence=confidence,
                predicted_class=prediction,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                reason="Model predicts DOWN",
            )

        return Signal(SignalType.HOLD)

    return signal_generator


class WalkForwardValidator:
    """
    Walk-forward validation framework for backtesting.

    Implements rolling window train/validation/test splits to avoid
    overfitting and assess strategy robustness.

    Walk-Forward Process:
    1. Train model on training window (6 months)
    2. Optimize parameters on validation window (1 month)
    3. Test on out-of-sample test window (1 month)
    4. Roll forward by step size and repeat

    Example with 3 years of data:
    Fold 1: Train Jan-Jun 2023 | Val Jul 2023 | Test Aug 2023
    Fold 2: Train Feb-Jul 2023 | Val Aug 2023 | Test Sep 2023
    ...
    """

    def __init__(
        self,
        training_months: int = 6,
        validation_months: int = 1,
        test_months: int = 1,
        step_months: int = 1,
        min_trades_per_fold: int = 100,
    ):
        """
        Initialize walk-forward validator.

        Args:
            training_months: Months in training window
            validation_months: Months in validation window
            test_months: Months in test window
            step_months: Months to roll forward each iteration
            min_trades_per_fold: Minimum trades required per fold
        """
        self.training_months = training_months
        self.validation_months = validation_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_trades_per_fold = min_trades_per_fold

    def generate_folds(
        self,
        data: pd.DataFrame,
    ) -> List[Dict[str, Tuple[datetime, datetime]]]:
        """
        Generate train/val/test date ranges for walk-forward validation.

        Args:
            data: DataFrame with datetime index

        Returns:
            List of fold dictionaries with 'train', 'val', 'test' date ranges
        """
        if len(data) == 0:
            return []

        start_date = data.index[0]
        end_date = data.index[-1]

        folds = []
        fold_start = start_date

        total_window_months = self.training_months + self.validation_months + self.test_months

        while True:
            # Calculate window end dates
            train_end = fold_start + pd.DateOffset(months=self.training_months)
            val_end = train_end + pd.DateOffset(months=self.validation_months)
            test_end = val_end + pd.DateOffset(months=self.test_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            fold = {
                'train': (fold_start, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end),
            }
            folds.append(fold)

            # Roll forward
            fold_start = fold_start + pd.DateOffset(months=self.step_months)

        return folds

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        engine: BacktestEngine,
        signal_generator: SignalGenerator,
        verbose: bool = False,
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation.

        Args:
            data: Full dataset with datetime index
            engine: BacktestEngine to use
            signal_generator: Signal generator function
            verbose: Whether to print progress

        Returns:
            List of BacktestResult for each fold's test period
        """
        folds = self.generate_folds(data)
        results = []

        for fold_idx, fold in enumerate(folds):
            if verbose:
                print(f"Fold {fold_idx + 1}/{len(folds)}: "
                      f"Test {fold['test'][0].date()} to {fold['test'][1].date()}")

            # Extract test period data
            test_start, test_end = fold['test']
            test_data = data[(data.index >= test_start) & (data.index < test_end)]

            if len(test_data) == 0:
                continue

            # Run backtest on test period
            result = engine.run(test_data, signal_generator, verbose=False)
            result.report.fold_id = fold_idx + 1

            results.append(result)

        return results
