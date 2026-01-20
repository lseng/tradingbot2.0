"""
Simplified Backtest Engine for 5-Minute Scalping System

This module provides a bar-by-bar backtesting engine optimized for the 5-minute
scalping strategy. It is simpler than the main BacktestEngine in src/backtest/
because the 5-minute system has fixed rules:

Trading Rules (from spec):
- Entry: Model confidence >= 60% for direction
- Exit: Profit target (6 ticks), Stop loss (8 ticks), or Time stop (30 min)
- No new positions after 3:45 PM
- Flatten by 3:55 PM
- Max position: 1 contract

Execution Model:
- Entry at next bar open + 1 tick slippage
- Commission: $0.84 round-trip

Why a separate backtest engine:
1. The main engine is designed for 1-second data with complex features
2. This engine is optimized for 5-minute bars with simple rules
3. Simpler code = fewer bugs = more trustworthy results
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Reuse existing components
from src.backtest.metrics import PerformanceMetrics, calculate_metrics
from src.backtest.slippage import SlippageModel, SlippageConfig, OrderType, MarketCondition
from src.backtest.costs import TransactionCostModel, MESCostConfig

logger = logging.getLogger(__name__)

# Constants
NY_TZ = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)
NO_NEW_ENTRIES_TIME = time(15, 45)  # 3:45 PM - no new positions after this
FLATTEN_TIME = time(15, 55)  # 3:55 PM - flatten all positions
TICK_SIZE = 0.25
TICK_VALUE = 1.25
POINT_VALUE = 5.0  # $5 per point for MES


class ExitReason(Enum):
    """Reason why a position was exited."""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    EOD_FLATTEN = "eod_flatten"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class BacktestConfig:
    """
    Configuration for the scalping backtest.

    All values have sensible defaults from the spec. Customize only if needed.
    """
    # Capital
    initial_capital: float = 1000.0

    # Entry/Exit Parameters (in ticks)
    profit_target_ticks: int = 6  # 6 ticks = $7.50 profit
    stop_loss_ticks: int = 8  # 8 ticks = $10.00 loss
    time_stop_bars: int = 6  # 6 bars = 30 minutes on 5M

    # Confidence threshold
    min_confidence: float = 0.60

    # Position sizing
    max_position: int = 1  # Max 1 contract

    # Costs (MES defaults)
    commission_per_side: float = 0.42  # $0.42 per side ($0.20 broker + $0.22 exchange)
    slippage_ticks: float = 1.0  # 1 tick slippage per fill

    # Time constraints
    no_new_entries_time: time = field(default_factory=lambda: NO_NEW_ENTRIES_TIME)
    flatten_time: time = field(default_factory=lambda: FLATTEN_TIME)

    # Daily risk limit
    max_daily_loss: float = 100.0  # $100 max loss per day

    # Logging
    verbose: bool = True


@dataclass
class Trade:
    """Record of a single completed trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int  # 1 = LONG, -1 = SHORT
    contracts: int
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    exit_reason: ExitReason
    confidence: float
    bars_held: int
    mfe: float  # Max Favorable Excursion (best unrealized profit)
    mae: float  # Max Adverse Excursion (worst unrealized loss)

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "contracts": self.contracts,
            "gross_pnl": round(self.gross_pnl, 2),
            "commission": round(self.commission, 2),
            "slippage": round(self.slippage, 2),
            "net_pnl": round(self.net_pnl, 2),
            "exit_reason": self.exit_reason.value,
            "confidence": round(self.confidence, 4),
            "bars_held": self.bars_held,
            "mfe": round(self.mfe, 2),
            "mae": round(self.mae, 2),
        }


@dataclass
class Position:
    """An open position being tracked."""
    entry_time: datetime
    entry_price: float
    direction: int
    contracts: int
    confidence: float
    stop_price: float
    target_price: float
    entry_bar_idx: int
    mfe: float = 0.0
    mae: float = 0.0


@dataclass
class BacktestResult:
    """Result of running a backtest."""
    trades: List[Trade]
    equity_curve: List[float]
    daily_pnls: Dict[str, float]  # date string -> daily PnL
    metrics: PerformanceMetrics
    config: BacktestConfig

    # Summary stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Exit reason breakdown
    exits_by_reason: Dict[str, int] = field(default_factory=dict)

    # By confidence tier
    trades_by_confidence: Dict[str, List[Trade]] = field(default_factory=dict)

    # By time of day
    trades_by_hour: Dict[int, List[Trade]] = field(default_factory=dict)


class ScalpingBacktest:
    """
    Simplified backtest engine for 5-minute scalping strategy.

    This engine processes bar-by-bar data and applies the trading rules
    from the 5M_SCALPING_SYSTEM spec. It tracks positions, applies
    slippage and commission, and calculates performance metrics.

    Example usage:
        model = ScalpingModel.load("models/5m_scalper")
        backtest = ScalpingBacktest()
        result = backtest.run(df, model)
        print(result.metrics.to_dict())
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()

        # Initialize cost models
        self.slippage_model = SlippageModel(SlippageConfig(
            normal_slippage_ticks=self.config.slippage_ticks,
        ))
        self.cost_model = TransactionCostModel(MESCostConfig(
            commission_per_side=self.config.commission_per_side / 2,  # Split to per-side
            exchange_fee_per_side=self.config.commission_per_side / 2,
        ))

        # State
        self.position: Optional[Position] = None
        self.equity = self.config.initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_pnls: Dict[str, float] = {}
        self.current_day_pnl = 0.0
        self.current_day: Optional[datetime] = None

    def run(
        self,
        df: pd.DataFrame,
        model: "ScalpingModel",  # Forward reference to avoid circular import
        feature_cols: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Run backtest on the provided data.

        Args:
            df: DataFrame with OHLCV data and features (DatetimeIndex in NY timezone)
            model: Trained ScalpingModel for signal generation
            feature_cols: List of feature column names (auto-detected if None)

        Returns:
            BacktestResult with trades, metrics, and analysis
        """
        logger.info(f"Starting backtest on {len(df):,} bars")

        # Validate input
        if df.empty:
            raise ValueError("DataFrame is empty")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Auto-detect feature columns if not provided
        if feature_cols is None:
            feature_cols = self._detect_feature_cols(df)

        # Verify all feature columns exist
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Reset state
        self._reset()

        # Extract feature matrix for batch prediction
        X = df[feature_cols].values

        # Get model signals for all bars at once (efficient)
        signals, confidences, should_trade = model.get_trading_signals(
            X, min_confidence=self.config.min_confidence
        )

        # Process bar by bar
        for i in range(len(df)):
            bar_time = df.index[i]
            bar = df.iloc[i]
            signal = signals[i]
            confidence = confidences[i]
            trade_flag = should_trade[i]

            self._process_bar(
                bar_idx=i,
                bar_time=bar_time,
                bar=bar,
                signal=signal,
                confidence=confidence,
                should_trade=trade_flag,
            )

            # Record equity
            self.equity_curve.append(self.equity)

        # Close any remaining position at end
        if self.position is not None:
            self._close_position(
                exit_time=df.index[-1],
                exit_price=df.iloc[-1]["close"],
                exit_reason=ExitReason.EOD_FLATTEN,
                bar_idx=len(df) - 1,
            )

        # Finalize daily P&L for last day
        if self.current_day is not None:
            day_str = self.current_day.strftime("%Y-%m-%d")
            self.daily_pnls[day_str] = self.current_day_pnl

        # Calculate metrics
        result = self._build_result(df)

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Win rate: {result.metrics.win_rate_pct:.1f}%, "
                   f"Net PnL: ${result.metrics.total_return_dollars:.2f}")

        return result

    def _reset(self) -> None:
        """Reset backtest state."""
        self.position = None
        self.equity = self.config.initial_capital
        self.trades = []
        self.equity_curve = []
        self.daily_pnls = {}
        self.current_day_pnl = 0.0
        self.current_day = None
        self.slippage_model.reset()
        self.cost_model.reset()

    def _detect_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns from DataFrame."""
        # Exclude OHLCV and target columns
        exclude = {"open", "high", "low", "close", "volume",
                   "target", "target_3bar", "target_6bar", "target_12bar"}
        return [c for c in df.columns if c.lower() not in exclude]

    def _process_bar(
        self,
        bar_idx: int,
        bar_time: datetime,
        bar: pd.Series,
        signal: int,
        confidence: float,
        should_trade: bool,
    ) -> None:
        """
        Process a single bar.

        This is the core logic that checks for:
        1. Day change (reset daily P&L)
        2. Existing position exits (target, stop, time, EOD)
        3. New entry signals
        """
        # Check for day change
        bar_date = bar_time.date()
        if self.current_day is None:
            self.current_day = bar_date
        elif bar_date != self.current_day:
            # Save previous day's P&L
            day_str = self.current_day.strftime("%Y-%m-%d")
            self.daily_pnls[day_str] = self.current_day_pnl
            self.current_day = bar_date
            self.current_day_pnl = 0.0

        # Get current time for EOD checks
        bar_local_time = bar_time.time() if hasattr(bar_time, 'time') else time(0, 0)

        # Check existing position for exits
        if self.position is not None:
            should_exit, exit_reason, exit_price = self._check_exits(
                bar_idx=bar_idx,
                bar_time=bar_time,
                bar=bar,
            )

            if should_exit:
                self._close_position(
                    exit_time=bar_time,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    bar_idx=bar_idx,
                )

        # Check for new entry (only if flat and conditions met)
        if self.position is None and should_trade and signal != 0:
            # Time restriction: no new entries after 3:45 PM
            if bar_local_time >= self.config.no_new_entries_time:
                return

            # Daily loss limit check
            if self.current_day_pnl <= -self.config.max_daily_loss:
                return

            # Enter position
            self._open_position(
                bar_idx=bar_idx,
                bar_time=bar_time,
                bar=bar,
                direction=signal,
                confidence=confidence,
            )

    def _check_exits(
        self,
        bar_idx: int,
        bar_time: datetime,
        bar: pd.Series,
    ) -> Tuple[bool, Optional[ExitReason], float]:
        """
        Check if current position should be exited.

        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        if self.position is None:
            return False, None, 0.0

        # Get current time for EOD check
        bar_local_time = bar_time.time() if hasattr(bar_time, 'time') else time(0, 0)

        # 1. EOD flatten check (highest priority)
        if bar_local_time >= self.config.flatten_time:
            return True, ExitReason.EOD_FLATTEN, bar["close"]

        # 2. Stop loss check (uses low for long, high for short)
        if self.position.direction == 1:  # LONG
            # Check if low hit stop
            if bar["low"] <= self.position.stop_price:
                return True, ExitReason.STOP_LOSS, self.position.stop_price
        else:  # SHORT
            # Check if high hit stop
            if bar["high"] >= self.position.stop_price:
                return True, ExitReason.STOP_LOSS, self.position.stop_price

        # 3. Profit target check (uses high for long, low for short)
        if self.position.direction == 1:  # LONG
            # Check if high hit target
            if bar["high"] >= self.position.target_price:
                return True, ExitReason.PROFIT_TARGET, self.position.target_price
        else:  # SHORT
            # Check if low hit target
            if bar["low"] <= self.position.target_price:
                return True, ExitReason.PROFIT_TARGET, self.position.target_price

        # 4. Time stop check (30 min = 6 bars)
        bars_held = bar_idx - self.position.entry_bar_idx
        if bars_held >= self.config.time_stop_bars:
            return True, ExitReason.TIME_STOP, bar["close"]

        # Update MFE/MAE
        if self.position.direction == 1:  # LONG
            unrealized = (bar["high"] - self.position.entry_price) * POINT_VALUE
            unrealized_low = (bar["low"] - self.position.entry_price) * POINT_VALUE
        else:  # SHORT
            unrealized = (self.position.entry_price - bar["low"]) * POINT_VALUE
            unrealized_low = (self.position.entry_price - bar["high"]) * POINT_VALUE

        self.position.mfe = max(self.position.mfe, unrealized)
        self.position.mae = min(self.position.mae, unrealized_low)

        return False, None, 0.0

    def _open_position(
        self,
        bar_idx: int,
        bar_time: datetime,
        bar: pd.Series,
        direction: int,
        confidence: float,
    ) -> None:
        """Open a new position."""
        # Entry price is bar open + slippage
        raw_entry_price = bar["open"]
        entry_price = self.slippage_model.apply_slippage(
            price=raw_entry_price,
            direction=direction,
            order_type=OrderType.MARKET,
            record=True,
        )

        # Calculate stop and target prices
        if direction == 1:  # LONG
            stop_price = entry_price - (self.config.stop_loss_ticks * TICK_SIZE)
            target_price = entry_price + (self.config.profit_target_ticks * TICK_SIZE)
        else:  # SHORT
            stop_price = entry_price + (self.config.stop_loss_ticks * TICK_SIZE)
            target_price = entry_price - (self.config.profit_target_ticks * TICK_SIZE)

        self.position = Position(
            entry_time=bar_time,
            entry_price=entry_price,
            direction=direction,
            contracts=self.config.max_position,
            confidence=confidence,
            stop_price=stop_price,
            target_price=target_price,
            entry_bar_idx=bar_idx,
        )

        if self.config.verbose:
            direction_str = "LONG" if direction == 1 else "SHORT"
            logger.debug(f"{bar_time}: {direction_str} entry at {entry_price:.2f}, "
                        f"stop={stop_price:.2f}, target={target_price:.2f}, "
                        f"confidence={confidence:.2%}")

    def _close_position(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: ExitReason,
        bar_idx: int,
    ) -> None:
        """Close the current position and record the trade."""
        if self.position is None:
            return

        # Apply slippage to exit (market order)
        slipped_exit_price = self.slippage_model.apply_slippage(
            price=exit_price,
            direction=-self.position.direction,  # Opposite direction to close
            order_type=OrderType.MARKET,
            record=True,
        )

        # Calculate P&L
        if self.position.direction == 1:  # LONG
            gross_pnl = (slipped_exit_price - self.position.entry_price) * POINT_VALUE
        else:  # SHORT
            gross_pnl = (self.position.entry_price - slipped_exit_price) * POINT_VALUE

        gross_pnl *= self.position.contracts

        # Calculate costs
        commission = self.cost_model.record_trade(self.position.contracts)
        slippage_cost = self.config.slippage_ticks * TICK_VALUE * 2  # Entry + exit

        net_pnl = gross_pnl - commission

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=exit_time,
            entry_price=self.position.entry_price,
            exit_price=slipped_exit_price,
            direction=self.position.direction,
            contracts=self.position.contracts,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage_cost,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            confidence=self.position.confidence,
            bars_held=bar_idx - self.position.entry_bar_idx,
            mfe=self.position.mfe,
            mae=self.position.mae,
        )

        self.trades.append(trade)

        # Update equity and daily P&L
        self.equity += net_pnl
        self.current_day_pnl += net_pnl

        if self.config.verbose:
            direction_str = "LONG" if self.position.direction == 1 else "SHORT"
            logger.debug(f"{exit_time}: {direction_str} exit at {slipped_exit_price:.2f}, "
                        f"reason={exit_reason.value}, net_pnl=${net_pnl:.2f}")

        # Clear position
        self.position = None

    def _build_result(self, df: pd.DataFrame) -> BacktestResult:
        """Build the final backtest result."""
        # Calculate trading days
        trading_days = len(self.daily_pnls)
        if trading_days == 0:
            trading_days = 1

        # Get trade P&Ls
        trade_pnls = [t.net_pnl for t in self.trades]
        daily_pnl_list = list(self.daily_pnls.values())

        # Calculate metrics
        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=self.equity_curve,
            initial_capital=self.config.initial_capital,
            trading_days=trading_days,
            total_commission=self.cost_model.get_total_commission(),
            total_slippage=self.slippage_model.get_total_slippage_dollars(),
            daily_pnls=daily_pnl_list,
            start_date=df.index[0].to_pydatetime() if len(df) > 0 else None,
            end_date=df.index[-1].to_pydatetime() if len(df) > 0 else None,
        )

        # Build result
        result = BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_pnls=self.daily_pnls,
            metrics=metrics,
            config=self.config,
            total_trades=len(self.trades),
            winning_trades=sum(1 for t in self.trades if t.net_pnl > 0),
            losing_trades=sum(1 for t in self.trades if t.net_pnl < 0),
        )

        # Exit reason breakdown
        for reason in ExitReason:
            result.exits_by_reason[reason.value] = sum(
                1 for t in self.trades if t.exit_reason == reason
            )

        # Trades by confidence tier
        result.trades_by_confidence = {
            "high": [t for t in self.trades if t.confidence >= 0.70],
            "medium": [t for t in self.trades if 0.60 <= t.confidence < 0.70],
            "low": [t for t in self.trades if t.confidence < 0.60],
        }

        # Trades by hour
        for trade in self.trades:
            hour = trade.entry_time.hour
            if hour not in result.trades_by_hour:
                result.trades_by_hour[hour] = []
            result.trades_by_hour[hour].append(trade)

        return result


def run_backtest(
    df: pd.DataFrame,
    model: "ScalpingModel",
    config: Optional[BacktestConfig] = None,
    feature_cols: Optional[List[str]] = None,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        df: DataFrame with OHLCV and features
        model: Trained ScalpingModel
        config: Backtest configuration (optional)
        feature_cols: Feature column names (auto-detected if None)

    Returns:
        BacktestResult with trades and metrics
    """
    backtest = ScalpingBacktest(config)
    return backtest.run(df, model, feature_cols)


def analyze_results(result: BacktestResult) -> Dict:
    """
    Generate detailed analysis of backtest results.

    Args:
        result: BacktestResult from a completed backtest

    Returns:
        Dictionary with analysis by various dimensions
    """
    analysis = {
        "summary": result.metrics.to_dict(),
        "exit_reasons": result.exits_by_reason,
        "by_confidence": {},
        "by_hour": {},
        "worst_days": [],
        "best_days": [],
    }

    # Analyze by confidence tier
    for tier, trades in result.trades_by_confidence.items():
        if trades:
            wins = sum(1 for t in trades if t.net_pnl > 0)
            total_pnl = sum(t.net_pnl for t in trades)
            analysis["by_confidence"][tier] = {
                "count": len(trades),
                "win_rate": wins / len(trades) * 100 if trades else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
            }

    # Analyze by hour
    for hour, trades in sorted(result.trades_by_hour.items()):
        if trades:
            wins = sum(1 for t in trades if t.net_pnl > 0)
            total_pnl = sum(t.net_pnl for t in trades)
            analysis["by_hour"][hour] = {
                "count": len(trades),
                "win_rate": wins / len(trades) * 100 if trades else 0,
                "total_pnl": round(total_pnl, 2),
            }

    # Best and worst days
    sorted_days = sorted(result.daily_pnls.items(), key=lambda x: x[1])
    analysis["worst_days"] = sorted_days[:5]  # 5 worst days
    analysis["best_days"] = sorted_days[-5:][::-1]  # 5 best days

    return analysis


def export_trades_csv(result: BacktestResult, filepath: str) -> None:
    """Export trades to CSV file."""
    if not result.trades:
        logger.warning("No trades to export")
        return

    import csv

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.trades[0].to_dict().keys())
        writer.writeheader()
        for trade in result.trades:
            writer.writerow(trade.to_dict())

    logger.info(f"Exported {len(result.trades)} trades to {filepath}")


def export_summary_json(result: BacktestResult, filepath: str) -> None:
    """Export summary to JSON file."""
    import json

    analysis = analyze_results(result)

    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Exported summary to {filepath}")
