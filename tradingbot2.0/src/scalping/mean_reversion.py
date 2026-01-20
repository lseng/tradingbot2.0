"""
Mean-Reversion Strategy for 5-Minute Scalping System

This module implements a mean-reversion strategy that exploits the one signal that WORKS:
volatility prediction (AUC 0.855).

Key Insight: During LOW volatility periods, prices tend to mean-revert.
Instead of trying to predict breakout direction (which failed), we trade mean-reversion
during quiet periods.

Strategy Logic:
1. Volatility model predicts LOW volatility (< 40% probability of high vol)
2. RSI shows extreme reading (< 30 or > 70)
3. Price is extended from EMA (deviation > threshold)
4. Enter against the extreme, expecting reversion to mean

Why this approach:
- Volatility prediction IS proven (AUC 0.855)
- Mean-reversion in low-vol is well-documented in market microstructure
- Tighter stops reduce exposure to sudden vol spikes
- Quick exits avoid being caught in regime change
- RSI extremes + vol filter combines two uncorrelated signals

Previous Approaches That Failed:
- Direction prediction (24 features): Win rate 38.8%, PF 0.28
- Breakout detection: Win rate 39.1%, PF 0.50
- Both failed because direction is unpredictable on 5-minute bars
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scalping.features import (
    ScalpingFeatureGenerator,
    FeatureConfig,
    _calculate_atr,
    _calculate_rsi,
)

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig:
    """Configuration for mean-reversion strategy."""

    # Volatility filter parameters
    low_vol_threshold: float = 0.40  # Predict LOW vol when prob < this

    # RSI parameters
    rsi_period: int = 7  # Shorter period for faster signal
    rsi_oversold: float = 30.0  # RSI below this = oversold
    rsi_overbought: float = 70.0  # RSI above this = overbought

    # EMA deviation parameters
    ema_period: int = 21  # EMA period for deviation
    min_ema_deviation: float = 0.003  # Min 0.3% deviation from EMA

    # Trade parameters
    profit_target_ticks: float = 4.0  # Smaller target (low vol = smaller moves)
    stop_loss_ticks: float = 4.0  # Symmetric 1:1 R:R
    time_stop_bars: int = 3  # Quicker exit (15 min on 5M bars)

    # Time filters
    avoid_first_hour: bool = True  # First hour often has breakouts
    avoid_last_hour: bool = True  # Last hour often has breakouts

    # Contract parameters
    tick_size: float = 0.25
    tick_value: float = 1.25
    commission: float = 0.84


def add_mean_reversion_features(
    df: pd.DataFrame,
    config: Optional[MeanReversionConfig] = None,
) -> pd.DataFrame:
    """
    Add features specifically for mean-reversion detection.

    Args:
        df: DataFrame with OHLCV data and base features
        config: Mean-reversion configuration

    Returns:
        DataFrame with mean-reversion features added
    """
    config = config or MeanReversionConfig()
    result = df.copy()

    # RSI for overbought/oversold detection
    if f"rsi_{config.rsi_period}_raw" not in result.columns:
        result[f"rsi_{config.rsi_period}_raw"] = _calculate_rsi(
            result["close"], config.rsi_period
        )

    # EMA for deviation
    ema = result["close"].ewm(span=config.ema_period, adjust=False).mean()
    result[f"ema_{config.ema_period}"] = ema
    result["ema_deviation"] = (result["close"] - ema) / result["close"]

    # Z-score of price relative to recent mean
    rolling_mean = result["close"].rolling(window=20).mean()
    rolling_std = result["close"].rolling(window=20).std()
    result["price_zscore"] = (result["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # Distance from recent VWAP
    typical_price = (result["high"] + result["low"] + result["close"]) / 3
    cumulative_tp_vol = (typical_price * result["volume"]).rolling(window=20).sum()
    cumulative_vol = result["volume"].rolling(window=20).sum()
    rolling_vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    result["vwap_deviation_mr"] = (result["close"] - rolling_vwap) / result["close"]

    # Consecutive bars in same direction (extreme = likely to revert)
    returns = result["close"].diff()
    up_streak = returns.gt(0).groupby(
        (~returns.gt(0)).cumsum()
    ).cumsum()
    down_streak = returns.lt(0).groupby(
        (~returns.lt(0)).cumsum()
    ).cumsum()
    result["up_streak"] = up_streak.where(returns > 0, 0)
    result["down_streak"] = down_streak.where(returns < 0, 0)

    # Rate of mean reversion (how quickly price returns to mean)
    # Higher = more mean reverting environment
    deviation_from_mean = result["close"] - rolling_mean
    reversion_next = deviation_from_mean.shift(1).abs() - deviation_from_mean.abs()
    result["reversion_rate"] = reversion_next.rolling(window=20).mean()

    return result


def identify_mean_reversion_setups(
    df: pd.DataFrame,
    vol_predictions: np.ndarray,
    config: Optional[MeanReversionConfig] = None,
) -> pd.DataFrame:
    """
    Identify potential mean-reversion trading setups.

    A setup is valid when:
    1. Volatility model predicts LOW volatility
    2. RSI shows extreme reading (oversold or overbought)
    3. Price is extended from EMA
    4. Not in first/last hour (optional)

    Args:
        df: DataFrame with features
        vol_predictions: Volatility model predictions (probability of HIGH vol)
        config: Mean-reversion configuration

    Returns:
        DataFrame with setup signals added
    """
    config = config or MeanReversionConfig()
    result = df.copy()

    # Add volatility predictions
    result["vol_prediction"] = vol_predictions

    # LOW volatility filter (we want LOW vol for mean-reversion)
    low_vol = vol_predictions < config.low_vol_threshold
    result["is_low_vol"] = low_vol.astype(int)

    # RSI extreme readings
    rsi_col = f"rsi_{config.rsi_period}_raw"
    if rsi_col not in result.columns:
        result[rsi_col] = _calculate_rsi(result["close"], config.rsi_period)

    rsi = result[rsi_col]
    is_oversold = rsi < config.rsi_oversold
    is_overbought = rsi > config.rsi_overbought
    result["is_oversold"] = is_oversold.astype(int)
    result["is_overbought"] = is_overbought.astype(int)

    # EMA deviation filter
    if "ema_deviation" not in result.columns:
        ema = result["close"].ewm(span=config.ema_period, adjust=False).mean()
        result["ema_deviation"] = (result["close"] - ema) / result["close"]

    extended_below = result["ema_deviation"] < -config.min_ema_deviation
    extended_above = result["ema_deviation"] > config.min_ema_deviation
    result["extended_below_ema"] = extended_below.astype(int)
    result["extended_above_ema"] = extended_above.astype(int)

    # Time filters
    times = result.index.time
    minutes = np.array([t.hour * 60 + t.minute for t in times])

    # First hour (9:30 - 10:30)
    in_first_hour = (minutes >= 570) & (minutes < 630)
    # Last hour (3:00 - 4:00)
    in_last_hour = (minutes >= 900) & (minutes < 960)

    time_ok = np.ones(len(result), dtype=bool)
    if config.avoid_first_hour:
        time_ok = time_ok & ~in_first_hour
    if config.avoid_last_hour:
        time_ok = time_ok & ~in_last_hour

    result["time_ok"] = time_ok.astype(int)

    # Combined setup conditions
    # LONG setup: oversold + extended below EMA + low vol + time ok
    long_setup = is_oversold & extended_below & low_vol & time_ok
    result["is_long_setup"] = long_setup.astype(int)

    # SHORT setup: overbought + extended above EMA + low vol + time ok
    short_setup = is_overbought & extended_above & low_vol & time_ok
    result["is_short_setup"] = short_setup.astype(int)

    # Combined setup direction
    result["setup_direction"] = np.where(
        long_setup, 1,
        np.where(short_setup, -1, 0)
    )

    # Log statistics
    n_long = long_setup.sum()
    n_short = short_setup.sum()
    n_low_vol = low_vol.sum()

    logger.info(f"Identified {n_long + n_short} mean-reversion setups: "
                f"{n_long} long (oversold), {n_short} short (overbought)")
    logger.info(f"Low volatility periods: {n_low_vol} ({n_low_vol/len(result)*100:.1f}%)")

    return result


class MeanReversionTrader:
    """
    Trading logic for mean-reversion strategy.

    Entry rules:
    1. Wait for LOW volatility prediction
    2. Wait for RSI extreme (oversold or overbought)
    3. Verify price extended from EMA
    4. Enter against the extreme (buy oversold, sell overbought)

    Exit rules:
    1. Profit target (4 ticks)
    2. Stop loss (4 ticks)
    3. Time stop (3 bars = 15 minutes)
    4. Volatility regime change (predict HIGH vol)
    """

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        """Initialize trader with configuration."""
        self.config = config or MeanReversionConfig()

        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0.0
        self.entry_bar = 0
        self.trades: List[dict] = []

    def should_enter(
        self,
        vol_prediction: float,
        rsi: float,
        ema_deviation: float,
        is_first_hour: bool = False,
        is_last_hour: bool = False,
        bar_idx: int = 0,
    ) -> int:
        """
        Determine if we should enter a trade.

        Returns:
            1 for long (buy oversold), -1 for short (sell overbought), 0 for no trade
        """
        if self.position != 0:
            return 0

        # Must have LOW volatility prediction
        if vol_prediction >= self.config.low_vol_threshold:
            return 0

        # Time filters
        if self.config.avoid_first_hour and is_first_hour:
            return 0
        if self.config.avoid_last_hour and is_last_hour:
            return 0

        # LONG: oversold + extended below EMA
        if (rsi < self.config.rsi_oversold and
            ema_deviation < -self.config.min_ema_deviation):
            return 1

        # SHORT: overbought + extended above EMA
        if (rsi > self.config.rsi_overbought and
            ema_deviation > self.config.min_ema_deviation):
            return -1

        return 0

    def should_exit(
        self,
        current_price: float,
        bar_idx: int,
        high: float,
        low: float,
        vol_prediction: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit current position.

        Returns:
            Tuple of (should_exit, reason)
        """
        if self.position == 0:
            return False, ""

        # Calculate P&L in ticks
        if self.position == 1:
            # Long position
            worst_price = low
            worst_pnl_ticks = (worst_price - self.entry_price) / self.config.tick_size
            best_price = high
            best_pnl_ticks = (best_price - self.entry_price) / self.config.tick_size
        else:
            # Short position
            worst_price = high
            worst_pnl_ticks = (self.entry_price - worst_price) / self.config.tick_size
            best_price = low
            best_pnl_ticks = (self.entry_price - best_price) / self.config.tick_size

        # Stop loss (check worst price first)
        if worst_pnl_ticks <= -self.config.stop_loss_ticks:
            return True, "stop_loss"

        # Profit target (check best price)
        if best_pnl_ticks >= self.config.profit_target_ticks:
            return True, "profit_target"

        # Time stop
        bars_held = bar_idx - self.entry_bar
        if bars_held >= self.config.time_stop_bars:
            return True, "time_stop"

        # Volatility regime change (optional early exit)
        if vol_prediction is not None and vol_prediction >= 0.60:
            # High vol predicted - exit to avoid being caught in breakout
            return True, "vol_regime_change"

        return False, ""

    def enter_trade(
        self,
        direction: int,
        price: float,
        bar_idx: int,
        timestamp: pd.Timestamp,
    ) -> dict:
        """Enter a new trade."""
        # Apply 1 tick slippage
        if direction == 1:
            fill_price = price + self.config.tick_size
        else:
            fill_price = price - self.config.tick_size

        self.position = direction
        self.entry_price = fill_price
        self.entry_bar = bar_idx

        trade = {
            "entry_time": timestamp,
            "entry_bar": bar_idx,
            "direction": direction,
            "entry_price": fill_price,
        }

        return trade

    def exit_trade(
        self,
        price: float,
        bar_idx: int,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> dict:
        """Exit current trade."""
        # Apply 1 tick slippage
        if self.position == 1:
            fill_price = price - self.config.tick_size
        else:
            fill_price = price + self.config.tick_size

        # Calculate P&L
        if self.position == 1:
            pnl_ticks = (fill_price - self.entry_price) / self.config.tick_size
        else:
            pnl_ticks = (self.entry_price - fill_price) / self.config.tick_size

        pnl_dollars = pnl_ticks * self.config.tick_value - self.config.commission

        trade = {
            "exit_time": timestamp,
            "exit_bar": bar_idx,
            "exit_price": fill_price,
            "exit_reason": reason,
            "pnl_ticks": pnl_ticks,
            "pnl_dollars": pnl_dollars,
            "bars_held": bar_idx - self.entry_bar,
        }

        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0

        return trade

    def reset(self):
        """Reset trader state."""
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.trades = []


def create_mean_reversion_target(
    df: pd.DataFrame,
    horizon_bars: int = 3,
    min_reversion_ticks: float = 2.0,
    tick_size: float = 0.25,
) -> Tuple[pd.DataFrame, dict]:
    """
    Create target variable for mean-reversion prediction.

    The target indicates whether price reverted toward the mean within horizon.

    Args:
        df: DataFrame with OHLCV data and RSI/EMA features
        horizon_bars: Bars to look ahead for reversion
        min_reversion_ticks: Minimum move in ticks for successful reversion
        tick_size: Contract tick size

    Returns:
        Tuple of (DataFrame with target, stats dict)
    """
    result = df.copy()

    min_reversion = min_reversion_ticks * tick_size

    # Calculate 21-period EMA if not present
    if "ema_21" not in result.columns:
        result["ema_21"] = result["close"].ewm(span=21, adjust=False).mean()

    # For each bar, check if price reverted toward EMA within horizon
    reversion_success = []

    for i in range(len(df)):
        if i + horizon_bars >= len(df):
            reversion_success.append(np.nan)
            continue

        current_close = df["close"].iloc[i]
        current_ema = result["ema_21"].iloc[i]
        current_deviation = current_close - current_ema

        # Check future bars for reversion
        future_closes = df["close"].iloc[i+1:i+horizon_bars+1]

        if abs(current_deviation) < min_reversion:
            # Not extended enough, no reversion expected
            reversion_success.append(0)
            continue

        if current_deviation > 0:
            # Price above EMA, reversion = price drops
            best_drop = current_close - future_closes.min()
            success = best_drop >= min_reversion
        else:
            # Price below EMA, reversion = price rises
            best_rise = future_closes.max() - current_close
            success = best_rise >= min_reversion

        reversion_success.append(int(success))

    result["target_reversion"] = reversion_success

    # Calculate statistics
    valid = pd.Series(reversion_success).dropna()
    stats = {
        "total_samples": len(valid),
        "reversion_rate": valid.mean() * 100 if len(valid) > 0 else 0,
        "horizon_bars": horizon_bars,
        "min_reversion_ticks": min_reversion_ticks,
    }

    logger.info(f"Mean-reversion target created: {stats['reversion_rate']:.1f}% reversion rate")

    return result, stats


def run_mean_reversion_backtest(
    df: pd.DataFrame,
    vol_predictions: np.ndarray,
    config: Optional[MeanReversionConfig] = None,
) -> Tuple[List[dict], dict]:
    """
    Run backtest using mean-reversion strategy.

    Args:
        df: DataFrame with features
        vol_predictions: Volatility model predictions (probability of HIGH vol)
        config: Mean-reversion configuration

    Returns:
        Tuple of (list of trades, summary statistics)
    """
    config = config or MeanReversionConfig()

    trader = MeanReversionTrader(config)

    trades = []
    current_trade = None

    # Ensure we have required features
    rsi_col = f"rsi_{config.rsi_period}_raw"
    if rsi_col not in df.columns:
        df[rsi_col] = _calculate_rsi(df["close"], config.rsi_period)

    if "ema_deviation" not in df.columns:
        ema = df["close"].ewm(span=config.ema_period, adjust=False).mean()
        df["ema_deviation"] = (df["close"] - ema) / df["close"]

    for i, (idx, row) in enumerate(df.iterrows()):
        # Skip if no predictions
        if i >= len(vol_predictions):
            break

        vol_pred = vol_predictions[i]

        # Time flags
        t = idx.time()
        minutes = t.hour * 60 + t.minute
        is_first_hour = 570 <= minutes < 630
        is_last_hour = 900 <= minutes < 960

        # Check for exit first
        if trader.position != 0:
            should_exit, reason = trader.should_exit(
                row["close"], i, row["high"], row["low"], vol_pred
            )
            if should_exit:
                exit_info = trader.exit_trade(row["close"], i, idx, reason)
                current_trade.update(exit_info)
                trades.append(current_trade)
                current_trade = None

        # Check for entry
        if trader.position == 0:
            direction = trader.should_enter(
                vol_pred,
                row.get(rsi_col, 50),
                row.get("ema_deviation", 0),
                is_first_hour,
                is_last_hour,
                i,
            )
            if direction != 0:
                current_trade = trader.enter_trade(direction, row["open"], i, idx)

    # Close any open trade at end
    if trader.position != 0 and current_trade is not None:
        last_row = df.iloc[-1]
        exit_info = trader.exit_trade(
            last_row["close"], len(df) - 1, df.index[-1], "end_of_data"
        )
        current_trade.update(exit_info)
        trades.append(current_trade)

    # Calculate summary statistics
    if trades:
        total_pnl = sum(t.get("pnl_dollars", 0) for t in trades)
        wins = [t for t in trades if t.get("pnl_dollars", 0) > 0]
        losses = [t for t in trades if t.get("pnl_dollars", 0) <= 0]

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t["pnl_dollars"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_dollars"] for t in losses]) if losses else 0

        total_win = sum(t["pnl_dollars"] for t in wins) if wins else 0
        total_loss = abs(sum(t["pnl_dollars"] for t in losses)) if losses else 0
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        summary = {
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "long_trades": len([t for t in trades if t.get("direction") == 1]),
            "short_trades": len([t for t in trades if t.get("direction") == -1]),
            "exit_reasons": exit_reasons,
        }
    else:
        summary = {
            "total_trades": 0,
            "total_pnl": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "long_trades": 0,
            "short_trades": 0,
            "exit_reasons": {},
        }

    return trades, summary
