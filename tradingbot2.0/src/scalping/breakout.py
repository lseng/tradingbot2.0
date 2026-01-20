"""
Breakout Detection Strategy for 5-Minute Scalping System

This module implements a breakout detection approach after the direction prediction
approach failed validation (AUC 0.51). The key insight is that:

1. Volatility IS predictable (AUC 0.856)
2. High volatility often follows periods of consolidation (breakouts)
3. By combining consolidation detection with volatility prediction, we can:
   - Identify when a breakout is likely to occur
   - Trade the breakout direction based on price position in the range

Strategy Logic:
1. Detect consolidation periods (low ATR, tight BB width, small ranges)
2. Identify position within consolidation range (near top vs near bottom)
3. Use volatility prediction to time breakout entry
4. Trade in direction indicated by price position in range

Why this approach:
- Avoids direct direction prediction which showed no signal
- Leverages proven volatility prediction capability
- Consolidation breakouts are a well-documented market phenomenon
- Clear entry/exit rules reduce curve fitting risk
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scalping.features import (
    ScalpingFeatureGenerator,
    FeatureConfig,
    _calculate_atr,
)

logger = logging.getLogger(__name__)


class ConsolidationType(Enum):
    """Type of consolidation/breakout detected."""
    NONE = 0           # Not in consolidation
    CONSOLIDATION = 1  # In consolidation, no breakout signal
    BREAKOUT_UP = 2    # Breakout upward likely
    BREAKOUT_DOWN = 3  # Breakout downward likely


@dataclass
class BreakoutConfig:
    """Configuration for breakout detection."""

    # Consolidation detection parameters
    lookback_bars: int = 12  # Bars to look back for consolidation detection
    atr_percentile_threshold: float = 30.0  # Below this percentile = low volatility
    bb_width_percentile_threshold: float = 30.0  # Below this percentile = tight range
    consolidation_threshold: float = 0.60  # Min consolidation score to consider valid

    # Breakout signal parameters
    vol_prediction_threshold: float = 0.60  # Min volatility model confidence
    position_in_range_threshold: float = 0.25  # Distance from range boundary (0-0.5)

    # Target creation parameters
    horizon_bars: int = 6  # 30 minutes on 5M bars
    breakout_move_ticks: float = 4.0  # Min move in ticks for successful breakout
    tick_size: float = 0.25

    # Feature calculation
    range_lookback: int = 20  # Bars for range calculation
    squeeze_lookback: int = 10  # Bars for squeeze detection


def _calculate_range_position(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate price position within recent trading range.

    Returns value from 0 (at range low) to 1 (at range high).
    Values near 0 suggest price is near support (breakout up possible).
    Values near 1 suggest price is near resistance (breakout down possible).

    Args:
        df: DataFrame with high, low, close columns
        lookback: Number of bars to use for range calculation

    Returns:
        Series with range position values (0-1)
    """
    rolling_high = df["high"].rolling(window=lookback).max()
    rolling_low = df["low"].rolling(window=lookback).min()
    range_size = rolling_high - rolling_low

    # Position in range: (close - low) / (high - low)
    # Avoid division by zero
    position = (df["close"] - rolling_low) / range_size.replace(0, np.nan)

    return position.clip(0, 1)


def _detect_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.Series:
    """
    Detect Bollinger Band squeeze (BB inside Keltner Channel).

    A squeeze indicates low volatility/consolidation and often precedes breakouts.

    Returns:
        Series with 1 where squeeze detected, 0 otherwise
    """
    # Bollinger Bands
    sma = df["close"].rolling(window=bb_period).mean()
    std = df["close"].rolling(window=bb_period).std()
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std

    # Keltner Channel
    atr = _calculate_atr(df, kc_period)
    kc_upper = sma + kc_mult * atr
    kc_lower = sma - kc_mult * atr

    # Squeeze: BB inside KC
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)

    return squeeze


def _calculate_consolidation_score(
    df: pd.DataFrame,
    lookback: int = 12,
) -> pd.Series:
    """
    Calculate a consolidation score (0-1) based on multiple factors.

    Higher score = more consolidated/tighter range.

    Factors:
    - ATR relative to recent history (lower = more consolidated)
    - Bar ranges relative to recent history (smaller = more consolidated)
    - Price movement relative to range (lower = more consolidated)

    Args:
        df: DataFrame with OHLCV data and ATR/BB features
        lookback: Number of bars for comparison

    Returns:
        Series with consolidation scores (0-1, higher = more consolidated)
    """
    scores = []

    # Use ATR normalized by close
    atr_normalized = _calculate_atr(df, 14) / df["close"]

    # Bar range normalized
    bar_range = (df["high"] - df["low"]) / df["close"]

    # Price movement over lookback
    price_change = df["close"].diff(lookback).abs() / df["close"]

    # Calculate rolling percentiles
    atr_percentile = atr_normalized.rolling(window=lookback * 5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
        raw=True
    )

    range_percentile = bar_range.rolling(window=lookback * 5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
        raw=True
    )

    # Consolidation score: lower percentile = more consolidated
    # Invert so higher score = more consolidated
    consolidation_score = 1 - (atr_percentile * 0.4 + range_percentile * 0.4 +
                               price_change.clip(0, 0.02) / 0.02 * 0.2)

    return consolidation_score.fillna(0.5)


def _calculate_momentum_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Calculate momentum divergence to hint at breakout direction.

    Positive divergence (price making lower lows but momentum making higher lows)
    suggests upward breakout. Negative divergence suggests downward breakout.

    Returns:
        Series with divergence signal (-1 to 1)
    """
    # Simple momentum: rate of change
    momentum = df["close"].diff(lookback)

    # Rolling correlation between price and momentum
    price_roc = df["close"].pct_change(lookback)

    # If price is down but momentum is improving, bullish divergence
    # If price is up but momentum is weakening, bearish divergence
    price_direction = np.sign(price_roc)
    momentum_change = momentum.diff(lookback // 2)
    momentum_direction = np.sign(momentum_change)

    # Divergence: opposite signs = divergence
    divergence = -price_direction * momentum_direction

    return divergence.fillna(0)


class BreakoutFeatureGenerator:
    """
    Feature generator for breakout detection strategy.

    Generates features specifically designed to:
    1. Detect consolidation periods
    2. Predict breakout timing
    3. Indicate likely breakout direction
    """

    def __init__(self, config: Optional[BreakoutConfig] = None):
        """Initialize with configuration."""
        self.config = config or BreakoutConfig()
        self._feature_names: List[str] = []

        # Also use base scalping features
        self.base_generator = ScalpingFeatureGenerator()

    def add_consolidation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that indicate consolidation."""
        result = df.copy()

        # Bollinger Band squeeze
        result["bb_squeeze"] = _detect_squeeze(df)

        # Consolidation score
        result["consolidation_score"] = _calculate_consolidation_score(
            df, lookback=self.config.lookback_bars
        )

        # ATR percentile (lower = more consolidated)
        atr = _calculate_atr(df, 14) / df["close"]
        result["atr_percentile"] = atr.rolling(window=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=True
        )

        # Number of bars in squeeze
        squeeze_runs = result["bb_squeeze"].groupby(
            (result["bb_squeeze"] != result["bb_squeeze"].shift()).cumsum()
        ).cumcount() + 1
        result["squeeze_duration"] = squeeze_runs * result["bb_squeeze"]

        # Range contraction (current range vs recent average)
        bar_range = df["high"] - df["low"]
        avg_range = bar_range.rolling(window=20).mean()
        result["range_contraction"] = bar_range / avg_range.replace(0, np.nan)

        return result

    def add_breakout_direction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that hint at breakout direction."""
        result = df.copy()

        # Position in range (0 = at low, 1 = at high)
        result["range_position"] = _calculate_range_position(
            df, lookback=self.config.range_lookback
        )

        # Distance from range boundaries
        rolling_high = df["high"].rolling(window=self.config.range_lookback).max()
        rolling_low = df["low"].rolling(window=self.config.range_lookback).min()
        range_size = rolling_high - rolling_low

        result["dist_from_high"] = (rolling_high - df["close"]) / range_size.replace(0, np.nan)
        result["dist_from_low"] = (df["close"] - rolling_low) / range_size.replace(0, np.nan)

        # Momentum divergence
        result["momentum_divergence"] = _calculate_momentum_divergence(df)

        # Recent trend within consolidation
        result["micro_trend"] = df["close"].diff(3) / df["close"]

        # Volume trend (increasing volume often precedes breakout)
        vol_ma = df["volume"].rolling(window=10).mean()
        vol_current = df["volume"].rolling(window=3).mean()
        result["volume_expansion"] = vol_current / vol_ma.replace(0, np.nan)

        return result

    def add_breakout_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that help predict breakout timing."""
        result = df.copy()

        # How long in consolidation (longer = more likely to break)
        if "consolidation_score" not in result.columns:
            result["consolidation_score"] = _calculate_consolidation_score(
                df, lookback=self.config.lookback_bars
            )

        # Rolling sum of consolidation score (cumulative consolidation)
        result["cumulative_consolidation"] = result["consolidation_score"].rolling(
            window=20
        ).sum() / 20

        # Time features (breakouts often occur at certain times)
        times = df.index.time
        minutes = np.array([t.hour * 60 + t.minute for t in times])
        rth_open_minutes = 9 * 60 + 30
        minutes_since_open = minutes - rth_open_minutes

        # First 30 minutes flag (high breakout probability)
        result["is_opening_range"] = ((minutes_since_open >= 0) &
                                       (minutes_since_open < 30)).astype(int)

        # Pre-close flag (breakouts before EOD)
        result["is_pre_close"] = ((minutes >= 15 * 60 + 30) &
                                   (minutes < 16 * 60)).astype(int)

        return result

    def generate_all(self, df: pd.DataFrame, include_base: bool = True) -> pd.DataFrame:
        """
        Generate all breakout-specific features.

        Args:
            df: DataFrame with OHLCV data
            include_base: Whether to include base scalping features

        Returns:
            DataFrame with all features added
        """
        logger.info(f"Generating breakout features for {len(df):,} bars")

        result = df.copy()

        # Add base features if requested
        if include_base:
            result = self.base_generator.generate_all(result, drop_warmup=False)

        # Add breakout-specific features
        result = self.add_consolidation_features(result)
        result = self.add_breakout_direction_features(result)
        result = self.add_breakout_timing_features(result)

        # Drop warmup period
        warmup = max(200, self.config.range_lookback * 5)
        result = result.iloc[warmup:]

        # Store feature names
        self._feature_names = self._get_feature_names(include_base)

        # Fill NaN with 0
        result[self._feature_names] = result[self._feature_names].fillna(0)

        logger.info(f"Generated {len(self._feature_names)} breakout features for {len(result):,} bars")

        return result

    def _get_feature_names(self, include_base: bool = True) -> List[str]:
        """Get list of all feature names."""
        names = []

        if include_base:
            names.extend(self.base_generator.get_feature_names())

        # Consolidation features
        names.extend([
            "bb_squeeze", "consolidation_score", "atr_percentile",
            "squeeze_duration", "range_contraction"
        ])

        # Direction features
        names.extend([
            "range_position", "dist_from_high", "dist_from_low",
            "momentum_divergence", "micro_trend", "volume_expansion"
        ])

        # Timing features
        names.extend([
            "cumulative_consolidation", "is_opening_range", "is_pre_close"
        ])

        return names

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not self._feature_names:
            self._feature_names = self._get_feature_names(include_base=True)
        return self._feature_names


def create_breakout_target(
    df: pd.DataFrame,
    horizon_bars: int = 6,
    breakout_threshold_ticks: float = 4.0,
    tick_size: float = 0.25,
    consolidation_col: str = "consolidation_score",
    consolidation_threshold: float = 0.6,
) -> Tuple[pd.DataFrame, dict]:
    """
    Create target variable for breakout prediction.

    The target is a 3-class variable:
    - 0: No significant breakout (price stays within range)
    - 1: Upward breakout (price moves up significantly)
    - 2: Downward breakout (price moves down significantly)

    Only samples where consolidation was detected are considered valid.

    Args:
        df: DataFrame with OHLCV data and consolidation_score
        horizon_bars: Bars to look ahead for breakout
        breakout_threshold_ticks: Min move in ticks for breakout
        tick_size: Contract tick size
        consolidation_col: Column name for consolidation indicator
        consolidation_threshold: Min consolidation score to consider

    Returns:
        Tuple of (DataFrame with target, stats dict)
    """
    result = df.copy()

    breakout_threshold = breakout_threshold_ticks * tick_size

    # Calculate max move in horizon period
    future_returns = []
    for i in range(len(df)):
        end_idx = min(i + horizon_bars + 1, len(df))
        if i + 1 < len(df) and end_idx > i + 1:
            future_closes = df["close"].iloc[i+1:end_idx]
            current_close = df["close"].iloc[i]
            if len(future_closes) > 0:
                max_up = (future_closes.max() - current_close)
                max_down = (current_close - future_closes.min())
                # Net move (positive = up, negative = down)
                if max_up > max_down:
                    future_returns.append(max_up)
                else:
                    future_returns.append(-max_down)
            else:
                future_returns.append(np.nan)
        else:
            future_returns.append(np.nan)

    future_move = pd.Series(future_returns, index=df.index)

    # Create 3-class target
    # 0 = no breakout, 1 = up breakout, 2 = down breakout
    target = np.where(
        future_move.isna(), np.nan,
        np.where(
            future_move > breakout_threshold, 1,  # Up breakout
            np.where(
                future_move < -breakout_threshold, 2,  # Down breakout
                0  # No breakout
            )
        )
    )

    result["target_breakout"] = target
    result["future_move"] = future_move

    # Also create binary targets for up/down
    result["target_breakout_up"] = np.where(
        future_move.isna(), np.nan,
        (future_move > breakout_threshold).astype(float)
    )
    result["target_breakout_down"] = np.where(
        future_move.isna(), np.nan,
        (future_move < -breakout_threshold).astype(float)
    )

    # Filter to consolidation periods only if column exists
    if consolidation_col in df.columns:
        is_consolidated = df[consolidation_col] >= consolidation_threshold
        result["is_valid_setup"] = is_consolidated.astype(int)
    else:
        result["is_valid_setup"] = 1

    # Calculate statistics
    valid = result["target_breakout"].dropna()
    stats = {
        "total_samples": len(valid),
        "no_breakout_pct": (valid == 0).mean() * 100,
        "up_breakout_pct": (valid == 1).mean() * 100,
        "down_breakout_pct": (valid == 2).mean() * 100,
        "threshold_ticks": breakout_threshold_ticks,
    }

    if consolidation_col in df.columns and "is_valid_setup" in result.columns:
        valid_setups = result[result["is_valid_setup"] == 1]["target_breakout"].dropna()
        if len(valid_setups) > 0:
            stats["valid_setups"] = len(valid_setups)
            stats["valid_up_breakout_pct"] = (valid_setups == 1).mean() * 100
            stats["valid_down_breakout_pct"] = (valid_setups == 2).mean() * 100

    logger.info(f"Breakout target created: {stats['up_breakout_pct']:.1f}% up, "
               f"{stats['down_breakout_pct']:.1f}% down, "
               f"{stats['no_breakout_pct']:.1f}% no breakout")

    return result, stats


def identify_breakout_setups(
    df: pd.DataFrame,
    vol_predictions: np.ndarray,
    vol_threshold: float = 0.60,
    consolidation_threshold: float = 0.60,
    consolidation_col: str = "consolidation_score",
    range_position_col: str = "range_position",
) -> pd.DataFrame:
    """
    Identify potential breakout trading setups.

    A setup is valid when:
    1. Price is in consolidation (high consolidation score)
    2. Volatility model predicts HIGH volatility coming
    3. Price position in range gives direction hint

    Args:
        df: DataFrame with features
        vol_predictions: Volatility model predictions (probability of HIGH vol)
        vol_threshold: Minimum volatility prediction to consider setup
        consolidation_threshold: Minimum consolidation score
        consolidation_col: Name of consolidation score column
        range_position_col: Name of range position column

    Returns:
        DataFrame with setup signals added
    """
    result = df.copy()

    # Add volatility predictions
    result["vol_prediction"] = vol_predictions

    # Identify consolidation
    is_consolidated = result[consolidation_col] >= consolidation_threshold

    # High volatility predicted
    vol_breakout_signal = vol_predictions >= vol_threshold

    # Setup condition: consolidated AND high vol predicted
    setup = is_consolidated & vol_breakout_signal
    result["is_breakout_setup"] = setup.astype(int)

    # Direction hint from range position
    # Near bottom (< 0.3) = likely up breakout
    # Near top (> 0.7) = likely down breakout
    range_pos = result[range_position_col]

    result["setup_direction"] = np.where(
        ~setup, 0,  # No setup
        np.where(
            range_pos < 0.35, 1,  # Up breakout (near range bottom)
            np.where(
                range_pos > 0.65, -1,  # Down breakout (near range top)
                0  # No clear direction
            )
        )
    )

    # Log statistics
    n_setups = setup.sum()
    n_up = (result["setup_direction"] == 1).sum()
    n_down = (result["setup_direction"] == -1).sum()

    logger.info(f"Identified {n_setups} breakout setups: {n_up} up, {n_down} down")

    return result


class BreakoutTrader:
    """
    Trading logic for breakout strategy.

    Entry rules:
    1. Wait for consolidation (high consolidation score)
    2. Wait for volatility model to predict HIGH volatility
    3. Enter in direction indicated by range position

    Exit rules:
    1. Profit target (4-8 ticks based on confidence)
    2. Stop loss (6 ticks)
    3. Time stop (6 bars = 30 minutes)
    4. Re-entry to consolidation (breakout failed)
    """

    def __init__(
        self,
        profit_target_ticks: float = 6.0,
        stop_loss_ticks: float = 8.0,
        time_stop_bars: int = 6,
        min_vol_confidence: float = 0.60,
        min_consolidation_score: float = 0.60,
        tick_size: float = 0.25,
        tick_value: float = 1.25,
        commission: float = 0.84,
    ):
        """Initialize trader with parameters."""
        self.profit_target_ticks = profit_target_ticks
        self.stop_loss_ticks = stop_loss_ticks
        self.time_stop_bars = time_stop_bars
        self.min_vol_confidence = min_vol_confidence
        self.min_consolidation_score = min_consolidation_score
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.commission = commission

        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0.0
        self.entry_bar = 0
        self.trades = []

    def should_enter(
        self,
        consolidation_score: float,
        vol_prediction: float,
        range_position: float,
        bar_idx: int,
    ) -> int:
        """
        Determine if we should enter a trade.

        Returns:
            1 for long, -1 for short, 0 for no trade
        """
        if self.position != 0:
            return 0

        # Must be in consolidation
        if consolidation_score < self.min_consolidation_score:
            return 0

        # Must predict high volatility
        if vol_prediction < self.min_vol_confidence:
            return 0

        # Direction based on range position
        if range_position < 0.35:
            return 1  # Long (near range low, expect up breakout)
        elif range_position > 0.65:
            return -1  # Short (near range high, expect down breakout)

        return 0  # No clear direction

    def should_exit(
        self,
        current_price: float,
        bar_idx: int,
        high: float,
        low: float,
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit current position.

        Returns:
            Tuple of (should_exit, reason)
        """
        if self.position == 0:
            return False, ""

        # Calculate unrealized P&L in ticks
        if self.position == 1:
            pnl_ticks = (current_price - self.entry_price) / self.tick_size
            # Check if stop was hit (using low)
            worst_price = low
            worst_pnl_ticks = (worst_price - self.entry_price) / self.tick_size
            # Check if target was hit (using high)
            best_price = high
            best_pnl_ticks = (best_price - self.entry_price) / self.tick_size
        else:  # short
            pnl_ticks = (self.entry_price - current_price) / self.tick_size
            # Check if stop was hit (using high)
            worst_price = high
            worst_pnl_ticks = (self.entry_price - worst_price) / self.tick_size
            # Check if target was hit (using low)
            best_price = low
            best_pnl_ticks = (self.entry_price - best_price) / self.tick_size

        # Stop loss (check worst price first)
        if worst_pnl_ticks <= -self.stop_loss_ticks:
            return True, "stop_loss"

        # Profit target (check best price)
        if best_pnl_ticks >= self.profit_target_ticks:
            return True, "profit_target"

        # Time stop
        bars_held = bar_idx - self.entry_bar
        if bars_held >= self.time_stop_bars:
            return True, "time_stop"

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
            fill_price = price + self.tick_size
        else:
            fill_price = price - self.tick_size

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
            fill_price = price - self.tick_size
        else:
            fill_price = price + self.tick_size

        # Calculate P&L
        if self.position == 1:
            pnl_ticks = (fill_price - self.entry_price) / self.tick_size
        else:
            pnl_ticks = (self.entry_price - fill_price) / self.tick_size

        pnl_dollars = pnl_ticks * self.tick_value - self.commission

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


def run_breakout_backtest(
    df: pd.DataFrame,
    vol_predictions: np.ndarray,
    feature_names: List[str],
    config: Optional[BreakoutConfig] = None,
) -> Tuple[List[dict], dict]:
    """
    Run backtest using breakout strategy.

    Args:
        df: DataFrame with features
        vol_predictions: Volatility model predictions
        feature_names: List of feature column names
        config: Breakout configuration

    Returns:
        Tuple of (list of trades, summary statistics)
    """
    config = config or BreakoutConfig()

    trader = BreakoutTrader(
        profit_target_ticks=6.0,
        stop_loss_ticks=8.0,
        time_stop_bars=config.horizon_bars,
        min_vol_confidence=config.vol_prediction_threshold,
        min_consolidation_score=config.consolidation_threshold,
    )

    trades = []
    current_trade = None

    for i, (idx, row) in enumerate(df.iterrows()):
        # Skip if no predictions
        if i >= len(vol_predictions):
            break

        vol_pred = vol_predictions[i]

        # Check for exit first
        if trader.position != 0:
            should_exit, reason = trader.should_exit(
                row["close"], i, row["high"], row["low"]
            )
            if should_exit:
                exit_info = trader.exit_trade(row["close"], i, idx, reason)
                current_trade.update(exit_info)
                trades.append(current_trade)
                current_trade = None

        # Check for entry
        if trader.position == 0:
            direction = trader.should_enter(
                row.get("consolidation_score", 0),
                vol_pred,
                row.get("range_position", 0.5),
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
        profit_factor = abs(sum(t["pnl_dollars"] for t in wins) /
                           sum(t["pnl_dollars"] for t in losses)) if losses and sum(t["pnl_dollars"] for t in losses) != 0 else 0

        summary = {
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "long_trades": len([t for t in trades if t.get("direction") == 1]),
            "short_trades": len([t for t in trades if t.get("direction") == -1]),
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
        }

    return trades, summary
