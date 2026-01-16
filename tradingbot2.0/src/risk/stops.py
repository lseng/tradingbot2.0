"""
Stop Loss Management Module for MES Futures Scalping Bot.

Provides multiple stop loss strategies:
1. ATR-Based Stops (recommended) - Dynamic based on volatility
2. Fixed Tick Stops - Simple fixed distance
3. Structure-Based Stops - Based on swing highs/lows
4. Trailing Stops - Move stop to protect profits

Key Parameters (from spec):
- Default ATR multiplier: 1.5
- Default fixed stop: 8 ticks ($10)
- Trailing stop: Move to breakeven after X profit, then trail
- EOD tightening: Tighter stops near market close
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StopType(Enum):
    """Type of stop loss strategy."""
    ATR = "atr"  # Volatility-based
    FIXED = "fixed"  # Fixed tick distance
    STRUCTURE = "structure"  # Swing high/low based
    TRAILING = "trailing"  # Trailing stop


@dataclass
class StopConfig:
    """Configuration for stop loss calculations."""
    # MES specifications
    tick_size: float = 0.25
    tick_value: float = 1.25

    # ATR-based stops
    atr_multiplier: float = 1.5
    atr_period: int = 14
    min_atr_ticks: int = 4  # Minimum stop distance
    max_atr_ticks: int = 16  # Maximum stop distance

    # Fixed stops
    default_stop_ticks: int = 8  # 8 ticks = $10 per contract

    # Trailing stops
    trail_trigger_ticks: int = 4  # Move to BE after 4 ticks profit
    trail_distance_ticks: int = 4  # Trail by 4 ticks

    # EOD adjustment
    eod_tighten_factor: float = 0.75  # Reduce stop by 25% near EOD


@dataclass
class StopResult:
    """Result of stop price calculation."""
    stop_price: float
    stop_ticks: int
    stop_dollars: float  # Dollar risk per contract
    stop_type: StopType
    reason: str


class StopLossManager:
    """
    Stop loss calculator and manager.

    Calculates stop prices using various strategies and manages
    trailing stop logic.

    Usage:
        manager = StopLossManager()

        # ATR-based stop for long entry
        result = manager.calculate_atr_stop(
            entry_price=6050.25,
            direction=1,  # long
            atr=2.5,  # 2.5 point ATR
        )
        print(f"Stop at {result.stop_price}, risk ${result.stop_dollars}")

        # Trailing stop update
        new_stop = manager.calculate_trailing_stop(
            entry_price=6050.25,
            current_price=6054.50,
            current_stop=6048.00,
            direction=1,
        )
    """

    def __init__(self, config: Optional[StopConfig] = None):
        """
        Initialize stop loss manager.

        Args:
            config: Stop configuration (uses defaults if None)
        """
        self.config = config or StopConfig()

    def calculate_atr_stop(
        self,
        entry_price: float,
        direction: int,
        atr: float,
        atr_multiplier: Optional[float] = None,
    ) -> StopResult:
        """
        Calculate stop price based on ATR (Average True Range).

        Recommended stop strategy - adapts to market volatility.

        Args:
            entry_price: Entry price
            direction: Trade direction (1=long, -1=short)
            atr: Current ATR value (in points, not ticks)
            atr_multiplier: Override default ATR multiplier

        Returns:
            StopResult with calculated stop price and details
        """
        mult = atr_multiplier or self.config.atr_multiplier

        # Calculate stop distance in points
        stop_distance = atr * mult

        # Convert to ticks and clamp
        stop_ticks = int(stop_distance / self.config.tick_size)
        stop_ticks = max(self.config.min_atr_ticks, min(stop_ticks, self.config.max_atr_ticks))

        # Convert back to points for stop price
        stop_points = stop_ticks * self.config.tick_size

        # Calculate stop price based on direction
        if direction > 0:  # Long
            stop_price = entry_price - stop_points
        else:  # Short
            stop_price = entry_price + stop_points

        # Round to tick size
        stop_price = round(stop_price / self.config.tick_size) * self.config.tick_size

        stop_dollars = stop_ticks * self.config.tick_value

        return StopResult(
            stop_price=stop_price,
            stop_ticks=stop_ticks,
            stop_dollars=stop_dollars,
            stop_type=StopType.ATR,
            reason=f"ATR={atr:.2f}, mult={mult}, stop={stop_ticks} ticks",
        )

    def calculate_fixed_stop(
        self,
        entry_price: float,
        direction: int,
        stop_ticks: Optional[int] = None,
    ) -> StopResult:
        """
        Calculate stop price with fixed tick distance.

        Simple strategy - good for consistent risk per trade.

        Args:
            entry_price: Entry price
            direction: Trade direction (1=long, -1=short)
            stop_ticks: Stop distance in ticks (uses default if None)

        Returns:
            StopResult with calculated stop price and details
        """
        ticks = stop_ticks or self.config.default_stop_ticks
        stop_points = ticks * self.config.tick_size

        if direction > 0:  # Long
            stop_price = entry_price - stop_points
        else:  # Short
            stop_price = entry_price + stop_points

        # Round to tick size
        stop_price = round(stop_price / self.config.tick_size) * self.config.tick_size

        stop_dollars = ticks * self.config.tick_value

        return StopResult(
            stop_price=stop_price,
            stop_ticks=ticks,
            stop_dollars=stop_dollars,
            stop_type=StopType.FIXED,
            reason=f"Fixed {ticks} ticks from entry",
        )

    def calculate_structure_stop(
        self,
        entry_price: float,
        direction: int,
        swing_prices: List[float],
        buffer_ticks: int = 2,
    ) -> StopResult:
        """
        Calculate stop price based on market structure (swing high/low).

        Places stop beyond recent swing point with buffer.

        Args:
            entry_price: Entry price
            direction: Trade direction (1=long, -1=short)
            swing_prices: List of recent swing high/low prices
            buffer_ticks: Additional buffer beyond swing point

        Returns:
            StopResult with calculated stop price and details
        """
        if not swing_prices:
            # Fall back to fixed stop if no swing data
            return self.calculate_fixed_stop(entry_price, direction)

        buffer_points = buffer_ticks * self.config.tick_size

        if direction > 0:  # Long - stop below swing low
            swing_low = min(swing_prices)
            stop_price = swing_low - buffer_points
        else:  # Short - stop above swing high
            swing_high = max(swing_prices)
            stop_price = swing_high + buffer_points

        # Round to tick size
        stop_price = round(stop_price / self.config.tick_size) * self.config.tick_size

        # Calculate ticks and dollars
        stop_ticks = int(abs(entry_price - stop_price) / self.config.tick_size)
        stop_dollars = stop_ticks * self.config.tick_value

        return StopResult(
            stop_price=stop_price,
            stop_ticks=stop_ticks,
            stop_dollars=stop_dollars,
            stop_type=StopType.STRUCTURE,
            reason=f"Structure-based with {buffer_ticks} tick buffer",
        )

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        direction: int,
        trail_trigger_ticks: Optional[int] = None,
        trail_distance_ticks: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate new trailing stop price.

        Logic:
        1. If profit reaches trigger threshold, move stop to breakeven
        2. After breakeven, trail by specified distance
        3. Stop only moves in favorable direction (never loosens)

        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_stop: Current stop price
            direction: Trade direction (1=long, -1=short)
            trail_trigger_ticks: Ticks profit to trigger trailing
            trail_distance_ticks: Distance to trail behind price

        Returns:
            New stop price if should be updated, None otherwise
        """
        trigger_ticks = trail_trigger_ticks or self.config.trail_trigger_ticks
        trail_ticks = trail_distance_ticks or self.config.trail_distance_ticks

        trigger_points = trigger_ticks * self.config.tick_size
        trail_points = trail_ticks * self.config.tick_size

        if direction > 0:  # Long position
            profit_points = current_price - entry_price

            # Check if we've hit trailing trigger
            if profit_points < trigger_points:
                return None  # Not enough profit yet

            # Calculate potential new stop
            new_stop = current_price - trail_points

            # Minimum: breakeven (entry price)
            new_stop = max(new_stop, entry_price)

            # Round to tick size
            new_stop = round(new_stop / self.config.tick_size) * self.config.tick_size

            # Only update if new stop is higher (tighter) than current
            if new_stop > current_stop:
                return new_stop

        else:  # Short position
            profit_points = entry_price - current_price

            # Check if we've hit trailing trigger
            if profit_points < trigger_points:
                return None  # Not enough profit yet

            # Calculate potential new stop
            new_stop = current_price + trail_points

            # Maximum: breakeven (entry price)
            new_stop = min(new_stop, entry_price)

            # Round to tick size
            new_stop = round(new_stop / self.config.tick_size) * self.config.tick_size

            # Only update if new stop is lower (tighter) than current
            if new_stop < current_stop:
                return new_stop

        return None  # No update needed

    def apply_eod_tightening(
        self,
        stop_result: StopResult,
        entry_price: float,
        direction: int,
        tighten_factor: Optional[float] = None,
    ) -> StopResult:
        """
        Tighten stop for end-of-day management.

        Reduces stop distance as market close approaches to
        protect profits and limit overnight exposure risk.

        Args:
            stop_result: Original stop calculation
            entry_price: Entry price
            direction: Trade direction (1=long, -1=short)
            tighten_factor: Factor to reduce stop (default 0.75 = 25% tighter)

        Returns:
            New StopResult with tightened stop
        """
        factor = tighten_factor or self.config.eod_tighten_factor

        # Calculate new (tighter) tick distance
        new_ticks = max(self.config.min_atr_ticks, int(stop_result.stop_ticks * factor))
        new_points = new_ticks * self.config.tick_size

        if direction > 0:  # Long
            new_stop = entry_price - new_points
        else:  # Short
            new_stop = entry_price + new_points

        # Round to tick size
        new_stop = round(new_stop / self.config.tick_size) * self.config.tick_size

        new_dollars = new_ticks * self.config.tick_value

        return StopResult(
            stop_price=new_stop,
            stop_ticks=new_ticks,
            stop_dollars=new_dollars,
            stop_type=stop_result.stop_type,
            reason=f"EOD tightened: {stop_result.stop_ticks} -> {new_ticks} ticks",
        )

    def calculate_target_price(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        rr_ratio: float = 2.0,
    ) -> Tuple[float, int, float]:
        """
        Calculate take profit target based on risk:reward ratio.

        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            direction: Trade direction (1=long, -1=short)
            rr_ratio: Risk:reward ratio (e.g., 2.0 for 1:2 R:R)

        Returns:
            Tuple of (target_price, target_ticks, target_dollars)
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)

        # Calculate target distance
        target_distance = stop_distance * rr_ratio

        # Calculate target price
        if direction > 0:  # Long
            target_price = entry_price + target_distance
        else:  # Short
            target_price = entry_price - target_distance

        # Round to tick size
        target_price = round(target_price / self.config.tick_size) * self.config.tick_size

        # Calculate ticks and dollars
        target_ticks = int(target_distance / self.config.tick_size)
        target_dollars = target_ticks * self.config.tick_value

        return target_price, target_ticks, target_dollars


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ATR period (default 14)

    Returns:
        Array of ATR values (same length as input, first period-1 will be NaN)
    """
    n = len(closes)
    if n < 2:
        return np.array([np.nan])

    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]  # First value uses itself

    tr1 = highs - lows
    tr2 = np.abs(highs - prev_close)
    tr3 = np.abs(lows - prev_close)

    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    # Calculate ATR using exponential moving average
    atr = np.full(n, np.nan)

    if n >= period:
        # First ATR is simple average
        atr[period - 1] = np.mean(true_range[:period])

        # Subsequent ATR uses smoothed average
        multiplier = 2 / (period + 1)
        for i in range(period, n):
            atr[i] = (true_range[i] * multiplier) + (atr[i - 1] * (1 - multiplier))

    return atr
