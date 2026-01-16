"""
Slippage Model for MES Futures

This module models execution slippage - the difference between the expected
fill price and actual fill price. For MES futures, slippage is measured in
ticks (0.25 points = $1.25 per contract).

Slippage Sources:
1. Market orders: Fill at best available price, usually 1 tick worse
2. Liquidity: Thin order books cause larger slippage
3. Volatility: Fast markets have wider spreads and more slippage
4. Order size: Large orders may need multiple price levels

MES Slippage Estimates:
- Normal conditions: 1 tick ($1.25)
- Low liquidity (ETH, thin book): 2 ticks ($2.50)
- High volatility (news, FOMC): 2-4 ticks ($2.50-$5.00)
- Limit orders (if filled): 0 ticks

NOTE: The legacy evaluation.py uses 0.0001 (0.01%) which is WRONG for futures.
Futures use tick-based slippage, not percentage-based.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class MarketCondition(Enum):
    """Market condition affects slippage amount."""
    NORMAL = "normal"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME = "extreme"  # News events, circuit breakers


class OrderType(Enum):
    """Order type determines slippage application."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class SlippageConfig:
    """
    Configuration for slippage model.

    All slippage values are in TICKS (not points or dollars).
    For MES: 1 tick = 0.25 points = $1.25 per contract.

    Attributes:
        tick_size: Minimum price movement (0.25 for MES)
        tick_value: Dollar value per tick (1.25 for MES)
        normal_slippage_ticks: Expected slippage under normal conditions
        low_liquidity_slippage_ticks: Slippage during thin markets
        high_volatility_slippage_ticks: Slippage during volatile periods
        extreme_slippage_ticks: Slippage during extreme events
        atr_threshold_multiplier: ATR multiple that triggers high volatility mode
    """
    tick_size: float = 0.25
    tick_value: float = 1.25
    normal_slippage_ticks: float = 1.0
    low_liquidity_slippage_ticks: float = 2.0
    high_volatility_slippage_ticks: float = 2.0
    extreme_slippage_ticks: float = 4.0
    atr_threshold_multiplier: float = 3.0  # 3x ATR = high volatility


class SlippageModel:
    """
    Slippage calculator for MES futures trading.

    This model calculates realistic fill prices based on order type,
    market conditions, and volatility. It's designed to be pessimistic
    to avoid over-optimistic backtest results.

    Key Design Decisions:
    1. Market orders ALWAYS get slippage (minimum 1 tick)
    2. Limit orders get 0 slippage IF they fill (fill probability not modeled)
    3. Stop orders get slippage because they become market orders
    4. Volatility-based adjustments use ATR as a proxy

    Usage:
        model = SlippageModel()

        # Market order during normal conditions
        fill_price = model.apply_slippage(
            price=4500.00,
            direction=1,  # buying
            order_type=OrderType.MARKET
        )  # Returns 4500.25 (1 tick worse)
    """

    def __init__(self, config: Optional[SlippageConfig] = None):
        """
        Initialize the slippage model.

        Args:
            config: Slippage configuration. Uses defaults if not provided.
        """
        self.config = config or SlippageConfig()
        self._total_slippage_dollars = 0.0
        self._total_slippage_ticks = 0.0
        self._slippage_events = 0

    def get_slippage_ticks(
        self,
        order_type: OrderType,
        condition: MarketCondition = MarketCondition.NORMAL,
        current_atr: Optional[float] = None,
        normal_atr: Optional[float] = None,
    ) -> float:
        """
        Get expected slippage in ticks for given conditions.

        Args:
            order_type: Type of order (market, limit, stop)
            condition: Current market condition
            current_atr: Current ATR value (optional, for dynamic adjustment)
            normal_atr: Baseline ATR for comparison (optional)

        Returns:
            Expected slippage in ticks (can be fractional)
        """
        # Limit orders have no slippage if filled
        if order_type == OrderType.LIMIT:
            return 0.0

        # Stop-limit orders: assume limit portion fills, so no extra slippage
        if order_type == OrderType.STOP_LIMIT:
            return 0.0

        # Market and stop orders get slippage
        # Base slippage depends on market condition
        base_slippage = {
            MarketCondition.NORMAL: self.config.normal_slippage_ticks,
            MarketCondition.LOW_LIQUIDITY: self.config.low_liquidity_slippage_ticks,
            MarketCondition.HIGH_VOLATILITY: self.config.high_volatility_slippage_ticks,
            MarketCondition.EXTREME: self.config.extreme_slippage_ticks,
        }.get(condition, self.config.normal_slippage_ticks)

        # ATR-based adjustment: if current volatility is elevated
        if current_atr is not None and normal_atr is not None and normal_atr > 0:
            vol_ratio = current_atr / normal_atr
            if vol_ratio > self.config.atr_threshold_multiplier:
                # Add extra slippage for high volatility
                base_slippage += (vol_ratio - 1) * 0.5  # 0.5 tick per 1x ATR excess

        return base_slippage

    def apply_slippage(
        self,
        price: float,
        direction: int,
        order_type: OrderType = OrderType.MARKET,
        condition: MarketCondition = MarketCondition.NORMAL,
        current_atr: Optional[float] = None,
        normal_atr: Optional[float] = None,
        contracts: int = 1,
        record: bool = True,
    ) -> float:
        """
        Calculate fill price after slippage.

        Slippage always works AGAINST the trader:
        - Buying (direction=1): price increases (fill higher)
        - Selling (direction=-1): price decreases (fill lower)

        Args:
            price: Intended fill price
            direction: Trade direction (1=buy, -1=sell)
            order_type: Type of order
            condition: Current market condition
            current_atr: Current ATR for volatility adjustment
            normal_atr: Baseline ATR for comparison
            contracts: Number of contracts (for tracking total cost)
            record: Whether to record this slippage event

        Returns:
            Adjusted fill price after slippage
        """
        slippage_ticks = self.get_slippage_ticks(
            order_type=order_type,
            condition=condition,
            current_atr=current_atr,
            normal_atr=normal_atr,
        )

        # Convert ticks to price movement
        slippage_points = slippage_ticks * self.config.tick_size

        # Apply slippage against the trader's favor
        # direction = 1 (buy): add to price (pay more)
        # direction = -1 (sell): subtract from price (receive less)
        fill_price = price + (slippage_points * direction)

        # Round to tick size
        fill_price = round(fill_price / self.config.tick_size) * self.config.tick_size

        # Track slippage costs
        if record and slippage_ticks > 0:
            self._slippage_events += 1
            self._total_slippage_ticks += slippage_ticks * contracts
            self._total_slippage_dollars += (
                slippage_ticks * self.config.tick_value * contracts
            )

        return fill_price

    def get_slippage_cost(
        self,
        slippage_ticks: float,
        contracts: int = 1
    ) -> float:
        """
        Convert slippage ticks to dollar cost.

        Args:
            slippage_ticks: Amount of slippage in ticks
            contracts: Number of contracts

        Returns:
            Slippage cost in dollars
        """
        return slippage_ticks * self.config.tick_value * contracts

    def detect_market_condition(
        self,
        current_atr: float,
        normal_atr: float,
        spread_ticks: Optional[float] = None,
        volume_ratio: Optional[float] = None,
    ) -> MarketCondition:
        """
        Detect market condition based on volatility and liquidity indicators.

        Args:
            current_atr: Current ATR value
            normal_atr: Baseline ATR (e.g., 20-day average)
            spread_ticks: Current bid-ask spread in ticks (optional)
            volume_ratio: Current volume / average volume (optional)

        Returns:
            Detected market condition
        """
        # Start with normal assumption
        condition = MarketCondition.NORMAL

        # Check ATR-based volatility
        if normal_atr > 0:
            vol_ratio = current_atr / normal_atr
            if vol_ratio > 5:
                return MarketCondition.EXTREME
            elif vol_ratio > self.config.atr_threshold_multiplier:
                condition = MarketCondition.HIGH_VOLATILITY

        # Check spread-based liquidity (if available)
        if spread_ticks is not None and spread_ticks > 2:
            condition = MarketCondition.LOW_LIQUIDITY

        # Check volume-based liquidity (if available)
        if volume_ratio is not None and volume_ratio < 0.1:
            condition = MarketCondition.LOW_LIQUIDITY

        return condition

    def get_total_slippage_dollars(self) -> float:
        """Get total slippage cost in dollars across all recorded events."""
        return self._total_slippage_dollars

    def get_total_slippage_ticks(self) -> float:
        """Get total slippage in ticks across all recorded events."""
        return self._total_slippage_ticks

    def get_slippage_events(self) -> int:
        """Get number of slippage events recorded."""
        return self._slippage_events

    def get_average_slippage_ticks(self) -> float:
        """Get average slippage per event in ticks."""
        if self._slippage_events == 0:
            return 0.0
        return self._total_slippage_ticks / self._slippage_events

    def reset(self) -> None:
        """Reset cumulative tracking (e.g., for new backtest run)."""
        self._total_slippage_dollars = 0.0
        self._total_slippage_ticks = 0.0
        self._slippage_events = 0

    def __repr__(self) -> str:
        return (
            f"SlippageModel("
            f"normal={self.config.normal_slippage_ticks} ticks, "
            f"tick_value=${self.config.tick_value})"
        )


def calculate_realistic_slippage(
    price: float,
    direction: int,
    is_market_order: bool = True,
    volatility_multiple: float = 1.0,
) -> float:
    """
    Convenience function for quick slippage calculation.

    Args:
        price: Intended price
        direction: 1 for buy, -1 for sell
        is_market_order: True for market order (gets slippage)
        volatility_multiple: Multiplier for volatile conditions (1.0 = normal)

    Returns:
        Fill price after slippage
    """
    model = SlippageModel()

    # Determine condition based on volatility multiple
    if volatility_multiple <= 1.0:
        condition = MarketCondition.NORMAL
    elif volatility_multiple <= 3.0:
        condition = MarketCondition.HIGH_VOLATILITY
    else:
        condition = MarketCondition.EXTREME

    order_type = OrderType.MARKET if is_market_order else OrderType.LIMIT

    return model.apply_slippage(
        price=price,
        direction=direction,
        order_type=order_type,
        condition=condition,
        record=False,
    )
