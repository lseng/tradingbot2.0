"""
Position Manager for Live Trading.

Tracks open positions, calculates real-time P&L, and syncs with API state.
This is the source of truth for position state in the trading system.

Key responsibilities:
- Track current position (direction, size, entry price)
- Calculate unrealized P&L tick-by-tick
- Manage stop loss and take profit order IDs
- Sync with API on reconnection (API is source of truth)
- Provide position change notifications

Reference: specs/live-trading-execution.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, List
from enum import Enum
import logging
import threading

from src.lib.constants import MES_TICK_SIZE, MES_TICK_VALUE, MES_POINT_VALUE

logger = logging.getLogger(__name__)


class PositionDirection(Enum):
    """Position direction enum."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    """
    Represents an open trading position.

    Attributes:
        contract_id: Contract identifier (e.g., "CON.F.US.MES.H26")
        direction: 1 (long), -1 (short), 0 (flat)
        size: Number of contracts (always positive)
        entry_price: Average entry price
        entry_time: Time position was opened
        stop_price: Stop loss price
        target_price: Take profit price
        stop_order_id: ID of the stop loss order
        target_order_id: ID of the take profit order
        unrealized_pnl: Current unrealized P&L in dollars
        realized_pnl: P&L from partial closes (if any)
    """
    contract_id: str
    direction: int = 0  # 1=long, -1=short, 0=flat
    size: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_price: float = 0.0
    target_price: float = 0.0
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.direction == 0 or self.size == 0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.direction == 1 and self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.direction == -1 and self.size > 0

    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in dollars
        """
        if self.is_flat:
            return 0.0

        # Price difference in points
        price_diff = current_price - self.entry_price

        # P&L = direction * price_diff * size * point_value
        pnl = self.direction * price_diff * self.size * MES_POINT_VALUE

        return pnl

    def calculate_pnl_ticks(self, current_price: float) -> float:
        """
        Calculate unrealized P&L in ticks.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in ticks
        """
        if self.is_flat:
            return 0.0

        tick_diff = (current_price - self.entry_price) / MES_TICK_SIZE
        return self.direction * tick_diff * self.size

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/persistence."""
        return {
            "contract_id": self.contract_id,
            "direction": self.direction,
            "size": self.size,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "stop_order_id": self.stop_order_id,
            "target_order_id": self.target_order_id,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class Fill:
    """Represents an order fill event."""
    order_id: str
    contract_id: str
    side: int  # 1=buy, 2=sell
    size: int
    price: float
    timestamp: datetime
    is_entry: bool = True
    custom_tag: Optional[str] = None


@dataclass
class PositionChange:
    """Event emitted when position changes."""
    old_position: Optional[Position]
    new_position: Position
    fill: Optional[Fill]
    timestamp: datetime
    change_type: str  # "open", "close", "partial_close", "modify"


class PositionManager:
    """
    Manages position state and provides position tracking utilities.

    Thread-safe implementation with callbacks for position changes.

    Usage:
        manager = PositionManager(contract_id="CON.F.US.MES.H26")

        # Register callback for position changes
        manager.on_position_change(my_callback)

        # Update from fill
        manager.update_from_fill(fill)

        # Update P&L
        manager.update_pnl(current_price=6050.25)

        # Sync from API
        manager.sync_from_api(api_position)
    """

    def __init__(self, contract_id: str):
        """
        Initialize position manager.

        Args:
            contract_id: Contract to track (e.g., "CON.F.US.MES.H26")
        """
        self.contract_id = contract_id
        self._position = Position(contract_id=contract_id)
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[PositionChange], None]] = []
        self._last_price: Optional[float] = None

        logger.info(f"PositionManager initialized for {contract_id}")

    @property
    def position(self) -> Position:
        """Get current position (read-only copy)."""
        with self._lock:
            return Position(
                contract_id=self._position.contract_id,
                direction=self._position.direction,
                size=self._position.size,
                entry_price=self._position.entry_price,
                entry_time=self._position.entry_time,
                stop_price=self._position.stop_price,
                target_price=self._position.target_price,
                stop_order_id=self._position.stop_order_id,
                target_order_id=self._position.target_order_id,
                unrealized_pnl=self._position.unrealized_pnl,
                realized_pnl=self._position.realized_pnl,
            )

    def is_flat(self) -> bool:
        """Check if position is flat."""
        with self._lock:
            return self._position.is_flat

    def is_long(self) -> bool:
        """Check if position is long."""
        with self._lock:
            return self._position.is_long

    def is_short(self) -> bool:
        """Check if position is short."""
        with self._lock:
            return self._position.is_short

    def get_direction(self) -> int:
        """Get position direction (1=long, -1=short, 0=flat)."""
        with self._lock:
            return self._position.direction

    def get_size(self) -> int:
        """Get position size."""
        with self._lock:
            return self._position.size

    def get_unrealized_pnl(self) -> float:
        """Get current unrealized P&L."""
        with self._lock:
            return self._position.unrealized_pnl

    def on_position_change(self, callback: Callable[[PositionChange], None]) -> None:
        """
        Register callback for position changes.

        Args:
            callback: Function to call when position changes
        """
        self._callbacks.append(callback)
        logger.debug(f"Position change callback registered: {callback}")

    def update_from_fill(self, fill: Fill) -> None:
        """
        Update position from an order fill.

        Handles:
        - Opening new positions
        - Closing existing positions
        - Partial fills
        - Reversals

        Args:
            fill: Fill event from order execution
        """
        with self._lock:
            old_position = self.position  # Copy before modification

            # Determine fill direction: BUY=1, SELL=2 -> direction: 1 or -1
            fill_direction = 1 if fill.side == 1 else -1

            if self._position.is_flat:
                # Opening new position
                self._open_position(fill, fill_direction)
                change_type = "open"

            elif self._position.direction == fill_direction:
                # Adding to existing position (same direction)
                self._add_to_position(fill)
                change_type = "add"

            else:
                # Opposite direction - closing or reversing
                if fill.size >= self._position.size:
                    # Full close or reversal
                    remaining = fill.size - self._position.size
                    self._close_position(fill, partial=False)
                    change_type = "close"

                    if remaining > 0:
                        # Reversal - open in opposite direction
                        reversal_fill = Fill(
                            order_id=fill.order_id,
                            contract_id=fill.contract_id,
                            side=fill.side,
                            size=remaining,
                            price=fill.price,
                            timestamp=fill.timestamp,
                            is_entry=True,
                        )
                        self._open_position(reversal_fill, fill_direction)
                        change_type = "reversal"
                else:
                    # Partial close
                    self._partial_close(fill)
                    change_type = "partial_close"

            # Notify callbacks
            change = PositionChange(
                old_position=old_position,
                new_position=self.position,
                fill=fill,
                timestamp=fill.timestamp,
                change_type=change_type,
            )
            self._notify_callbacks(change)

            logger.info(
                f"Position updated: {change_type}, "
                f"direction={self._position.direction}, "
                f"size={self._position.size}, "
                f"entry_price={self._position.entry_price:.2f}"
            )

    def _open_position(self, fill: Fill, direction: int) -> None:
        """Open a new position."""
        self._position.direction = direction
        self._position.size = fill.size
        self._position.entry_price = fill.price
        self._position.entry_time = fill.timestamp
        self._position.unrealized_pnl = 0.0
        self._position.realized_pnl = 0.0
        self._position.stop_order_id = None
        self._position.target_order_id = None

    def _add_to_position(self, fill: Fill) -> None:
        """Add to existing position (average in)."""
        total_size = self._position.size + fill.size

        # Calculate weighted average entry price
        old_value = self._position.entry_price * self._position.size
        new_value = fill.price * fill.size
        self._position.entry_price = (old_value + new_value) / total_size

        self._position.size = total_size

    def _close_position(self, fill: Fill, partial: bool = False) -> None:
        """Close position (full or partial)."""
        # Calculate realized P&L
        price_diff = fill.price - self._position.entry_price
        pnl = self._position.direction * price_diff * self._position.size * MES_POINT_VALUE
        self._position.realized_pnl += pnl

        # Reset position
        self._position.direction = 0
        self._position.size = 0
        self._position.entry_price = 0.0
        self._position.entry_time = None
        self._position.unrealized_pnl = 0.0
        self._position.stop_price = 0.0
        self._position.target_price = 0.0
        self._position.stop_order_id = None
        self._position.target_order_id = None

    def _partial_close(self, fill: Fill) -> None:
        """Partially close position."""
        # Calculate realized P&L for closed portion
        price_diff = fill.price - self._position.entry_price
        pnl = self._position.direction * price_diff * fill.size * MES_POINT_VALUE
        self._position.realized_pnl += pnl

        # Reduce position size
        self._position.size -= fill.size

        # If fully closed, reset
        if self._position.size <= 0:
            self._close_position(fill)

    def update_pnl(self, current_price: float) -> float:
        """
        Update unrealized P&L based on current price.

        Args:
            current_price: Current market price

        Returns:
            Updated unrealized P&L
        """
        with self._lock:
            self._last_price = current_price
            self._position.unrealized_pnl = self._position.calculate_pnl(current_price)
            return self._position.unrealized_pnl

    def set_stop_price(self, price: float, order_id: Optional[str] = None) -> None:
        """
        Set stop loss price and order ID.

        Args:
            price: Stop loss price
            order_id: Stop loss order ID
        """
        with self._lock:
            self._position.stop_price = price
            if order_id:
                self._position.stop_order_id = order_id
            logger.debug(f"Stop price set: {price}, order_id={order_id}")

    def set_target_price(self, price: float, order_id: Optional[str] = None) -> None:
        """
        Set take profit price and order ID.

        Args:
            price: Take profit price
            order_id: Take profit order ID
        """
        with self._lock:
            self._position.target_price = price
            if order_id:
                self._position.target_order_id = order_id
            logger.debug(f"Target price set: {price}, order_id={order_id}")

    def sync_from_api(self, api_position: 'PositionData') -> bool:
        """
        Sync position state from API.

        API is always the source of truth. This method reconciles local
        state with API state, logging any discrepancies.

        Args:
            api_position: Position data from API

        Returns:
            True if positions match, False if discrepancy was detected
        """
        with self._lock:
            old_position = self.position

            # Check for discrepancies
            has_discrepancy = False

            if api_position.contract_id != self.contract_id:
                logger.warning(
                    f"Contract ID mismatch: local={self.contract_id}, "
                    f"api={api_position.contract_id}"
                )
                return False

            api_direction = api_position.direction
            api_size = abs(api_position.size)

            if api_direction != self._position.direction:
                logger.warning(
                    f"Direction mismatch: local={self._position.direction}, "
                    f"api={api_direction}"
                )
                has_discrepancy = True

            if api_size != self._position.size:
                logger.warning(
                    f"Size mismatch: local={self._position.size}, "
                    f"api={api_size}"
                )
                has_discrepancy = True

            # Update from API (API is source of truth)
            self._position.direction = api_direction
            self._position.size = api_size
            self._position.entry_price = api_position.avg_price
            self._position.unrealized_pnl = api_position.unrealized_pnl

            if has_discrepancy:
                logger.warning("Position synced from API (API is source of truth)")

                # Notify callbacks
                change = PositionChange(
                    old_position=old_position,
                    new_position=self.position,
                    fill=None,
                    timestamp=datetime.now(),
                    change_type="sync",
                )
                self._notify_callbacks(change)

            return not has_discrepancy

    def flatten(self) -> None:
        """
        Mark position as flat (for emergency/EOD flatten).

        Note: This only updates local state. The actual close order
        must be placed separately via OrderExecutor.
        """
        with self._lock:
            old_position = self.position

            self._position.direction = 0
            self._position.size = 0
            self._position.entry_price = 0.0
            self._position.entry_time = None
            self._position.unrealized_pnl = 0.0
            self._position.stop_price = 0.0
            self._position.target_price = 0.0
            self._position.stop_order_id = None
            self._position.target_order_id = None

            logger.info("Position flattened (local state)")

            # Notify callbacks
            change = PositionChange(
                old_position=old_position,
                new_position=self.position,
                fill=None,
                timestamp=datetime.now(),
                change_type="flatten",
            )
            self._notify_callbacks(change)

    def _notify_callbacks(self, change: PositionChange) -> None:
        """Notify all registered callbacks of position change."""
        for callback in self._callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Error in position change callback: {e}")

    def get_metrics(self) -> dict:
        """
        Get position metrics for logging/display.

        Returns:
            Dictionary of position metrics
        """
        with self._lock:
            pos = self._position
            direction_str = "FLAT"
            if pos.direction == 1:
                direction_str = "LONG"
            elif pos.direction == -1:
                direction_str = "SHORT"

            return {
                "contract_id": pos.contract_id,
                "direction": direction_str,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "stop_price": pos.stop_price,
                "target_price": pos.target_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "total_pnl": pos.unrealized_pnl + pos.realized_pnl,
                "last_price": self._last_price,
                "pnl_ticks": pos.calculate_pnl_ticks(self._last_price) if self._last_price else 0,
            }


# For convenience - re-export PositionData from API module
try:
    from src.api.topstepx_rest import PositionData
except ImportError:
    # Define a minimal PositionData for testing
    @dataclass
    class PositionData:
        """Minimal PositionData for standalone testing."""
        contract_id: str
        size: int
        avg_price: float
        unrealized_pnl: float = 0.0
        realized_pnl: float = 0.0

        @property
        def direction(self) -> int:
            if self.size > 0:
                return 1
            elif self.size < 0:
                return -1
            return 0
