"""
Order Executor for Live Trading.

Handles all order placement, management, and execution with TopstepX API.
Manages the complete order lifecycle:
- Entry orders (market for speed)
- Stop loss orders (placed immediately after entry fill)
- Take profit orders (limit orders)
- OCO management (manual since API doesn't support brackets)
- Order tracking and cancellation

Performance Requirements (specs/live-trading-execution.md):
- Order placement round-trip < 500ms
- Market orders execute within 1 second of signal

Reference: specs/live-trading-execution.md
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum

from src.api import (
    TopstepXREST,
    TopstepXWebSocket,
    OrderType,
    OrderSide,
    OrderResponse,
    OrderFill,
)
from src.trading.signal_generator import Signal, SignalType
from src.trading.position_manager import PositionManager, Fill

logger = logging.getLogger(__name__)

# MES Contract Constants
MES_TICK_SIZE = 0.25  # Minimum price movement


class ExecutionStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ExecutionTiming:
    """
    Timing metrics for order execution.

    Performance Requirements:
    - Order placement round-trip < 500ms
    - Signal to fill < 1000ms
    """
    signal_time: Optional[float] = None  # perf_counter when signal received
    order_placed_time: Optional[float] = None  # perf_counter when order placed
    fill_received_time: Optional[float] = None  # perf_counter when fill confirmed

    @property
    def placement_latency_ms(self) -> Optional[float]:
        """Time from signal to order placement (ms). Target: part of <500ms round-trip."""
        if self.signal_time is not None and self.order_placed_time is not None:
            return (self.order_placed_time - self.signal_time) * 1000
        return None

    @property
    def fill_latency_ms(self) -> Optional[float]:
        """Time from order placement to fill (ms)."""
        if self.order_placed_time is not None and self.fill_received_time is not None:
            return (self.fill_received_time - self.order_placed_time) * 1000
        return None

    @property
    def total_latency_ms(self) -> Optional[float]:
        """Total time from signal to fill (ms). Target: <1000ms."""
        if self.signal_time is not None and self.fill_received_time is not None:
            return (self.fill_received_time - self.signal_time) * 1000
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "placement_latency_ms": round(self.placement_latency_ms, 2) if self.placement_latency_ms else None,
            "fill_latency_ms": round(self.fill_latency_ms, 2) if self.fill_latency_ms else None,
            "total_latency_ms": round(self.total_latency_ms, 2) if self.total_latency_ms else None,
        }


@dataclass
class EntryResult:
    """
    Result of an entry order execution.

    Contains the entry fill and associated stop/target orders.
    """
    status: ExecutionStatus
    entry_fill_price: Optional[float] = None
    entry_fill_size: int = 0
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    timing: Optional[ExecutionTiming] = None

    @property
    def success(self) -> bool:
        """Check if entry was successful."""
        return self.status == ExecutionStatus.FILLED and self.entry_fill_price is not None

    @property
    def execution_latency_ms(self) -> Optional[float]:
        """Get total execution latency in milliseconds."""
        if self.timing:
            return self.timing.total_latency_ms
        return None


@dataclass
class ExecutorConfig:
    """Configuration for order executor."""
    # Timeouts
    fill_timeout_seconds: float = 5.0  # Time to wait for fill
    order_timeout_seconds: float = 10.0  # Time to wait for order acknowledgment

    # Entry order type
    use_market_orders: bool = True  # Market (fast) vs limit (price)
    limit_buffer_ticks: int = 1  # Buffer for limit orders

    # Retry settings
    max_retries: int = 2
    retry_delay_seconds: float = 1.0

    # Order tags
    entry_tag: str = "SCALPER_ENTRY"
    stop_tag: str = "SCALPER_STOP"
    target_tag: str = "SCALPER_TARGET"
    flatten_tag: str = "SCALPER_FLATTEN"


class OrderExecutor:
    """
    Executes trading orders via TopstepX API.

    Handles the complete order lifecycle with proper error handling
    and recovery mechanisms.

    Usage:
        executor = OrderExecutor(rest_client, ws_client, position_manager)

        # Execute entry
        result = await executor.execute_entry(
            contract_id="CON.F.US.MES.H26",
            direction=1,  # Long
            size=1,
            stop_ticks=8,
            target_ticks=12,
        )

        # Execute exit
        await executor.execute_exit(contract_id, size)

        # Flatten all
        await executor.flatten_all(contract_id)
    """

    def __init__(
        self,
        rest_client: TopstepXREST,
        ws_client: Optional[TopstepXWebSocket],
        position_manager: PositionManager,
        config: Optional[ExecutorConfig] = None,
    ):
        """
        Initialize order executor.

        Args:
            rest_client: TopstepX REST API client
            ws_client: TopstepX WebSocket client (for fill notifications)
            position_manager: Position manager for state tracking
            config: Executor configuration
        """
        self._rest = rest_client
        self._ws = ws_client
        self._position_manager = position_manager
        self.config = config or ExecutorConfig()

        # Track open orders
        self._open_orders: Dict[str, OrderResponse] = {}
        self._pending_fills: Dict[str, asyncio.Future] = {}
        self._pending_oco_cancellations: set[asyncio.Task] = set()

        # Register WebSocket callbacks if available
        if self._ws:
            self._ws.on_fill(self._handle_fill)

        logger.info("OrderExecutor initialized")

    async def execute_signal(
        self,
        signal: Signal,
        contract_id: str,
        size: int,
        current_price: float,
    ) -> Optional[EntryResult]:
        """
        Execute a trading signal.

        Args:
            signal: Signal to execute
            contract_id: Contract to trade
            size: Number of contracts
            current_price: Current market price for stop/target calculation

        Returns:
            EntryResult for entries, None for exits/holds
        """
        if signal.signal_type == SignalType.HOLD:
            return None

        elif signal.signal_type == SignalType.LONG_ENTRY:
            return await self.execute_entry(
                contract_id=contract_id,
                direction=1,
                size=size,
                stop_ticks=signal.stop_ticks,
                target_ticks=signal.target_ticks,
                current_price=current_price,
            )

        elif signal.signal_type == SignalType.SHORT_ENTRY:
            return await self.execute_entry(
                contract_id=contract_id,
                direction=-1,
                size=size,
                stop_ticks=signal.stop_ticks,
                target_ticks=signal.target_ticks,
                current_price=current_price,
            )

        elif signal.signal_type == SignalType.EXIT_LONG:
            await self.execute_exit(contract_id, size, current_direction=1)
            return None

        elif signal.signal_type == SignalType.EXIT_SHORT:
            await self.execute_exit(contract_id, size, current_direction=-1)
            return None

        elif signal.signal_type == SignalType.REVERSE_TO_LONG:
            await self.execute_exit(contract_id, size, current_direction=-1)
            return await self.execute_entry(
                contract_id=contract_id,
                direction=1,
                size=size,
                stop_ticks=signal.stop_ticks,
                target_ticks=signal.target_ticks,
                current_price=current_price,
            )

        elif signal.signal_type == SignalType.REVERSE_TO_SHORT:
            await self.execute_exit(contract_id, size, current_direction=1)
            return await self.execute_entry(
                contract_id=contract_id,
                direction=-1,
                size=size,
                stop_ticks=signal.stop_ticks,
                target_ticks=signal.target_ticks,
                current_price=current_price,
            )

        elif signal.signal_type == SignalType.FLATTEN:
            await self.flatten_all(contract_id)
            return None

        else:
            logger.warning(f"Unknown signal type: {signal.signal_type}")
            return None

    async def execute_entry(
        self,
        contract_id: str,
        direction: int,
        size: int,
        stop_ticks: float,
        target_ticks: float,
        current_price: Optional[float] = None,
    ) -> EntryResult:
        """
        Execute an entry order with stop loss and take profit.

        Workflow:
        1. Place entry order (market or limit)
        2. Wait for fill confirmation
        3. Place stop loss order
        4. Place take profit order
        5. Track all orders for OCO management

        Performance targets:
        - Order placement round-trip < 500ms
        - Total signal to fill < 1000ms

        Args:
            contract_id: Contract to trade
            direction: 1 for long, -1 for short
            size: Number of contracts
            stop_ticks: Stop loss distance in ticks
            target_ticks: Take profit distance in ticks
            current_price: Current price for limit orders

        Returns:
            EntryResult with fill details, order IDs, and timing metrics
        """
        # Start timing from signal receipt
        timing = ExecutionTiming(signal_time=time.perf_counter())

        side = OrderSide.BUY if direction == 1 else OrderSide.SELL

        logger.info(
            f"Executing entry: {side.name} {size} {contract_id}, "
            f"stop={stop_ticks} ticks, target={target_ticks} ticks"
        )

        try:
            # 1. Place entry order
            entry_order = await self._place_entry_order(
                contract_id, side, size, current_price
            )
            timing.order_placed_time = time.perf_counter()

            # Log placement latency
            placement_ms = timing.placement_latency_ms
            if placement_ms is not None:
                logger.debug(f"Order placement latency: {placement_ms:.2f}ms")
                if placement_ms > 500:
                    logger.warning(
                        f"Order placement exceeded 500ms threshold: {placement_ms:.2f}ms"
                    )

            if entry_order.is_rejected:
                logger.error(f"Entry order rejected: {entry_order.error_message}")
                return EntryResult(
                    status=ExecutionStatus.REJECTED,
                    error_message=entry_order.error_message,
                    timing=timing,
                )

            # 2. Wait for fill
            fill_price = await self._wait_for_fill(entry_order.order_id)
            timing.fill_received_time = time.perf_counter()

            if fill_price is None:
                logger.error(f"Entry order fill timeout: {entry_order.order_id}")
                # Try to cancel the order
                await self._rest.cancel_order(entry_order.order_id)
                return EntryResult(
                    status=ExecutionStatus.FAILED,
                    entry_order_id=entry_order.order_id,
                    error_message="Fill timeout",
                    timing=timing,
                )

            # Log execution timing
            total_ms = timing.total_latency_ms
            if total_ms is not None:
                logger.info(f"Entry filled: {fill_price} (latency: {total_ms:.2f}ms)")
                if total_ms > 1000:
                    logger.warning(
                        f"Order execution exceeded 1000ms threshold: {total_ms:.2f}ms"
                    )
            else:
                logger.info(f"Entry filled: {fill_price}")

            # 3. Calculate stop and target prices
            stop_distance = stop_ticks * MES_TICK_SIZE
            target_distance = target_ticks * MES_TICK_SIZE

            if direction == 1:  # Long
                stop_price = fill_price - stop_distance
                target_price = fill_price + target_distance
            else:  # Short
                stop_price = fill_price + stop_distance
                target_price = fill_price - target_distance

            # 4. Place stop loss order
            stop_order = await self._place_stop_order(
                contract_id=contract_id,
                side=OrderSide.SELL if direction == 1 else OrderSide.BUY,
                size=size,
                stop_price=stop_price,
            )

            # 5. Place take profit order
            target_order = await self._place_target_order(
                contract_id=contract_id,
                side=OrderSide.SELL if direction == 1 else OrderSide.BUY,
                size=size,
                target_price=target_price,
            )

            # 6. Update position manager
            fill = Fill(
                order_id=entry_order.order_id,
                contract_id=contract_id,
                side=side,
                size=size,
                price=fill_price,
                timestamp=datetime.now(),
                is_entry=True,
            )
            self._position_manager.update_from_fill(fill)
            self._position_manager.set_stop_price(stop_price, stop_order.order_id if stop_order else None)
            self._position_manager.set_target_price(target_price, target_order.order_id if target_order else None)

            # 7. Track orders for OCO
            self._track_oco_orders(stop_order, target_order)

            return EntryResult(
                status=ExecutionStatus.FILLED,
                entry_fill_price=fill_price,
                entry_fill_size=size,
                entry_order_id=entry_order.order_id,
                stop_order_id=stop_order.order_id if stop_order else None,
                target_order_id=target_order.order_id if target_order else None,
                timing=timing,
            )

        except Exception as e:
            logger.error(f"Entry execution failed: {e}")
            return EntryResult(
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                timing=timing,
            )

    async def execute_exit(
        self,
        contract_id: str,
        size: int,
        current_direction: int,
    ) -> bool:
        """
        Execute an exit order to close position.

        Args:
            contract_id: Contract to close
            size: Number of contracts to close
            current_direction: Current position direction (1=long, -1=short)

        Returns:
            True if exit successful
        """
        # Exit is opposite of current direction
        side = OrderSide.SELL if current_direction == 1 else OrderSide.BUY

        logger.info(f"Executing exit: {side.name} {size} {contract_id}")

        try:
            # Cancel existing stop/target orders
            await self._cancel_oco_orders()

            # Place market exit order
            exit_order = await self._rest.place_order(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=OrderType.MARKET,
                custom_tag=self.config.flatten_tag,
            )

            # Wait for fill
            fill_price = await self._wait_for_fill(exit_order.order_id)

            if fill_price:
                logger.info(f"Exit filled: {fill_price}")

                # Update position manager
                fill = Fill(
                    order_id=exit_order.order_id,
                    contract_id=contract_id,
                    side=side,
                    size=size,
                    price=fill_price,
                    timestamp=datetime.now(),
                    is_entry=False,
                )
                self._position_manager.update_from_fill(fill)
                return True
            else:
                logger.error("Exit order fill timeout")
                return False

        except Exception as e:
            logger.error(f"Exit execution failed: {e}")
            return False

    async def flatten_all(self, contract_id: str) -> bool:
        """
        Flatten (close) all positions for contract.

        Used for:
        - EOD flatten
        - Emergency exit
        - Manual intervention

        Args:
            contract_id: Contract to flatten

        Returns:
            True if successfully flattened
        """
        logger.warning(f"Flattening all positions: {contract_id}")

        try:
            # Cancel all pending orders first
            await self._cancel_all_orders(contract_id)

            # Use REST flatten endpoint
            result = await self._rest.flatten_position(contract_id)

            if result:
                # Wait for fill confirmation
                fill_price = await self._wait_for_fill(result.order_id)
                if fill_price:
                    logger.info(f"Flatten complete: {fill_price}")
                    self._position_manager.flatten()
                    return True

            # Already flat or no position
            logger.info("No position to flatten or already flat")
            return True

        except Exception as e:
            logger.error(f"Flatten failed: {e}")
            return False

    async def _place_entry_order(
        self,
        contract_id: str,
        side: OrderSide,
        size: int,
        current_price: Optional[float] = None,
    ) -> OrderResponse:
        """Place entry order (market or limit)."""
        if self.config.use_market_orders:
            return await self._rest.place_order(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=OrderType.MARKET,
                custom_tag=self.config.entry_tag,
            )
        else:
            # Limit order with buffer
            if current_price is None:
                raise ValueError("Current price required for limit orders")

            buffer = self.config.limit_buffer_ticks * MES_TICK_SIZE
            if side == OrderSide.BUY:
                limit_price = current_price + buffer  # Pay up for buys
            else:
                limit_price = current_price - buffer  # Accept less for sells

            return await self._rest.place_order(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=OrderType.LIMIT,
                price=limit_price,
                custom_tag=self.config.entry_tag,
            )

    async def _place_stop_order(
        self,
        contract_id: str,
        side: OrderSide,
        size: int,
        stop_price: float,
    ) -> Optional[OrderResponse]:
        """Place stop loss order."""
        try:
            order = await self._rest.place_order(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=OrderType.STOP,
                stop_price=stop_price,
                custom_tag=self.config.stop_tag,
            )
            self._open_orders[order.order_id] = order
            logger.debug(f"Stop order placed: {order.order_id} @ {stop_price}")
            return order
        except Exception as e:
            logger.error(f"Failed to place stop order: {e}")
            return None

    async def _place_target_order(
        self,
        contract_id: str,
        side: OrderSide,
        size: int,
        target_price: float,
    ) -> Optional[OrderResponse]:
        """Place take profit limit order."""
        try:
            order = await self._rest.place_order(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=OrderType.LIMIT,
                price=target_price,
                custom_tag=self.config.target_tag,
            )
            self._open_orders[order.order_id] = order
            logger.debug(f"Target order placed: {order.order_id} @ {target_price}")
            return order
        except Exception as e:
            logger.error(f"Failed to place target order: {e}")
            return None

    async def _wait_for_fill(self, order_id: str) -> Optional[float]:
        """
        Wait for order fill notification.

        Args:
            order_id: Order ID to wait for

        Returns:
            Fill price if filled, None if timeout
        """
        # Create a future for this fill
        fill_future: asyncio.Future = asyncio.Future()
        self._pending_fills[order_id] = fill_future

        try:
            # Wait with timeout
            fill_price = await asyncio.wait_for(
                fill_future,
                timeout=self.config.fill_timeout_seconds
            )
            return fill_price
        except asyncio.TimeoutError:
            logger.warning(f"Fill timeout for order: {order_id}")
            # Check order status via REST as fallback
            try:
                order = await self._rest.get_order(order_id)
                if order.is_filled:
                    return order.avg_fill_price
            except Exception:
                pass
            return None
        finally:
            self._pending_fills.pop(order_id, None)

    def _handle_fill(self, fill: OrderFill) -> None:
        """
        Handle fill notification from WebSocket.

        This is called by the WebSocket client when an order is filled.
        """
        order_id = fill.order_id
        logger.debug(f"Fill received: {order_id} @ {fill.fill_price}")

        # Resolve pending fill future
        if order_id in self._pending_fills:
            future = self._pending_fills[order_id]
            if not future.done():
                future.set_result(fill.fill_price)

        # Handle OCO cancellation
        self._handle_oco_fill(order_id)

        # Update position manager if this is a stop/target fill
        if order_id in self._open_orders:
            order = self._open_orders[order_id]
            if order.custom_tag in (self.config.stop_tag, self.config.target_tag):
                # This is a stop or target being hit - position closed
                fill_obj = Fill(
                    order_id=order_id,
                    contract_id=order.contract_id,
                    side=order.side,
                    size=fill.fill_size,
                    price=fill.fill_price,
                    timestamp=datetime.now(),
                    is_entry=False,
                )
                self._position_manager.update_from_fill(fill_obj)

            del self._open_orders[order_id]

    def _track_oco_orders(
        self,
        stop_order: Optional[OrderResponse],
        target_order: Optional[OrderResponse],
    ) -> None:
        """Track stop and target orders for OCO management."""
        if stop_order:
            self._open_orders[stop_order.order_id] = stop_order
        if target_order:
            self._open_orders[target_order.order_id] = target_order

    def _handle_oco_fill(self, filled_order_id: str) -> None:
        """
        Handle OCO logic - cancel other order when one fills.

        Since TopstepX doesn't support bracket orders, we manually
        cancel the other side when stop or target is hit.

        This schedules an async task to handle cancellation with proper
        timeout and error handling to prevent race conditions.
        """
        if filled_order_id not in self._open_orders:
            return

        filled_order = self._open_orders[filled_order_id]

        # Find the other OCO order to cancel
        if filled_order.custom_tag == self.config.stop_tag:
            # Stop hit - cancel target
            cancel_tag = self.config.target_tag
        elif filled_order.custom_tag == self.config.target_tag:
            # Target hit - cancel stop
            cancel_tag = self.config.stop_tag
        else:
            return

        # Find and cancel the other order
        to_cancel = [
            oid for oid, order in self._open_orders.items()
            if order.custom_tag == cancel_tag and oid != filled_order_id
        ]

        if to_cancel:
            # Schedule async cancellation with timeout and verification
            task = asyncio.create_task(self._cancel_oco_orders_with_timeout(to_cancel))
            # Track the task for potential cleanup
            self._pending_oco_cancellations.add(task)
            task.add_done_callback(lambda t: self._pending_oco_cancellations.discard(t))

    async def _cancel_oco_orders_with_timeout(
        self, order_ids: list[str], timeout: float = 5.0
    ) -> None:
        """
        Cancel OCO orders with timeout and verification.

        This ensures cancellations complete or are retried, preventing
        the race condition where both stop and target could fill.
        """
        cancel_tasks = [self._cancel_order_safe(oid) for oid in order_ids]

        try:
            await asyncio.wait_for(
                asyncio.gather(*cancel_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"OCO cancellation timed out after {timeout}s - "
                f"verifying order states for {order_ids}"
            )
            # Verify which orders are still pending
            await self._verify_oco_cancellation_state(order_ids)

    async def _verify_oco_cancellation_state(self, order_ids: list[str]) -> None:
        """
        Verify and reconcile OCO order states after timeout.

        If cancellations timed out, we need to verify which orders are
        still active and retry cancellation or log critical error.
        """
        for order_id in order_ids:
            if order_id in self._open_orders:
                logger.warning(
                    f"Order {order_id} still in local state after cancellation - "
                    "attempting retry"
                )
                try:
                    # Retry cancellation
                    await asyncio.wait_for(
                        self._cancel_order_safe(order_id),
                        timeout=3.0,
                    )
                except asyncio.TimeoutError:
                    logger.critical(
                        f"CRITICAL: Order {order_id} cancellation failed after retry - "
                        "POSSIBLE DUAL FILL RISK"
                    )
                except Exception as e:
                    logger.error(f"Cancellation retry failed for {order_id}: {e}")

    async def _cancel_order_safe(self, order_id: str) -> None:
        """Cancel order with error handling."""
        try:
            await self._rest.cancel_order(order_id)
            logger.debug(f"Cancelled order: {order_id}")
            self._open_orders.pop(order_id, None)
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")

    async def _cancel_oco_orders(self) -> None:
        """Cancel all open stop/target orders."""
        to_cancel = [
            oid for oid, order in self._open_orders.items()
            if order.custom_tag in (self.config.stop_tag, self.config.target_tag)
        ]

        for order_id in to_cancel:
            await self._cancel_order_safe(order_id)

    async def _cancel_all_orders(self, contract_id: str) -> None:
        """Cancel all pending orders for contract."""
        try:
            count = await self._rest.cancel_all_orders(contract_id)
            logger.info(f"Cancelled {count} orders for {contract_id}")
            self._open_orders.clear()
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")

    def get_open_orders(self) -> Dict[str, OrderResponse]:
        """Get all tracked open orders."""
        return dict(self._open_orders)

    def has_open_orders(self) -> bool:
        """Check if there are open orders."""
        return len(self._open_orders) > 0
