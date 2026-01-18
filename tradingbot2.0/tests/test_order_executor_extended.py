"""
Extended tests for OrderExecutor covering advanced scenarios.

Tests cover:
- Limit order placement (non-market orders)
- WebSocket fill handling
- OCO order management
- Error handling paths
- Timing and latency warnings
- Cancel operations
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import time

from src.trading.order_executor import (
    ExecutionStatus,
    ExecutionTiming,
    EntryResult,
    ExecutorConfig,
    OrderExecutor,
    MES_TICK_SIZE,
)
from src.trading.signal_generator import Signal, SignalType
from src.api import OrderResponse, OrderStatus, OrderType, OrderSide, OrderFill


# ============================================================================
# ExecutionTiming Tests
# ============================================================================

class TestExecutionTiming:
    """Tests for ExecutionTiming dataclass and properties."""

    def test_timing_default_values(self):
        """Test default values are None."""
        timing = ExecutionTiming()
        assert timing.signal_time is None
        assert timing.order_placed_time is None
        assert timing.fill_received_time is None

    def test_placement_latency_ms_calculated(self):
        """Test placement latency calculation."""
        timing = ExecutionTiming(
            signal_time=100.0,
            order_placed_time=100.5,  # 500ms later
        )
        assert timing.placement_latency_ms == 500.0

    def test_placement_latency_ms_none_when_missing_times(self):
        """Test placement latency returns None with missing times."""
        timing = ExecutionTiming(signal_time=100.0)
        assert timing.placement_latency_ms is None

        timing2 = ExecutionTiming(order_placed_time=100.5)
        assert timing2.placement_latency_ms is None

    def test_fill_latency_ms_calculated(self):
        """Test fill latency calculation."""
        timing = ExecutionTiming(
            order_placed_time=100.0,
            fill_received_time=100.8,  # 800ms later
        )
        assert timing.fill_latency_ms == pytest.approx(800.0, rel=1e-6)

    def test_fill_latency_ms_none_when_missing_times(self):
        """Test fill latency returns None with missing times."""
        timing = ExecutionTiming(order_placed_time=100.0)
        assert timing.fill_latency_ms is None

    def test_total_latency_ms_calculated(self):
        """Test total latency calculation."""
        timing = ExecutionTiming(
            signal_time=100.0,
            fill_received_time=100.9,  # 900ms later
        )
        assert timing.total_latency_ms == pytest.approx(900.0, rel=1e-6)

    def test_total_latency_ms_none_when_missing_times(self):
        """Test total latency returns None with missing times."""
        timing = ExecutionTiming(signal_time=100.0)
        assert timing.total_latency_ms is None

    def test_to_dict_all_values(self):
        """Test to_dict with all latency values."""
        timing = ExecutionTiming(
            signal_time=100.0,
            order_placed_time=100.5,
            fill_received_time=101.0,
        )
        result = timing.to_dict()
        assert result['placement_latency_ms'] == 500.0
        assert result['fill_latency_ms'] == 500.0
        assert result['total_latency_ms'] == 1000.0

    def test_to_dict_none_values(self):
        """Test to_dict when latencies are None."""
        timing = ExecutionTiming()
        result = timing.to_dict()
        assert result['placement_latency_ms'] is None
        assert result['fill_latency_ms'] is None
        assert result['total_latency_ms'] is None


class TestEntryResultExtended:
    """Extended tests for EntryResult."""

    def test_execution_latency_ms_with_timing(self):
        """Test execution_latency_ms property returns total latency."""
        timing = ExecutionTiming(
            signal_time=100.0,
            fill_received_time=100.75,
        )
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.0,
            timing=timing,
        )
        assert result.execution_latency_ms == 750.0

    def test_execution_latency_ms_none_without_timing(self):
        """Test execution_latency_ms returns None without timing."""
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.0,
        )
        assert result.execution_latency_ms is None


# ============================================================================
# Limit Order Tests
# ============================================================================

class TestLimitOrderPlacement:
    """Tests for limit order placement (use_market_orders=False)."""

    @pytest.mark.asyncio
    async def test_place_entry_limit_buy_order(self, mock_rest_client, mock_position_manager):
        """Test placing a limit buy order with buffer."""
        config = ExecutorConfig(
            use_market_orders=False,
            limit_buffer_ticks=2,
        )

        mock_entry_order = OrderResponse(
            order_id="LIMIT001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_entry_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=config,
        )

        # 10C.10 FIX: Mock stop and target orders to return valid responses
        mock_stop_order = OrderResponse(
            order_id="STOP001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.STOP,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_target_order = OrderResponse(
            order_id="TARGET001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.5)):
            with patch.object(executor, '_place_stop_order', AsyncMock(return_value=mock_stop_order)):
                with patch.object(executor, '_place_target_order', AsyncMock(return_value=mock_target_order)):
                    result = await executor.execute_entry(
                        contract_id="CON.F.US.MES.H26",
                        direction=1,
                        size=1,
                        stop_ticks=8,
                        target_ticks=12,
                        current_price=5000.0,
                    )

        # Verify limit order was placed
        mock_rest_client.place_order.assert_called()
        call_args = mock_rest_client.place_order.call_args
        # Buy with buffer: current_price + buffer_ticks * tick_size = 5000.0 + 2 * 0.25 = 5000.5
        assert call_args.kwargs['order_type'] == OrderType.LIMIT

    @pytest.mark.asyncio
    async def test_place_entry_limit_sell_order(self, mock_rest_client, mock_position_manager):
        """Test placing a limit sell order with buffer (short entry)."""
        config = ExecutorConfig(
            use_market_orders=False,
            limit_buffer_ticks=2,
        )

        mock_entry_order = OrderResponse(
            order_id="LIMIT002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_entry_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=config,
        )

        # 10C.10 FIX: Mock stop and target orders to return valid responses
        mock_stop_order = OrderResponse(
            order_id="STOP002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.STOP,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_target_order = OrderResponse(
            order_id="TARGET002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=4999.5)):
            with patch.object(executor, '_place_stop_order', AsyncMock(return_value=mock_stop_order)):
                with patch.object(executor, '_place_target_order', AsyncMock(return_value=mock_target_order)):
                    result = await executor.execute_entry(
                        contract_id="CON.F.US.MES.H26",
                        direction=-1,  # Short
                        size=1,
                        stop_ticks=8,
                        target_ticks=12,
                        current_price=5000.0,
                    )

        # Verify limit order was placed
        mock_rest_client.place_order.assert_called()
        call_args = mock_rest_client.place_order.call_args
        # Sell with buffer: current_price - buffer_ticks * tick_size = 5000.0 - 2 * 0.25 = 4999.5
        assert call_args.kwargs['order_type'] == OrderType.LIMIT

    @pytest.mark.asyncio
    async def test_limit_order_requires_current_price(self, mock_rest_client, mock_position_manager):
        """Test that limit orders require current_price parameter."""
        config = ExecutorConfig(use_market_orders=False)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=config,
        )

        result = await executor.execute_entry(
            contract_id="CON.F.US.MES.H26",
            direction=1,
            size=1,
            stop_ticks=8,
            target_ticks=12,
            current_price=None,  # No price provided
        )

        # Should fail with error
        assert result.status == ExecutionStatus.FAILED
        assert result.error_message is not None


# ============================================================================
# WebSocket Fill Handling Tests
# ============================================================================

class TestWebSocketFillHandling:
    """Tests for WebSocket fill callback handling."""

    @pytest.mark.asyncio
    async def test_handle_fill_resolves_pending_future(self, mock_rest_client, mock_position_manager):
        """Test that _handle_fill resolves pending fill futures."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Create a pending fill future
        fill_future = asyncio.get_event_loop().create_future()
        executor._pending_fills["ORD123"] = fill_future

        # Create fill notification
        fill = OrderFill(
            order_id="ORD123",
            contract_id="CON.F.US.MES.H26",
            side=1,  # Buy
            fill_price=5000.25,
            fill_size=1,
            timestamp=datetime.now(),
        )

        # Handle the fill
        executor._handle_fill(fill)

        # Future should be resolved
        assert fill_future.done()
        assert fill_future.result() == 5000.25

    def test_handle_fill_ignores_nonexistent_order(self, mock_rest_client, mock_position_manager):
        """Test _handle_fill ignores fills for orders not being tracked."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        fill = OrderFill(
            order_id="UNKNOWN",
            contract_id="CON.F.US.MES.H26",
            side=1,  # Buy
            fill_price=5000.25,
            fill_size=1,
            timestamp=datetime.now(),
        )

        # Should not raise exception
        executor._handle_fill(fill)

    def test_handle_fill_updates_position_for_stop_target(self, mock_rest_client, mock_position_manager):
        """Test _handle_fill updates position manager for stop/target fills."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add a stop order to tracking
        mock_stop_order = MagicMock()
        mock_stop_order.order_id = "STOP001"
        mock_stop_order.custom_tag = "SCALPER_STOP"
        mock_stop_order.contract_id = "CON.F.US.MES.H26"
        mock_stop_order.side = OrderSide.SELL
        executor._open_orders["STOP001"] = mock_stop_order

        fill = OrderFill(
            order_id="STOP001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # Sell
            fill_price=4998.0,
            fill_size=1,
            timestamp=datetime.now(),
        )

        executor._handle_fill(fill)

        # Position manager should be updated
        mock_position_manager.update_from_fill.assert_called()

    @pytest.mark.asyncio
    async def test_handle_fill_skips_done_futures(self, mock_rest_client, mock_position_manager):
        """Test _handle_fill skips already done futures."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Create an already done future
        fill_future = asyncio.get_event_loop().create_future()
        fill_future.set_result(5000.0)  # Already done
        executor._pending_fills["ORD123"] = fill_future

        fill = OrderFill(
            order_id="ORD123",
            contract_id="CON.F.US.MES.H26",
            side=1,  # Buy
            fill_price=5001.0,  # Different price
            fill_size=1,
            timestamp=datetime.now(),
        )

        # Should not raise exception
        executor._handle_fill(fill)

        # Original result should be unchanged
        assert fill_future.result() == 5000.0


# ============================================================================
# OCO Order Management Tests
# ============================================================================

class TestOCOOrderManagement:
    """Tests for OCO (one-cancels-other) order management."""

    @pytest.mark.asyncio
    async def test_handle_oco_fill_stop_hit_cancels_target(self, mock_rest_client, mock_position_manager):
        """Test that when stop is hit, target is cancelled."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = "SCALPER_STOP"

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = "SCALPER_TARGET"

        executor._open_orders["STOP001"] = mock_stop
        executor._open_orders["TARGET001"] = mock_target

        # Handle stop fill - should schedule target cancellation
        executor._handle_oco_fill("STOP001")

        # Give the event loop a chance to run the scheduled task
        await asyncio.sleep(0.01)

        # The cancel task should have been created

    @pytest.mark.asyncio
    async def test_handle_oco_fill_target_hit_cancels_stop(self, mock_rest_client, mock_position_manager):
        """Test that when target is hit, stop is cancelled."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = "SCALPER_STOP"

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = "SCALPER_TARGET"

        executor._open_orders["STOP001"] = mock_stop
        executor._open_orders["TARGET001"] = mock_target

        # Handle target fill - should schedule stop cancellation
        executor._handle_oco_fill("TARGET001")

        # Give the event loop a chance to run the scheduled task
        await asyncio.sleep(0.01)

    def test_handle_oco_fill_ignores_unknown_order(self, mock_rest_client, mock_position_manager):
        """Test _handle_oco_fill ignores fills for unknown orders."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Should not raise exception
        executor._handle_oco_fill("UNKNOWN_ORDER")

    def test_handle_oco_fill_ignores_non_stop_target(self, mock_rest_client, mock_position_manager):
        """Test _handle_oco_fill ignores orders that aren't stop/target."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_entry = MagicMock()
        mock_entry.order_id = "ENTRY001"
        mock_entry.custom_tag = "SCALPER_ENTRY"  # Not stop or target

        executor._open_orders["ENTRY001"] = mock_entry

        with patch('asyncio.create_task') as mock_create_task:
            executor._handle_oco_fill("ENTRY001")
            # Should not create cancel task for non-OCO orders
            mock_create_task.assert_not_called()


# ============================================================================
# Cancel Operations Tests
# ============================================================================

class TestCancelOperations:
    """Tests for order cancellation operations."""

    @pytest.mark.asyncio
    async def test_cancel_order_safe_success(self, mock_rest_client, mock_position_manager):
        """Test _cancel_order_safe removes order from tracking on success."""
        mock_rest_client.cancel_order = AsyncMock()

        # 10C.9 FIX: Mock get_order to return cancelled order for verification
        cancelled_order = MagicMock()
        cancelled_order.is_cancelled = True
        cancelled_order.is_filled = False
        mock_rest_client.get_order = AsyncMock(return_value=cancelled_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_order = MagicMock()
        mock_order.order_id = "ORD123"
        executor._open_orders["ORD123"] = mock_order

        await executor._cancel_order_safe("ORD123")

        assert "ORD123" not in executor._open_orders
        mock_rest_client.cancel_order.assert_called_once_with("ORD123")

    @pytest.mark.asyncio
    async def test_cancel_order_safe_handles_error(self, mock_rest_client, mock_position_manager):
        """Test _cancel_order_safe handles API errors gracefully."""
        mock_rest_client.cancel_order = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_order = MagicMock()
        mock_order.order_id = "ORD123"
        executor._open_orders["ORD123"] = mock_order

        # Should not raise exception
        await executor._cancel_order_safe("ORD123")

    @pytest.mark.asyncio
    async def test_cancel_oco_orders(self, mock_rest_client, mock_position_manager):
        """Test _cancel_oco_orders cancels stop and target orders."""
        mock_rest_client.cancel_order = AsyncMock()

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = "SCALPER_STOP"

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = "SCALPER_TARGET"

        mock_entry = MagicMock()
        mock_entry.order_id = "ENTRY001"
        mock_entry.custom_tag = "SCALPER_ENTRY"

        executor._open_orders["STOP001"] = mock_stop
        executor._open_orders["TARGET001"] = mock_target
        executor._open_orders["ENTRY001"] = mock_entry

        await executor._cancel_oco_orders()

        # Should have cancelled stop and target but not entry
        assert mock_rest_client.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_cancel_all_orders_success(self, mock_rest_client, mock_position_manager):
        """Test _cancel_all_orders clears all tracked orders."""
        mock_rest_client.cancel_all_orders = AsyncMock(return_value=3)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add some orders
        executor._open_orders["ORD1"] = MagicMock()
        executor._open_orders["ORD2"] = MagicMock()

        await executor._cancel_all_orders("CON.F.US.MES.H26")

        assert len(executor._open_orders) == 0

    @pytest.mark.asyncio
    async def test_cancel_all_orders_handles_error(self, mock_rest_client, mock_position_manager):
        """Test _cancel_all_orders handles API errors."""
        mock_rest_client.cancel_all_orders = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        executor._open_orders["ORD1"] = MagicMock()

        # Should not raise exception
        await executor._cancel_all_orders("CON.F.US.MES.H26")


# ============================================================================
# Wait for Fill Tests
# ============================================================================

class TestWaitForFill:
    """Tests for _wait_for_fill method."""

    @pytest.mark.asyncio
    async def test_wait_for_fill_returns_price_on_success(self, mock_rest_client, mock_position_manager):
        """Test _wait_for_fill returns fill price when future resolves."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=ExecutorConfig(fill_timeout_seconds=1.0),
        )

        async def resolve_fill():
            await asyncio.sleep(0.1)
            executor._pending_fills["ORD123"].set_result(5000.25)

        asyncio.create_task(resolve_fill())

        result = await executor._wait_for_fill("ORD123")
        assert result == 5000.25

    @pytest.mark.asyncio
    async def test_wait_for_fill_timeout_checks_rest(self, mock_rest_client, mock_position_manager):
        """Test _wait_for_fill falls back to REST on timeout."""
        # Mock REST to return filled order
        mock_order = MagicMock()
        mock_order.is_filled = True
        mock_order.avg_fill_price = 5000.50
        mock_rest_client.get_order = AsyncMock(return_value=mock_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=ExecutorConfig(fill_timeout_seconds=0.1),  # Very short timeout
        )

        result = await executor._wait_for_fill("ORD123")

        # Should fall back to REST and get fill price
        assert result == 5000.50
        mock_rest_client.get_order.assert_called_once_with("ORD123")

    @pytest.mark.asyncio
    async def test_wait_for_fill_timeout_rest_error(self, mock_rest_client, mock_position_manager):
        """Test _wait_for_fill returns None when REST fallback also fails."""
        mock_rest_client.get_order = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=ExecutorConfig(fill_timeout_seconds=0.1),
        )

        result = await executor._wait_for_fill("ORD123")
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_fill_timeout_order_not_filled(self, mock_rest_client, mock_position_manager):
        """Test _wait_for_fill returns None when order not filled via REST."""
        mock_order = MagicMock()
        mock_order.is_filled = False
        mock_rest_client.get_order = AsyncMock(return_value=mock_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=ExecutorConfig(fill_timeout_seconds=0.1),
        )

        result = await executor._wait_for_fill("ORD123")
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_fill_cleans_up_pending_fills(self, mock_rest_client, mock_position_manager):
        """Test _wait_for_fill removes order from pending_fills after completion."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=ExecutorConfig(fill_timeout_seconds=0.1),
        )

        mock_rest_client.get_order = AsyncMock(side_effect=Exception("Error"))

        await executor._wait_for_fill("ORD123")

        # Should be cleaned up
        assert "ORD123" not in executor._pending_fills


# ============================================================================
# Stop/Target Order Placement Error Tests
# ============================================================================

class TestStopTargetPlacementErrors:
    """Tests for error handling in stop/target order placement."""

    @pytest.mark.asyncio
    async def test_place_stop_order_handles_error(self, mock_rest_client, mock_position_manager):
        """Test _place_stop_order returns None on API error."""
        mock_rest_client.place_order = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        result = await executor._place_stop_order(
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.SELL,
            size=1,
            stop_price=4998.0,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_place_target_order_handles_error(self, mock_rest_client, mock_position_manager):
        """Test _place_target_order returns None on API error."""
        mock_rest_client.place_order = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        result = await executor._place_target_order(
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.SELL,
            size=1,
            target_price=5003.0,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_place_stop_order_success_tracks_order(self, mock_rest_client, mock_position_manager):
        """Test _place_stop_order tracks order on success."""
        mock_order = OrderResponse(
            order_id="STOP001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.STOP,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_rest_client.place_order = AsyncMock(return_value=mock_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        result = await executor._place_stop_order(
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.SELL,
            size=1,
            stop_price=4998.0,
        )

        assert result.order_id == "STOP001"
        assert "STOP001" in executor._open_orders

    @pytest.mark.asyncio
    async def test_place_target_order_success_tracks_order(self, mock_rest_client, mock_position_manager):
        """Test _place_target_order tracks order on success."""
        mock_order = OrderResponse(
            order_id="TARGET001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_rest_client.place_order = AsyncMock(return_value=mock_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        result = await executor._place_target_order(
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.SELL,
            size=1,
            target_price=5003.0,
        )

        assert result.order_id == "TARGET001"
        assert "TARGET001" in executor._open_orders


# ============================================================================
# Exit Execution Error Tests
# ============================================================================

class TestExitExecutionErrors:
    """Tests for error handling in exit execution."""

    @pytest.mark.asyncio
    async def test_execute_exit_fill_timeout(self, mock_rest_client, mock_position_manager):
        """Test execute_exit returns False on fill timeout."""
        mock_exit_order = OrderResponse(
            order_id="EXIT001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_exit_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_oco_orders', AsyncMock()):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=None)):
                result = await executor.execute_exit(
                    contract_id="CON.F.US.MES.H26",
                    size=1,
                    current_direction=1,
                )

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_exit_api_error(self, mock_rest_client, mock_position_manager):
        """Test execute_exit returns False on API error."""
        mock_rest_client.place_order = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_oco_orders', AsyncMock()):
            result = await executor.execute_exit(
                contract_id="CON.F.US.MES.H26",
                size=1,
                current_direction=1,
            )

        assert result is False


# ============================================================================
# Flatten All Error Tests
# ============================================================================

class TestFlattenAllErrors:
    """Tests for error handling in flatten_all."""

    @pytest.mark.asyncio
    async def test_flatten_all_no_result(self, mock_rest_client, mock_position_manager):
        """Test flatten_all handles case when flatten returns no result."""
        mock_rest_client.flatten_position = AsyncMock(return_value=None)
        mock_rest_client.cancel_all_orders = AsyncMock(return_value=0)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_all_orders', AsyncMock()):
            result = await executor.flatten_all("CON.F.US.MES.H26")

        # Should return True (already flat)
        assert result is True

    @pytest.mark.asyncio
    async def test_flatten_all_api_error(self, mock_rest_client, mock_position_manager):
        """Test flatten_all returns False on API error."""
        mock_rest_client.flatten_position = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_all_orders', AsyncMock()):
            result = await executor.flatten_all("CON.F.US.MES.H26")

        assert result is False

    @pytest.mark.asyncio
    async def test_flatten_all_fill_timeout(self, mock_rest_client, mock_position_manager):
        """Test flatten_all handles fill timeout."""
        mock_flatten_order = OrderResponse(
            order_id="FLAT001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_rest_client.flatten_position = AsyncMock(return_value=mock_flatten_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_all_orders', AsyncMock()):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=None)):
                result = await executor.flatten_all("CON.F.US.MES.H26")

        # Returns True because no position to flatten
        assert result is True


# ============================================================================
# Unknown Signal Type Tests
# ============================================================================

class TestUnknownSignalType:
    """Tests for handling unknown signal types."""

    @pytest.mark.asyncio
    async def test_execute_signal_unknown_type(self, mock_rest_client, mock_position_manager):
        """Test execute_signal handles unknown signal types gracefully."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Create a mock signal with an unknown type
        signal = MagicMock()
        signal.signal_type = MagicMock()
        signal.signal_type.value = "unknown_type"
        signal.signal_type.__eq__ = lambda self, other: False  # Never equal

        result = await executor.execute_signal(
            signal=signal,
            contract_id="CON.F.US.MES.H26",
            size=1,
            current_price=5000.0,
        )

        # Should return None for unknown signal types
        assert result is None


# ============================================================================
# Latency Warning Tests
# ============================================================================

class TestLatencyWarnings:
    """Tests for latency threshold warnings."""

    @pytest.mark.asyncio
    async def test_execute_entry_logs_placement_warning_over_500ms(
        self, mock_rest_client, mock_position_manager
    ):
        """Test that placement latency > 500ms logs a warning."""
        mock_entry_order = OrderResponse(
            order_id="ENTRY001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_entry_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Simulate slow order placement (>500ms)
        original_perf_counter = time.perf_counter
        call_count = [0]

        def mock_perf_counter():
            call_count[0] += 1
            if call_count[0] == 1:
                return 100.0  # Signal time
            elif call_count[0] == 2:
                return 100.6  # Order placed (600ms later)
            else:
                return 100.8  # Fill received

        # 10C.10 FIX: Mock stop and target orders to return valid responses
        mock_stop_order = OrderResponse(
            order_id="STOP001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.STOP,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_target_order = OrderResponse(
            order_id="TARGET001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        with patch('time.perf_counter', mock_perf_counter):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
                with patch.object(executor, '_place_stop_order', AsyncMock(return_value=mock_stop_order)):
                    with patch.object(executor, '_place_target_order', AsyncMock(return_value=mock_target_order)):
                        result = await executor.execute_entry(
                            contract_id="CON.F.US.MES.H26",
                            direction=1,
                            size=1,
                            stop_ticks=8,
                            target_ticks=12,
                            current_price=5000.0,
                        )

        # Entry should still succeed
        assert result.status == ExecutionStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_entry_logs_total_warning_over_1000ms(
        self, mock_rest_client, mock_position_manager
    ):
        """Test that total latency > 1000ms logs a warning."""
        mock_entry_order = OrderResponse(
            order_id="ENTRY001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_entry_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Simulate slow total execution (>1000ms)
        call_count = [0]

        def mock_perf_counter():
            call_count[0] += 1
            if call_count[0] == 1:
                return 100.0  # Signal time
            elif call_count[0] == 2:
                return 100.3  # Order placed (300ms)
            else:
                return 101.5  # Fill received (1500ms total)

        # 10C.10 FIX: Mock stop and target orders to return valid responses
        mock_stop_order = OrderResponse(
            order_id="STOP002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.STOP,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )
        mock_target_order = OrderResponse(
            order_id="TARGET002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.LIMIT,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        with patch('time.perf_counter', mock_perf_counter):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
                with patch.object(executor, '_place_stop_order', AsyncMock(return_value=mock_stop_order)):
                    with patch.object(executor, '_place_target_order', AsyncMock(return_value=mock_target_order)):
                        result = await executor.execute_entry(
                            contract_id="CON.F.US.MES.H26",
                            direction=1,
                            size=1,
                            stop_ticks=8,
                            target_ticks=12,
                            current_price=5000.0,
                        )

        assert result.status == ExecutionStatus.FILLED


# ============================================================================
# Track OCO Orders Tests
# ============================================================================

class TestTrackOCOOrders:
    """Tests for _track_oco_orders method."""

    def test_track_both_orders(self, mock_rest_client, mock_position_manager):
        """Test _track_oco_orders adds both stop and target."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"

        executor._track_oco_orders(mock_stop, mock_target)

        assert "STOP001" in executor._open_orders
        assert "TARGET001" in executor._open_orders

    def test_track_only_stop(self, mock_rest_client, mock_position_manager):
        """Test _track_oco_orders handles None target."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"

        executor._track_oco_orders(mock_stop, None)

        assert "STOP001" in executor._open_orders
        assert len(executor._open_orders) == 1

    def test_track_only_target(self, mock_rest_client, mock_position_manager):
        """Test _track_oco_orders handles None stop."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"

        executor._track_oco_orders(None, mock_target)

        assert "TARGET001" in executor._open_orders
        assert len(executor._open_orders) == 1

    def test_track_neither(self, mock_rest_client, mock_position_manager):
        """Test _track_oco_orders handles both None."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        executor._track_oco_orders(None, None)

        assert len(executor._open_orders) == 0


# ============================================================================
# WebSocket Registration Tests
# ============================================================================

class TestWebSocketRegistration:
    """Tests for WebSocket callback registration."""

    def test_ws_callback_registered_on_init(self, mock_rest_client, mock_ws_client, mock_position_manager):
        """Test that fill callback is registered with WebSocket client."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=mock_ws_client,
            position_manager=mock_position_manager,
        )

        mock_ws_client.on_fill.assert_called_once()

    def test_no_ws_callback_without_ws_client(self, mock_rest_client, mock_position_manager):
        """Test that no callback registration occurs without WS client."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Should not raise exception
        assert executor._ws is None


# ============================================================================
# OCO Race Condition Tests (10.23 FIX)
# ============================================================================

class TestOCORaceConditionPrevention:
    """
    Tests for the OCO (one-cancels-other) race condition fix (10.23).

    The race condition occurs when both stop and target orders fill before
    our cancellation can complete. Without proper handling, this would cause:
    1. Position doubling (both exits processed)
    2. Duplicate position manager updates
    3. Orphaned tracking state

    The fix uses:
    - threading.Lock for thread-safe access to shared state
    - _filled_oco_orders set to track which OCO orders have filled
    - _orders_being_cancelled set to prevent duplicate cancellation scheduling
    - Early return when dual fill is detected to prevent double position updates
    """

    def test_filled_oco_orders_tracking_initialized(self, mock_rest_client, mock_position_manager):
        """Test that _filled_oco_orders tracking set is initialized."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )
        assert hasattr(executor, '_filled_oco_orders')
        assert isinstance(executor._filled_oco_orders, set)
        assert len(executor._filled_oco_orders) == 0

    def test_orders_being_cancelled_tracking_initialized(self, mock_rest_client, mock_position_manager):
        """Test that _orders_being_cancelled tracking set is initialized."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )
        assert hasattr(executor, '_orders_being_cancelled')
        assert isinstance(executor._orders_being_cancelled, set)
        assert len(executor._orders_being_cancelled) == 0

    def test_order_lock_is_threading_lock(self, mock_rest_client, mock_position_manager):
        """Test that _order_lock is a threading.Lock (not asyncio.Lock).

        This is important because _handle_fill is a synchronous callback
        and asyncio.Lock cannot be used in sync context.
        """
        import threading

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )
        assert hasattr(executor, '_order_lock')
        assert isinstance(executor._order_lock, type(threading.Lock()))

    def test_handle_fill_marks_oco_order_as_filled(self, mock_rest_client, mock_position_manager):
        """Test that _handle_fill marks OCO orders in _filled_oco_orders set."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add a stop order to tracking
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        mock_stop.contract_id = "CON.F.US.MES.H26"
        mock_stop.side = OrderSide.SELL
        executor._open_orders["STOP001"] = mock_stop

        # Simulate fill notification
        fill = OrderFill(
            order_id="STOP001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            fill_price=5000.25,
            fill_size=1,
            timestamp=datetime.now(),
        )
        executor._handle_fill(fill)

        # Order should be marked as filled
        assert "STOP001" in executor._filled_oco_orders

    @pytest.mark.asyncio
    async def test_handle_fill_detects_dual_fill(self, mock_rest_client, mock_position_manager):
        """Test that dual fill is detected when both OCO orders fill.

        This simulates the race condition where both stop and target fill
        before the cancellation can complete.
        """
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add both stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        mock_stop.contract_id = "CON.F.US.MES.H26"
        mock_stop.side = OrderSide.SELL
        executor._open_orders["STOP001"] = mock_stop

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = executor.config.target_tag
        mock_target.contract_id = "CON.F.US.MES.H26"
        mock_target.side = OrderSide.SELL
        executor._open_orders["TARGET001"] = mock_target

        # First fill: stop
        fill1 = OrderFill(
            order_id="STOP001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            fill_price=5000.25,
            fill_size=1,
            timestamp=datetime.now(),
        )
        executor._handle_fill(fill1)
        await asyncio.sleep(0.01)  # Let cancellation task start

        # STOP001 should now be in _filled_oco_orders
        assert "STOP001" in executor._filled_oco_orders

        # Second fill: target (dual fill scenario - target fills before cancellation completes)
        fill2 = OrderFill(
            order_id="TARGET001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            fill_price=5010.00,
            fill_size=1,
            timestamp=datetime.now(),
        )

        # Capture log output to verify CRITICAL log
        with patch('src.trading.order_executor.logger') as mock_logger:
            executor._handle_fill(fill2)
            # Should log CRITICAL for dual fill because STOP001 is already filled
            mock_logger.critical.assert_called()
            call_args = str(mock_logger.critical.call_args)
            assert "DUAL FILL DETECTED" in call_args

        # Clean up pending tasks
        for task in executor._pending_oco_cancellations:
            task.cancel()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_handle_fill_skips_position_update_on_dual_fill(self, mock_rest_client, mock_position_manager):
        """Test that position manager is NOT updated twice on dual fill.

        This is critical - without this fix, both fills would call
        update_from_fill, causing position to be closed twice.
        """
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add both stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        mock_stop.contract_id = "CON.F.US.MES.H26"
        mock_stop.side = OrderSide.SELL
        executor._open_orders["STOP001"] = mock_stop

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = executor.config.target_tag
        mock_target.contract_id = "CON.F.US.MES.H26"
        mock_target.side = OrderSide.SELL
        executor._open_orders["TARGET001"] = mock_target

        # First fill: stop
        fill1 = OrderFill(
            order_id="STOP001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            fill_price=5000.25,
            fill_size=1,
            timestamp=datetime.now(),
        )
        executor._handle_fill(fill1)
        await asyncio.sleep(0.01)  # Let cancellation task start

        # Position manager should be called once
        assert mock_position_manager.update_from_fill.call_count == 1

        # Simulate second fill coming in before cancellation completes
        # This is the race condition scenario
        fill2 = OrderFill(
            order_id="TARGET001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            fill_price=5010.00,
            fill_size=1,
            timestamp=datetime.now(),
        )
        executor._handle_fill(fill2)

        # Position manager should NOT be called again (dual fill prevention)
        # Because target was detected as dual fill, it should skip update
        assert mock_position_manager.update_from_fill.call_count == 1

        # Clean up pending tasks
        for task in executor._pending_oco_cancellations:
            task.cancel()
        await asyncio.sleep(0.01)

    def test_handle_fill_ignores_duplicate_fill_notification(self, mock_rest_client, mock_position_manager):
        """Test that duplicate fill notifications for same order are ignored."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add stop order
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        mock_stop.contract_id = "CON.F.US.MES.H26"
        mock_stop.side = OrderSide.SELL
        executor._open_orders["STOP001"] = mock_stop

        # First fill
        fill = OrderFill(
            order_id="STOP001",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            fill_price=5000.25,
            fill_size=1,
            timestamp=datetime.now(),
        )
        executor._handle_fill(fill)
        assert mock_position_manager.update_from_fill.call_count == 1

        # Duplicate fill notification (same order ID)
        executor._handle_fill(fill)

        # Position manager should not be called again
        assert mock_position_manager.update_from_fill.call_count == 1

    def test_track_oco_orders_clears_tracking_state(self, mock_rest_client, mock_position_manager):
        """Test that _track_oco_orders clears previous tracking state.

        When entering a new position, old OCO tracking must be cleared
        to prevent stale state from affecting new orders.
        """
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Simulate state from previous position
        executor._filled_oco_orders.add("OLD_STOP")
        executor._orders_being_cancelled.add("OLD_TARGET")

        # Track new OCO orders
        mock_stop = MagicMock()
        mock_stop.order_id = "NEW_STOP"
        mock_target = MagicMock()
        mock_target.order_id = "NEW_TARGET"

        executor._track_oco_orders(mock_stop, mock_target)

        # Old state should be cleared
        assert "OLD_STOP" not in executor._filled_oco_orders
        assert "OLD_TARGET" not in executor._orders_being_cancelled
        assert len(executor._filled_oco_orders) == 0
        assert len(executor._orders_being_cancelled) == 0

    @pytest.mark.asyncio
    async def test_handle_oco_fill_marks_order_being_cancelled(self, mock_rest_client, mock_position_manager):
        """Test that _handle_oco_fill marks orders as being cancelled.

        This prevents duplicate cancellation tasks from being scheduled
        if multiple fills come in quickly.
        """
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add both stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        executor._open_orders["STOP001"] = mock_stop

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = executor.config.target_tag
        executor._open_orders["TARGET001"] = mock_target

        # Call _handle_oco_fill as if stop was filled (needs event loop for create_task)
        executor._handle_oco_fill("STOP001")

        # Target should be marked as being cancelled
        assert "TARGET001" in executor._orders_being_cancelled

        # Clean up pending tasks
        for task in executor._pending_oco_cancellations:
            task.cancel()
        await asyncio.sleep(0.01)

    def test_handle_oco_fill_skips_already_being_cancelled(self, mock_rest_client, mock_position_manager):
        """Test that _handle_oco_fill doesn't schedule duplicate cancellation."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add both stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        executor._open_orders["STOP001"] = mock_stop

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = executor.config.target_tag
        executor._open_orders["TARGET001"] = mock_target

        # Pre-mark target as being cancelled
        executor._orders_being_cancelled.add("TARGET001")

        # Count pending tasks before
        tasks_before = len(executor._pending_oco_cancellations)

        # Call _handle_oco_fill - should NOT schedule another cancellation
        executor._handle_oco_fill("STOP001")

        # No new tasks should have been scheduled
        assert len(executor._pending_oco_cancellations) == tasks_before

    def test_handle_oco_fill_skips_already_filled_orders(self, mock_rest_client, mock_position_manager):
        """Test that _handle_oco_fill doesn't try to cancel already filled orders."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add both stop and target orders
        mock_stop = MagicMock()
        mock_stop.order_id = "STOP001"
        mock_stop.custom_tag = executor.config.stop_tag
        executor._open_orders["STOP001"] = mock_stop

        mock_target = MagicMock()
        mock_target.order_id = "TARGET001"
        mock_target.custom_tag = executor.config.target_tag
        executor._open_orders["TARGET001"] = mock_target

        # Pre-mark target as filled (simulating it already filled)
        executor._filled_oco_orders.add("TARGET001")

        # Count pending tasks before
        tasks_before = len(executor._pending_oco_cancellations)

        # Call _handle_oco_fill - should NOT schedule cancellation for filled order
        executor._handle_oco_fill("STOP001")

        # No new tasks should have been scheduled
        assert len(executor._pending_oco_cancellations) == tasks_before

    def test_get_open_orders_uses_lock(self, mock_rest_client, mock_position_manager):
        """Test that get_open_orders acquires lock for thread-safe access."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add an order
        mock_order = MagicMock()
        mock_order.order_id = "ORD001"
        executor._open_orders["ORD001"] = mock_order

        # Verify we get a copy (thread-safe snapshot)
        result = executor.get_open_orders()
        assert "ORD001" in result

        # Verify it's a copy, not the original
        result["NEW_ORDER"] = MagicMock()
        assert "NEW_ORDER" not in executor._open_orders

    def test_has_open_orders_uses_lock(self, mock_rest_client, mock_position_manager):
        """Test that has_open_orders acquires lock for thread-safe access."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Initially no orders
        assert executor.has_open_orders() is False

        # Add an order
        with executor._order_lock:
            mock_order = MagicMock()
            executor._open_orders["ORD001"] = mock_order

        # Should see the order
        assert executor.has_open_orders() is True

    @pytest.mark.asyncio
    async def test_cancel_all_orders_clears_tracking_state(self, mock_rest_client, mock_position_manager):
        """Test that _cancel_all_orders clears all tracking state."""
        mock_rest_client.cancel_all_orders = AsyncMock(return_value=2)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Populate tracking state
        executor._open_orders["ORD001"] = MagicMock()
        executor._open_orders["ORD002"] = MagicMock()
        executor._filled_oco_orders.add("FILLED001")
        executor._orders_being_cancelled.add("CANCEL001")

        await executor._cancel_all_orders("CON.F.US.MES.H26")

        # All state should be cleared
        assert len(executor._open_orders) == 0
        assert len(executor._filled_oco_orders) == 0
        assert len(executor._orders_being_cancelled) == 0

    @pytest.mark.asyncio
    async def test_cancel_order_safe_uses_lock(self, mock_rest_client, mock_position_manager):
        """Test that _cancel_order_safe acquires lock when modifying state."""
        mock_rest_client.cancel_order = AsyncMock()
        mock_order = MagicMock()
        mock_order.is_cancelled = True
        mock_order.is_filled = False  # Important: must set both attributes
        mock_rest_client.get_order = AsyncMock(return_value=mock_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add order
        executor._open_orders["ORD001"] = MagicMock()

        # Cancel should remove from tracking
        result = await executor._cancel_order_safe("ORD001")

        assert result is True
        assert "ORD001" not in executor._open_orders

    @pytest.mark.asyncio
    async def test_cancel_order_safe_tracks_filled_orders(self, mock_rest_client, mock_position_manager):
        """Test that _cancel_order_safe adds to _filled_oco_orders when order filled before cancel."""
        mock_rest_client.cancel_order = AsyncMock()
        mock_order = MagicMock()
        mock_order.is_filled = True  # Order filled before cancel completed
        mock_order.is_cancelled = False
        mock_rest_client.get_order = AsyncMock(return_value=mock_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add order
        executor._open_orders["ORD001"] = MagicMock()

        # Cancel should detect the order was filled
        result = await executor._cancel_order_safe("ORD001")

        assert result is False  # Cancel wasn't successful (order filled instead)
        assert "ORD001" in executor._filled_oco_orders
        assert "ORD001" not in executor._open_orders


# ============================================================================
# EntryResult.requires_halt Tests (1.16 FIX)
# ============================================================================

class TestEntryResultRequiresHalt:
    """Tests for EntryResult.requires_halt field (1.16 FIX)."""

    def test_requires_halt_default_false(self):
        """Test that requires_halt defaults to False."""
        result = EntryResult(status=ExecutionStatus.FILLED)
        assert result.requires_halt is False

    def test_requires_halt_can_be_set_true(self):
        """Test that requires_halt can be set to True."""
        result = EntryResult(
            status=ExecutionStatus.FAILED,
            error_message="Unprotected position",
            requires_halt=True,
        )
        assert result.requires_halt is True

    def test_requires_halt_on_successful_entry(self):
        """Test successful entry has requires_halt=False."""
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.0,
            entry_fill_size=1,
            stop_order_id="STOP001",
            target_order_id="TARGET001",
        )
        assert result.requires_halt is False
        assert result.success is True

    def test_requires_halt_unprotected_position(self):
        """Test entry with unprotected position has requires_halt=True."""
        result = EntryResult(
            status=ExecutionStatus.FAILED,
            entry_fill_price=5000.0,
            entry_fill_size=1,
            error_message="CRITICAL: Unprotected position - emergency exit failed",
            requires_halt=True,
        )
        assert result.requires_halt is True
        assert result.success is False
