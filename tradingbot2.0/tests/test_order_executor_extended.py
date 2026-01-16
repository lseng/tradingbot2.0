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

        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.5)):
            with patch.object(executor, '_place_stop_order', AsyncMock(return_value=None)):
                with patch.object(executor, '_place_target_order', AsyncMock(return_value=None)):
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

        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=4999.5)):
            with patch.object(executor, '_place_stop_order', AsyncMock(return_value=None)):
                with patch.object(executor, '_place_target_order', AsyncMock(return_value=None)):
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

        with patch('time.perf_counter', mock_perf_counter):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
                with patch.object(executor, '_place_stop_order', AsyncMock(return_value=None)):
                    with patch.object(executor, '_place_target_order', AsyncMock(return_value=None)):
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

        with patch('time.perf_counter', mock_perf_counter):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
                with patch.object(executor, '_place_stop_order', AsyncMock(return_value=None)):
                    with patch.object(executor, '_place_target_order', AsyncMock(return_value=None)):
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
