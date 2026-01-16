"""
Unit tests for OrderExecutor and related components.

Tests cover:
- ExecutionStatus enum
- EntryResult dataclass
- ExecutorConfig dataclass
- OrderExecutor initialization
- Signal dispatch logic
- Order placement helpers
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from enum import Enum

from src.trading.order_executor import (
    ExecutionStatus,
    EntryResult,
    ExecutorConfig,
    OrderExecutor,
    MES_TICK_SIZE,
)
from src.trading.signal_generator import Signal, SignalType


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert hasattr(ExecutionStatus, 'PENDING')
        assert hasattr(ExecutionStatus, 'SUBMITTED')
        assert hasattr(ExecutionStatus, 'FILLED')
        assert hasattr(ExecutionStatus, 'PARTIALLY_FILLED')
        assert hasattr(ExecutionStatus, 'CANCELLED')
        assert hasattr(ExecutionStatus, 'REJECTED')
        assert hasattr(ExecutionStatus, 'FAILED')

    def test_status_values(self):
        """Test status string values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.FILLED.value == "filled"
        assert ExecutionStatus.CANCELLED.value == "cancelled"


class TestEntryResult:
    """Tests for EntryResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = EntryResult(status=ExecutionStatus.PENDING)

        assert result.status == ExecutionStatus.PENDING
        assert result.entry_fill_price is None
        assert result.entry_fill_size == 0
        assert result.entry_order_id is None
        assert result.stop_order_id is None
        assert result.target_order_id is None
        assert result.error_message is None

    def test_success_property_true(self):
        """Test success property when filled."""
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.0,
        )

        assert result.success is True

    def test_success_property_false_not_filled(self):
        """Test success property when not filled."""
        result = EntryResult(
            status=ExecutionStatus.PENDING,
            entry_fill_price=5000.0,
        )

        assert result.success is False

    def test_success_property_false_no_price(self):
        """Test success property when no fill price."""
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=None,
        )

        assert result.success is False

    def test_timestamp_auto_populated(self):
        """Test that timestamp is auto-populated."""
        result = EntryResult(status=ExecutionStatus.PENDING)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_full_result(self):
        """Test fully populated result."""
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.25,
            entry_fill_size=2,
            entry_order_id="ORD123",
            stop_order_id="ORD124",
            target_order_id="ORD125",
        )

        assert result.success is True
        assert result.entry_fill_price == 5000.25
        assert result.entry_fill_size == 2
        assert result.entry_order_id == "ORD123"
        assert result.stop_order_id == "ORD124"
        assert result.target_order_id == "ORD125"


class TestExecutorConfig:
    """Tests for ExecutorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExecutorConfig()

        assert config.fill_timeout_seconds == 5.0
        assert config.order_timeout_seconds == 10.0
        assert config.use_market_orders is True
        assert config.limit_buffer_ticks == 1
        assert config.max_retries == 2
        assert config.retry_delay_seconds == 1.0

    def test_default_tags(self):
        """Test default order tags."""
        config = ExecutorConfig()

        assert config.entry_tag == "SCALPER_ENTRY"
        assert config.stop_tag == "SCALPER_STOP"
        assert config.target_tag == "SCALPER_TARGET"
        assert config.flatten_tag == "SCALPER_FLATTEN"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExecutorConfig(
            fill_timeout_seconds=10.0,
            use_market_orders=False,
            max_retries=5,
        )

        assert config.fill_timeout_seconds == 10.0
        assert config.use_market_orders is False
        assert config.max_retries == 5


class TestMESConstants:
    """Tests for MES contract constants."""

    def test_tick_size(self):
        """Test MES tick size."""
        assert MES_TICK_SIZE == 0.25


class TestOrderExecutorInit:
    """Tests for OrderExecutor initialization."""

    def test_init_with_required_args(self, mock_rest_client, mock_position_manager):
        """Test initialization with required arguments."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        assert executor._rest == mock_rest_client
        assert executor._ws is None
        assert executor._position_manager == mock_position_manager

    def test_init_with_config(self, mock_rest_client, mock_position_manager):
        """Test initialization with custom config."""
        config = ExecutorConfig(fill_timeout_seconds=10.0)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
            config=config,
        )

        assert executor.config.fill_timeout_seconds == 10.0

    def test_init_default_config(self, mock_rest_client, mock_position_manager):
        """Test initialization creates default config."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        assert executor.config is not None
        assert isinstance(executor.config, ExecutorConfig)

    def test_init_open_orders_empty(self, mock_rest_client, mock_position_manager):
        """Test that open orders dict is initialized empty."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        assert executor._open_orders == {}

    def test_init_with_ws_client(self, mock_rest_client, mock_ws_client, mock_position_manager):
        """Test initialization with WebSocket client."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=mock_ws_client,
            position_manager=mock_position_manager,
        )

        assert executor._ws == mock_ws_client


class TestOrderExecutorSignalDispatch:
    """Tests for signal dispatch logic."""

    @pytest.mark.asyncio
    async def test_execute_signal_hold(self, mock_rest_client, mock_position_manager):
        """Test that HOLD signal does nothing."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            predicted_class=1,  # FLAT
            timestamp=datetime.now(),
        )

        result = await executor.execute_signal(
            signal=signal,
            contract_id="CON.F.US.MES.H26",
            size=1,
            current_price=5000.0,
        )

        # HOLD should return None or success with no action
        assert result is None or result.status == ExecutionStatus.FILLED


class TestOrderExecutorOpenOrders:
    """Tests for open orders tracking."""

    def test_get_open_orders_empty(self, mock_rest_client, mock_position_manager):
        """Test getting open orders when empty."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        orders = executor.get_open_orders()

        assert isinstance(orders, dict)
        assert len(orders) == 0

    def test_has_open_orders_false(self, mock_rest_client, mock_position_manager):
        """Test has_open_orders when empty."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        assert executor.has_open_orders() is False


class TestSignalTypes:
    """Tests for different signal types."""

    def test_signal_type_values(self):
        """Test signal type enum values."""
        assert SignalType.LONG_ENTRY.value == "long_entry"
        assert SignalType.SHORT_ENTRY.value == "short_entry"
        assert SignalType.EXIT_LONG.value == "exit_long"
        assert SignalType.EXIT_SHORT.value == "exit_short"
        assert SignalType.FLATTEN.value == "flatten"
        assert SignalType.HOLD.value == "hold"


class TestSignalCreation:
    """Tests for creating Signal objects."""

    def test_signal_for_long_entry(self):
        """Test creating a long entry signal."""
        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.75,
            stop_ticks=8,
            target_ticks=12,
            predicted_class=2,  # UP
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.LONG_ENTRY
        assert signal.predicted_class == 2  # UP
        assert signal.confidence == 0.75
        assert signal.stop_ticks == 8
        assert signal.target_ticks == 12

    def test_signal_for_short_entry(self):
        """Test creating a short entry signal."""
        signal = Signal(
            signal_type=SignalType.SHORT_ENTRY,
            confidence=0.80,
            stop_ticks=10,
            target_ticks=15,
            predicted_class=0,  # DOWN
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.SHORT_ENTRY
        assert signal.predicted_class == 0  # DOWN

    def test_signal_for_exit(self):
        """Test creating an exit signal."""
        signal = Signal(
            signal_type=SignalType.EXIT_LONG,
            confidence=0.65,
            predicted_class=1,  # FLAT (exiting position)
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.EXIT_LONG
        assert signal.predicted_class == 1  # FLAT

    def test_signal_for_flatten(self):
        """Test creating a flatten signal."""
        signal = Signal(
            signal_type=SignalType.FLATTEN,
            confidence=1.0,  # Flatten always has max confidence
            predicted_class=1,  # FLAT
            reason="EOD",
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.FLATTEN
        assert signal.reason == "EOD"


class TestFlattenAll:
    """Tests for flatten_all method."""

    @pytest.mark.asyncio
    async def test_flatten_all_no_position(self, mock_rest_client, mock_position_manager):
        """Test flatten when no position exists."""
        mock_position_manager.is_flat = MagicMock(return_value=True)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Should complete without errors
        await executor.flatten_all("CON.F.US.MES.H26")


class TestOrderTracking:
    """Tests for order tracking functionality."""

    def test_track_order(self, mock_rest_client, mock_position_manager):
        """Test that orders can be tracked."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Simulate adding an order to tracking
        mock_order = MagicMock()
        mock_order.order_id = "ORD123"

        executor._open_orders["ORD123"] = mock_order

        assert "ORD123" in executor._open_orders
        assert executor.has_open_orders() is True


class TestPriceLevelCalculation:
    """Tests for price level calculation helpers."""

    def test_stop_price_long(self):
        """Test stop price calculation for long position."""
        entry_price = 5000.0
        stop_ticks = 8

        # Stop should be below entry for long
        stop_price = entry_price - (stop_ticks * MES_TICK_SIZE)

        assert stop_price == 4998.0  # 5000 - 2.0

    def test_stop_price_short(self):
        """Test stop price calculation for short position."""
        entry_price = 5000.0
        stop_ticks = 8

        # Stop should be above entry for short
        stop_price = entry_price + (stop_ticks * MES_TICK_SIZE)

        assert stop_price == 5002.0  # 5000 + 2.0

    def test_target_price_long(self):
        """Test target price calculation for long position."""
        entry_price = 5000.0
        target_ticks = 12

        # Target should be above entry for long
        target_price = entry_price + (target_ticks * MES_TICK_SIZE)

        assert target_price == 5003.0  # 5000 + 3.0

    def test_target_price_short(self):
        """Test target price calculation for short position."""
        entry_price = 5000.0
        target_ticks = 12

        # Target should be below entry for short
        target_price = entry_price - (target_ticks * MES_TICK_SIZE)

        assert target_price == 4997.0  # 5000 - 3.0


class TestOrderExecutorErrorHandling:
    """Tests for error handling in order executor."""

    @pytest.mark.asyncio
    async def test_execute_entry_api_error(self, mock_rest_client, mock_position_manager):
        """Test handling of API errors during entry."""
        # Mock API to raise exception
        mock_rest_client.place_order = AsyncMock(side_effect=Exception("API Error"))

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.75,
            stop_ticks=8,
            target_ticks=12,
            predicted_class=2,  # UP
            timestamp=datetime.now(),
        )

        # Should handle error gracefully
        result = await executor.execute_signal(
            signal=signal,
            contract_id="CON.F.US.MES.H26",
            size=1,
            current_price=5000.0,
        )

        # Should return failed result
        if result:
            assert result.status == ExecutionStatus.FAILED or result.error_message is not None


class TestOCOManagement:
    """Tests for OCO (one-cancels-other) order management."""

    def test_oco_order_tracking_structure(self, mock_rest_client, mock_position_manager):
        """Test that OCO orders can be tracked."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Verify internal structure supports OCO tracking
        assert hasattr(executor, '_open_orders')
        assert isinstance(executor._open_orders, dict)


# ============================================================================
# Execute Entry Tests with Mocked API Responses
# ============================================================================

class TestExecuteEntryAsync:
    """Tests for execute_entry method with mocked API responses.

    Why test execute_entry:
    - Core execution logic that handles the complete entry workflow
    - Must handle API responses correctly (success, rejection, timeout)
    - Critical for ensuring orders are placed and tracked correctly
    """

    @pytest.fixture
    def mock_order_response(self):
        """Create a mock OrderResponse."""
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        return OrderResponse(
            order_id="ORD123",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

    @pytest.fixture
    def mock_rejected_response(self):
        """Create a mock rejected OrderResponse."""
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        return OrderResponse(
            order_id="ORD456",
            account_id=12345,
            status=OrderStatus.REJECTED,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
            error_message="Insufficient margin",
        )

    @pytest.mark.asyncio
    async def test_execute_entry_long_success(self, mock_rest_client, mock_position_manager):
        """Test successful long entry execution.

        Why: Verifies the complete happy path for long entries.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        # Setup mock responses
        mock_entry_order = OrderResponse(
            order_id="ENTRY001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

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

        # Configure mock REST client
        mock_rest_client.place_order = AsyncMock(
            side_effect=[mock_entry_order, mock_stop_order, mock_target_order]
        )
        mock_rest_client.get_order = AsyncMock(return_value={
            'order_id': 'ENTRY001',
            'status': 'FILLED',
            'fill_price': 5000.0
        })

        # Create executor
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Mock _wait_for_fill to return fill price
        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
            result = await executor.execute_entry(
                contract_id="CON.F.US.MES.H26",
                direction=1,  # Long
                size=1,
                stop_ticks=8,
                target_ticks=12,
                current_price=5000.0,
            )

        # Verify result
        assert result.status == ExecutionStatus.FILLED
        assert result.entry_fill_price == 5000.0
        assert result.entry_fill_size == 1
        assert result.entry_order_id == "ENTRY001"

    @pytest.mark.asyncio
    async def test_execute_entry_short_success(self, mock_rest_client, mock_position_manager):
        """Test successful short entry execution.

        Why: Short entries have different stop/target price calculations.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        mock_entry_order = OrderResponse(
            order_id="ENTRY002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_entry_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
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

        assert result.status == ExecutionStatus.FILLED
        assert result.entry_fill_price == 5000.0

    @pytest.mark.asyncio
    async def test_execute_entry_rejected(self, mock_rest_client, mock_position_manager):
        """Test entry order rejection handling.

        Why: Must handle API rejections gracefully without crashing.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        mock_rejected = OrderResponse(
            order_id="REJ001",
            account_id=12345,
            status=OrderStatus.REJECTED,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
            error_message="Insufficient margin",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_rejected)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        result = await executor.execute_entry(
            contract_id="CON.F.US.MES.H26",
            direction=1,
            size=1,
            stop_ticks=8,
            target_ticks=12,
            current_price=5000.0,
        )

        assert result.status == ExecutionStatus.REJECTED
        assert "margin" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_entry_fill_timeout(self, mock_rest_client, mock_position_manager):
        """Test handling of fill timeout.

        Why: Ensures orders are cancelled when fills don't arrive in time.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        mock_entry_order = OrderResponse(
            order_id="TIMEOUT001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_rest_client.place_order = AsyncMock(return_value=mock_entry_order)
        mock_rest_client.cancel_order = AsyncMock(return_value={'success': True})

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Mock fill timeout
        with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=None)):
            result = await executor.execute_entry(
                contract_id="CON.F.US.MES.H26",
                direction=1,
                size=1,
                stop_ticks=8,
                target_ticks=12,
                current_price=5000.0,
            )

        assert result.status == ExecutionStatus.FAILED
        assert "timeout" in result.error_message.lower()
        mock_rest_client.cancel_order.assert_called_once()


# ============================================================================
# Execute Exit Tests
# ============================================================================

class TestExecuteExitAsync:
    """Tests for execute_exit method."""

    @pytest.mark.asyncio
    async def test_execute_exit_long(self, mock_rest_client, mock_position_manager):
        """Test exiting a long position.

        Why: Exit must cancel pending orders and place market exit.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

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
        mock_rest_client.cancel_all_orders = AsyncMock(return_value={'cancelled': 2})

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_oco_orders', AsyncMock()):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5001.0)):
                result = await executor.execute_exit(
                    contract_id="CON.F.US.MES.H26",
                    size=1,
                    current_direction=1,
                )

        # Exit returns True on success
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_exit_short(self, mock_rest_client, mock_position_manager):
        """Test exiting a short position.

        Why: Short exits buy to cover, opposite of long exits.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        mock_exit_order = OrderResponse(
            order_id="EXIT002",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
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
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=4999.0)):
                result = await executor.execute_exit(
                    contract_id="CON.F.US.MES.H26",
                    size=1,
                    current_direction=-1,
                )

        assert result is True


# ============================================================================
# Flatten All Tests
# ============================================================================

class TestFlattenAllAsync:
    """Tests for flatten_all method."""

    @pytest.mark.asyncio
    async def test_flatten_all_with_position(self, mock_rest_client, mock_position_manager):
        """Test flattening when position exists.

        Why: Must cancel all orders and place market flatten order.
        """
        from src.api import OrderResponse, OrderStatus, OrderType, OrderSide

        mock_flatten_order = OrderResponse(
            order_id="FLAT001",
            account_id=12345,
            status=OrderStatus.WORKING,
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            size=1,
            contract_id="CON.F.US.MES.H26",
        )

        mock_position_manager.is_flat = MagicMock(return_value=False)
        mock_position_manager.current_position = MagicMock()
        mock_position_manager.current_position.direction = 1
        mock_position_manager.current_position.size = 1
        mock_position_manager.flatten = MagicMock()

        mock_rest_client.place_order = AsyncMock(return_value=mock_flatten_order)
        mock_rest_client.cancel_all_orders = AsyncMock(return_value={'cancelled': 2})
        mock_rest_client.flatten_position = AsyncMock(return_value=mock_flatten_order)

        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        with patch.object(executor, '_cancel_all_orders', AsyncMock()):
            with patch.object(executor, '_wait_for_fill', AsyncMock(return_value=5000.0)):
                result = await executor.flatten_all("CON.F.US.MES.H26")

        assert result is True
        mock_position_manager.flatten.assert_called_once()


# ============================================================================
# Signal Dispatch Complete Tests
# ============================================================================

class TestSignalDispatchComplete:
    """Complete tests for all signal type dispatching.

    Why: Ensures each signal type is handled correctly.
    """

    @pytest.mark.asyncio
    async def test_dispatch_long_entry(self, mock_rest_client, mock_position_manager):
        """Test LONG_ENTRY signal dispatches correctly."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.75,
            stop_ticks=8,
            target_ticks=12,
            predicted_class=2,
            timestamp=datetime.now(),
        )

        # Mock execute_entry
        with patch.object(
            executor, 'execute_entry',
            AsyncMock(return_value=EntryResult(
                status=ExecutionStatus.FILLED,
                entry_fill_price=5000.0,
                entry_fill_size=1,
            ))
        ) as mock_entry:
            result = await executor.execute_signal(
                signal=signal,
                contract_id="CON.F.US.MES.H26",
                size=1,
                current_price=5000.0,
            )

            mock_entry.assert_called_once()
            # Verify direction is 1 for long
            assert mock_entry.call_args.kwargs['direction'] == 1

    @pytest.mark.asyncio
    async def test_dispatch_short_entry(self, mock_rest_client, mock_position_manager):
        """Test SHORT_ENTRY signal dispatches correctly."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.SHORT_ENTRY,
            confidence=0.80,
            stop_ticks=8,
            target_ticks=12,
            predicted_class=0,
            timestamp=datetime.now(),
        )

        with patch.object(
            executor, 'execute_entry',
            AsyncMock(return_value=EntryResult(
                status=ExecutionStatus.FILLED,
                entry_fill_price=5000.0,
                entry_fill_size=1,
            ))
        ) as mock_entry:
            result = await executor.execute_signal(
                signal=signal,
                contract_id="CON.F.US.MES.H26",
                size=1,
                current_price=5000.0,
            )

            mock_entry.assert_called_once()
            # Verify direction is -1 for short
            assert mock_entry.call_args.kwargs['direction'] == -1

    @pytest.mark.asyncio
    async def test_dispatch_exit_long(self, mock_rest_client, mock_position_manager):
        """Test EXIT_LONG signal dispatches correctly."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.EXIT_LONG,
            confidence=0.70,
            predicted_class=1,
            timestamp=datetime.now(),
        )

        with patch.object(executor, 'execute_exit', AsyncMock(return_value=True)) as mock_exit:
            result = await executor.execute_signal(
                signal=signal,
                contract_id="CON.F.US.MES.H26",
                size=1,
                current_price=5000.0,
            )

            mock_exit.assert_called_once()
            # Result should be None for exit signals
            assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_exit_short(self, mock_rest_client, mock_position_manager):
        """Test EXIT_SHORT signal dispatches correctly."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.EXIT_SHORT,
            confidence=0.70,
            predicted_class=1,
            timestamp=datetime.now(),
        )

        with patch.object(executor, 'execute_exit', AsyncMock(return_value=True)) as mock_exit:
            result = await executor.execute_signal(
                signal=signal,
                contract_id="CON.F.US.MES.H26",
                size=1,
                current_price=5000.0,
            )

            mock_exit.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_reverse_to_long(self, mock_rest_client, mock_position_manager):
        """Test REVERSE_TO_LONG signal dispatches correctly.

        Why: Reverse signals must exit current position then enter opposite direction.
        """
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.REVERSE_TO_LONG,
            confidence=0.85,
            stop_ticks=8,
            target_ticks=12,
            predicted_class=2,
            timestamp=datetime.now(),
        )

        with patch.object(executor, 'execute_exit', AsyncMock(return_value=True)) as mock_exit:
            with patch.object(
                executor, 'execute_entry',
                AsyncMock(return_value=EntryResult(
                    status=ExecutionStatus.FILLED,
                    entry_fill_price=5000.0,
                    entry_fill_size=1,
                ))
            ) as mock_entry:
                result = await executor.execute_signal(
                    signal=signal,
                    contract_id="CON.F.US.MES.H26",
                    size=1,
                    current_price=5000.0,
                )

                # Both exit and entry should be called
                mock_exit.assert_called_once()
                mock_entry.assert_called_once()
                assert mock_entry.call_args.kwargs['direction'] == 1

    @pytest.mark.asyncio
    async def test_dispatch_reverse_to_short(self, mock_rest_client, mock_position_manager):
        """Test REVERSE_TO_SHORT signal dispatches correctly."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.REVERSE_TO_SHORT,
            confidence=0.85,
            stop_ticks=8,
            target_ticks=12,
            predicted_class=0,
            timestamp=datetime.now(),
        )

        with patch.object(executor, 'execute_exit', AsyncMock(return_value=True)) as mock_exit:
            with patch.object(
                executor, 'execute_entry',
                AsyncMock(return_value=EntryResult(
                    status=ExecutionStatus.FILLED,
                    entry_fill_price=5000.0,
                    entry_fill_size=1,
                ))
            ) as mock_entry:
                result = await executor.execute_signal(
                    signal=signal,
                    contract_id="CON.F.US.MES.H26",
                    size=1,
                    current_price=5000.0,
                )

                mock_exit.assert_called_once()
                mock_entry.assert_called_once()
                assert mock_entry.call_args.kwargs['direction'] == -1

    @pytest.mark.asyncio
    async def test_dispatch_flatten(self, mock_rest_client, mock_position_manager):
        """Test FLATTEN signal dispatches correctly."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.FLATTEN,
            confidence=1.0,
            predicted_class=1,
            reason="EOD",
            timestamp=datetime.now(),
        )

        with patch.object(executor, 'flatten_all', AsyncMock(return_value=True)) as mock_flatten:
            result = await executor.execute_signal(
                signal=signal,
                contract_id="CON.F.US.MES.H26",
                size=1,
                current_price=5000.0,
            )

            mock_flatten.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_hold(self, mock_rest_client, mock_position_manager):
        """Test HOLD signal does nothing."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            predicted_class=1,
            timestamp=datetime.now(),
        )

        # No methods should be called for HOLD
        with patch.object(executor, 'execute_entry', AsyncMock()) as mock_entry:
            with patch.object(executor, 'execute_exit', AsyncMock()) as mock_exit:
                with patch.object(executor, 'flatten_all', AsyncMock()) as mock_flatten:
                    result = await executor.execute_signal(
                        signal=signal,
                        contract_id="CON.F.US.MES.H26",
                        size=1,
                        current_price=5000.0,
                    )

                    mock_entry.assert_not_called()
                    mock_exit.assert_not_called()
                    mock_flatten.assert_not_called()
                    assert result is None


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestOrderExecutorHelpers:
    """Tests for helper methods in OrderExecutor."""

    def test_get_open_orders_returns_copy(self, mock_rest_client, mock_position_manager):
        """Test get_open_orders returns a copy, not the original dict.

        Why: Prevents external modification of internal state.
        """
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        # Add an order
        mock_order = MagicMock()
        mock_order.order_id = "ORD123"
        executor._open_orders["ORD123"] = mock_order

        # Get orders
        orders = executor.get_open_orders()

        # Modify the returned dict
        orders["NEW"] = "should not affect original"

        # Original should be unchanged
        assert "NEW" not in executor._open_orders

    def test_has_open_orders_true(self, mock_rest_client, mock_position_manager):
        """Test has_open_orders returns True when orders exist."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        mock_order = MagicMock()
        mock_order.order_id = "ORD123"
        executor._open_orders["ORD123"] = mock_order

        assert executor.has_open_orders() is True

    def test_pending_fills_tracking(self, mock_rest_client, mock_position_manager):
        """Test that pending fills dict is initialized."""
        executor = OrderExecutor(
            rest_client=mock_rest_client,
            ws_client=None,
            position_manager=mock_position_manager,
        )

        assert hasattr(executor, '_pending_fills')
        assert isinstance(executor._pending_fills, dict)
        assert len(executor._pending_fills) == 0
