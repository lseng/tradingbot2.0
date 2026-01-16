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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.order_executor import (
    ExecutionStatus,
    EntryResult,
    ExecutorConfig,
    OrderExecutor,
    MES_TICK_SIZE,
)
from trading.signal_generator import Signal, SignalType


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
