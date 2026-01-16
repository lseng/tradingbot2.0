"""
Extended Unit Tests for Position Manager.

This file provides additional test coverage for lines not covered in test_trading.py.
Focus areas:
- Reversal fill handling (lines 295-309)
- Adding to existing positions (lines 341-348)
- Partial position closure (lines 371-380)
- Stop/target price setters (lines 405-423)
- API sync with discrepancy handling (lines 438-487)
- Callback exception handling (lines 526-527)
- get_metrics for SHORT direction (lines 541-542)
- calculate_pnl_ticks for flat position (line 116)
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from src.trading.position_manager import (
    Position,
    PositionDirection,
    PositionManager,
    PositionChange,
    Fill,
    PositionData,
    MES_TICK_SIZE,
    MES_POINT_VALUE,
)


class TestPositionPnlTicksFlat:
    """Tests for calculate_pnl_ticks when position is flat (line 116)."""

    def test_calculate_pnl_ticks_flat_position(self):
        """Flat position returns 0 ticks."""
        pos = Position(contract_id="MES", direction=0, size=0)
        result = pos.calculate_pnl_ticks(6000.0)
        assert result == 0.0

    def test_calculate_pnl_ticks_zero_size(self):
        """Zero size with direction returns 0 ticks."""
        pos = Position(contract_id="MES", direction=1, size=0)
        result = pos.calculate_pnl_ticks(6000.0)
        assert result == 0.0

    def test_calculate_pnl_ticks_short_position(self):
        """Short position calculates ticks correctly."""
        pos = Position(
            contract_id="MES",
            direction=-1,
            size=1,
            entry_price=6000.0,
        )
        # Price down 1 point (4 ticks) = +4 ticks for short
        result = pos.calculate_pnl_ticks(5999.0)
        assert result == pytest.approx(4.0)

    def test_calculate_pnl_ticks_multiple_contracts(self):
        """Multiple contracts multiply tick P&L."""
        pos = Position(
            contract_id="MES",
            direction=1,
            size=3,
            entry_price=6000.0,
        )
        # 2 points up (8 ticks) * 3 contracts = 24 ticks
        result = pos.calculate_pnl_ticks(6002.0)
        assert result == pytest.approx(24.0)


class TestPositionManagerAddToPosition:
    """Tests for _add_to_position method (lines 341-348)."""

    def test_add_to_long_position(self):
        """Adding to long position averages price."""
        manager = PositionManager("MES")

        # Open initial position at 6000
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,  # BUY
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)
        assert manager.get_size() == 1
        assert manager.position.entry_price == 6000.0

        # Add another contract at 6002
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=1,  # BUY (same direction)
            size=1,
            price=6002.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.get_size() == 2
        # Average: (6000 * 1 + 6002 * 1) / 2 = 6001
        assert manager.position.entry_price == pytest.approx(6001.0)

    def test_add_to_short_position(self):
        """Adding to short position averages price."""
        manager = PositionManager("MES")

        # Open initial short at 6000
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=2,  # SELL
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)
        assert manager.is_short()

        # Add more contracts at 6010
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=2,  # SELL (same direction)
            size=1,
            price=6010.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.get_size() == 3
        # Average: (6000 * 2 + 6010 * 1) / 3 = 6003.33
        assert manager.position.entry_price == pytest.approx(6003.33, rel=0.01)

    def test_add_callback_type(self):
        """Adding to position triggers 'add' change type callback."""
        manager = PositionManager("MES")
        changes = []

        def callback(change):
            changes.append(change)

        manager.on_position_change(callback)

        # Open position
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)

        # Add to position
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=1,
            size=1,
            price=6002.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert len(changes) == 2
        assert changes[0].change_type == "open"
        assert changes[1].change_type == "add"


class TestPositionManagerPartialClose:
    """Tests for _partial_close method (lines 368-380)."""

    def test_partial_close_long(self):
        """Partial close reduces position size."""
        manager = PositionManager("MES")

        # Open 3 contracts
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=3,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)
        assert manager.get_size() == 3

        # Sell 1 contract (partial close)
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=2,  # SELL
            size=1,
            price=6002.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.get_size() == 2
        assert manager.is_long()  # Still long
        # Realized P&L: 1 contract * 2 points * $5 = $10
        assert manager.position.realized_pnl == pytest.approx(10.0)

    def test_partial_close_short(self):
        """Partial close of short position."""
        manager = PositionManager("MES")

        # Open 2 short contracts
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=2,
            size=2,
            price=6010.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)

        # Buy 1 contract (partial close)
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=1,  # BUY
            size=1,
            price=6005.0,  # 5 points profit
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.get_size() == 1
        assert manager.is_short()
        # Realized P&L: 1 contract * 5 points * $5 = $25
        assert manager.position.realized_pnl == pytest.approx(25.0)

    def test_partial_close_callback_type(self):
        """Partial close triggers 'partial_close' callback."""
        manager = PositionManager("MES")
        changes = []

        def callback(change):
            changes.append(change)

        manager.on_position_change(callback)

        # Open 3 contracts
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=3,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)

        # Partial close
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=2,
            size=1,
            price=6001.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert changes[-1].change_type == "partial_close"


class TestPositionManagerReversal:
    """Tests for reversal logic in update_from_fill (lines 293-309)."""

    def test_reversal_long_to_short(self):
        """Reversing from long to short position."""
        manager = PositionManager("MES")
        changes = []

        def callback(change):
            changes.append(change)

        manager.on_position_change(callback)

        # Open 1 long
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)
        assert manager.is_long()

        # Sell 2 (close 1 + open 1 short)
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=2,
            size=2,
            price=6002.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.is_short()
        assert manager.get_size() == 1
        assert manager.position.entry_price == 6002.0  # New entry at reversal price
        # The last callback should be 'reversal'
        assert changes[-1].change_type == "reversal"

    def test_reversal_short_to_long(self):
        """Reversing from short to long position."""
        manager = PositionManager("MES")

        # Open 2 short
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=2,
            size=2,
            price=6010.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)
        assert manager.is_short()

        # Buy 3 (close 2 + open 1 long)
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=1,
            size=3,
            price=6005.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.is_long()
        assert manager.get_size() == 1
        assert manager.position.entry_price == 6005.0

    def test_reversal_with_larger_size(self):
        """Reversal with significantly larger opposite size."""
        manager = PositionManager("MES")

        # Open 1 long
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)

        # Sell 5 (close 1 + open 4 short)
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=2,
            size=5,
            price=6001.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        assert manager.is_short()
        assert manager.get_size() == 4


class TestPositionManagerStopTarget:
    """Tests for set_stop_price and set_target_price (lines 397-423)."""

    def test_set_stop_price(self):
        """Set stop price updates position."""
        manager = PositionManager("MES")

        # Open position
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        manager.set_stop_price(5995.0, order_id="STOP123")

        assert manager.position.stop_price == 5995.0
        assert manager.position.stop_order_id == "STOP123"

    def test_set_stop_price_without_order_id(self):
        """Set stop price without order ID."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        manager.set_stop_price(5990.0)

        assert manager.position.stop_price == 5990.0
        assert manager.position.stop_order_id is None

    def test_set_target_price(self):
        """Set target price updates position."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        manager.set_target_price(6010.0, order_id="TARGET456")

        assert manager.position.target_price == 6010.0
        assert manager.position.target_order_id == "TARGET456"

    def test_set_target_price_without_order_id(self):
        """Set target price without order ID."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        manager.set_target_price(6015.0)

        assert manager.position.target_price == 6015.0
        assert manager.position.target_order_id is None


class TestPositionManagerSyncFromApi:
    """Tests for sync_from_api method (lines 425-487)."""

    def test_sync_matching_position(self):
        """Sync with matching API position returns True."""
        manager = PositionManager("MES")

        # Open local position
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        # API position matches
        api_pos = PositionData(
            account_id=12345,
            contract_id="MES",
            size=2,  # Positive = long
            avg_price=6000.0,
            unrealized_pnl=10.0,
        )

        result = manager.sync_from_api(api_pos)

        assert result is True
        assert manager.position.unrealized_pnl == 10.0

    def test_sync_direction_mismatch(self):
        """Sync with direction mismatch returns False and updates."""
        manager = PositionManager("MES")

        # Open local long position
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        # API says we're short (size negative = short)
        api_pos = PositionData(
            account_id=12345,
            contract_id="MES",
            size=-1,  # Negative = short
            avg_price=6005.0,
            unrealized_pnl=-5.0,
        )

        result = manager.sync_from_api(api_pos)

        assert result is False  # Discrepancy detected
        assert manager.is_short()  # API is source of truth
        assert manager.position.entry_price == 6005.0

    def test_sync_size_mismatch(self):
        """Sync with size mismatch returns False and updates."""
        manager = PositionManager("MES")
        changes = []

        def callback(change):
            changes.append(change)

        manager.on_position_change(callback)

        # Open local position with 1 contract
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        # API says we have 3 contracts
        api_pos = PositionData(
            account_id=12345,
            contract_id="MES",
            size=3,
            avg_price=6002.0,
            unrealized_pnl=15.0,
        )

        result = manager.sync_from_api(api_pos)

        assert result is False
        assert manager.get_size() == 3
        # Should trigger sync callback
        assert changes[-1].change_type == "sync"

    def test_sync_contract_id_mismatch(self):
        """Sync with different contract ID returns False without update."""
        manager = PositionManager("MES")

        # API has different contract
        api_pos = PositionData(
            account_id=12345,
            contract_id="ES",  # Different contract
            size=1,
            avg_price=6000.0,
        )

        result = manager.sync_from_api(api_pos)

        assert result is False
        # Position should remain unchanged
        assert manager.is_flat()


class TestPositionManagerCallbackException:
    """Tests for callback exception handling (lines 521-527)."""

    def test_callback_exception_caught(self):
        """Callback exceptions don't crash the manager."""
        manager = PositionManager("MES")

        def bad_callback(change):
            raise RuntimeError("Callback error!")

        def good_callback(change):
            pass

        manager.on_position_change(bad_callback)
        manager.on_position_change(good_callback)

        # Should not raise
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        # Position should still be updated
        assert manager.is_long()

    def test_multiple_callback_exceptions(self):
        """Multiple callback exceptions don't prevent position update."""
        manager = PositionManager("MES")

        def bad_callback1(change):
            raise ValueError("Error 1")

        def bad_callback2(change):
            raise KeyError("Error 2")

        manager.on_position_change(bad_callback1)
        manager.on_position_change(bad_callback2)

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )

        # Should not raise
        manager.update_from_fill(fill)
        assert manager.is_long()


class TestPositionManagerGetMetrics:
    """Tests for get_metrics method (lines 529-557)."""

    def test_get_metrics_flat(self):
        """Get metrics for flat position."""
        manager = PositionManager("MES")

        metrics = manager.get_metrics()

        assert metrics["direction"] == "FLAT"
        assert metrics["size"] == 0
        assert metrics["entry_price"] == 0.0

    def test_get_metrics_long(self):
        """Get metrics for long position."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)
        manager.update_pnl(6002.0)

        metrics = manager.get_metrics()

        assert metrics["direction"] == "LONG"
        assert metrics["size"] == 2
        assert metrics["entry_price"] == 6000.0
        assert metrics["last_price"] == 6002.0
        assert metrics["pnl_ticks"] == pytest.approx(16.0)  # 8 ticks * 2 contracts

    def test_get_metrics_short(self):
        """Get metrics for short position (line 541-542 coverage)."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=2,  # SELL
            size=1,
            price=6010.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)
        manager.update_pnl(6005.0)

        metrics = manager.get_metrics()

        assert metrics["direction"] == "SHORT"
        assert metrics["size"] == 1
        assert metrics["entry_price"] == 6010.0
        # Short profits when price goes down
        assert metrics["unrealized_pnl"] == pytest.approx(25.0)

    def test_get_metrics_total_pnl(self):
        """Get metrics includes total P&L (unrealized + realized)."""
        manager = PositionManager("MES")

        # Open 2 contracts
        fill1 = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill1)

        # Partial close 1 at profit
        fill2 = Fill(
            order_id="2",
            contract_id="MES",
            side=2,
            size=1,
            price=6002.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill2)

        # Update unrealized P&L
        manager.update_pnl(6003.0)

        metrics = manager.get_metrics()

        # Realized: 1 contract * 2 points * $5 = $10
        # Unrealized: 1 contract * 3 points * $5 = $15
        # Total: $25
        assert metrics["realized_pnl"] == pytest.approx(10.0)
        assert metrics["unrealized_pnl"] == pytest.approx(15.0)
        assert metrics["total_pnl"] == pytest.approx(25.0)

    def test_get_metrics_no_last_price(self):
        """Get metrics with no last price."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)
        # Don't update P&L (no last price)

        metrics = manager.get_metrics()

        assert metrics["last_price"] is None
        assert metrics["pnl_ticks"] == 0


class TestPositionDataFallback:
    """Tests for PositionData class (lines 195-213 in topstepx_rest.py)."""

    def test_position_data_long(self):
        """PositionData direction property for long position."""
        pos = PositionData(
            account_id=12345,
            contract_id="MES",
            size=2,
            avg_price=6000.0,
        )

        assert pos.direction == 1

    def test_position_data_short(self):
        """PositionData direction property for short position."""
        pos = PositionData(
            account_id=12345,
            contract_id="MES",
            size=-3,
            avg_price=6000.0,
        )

        assert pos.direction == -1

    def test_position_data_flat(self):
        """PositionData direction property for flat position."""
        pos = PositionData(
            account_id=12345,
            contract_id="MES",
            size=0,
            avg_price=0.0,
        )

        assert pos.direction == 0


class TestPositionManagerThreadSafety:
    """Tests for thread-safety of PositionManager."""

    def test_concurrent_pnl_updates(self):
        """Multiple P&L updates don't corrupt state."""
        import threading

        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        results = []

        def update_pnl(price):
            for _ in range(100):
                pnl = manager.update_pnl(price)
                results.append(pnl)

        threads = [
            threading.Thread(target=update_pnl, args=(6001.0,)),
            threading.Thread(target=update_pnl, args=(6002.0,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Position should still be valid
        assert manager.is_long()
        assert manager.get_size() == 1

    def test_concurrent_fill_and_read(self):
        """Concurrent fill updates and position reads don't crash."""
        import threading

        manager = PositionManager("MES")

        def fill_loop():
            for i in range(50):
                fill = Fill(
                    order_id=str(i),
                    contract_id="MES",
                    side=1 if i % 2 == 0 else 2,
                    size=1,
                    price=6000.0 + i,
                    timestamp=datetime.now(),
                )
                manager.update_from_fill(fill)

        def read_loop():
            for _ in range(100):
                _ = manager.position
                _ = manager.get_size()
                _ = manager.is_flat()

        threads = [
            threading.Thread(target=fill_loop),
            threading.Thread(target=read_loop),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash


class TestPositionManagerFlatten:
    """Tests for flatten method with callbacks."""

    def test_flatten_triggers_callback(self):
        """Flatten triggers callback with 'flatten' type."""
        manager = PositionManager("MES")
        changes = []

        def callback(change):
            changes.append(change)

        manager.on_position_change(callback)

        # Open position
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)

        # Flatten
        manager.flatten()

        assert len(changes) == 2
        assert changes[-1].change_type == "flatten"
        assert changes[-1].fill is None

    def test_flatten_resets_all_fields(self):
        """Flatten resets all position fields."""
        manager = PositionManager("MES")

        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)
        manager.set_stop_price(5990.0, "STOP1")
        manager.set_target_price(6020.0, "TARGET1")
        manager.update_pnl(6005.0)

        manager.flatten()

        pos = manager.position
        assert pos.is_flat
        assert pos.direction == 0
        assert pos.size == 0
        assert pos.entry_price == 0.0
        assert pos.entry_time is None
        assert pos.stop_price == 0.0
        assert pos.target_price == 0.0
        assert pos.stop_order_id is None
        assert pos.target_order_id is None
        assert pos.unrealized_pnl == 0.0


class TestPositionManagerGetDirection:
    """Tests for get_direction and related methods (lines 231-244)."""

    def test_get_direction_flat(self):
        """Get direction returns 0 for flat."""
        manager = PositionManager("MES")
        assert manager.get_direction() == 0

    def test_get_direction_long(self):
        """Get direction returns 1 for long."""
        manager = PositionManager("MES")
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=1,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)
        assert manager.get_direction() == 1

    def test_get_direction_short(self):
        """Get direction returns -1 for short."""
        manager = PositionManager("MES")
        fill = Fill(
            order_id="1",
            contract_id="MES",
            side=2,
            size=1,
            price=6000.0,
            timestamp=datetime.now(),
        )
        manager.update_from_fill(fill)
        assert manager.get_direction() == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
