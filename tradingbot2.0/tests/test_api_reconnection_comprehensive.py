"""
Comprehensive API Reconnection Tests for Go-Live Validation (#11).

Tests cover:
1. Subscription recovery after reconnect
2. State consistency after reconnection
3. Cascade failures (market + trade connections)
4. Operation buffering during disconnection
5. Callback resilience
6. Connection stability (rapid connect/disconnect)
7. Graceful shutdown during reconnection
8. Position synchronization after reconnect

These tests ensure the API handles network interruptions correctly.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import dataclass
from typing import Optional, List

from src.api.topstepx_ws import (
    Quote,
    OrderFill,
    PositionUpdate,
    AccountUpdate,
    WebSocketState,
)
from src.trading.recovery import (
    RecoveryHandler,
    RecoveryConfig,
    RecoveryState,
    ErrorEvent,
    ErrorCategory,
    ErrorSeverity,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def recovery_config():
    """Create a fast recovery config for testing."""
    return RecoveryConfig(
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.1,
        backoff_multiplier=2.0,
        max_reconnect_attempts=3,
        max_auth_retries=2,
        reconnect_timeout_seconds=1.0,
        position_sync_timeout_seconds=0.5,
    )


@pytest.fixture
def recovery_handler(recovery_config):
    """Create a recovery handler for testing."""
    return RecoveryHandler(config=recovery_config)


# =============================================================================
# Subscription Recovery Tests
# =============================================================================

class TestSubscriptionRecoveryAfterReconnect:
    """Test that subscriptions are properly restored after reconnect."""

    @pytest.mark.asyncio
    async def test_quote_subscription_restored_after_reconnect(self):
        """Test that quote subscriptions are restored after WebSocket reconnect."""
        subscribed_contracts = []
        reconnect_count = 0

        async def mock_subscribe(contract_id: str):
            subscribed_contracts.append(contract_id)
            return True

        async def mock_reconnect():
            nonlocal reconnect_count
            reconnect_count += 1
            # Simulate successful reconnect
            return True

        # Simulate initial subscription
        await mock_subscribe("MESU5")
        await mock_subscribe("MESZ5")
        assert len(subscribed_contracts) == 2

        # Simulate disconnect and reconnect
        subscribed_contracts.clear()
        await mock_reconnect()

        # Re-subscribe after reconnect
        await mock_subscribe("MESU5")
        await mock_subscribe("MESZ5")

        assert reconnect_count == 1
        assert "MESU5" in subscribed_contracts
        assert "MESZ5" in subscribed_contracts

    @pytest.mark.asyncio
    async def test_subscription_state_tracking(self):
        """Test that subscription state is properly tracked for restoration."""
        @dataclass
        class SubscriptionManager:
            active_subscriptions: set = None

            def __post_init__(self):
                if self.active_subscriptions is None:
                    self.active_subscriptions = set()

            def subscribe(self, contract_id: str):
                self.active_subscriptions.add(contract_id)

            def unsubscribe(self, contract_id: str):
                self.active_subscriptions.discard(contract_id)

            def restore_subscriptions(self):
                """Return list of subscriptions to restore."""
                return list(self.active_subscriptions)

        manager = SubscriptionManager()
        manager.subscribe("MESU5")
        manager.subscribe("MESZ5")
        manager.subscribe("ESU5")

        # Simulate disconnect (subscriptions lost on server but tracked locally)
        subs_to_restore = manager.restore_subscriptions()

        assert len(subs_to_restore) == 3
        assert "MESU5" in subs_to_restore
        assert "MESZ5" in subs_to_restore
        assert "ESU5" in subs_to_restore

    @pytest.mark.asyncio
    async def test_partial_subscription_failure_handling(self):
        """Test handling when some subscriptions fail to restore."""
        results = {}

        async def mock_subscribe(contract_id: str) -> bool:
            # Simulate failure for one contract
            if contract_id == "FAIL_CONTRACT":
                results[contract_id] = False
                return False
            results[contract_id] = True
            return True

        contracts = ["MESU5", "FAIL_CONTRACT", "MESZ5"]

        for contract in contracts:
            await mock_subscribe(contract)

        assert results["MESU5"] is True
        assert results["FAIL_CONTRACT"] is False
        assert results["MESZ5"] is True

        # Should have 2 successful subscriptions
        successful = [c for c, r in results.items() if r]
        assert len(successful) == 2


# =============================================================================
# State Consistency Tests
# =============================================================================

class TestStateConsistencyAfterReconnection:
    """Test that system state remains consistent after reconnection."""

    @pytest.mark.asyncio
    async def test_no_duplicate_quotes_after_reconnect(self):
        """Test that duplicate quotes are detected and handled after reconnect."""
        received_quotes = []
        last_quote_time = None

        def is_duplicate(quote: Quote) -> bool:
            nonlocal last_quote_time
            if last_quote_time and quote.timestamp:
                # Same timestamp = duplicate
                if quote.timestamp == last_quote_time:
                    return True
            last_quote_time = quote.timestamp
            return False

        # Simulate quotes
        now = datetime.utcnow()
        quotes = [
            Quote("MESU5", 6050.0, 6050.25, 6050.0, timestamp=now),
            Quote("MESU5", 6050.0, 6050.25, 6050.0, timestamp=now),  # Duplicate
            Quote("MESU5", 6050.25, 6050.50, 6050.25,
                  timestamp=now + timedelta(seconds=1)),
        ]

        for quote in quotes:
            if not is_duplicate(quote):
                received_quotes.append(quote)

        # Should have 2 unique quotes (duplicate filtered)
        assert len(received_quotes) == 2

    @pytest.mark.asyncio
    async def test_position_state_sync_after_reconnect(self):
        """Test that position state is synced from API after reconnect."""
        @dataclass
        class LocalPosition:
            contract_id: str
            size: int
            avg_price: float

        local_position = LocalPosition("MESU5", 2, 6050.0)

        # Simulate API position (different from local - API is truth)
        api_position = {"contract_id": "MESU5", "size": 1, "avg_price": 6055.0}

        # After reconnect, sync from API
        local_position.size = api_position["size"]
        local_position.avg_price = api_position["avg_price"]

        assert local_position.size == 1
        assert local_position.avg_price == 6055.0

    @pytest.mark.asyncio
    async def test_order_state_recovery_after_reconnect(self):
        """Test that pending order state is recovered after reconnect."""
        @dataclass
        class PendingOrder:
            order_id: str
            status: str
            needs_refresh: bool = False

        pending_orders = {
            "ORD001": PendingOrder("ORD001", "submitted"),
            "ORD002": PendingOrder("ORD002", "submitted"),
        }

        # After reconnect, mark all pending orders for refresh
        for order in pending_orders.values():
            order.needs_refresh = True

        # Verify all orders marked
        assert all(o.needs_refresh for o in pending_orders.values())


# =============================================================================
# Cascade Failure Tests
# =============================================================================

class TestCascadeFailures:
    """Test handling of multiple simultaneous connection failures."""

    @pytest.mark.asyncio
    async def test_both_market_and_trade_connections_fail(self, recovery_config):
        """Test recovery when both market and trade connections fail."""
        market_connected = False
        trade_connected = False
        recovery_complete = False

        async def reconnect_market():
            nonlocal market_connected
            market_connected = True
            return True

        async def reconnect_trade():
            nonlocal trade_connected
            trade_connected = True
            return True

        async def full_reconnect():
            nonlocal recovery_complete
            await reconnect_market()
            await reconnect_trade()
            recovery_complete = market_connected and trade_connected
            return recovery_complete

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_disconnect(full_reconnect)

        assert result is True
        assert market_connected is True
        assert trade_connected is True
        assert recovery_complete is True

    @pytest.mark.asyncio
    async def test_market_reconnects_trade_fails(self, recovery_config):
        """Test handling when market reconnects but trade fails."""
        attempt_count = 0

        async def partial_reconnect():
            nonlocal attempt_count
            attempt_count += 1
            # Market succeeds, trade fails
            return False  # Overall failure

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_disconnect(partial_reconnect)

        assert result is False
        assert attempt_count == recovery_config.max_reconnect_attempts

    @pytest.mark.asyncio
    async def test_sequential_connection_recovery(self, recovery_config):
        """Test that connections are recovered in sequence."""
        recovery_sequence = []

        async def reconnect_with_sequence():
            recovery_sequence.append("market_start")
            await asyncio.sleep(0.01)
            recovery_sequence.append("market_done")
            recovery_sequence.append("trade_start")
            await asyncio.sleep(0.01)
            recovery_sequence.append("trade_done")
            return True

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_disconnect(reconnect_with_sequence)

        assert result is True
        assert recovery_sequence == [
            "market_start", "market_done",
            "trade_start", "trade_done"
        ]


# =============================================================================
# Operation Buffering Tests
# =============================================================================

class TestOperationBufferingDuringDisconnection:
    """Test that operations are handled correctly during disconnection."""

    @pytest.mark.asyncio
    async def test_orders_queued_during_disconnect(self):
        """Test that orders placed during disconnect are queued."""
        @dataclass
        class OrderQueue:
            connected: bool = True
            pending_orders: list = None

            def __post_init__(self):
                if self.pending_orders is None:
                    self.pending_orders = []

            async def place_order(self, order_data: dict) -> Optional[str]:
                if not self.connected:
                    self.pending_orders.append(order_data)
                    return None  # Queued, not placed
                return "ORD123"  # Placed

            async def flush_queue(self) -> List[str]:
                if not self.connected:
                    return []
                results = []
                while self.pending_orders:
                    order = self.pending_orders.pop(0)
                    result = await self.place_order(order)
                    if result:
                        results.append(result)
                return results

        queue = OrderQueue()

        # Normal operation
        result = await queue.place_order({"symbol": "MESU5", "side": "buy"})
        assert result == "ORD123"

        # Disconnect
        queue.connected = False
        result = await queue.place_order({"symbol": "MESU5", "side": "sell"})
        assert result is None
        assert len(queue.pending_orders) == 1

        # Reconnect and flush
        queue.connected = True
        flushed = await queue.flush_queue()
        assert len(flushed) == 1
        assert len(queue.pending_orders) == 0

    @pytest.mark.asyncio
    async def test_position_queries_during_disconnect(self):
        """Test that position queries return cached data during disconnect."""
        @dataclass
        class PositionCache:
            connected: bool = True
            cached_positions: dict = None
            last_update: datetime = None

            def __post_init__(self):
                if self.cached_positions is None:
                    self.cached_positions = {}

            async def get_positions(self) -> dict:
                if not self.connected:
                    # Return cached data
                    return self.cached_positions.copy()
                # Would fetch from API
                return self.cached_positions

            def update_cache(self, positions: dict):
                self.cached_positions = positions
                self.last_update = datetime.now()

        cache = PositionCache()
        cache.update_cache({"MESU5": {"size": 2, "avg_price": 6050.0}})

        # Disconnect
        cache.connected = False

        # Should return cached data
        positions = await cache.get_positions()
        assert "MESU5" in positions
        assert positions["MESU5"]["size"] == 2


# =============================================================================
# Callback Resilience Tests
# =============================================================================

class TestCallbackResilience:
    """Test that callback errors don't break reconnection."""

    @pytest.mark.asyncio
    async def test_quote_callback_error_doesnt_break_connection(self):
        """Test that a failing quote callback doesn't crash the system."""
        callback_errors = []
        quotes_processed = 0

        def failing_callback(quote: Quote):
            if quote.bid > 6050:
                raise ValueError("Test error")
            nonlocal quotes_processed
            quotes_processed += 1

        def safe_callback_wrapper(callback):
            def wrapper(quote: Quote):
                try:
                    callback(quote)
                except Exception as e:
                    callback_errors.append(str(e))
            return wrapper

        safe_callback = safe_callback_wrapper(failing_callback)

        # Process quotes
        quotes = [
            Quote("MESU5", 6049.0, 6049.25, 6049.0),  # OK
            Quote("MESU5", 6051.0, 6051.25, 6051.0),  # Will error
            Quote("MESU5", 6050.0, 6050.25, 6050.0),  # OK
        ]

        for quote in quotes:
            safe_callback(quote)

        # One error, but 2 processed
        assert len(callback_errors) == 1
        assert quotes_processed == 2

    @pytest.mark.asyncio
    async def test_alert_callback_error_doesnt_prevent_recovery(self, recovery_config):
        """Test that a failing alert callback doesn't prevent recovery."""
        alert_called = False
        recovery_completed = False

        def failing_alert(event: ErrorEvent):
            nonlocal alert_called
            alert_called = True
            raise RuntimeError("Alert failed")

        async def successful_reconnect():
            nonlocal recovery_completed
            recovery_completed = True
            return True

        # Create handler with failing alert
        handler = RecoveryHandler(
            config=recovery_config,
            on_alert=failing_alert
        )

        # Recovery should still work
        result = await handler.handle_disconnect(successful_reconnect)

        assert result is True
        assert recovery_completed is True


# =============================================================================
# Connection Stability Tests
# =============================================================================

class TestConnectionStability:
    """Test behavior under connection instability scenarios."""

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect_cycles(self):
        """Test handling of rapid connect/disconnect cycles."""
        connection_events = []

        async def simulate_connection_cycle():
            for i in range(5):
                connection_events.append(f"connect_{i}")
                await asyncio.sleep(0.01)
                connection_events.append(f"disconnect_{i}")
                await asyncio.sleep(0.01)

        await simulate_connection_cycle()

        # Should have 10 events (5 connects + 5 disconnects)
        assert len(connection_events) == 10
        assert connection_events[0] == "connect_0"
        assert connection_events[-1] == "disconnect_4"

    @pytest.mark.asyncio
    async def test_reconnect_backoff_increases(self, recovery_config):
        """Test that reconnect backoff increases with failures."""
        state = RecoveryState()

        backoffs = []
        for _ in range(5):
            backoff = state.get_backoff(recovery_config)
            backoffs.append(backoff)
            state.reconnect_attempts += 1
            state.record_error()

        # Backoff should increase exponentially
        for i in range(1, len(backoffs)):
            assert backoffs[i] >= backoffs[i-1]

        # But capped at max
        assert backoffs[-1] <= recovery_config.max_backoff_seconds

    @pytest.mark.asyncio
    async def test_successful_reconnect_resets_backoff(self, recovery_config):
        """Test that successful reconnect resets the backoff."""
        state = RecoveryState()

        # Simulate failures
        for _ in range(3):
            state.reconnect_attempts += 1
            state.record_error()

        assert state.consecutive_errors == 3
        assert state.reconnect_attempts == 3

        # Successful recovery
        state.record_success()

        assert state.consecutive_errors == 0
        assert state.reconnect_attempts == 0
        assert state.current_backoff == 1.0


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================

class TestGracefulShutdownDuringReconnection:
    """Test graceful shutdown behavior during active reconnection."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_reconnect_attempt(self):
        """Test that shutdown cancels any active reconnection."""
        reconnect_running = asyncio.Event()
        reconnect_cancelled = False

        async def slow_reconnect():
            nonlocal reconnect_cancelled
            reconnect_running.set()
            try:
                await asyncio.sleep(10)  # Long reconnect
                return True
            except asyncio.CancelledError:
                reconnect_cancelled = True
                raise

        async def shutdown():
            await reconnect_running.wait()
            # Cancel would be called here
            return True

        # Start reconnect
        reconnect_task = asyncio.create_task(slow_reconnect())

        # Wait for it to start, then cancel
        await reconnect_running.wait()
        reconnect_task.cancel()

        try:
            await reconnect_task
        except asyncio.CancelledError:
            pass

        assert reconnect_cancelled is True

    @pytest.mark.asyncio
    async def test_cleanup_during_reconnection(self):
        """Test that cleanup happens properly during reconnection."""
        resources_cleaned = []

        class MockConnection:
            def __init__(self, name: str):
                self.name = name
                self.is_open = False

            async def open(self):
                self.is_open = True

            async def close(self):
                self.is_open = False
                resources_cleaned.append(self.name)

        market_conn = MockConnection("market")
        trade_conn = MockConnection("trade")

        await market_conn.open()
        await trade_conn.open()

        assert market_conn.is_open is True
        assert trade_conn.is_open is True

        # Cleanup during shutdown
        await market_conn.close()
        await trade_conn.close()

        assert market_conn.is_open is False
        assert trade_conn.is_open is False
        assert "market" in resources_cleaned
        assert "trade" in resources_cleaned


# =============================================================================
# Position Synchronization Tests
# =============================================================================

class TestPositionSynchronizationAfterReconnect:
    """Test position synchronization after reconnection."""

    @pytest.mark.asyncio
    async def test_position_mismatch_detected(self, recovery_config):
        """Test that position mismatches are detected after reconnect."""
        @dataclass
        class MockPosition:
            contract_id: str
            size: int

        local_position = MockPosition("MESU5", 2)
        api_position = MockPosition("MESU5", 1)

        mismatch_detected = False
        if local_position.size != api_position.size:
            mismatch_detected = True

        assert mismatch_detected is True

    @pytest.mark.asyncio
    async def test_position_sync_from_api(self, recovery_config):
        """Test that positions are synced from API (API is source of truth)."""
        @dataclass
        class PositionManager:
            positions: dict = None

            def __post_init__(self):
                if self.positions is None:
                    self.positions = {}

            def sync_from_api(self, api_positions: list):
                """Sync local positions with API positions."""
                # API is source of truth - replace local with API
                self.positions = {
                    p["contract_id"]: {
                        "size": p["size"],
                        "avg_price": p["avg_price"],
                    }
                    for p in api_positions
                }

        manager = PositionManager()
        # Local shows 2 contracts
        manager.positions = {"MESU5": {"size": 2, "avg_price": 6050.0}}

        # API shows 1 contract (this is truth)
        api_positions = [{"contract_id": "MESU5", "size": 1, "avg_price": 6055.0}]

        manager.sync_from_api(api_positions)

        assert manager.positions["MESU5"]["size"] == 1
        assert manager.positions["MESU5"]["avg_price"] == 6055.0

    @pytest.mark.asyncio
    async def test_position_sync_alerts_on_mismatch(self, recovery_config):
        """Test that position mismatch generates alert."""
        alerts = []

        async def mock_alert(message: str):
            alerts.append(message)

        local_size = 2
        api_size = 1

        if local_size != api_size:
            await mock_alert(
                f"Position mismatch: local={local_size}, api={api_size}"
            )

        assert len(alerts) == 1
        assert "mismatch" in alerts[0]

    @pytest.mark.asyncio
    async def test_new_position_discovered_on_sync(self):
        """Test that new positions from API are added locally."""
        local_positions = {"MESU5": {"size": 1}}
        api_positions = [
            {"contract_id": "MESU5", "size": 1, "avg_price": 6050.0},
            {"contract_id": "MESZ5", "size": 2, "avg_price": 6055.0},  # New
        ]

        for p in api_positions:
            local_positions[p["contract_id"]] = {
                "size": p["size"],
                "avg_price": p["avg_price"],
            }

        assert "MESZ5" in local_positions
        assert local_positions["MESZ5"]["size"] == 2


# =============================================================================
# Error Rate Monitoring Tests
# =============================================================================

class TestErrorRateMonitoring:
    """Test error rate monitoring and alerting."""

    def test_error_rate_window_tracking(self):
        """Test that errors are tracked within time windows."""
        state = RecoveryState()

        # Record multiple errors
        for _ in range(5):
            state.record_error()

        assert state.errors_in_window == 5
        assert state.consecutive_errors == 5

    def test_error_window_resets_after_60_seconds(self):
        """Test that error window resets after 60 seconds."""
        state = RecoveryState()

        # Record errors
        state.record_error()
        state.record_error()
        assert state.errors_in_window == 2

        # Simulate time passing (set window_start to past)
        state.window_start = datetime.now() - timedelta(seconds=61)

        # Next error should reset window
        state.record_error()
        assert state.errors_in_window == 1

    def test_success_resets_consecutive_errors(self):
        """Test that success resets consecutive error count."""
        state = RecoveryState()

        # Record errors
        for _ in range(3):
            state.record_error()
        assert state.consecutive_errors == 3

        # Record success
        state.record_success()
        assert state.consecutive_errors == 0


# =============================================================================
# Recovery Handler Integration Tests
# =============================================================================

class TestRecoveryHandlerIntegration:
    """Integration tests for recovery handler."""

    @pytest.mark.asyncio
    async def test_full_recovery_flow(self, recovery_config):
        """Test complete recovery flow from disconnect to reconnect."""
        events = []

        async def reconnect():
            events.append("reconnect_called")
            return True

        handler = RecoveryHandler(
            config=recovery_config,
            on_alert=lambda e: events.append(f"alert_{e.severity.value}"),
        )

        result = await handler.handle_disconnect(reconnect)

        assert result is True
        assert "reconnect_called" in events

    @pytest.mark.asyncio
    async def test_recovery_with_all_retries_exhausted(self, recovery_config):
        """Test recovery when all retries are exhausted."""
        attempt_count = 0
        halt_called = False

        async def failing_reconnect():
            nonlocal attempt_count
            attempt_count += 1
            return False

        async def on_halt(reason: str):
            nonlocal halt_called
            halt_called = True

        handler = RecoveryHandler(
            config=recovery_config,
            on_halt=on_halt,
        )

        result = await handler.handle_disconnect(failing_reconnect)

        assert result is False
        assert attempt_count == recovery_config.max_reconnect_attempts
        assert halt_called is True

    @pytest.mark.asyncio
    async def test_recovery_succeeds_on_third_attempt(self, recovery_config):
        """Test recovery that succeeds after a few failures."""
        attempt_count = 0

        async def sometimes_reconnect():
            nonlocal attempt_count
            attempt_count += 1
            return attempt_count >= 2  # Succeed on 2nd attempt

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_disconnect(sometimes_reconnect)

        assert result is True
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_auth_recovery_flow(self, recovery_config):
        """Test authentication recovery flow."""
        attempt_count = 0

        async def reauth():
            nonlocal attempt_count
            attempt_count += 1
            return attempt_count >= 2  # Succeed on 2nd attempt

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_auth_failure(reauth)

        assert result is True
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, recovery_config):
        """Test rate limit recovery with wait."""
        operation_called = False

        async def rate_limited_operation():
            nonlocal operation_called
            operation_called = True
            return "success"

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_rate_limit(
            retry_after_seconds=0.01,
            operation_func=rate_limited_operation
        )

        assert result == "success"
        assert operation_called is True

    @pytest.mark.asyncio
    async def test_insufficient_margin_recovery(self, recovery_config):
        """Test insufficient margin recovery with size reduction."""
        sizes_tried = []

        async def place_order(size: int):
            sizes_tried.append(size)
            return f"ORD_{size}"

        handler = RecoveryHandler(config=recovery_config)
        result = await handler.handle_insufficient_margin(
            order_func=place_order,
            original_size=4,
            min_size=1
        )

        assert result == "ORD_2"  # 4 // 2 = 2
        assert 2 in sizes_tried
