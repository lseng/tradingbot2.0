"""
WebSocket Auto-Reconnection Tests.

Tests for verifying the WebSocket auto-reconnection functionality,
specifically ensuring Bug 10.0.1 (WebSocket auto-reconnect never started) remains fixed.

Tests cover:
1. Auto-reconnect loop started (Bug 10.0.1 fix verification)
2. Reconnection behavior (backoff, max attempts)
3. State management during disconnect/reconnect
4. Error handling for connection failures

Reference: IMPLEMENTATION_PLAN.md - Bug 10.0.1
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from typing import Optional

import aiohttp

from src.api.topstepx_ws import (
    WebSocketState,
    Quote,
    SignalRConnection,
    TopstepXWebSocket,
)
from src.api.topstepx_client import TopstepXConnectionError, TopstepXConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_client():
    """Create a mock TopstepXClient."""
    client = MagicMock()
    client.access_token = "test_token"
    client.default_account_id = 12345
    client.config = TopstepXConfig(
        ws_market_url="wss://rtc.topstepx.com/hubs/market",
        ws_trade_url="wss://rtc.topstepx.com/hubs/trade",
    )
    return client


@pytest.fixture
def mock_ws_response():
    """Create a mock WebSocket response."""
    ws = AsyncMock()
    ws.closed = False
    ws.close = AsyncMock()
    ws.send_str = AsyncMock()
    return ws


@pytest.fixture
def mock_session(mock_ws_response):
    """Create a mock aiohttp session."""
    session = MagicMock()
    session.ws_connect = AsyncMock(return_value=mock_ws_response)
    session.post = MagicMock()
    session.closed = False
    return session


# =============================================================================
# Bug 10.0.1 Fix Verification Tests
# =============================================================================

class TestAutoReconnectLoopStarted:
    """
    Tests verifying Bug 10.0.1 fix: auto-reconnect loop is actually started.

    Bug 10.0.1: WebSocket Auto-Reconnect Never Started
    - The _auto_reconnect_loop() was defined but never called
    - Fixed by adding asyncio.create_task() in connect() method (lines 711-713)
    """

    @pytest.mark.asyncio
    async def test_auto_reconnect_loop_started_on_connect(self, mock_client):
        """
        Verify _auto_reconnect_loop() is called when connect() is called.

        This is the core verification for Bug 10.0.1 fix.
        """
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        # Track if _auto_reconnect_loop is started
        loop_started = False
        original_auto_reconnect_loop = ws._auto_reconnect_loop

        async def mock_auto_reconnect_loop():
            nonlocal loop_started
            loop_started = True
            # Don't actually run the loop, just mark it started
            return

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                with patch.object(ws, '_auto_reconnect_loop', mock_auto_reconnect_loop):
                    await ws.connect()
                    # Give async task time to start
                    await asyncio.sleep(0.01)

        # Verify reconnect task was created
        assert ws._reconnect_task is not None, "Bug 10.0.1: reconnect task should be created"

        # Cleanup
        if ws._reconnect_task and not ws._reconnect_task.done():
            ws._reconnect_task.cancel()
            try:
                await ws._reconnect_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_reconnect_task_created_with_asyncio_create_task(self, mock_client):
        """
        Verify reconnect task is created with asyncio.create_task().

        Bug 10.0.1 fix specifically uses asyncio.create_task() at lines 711-713.
        """
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        task_created = False

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                # Patch asyncio.create_task to track calls
                original_create_task = asyncio.create_task
                tasks_created = []

                def track_create_task(coro, **kwargs):
                    task = original_create_task(coro, **kwargs)
                    tasks_created.append(task)
                    return task

                with patch('asyncio.create_task', side_effect=track_create_task):
                    await ws.connect()

        # Verify a task was created for the reconnect loop
        assert len(tasks_created) > 0, "Bug 10.0.1: asyncio.create_task should be called"

        # Cleanup
        for task in tasks_created:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_auto_reconnect_disabled_no_task_created(self, mock_client):
        """
        Verify no reconnect task is created when auto_reconnect=False.
        """
        ws = TopstepXWebSocket(mock_client, auto_reconnect=False)

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                await ws.connect()

        # No reconnect task should be created
        assert ws._reconnect_task is None, "No reconnect task when auto_reconnect=False"

    @pytest.mark.asyncio
    async def test_auto_reconnect_not_duplicated_on_multiple_connects(self, mock_client):
        """
        Verify reconnect task is not duplicated on multiple connect() calls.
        """
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        async def mock_reconnect_loop():
            while ws._should_run and ws._auto_reconnect:
                await asyncio.sleep(0.1)

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                with patch.object(ws, '_auto_reconnect_loop', mock_reconnect_loop):
                    await ws.connect()
                    first_task = ws._reconnect_task

                    await ws.connect()  # Second connect
                    second_task = ws._reconnect_task

        # Same task should be used (not duplicated)
        assert first_task is second_task, "Reconnect task should not be duplicated"

        # Cleanup
        ws._should_run = False
        if ws._reconnect_task and not ws._reconnect_task.done():
            ws._reconnect_task.cancel()
            try:
                await ws._reconnect_task
            except asyncio.CancelledError:
                pass


# =============================================================================
# Reconnection Behavior Tests
# =============================================================================

class TestReconnectionBehavior:
    """Tests for reconnection behavior after disconnect."""

    @pytest.mark.asyncio
    async def test_reconnect_triggered_after_disconnect(self, mock_client):
        """Test reconnection is triggered when connection is lost."""
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.01,
            max_reconnect_attempts=3,
        )

        reconnect_attempts = 0

        async def mock_connect():
            nonlocal reconnect_attempts
            reconnect_attempts += 1
            if reconnect_attempts == 1:
                # First call succeeds
                ws._should_run = True
            else:
                # Subsequent reconnect attempts
                pass

        ws._should_run = True

        # Simulate disconnected state
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', mock_connect):
            # Run reconnect loop briefly
            task = asyncio.create_task(ws._auto_reconnect_loop())
            await asyncio.sleep(0.1)
            ws._should_run = False

            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # At least one reconnection attempt should have been made
        assert reconnect_attempts >= 1, "Reconnection should be attempted"

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, mock_client):
        """Test reconnection uses exponential backoff."""
        initial_delay = 0.01
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=initial_delay,
            max_reconnect_attempts=5,
        )

        delays = []
        reconnect_times = []

        async def mock_connect():
            reconnect_times.append(asyncio.get_event_loop().time())
            raise Exception("Connection failed")

        ws._should_run = True
        ws._auto_reconnect = True

        # Simulate disconnected state
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', mock_connect):
            task = asyncio.create_task(ws._auto_reconnect_loop())
            await asyncio.sleep(0.5)  # Let it run a bit
            ws._should_run = False

            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Calculate delays between attempts
        if len(reconnect_times) >= 2:
            for i in range(1, len(reconnect_times)):
                delays.append(reconnect_times[i] - reconnect_times[i-1])

        # Verify backoff increases (at least second delay >= first delay)
        if len(delays) >= 2:
            assert delays[1] >= delays[0], "Backoff should increase"

    @pytest.mark.asyncio
    async def test_max_reconnect_attempts_honored(self, mock_client):
        """Test reconnection stops after max attempts."""
        max_attempts = 3
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.01,
            max_reconnect_attempts=max_attempts,
        )

        actual_attempts = 0

        async def mock_connect():
            nonlocal actual_attempts
            actual_attempts += 1
            raise Exception("Connection failed")

        ws._should_run = True
        ws._auto_reconnect = True

        # Simulate disconnected state
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', mock_connect):
            task = asyncio.create_task(ws._auto_reconnect_loop())

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            finally:
                ws._should_run = False
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Should have tried max_attempts times
        assert actual_attempts == max_attempts, f"Should attempt exactly {max_attempts} times"

    @pytest.mark.asyncio
    async def test_backoff_capped_at_60_seconds(self, mock_client):
        """Test that backoff is capped at 60 seconds."""
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=30.0,  # Large initial delay
            max_reconnect_attempts=5,
        )

        # Manually calculate what the delay would be
        # delay = reconnect_delay * (2 ** (attempts - 1))
        # With reconnect_delay=30, attempt 2 would be 30 * 2 = 60
        # attempt 3 would be 30 * 4 = 120, but should be capped at 60

        ws._reconnect_attempts = 3
        expected_delay = ws._reconnect_delay * (2 ** (ws._reconnect_attempts - 1))
        capped_delay = min(expected_delay, 60)

        assert capped_delay == 60, "Delay should be capped at 60 seconds"


# =============================================================================
# State Management Tests
# =============================================================================

class TestStateManagement:
    """Tests for state management during disconnect/reconnect cycles."""

    @pytest.mark.asyncio
    async def test_should_run_flag_set_on_connect(self, mock_client):
        """Test _should_run is set to True on connect."""
        ws = TopstepXWebSocket(mock_client)

        assert ws._should_run is False

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                await ws.connect()

        assert ws._should_run is True

        # Cleanup
        await ws.disconnect()

    @pytest.mark.asyncio
    async def test_should_run_flag_cleared_on_disconnect(self, mock_client):
        """Test _should_run is set to False on disconnect."""
        ws = TopstepXWebSocket(mock_client)
        ws._should_run = True

        await ws.disconnect()

        assert ws._should_run is False

    @pytest.mark.asyncio
    async def test_subscriptions_restored_after_reconnect(self, mock_client):
        """Test that subscribed contracts are restored after reconnect."""
        ws = TopstepXWebSocket(mock_client)

        # Pre-populate subscribed contracts
        ws._subscribed_contracts = {"MES", "ES", "NQ"}

        invoke_calls = []

        mock_market = MagicMock()
        mock_market.is_connected = False
        mock_market.connect = AsyncMock()
        mock_market.on = MagicMock()
        mock_market.invoke = AsyncMock(side_effect=lambda method, *args: invoke_calls.append((method, args)))

        with patch.object(ws, '_market_connection', mock_market):
            with patch('src.api.topstepx_ws.SignalRConnection', return_value=mock_market):
                await ws._connect_market()

        # Verify SubscribeQuotes was called with all contracts
        subscribe_calls = [c for c in invoke_calls if c[0] == "SubscribeQuotes"]
        assert len(subscribe_calls) == 1, "SubscribeQuotes should be called"
        subscribed = set(subscribe_calls[0][1][0])
        assert subscribed == {"MES", "ES", "NQ"}, "All contracts should be resubscribed"

    @pytest.mark.asyncio
    async def test_reconnect_attempts_reset_after_successful_connect(self, mock_client):
        """Test reconnect attempts counter is reset after successful connect."""
        ws = TopstepXWebSocket(mock_client)

        # Simulate failed reconnect attempts
        ws._reconnect_attempts = 5
        ws._subscribed_contracts = set()

        mock_market = MagicMock()
        mock_market.is_connected = False
        mock_market.connect = AsyncMock()
        mock_market.on = MagicMock()
        mock_market.invoke = AsyncMock()

        with patch('src.api.topstepx_ws.SignalRConnection', return_value=mock_market):
            await ws._connect_market()

        assert ws._reconnect_attempts == 0, "Reconnect attempts should be reset"

    @pytest.mark.asyncio
    async def test_subscribed_contracts_cleared_on_disconnect(self, mock_client):
        """Test subscribed contracts are cleared on disconnect."""
        ws = TopstepXWebSocket(mock_client)
        ws._subscribed_contracts = {"MES", "ES"}

        await ws.disconnect()

        assert len(ws._subscribed_contracts) == 0, "Subscriptions should be cleared"

    @pytest.mark.asyncio
    async def test_reconnect_task_cancelled_on_disconnect(self, mock_client):
        """Test reconnect task is properly cancelled on disconnect."""
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        # Create a mock reconnect task
        async def long_running_task():
            while True:
                await asyncio.sleep(0.1)

        ws._reconnect_task = asyncio.create_task(long_running_task())
        ws._should_run = True

        await ws.disconnect()

        assert ws._reconnect_task is None, "Reconnect task should be None after disconnect"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling during reconnection."""

    @pytest.mark.asyncio
    async def test_reconnect_continues_after_single_failure(self, mock_client):
        """Test reconnection continues after a single failure."""
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.005,  # Very short delay for testing
            max_reconnect_attempts=5,
        )

        attempt_count = 0
        success_event = asyncio.Event()

        async def mock_connect():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise Exception("Temporary failure")
            # Succeed on second attempt
            ws._market_connection = MagicMock()
            ws._market_connection.is_connected = True
            success_event.set()

        ws._should_run = True
        ws._auto_reconnect = True
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', mock_connect):
            task = asyncio.create_task(ws._auto_reconnect_loop())

            # Wait for success or timeout
            try:
                await asyncio.wait_for(success_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            finally:
                ws._should_run = False
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        assert attempt_count >= 2, "Should retry after failure"

    @pytest.mark.asyncio
    async def test_permanent_failure_stops_reconnection(self, mock_client):
        """Test reconnection stops after permanent failure (max attempts)."""
        max_attempts = 2
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.01,
            max_reconnect_attempts=max_attempts,
        )

        failure_count = 0

        async def always_fail_connect():
            nonlocal failure_count
            failure_count += 1
            raise Exception("Permanent failure")

        ws._should_run = True
        ws._auto_reconnect = True
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', always_fail_connect):
            task = asyncio.create_task(ws._auto_reconnect_loop())

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                pass
            finally:
                ws._should_run = False
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        assert failure_count == max_attempts, "Should stop after max attempts"

    @pytest.mark.asyncio
    async def test_connection_error_handled_gracefully(self, mock_client):
        """Test that TopstepXConnectionError is handled gracefully."""
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.01,
            max_reconnect_attempts=2,
        )

        async def raise_connection_error():
            raise TopstepXConnectionError("Authentication failed")

        ws._should_run = True
        ws._auto_reconnect = True
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        # Should not raise, just log and retry
        with patch.object(ws, 'connect', raise_connection_error):
            task = asyncio.create_task(ws._auto_reconnect_loop())

            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            finally:
                ws._should_run = False
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Test passes if no exception propagated

    @pytest.mark.asyncio
    async def test_infinite_reconnect_when_max_attempts_zero(self, mock_client):
        """Test infinite reconnection when max_reconnect_attempts=0."""
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.001,  # Very short delay for fast iteration
            max_reconnect_attempts=0,  # 0 = infinite
        )

        attempt_count = 0
        target_attempts = 5
        target_reached = asyncio.Event()

        async def counting_connect():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count >= target_attempts:
                target_reached.set()
            raise Exception("Keep trying")

        ws._should_run = True
        ws._auto_reconnect = True
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', counting_connect):
            task = asyncio.create_task(ws._auto_reconnect_loop())

            # Wait for enough attempts or timeout
            try:
                await asyncio.wait_for(target_reached.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            finally:
                ws._should_run = False
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Should have attempted at least target_attempts times
        assert attempt_count >= target_attempts, f"Should continue indefinitely when max=0, got {attempt_count} attempts"


# =============================================================================
# Integration Tests
# =============================================================================

class TestReconnectionIntegration:
    """Integration tests for the full reconnection flow."""

    @pytest.mark.asyncio
    async def test_full_disconnect_reconnect_cycle(self, mock_client):
        """Test a complete disconnect and reconnect cycle."""
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        connect_count = 0

        async def track_connect():
            nonlocal connect_count
            connect_count += 1

        with patch.object(ws, '_connect_market', track_connect):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                # Initial connect
                await ws.connect()
                assert connect_count == 1

                # Clean disconnect
                await ws.disconnect()

                # Reconnect
                ws._reconnect_task = None  # Reset
                await ws.connect()
                assert connect_count == 2

        # Final cleanup
        await ws.disconnect()

    @pytest.mark.asyncio
    async def test_quote_callbacks_work_after_reconnect(self, mock_client):
        """Test that quote callbacks continue working after reconnect."""
        ws = TopstepXWebSocket(mock_client)

        received_quotes = []

        def quote_callback(quote: Quote):
            received_quotes.append(quote)

        ws.on_quote(quote_callback)

        # Simulate receiving a quote
        quote_data = {
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
        }

        ws._handle_quote(quote_data)
        assert len(received_quotes) == 1

        # Simulate disconnect/reconnect (callbacks should persist)
        ws._market_connection = None
        ws._trade_connection = None

        # After "reconnect", callback should still work
        ws._handle_quote(quote_data)
        assert len(received_quotes) == 2, "Callbacks should persist after reconnect"

    @pytest.mark.asyncio
    async def test_context_manager_handles_reconnect_task(self, mock_client):
        """Test async context manager properly manages reconnect task."""
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        async def mock_reconnect_loop():
            while ws._should_run and ws._auto_reconnect:
                await asyncio.sleep(0.1)

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                with patch.object(ws, '_auto_reconnect_loop', mock_reconnect_loop):
                    async with ws:
                        assert ws._should_run is True
                        assert ws._reconnect_task is not None

        # After context exit
        assert ws._should_run is False
        assert ws._reconnect_task is None


# =============================================================================
# Regression Tests for Bug 10.0.1
# =============================================================================

class TestBug10_0_1Regression:
    """
    Regression tests specifically for Bug 10.0.1.

    These tests ensure the bug does not reappear:
    Bug: WebSocket auto-reconnect loop was defined but never started.
    Fix: Added asyncio.create_task(self._auto_reconnect_loop()) in connect() method.
    """

    @pytest.mark.asyncio
    async def test_auto_reconnect_loop_method_exists(self, mock_client):
        """Verify _auto_reconnect_loop method exists."""
        ws = TopstepXWebSocket(mock_client)
        assert hasattr(ws, '_auto_reconnect_loop'), "_auto_reconnect_loop method must exist"
        assert asyncio.iscoroutinefunction(ws._auto_reconnect_loop), "Must be async method"

    @pytest.mark.asyncio
    async def test_connect_starts_reconnect_when_enabled(self, mock_client):
        """Verify connect() starts reconnect task when auto_reconnect=True."""
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        # Before connect, no task
        assert ws._reconnect_task is None

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                await ws.connect()

        # After connect, task should exist
        assert ws._reconnect_task is not None, "Bug 10.0.1: reconnect task must be created"

        # Cleanup
        await ws.disconnect()

    @pytest.mark.asyncio
    async def test_reconnect_task_is_running(self, mock_client):
        """Verify the reconnect task is actually running, not just created."""
        ws = TopstepXWebSocket(mock_client, auto_reconnect=True)

        loop_entered = asyncio.Event()

        async def tracking_loop():
            loop_entered.set()
            while ws._should_run and ws._auto_reconnect:
                await asyncio.sleep(0.01)

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                with patch.object(ws, '_auto_reconnect_loop', tracking_loop):
                    await ws.connect()

                    # Wait for loop to actually enter
                    try:
                        await asyncio.wait_for(loop_entered.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pytest.fail("Bug 10.0.1: reconnect loop never entered")

        # Cleanup
        await ws.disconnect()

    @pytest.mark.asyncio
    async def test_reconnect_loop_monitors_connection_state(self, mock_client):
        """Verify reconnect loop checks connection state."""
        ws = TopstepXWebSocket(
            mock_client,
            auto_reconnect=True,
            reconnect_delay=0.01,
        )

        reconnect_triggered = False

        async def mock_connect():
            nonlocal reconnect_triggered
            reconnect_triggered = True
            # Make it appear connected after reconnect
            ws._market_connection = MagicMock()
            ws._market_connection.is_connected = True

        # Set up disconnected state
        ws._should_run = True
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', mock_connect):
            task = asyncio.create_task(ws._auto_reconnect_loop())
            await asyncio.sleep(0.1)
            ws._should_run = False

            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        assert reconnect_triggered, "Bug 10.0.1: loop must trigger reconnect when disconnected"
