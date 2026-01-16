"""
Extended tests for Recovery module to increase coverage.

Tests cover:
- RecoveryState: window reset, backoff calculation
- RecoveryHandler: all handle_* methods
- Error history management
- Alert callbacks
- with_retry and with_timeout decorators
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from src.trading.recovery import (
    RecoveryHandler,
    RecoveryConfig,
    RecoveryState,
    ErrorEvent,
    ErrorSeverity,
    ErrorCategory,
    with_retry,
    with_timeout,
)


# =============================================================================
# RecoveryState Tests
# =============================================================================

class TestRecoveryStateWindowReset:
    """Tests for RecoveryState window reset logic."""

    def test_record_error_increments_consecutive(self):
        """Test record_error increments consecutive errors."""
        state = RecoveryState()
        assert state.consecutive_errors == 0

        state.record_error()
        assert state.consecutive_errors == 1

        state.record_error()
        assert state.consecutive_errors == 2

    def test_record_error_tracks_window(self):
        """Test record_error tracks errors in window."""
        state = RecoveryState()
        state.record_error()
        assert state.errors_in_window == 1

    def test_record_error_resets_window_after_60_seconds(self):
        """Test record_error resets window after 60 seconds."""
        state = RecoveryState()

        # Set window start to 61 seconds ago
        state.window_start = datetime.now() - timedelta(seconds=61)
        state.errors_in_window = 5

        state.record_error()

        # Window should reset
        assert state.errors_in_window == 1

    def test_record_error_within_window(self):
        """Test record_error increments within window."""
        state = RecoveryState()

        # Set window start to 30 seconds ago (within 60s window)
        state.window_start = datetime.now() - timedelta(seconds=30)
        state.errors_in_window = 5

        state.record_error()

        # Should increment, not reset
        assert state.errors_in_window == 6

    def test_record_success_resets_counters(self):
        """Test record_success resets all counters."""
        state = RecoveryState()
        state.consecutive_errors = 5
        state.reconnect_attempts = 3
        state.auth_retries = 2
        state.is_recovering = True
        state.current_backoff = 10.0

        state.record_success()

        assert state.consecutive_errors == 0
        assert state.reconnect_attempts == 0
        assert state.auth_retries == 0
        assert state.is_recovering is False
        assert state.current_backoff == 1.0

    def test_get_backoff_exponential(self):
        """Test backoff calculation is exponential."""
        state = RecoveryState()
        config = RecoveryConfig(
            initial_backoff_seconds=1.0,
            max_backoff_seconds=30.0,
            backoff_multiplier=2.0,
        )

        state.reconnect_attempts = 0
        assert state.get_backoff(config) == 1.0

        state.reconnect_attempts = 1
        assert state.get_backoff(config) == 2.0

        state.reconnect_attempts = 2
        assert state.get_backoff(config) == 4.0

        state.reconnect_attempts = 3
        assert state.get_backoff(config) == 8.0

    def test_get_backoff_caps_at_max(self):
        """Test backoff is capped at max."""
        state = RecoveryState()
        config = RecoveryConfig(
            initial_backoff_seconds=1.0,
            max_backoff_seconds=30.0,
            backoff_multiplier=2.0,
        )

        state.reconnect_attempts = 10  # Would be 1024 without cap
        assert state.get_backoff(config) == 30.0


# =============================================================================
# RecoveryHandler Tests
# =============================================================================

class TestRecoveryHandlerInit:
    """Tests for RecoveryHandler initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        handler = RecoveryHandler()

        assert handler.config is not None
        assert handler._on_alert is None
        assert handler._on_halt is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = RecoveryConfig(max_reconnect_attempts=5)
        handler = RecoveryHandler(config=config)

        assert handler.config.max_reconnect_attempts == 5

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        alert_cb = MagicMock()
        halt_cb = AsyncMock()

        handler = RecoveryHandler(on_alert=alert_cb, on_halt=halt_cb)

        assert handler._on_alert == alert_cb
        assert handler._on_halt == halt_cb


class TestRecoveryHandlerDisconnect:
    """Tests for RecoveryHandler.handle_disconnect()."""

    @pytest.mark.asyncio
    async def test_handle_disconnect_success(self):
        """Test successful reconnection."""
        handler = RecoveryHandler()

        reconnect_func = AsyncMock(return_value=True)

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_disconnect(reconnect_func)

        assert result is True
        assert handler._state.is_recovering is False

    @pytest.mark.asyncio
    async def test_handle_disconnect_timeout(self):
        """Test reconnection failure handling when all attempts fail."""
        handler = RecoveryHandler(config=RecoveryConfig(
            max_reconnect_attempts=2,
            reconnect_timeout_seconds=0.01,
        ))

        async def failing_reconnect():
            return False

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_disconnect(failing_reconnect)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_disconnect_exception(self):
        """Test reconnection exception handling."""
        handler = RecoveryHandler(config=RecoveryConfig(
            max_reconnect_attempts=2,
        ))

        reconnect_func = AsyncMock(side_effect=Exception("Connection error"))

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_disconnect(reconnect_func)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_disconnect_max_attempts_calls_halt(self):
        """Test that max attempts triggers halt."""
        halt_cb = AsyncMock()
        handler = RecoveryHandler(
            config=RecoveryConfig(max_reconnect_attempts=1),
            on_halt=halt_cb,
        )

        reconnect_func = AsyncMock(return_value=False)

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_disconnect(reconnect_func)

        assert result is False
        halt_cb.assert_called_once()


class TestRecoveryHandlerAuthFailure:
    """Tests for RecoveryHandler.handle_auth_failure()."""

    @pytest.mark.asyncio
    async def test_handle_auth_failure_success(self):
        """Test successful re-authentication."""
        handler = RecoveryHandler()

        reauth_func = AsyncMock(return_value=True)

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_auth_failure(reauth_func)

        assert result is True
        assert handler._state.auth_retries == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_handle_auth_failure_retry_then_success(self):
        """Test retry then success."""
        handler = RecoveryHandler()

        # Fail first, succeed second
        reauth_func = AsyncMock(side_effect=[False, True])

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_auth_failure(reauth_func)

        assert result is True
        assert reauth_func.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_auth_failure_max_retries(self):
        """Test max auth retries triggers halt."""
        halt_cb = AsyncMock()
        handler = RecoveryHandler(
            config=RecoveryConfig(max_auth_retries=2),
            on_halt=halt_cb,
        )

        reauth_func = AsyncMock(return_value=False)

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_auth_failure(reauth_func)

        assert result is False
        halt_cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_auth_failure_exception(self):
        """Test exception during re-auth."""
        handler = RecoveryHandler(config=RecoveryConfig(max_auth_retries=2))

        reauth_func = AsyncMock(side_effect=Exception("Auth error"))

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_auth_failure(reauth_func)

        assert result is False


class TestRecoveryHandlerOrderRejection:
    """Tests for RecoveryHandler.handle_order_rejection()."""

    @pytest.mark.asyncio
    async def test_handle_order_rejection_logs_error(self):
        """Test order rejection is logged."""
        handler = RecoveryHandler()

        await handler.handle_order_rejection(
            order_id="ORD123",
            error_message="Insufficient margin",
            order_details={"side": "BUY", "size": 5},
        )

        assert len(handler._error_history) == 1
        error = handler._error_history[0]
        assert error.category == ErrorCategory.ORDER
        assert "ORD123" in str(error.details)

    @pytest.mark.asyncio
    async def test_handle_order_rejection_alerts_after_threshold(self):
        """Test alert sent after consecutive error threshold."""
        alert_cb = MagicMock()
        handler = RecoveryHandler(
            config=RecoveryConfig(consecutive_errors_for_alert=3),
            on_alert=alert_cb,
        )

        # First 2 rejections - no alert
        for i in range(2):
            await handler.handle_order_rejection(f"ORD{i}", "Error")

        assert alert_cb.call_count == 0

        # Third rejection - triggers alert
        await handler.handle_order_rejection("ORD3", "Error")

        assert alert_cb.call_count == 1


class TestRecoveryHandlerInsufficientMargin:
    """Tests for RecoveryHandler.handle_insufficient_margin()."""

    @pytest.mark.asyncio
    async def test_handle_insufficient_margin_success(self):
        """Test successful order with reduced size."""
        handler = RecoveryHandler()

        order_func = AsyncMock(return_value={"order_id": "123"})

        result = await handler.handle_insufficient_margin(
            order_func=order_func,
            original_size=4,
        )

        assert result == {"order_id": "123"}
        order_func.assert_called_once_with(2)  # 4 // 2

    @pytest.mark.asyncio
    async def test_handle_insufficient_margin_uses_min_size(self):
        """Test uses min_size when reduction too small."""
        handler = RecoveryHandler()

        order_func = AsyncMock(return_value={"order_id": "123"})

        result = await handler.handle_insufficient_margin(
            order_func=order_func,
            original_size=1,  # Half would be 0
            min_size=1,
        )

        assert result == {"order_id": "123"}
        order_func.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_handle_insufficient_margin_failure(self):
        """Test failure even with reduced size."""
        handler = RecoveryHandler()

        order_func = AsyncMock(side_effect=Exception("Still insufficient"))

        result = await handler.handle_insufficient_margin(
            order_func=order_func,
            original_size=4,
        )

        assert result is None
        assert len(handler._error_history) == 1


class TestRecoveryHandlerRateLimit:
    """Tests for RecoveryHandler.handle_rate_limit()."""

    @pytest.mark.asyncio
    async def test_handle_rate_limit_success(self):
        """Test successful operation after rate limit wait."""
        handler = RecoveryHandler()

        operation_func = AsyncMock(return_value={"success": True})

        with patch('asyncio.sleep', new=AsyncMock()) as mock_sleep:
            result = await handler.handle_rate_limit(
                retry_after_seconds=5.0,
                operation_func=operation_func,
            )

        assert result == {"success": True}
        mock_sleep.assert_called_once_with(5.0)

    @pytest.mark.asyncio
    async def test_handle_rate_limit_failure(self):
        """Test failure after rate limit wait."""
        handler = RecoveryHandler()

        operation_func = AsyncMock(side_effect=Exception("Still rate limited"))

        with patch('asyncio.sleep', new=AsyncMock()):
            result = await handler.handle_rate_limit(
                retry_after_seconds=5.0,
                operation_func=operation_func,
            )

        assert result is None


class TestRecoveryHandlerPositionMismatch:
    """Tests for RecoveryHandler.handle_position_mismatch()."""

    @dataclass
    class MockPosition:
        direction: int
        size: int

    @dataclass
    class MockAPIPosition:
        direction: int
        size: int

    @pytest.mark.asyncio
    async def test_handle_position_mismatch_syncs(self):
        """Test position mismatch syncs from API."""
        handler = RecoveryHandler(
            config=RecoveryConfig(halt_on_position_mismatch=False)
        )

        local_pos = self.MockPosition(direction=1, size=2)
        api_pos = self.MockAPIPosition(direction=-1, size=-3)
        sync_func = MagicMock()

        await handler.handle_position_mismatch(local_pos, api_pos, sync_func)

        sync_func.assert_called_once_with(api_pos)
        assert len(handler._error_history) == 1

    @pytest.mark.asyncio
    async def test_handle_position_mismatch_alerts(self):
        """Test position mismatch sends alert."""
        alert_cb = MagicMock()
        handler = RecoveryHandler(
            config=RecoveryConfig(halt_on_position_mismatch=False),
            on_alert=alert_cb,
        )

        local_pos = self.MockPosition(direction=1, size=2)
        api_pos = self.MockAPIPosition(direction=-1, size=-3)
        sync_func = MagicMock()

        await handler.handle_position_mismatch(local_pos, api_pos, sync_func)

        alert_cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_position_mismatch_halts_when_configured(self):
        """Test position mismatch triggers halt when configured."""
        halt_cb = AsyncMock()
        handler = RecoveryHandler(
            config=RecoveryConfig(halt_on_position_mismatch=True),
            on_halt=halt_cb,
        )

        local_pos = self.MockPosition(direction=1, size=2)
        api_pos = self.MockAPIPosition(direction=-1, size=-3)
        sync_func = MagicMock()

        await handler.handle_position_mismatch(local_pos, api_pos, sync_func)

        halt_cb.assert_called_once()


class TestRecoveryHandlerCriticalError:
    """Tests for RecoveryHandler.handle_critical_error()."""

    @pytest.mark.asyncio
    async def test_handle_critical_error_logs(self):
        """Test critical error is logged."""
        handler = RecoveryHandler()

        await handler.handle_critical_error(
            exception=ValueError("Test error"),
            context="test_context",
        )

        assert len(handler._error_history) == 1
        error = handler._error_history[0]
        assert error.severity == ErrorSeverity.CRITICAL
        assert "test_context" in error.message

    @pytest.mark.asyncio
    async def test_handle_critical_error_includes_traceback(self):
        """Test critical error includes traceback."""
        handler = RecoveryHandler()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            await handler.handle_critical_error(exception=e, context="test")

        error = handler._error_history[0]
        assert "traceback" in error.details

    @pytest.mark.asyncio
    async def test_handle_critical_error_calls_halt(self):
        """Test critical error triggers halt."""
        halt_cb = AsyncMock()
        handler = RecoveryHandler(
            config=RecoveryConfig(flatten_on_critical_error=True),
            on_halt=halt_cb,
        )

        await handler.handle_critical_error(
            exception=ValueError("Test error"),
            context="test",
        )

        halt_cb.assert_called_once()


class TestRecoveryHandlerErrorHistory:
    """Tests for error history management."""

    def test_log_error_trims_history(self):
        """Test error history is trimmed at 1000 entries."""
        handler = RecoveryHandler()

        # Add 1001 errors
        for i in range(1001):
            error = ErrorEvent(
                timestamp=datetime.now(),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.DEBUG,
                message=f"Error {i}",
            )
            handler._log_error(error)

        # Should be trimmed to 500
        assert len(handler._error_history) == 500

        # Should have most recent errors
        assert "Error 1000" in handler._error_history[-1].message

    def test_get_error_history_filter_by_since(self):
        """Test get_error_history filters by timestamp."""
        handler = RecoveryHandler()

        # Add old error
        old_error = ErrorEvent(
            timestamp=datetime.now() - timedelta(hours=2),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.DEBUG,
            message="Old error",
        )
        handler._error_history.append(old_error)

        # Add recent error
        recent_error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.DEBUG,
            message="Recent error",
        )
        handler._error_history.append(recent_error)

        # Filter for last hour
        since = datetime.now() - timedelta(hours=1)
        filtered = handler.get_error_history(since=since)

        assert len(filtered) == 1
        assert filtered[0].message == "Recent error"

    def test_get_error_history_filter_by_category(self):
        """Test get_error_history filters by category."""
        handler = RecoveryHandler()

        handler._error_history = [
            ErrorEvent(
                timestamp=datetime.now(),
                category=ErrorCategory.CONNECTION,
                severity=ErrorSeverity.DEBUG,
                message="Connection error",
            ),
            ErrorEvent(
                timestamp=datetime.now(),
                category=ErrorCategory.ORDER,
                severity=ErrorSeverity.DEBUG,
                message="Order error",
            ),
        ]

        filtered = handler.get_error_history(category=ErrorCategory.ORDER)

        assert len(filtered) == 1
        assert filtered[0].message == "Order error"

    def test_get_error_history_filter_by_severity(self):
        """Test get_error_history filters by severity."""
        handler = RecoveryHandler()

        handler._error_history = [
            ErrorEvent(
                timestamp=datetime.now(),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.WARNING,
                message="Warning",
            ),
            ErrorEvent(
                timestamp=datetime.now(),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.ERROR,
                message="Error",
            ),
        ]

        filtered = handler.get_error_history(severity=ErrorSeverity.ERROR)

        assert len(filtered) == 1
        assert filtered[0].message == "Error"

    def test_get_error_stats(self):
        """Test get_error_stats returns statistics."""
        handler = RecoveryHandler()
        handler._state.consecutive_errors = 3
        handler._state.is_recovering = True

        stats = handler.get_error_stats()

        assert "total_errors" in stats
        assert stats["consecutive_errors"] == 3
        assert stats["is_recovering"] is True


class TestRecoveryHandlerAlerts:
    """Tests for alert handling."""

    @pytest.mark.asyncio
    async def test_send_alert_calls_callback(self):
        """Test _send_alert calls callback."""
        alert_cb = MagicMock()
        handler = RecoveryHandler(on_alert=alert_cb)

        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            message="Test error",
        )

        await handler._send_alert(error)

        alert_cb.assert_called_once_with(error)

    @pytest.mark.asyncio
    async def test_send_alert_handles_callback_exception(self):
        """Test _send_alert handles callback exception."""
        def bad_callback(error):
            raise ValueError("Callback error")

        handler = RecoveryHandler(on_alert=bad_callback)

        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            message="Test error",
        )

        # Should not raise
        await handler._send_alert(error)


# =============================================================================
# Decorator Tests
# =============================================================================

class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    @pytest.mark.asyncio
    async def test_with_retry_success_first_attempt(self):
        """Test success on first attempt."""

        @with_retry(max_retries=3)
        async def success_func():
            return "success"

        result = await success_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_retry_success_after_retries(self):
        """Test success after retries."""
        call_count = 0

        @with_retry(max_retries=3, backoff_base=0.01)
        async def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry")
            return "success"

        result = await retry_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_exhausted(self):
        """Test raises after max retries exhausted."""

        @with_retry(max_retries=2, backoff_base=0.01)
        async def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(ValueError) as exc_info:
            await always_fail()

        assert "Always fails" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_retry_specific_exceptions(self):
        """Test only catches specified exceptions."""

        @with_retry(max_retries=2, exceptions=(ValueError,))
        async def raise_type_error():
            raise TypeError("Type error")

        # TypeError should not be retried
        with pytest.raises(TypeError):
            await raise_type_error()

    @pytest.mark.asyncio
    async def test_with_retry_exponential_backoff(self):
        """Test exponential backoff between retries."""
        sleeps = []

        async def mock_sleep(duration):
            sleeps.append(duration)

        @with_retry(max_retries=3, backoff_base=1.0)
        async def always_fail():
            raise ValueError("Fail")

        with patch('asyncio.sleep', side_effect=mock_sleep):
            with pytest.raises(ValueError):
                await always_fail()

        # Backoff should be 1.0, 2.0, 4.0
        assert len(sleeps) == 3
        assert sleeps[0] == 1.0
        assert sleeps[1] == 2.0
        assert sleeps[2] == 4.0


class TestWithTimeoutDecorator:
    """Tests for with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_with_timeout_completes_in_time(self):
        """Test function completes within timeout."""

        @with_timeout(timeout_seconds=5.0)
        async def fast_func():
            return "fast"

        result = await fast_func()
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_with_timeout_exceeds_limit(self):
        """Test timeout raises when exceeded."""

        @with_timeout(timeout_seconds=0.01)
        async def slow_func():
            await asyncio.sleep(1.0)
            return "slow"

        with pytest.raises(asyncio.TimeoutError):
            await slow_func()


# =============================================================================
# Error Event Tests
# =============================================================================

class TestErrorEvent:
    """Tests for ErrorEvent dataclass."""

    def test_error_event_to_dict(self):
        """Test ErrorEvent serialization."""
        error = ErrorEvent(
            timestamp=datetime(2026, 1, 16, 10, 30, 0),
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.ERROR,
            message="Test error",
            details={"key": "value"},
            exception=ValueError("Test"),
            recoverable=True,
            recovery_action="retry",
        )

        d = error.to_dict()

        assert d["timestamp"] == "2026-01-16T10:30:00"
        assert d["category"] == "connection"
        assert d["severity"] == "error"
        assert d["message"] == "Test error"
        assert d["details"] == {"key": "value"}
        assert "Test" in d["exception"]
        assert d["recoverable"] is True
        assert d["recovery_action"] == "retry"

    def test_error_event_to_dict_no_exception(self):
        """Test ErrorEvent serialization without exception."""
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.DEBUG,
            message="Test",
        )

        d = error.to_dict()
        assert d["exception"] is None
