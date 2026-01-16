"""
Error Handling and Recovery Module for Live Trading.

Provides robust error handling, automatic recovery, and alerting
for the live trading system.

Key responsibilities:
- WebSocket disconnect recovery with exponential backoff
- Position mismatch detection and resolution
- Order rejection handling
- Rate limiting recovery
- Authentication failure recovery
- Unhandled exception recovery (flatten and halt)

Reference: specs/live-trading-execution.md
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable, Any, Dict
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"  # Requires halt and manual intervention


class ErrorCategory(Enum):
    """Error categories for classification."""
    CONNECTION = "connection"  # WebSocket/API connection issues
    AUTHENTICATION = "authentication"  # Auth failures
    ORDER = "order"  # Order placement/execution issues
    POSITION = "position"  # Position mismatch
    RATE_LIMIT = "rate_limit"  # API rate limiting
    DATA = "data"  # Data quality issues
    SYSTEM = "system"  # System/infrastructure errors
    UNKNOWN = "unknown"


@dataclass
class ErrorEvent:
    """Represents an error event."""
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    exception: Optional[Exception] = None
    recoverable: bool = True
    recovery_action: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "exception": str(self.exception) if self.exception else None,
            "recoverable": self.recoverable,
            "recovery_action": self.recovery_action,
        }


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""
    # Backoff settings
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0

    # Retry limits
    max_reconnect_attempts: int = 10
    max_auth_retries: int = 3
    max_order_retries: int = 2

    # Timeouts
    reconnect_timeout_seconds: float = 60.0
    position_sync_timeout_seconds: float = 10.0

    # Alert thresholds
    consecutive_errors_for_alert: int = 3
    error_rate_window_seconds: float = 60.0
    max_errors_per_window: int = 10

    # Critical error behavior
    halt_on_position_mismatch: bool = False  # API is source of truth
    flatten_on_critical_error: bool = True


class RecoveryState:
    """Tracks recovery state."""

    def __init__(self):
        self.reconnect_attempts: int = 0
        self.auth_retries: int = 0
        self.last_error_time: Optional[datetime] = None
        self.consecutive_errors: int = 0
        self.errors_in_window: int = 0
        self.window_start: datetime = datetime.now()
        self.is_recovering: bool = False
        self.current_backoff: float = 1.0

    def record_error(self) -> None:
        """Record an error occurrence."""
        now = datetime.now()
        self.last_error_time = now
        self.consecutive_errors += 1

        # Track errors in window
        window_duration = (now - self.window_start).total_seconds()
        if window_duration > 60.0:
            # Reset window
            self.window_start = now
            self.errors_in_window = 1
        else:
            self.errors_in_window += 1

    def record_success(self) -> None:
        """Record a successful recovery."""
        self.consecutive_errors = 0
        self.reconnect_attempts = 0
        self.auth_retries = 0
        self.is_recovering = False
        self.current_backoff = 1.0

    def get_backoff(self, config: RecoveryConfig) -> float:
        """Get current backoff duration."""
        backoff = config.initial_backoff_seconds * (config.backoff_multiplier ** self.reconnect_attempts)
        return min(backoff, config.max_backoff_seconds)


class RecoveryHandler:
    """
    Handles error recovery for the trading system.

    Provides automatic recovery for various error types with
    exponential backoff and alerting.

    Usage:
        handler = RecoveryHandler(config)

        # Handle WebSocket disconnect
        await handler.handle_disconnect(reconnect_func)

        # Handle order rejection
        await handler.handle_order_rejection(order, error)

        # Handle position mismatch
        await handler.handle_position_mismatch(local, api)
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        on_alert: Optional[Callable[[ErrorEvent], None]] = None,
        on_halt: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """
        Initialize recovery handler.

        Args:
            config: Recovery configuration
            on_alert: Callback for alerts (logging, notifications)
            on_halt: Callback for halt (flatten positions, stop trading)
        """
        self.config = config or RecoveryConfig()
        self._on_alert = on_alert
        self._on_halt = on_halt
        self._state = RecoveryState()
        self._error_history: list[ErrorEvent] = []

        logger.info("RecoveryHandler initialized")

    async def handle_disconnect(
        self,
        reconnect_func: Callable[[], Awaitable[bool]],
    ) -> bool:
        """
        Handle WebSocket disconnect with auto-reconnect.

        Args:
            reconnect_func: Async function to attempt reconnection

        Returns:
            True if successfully reconnected
        """
        self._state.is_recovering = True
        logger.warning("Handling disconnect, starting recovery...")

        while self._state.reconnect_attempts < self.config.max_reconnect_attempts:
            self._state.reconnect_attempts += 1
            backoff = self._state.get_backoff(self.config)

            logger.info(
                f"Reconnect attempt {self._state.reconnect_attempts}/"
                f"{self.config.max_reconnect_attempts}, "
                f"waiting {backoff:.1f}s..."
            )

            await asyncio.sleep(backoff)

            try:
                success = await asyncio.wait_for(
                    reconnect_func(),
                    timeout=self.config.reconnect_timeout_seconds
                )

                if success:
                    logger.info("Reconnection successful")
                    self._state.record_success()
                    return True

            except asyncio.TimeoutError:
                logger.warning("Reconnection attempt timed out")
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")

            self._state.record_error()

        # Max attempts reached
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.CRITICAL,
            message=f"Max reconnect attempts ({self.config.max_reconnect_attempts}) reached",
            recoverable=False,
            recovery_action="halt",
        )
        await self._handle_critical_error(error)
        return False

    async def handle_auth_failure(
        self,
        reauth_func: Callable[[], Awaitable[bool]],
    ) -> bool:
        """
        Handle authentication failure with retry.

        Args:
            reauth_func: Async function to attempt re-authentication

        Returns:
            True if successfully re-authenticated
        """
        logger.warning("Handling auth failure...")

        while self._state.auth_retries < self.config.max_auth_retries:
            self._state.auth_retries += 1
            backoff = self._state.get_backoff(self.config)

            logger.info(
                f"Re-auth attempt {self._state.auth_retries}/"
                f"{self.config.max_auth_retries}, waiting {backoff:.1f}s..."
            )

            await asyncio.sleep(backoff)

            try:
                success = await reauth_func()
                if success:
                    logger.info("Re-authentication successful")
                    self._state.record_success()
                    return True
            except Exception as e:
                logger.error(f"Re-auth attempt failed: {e}")

            self._state.record_error()

        # Max retries reached
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.CRITICAL,
            message=f"Max auth retries ({self.config.max_auth_retries}) reached",
            recoverable=False,
            recovery_action="halt",
        )
        await self._handle_critical_error(error)
        return False

    async def handle_order_rejection(
        self,
        order_id: str,
        error_message: str,
        order_details: Optional[Dict] = None,
    ) -> None:
        """
        Handle order rejection.

        Order rejections are logged but NOT retried (as per spec).

        Args:
            order_id: Rejected order ID
            error_message: Rejection reason
            order_details: Additional order details
        """
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.ORDER,
            severity=ErrorSeverity.ERROR,
            message=f"Order rejected: {error_message}",
            details={"order_id": order_id, **(order_details or {})},
            recoverable=False,  # Do not retry rejected orders
            recovery_action="log_and_continue",
        )

        self._log_error(error)

        # Check if this indicates a systemic issue
        self._state.record_error()
        if self._state.consecutive_errors >= self.config.consecutive_errors_for_alert:
            await self._send_alert(error)

    async def handle_insufficient_margin(
        self,
        order_func: Callable[[int], Awaitable[Any]],
        original_size: int,
        min_size: int = 1,
    ) -> Optional[Any]:
        """
        Handle insufficient margin by reducing size.

        Args:
            order_func: Function to place order with reduced size
            original_size: Original order size
            min_size: Minimum order size to attempt

        Returns:
            Order result if successful, None otherwise
        """
        logger.warning(f"Insufficient margin for size {original_size}, reducing...")

        # Try with reduced size
        new_size = original_size // 2
        if new_size < min_size:
            new_size = min_size

        try:
            result = await order_func(new_size)
            logger.info(f"Order successful with reduced size: {new_size}")
            return result
        except Exception as e:
            logger.error(f"Order with reduced size also failed: {e}")

            error = ErrorEvent(
                timestamp=datetime.now(),
                category=ErrorCategory.ORDER,
                severity=ErrorSeverity.WARNING,
                message=f"Insufficient margin even with reduced size ({new_size})",
                details={"original_size": original_size, "reduced_size": new_size},
            )
            self._log_error(error)
            return None

    async def handle_rate_limit(
        self,
        retry_after_seconds: float,
        operation_func: Callable[[], Awaitable[Any]],
    ) -> Optional[Any]:
        """
        Handle rate limiting by waiting and retrying.

        Args:
            retry_after_seconds: Seconds to wait before retry
            operation_func: Function to retry

        Returns:
            Operation result if successful, None otherwise
        """
        logger.warning(f"Rate limited, waiting {retry_after_seconds}s...")

        await asyncio.sleep(retry_after_seconds)

        try:
            result = await operation_func()
            logger.info("Operation successful after rate limit wait")
            return result
        except Exception as e:
            logger.error(f"Operation failed after rate limit wait: {e}")
            return None

    async def handle_position_mismatch(
        self,
        local_position: 'Position',
        api_position: 'PositionData',
        sync_func: Callable[['PositionData'], None],
    ) -> None:
        """
        Handle position mismatch between local and API state.

        API is always the source of truth.

        Args:
            local_position: Local position state
            api_position: API position state
            sync_func: Function to sync local state from API
        """
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.POSITION,
            severity=ErrorSeverity.WARNING,
            message="Position mismatch detected",
            details={
                "local_direction": local_position.direction,
                "local_size": local_position.size,
                "api_direction": api_position.direction,
                "api_size": abs(api_position.size),
            },
            recovery_action="sync_from_api",
        )

        self._log_error(error)
        await self._send_alert(error)

        # Sync from API (API is source of truth)
        logger.info("Syncing position from API (API is source of truth)")
        sync_func(api_position)

        if self.config.halt_on_position_mismatch:
            await self._handle_critical_error(error)

    async def handle_critical_error(
        self,
        exception: Exception,
        context: str = "",
    ) -> None:
        """
        Handle critical/unhandled exception.

        Flattens positions and halts trading.

        Args:
            exception: The exception that occurred
            context: Context where the error occurred
        """
        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            message=f"Critical error in {context}: {str(exception)}",
            exception=exception,
            details={"traceback": traceback.format_exc()},
            recoverable=False,
            recovery_action="flatten_and_halt",
        )

        await self._handle_critical_error(error)

    async def _handle_critical_error(self, error: ErrorEvent) -> None:
        """Handle a critical error."""
        self._log_error(error)
        await self._send_alert(error)

        if self.config.flatten_on_critical_error and self._on_halt:
            logger.critical(f"CRITICAL ERROR - Initiating halt: {error.message}")
            await self._on_halt(error.message)

    def _log_error(self, error: ErrorEvent) -> None:
        """Log an error event."""
        self._error_history.append(error)

        # Limit history size
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]

        log_method = {
            ErrorSeverity.DEBUG: logger.debug,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.CRITICAL: logger.critical,
        }.get(error.severity, logger.error)

        log_method(f"[{error.category.value}] {error.message}")
        if error.details:
            log_method(f"  Details: {error.details}")

    async def _send_alert(self, error: ErrorEvent) -> None:
        """Send alert via callback."""
        if self._on_alert:
            try:
                self._on_alert(error)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    def get_error_history(
        self,
        since: Optional[datetime] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
    ) -> list[ErrorEvent]:
        """
        Get error history with optional filters.

        Args:
            since: Only errors after this time
            category: Filter by category
            severity: Filter by severity

        Returns:
            Filtered error list
        """
        errors = self._error_history

        if since:
            errors = [e for e in errors if e.timestamp >= since]
        if category:
            errors = [e for e in errors if e.category == category]
        if severity:
            errors = [e for e in errors if e.severity == severity]

        return errors

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)

        hour_errors = [e for e in self._error_history if e.timestamp >= last_hour]
        day_errors = [e for e in self._error_history if e.timestamp >= last_day]

        return {
            "total_errors": len(self._error_history),
            "errors_last_hour": len(hour_errors),
            "errors_last_day": len(day_errors),
            "consecutive_errors": self._state.consecutive_errors,
            "is_recovering": self._state.is_recovering,
            "reconnect_attempts": self._state.reconnect_attempts,
            "last_error_time": self._state.last_error_time.isoformat() if self._state.last_error_time else None,
        }

    def reset(self) -> None:
        """Reset recovery state."""
        self._state = RecoveryState()
        logger.info("Recovery state reset")


# Convenience decorators for error handling


def with_retry(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for async functions with retry logic.

    Args:
        max_retries: Maximum retry attempts
        backoff_base: Base backoff in seconds
        exceptions: Exception types to catch
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        backoff = backoff_base * (2 ** attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}, "
                            f"waiting {backoff:.1f}s: {e}"
                        )
                        await asyncio.sleep(backoff)
            raise last_exception
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator for async functions with timeout.

    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds
            )
        return wrapper
    return decorator
