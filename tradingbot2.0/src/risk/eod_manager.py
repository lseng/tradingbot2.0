"""
End-of-Day (EOD) Manager for MES Futures Scalping Bot.

HARD REQUIREMENT: All positions must be flat by 4:30 PM NY.

EOD Timeline (from spec):
- 4:00 PM NY: Reduce position sizing by 50%
- 4:15 PM NY: No new positions, close existing only
- 4:25 PM NY: Begin market order exits (aggressive)
- 4:30 PM NY: MUST be flat (no exceptions)

This module handles:
- Time-based phase detection
- Position sizing adjustments
- Flatten signal generation
- DST-aware timezone handling
"""

from dataclasses import dataclass
from datetime import datetime, time, date
from enum import Enum
from typing import Optional, Tuple
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

logger = logging.getLogger(__name__)


# New York timezone for all trading time calculations
NY_TZ = ZoneInfo("America/New_York")


class EODPhase(Enum):
    """End-of-day trading phase."""
    NORMAL = "normal"  # Normal trading
    REDUCED_SIZE = "reduced_size"  # After 4:00 PM - 50% size
    CLOSE_ONLY = "close_only"  # After 4:15 PM - no new positions
    AGGRESSIVE_EXIT = "aggressive_exit"  # After 4:25 PM - market orders
    MUST_BE_FLAT = "must_be_flat"  # 4:30 PM - force flatten
    AFTER_HOURS = "after_hours"  # After 4:30 PM - no trading


@dataclass
class EODConfig:
    """Configuration for EOD management."""
    # Session times (NY timezone)
    session_start: time = time(9, 30)  # 9:30 AM NY
    session_end: time = time(16, 30)  # 4:30 PM NY

    # EOD phase times
    reduced_size_time: time = time(16, 0)   # 4:00 PM
    close_only_time: time = time(16, 15)    # 4:15 PM
    aggressive_exit_time: time = time(16, 25)  # 4:25 PM
    must_be_flat_time: time = time(16, 30)  # 4:30 PM

    # Size reduction factor
    reduced_size_factor: float = 0.5  # 50% of normal size

    # Pre-market buffer (no trading before this)
    pre_market_buffer_minutes: int = 5  # Start 5 mins after open


@dataclass
class EODStatus:
    """Current EOD status."""
    phase: EODPhase
    current_time_ny: datetime
    minutes_to_close: int
    can_open_new_positions: bool
    position_size_multiplier: float
    should_flatten: bool
    reason: str


class EODManager:
    """
    End-of-day position management.

    Ensures all positions are closed by 4:30 PM NY to comply
    with day trading requirements and avoid overnight exposure.

    Usage:
        manager = EODManager()

        # Check current status
        status = manager.get_status()
        if not status.can_open_new_positions:
            # Don't open new trades
            pass

        if status.should_flatten:
            # Close all positions immediately
            pass

        # Adjust position size
        adjusted_size = int(base_size * status.position_size_multiplier)
    """

    def __init__(self, config: Optional[EODConfig] = None):
        """
        Initialize EOD manager.

        Args:
            config: EOD configuration (uses defaults if None)
        """
        self.config = config or EODConfig()

    def get_status(self, current_time: Optional[datetime] = None) -> EODStatus:
        """
        Get current EOD status.

        Args:
            current_time: Time to check (uses current time if None)

        Returns:
            EODStatus with current phase and trading restrictions
        """
        # Get current NY time
        if current_time is None:
            now_ny = datetime.now(NY_TZ)
        else:
            # Convert to NY if needed
            if current_time.tzinfo is None:
                now_ny = current_time.replace(tzinfo=NY_TZ)
            else:
                now_ny = current_time.astimezone(NY_TZ)

        current_time_only = now_ny.time()

        # Calculate minutes to close
        close_dt = datetime.combine(now_ny.date(), self.config.must_be_flat_time)
        close_dt = close_dt.replace(tzinfo=NY_TZ)
        minutes_to_close = int((close_dt - now_ny).total_seconds() / 60)

        # Determine phase and restrictions
        phase, can_open, size_mult, should_flatten, reason = self._determine_phase(
            current_time_only, minutes_to_close
        )

        return EODStatus(
            phase=phase,
            current_time_ny=now_ny,
            minutes_to_close=max(0, minutes_to_close),
            can_open_new_positions=can_open,
            position_size_multiplier=size_mult,
            should_flatten=should_flatten,
            reason=reason,
        )

    def _determine_phase(
        self,
        current_time: time,
        minutes_to_close: int,
    ) -> Tuple[EODPhase, bool, float, bool, str]:
        """
        Determine EOD phase and trading restrictions.

        Returns:
            Tuple of (phase, can_open_new, size_multiplier, should_flatten, reason)
        """
        # Pre-market
        session_start_with_buffer = time(
            self.config.session_start.hour,
            self.config.session_start.minute + self.config.pre_market_buffer_minutes
        )

        if current_time < session_start_with_buffer:
            return (
                EODPhase.AFTER_HOURS,
                False,
                0.0,
                False,
                f"Before session start ({self.config.session_start})"
            )

        # Must be flat (4:30 PM or later)
        if current_time >= self.config.must_be_flat_time:
            return (
                EODPhase.MUST_BE_FLAT,
                False,
                0.0,
                True,
                "MUST BE FLAT - 4:30 PM NY reached"
            )

        # Aggressive exit (4:25 PM - 4:30 PM)
        if current_time >= self.config.aggressive_exit_time:
            return (
                EODPhase.AGGRESSIVE_EXIT,
                False,
                0.0,
                True,
                f"Aggressive exit phase - {minutes_to_close} mins to close"
            )

        # Close only (4:15 PM - 4:25 PM)
        if current_time >= self.config.close_only_time:
            return (
                EODPhase.CLOSE_ONLY,
                False,
                0.0,
                False,
                f"Close only - no new positions, {minutes_to_close} mins to close"
            )

        # Reduced size (4:00 PM - 4:15 PM)
        if current_time >= self.config.reduced_size_time:
            return (
                EODPhase.REDUCED_SIZE,
                True,
                self.config.reduced_size_factor,
                False,
                f"Reduced size ({self.config.reduced_size_factor:.0%}), {minutes_to_close} mins to close"
            )

        # Normal trading
        return (
            EODPhase.NORMAL,
            True,
            1.0,
            False,
            f"Normal trading, {minutes_to_close} mins to close"
        )

    def get_minutes_to_close(self, current_time: Optional[datetime] = None) -> int:
        """
        Get minutes remaining until market close (4:30 PM NY).

        Args:
            current_time: Time to check (uses current time if None)

        Returns:
            Minutes until 4:30 PM NY (negative if after close)
        """
        status = self.get_status(current_time)
        return status.minutes_to_close

    def is_trading_session(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if currently in trading session (9:30 AM - 4:30 PM NY).

        Args:
            current_time: Time to check (uses current time if None)

        Returns:
            True if within trading session
        """
        status = self.get_status(current_time)
        return status.phase not in (EODPhase.AFTER_HOURS, EODPhase.MUST_BE_FLAT)

    def can_open_new_position(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if new positions can be opened.

        Args:
            current_time: Time to check (uses current time if None)

        Returns:
            True if new positions are allowed
        """
        status = self.get_status(current_time)
        return status.can_open_new_positions

    def should_flatten_now(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if positions should be flattened immediately.

        Args:
            current_time: Time to check (uses current time if None)

        Returns:
            True if flatten is required
        """
        status = self.get_status(current_time)
        return status.should_flatten

    def get_position_size_multiplier(self, current_time: Optional[datetime] = None) -> float:
        """
        Get position size multiplier for current time.

        Args:
            current_time: Time to check (uses current time if None)

        Returns:
            Multiplier (1.0 for normal, 0.5 for reduced, 0.0 for no trading)
        """
        status = self.get_status(current_time)
        return status.position_size_multiplier

    def get_next_trading_session_start(
        self,
        current_time: Optional[datetime] = None
    ) -> datetime:
        """
        Get the start time of the next trading session.

        Useful for scheduling daily state resets.

        Args:
            current_time: Current time (uses now if None)

        Returns:
            Datetime of next 9:30 AM NY
        """
        if current_time is None:
            now_ny = datetime.now(NY_TZ)
        else:
            if current_time.tzinfo is None:
                now_ny = current_time.replace(tzinfo=NY_TZ)
            else:
                now_ny = current_time.astimezone(NY_TZ)

        # Today's session start
        today_start = datetime.combine(
            now_ny.date(),
            self.config.session_start
        ).replace(tzinfo=NY_TZ)

        # If before today's start, return today
        if now_ny < today_start:
            return today_start

        # Otherwise, return tomorrow's start
        from datetime import timedelta
        tomorrow = now_ny.date() + timedelta(days=1)

        # Skip weekends
        while tomorrow.weekday() >= 5:  # Saturday=5, Sunday=6
            tomorrow += timedelta(days=1)

        return datetime.combine(
            tomorrow,
            self.config.session_start
        ).replace(tzinfo=NY_TZ)


def time_to_ny(dt: datetime) -> datetime:
    """
    Convert datetime to New York timezone.

    Args:
        dt: Datetime to convert (assumes UTC if no tzinfo)

    Returns:
        Datetime in NY timezone
    """
    if dt.tzinfo is None:
        # Assume UTC for naive datetime
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(NY_TZ)


def get_ny_time() -> datetime:
    """Get current time in New York timezone."""
    return datetime.now(NY_TZ)


def is_market_open(current_time: Optional[datetime] = None) -> bool:
    """
    Quick check if US equity/futures market is open.

    Args:
        current_time: Time to check (uses current time if None)

    Returns:
        True if market is open (9:30 AM - 4:30 PM NY, weekdays)
    """
    manager = EODManager()
    status = manager.get_status(current_time)

    if status.phase in (EODPhase.AFTER_HOURS, EODPhase.MUST_BE_FLAT):
        return False

    # Check weekday
    if status.current_time_ny.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    return True
