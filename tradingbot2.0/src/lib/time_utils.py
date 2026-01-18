"""
Time utilities for trading operations.

This module provides timezone-aware utilities for:
- Converting between timezones (UTC, NY)
- Detecting trading sessions (RTH, ETH)
- Market calendar operations (trading days, holidays)
- End-of-day management (EOD phases)

All times default to New York timezone to match CME/TopstepX conventions.
"""

from datetime import datetime, date, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.lib.constants import (
    NY_TIMEZONE,
    RTH_START,
    RTH_END,
    ETH_START,
    ETH_END,
    EOD_REDUCED_SIZE_TIME,
    EOD_CLOSE_ONLY_TIME,
    EOD_FLATTEN_START_TIME,
    EOD_FLATTEN_TIME,
    RTH_DURATION_MINUTES,
)

# Import canonical EODPhase from risk module to avoid duplicate definitions
# The canonical version is in src/risk/eod_manager.py with string values and 6 phases
from src.risk.eod_manager import EODPhase


def get_ny_now() -> datetime:
    """
    Get current time in New York timezone.

    Returns:
        Current datetime in NY timezone
    """
    return datetime.now(NY_TIMEZONE)


def to_ny_time(dt: datetime) -> datetime:
    """
    Convert a datetime to New York timezone.

    Handles both naive and aware datetimes:
    - Naive datetimes are assumed to be UTC
    - Aware datetimes are converted to NY

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in NY timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(NY_TIMEZONE)


def to_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC timezone.

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in UTC timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetime is NY
        dt = dt.replace(tzinfo=NY_TIMEZONE)
    return dt.astimezone(ZoneInfo("UTC"))


def is_rth(dt: Optional[datetime] = None) -> bool:
    """
    Check if the given time is during Regular Trading Hours (RTH).

    RTH is 9:30 AM - 4:00 PM NY time, Monday-Friday.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if during RTH, False otherwise
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    # Check if it's a weekday
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    current_time = dt.time()
    return RTH_START <= current_time < RTH_END


def is_eth(dt: Optional[datetime] = None) -> bool:
    """
    Check if the given time is during Extended Trading Hours (ETH).

    ETH is the Globex session: 6:00 PM - 5:00 PM NY next day
    (with a 15-minute break from 5:00 PM - 5:15 PM for CME reset)

    Note: This returns True for times outside RTH but during Globex hours.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if during ETH (but not RTH), False otherwise
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    # Weekend check (more complex for ETH due to Sunday open)
    weekday = dt.weekday()
    current_time = dt.time()

    # Saturday is always closed
    if weekday == 5:
        return False

    # Sunday: ETH starts at 6:00 PM
    if weekday == 6:
        return current_time >= ETH_START

    # Friday: Globex closes at 5:00 PM (no overnight session)
    if weekday == 4:
        if current_time < RTH_START:
            return True  # Early morning before RTH
        if current_time >= RTH_END and current_time < time(17, 0):
            return True  # After RTH but before 5 PM close
        return False

    # Monday-Thursday
    # ETH includes: midnight to RTH start, and after RTH to midnight
    if current_time < RTH_START or current_time >= RTH_END:
        # Exclude the 5:00 PM - 5:15 PM CME reset window
        if time(17, 0) <= current_time < time(17, 15):
            return False
        return True

    return False


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if the market is open (either RTH or ETH).

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if market is open, False otherwise
    """
    return is_rth(dt) or is_eth(dt)


def get_session_start(dt: Optional[datetime] = None, rth_only: bool = True) -> datetime:
    """
    Get the start time of the trading session for the given date.

    Args:
        dt: Date to get session start for (default: today)
        rth_only: If True, return RTH start; if False, return ETH start

    Returns:
        Datetime of session start in NY timezone
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    session_date = dt.date()

    if rth_only:
        return datetime.combine(session_date, RTH_START, tzinfo=NY_TIMEZONE)
    else:
        # ETH starts at 6 PM the previous day
        previous_day = session_date - timedelta(days=1)
        return datetime.combine(previous_day, ETH_START, tzinfo=NY_TIMEZONE)


def get_session_end(dt: Optional[datetime] = None, rth_only: bool = True) -> datetime:
    """
    Get the end time of the trading session for the given date.

    Args:
        dt: Date to get session end for (default: today)
        rth_only: If True, return RTH end; if False, return ETH end

    Returns:
        Datetime of session end in NY timezone
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    session_date = dt.date()

    if rth_only:
        return datetime.combine(session_date, RTH_END, tzinfo=NY_TIMEZONE)
    else:
        return datetime.combine(session_date, ETH_END, tzinfo=NY_TIMEZONE)


def minutes_to_close(dt: Optional[datetime] = None, rth_only: bool = True) -> float:
    """
    Calculate minutes until session close.

    This is used for:
    - EOD management (when to reduce size, flatten)
    - Feature engineering (minutes-to-close feature)

    Args:
        dt: Datetime to calculate from (default: current time)
        rth_only: If True, calculate to RTH close (4:00 PM)

    Returns:
        Minutes until close (negative if already past close)
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    session_end = get_session_end(dt, rth_only)
    delta = session_end - dt
    return delta.total_seconds() / 60.0


def minutes_to_eod_flatten(dt: Optional[datetime] = None) -> float:
    """
    Calculate minutes until EOD flatten time (4:30 PM NY).

    This is the HARD REQUIREMENT - all positions must be flat by this time.

    Args:
        dt: Datetime to calculate from (default: current time)

    Returns:
        Minutes until flatten time (negative if already past)
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    flatten_time = datetime.combine(dt.date(), EOD_FLATTEN_TIME, tzinfo=NY_TIMEZONE)
    delta = flatten_time - dt
    return delta.total_seconds() / 60.0


def get_eod_phase(dt: Optional[datetime] = None) -> EODPhase:
    """
    Determine the current EOD management phase.

    Timeline:
    - NORMAL: Before 4:00 PM
    - REDUCED_SIZE: 4:00 PM - 4:15 PM (reduce sizing 50%)
    - CLOSE_ONLY: 4:15 PM - 4:25 PM (no new positions)
    - AGGRESSIVE_EXIT: 4:25 PM - 4:30 PM (aggressive exits)
    - MUST_BE_FLAT: 4:30 PM+ (must have no positions)

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        Current EODPhase
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    current_time = dt.time()

    if current_time >= EOD_FLATTEN_TIME:
        return EODPhase.MUST_BE_FLAT
    elif current_time >= EOD_FLATTEN_START_TIME:
        return EODPhase.AGGRESSIVE_EXIT
    elif current_time >= EOD_CLOSE_ONLY_TIME:
        return EODPhase.CLOSE_ONLY
    elif current_time >= EOD_REDUCED_SIZE_TIME:
        return EODPhase.REDUCED_SIZE
    else:
        return EODPhase.NORMAL


def get_eod_size_multiplier(dt: Optional[datetime] = None) -> float:
    """
    Get the position size multiplier based on EOD phase.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        Multiplier for position sizing (0.0 to 1.0)
    """
    phase = get_eod_phase(dt)

    if phase == EODPhase.NORMAL:
        return 1.0
    elif phase == EODPhase.REDUCED_SIZE:
        return 0.5
    else:
        # CLOSE_ONLY, FLATTEN, MUST_BE_FLAT - no new positions
        return 0.0


def can_open_new_position(dt: Optional[datetime] = None) -> bool:
    """
    Check if new positions can be opened at the given time.

    After 4:15 PM NY, no new positions should be opened.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if new positions are allowed
    """
    phase = get_eod_phase(dt)
    return phase in (EODPhase.NORMAL, EODPhase.REDUCED_SIZE)


def should_flatten(dt: Optional[datetime] = None) -> bool:
    """
    Check if positions should be flattened immediately.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if in AGGRESSIVE_EXIT or MUST_BE_FLAT phase
    """
    phase = get_eod_phase(dt)
    return phase in (EODPhase.AGGRESSIVE_EXIT, EODPhase.MUST_BE_FLAT)


# =============================================================================
# Trading Day Calendar
# =============================================================================

# US market holidays (CME observed holidays for futures)
# Note: This is a simplified list - production should use a proper calendar API
US_HOLIDAYS_2026 = [
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving Day
    date(2026, 12, 25), # Christmas Day
]

US_HOLIDAYS_2025 = [
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # Martin Luther King Jr. Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving Day
    date(2025, 12, 25), # Christmas Day
]

# Combined holidays set
US_HOLIDAYS = set(US_HOLIDAYS_2025 + US_HOLIDAYS_2026)


def is_trading_day(d: Optional[date] = None) -> bool:
    """
    Check if the given date is a trading day.

    A trading day is a weekday that is not a US market holiday.

    Args:
        d: Date to check (default: today)

    Returns:
        True if it's a trading day
    """
    if d is None:
        d = get_ny_now().date()
    elif isinstance(d, datetime):
        d = d.date()

    # Weekend check
    if d.weekday() >= 5:
        return False

    # Holiday check
    if d in US_HOLIDAYS:
        return False

    return True


def get_next_trading_day(d: Optional[date] = None) -> date:
    """
    Get the next trading day after the given date.

    Args:
        d: Starting date (default: today)

    Returns:
        Next trading day
    """
    if d is None:
        d = get_ny_now().date()
    elif isinstance(d, datetime):
        d = d.date()

    next_day = d + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)

    return next_day


def get_previous_trading_day(d: Optional[date] = None) -> date:
    """
    Get the previous trading day before the given date.

    Args:
        d: Starting date (default: today)

    Returns:
        Previous trading day
    """
    if d is None:
        d = get_ny_now().date()
    elif isinstance(d, datetime):
        d = d.date()

    prev_day = d - timedelta(days=1)
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)

    return prev_day


def get_trading_days_in_range(start: date, end: date) -> list[date]:
    """
    Get all trading days in the given date range (inclusive).

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        List of trading days
    """
    trading_days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days


def count_trading_days(start: date, end: date) -> int:
    """
    Count the number of trading days in the given range.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        Number of trading days
    """
    return len(get_trading_days_in_range(start, end))


# =============================================================================
# Session Normalization
# =============================================================================

def normalize_to_session(dt: datetime) -> datetime:
    """
    Normalize a datetime to its trading session date.

    For times before midnight (ETH overnight), maps to the next trading day.
    This ensures consistent session assignment for overnight trading.

    Args:
        dt: Datetime to normalize

    Returns:
        Datetime with date set to the logical trading session date
    """
    dt = to_ny_time(dt)
    current_time = dt.time()

    # If before 5 PM, it's part of that day's session
    # If after 6 PM, it's part of the next day's session
    if current_time >= time(18, 0):  # After 6 PM = next day's session
        session_date = dt.date() + timedelta(days=1)
    else:
        session_date = dt.date()

    return dt.replace(year=session_date.year, month=session_date.month, day=session_date.day)


def get_minutes_since_rth_open(dt: Optional[datetime] = None) -> float:
    """
    Calculate minutes since RTH open (9:30 AM NY).

    Useful for feature engineering (time of day feature).

    Args:
        dt: Datetime to calculate from (default: current time)

    Returns:
        Minutes since RTH open (negative if before open)
    """
    if dt is None:
        dt = get_ny_now()
    else:
        dt = to_ny_time(dt)

    rth_open = datetime.combine(dt.date(), RTH_START, tzinfo=NY_TIMEZONE)
    delta = dt - rth_open
    return delta.total_seconds() / 60.0


def get_normalized_session_time(dt: Optional[datetime] = None) -> float:
    """
    Get normalized time within RTH session (0.0 to 1.0).

    - 0.0 = RTH open (9:30 AM)
    - 1.0 = RTH close (4:00 PM)
    - Values outside [0, 1] indicate outside RTH

    Args:
        dt: Datetime to normalize (default: current time)

    Returns:
        Normalized session time
    """
    minutes_in = get_minutes_since_rth_open(dt)
    return minutes_in / RTH_DURATION_MINUTES
