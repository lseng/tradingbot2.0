"""
Extended Tests for src/lib/time_utils.py.

These tests target uncovered lines to improve coverage from 71% to 80%+.

Uncovered lines being addressed:
- Line 88: to_utc() with naive datetime (assumes NY timezone)
- Line 105: is_rth() with dt=None (uses get_ny_now)
- Line 133: is_eth() with dt=None (uses get_ny_now)
- Lines 147, 151-155: is_eth() Sunday and Friday edge cases
- Line 162: is_eth() CME reset window (5:00 PM - 5:15 PM)
- Lines 192-204: get_session_start() with dt=None and ETH mode
- Line 219, 228: get_session_end() with dt=None and ETH mode
- Line 247: minutes_to_close() with dt=None
- Lines 268-275: minutes_to_eod_flatten() with dt=None
- Line 296: get_eod_phase() with dt=None
- Lines 412, 414: is_trading_day() with datetime input
- Lines 438, 440: get_next_trading_day() with datetime input
- Lines 460, 462: get_previous_trading_day() with datetime input
- Lines 522-532: normalize_to_session() after 6 PM
- Lines 547-554: get_minutes_since_rth_open() with dt=None
- Lines 571-572: get_normalized_session_time() with dt=None
"""

import pytest
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

from src.risk.eod_manager import EODPhase
from src.lib.time_utils import (
    get_ny_now,
    to_ny_time,
    to_utc,
    is_rth,
    is_eth,
    is_market_open,
    get_session_start,
    get_session_end,
    minutes_to_close,
    minutes_to_eod_flatten,
    get_eod_phase,
    get_eod_size_multiplier,
    can_open_new_position,
    should_flatten,
    is_trading_day,
    get_next_trading_day,
    get_previous_trading_day,
    get_trading_days_in_range,
    count_trading_days,
    normalize_to_session,
    get_minutes_since_rth_open,
    get_normalized_session_time,
    US_HOLIDAYS,
)
from src.lib.constants import NY_TIMEZONE


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ny_time_rth():
    """Mock current time to be during RTH (10:00 AM NY)."""
    mock_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=NY_TIMEZONE)
    with patch('src.lib.time_utils.get_ny_now', return_value=mock_time):
        yield mock_time


@pytest.fixture
def mock_ny_time_eth_morning():
    """Mock current time to be during ETH (6:00 AM NY)."""
    mock_time = datetime(2026, 1, 15, 6, 0, 0, tzinfo=NY_TIMEZONE)  # Thursday
    with patch('src.lib.time_utils.get_ny_now', return_value=mock_time):
        yield mock_time


@pytest.fixture
def mock_ny_time_eth_evening():
    """Mock current time to be during ETH (8:00 PM NY)."""
    mock_time = datetime(2026, 1, 15, 20, 0, 0, tzinfo=NY_TIMEZONE)  # Thursday evening
    with patch('src.lib.time_utils.get_ny_now', return_value=mock_time):
        yield mock_time


# ============================================================================
# Test to_utc() with Naive Datetime (Line 88)
# ============================================================================

class TestToUtcNaiveDatetime:
    """Tests for to_utc with naive datetime (assumed NY)."""

    def test_to_utc_naive_datetime_winter(self):
        """Naive datetime should be assumed NY and converted to UTC (winter: +5 hours)."""
        naive_dt = datetime(2026, 1, 15, 9, 0, 0)  # 9 AM NY (naive)
        utc_dt = to_utc(naive_dt)

        # NY to UTC in winter is +5 hours
        assert utc_dt.hour == 14  # 9 AM NY = 2 PM UTC
        assert utc_dt.tzinfo == ZoneInfo("UTC")

    def test_to_utc_naive_datetime_summer(self):
        """Naive datetime should be assumed NY and converted to UTC (summer: +4 hours)."""
        naive_dt = datetime(2026, 6, 15, 9, 0, 0)  # 9 AM NY (naive), DST
        utc_dt = to_utc(naive_dt)

        # NY to UTC in summer is +4 hours
        assert utc_dt.hour == 13  # 9 AM NY = 1 PM UTC
        assert utc_dt.tzinfo == ZoneInfo("UTC")

    def test_to_utc_aware_datetime(self):
        """Aware datetime should be converted correctly."""
        utc_aware = datetime(2026, 1, 15, 14, 0, 0, tzinfo=ZoneInfo("UTC"))
        result = to_utc(utc_aware)

        assert result.hour == 14
        assert result.tzinfo == ZoneInfo("UTC")


# ============================================================================
# Test is_rth() with dt=None (Line 105)
# ============================================================================

class TestIsRthWithNone:
    """Tests for is_rth with dt=None."""

    def test_is_rth_none_uses_current_time(self, mock_ny_time_rth):
        """is_rth(None) should use current time."""
        result = is_rth()
        # Mock time is 10 AM Thursday, which is RTH
        assert result is True

    def test_is_rth_weekend_saturday(self):
        """Saturday should not be RTH."""
        saturday = datetime(2026, 1, 17, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_rth(saturday) is False

    def test_is_rth_weekend_sunday(self):
        """Sunday should not be RTH."""
        sunday = datetime(2026, 1, 18, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_rth(sunday) is False


# ============================================================================
# Test is_eth() Edge Cases (Lines 133, 147, 151-155, 162)
# ============================================================================

class TestIsEthEdgeCases:
    """Tests for is_eth edge cases."""

    def test_is_eth_none_uses_current_time(self, mock_ny_time_eth_morning):
        """is_eth(None) should use current time."""
        result = is_eth()
        # Mock time is 6 AM Thursday, which is ETH
        assert result is True

    def test_is_eth_saturday_always_false(self):
        """Saturday should never be ETH."""
        saturday = datetime(2026, 1, 17, 20, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(saturday) is False

    def test_is_eth_sunday_before_6pm_false(self):
        """Sunday before 6 PM should not be ETH."""
        sunday_early = datetime(2026, 1, 18, 15, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(sunday_early) is False

    def test_is_eth_sunday_after_6pm_true(self):
        """Sunday at/after 6 PM should be ETH."""
        sunday_evening = datetime(2026, 1, 18, 18, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(sunday_evening) is True

        sunday_late = datetime(2026, 1, 18, 21, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(sunday_late) is True

    def test_is_eth_friday_before_rth_true(self):
        """Friday before RTH should be ETH."""
        friday_early = datetime(2026, 1, 16, 6, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(friday_early) is True

    def test_is_eth_friday_after_rth_before_5pm_true(self):
        """Friday after RTH but before 5 PM should be ETH."""
        friday_post_rth = datetime(2026, 1, 16, 16, 30, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(friday_post_rth) is True

    def test_is_eth_friday_after_5pm_false(self):
        """Friday at/after 5 PM should not be ETH."""
        friday_late = datetime(2026, 1, 16, 17, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(friday_late) is False

    def test_is_eth_cme_reset_window_false(self):
        """CME reset window (5:00 PM - 5:15 PM) should not be ETH."""
        reset_start = datetime(2026, 1, 15, 17, 0, 0, tzinfo=NY_TIMEZONE)  # Thursday
        assert is_eth(reset_start) is False

        reset_middle = datetime(2026, 1, 15, 17, 7, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(reset_middle) is False

        reset_end = datetime(2026, 1, 15, 17, 14, 0, tzinfo=NY_TIMEZONE)
        assert is_eth(reset_end) is False

    def test_is_eth_after_reset_window_true(self):
        """After CME reset window (5:15 PM+) should be ETH."""
        after_reset = datetime(2026, 1, 15, 17, 15, 0, tzinfo=NY_TIMEZONE)  # Thursday
        assert is_eth(after_reset) is True

    def test_is_eth_weekday_evening_true(self):
        """Weekday evening (after 4 PM, not in reset) should be ETH."""
        evening = datetime(2026, 1, 15, 20, 0, 0, tzinfo=NY_TIMEZONE)  # Thursday 8 PM
        assert is_eth(evening) is True


# ============================================================================
# Test get_session_start() and get_session_end() with dt=None (Lines 192-204, 219, 228)
# ============================================================================

class TestSessionStartEndWithNone:
    """Tests for get_session_start and get_session_end with dt=None."""

    def test_get_session_start_none_rth(self, mock_ny_time_rth):
        """get_session_start(None) should use current time."""
        result = get_session_start(rth_only=True)
        # Should return 9:30 AM for the mock date (Jan 15, 2026)
        assert result.hour == 9
        assert result.minute == 30
        assert result.date() == mock_ny_time_rth.date()

    def test_get_session_start_none_eth(self, mock_ny_time_rth):
        """get_session_start(None, rth_only=False) should return ETH start."""
        result = get_session_start(rth_only=False)
        # ETH starts at 6 PM the previous day
        assert result.hour == 18
        assert result.date() == mock_ny_time_rth.date() - timedelta(days=1)

    def test_get_session_end_none_rth(self, mock_ny_time_rth):
        """get_session_end(None) should use current time."""
        result = get_session_end(rth_only=True)
        # Should return 4:00 PM for the mock date
        assert result.hour == 16
        assert result.minute == 0

    def test_get_session_end_none_eth(self, mock_ny_time_rth):
        """get_session_end(None, rth_only=False) should return ETH end."""
        result = get_session_end(rth_only=False)
        # ETH ends at 5:00 PM
        assert result.hour == 17
        assert result.minute == 0


# ============================================================================
# Test minutes_to_close() with dt=None (Line 247)
# ============================================================================

class TestMinutesToCloseWithNone:
    """Tests for minutes_to_close with dt=None."""

    def test_minutes_to_close_none_uses_current_time(self, mock_ny_time_rth):
        """minutes_to_close(None) should use current time."""
        result = minutes_to_close()
        # Mock time is 10 AM, RTH close is 4 PM = 6 hours = 360 minutes
        assert abs(result - 360) < 1


# ============================================================================
# Test minutes_to_eod_flatten() with dt=None (Lines 268-275)
# ============================================================================

class TestMinutesToEodFlattenWithNone:
    """Tests for minutes_to_eod_flatten with dt=None."""

    def test_minutes_to_eod_flatten_none_uses_current_time(self, mock_ny_time_rth):
        """minutes_to_eod_flatten(None) should use current time."""
        result = minutes_to_eod_flatten()
        # Mock time is 10 AM, flatten time is 4:30 PM = 6.5 hours = 390 minutes
        assert abs(result - 390) < 1

    def test_minutes_to_eod_flatten_past_flatten_time(self):
        """After flatten time, result should be negative."""
        past_flatten = datetime(2026, 1, 15, 17, 0, 0, tzinfo=NY_TIMEZONE)
        result = minutes_to_eod_flatten(past_flatten)
        assert result < 0


# ============================================================================
# Test get_eod_phase() with dt=None (Line 296)
# ============================================================================

class TestGetEodPhaseWithNone:
    """Tests for get_eod_phase with dt=None."""

    def test_get_eod_phase_none_uses_current_time(self, mock_ny_time_rth):
        """get_eod_phase(None) should use current time."""
        result = get_eod_phase()
        # Mock time is 10 AM, which is NORMAL phase
        assert result == EODPhase.NORMAL


# ============================================================================
# Test is_trading_day() with datetime input (Lines 412, 414)
# ============================================================================

class TestIsTradingDayWithDatetime:
    """Tests for is_trading_day with datetime input."""

    def test_is_trading_day_none_uses_today(self, mock_ny_time_rth):
        """is_trading_day(None) should use today."""
        result = is_trading_day()
        # Mock time is Thursday Jan 15, 2026 - a trading day
        assert result is True

    def test_is_trading_day_with_datetime(self):
        """is_trading_day with datetime should convert to date."""
        weekday_dt = datetime(2026, 1, 15, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_trading_day(weekday_dt) is True

        weekend_dt = datetime(2026, 1, 17, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_trading_day(weekend_dt) is False

    def test_is_trading_day_holiday(self):
        """Holidays should not be trading days."""
        # New Year's Day 2026
        new_years = date(2026, 1, 1)
        assert is_trading_day(new_years) is False


# ============================================================================
# Test get_next_trading_day() with datetime input (Lines 438, 440)
# ============================================================================

class TestGetNextTradingDayWithDatetime:
    """Tests for get_next_trading_day with datetime input."""

    def test_get_next_trading_day_none(self, mock_ny_time_rth):
        """get_next_trading_day(None) should use today."""
        result = get_next_trading_day()
        # From Thursday Jan 15, next trading day is Friday Jan 16
        assert result == date(2026, 1, 16)

    def test_get_next_trading_day_with_datetime(self):
        """get_next_trading_day with datetime should convert to date."""
        friday_dt = datetime(2026, 1, 16, 10, 0, 0, tzinfo=NY_TIMEZONE)
        result = get_next_trading_day(friday_dt)
        # From Friday Jan 16, next is Tuesday Jan 20 (skipping weekend AND MLK Day Jan 19)
        assert result == date(2026, 1, 20)

    def test_get_next_trading_day_skips_holiday(self):
        """Should skip holidays."""
        # Day before New Year's (Wednesday Dec 31, 2025)
        dec_31 = date(2025, 12, 31)
        result = get_next_trading_day(dec_31)
        # Jan 1 is a holiday, so next trading day is Jan 2
        assert result == date(2026, 1, 2)


# ============================================================================
# Test get_previous_trading_day() with datetime input (Lines 460, 462)
# ============================================================================

class TestGetPreviousTradingDayWithDatetime:
    """Tests for get_previous_trading_day with datetime input."""

    def test_get_previous_trading_day_none(self, mock_ny_time_rth):
        """get_previous_trading_day(None) should use today."""
        result = get_previous_trading_day()
        # From Thursday Jan 15, previous is Wednesday Jan 14
        assert result == date(2026, 1, 14)

    def test_get_previous_trading_day_with_datetime(self):
        """get_previous_trading_day with datetime should convert to date."""
        monday_dt = datetime(2026, 1, 19, 10, 0, 0, tzinfo=NY_TIMEZONE)
        result = get_previous_trading_day(monday_dt)
        # From Monday, previous is Friday
        assert result == date(2026, 1, 16)


# ============================================================================
# Test normalize_to_session() After 6 PM (Lines 522-532)
# ============================================================================

class TestNormalizeToSessionAfter6PM:
    """Tests for normalize_to_session after 6 PM."""

    def test_normalize_to_session_after_6pm(self):
        """After 6 PM, session date should be next day."""
        evening_dt = datetime(2026, 1, 15, 20, 0, 0, tzinfo=NY_TIMEZONE)  # 8 PM Thursday
        result = normalize_to_session(evening_dt)

        # After 6 PM Thursday = Friday's session
        assert result.date() == date(2026, 1, 16)

    def test_normalize_to_session_exactly_6pm(self):
        """Exactly at 6 PM, session date should be next day."""
        six_pm = datetime(2026, 1, 15, 18, 0, 0, tzinfo=NY_TIMEZONE)
        result = normalize_to_session(six_pm)

        assert result.date() == date(2026, 1, 16)

    def test_normalize_to_session_before_6pm(self):
        """Before 6 PM, session date should be same day."""
        afternoon = datetime(2026, 1, 15, 15, 0, 0, tzinfo=NY_TIMEZONE)  # 3 PM
        result = normalize_to_session(afternoon)

        assert result.date() == date(2026, 1, 15)

    def test_normalize_to_session_naive_datetime(self):
        """Naive datetime should be converted to NY first."""
        naive_evening = datetime(2026, 1, 15, 20, 0, 0)  # Naive, assumes UTC
        result = normalize_to_session(naive_evening)

        # Naive assumed UTC, 8 PM UTC = 3 PM NY (winter), same day
        assert result.date() == date(2026, 1, 15)


# ============================================================================
# Test get_minutes_since_rth_open() with dt=None (Lines 547-554)
# ============================================================================

class TestGetMinutesSinceRthOpenWithNone:
    """Tests for get_minutes_since_rth_open with dt=None."""

    def test_get_minutes_since_rth_open_none(self, mock_ny_time_rth):
        """get_minutes_since_rth_open(None) should use current time."""
        result = get_minutes_since_rth_open()
        # Mock time is 10 AM, RTH opens at 9:30 AM = 30 minutes since open
        assert abs(result - 30) < 1

    def test_get_minutes_since_rth_open_before_open(self):
        """Before RTH open, result should be negative."""
        early = datetime(2026, 1, 15, 8, 0, 0, tzinfo=NY_TIMEZONE)
        result = get_minutes_since_rth_open(early)
        # 8 AM is 1.5 hours before 9:30 AM = -90 minutes
        assert abs(result - (-90)) < 1


# ============================================================================
# Test get_normalized_session_time() with dt=None (Lines 571-572)
# ============================================================================

class TestGetNormalizedSessionTimeWithNone:
    """Tests for get_normalized_session_time with dt=None."""

    def test_get_normalized_session_time_none(self, mock_ny_time_rth):
        """get_normalized_session_time(None) should use current time."""
        result = get_normalized_session_time()
        # Mock time is 10 AM = 30 min since open, RTH is 390 min = 30/390 ≈ 0.077
        assert 0 < result < 0.2

    def test_get_normalized_session_time_at_open(self):
        """At RTH open, normalized time should be 0."""
        rth_open = datetime(2026, 1, 15, 9, 30, 0, tzinfo=NY_TIMEZONE)
        result = get_normalized_session_time(rth_open)
        assert abs(result - 0.0) < 0.01

    def test_get_normalized_session_time_at_close(self):
        """At RTH close, normalized time should be 1.0."""
        rth_close = datetime(2026, 1, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)
        result = get_normalized_session_time(rth_close)
        assert abs(result - 1.0) < 0.01

    def test_get_normalized_session_time_midday(self):
        """At noon, normalized time should be ~0.38."""
        noon = datetime(2026, 1, 15, 12, 0, 0, tzinfo=NY_TIMEZONE)
        result = get_normalized_session_time(noon)
        # 12:00 - 9:30 = 2.5 hours = 150 minutes, 150/390 ≈ 0.385
        assert 0.35 < result < 0.45


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestMarketOpenFunction:
    """Tests for is_market_open helper function."""

    def test_market_open_during_rth(self):
        """Market should be open during RTH."""
        rth_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_market_open(rth_time) is True

    def test_market_open_during_eth(self):
        """Market should be open during ETH."""
        eth_time = datetime(2026, 1, 15, 6, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_market_open(eth_time) is True

    def test_market_closed_saturday(self):
        """Market should be closed on Saturday."""
        saturday = datetime(2026, 1, 17, 12, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_market_open(saturday) is False


class TestEodSizeMultiplierAndCanOpenPosition:
    """Tests for EOD-related helper functions."""

    def test_size_multiplier_normal(self):
        """Normal phase should have 1.0 multiplier."""
        normal = datetime(2026, 1, 15, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert get_eod_size_multiplier(normal) == 1.0

    def test_size_multiplier_reduced(self):
        """Reduced size phase should have 0.5 multiplier."""
        reduced = datetime(2026, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE)  # 4:05 PM
        assert get_eod_size_multiplier(reduced) == 0.5

    def test_size_multiplier_close_only(self):
        """Close only phase should have 0.0 multiplier."""
        close_only = datetime(2026, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)  # 4:20 PM
        assert get_eod_size_multiplier(close_only) == 0.0

    def test_can_open_position_normal(self):
        """Can open position during normal phase."""
        normal = datetime(2026, 1, 15, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert can_open_new_position(normal) is True

    def test_can_open_position_close_only(self):
        """Cannot open position during close only phase."""
        close_only = datetime(2026, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)
        assert can_open_new_position(close_only) is False


class TestShouldFlatten:
    """Tests for should_flatten function."""

    def test_should_flatten_normal_false(self):
        """Should not flatten during normal phase."""
        normal = datetime(2026, 1, 15, 10, 0, 0, tzinfo=NY_TIMEZONE)
        assert should_flatten(normal) is False

    def test_should_flatten_flatten_phase_true(self):
        """Should flatten during flatten phase."""
        flatten = datetime(2026, 1, 15, 16, 26, 0, tzinfo=NY_TIMEZONE)  # 4:26 PM
        assert should_flatten(flatten) is True

    def test_should_flatten_must_be_flat_true(self):
        """Should flatten during must_be_flat phase."""
        must_flat = datetime(2026, 1, 15, 16, 35, 0, tzinfo=NY_TIMEZONE)  # 4:35 PM
        assert should_flatten(must_flat) is True


class TestTradingDaysInRange:
    """Tests for get_trading_days_in_range and count_trading_days."""

    def test_trading_days_in_range_single_week(self):
        """Get trading days in a single week."""
        start = date(2026, 1, 12)  # Monday
        end = date(2026, 1, 16)    # Friday

        days = get_trading_days_in_range(start, end)
        assert len(days) == 5  # Mon-Fri
        assert all(d.weekday() < 5 for d in days)

    def test_trading_days_in_range_with_weekend(self):
        """Should skip weekends and holidays."""
        start = date(2026, 1, 16)  # Friday
        end = date(2026, 1, 20)    # Tuesday (Jan 19 is MLK Day holiday)

        days = get_trading_days_in_range(start, end)
        # Friday Jan 16, and Tuesday Jan 20 (Sat, Sun, MLK Day skipped)
        assert len(days) == 2

    def test_count_trading_days(self):
        """Count trading days should match list length."""
        start = date(2026, 1, 12)
        end = date(2026, 1, 16)

        count = count_trading_days(start, end)
        days = get_trading_days_in_range(start, end)
        assert count == len(days)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
