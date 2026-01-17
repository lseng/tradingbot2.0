"""
Tests for time parsing validation in run_live.py (10.6).

These tests verify that the parse_time function properly validates
HH:MM format and rejects invalid inputs with clear error messages.
"""

import pytest
from datetime import time

# Import the function to test
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from scripts.run_live import parse_time


class TestParseTimeValidFormat:
    """Test parse_time with valid inputs."""

    def test_parse_valid_time_morning(self):
        """Test parsing a valid morning time."""
        result = parse_time("09:30")
        assert result == time(9, 30)

    def test_parse_valid_time_afternoon(self):
        """Test parsing a valid afternoon time."""
        result = parse_time("16:00")
        assert result == time(16, 0)

    def test_parse_valid_time_midnight(self):
        """Test parsing midnight."""
        result = parse_time("00:00")
        assert result == time(0, 0)

    def test_parse_valid_time_end_of_day(self):
        """Test parsing end of day."""
        result = parse_time("23:59")
        assert result == time(23, 59)

    def test_parse_single_digit_hour(self):
        """Test parsing time with single digit hour."""
        result = parse_time("9:30")
        assert result == time(9, 30)

    def test_parse_leading_zeros(self):
        """Test parsing time with leading zeros."""
        result = parse_time("00:05")
        assert result == time(0, 5)


class TestParseTimeInvalidFormat:
    """Test parse_time rejects invalid formats."""

    def test_missing_colon(self):
        """Test that missing colon raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("0930")
        assert "Invalid time format" in str(exc_info.value)
        assert "HH:MM" in str(exc_info.value)

    def test_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("")
        assert "Invalid time format" in str(exc_info.value)

    def test_extra_colons(self):
        """Test that extra colons (HH:MM:SS) raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09:30:00")
        assert "Invalid time format" in str(exc_info.value)

    def test_non_numeric_hours(self):
        """Test that non-numeric hours raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("ab:30")
        assert "Invalid time format" in str(exc_info.value)

    def test_non_numeric_minutes(self):
        """Test that non-numeric minutes raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09:xy")
        assert "Invalid time format" in str(exc_info.value)

    def test_only_colon(self):
        """Test that just a colon raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time(":")
        assert "Invalid time format" in str(exc_info.value)

    def test_spaces_in_time(self):
        """Test that spaces in time raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09 :30")
        assert "Invalid time format" in str(exc_info.value)

    def test_partial_time(self):
        """Test that partial time (missing minutes) raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09:")
        assert "Invalid time format" in str(exc_info.value)


class TestParseTimeInvalidRange:
    """Test parse_time rejects out-of-range values."""

    def test_hour_24(self):
        """Test that hour 24 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("24:00")
        assert "Invalid hour" in str(exc_info.value)
        assert "0-23" in str(exc_info.value)

    def test_hour_25(self):
        """Test that hour 25 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("25:00")
        assert "Invalid hour" in str(exc_info.value)

    def test_negative_hour(self):
        """Test that negative hour raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("-1:00")
        assert "Invalid time format" in str(exc_info.value)

    def test_minute_60(self):
        """Test that minute 60 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09:60")
        assert "Invalid minute" in str(exc_info.value)
        assert "0-59" in str(exc_info.value)

    def test_minute_99(self):
        """Test that minute 99 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09:99")
        assert "Invalid minute" in str(exc_info.value)

    def test_negative_minute(self):
        """Test that negative minute raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_time("09:-5")
        assert "Invalid time format" in str(exc_info.value)


class TestParseTimeEdgeCases:
    """Test edge cases for parse_time."""

    def test_boundary_hour_0(self):
        """Test hour boundary at 0."""
        result = parse_time("00:30")
        assert result.hour == 0

    def test_boundary_hour_23(self):
        """Test hour boundary at 23."""
        result = parse_time("23:30")
        assert result.hour == 23

    def test_boundary_minute_0(self):
        """Test minute boundary at 0."""
        result = parse_time("09:00")
        assert result.minute == 0

    def test_boundary_minute_59(self):
        """Test minute boundary at 59."""
        result = parse_time("09:59")
        assert result.minute == 59

    def test_typical_trading_start(self):
        """Test typical trading session start time."""
        result = parse_time("09:30")
        assert result == time(9, 30)

    def test_typical_trading_end(self):
        """Test typical trading session end time."""
        result = parse_time("16:00")
        assert result == time(16, 0)

    def test_flatten_time(self):
        """Test typical flatten time."""
        result = parse_time("16:25")
        assert result == time(16, 25)
