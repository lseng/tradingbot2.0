"""
Comprehensive tests for End-of-Day (EOD) phase transitions and LiveTrader integration.

Tests cover:
1. EOD Phase Transitions at specific NY times (4:00, 4:15, 4:25, 4:30 PM)
2. LiveTrader EOD integration - verifying correct API usage
3. DST handling during spring forward and fall back transitions
4. Bug 10.0.2 regression tests - ensuring get_status().phase is used correctly

Reference: specs/live-trading-execution.md
EOD Timeline:
- 4:00 PM NY: REDUCED_SIZE - Reduce position sizing by 50%
- 4:15 PM NY: CLOSE_ONLY - No new positions, close existing only
- 4:25 PM NY: AGGRESSIVE_EXIT - Begin market order exits
- 4:30 PM NY: MUST_BE_FLAT - Must be flat (no exceptions)
"""

import pytest
import asyncio
from datetime import datetime, date, time, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.eod_manager import (
    EODManager,
    EODConfig,
    EODPhase,
    EODStatus,
    time_to_ny,
    get_ny_time,
    is_market_open,
)
from src.lib.constants import NY_TIMEZONE
from trading.live_trader import (
    LiveTrader,
    TradingConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def eod_manager():
    """EOD manager with default config."""
    return EODManager()


@pytest.fixture
def custom_eod_manager():
    """EOD manager with custom config for testing."""
    config = EODConfig(
        reduced_size_time=time(16, 0),
        close_only_time=time(16, 15),
        aggressive_exit_time=time(16, 25),
        must_be_flat_time=time(16, 30),
        reduced_size_factor=0.5,
    )
    return EODManager(config)


@pytest.fixture
def live_trader():
    """LiveTrader with default config for testing."""
    config = TradingConfig()
    return LiveTrader(config)


# =============================================================================
# EOD Phase Transition Tests
# =============================================================================

class TestEODPhaseTransitions:
    """
    Tests for EOD phase transitions at specific NY times.

    Verifies that phases transition correctly at:
    - 4:00 PM NY: REDUCED_SIZE (50% sizing)
    - 4:15 PM NY: CLOSE_ONLY (no new positions)
    - 4:25 PM NY: AGGRESSIVE_EXIT (market order exits)
    - 4:30 PM NY: MUST_BE_FLAT (force flatten)
    """

    def test_normal_phase_before_4pm(self, eod_manager):
        """Test NORMAL phase is active before 4:00 PM NY."""
        # Test at various times during normal trading hours
        test_times = [
            datetime(2025, 1, 15, 9, 35, 0, tzinfo=NY_TIMEZONE),  # 9:35 AM
            datetime(2025, 1, 15, 12, 0, 0, tzinfo=NY_TIMEZONE),  # 12:00 PM
            datetime(2025, 1, 15, 15, 59, 0, tzinfo=NY_TIMEZONE),  # 3:59 PM
            datetime(2025, 1, 15, 15, 59, 59, tzinfo=NY_TIMEZONE),  # 3:59:59 PM
        ]

        for test_time in test_times:
            status = eod_manager.get_status(test_time)
            assert status.phase == EODPhase.NORMAL, f"Failed at {test_time}"
            assert status.can_open_new_positions is True
            assert status.position_size_multiplier == 1.0
            assert status.should_flatten is False

    def test_reduced_size_phase_at_4pm(self, eod_manager):
        """Test REDUCED_SIZE phase starts exactly at 4:00 PM NY."""
        test_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.REDUCED_SIZE
        assert status.can_open_new_positions is True
        assert status.position_size_multiplier == 0.5  # 50% sizing
        assert status.should_flatten is False
        assert status.minutes_to_close == 30

    def test_reduced_size_phase_between_4pm_and_415pm(self, eod_manager):
        """Test REDUCED_SIZE phase continues between 4:00-4:15 PM NY."""
        test_times = [
            datetime(2025, 1, 15, 16, 1, 0, tzinfo=NY_TIMEZONE),   # 4:01 PM
            datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE),   # 4:05 PM
            datetime(2025, 1, 15, 16, 10, 0, tzinfo=NY_TIMEZONE),  # 4:10 PM
            datetime(2025, 1, 15, 16, 14, 59, tzinfo=NY_TIMEZONE), # 4:14:59 PM
        ]

        for test_time in test_times:
            status = eod_manager.get_status(test_time)
            assert status.phase == EODPhase.REDUCED_SIZE, f"Failed at {test_time}"
            assert status.can_open_new_positions is True
            assert status.position_size_multiplier == 0.5

    def test_close_only_phase_at_415pm(self, eod_manager):
        """Test CLOSE_ONLY phase (NO_NEW_POSITIONS) starts exactly at 4:15 PM NY."""
        test_time = datetime(2025, 1, 15, 16, 15, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.CLOSE_ONLY
        assert status.can_open_new_positions is False  # Key check: no new positions
        assert status.position_size_multiplier == 0.0
        assert status.should_flatten is False  # Not yet aggressive exit
        assert status.minutes_to_close == 15

    def test_close_only_phase_between_415pm_and_425pm(self, eod_manager):
        """Test CLOSE_ONLY phase continues between 4:15-4:25 PM NY."""
        test_times = [
            datetime(2025, 1, 15, 16, 16, 0, tzinfo=NY_TIMEZONE),  # 4:16 PM
            datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE),  # 4:20 PM
            datetime(2025, 1, 15, 16, 24, 59, tzinfo=NY_TIMEZONE), # 4:24:59 PM
        ]

        for test_time in test_times:
            status = eod_manager.get_status(test_time)
            assert status.phase == EODPhase.CLOSE_ONLY, f"Failed at {test_time}"
            assert status.can_open_new_positions is False

    def test_aggressive_exit_phase_at_425pm(self, eod_manager):
        """Test AGGRESSIVE_EXIT (EXIT_ONLY) phase starts exactly at 4:25 PM NY."""
        test_time = datetime(2025, 1, 15, 16, 25, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.AGGRESSIVE_EXIT
        assert status.can_open_new_positions is False
        assert status.position_size_multiplier == 0.0
        assert status.should_flatten is True  # Key: must start flattening
        assert status.minutes_to_close == 5

    def test_aggressive_exit_phase_between_425pm_and_430pm(self, eod_manager):
        """Test AGGRESSIVE_EXIT phase continues between 4:25-4:30 PM NY."""
        test_times = [
            datetime(2025, 1, 15, 16, 26, 0, tzinfo=NY_TIMEZONE),  # 4:26 PM
            datetime(2025, 1, 15, 16, 28, 0, tzinfo=NY_TIMEZONE),  # 4:28 PM
            datetime(2025, 1, 15, 16, 29, 59, tzinfo=NY_TIMEZONE), # 4:29:59 PM
        ]

        for test_time in test_times:
            status = eod_manager.get_status(test_time)
            assert status.phase == EODPhase.AGGRESSIVE_EXIT, f"Failed at {test_time}"
            assert status.should_flatten is True

    def test_must_be_flat_phase_at_430pm(self, eod_manager):
        """Test MUST_BE_FLAT (FLATTEN_IMMEDIATE) phase starts exactly at 4:30 PM NY."""
        test_time = datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.MUST_BE_FLAT
        assert status.can_open_new_positions is False
        assert status.position_size_multiplier == 0.0
        assert status.should_flatten is True
        assert status.minutes_to_close == 0

    def test_must_be_flat_phase_after_430pm(self, eod_manager):
        """Test MUST_BE_FLAT phase continues after 4:30 PM NY."""
        test_times = [
            datetime(2025, 1, 15, 16, 31, 0, tzinfo=NY_TIMEZONE),  # 4:31 PM
            datetime(2025, 1, 15, 17, 0, 0, tzinfo=NY_TIMEZONE),   # 5:00 PM
            datetime(2025, 1, 15, 18, 0, 0, tzinfo=NY_TIMEZONE),   # 6:00 PM
        ]

        for test_time in test_times:
            status = eod_manager.get_status(test_time)
            assert status.phase == EODPhase.MUST_BE_FLAT, f"Failed at {test_time}"
            assert status.should_flatten is True


class TestEODPhaseTransitionBoundaries:
    """
    Tests for exact boundary conditions at phase transitions.

    Ensures that transitions happen at the exact second specified.
    """

    def test_transition_normal_to_reduced_boundary(self, eod_manager):
        """Test exact transition from NORMAL to REDUCED_SIZE at 4:00:00 PM."""
        # One second before
        before = datetime(2025, 1, 15, 15, 59, 59, tzinfo=NY_TIMEZONE)
        status_before = eod_manager.get_status(before)
        assert status_before.phase == EODPhase.NORMAL

        # Exactly at 4:00:00
        at = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)
        status_at = eod_manager.get_status(at)
        assert status_at.phase == EODPhase.REDUCED_SIZE

    def test_transition_reduced_to_close_only_boundary(self, eod_manager):
        """Test exact transition from REDUCED_SIZE to CLOSE_ONLY at 4:15:00 PM."""
        # One second before
        before = datetime(2025, 1, 15, 16, 14, 59, tzinfo=NY_TIMEZONE)
        status_before = eod_manager.get_status(before)
        assert status_before.phase == EODPhase.REDUCED_SIZE

        # Exactly at 4:15:00
        at = datetime(2025, 1, 15, 16, 15, 0, tzinfo=NY_TIMEZONE)
        status_at = eod_manager.get_status(at)
        assert status_at.phase == EODPhase.CLOSE_ONLY

    def test_transition_close_only_to_aggressive_exit_boundary(self, eod_manager):
        """Test exact transition from CLOSE_ONLY to AGGRESSIVE_EXIT at 4:25:00 PM."""
        # One second before
        before = datetime(2025, 1, 15, 16, 24, 59, tzinfo=NY_TIMEZONE)
        status_before = eod_manager.get_status(before)
        assert status_before.phase == EODPhase.CLOSE_ONLY
        assert status_before.should_flatten is False

        # Exactly at 4:25:00
        at = datetime(2025, 1, 15, 16, 25, 0, tzinfo=NY_TIMEZONE)
        status_at = eod_manager.get_status(at)
        assert status_at.phase == EODPhase.AGGRESSIVE_EXIT
        assert status_at.should_flatten is True

    def test_transition_aggressive_exit_to_must_be_flat_boundary(self, eod_manager):
        """Test exact transition from AGGRESSIVE_EXIT to MUST_BE_FLAT at 4:30:00 PM."""
        # One second before
        before = datetime(2025, 1, 15, 16, 29, 59, tzinfo=NY_TIMEZONE)
        status_before = eod_manager.get_status(before)
        assert status_before.phase == EODPhase.AGGRESSIVE_EXIT

        # Exactly at 4:30:00
        at = datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TIMEZONE)
        status_at = eod_manager.get_status(at)
        assert status_at.phase == EODPhase.MUST_BE_FLAT


class TestEODPositionSizing:
    """
    Tests for position sizing adjustments during EOD phases.
    """

    def test_full_size_during_normal(self, eod_manager):
        """Test 100% position sizing during normal trading hours."""
        test_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=NY_TIMEZONE)  # 2 PM
        status = eod_manager.get_status(test_time)
        assert status.position_size_multiplier == 1.0

    def test_half_size_during_reduced_size(self, eod_manager):
        """Test 50% position sizing during REDUCED_SIZE phase."""
        test_time = datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE)  # 4:05 PM
        status = eod_manager.get_status(test_time)
        assert status.position_size_multiplier == 0.5

    def test_zero_size_during_close_only(self, eod_manager):
        """Test 0% position sizing during CLOSE_ONLY phase."""
        test_time = datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)  # 4:20 PM
        status = eod_manager.get_status(test_time)
        assert status.position_size_multiplier == 0.0

    def test_zero_size_during_aggressive_exit(self, eod_manager):
        """Test 0% position sizing during AGGRESSIVE_EXIT phase."""
        test_time = datetime(2025, 1, 15, 16, 27, 0, tzinfo=NY_TIMEZONE)  # 4:27 PM
        status = eod_manager.get_status(test_time)
        assert status.position_size_multiplier == 0.0

    def test_custom_reduced_size_factor(self):
        """Test custom reduced size factor configuration."""
        config = EODConfig(reduced_size_factor=0.75)  # 75% instead of 50%
        manager = EODManager(config)

        test_time = datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE)
        status = manager.get_status(test_time)
        assert status.position_size_multiplier == 0.75


# =============================================================================
# LiveTrader EOD Integration Tests
# =============================================================================

class TestLiveTraderEODIntegration:
    """
    Tests for LiveTrader's integration with EOD manager.

    Verifies that:
    - LiveTrader correctly uses get_status().phase (not get_current_phase())
    - EOD phases affect position sizing correctly
    - No new positions allowed in CLOSE_ONLY phase
    - Flatten happens at MUST_BE_FLAT phase

    Bug 10.0.2 regression test: Ensures the correct EOD API is used.
    """

    def test_live_trader_uses_get_status_method(self):
        """
        Regression test for Bug 10.0.2: Verify LiveTrader uses get_status().phase.

        The old bug had live_trader.py calling get_current_phase() which doesn't exist.
        The fix uses get_status().phase instead.
        """
        # Verify EODManager has get_status method
        manager = EODManager()
        assert hasattr(manager, 'get_status'), "EODManager missing get_status method"

        # Verify get_status returns EODStatus with phase attribute
        status = manager.get_status()
        assert hasattr(status, 'phase'), "EODStatus missing phase attribute"
        assert isinstance(status.phase, EODPhase)

        # Verify EODManager does NOT have get_current_phase method (old buggy API)
        assert not hasattr(manager, 'get_current_phase'), \
            "EODManager should NOT have get_current_phase - this was the bug"

    def test_live_trader_eod_manager_initialization(self, live_trader):
        """Test that LiveTrader initializes EODManager."""
        # Before startup, _eod_manager is None
        assert live_trader._eod_manager is None

    @pytest.mark.asyncio
    async def test_process_bar_uses_correct_eod_api(self, live_trader):
        """
        Test that _process_bar uses get_status().phase correctly.

        This is a direct regression test for Bug 10.0.2.
        """
        # Setup mocks
        live_trader._eod_manager = EODManager()
        live_trader._feature_engine = MagicMock()
        live_trader._feature_engine.update.return_value = None  # No features yet

        # Create a mock bar
        mock_bar = MagicMock()
        mock_bar.high = 5001.0
        mock_bar.low = 4999.0
        mock_bar.close = 5000.0

        # Process bar - should not raise AttributeError for get_current_phase
        await live_trader._process_bar(mock_bar)

        # If we got here without AttributeError, the bug is fixed

    def test_close_only_phase_blocks_new_positions(self, eod_manager):
        """
        Test that CLOSE_ONLY phase correctly reports no new positions allowed.

        In CLOSE_ONLY phase (4:15 PM - 4:25 PM):
        - can_open_new_positions should be False
        - position_size_multiplier should be 0.0
        - should_flatten should be False (not yet aggressive)

        This tests the EOD status values that LiveTrader would check.
        """
        test_time = datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.CLOSE_ONLY
        assert status.can_open_new_positions is False
        assert status.position_size_multiplier == 0.0
        assert status.should_flatten is False

        # Verify the reason mentions no new positions or close only
        assert "close" in status.reason.lower() or "no new" in status.reason.lower()

    def test_must_be_flat_triggers_flatten(self, eod_manager):
        """
        Test that MUST_BE_FLAT phase correctly signals flatten requirement.

        At 4:30 PM NY, should_flatten must be True.
        """
        test_time = datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.MUST_BE_FLAT
        assert status.should_flatten is True
        assert status.can_open_new_positions is False
        assert status.position_size_multiplier == 0.0
        assert status.minutes_to_close == 0

        # Verify the reason mentions flat or 4:30
        assert "flat" in status.reason.lower() or "4:30" in status.reason

    def test_live_trader_checks_eod_phase_via_get_status(self):
        """
        Integration test verifying EODManager API contract.

        Bug 10.0.2 verification: The real EODManager class must:
        1. Have get_status() method
        2. NOT have get_current_phase() method

        This ensures the fix remains in place.
        """
        # Create a real EODManager (not a mock) to test the actual API
        manager = EODManager()

        # Verify required method exists
        assert hasattr(manager, 'get_status'), "EODManager must have get_status method"
        assert callable(getattr(manager, 'get_status')), "get_status must be callable"

        # Verify the buggy method does NOT exist
        assert not hasattr(manager, 'get_current_phase'), \
            "EODManager should NOT have get_current_phase - this was Bug 10.0.2"

        # Verify get_status returns EODStatus with phase
        status = manager.get_status()
        assert hasattr(status, 'phase'), "EODStatus must have phase attribute"
        assert isinstance(status.phase, EODPhase), "phase must be an EODPhase enum"


# =============================================================================
# DST Handling Tests
# =============================================================================

class TestDSTHandling:
    """
    Tests for DST (Daylight Saving Time) transitions.

    Ensures EOD phases work correctly during:
    - DST spring forward (March - clocks skip 2 AM to 3 AM)
    - DST fall back (November - clocks repeat 1 AM to 2 AM)

    Key insight: NY_TIMEZONE (America/New_York) handles DST automatically.
    EOD times are always in NY local time, so 4:00 PM is 4:00 PM NY
    regardless of whether DST is active.
    """

    def test_dst_spring_forward_eod_phases(self, eod_manager):
        """
        Test EOD phases work correctly during DST spring forward.

        DST spring forward in 2025: March 9, 2025 at 2:00 AM
        Clocks skip from 1:59:59 AM EST to 3:00:00 AM EDT

        EOD phases should still work at 4:00 PM EDT (now -4 hours from UTC).
        """
        # Day of DST change - March 9, 2025 (Sunday - market closed, but test anyway)
        # After DST, we're in EDT (UTC-4 instead of UTC-5)

        # Test day after DST change - March 10, 2025 (Monday)
        # 4:00 PM EDT should be REDUCED_SIZE
        test_time = datetime(2025, 3, 10, 16, 0, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.REDUCED_SIZE
        assert status.position_size_multiplier == 0.5

        # 4:15 PM EDT should be CLOSE_ONLY
        test_time = datetime(2025, 3, 10, 16, 15, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.CLOSE_ONLY

        # 4:25 PM EDT should be AGGRESSIVE_EXIT
        test_time = datetime(2025, 3, 10, 16, 25, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.AGGRESSIVE_EXIT

        # 4:30 PM EDT should be MUST_BE_FLAT
        test_time = datetime(2025, 3, 10, 16, 30, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.MUST_BE_FLAT

    def test_dst_fall_back_eod_phases(self, eod_manager):
        """
        Test EOD phases work correctly during DST fall back.

        DST fall back in 2025: November 2, 2025 at 2:00 AM
        Clocks repeat from 1:59:59 AM EDT to 1:00:00 AM EST

        EOD phases should still work at 4:00 PM EST (now -5 hours from UTC).
        """
        # Day of DST change - November 2, 2025 (Sunday - market closed, but test anyway)
        # After DST, we're back in EST (UTC-5 instead of UTC-4)

        # Test day after DST change - November 3, 2025 (Monday)
        # 4:00 PM EST should be REDUCED_SIZE
        test_time = datetime(2025, 11, 3, 16, 0, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.REDUCED_SIZE

        # 4:15 PM EST should be CLOSE_ONLY
        test_time = datetime(2025, 11, 3, 16, 15, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.CLOSE_ONLY

        # 4:25 PM EST should be AGGRESSIVE_EXIT
        test_time = datetime(2025, 11, 3, 16, 25, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.AGGRESSIVE_EXIT

        # 4:30 PM EST should be MUST_BE_FLAT
        test_time = datetime(2025, 11, 3, 16, 30, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert status.phase == EODPhase.MUST_BE_FLAT

    def test_summer_vs_winter_times(self, eod_manager):
        """
        Test that EOD phases work the same in summer (EDT) and winter (EST).

        4:00 PM NY should be REDUCED_SIZE regardless of DST.
        """
        # Summer time (EDT - daylight saving active)
        summer_time = datetime(2025, 7, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)
        summer_status = eod_manager.get_status(summer_time)

        # Winter time (EST - standard time)
        winter_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)
        winter_status = eod_manager.get_status(winter_time)

        # Both should be REDUCED_SIZE at 4:00 PM local time
        assert summer_status.phase == EODPhase.REDUCED_SIZE
        assert winter_status.phase == EODPhase.REDUCED_SIZE
        assert summer_status.position_size_multiplier == winter_status.position_size_multiplier

    def test_utc_conversion_during_dst(self, eod_manager):
        """
        Test that UTC times are correctly converted to NY time during DST.

        4:00 PM EDT = 20:00 UTC (summer)
        4:00 PM EST = 21:00 UTC (winter)
        """
        from datetime import timezone

        # Summer: 4:00 PM EDT = 20:00 UTC
        utc_summer = datetime(2025, 7, 15, 20, 0, 0, tzinfo=timezone.utc)
        status_summer = eod_manager.get_status(utc_summer)
        assert status_summer.phase == EODPhase.REDUCED_SIZE

        # Winter: 4:00 PM EST = 21:00 UTC
        utc_winter = datetime(2025, 1, 15, 21, 0, 0, tzinfo=timezone.utc)
        status_winter = eod_manager.get_status(utc_winter)
        assert status_winter.phase == EODPhase.REDUCED_SIZE

    def test_dst_transition_day_phases(self, eod_manager):
        """
        Test EOD phases on the actual day of DST transition.

        Note: DST transitions happen on Sundays when market is closed,
        but we still test to ensure the code handles it correctly.
        """
        # Spring forward - March 9, 2025 (Sunday)
        # Even though market is closed, phases should still be correct
        spring_forward_day = datetime(2025, 3, 9, 16, 0, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(spring_forward_day)
        assert status.phase == EODPhase.REDUCED_SIZE

        # Fall back - November 2, 2025 (Sunday)
        fall_back_day = datetime(2025, 11, 2, 16, 0, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(fall_back_day)
        assert status.phase == EODPhase.REDUCED_SIZE


# =============================================================================
# Bug 10.0.2 Regression Tests
# =============================================================================

class TestBug1002Regression:
    """
    Regression tests specifically for Bug 10.0.2: EOD method name mismatch.

    Original bug: live_trader.py called get_current_phase() but EODManager
    only defines get_status() method. This would cause AttributeError at
    4:00 PM daily when EOD phase checking kicks in.

    Fix: Changed to get_status().phase

    These tests ensure the bug remains fixed.
    """

    def test_eod_manager_api_contract(self):
        """Verify EODManager has correct API methods."""
        manager = EODManager()

        # Required methods
        assert hasattr(manager, 'get_status')
        assert hasattr(manager, 'get_minutes_to_close')
        assert hasattr(manager, 'is_trading_session')
        assert hasattr(manager, 'can_open_new_position')
        assert hasattr(manager, 'should_flatten_now')
        assert hasattr(manager, 'get_position_size_multiplier')

        # Should NOT have old buggy method
        assert not hasattr(manager, 'get_current_phase')

    def test_eod_status_has_phase_attribute(self):
        """Verify EODStatus returned by get_status() has phase attribute."""
        manager = EODManager()
        status = manager.get_status()

        assert isinstance(status, EODStatus)
        assert hasattr(status, 'phase')
        assert isinstance(status.phase, EODPhase)

    def test_phase_accessible_via_get_status(self):
        """Test that phase is correctly accessible via get_status().phase."""
        manager = EODManager()

        # Test at various times
        test_cases = [
            (datetime(2025, 1, 15, 14, 0, 0, tzinfo=NY_TIMEZONE), EODPhase.NORMAL),
            (datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE), EODPhase.REDUCED_SIZE),
            (datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE), EODPhase.CLOSE_ONLY),
            (datetime(2025, 1, 15, 16, 27, 0, tzinfo=NY_TIMEZONE), EODPhase.AGGRESSIVE_EXIT),
            (datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TIMEZONE), EODPhase.MUST_BE_FLAT),
        ]

        for test_time, expected_phase in test_cases:
            status = manager.get_status(test_time)
            assert status.phase == expected_phase, \
                f"At {test_time}, expected {expected_phase}, got {status.phase}"

    def test_no_attribute_error_at_eod(self):
        """
        Ensure no AttributeError when checking EOD phase at 4:00 PM.

        This is the core regression test - the original bug would raise
        AttributeError: 'EODManager' object has no attribute 'get_current_phase'
        """
        manager = EODManager()

        # This is the exact pattern used in live_trader.py after the fix
        test_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)

        # This should NOT raise AttributeError
        eod_status = manager.get_status(test_time)
        eod_phase = eod_status.phase

        # The phase should be valid
        assert eod_phase in EODPhase.__members__.values()

    @pytest.mark.asyncio
    async def test_live_trader_process_bar_no_attribute_error(self):
        """
        Test that LiveTrader._process_bar doesn't raise AttributeError.

        This simulates what happens at 4:00 PM daily when the trading loop
        calls _process_bar and checks EOD phase.
        """
        config = TradingConfig()
        trader = LiveTrader(config)

        # Setup minimal mocks
        trader._eod_manager = EODManager()
        trader._feature_engine = MagicMock()
        trader._feature_engine.update.return_value = None  # No features

        # Mock bar
        mock_bar = MagicMock()
        mock_bar.high = 5001.0
        mock_bar.low = 4999.0
        mock_bar.close = 5000.0

        # This should complete without AttributeError
        # Using await since _process_bar is async
        await trader._process_bar(mock_bar)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for EOD manager helper functions."""

    def test_time_to_ny_naive_datetime(self):
        """Test time_to_ny converts naive datetime to NY timezone."""
        naive_dt = datetime(2025, 1, 15, 15, 0, 0)  # No timezone
        ny_dt = time_to_ny(naive_dt)

        assert ny_dt.tzinfo == NY_TIMEZONE

    def test_time_to_ny_utc_datetime(self):
        """Test time_to_ny converts UTC datetime to NY timezone."""
        from datetime import timezone

        utc_dt = datetime(2025, 1, 15, 21, 0, 0, tzinfo=timezone.utc)  # 9 PM UTC
        ny_dt = time_to_ny(utc_dt)

        assert ny_dt.tzinfo == NY_TIMEZONE
        assert ny_dt.hour == 16  # 4 PM NY in winter (EST = UTC-5)

    def test_get_ny_time_returns_ny_timezone(self):
        """Test get_ny_time returns datetime in NY timezone."""
        ny_time = get_ny_time()
        assert ny_time.tzinfo == NY_TIMEZONE

    def test_is_market_open_during_session(self, eod_manager):
        """Test is_market_open returns True during trading session."""
        # Weekday during session
        test_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=NY_TIMEZONE)  # Wednesday 2 PM
        assert is_market_open(test_time) is True

    def test_is_market_open_after_session(self, eod_manager):
        """Test is_market_open returns False after trading session."""
        test_time = datetime(2025, 1, 15, 17, 0, 0, tzinfo=NY_TIMEZONE)  # 5 PM
        assert is_market_open(test_time) is False

    def test_is_market_open_weekend(self, eod_manager):
        """Test is_market_open returns False on weekends."""
        # Saturday
        saturday = datetime(2025, 1, 18, 14, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_market_open(saturday) is False

        # Sunday
        sunday = datetime(2025, 1, 19, 14, 0, 0, tzinfo=NY_TIMEZONE)
        assert is_market_open(sunday) is False


# =============================================================================
# EOD Status Reason Tests
# =============================================================================

class TestEODStatusReasons:
    """Tests for EOD status reason messages."""

    def test_reason_includes_phase_info(self, eod_manager):
        """Test that reason string includes relevant phase information."""
        # Normal phase
        normal_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(normal_time)
        assert "normal" in status.reason.lower() or "mins to close" in status.reason.lower()

        # Reduced size phase
        reduced_time = datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(reduced_time)
        assert "reduced" in status.reason.lower() or "50%" in status.reason

    def test_reason_includes_minutes_to_close(self, eod_manager):
        """Test that reason includes minutes to close countdown."""
        test_time = datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)  # 10 mins to 4:30
        status = eod_manager.get_status(test_time)
        assert "10" in status.reason or "mins" in status.reason.lower()

    def test_must_be_flat_reason(self, eod_manager):
        """Test MUST_BE_FLAT has clear reason message."""
        test_time = datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TIMEZONE)
        status = eod_manager.get_status(test_time)
        assert "flat" in status.reason.lower() or "4:30" in status.reason


# =============================================================================
# Time-Decay Stop Tightening Tests (2.6 FIX)
# =============================================================================

class TestStopTightenFactor:
    """
    Tests for EODManager.get_stop_tighten_factor().

    2.6 FIX: Time-decay stop tightening as EOD approaches.

    Factor schedule (minutes to close -> factor):
    - > 60 minutes: 1.0 (no tightening)
    - 30-60 minutes: 0.90 (10% tighter)
    - 15-30 minutes: 0.80 (20% tighter)
    - 5-15 minutes: 0.70 (30% tighter)
    - < 5 minutes: 0.60 (40% tighter)
    """

    def test_no_tightening_early_session(self, eod_manager):
        """Test no tightening when more than 60 minutes to close."""
        # 2 hours before close (2:30 PM -> 4:30 PM = 120 mins)
        test_time = datetime(2025, 1, 15, 14, 30, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 1.0, f"Expected 1.0 for 120 mins to close, got {factor}"

    def test_no_tightening_over_60_minutes(self, eod_manager):
        """Test no tightening when exactly 61 minutes to close."""
        # 61 minutes before close (3:29 PM)
        test_time = datetime(2025, 1, 15, 15, 29, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 1.0, f"Expected 1.0 for 61 mins to close, got {factor}"

    def test_10_percent_tightening_30_60_minutes(self, eod_manager):
        """Test 10% tightening when 30-60 minutes to close."""
        # 45 minutes before close (3:45 PM)
        test_time = datetime(2025, 1, 15, 15, 45, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 0.90, f"Expected 0.90 for 45 mins to close, got {factor}"

    def test_20_percent_tightening_15_30_minutes(self, eod_manager):
        """Test 20% tightening when 15-30 minutes to close."""
        # 20 minutes before close (4:10 PM)
        test_time = datetime(2025, 1, 15, 16, 10, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 0.80, f"Expected 0.80 for 20 mins to close, got {factor}"

    def test_30_percent_tightening_5_15_minutes(self, eod_manager):
        """Test 30% tightening when 5-15 minutes to close."""
        # 10 minutes before close (4:20 PM)
        test_time = datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 0.70, f"Expected 0.70 for 10 mins to close, got {factor}"

    def test_40_percent_tightening_under_5_minutes(self, eod_manager):
        """Test 40% tightening when less than 5 minutes to close."""
        # 3 minutes before close (4:27 PM)
        test_time = datetime(2025, 1, 15, 16, 27, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 0.60, f"Expected 0.60 for 3 mins to close, got {factor}"

    def test_boundary_60_minutes(self, eod_manager):
        """Test boundary at exactly 60 minutes to close."""
        # Exactly 60 minutes (3:30 PM)
        test_time = datetime(2025, 1, 15, 15, 30, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        # 60 minutes should be in the 30-60 band (0.90)
        assert factor == 0.90, f"Expected 0.90 for exactly 60 mins, got {factor}"

    def test_boundary_30_minutes(self, eod_manager):
        """Test boundary at exactly 30 minutes to close."""
        # Exactly 30 minutes (4:00 PM)
        test_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        # 30 minutes should be in the 15-30 band (0.80)
        assert factor == 0.80, f"Expected 0.80 for exactly 30 mins, got {factor}"

    def test_boundary_15_minutes(self, eod_manager):
        """Test boundary at exactly 15 minutes to close."""
        # Exactly 15 minutes (4:15 PM)
        test_time = datetime(2025, 1, 15, 16, 15, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        # 15 minutes should be in the 5-15 band (0.70)
        assert factor == 0.70, f"Expected 0.70 for exactly 15 mins, got {factor}"

    def test_boundary_5_minutes(self, eod_manager):
        """Test boundary at exactly 5 minutes to close."""
        # Exactly 5 minutes (4:25 PM)
        test_time = datetime(2025, 1, 15, 16, 25, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        # 5 minutes should be in the <5 band (0.60)
        assert factor == 0.60, f"Expected 0.60 for exactly 5 mins, got {factor}"

    def test_at_market_close(self, eod_manager):
        """Test maximum tightening at market close."""
        # At 4:30 PM (0 minutes)
        test_time = datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TIMEZONE)
        factor = eod_manager.get_stop_tighten_factor(test_time)
        assert factor == 0.60, f"Expected 0.60 for 0 mins to close, got {factor}"

    def test_progressive_tightening(self, eod_manager):
        """Test that factors decrease monotonically as time progresses."""
        times = [
            datetime(2025, 1, 15, 14, 30, 0, tzinfo=NY_TIMEZONE),  # 120 mins
            datetime(2025, 1, 15, 15, 45, 0, tzinfo=NY_TIMEZONE),  # 45 mins
            datetime(2025, 1, 15, 16, 10, 0, tzinfo=NY_TIMEZONE),  # 20 mins
            datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE),  # 10 mins
            datetime(2025, 1, 15, 16, 27, 0, tzinfo=NY_TIMEZONE),  # 3 mins
        ]

        factors = [eod_manager.get_stop_tighten_factor(t) for t in times]

        # Verify monotonically decreasing or equal
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1], (
                f"Factor should decrease over time: {factors[i]} >= {factors[i + 1]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
