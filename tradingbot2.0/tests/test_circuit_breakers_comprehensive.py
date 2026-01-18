"""
Comprehensive Circuit Breaker Tests for Go-Live Validation (#10).

Tests cover:
1. Multiple simultaneous breakers
2. Boundary condition tests (exactly at thresholds)
3. Pause/Halt priority conflicts
4. Thread safety under concurrency
5. EOD + Circuit Breaker interaction
6. Breaker state transitions
7. Edge cases for volatility, spread, and volume

These tests ensure circuit breakers work correctly under all production scenarios.
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.risk.circuit_breakers import (
    CircuitBreakers,
    CircuitBreakerConfig,
    BreakerType,
    check_market_conditions,
)
from src.risk.eod_manager import EODManager, EODPhase
from src.lib.constants import NY_TIMEZONE


# =============================================================================
# Multiple Simultaneous Breakers Tests
# =============================================================================

class TestMultipleSimultaneousBreakers:
    """Test behavior when multiple circuit breakers trigger simultaneously."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    def test_high_volatility_and_low_volume_both_reduce_size(self, breakers):
        """Test that both HIGH_VOLATILITY and LOW_VOLUME reduce size to minimum."""
        # Trigger both conditions
        breakers.update_market_conditions(
            current_atr=4.0,  # 4x volatility (> 3x threshold)
            normal_atr=1.0,
            spread_ticks=1,  # Normal spread
            volume_pct=0.05,  # 5% volume (< 10% threshold)
        )

        # Both breakers should be active
        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        assert BreakerType.LOW_VOLUME in breakers.state.active_breakers

        # Size multiplier should be min(0.5, 0.5) = 0.5 (not 0.25)
        # Implementation uses min() which takes the lowest of all multipliers
        assert breakers.get_size_multiplier() == 0.5

    def test_wide_spread_pauses_even_with_other_breakers(self, breakers):
        """Test that WIDE_SPREAD pause takes priority over size reduction."""
        # Trigger volatility first
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )
        assert breakers.can_trade() is True
        assert breakers.get_size_multiplier() == 0.5

        # Now add wide spread - should pause
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=3,  # > 2 tick threshold
            volume_pct=0.85,
        )

        assert breakers.state.is_paused is True
        assert breakers.can_trade() is False
        assert BreakerType.WIDE_SPREAD in breakers.state.active_breakers

    def test_three_breakers_simultaneous(self, breakers):
        """Test all three market condition breakers triggering together."""
        breakers.update_market_conditions(
            current_atr=4.0,  # High volatility
            normal_atr=1.0,
            spread_ticks=3,  # Wide spread
            volume_pct=0.05,  # Low volume
        )

        # All three should be active
        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        assert BreakerType.WIDE_SPREAD in breakers.state.active_breakers
        assert BreakerType.LOW_VOLUME in breakers.state.active_breakers

        # Wide spread causes pause - trading not allowed
        assert breakers.can_trade() is False
        assert breakers.state.is_paused is True

    def test_consecutive_losses_during_high_volatility(self, breakers):
        """Test consecutive loss pause during high volatility conditions."""
        # First set high volatility (reduces size)
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )
        assert breakers.can_trade() is True
        assert breakers.get_size_multiplier() == 0.5

        # Now record 3 losses
        for _ in range(3):
            breakers.record_loss()

        # Both breakers should be active
        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        assert BreakerType.CONSECUTIVE_LOSSES in breakers.state.active_breakers

        # Should be paused (consecutive losses)
        assert breakers.can_trade() is False
        assert breakers.state.is_paused is True


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Test behavior at exact threshold boundaries."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    def test_volatility_exactly_at_threshold(self, breakers):
        """Test volatility exactly at 3.0x threshold (should NOT trigger)."""
        breakers.update_market_conditions(
            current_atr=3.0,  # Exactly 3x
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )

        # Exactly at threshold should NOT trigger (need > 3.0)
        assert BreakerType.HIGH_VOLATILITY not in breakers.state.active_breakers
        assert breakers.get_size_multiplier() == 1.0

    def test_volatility_just_above_threshold(self, breakers):
        """Test volatility just above 3.0x threshold (should trigger)."""
        breakers.update_market_conditions(
            current_atr=3.01,  # Just above 3x
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )

        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        assert breakers.get_size_multiplier() == 0.5

    def test_spread_exactly_at_threshold(self, breakers):
        """Test spread exactly at 2 tick threshold (should NOT trigger)."""
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=2,  # Exactly 2 ticks
            volume_pct=0.85,
        )

        # Exactly at threshold should NOT trigger (need > 2)
        assert BreakerType.WIDE_SPREAD not in breakers.state.active_breakers
        assert breakers.state.is_paused is False

    def test_spread_just_above_threshold(self, breakers):
        """Test spread just above 2 tick threshold (should trigger)."""
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=2.01,  # Just above 2 ticks
            volume_pct=0.85,
        )

        assert BreakerType.WIDE_SPREAD in breakers.state.active_breakers
        assert breakers.state.is_paused is True

    def test_volume_exactly_at_threshold(self, breakers):
        """Test volume exactly at 10% threshold (should NOT trigger)."""
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.10,  # Exactly 10%
        )

        # Exactly at threshold should NOT trigger (need < 10%)
        assert BreakerType.LOW_VOLUME not in breakers.state.active_breakers
        assert breakers.get_size_multiplier() == 1.0

    def test_volume_just_below_threshold(self, breakers):
        """Test volume just below 10% threshold (should trigger)."""
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.099,  # Just below 10%
        )

        assert BreakerType.LOW_VOLUME in breakers.state.active_breakers
        assert breakers.get_size_multiplier() == 0.5

    def test_zero_normal_atr_handles_division(self, breakers):
        """Test that zero normal ATR doesn't cause division by zero."""
        # Should not raise an error
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=0.0,  # Zero - could cause division by zero
            spread_ticks=1,
            volume_pct=0.85,
        )

        # Should not trigger volatility breaker (check is skipped)
        assert BreakerType.HIGH_VOLATILITY not in breakers.state.active_breakers

    def test_extreme_volatility_still_triggers(self, breakers):
        """Test that very extreme volatility (>10x) still triggers correctly."""
        breakers.update_market_conditions(
            current_atr=15.0,  # 15x volatility
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )

        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        # Size multiplier should still be 0.5 (not lower)
        assert breakers.get_size_multiplier() == 0.5


# =============================================================================
# Pause/Halt Priority Tests
# =============================================================================

class TestPauseHaltPriority:
    """Test priority between pauses and halts."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    def test_halt_takes_priority_over_pause(self, breakers):
        """Test that halt state prevents trading even if pause would expire."""
        # First trigger a pause
        for _ in range(3):
            breakers.record_loss()
        assert breakers.state.is_paused is True

        # Then trigger a halt (daily loss)
        breakers.trigger_daily_loss_stop(50.0, 50.0)

        # Halt should take priority
        assert breakers.state.is_halted is True
        assert breakers.can_trade() is False

        # Even after pause would normally expire
        breakers.state.pause_until = datetime.now() - timedelta(minutes=1)
        assert breakers.can_trade() is False  # Still halted

    def test_pause_cannot_be_triggered_when_halted(self, breakers):
        """Test that being halted doesn't reset to paused state."""
        # First halt
        breakers.trigger_daily_loss_stop(50.0, 50.0)
        assert breakers.state.is_halted is True

        # Now record losses (would normally trigger pause)
        for _ in range(3):
            breakers.record_loss()

        # Should still be halted, not just paused
        assert breakers.state.is_halted is True
        assert breakers.can_trade() is False

    def test_manual_review_halt_persists_through_daily_reset(self, breakers):
        """Test that max drawdown halt persists through daily reset."""
        # Trigger max drawdown (requires manual review)
        breakers.trigger_max_drawdown_halt(200.0, 200.0)
        assert breakers.state.requires_manual_review is True

        # Daily reset
        breakers.reset_daily()

        # Should still be halted
        assert breakers.state.is_halted is True
        assert breakers.state.requires_manual_review is True
        assert breakers.can_trade() is False

    def test_daily_loss_halt_clears_on_daily_reset(self, breakers):
        """Test that daily loss halt clears on daily reset."""
        breakers.trigger_daily_loss_stop(50.0, 50.0)
        assert breakers.state.is_halted is True

        # Daily reset
        breakers.reset_daily()

        # Should be able to trade again
        assert breakers.can_trade() is True


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of circuit breakers under concurrent access."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    def test_concurrent_record_loss_calls(self, breakers):
        """Test that concurrent record_loss calls don't cause race conditions."""
        errors = []

        def record_loss_thread():
            try:
                for _ in range(10):
                    breakers.record_loss()
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_loss_thread) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 5 threads * 10 losses = 50 total losses
        assert breakers._consecutive_losses == 50

    def test_concurrent_market_condition_updates(self, breakers):
        """Test that concurrent market condition updates don't cause issues."""
        errors = []

        def update_conditions_thread(vol_mult):
            try:
                for _ in range(10):
                    breakers.update_market_conditions(
                        current_atr=vol_mult,
                        normal_atr=1.0,
                        spread_ticks=1,
                        volume_pct=0.5,
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_conditions_thread, args=(i * 0.5,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # State should be consistent
        status = breakers.get_status()
        assert 'size_multiplier' in status

    def test_concurrent_can_trade_checks(self, breakers):
        """Test that concurrent can_trade checks don't cause issues."""
        results = []

        def check_can_trade_thread():
            for _ in range(100):
                result = breakers.can_trade()
                results.append(result)

        threads = [threading.Thread(target=check_can_trade_thread) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be True (no breakers triggered)
        assert all(results)
        assert len(results) == 500

    def test_concurrent_loss_and_win_recording(self, breakers):
        """Test concurrent loss and win recording."""
        errors = []

        def record_loss_thread():
            try:
                for _ in range(10):
                    breakers.record_loss()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def record_win_thread():
            try:
                for _ in range(10):
                    breakers.record_win()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_loss_thread),
            threading.Thread(target=record_win_thread),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final count depends on thread scheduling but should be >= 0
        assert breakers._consecutive_losses >= 0


# =============================================================================
# EOD + Circuit Breaker Interaction Tests
# =============================================================================

class TestEODCircuitBreakerInteraction:
    """Test interaction between EOD Manager and Circuit Breakers."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    @pytest.fixture
    def eod_manager(self):
        """Create EOD manager."""
        return EODManager()

    def test_circuit_breaker_pause_during_reduce_size_phase(self, breakers, eod_manager):
        """Test circuit breaker pause during EOD reduce size phase."""
        # 4:05 PM NY - reduce size phase
        test_time = datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TIMEZONE)
        eod_status = eod_manager.get_status(test_time)

        assert eod_status.phase == EODPhase.REDUCED_SIZE
        assert eod_status.position_size_multiplier == 0.5

        # Now also trigger high volatility
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )

        # Both should apply - combined multiplier would be 0.5 * 0.5 = 0.25
        # (but they're independent - EOD manager and circuit breakers)
        assert breakers.get_size_multiplier() == 0.5  # Circuit breaker
        assert eod_status.position_size_multiplier == 0.5  # EOD

    def test_circuit_breaker_pause_during_close_only_phase(self, breakers, eod_manager):
        """Test circuit breaker pause during close-only phase."""
        # 4:20 PM NY - close only phase
        test_time = datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TIMEZONE)
        eod_status = eod_manager.get_status(test_time)

        assert eod_status.phase == EODPhase.CLOSE_ONLY
        assert eod_status.can_open_new_positions is False

        # Circuit breaker pause shouldn't change EOD behavior
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=3,  # Wide spread - would pause
            volume_pct=0.85,
        )

        # EOD still prevents new positions
        assert eod_status.can_open_new_positions is False
        # Circuit breaker also pauses
        assert breakers.can_trade() is False

    def test_eod_flatten_takes_priority_over_circuit_breaker_pause(self, breakers, eod_manager):
        """Test that EOD flatten should take priority - must flatten even if paused."""
        # 4:26 PM NY - aggressive exit phase
        test_time = datetime(2025, 1, 15, 16, 26, 0, tzinfo=NY_TIMEZONE)
        eod_status = eod_manager.get_status(test_time)

        assert eod_status.phase == EODPhase.AGGRESSIVE_EXIT
        assert eod_status.can_open_new_positions is False

        # Even if circuit breaker is paused, we should still flatten
        for _ in range(3):
            breakers.record_loss()
        assert breakers.state.is_paused is True

        # EOD flatten should still be possible (closing, not opening)
        # The circuit breaker pause should not prevent exits
        assert eod_status.can_open_new_positions is False
        # Note: Circuit breakers apply to OPENING positions, not closing


# =============================================================================
# State Transition Tests
# =============================================================================

class TestStateTransitions:
    """Test circuit breaker state transitions."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    def test_pause_expiration_allows_trading(self, breakers):
        """Test that pause expiration allows trading to resume."""
        # Trigger a pause
        breakers._trigger_pause(
            BreakerType.CONSECUTIVE_LOSSES,
            1,  # 1 second pause
            "Test pause"
        )
        assert breakers.can_trade() is False

        # Wait for expiration
        time.sleep(1.1)

        # Should be able to trade now
        assert breakers.can_trade() is True
        assert breakers.state.is_paused is False

    def test_breaker_deactivation_removes_from_active_list(self, breakers):
        """Test that deactivating a breaker removes it from active_breakers."""
        # Activate high volatility
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )
        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers

        # Normalize conditions
        breakers.update_market_conditions(
            current_atr=1.5,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )

        # Should be removed from active_breakers
        assert BreakerType.HIGH_VOLATILITY not in breakers.state.active_breakers

    def test_consecutive_losses_reset_after_pause_expires(self, breakers):
        """Test that consecutive losses counter is handled correctly after pause."""
        # Record 3 losses (triggers 15-min pause)
        for _ in range(3):
            breakers.record_loss()
        assert breakers._consecutive_losses == 3
        assert breakers.state.is_paused is True

        # Manually expire the pause
        breakers.state.pause_until = datetime.now() - timedelta(minutes=1)

        # can_trade should clear the pause
        assert breakers.can_trade() is True

        # Consecutive losses counter should be maintained
        # (only reset by a win or manual reset)
        assert breakers._consecutive_losses == 3

    def test_consecutive_losses_counter_increments_correctly(self, breakers):
        """Test that consecutive losses increment even during pause."""
        # Record 3 losses (triggers pause)
        for _ in range(3):
            breakers.record_loss()
        assert breakers._consecutive_losses == 3
        assert breakers.state.is_paused is True

        # Record 2 more losses (should increment to 5)
        for _ in range(2):
            breakers.record_loss()

        # Should now be 5 consecutive losses
        assert breakers._consecutive_losses == 5


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def breakers(self):
        """Create circuit breakers with default config."""
        return CircuitBreakers()

    def test_none_values_in_market_conditions(self, breakers):
        """Test that None values are handled gracefully."""
        # All None - should not change state
        breakers.update_market_conditions(
            current_atr=None,
            normal_atr=None,
            spread_ticks=None,
            volume_pct=None,
        )

        assert breakers.can_trade() is True
        assert breakers.get_size_multiplier() == 1.0

    def test_partial_market_conditions(self, breakers):
        """Test that partial market conditions are handled."""
        # Only volatility data
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=None,  # Unknown
            volume_pct=None,    # Unknown
        )

        # Should trigger high volatility but not affect spread/volume
        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        assert BreakerType.WIDE_SPREAD not in breakers.state.active_breakers
        assert BreakerType.LOW_VOLUME not in breakers.state.active_breakers

    def test_negative_atr_handled(self, breakers):
        """Test that negative ATR values are handled."""
        # Negative ATR shouldn't crash
        breakers.update_market_conditions(
            current_atr=-1.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85,
        )

        # Negative ratio won't trigger high volatility
        assert BreakerType.HIGH_VOLATILITY not in breakers.state.active_breakers

    def test_zero_spread_handled(self, breakers):
        """Test that zero spread is handled correctly."""
        breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=0,  # Zero spread (very tight)
            volume_pct=0.85,
        )

        # Should not trigger wide spread
        assert BreakerType.WIDE_SPREAD not in breakers.state.active_breakers
        assert breakers.can_trade() is True

    def test_custom_config(self):
        """Test circuit breakers with custom configuration."""
        config = CircuitBreakerConfig(
            loss_3_pause_seconds=60,   # 1 minute
            loss_5_pause_seconds=120,  # 2 minutes
            volatility_multiplier_threshold=2.0,  # Lower threshold
            max_spread_ticks=1,  # Tighter spread limit
            min_volume_pct=0.20,  # Higher volume requirement
        )

        breakers = CircuitBreakers(config=config)

        # Test with values that would pass default config but fail custom
        breakers.update_market_conditions(
            current_atr=2.5,  # Would pass 3.0 default, fails 2.0 custom
            normal_atr=1.0,
            spread_ticks=1.5,  # Would pass 2 default, fails 1 custom
            volume_pct=0.15,  # Would pass 10% default, fails 20% custom
        )

        assert BreakerType.HIGH_VOLATILITY in breakers.state.active_breakers
        assert BreakerType.WIDE_SPREAD in breakers.state.active_breakers
        assert BreakerType.LOW_VOLUME in breakers.state.active_breakers

    def test_get_status_is_complete(self, breakers):
        """Test that get_status returns complete information."""
        # Trigger multiple conditions
        breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=3,
            volume_pct=0.05,
        )

        status = breakers.get_status()

        # Verify all expected keys are present
        assert 'can_trade' in status
        assert 'is_paused' in status
        assert 'pause_until' in status
        assert 'pause_reason' in status
        assert 'is_halted' in status
        assert 'halt_reason' in status
        assert 'requires_manual_review' in status
        assert 'size_multiplier' in status
        assert 'size_reduction_reasons' in status
        assert 'consecutive_losses' in status
        assert 'active_breakers' in status

        # Verify values make sense
        assert status['can_trade'] is False  # Paused due to wide spread
        assert status['is_paused'] is True
        assert status['size_multiplier'] == 0.5
        assert len(status['active_breakers']) >= 3


# =============================================================================
# Check Market Conditions Function Tests
# =============================================================================

class TestCheckMarketConditionsFunction:
    """Test the standalone check_market_conditions function."""

    def test_all_conditions_good(self):
        """Test when all market conditions are good."""
        result = check_market_conditions(
            atr=1.0,
            baseline_atr=1.0,
            spread_ticks=1,
            volume_ratio=0.50,
        )

        assert result['volatility_normal'] is True
        assert result['spread_acceptable'] is True
        assert result['volume_adequate'] is True
        assert result['tradeable'] is True

    def test_volatility_only_bad(self):
        """Test when only volatility is bad."""
        result = check_market_conditions(
            atr=4.0,  # 4x volatility
            baseline_atr=1.0,
            spread_ticks=1,
            volume_ratio=0.50,
        )

        assert result['volatility_normal'] is False
        assert result['spread_acceptable'] is True
        assert result['volume_adequate'] is True
        assert result['tradeable'] is False

    def test_spread_only_bad(self):
        """Test when only spread is bad."""
        result = check_market_conditions(
            atr=1.0,
            baseline_atr=1.0,
            spread_ticks=3,  # Wide spread
            volume_ratio=0.50,
        )

        assert result['volatility_normal'] is True
        assert result['spread_acceptable'] is False
        assert result['volume_adequate'] is True
        assert result['tradeable'] is False

    def test_volume_only_bad(self):
        """Test when only volume is bad."""
        result = check_market_conditions(
            atr=1.0,
            baseline_atr=1.0,
            spread_ticks=1,
            volume_ratio=0.05,  # Low volume
        )

        assert result['volatility_normal'] is True
        assert result['spread_acceptable'] is True
        assert result['volume_adequate'] is False
        assert result['tradeable'] is False

    def test_all_conditions_bad(self):
        """Test when all market conditions are bad."""
        result = check_market_conditions(
            atr=4.0,
            baseline_atr=1.0,
            spread_ticks=3,
            volume_ratio=0.05,
        )

        assert result['volatility_normal'] is False
        assert result['spread_acceptable'] is False
        assert result['volume_adequate'] is False
        assert result['tradeable'] is False

    def test_boundary_values(self):
        """Test boundary values for all conditions."""
        # All at exact boundaries (should be tradeable)
        result = check_market_conditions(
            atr=3.0,       # Exactly 3x
            baseline_atr=1.0,
            spread_ticks=2,    # Exactly 2
            volume_ratio=0.10,  # Exactly 10%
        )

        # All at boundary should be acceptable (condition is <= or >=)
        assert result['volatility_normal'] is True
        assert result['spread_acceptable'] is True
        assert result['volume_adequate'] is True
        assert result['tradeable'] is True
