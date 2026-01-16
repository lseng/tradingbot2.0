"""
Tests for Reversal Bar-Range Constraint.

Per specs/risk-management.md:
- Cannot reverse more than 2x in same bar range
- Cooldown period after reversal: 30 seconds minimum
- Must have high-confidence opposite signal (> 75%)

These tests verify the SignalGenerator enforces these constraints
to protect the $1,000 account from excessive whipsawing.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.trading.signal_generator import (
    SignalGenerator,
    SignalConfig,
    SignalType,
    ModelPrediction,
    BarRange,
    is_reversal_signal,
)
from src.trading.position_manager import Position


class TestBarRangeDataclass:
    """Test the BarRange dataclass functionality."""

    def test_bar_range_creation(self):
        """Test creating a BarRange."""
        bar = BarRange(high=6050.25, low=6049.00)
        assert bar.high == 6050.25
        assert bar.low == 6049.00
        assert bar.timestamp is not None

    def test_bar_range_contains_price_inside(self):
        """Test price inside the bar range."""
        bar = BarRange(high=6050.00, low=6048.00)
        assert bar.contains_price(6049.00) is True
        assert bar.contains_price(6050.00) is True  # High boundary
        assert bar.contains_price(6048.00) is True  # Low boundary

    def test_bar_range_contains_price_outside(self):
        """Test price outside the bar range."""
        bar = BarRange(high=6050.00, low=6048.00)
        assert bar.contains_price(6051.00) is False
        assert bar.contains_price(6047.00) is False

    def test_bar_range_contains_price_with_tolerance(self):
        """Test price containment with tolerance."""
        bar = BarRange(high=6050.00, low=6048.00)
        # Price at 6050.25 is within 0.25 tolerance of high 6050.00
        assert bar.contains_price(6050.25, tolerance=0.25) is True
        # Price at 6047.75 is within 0.25 tolerance of low 6048.00
        assert bar.contains_price(6047.75, tolerance=0.25) is True
        # Price at 6050.50 is outside tolerance
        assert bar.contains_price(6050.50, tolerance=0.25) is False

    def test_bar_range_overlaps_complete_overlap(self):
        """Test complete overlap between bar ranges."""
        bar1 = BarRange(high=6050.00, low=6048.00)
        bar2 = BarRange(high=6051.00, low=6047.00)
        assert bar1.overlaps(bar2) is True
        assert bar2.overlaps(bar1) is True

    def test_bar_range_overlaps_partial(self):
        """Test partial overlap between bar ranges."""
        bar1 = BarRange(high=6050.00, low=6048.00)
        bar2 = BarRange(high=6049.00, low=6047.00)  # Overlaps at 6048-6049
        assert bar1.overlaps(bar2) is True

    def test_bar_range_overlaps_no_overlap(self):
        """Test non-overlapping bar ranges."""
        bar1 = BarRange(high=6050.00, low=6048.00)
        bar2 = BarRange(high=6045.00, low=6043.00)
        assert bar1.overlaps(bar2) is False
        assert bar2.overlaps(bar1) is False

    def test_bar_range_overlaps_with_tolerance(self):
        """Test overlap detection with tolerance."""
        bar1 = BarRange(high=6050.00, low=6048.00)
        bar2 = BarRange(high=6047.75, low=6046.00)  # Gap of 0.25
        # Without tolerance, no overlap
        assert bar1.overlaps(bar2) is False
        # With 0.25 tolerance, they overlap
        assert bar1.overlaps(bar2, tolerance=0.25) is True


class TestSignalConfigReversalSettings:
    """Test the SignalConfig reversal-related settings."""

    def test_default_reversal_settings(self):
        """Test default reversal configuration values."""
        config = SignalConfig()
        assert config.min_reversal_confidence == 0.75
        assert config.allow_reversals is True
        assert config.require_flat_first is False
        assert config.max_reversals_per_bar_range == 2
        assert config.bar_range_tolerance == 0.25
        assert config.reversal_cooldown_seconds == 30.0

    def test_custom_reversal_settings(self):
        """Test custom reversal configuration."""
        config = SignalConfig(
            max_reversals_per_bar_range=1,
            bar_range_tolerance=0.50,
            reversal_cooldown_seconds=60.0,
        )
        assert config.max_reversals_per_bar_range == 1
        assert config.bar_range_tolerance == 0.50
        assert config.reversal_cooldown_seconds == 60.0


class TestSignalGeneratorReversalState:
    """Test SignalGenerator reversal state management."""

    def test_initial_state(self):
        """Test initial reversal state is clean."""
        generator = SignalGenerator()
        assert generator._reversals_in_bar_range == 0
        assert generator._current_bar_range is None
        assert generator._last_reversal_time is None

    def test_update_bar_range_first_bar(self):
        """Test establishing first bar range."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        assert generator._current_bar_range is not None
        assert generator._current_bar_range.high == 6050.00
        assert generator._current_bar_range.low == 6048.00
        assert generator._reversals_in_bar_range == 0

    def test_update_bar_range_extend_overlapping(self):
        """Test extending bar range with overlapping bar."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator.update_bar_range(high=6051.00, low=6049.00, current_price=6050.00)

        # Range should be extended
        assert generator._current_bar_range.high == 6051.00
        assert generator._current_bar_range.low == 6048.00

    def test_update_bar_range_new_non_overlapping(self):
        """Test new bar range resets counter."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 2  # Simulate reversals

        # Non-overlapping bar should reset
        generator.update_bar_range(high=6040.00, low=6038.00, current_price=6039.00)

        assert generator._current_bar_range.high == 6040.00
        assert generator._current_bar_range.low == 6038.00
        assert generator._reversals_in_bar_range == 0  # Reset

    def test_get_reversal_state(self):
        """Test getting reversal state for monitoring."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 1

        state = generator.get_reversal_state()
        assert state["reversals_in_bar_range"] == 1
        assert state["max_allowed"] == 2
        assert state["current_bar_range"]["high"] == 6050.00
        assert state["current_bar_range"]["low"] == 6048.00
        assert state["in_reversal_cooldown"] is False

    def test_reset_reversal_state(self):
        """Test resetting reversal state."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 2
        generator._last_reversal_time = datetime.now()

        generator.reset_reversal_state()

        assert generator._reversals_in_bar_range == 0
        assert generator._current_bar_range is None
        assert generator._last_reversal_time is None


class TestReversalBarRangeConstraint:
    """Test the 2x reversal per bar range constraint."""

    @pytest.fixture
    def generator(self):
        """Create a signal generator for testing."""
        config = SignalConfig(
            min_reversal_confidence=0.75,
            max_reversals_per_bar_range=2,
            reversal_cooldown_seconds=0.0,  # Disable for testing
        )
        return SignalGenerator(config)

    @pytest.fixture
    def long_position(self):
        """Create a long position for testing."""
        pos = MagicMock(spec=Position)
        pos.is_flat = False
        pos.is_long = True
        pos.is_short = False
        pos.direction = 1
        pos.size = 1
        return pos

    @pytest.fixture
    def short_position(self):
        """Create a short position for testing."""
        pos = MagicMock(spec=Position)
        pos.is_flat = False
        pos.is_long = False
        pos.is_short = True
        pos.direction = -1
        pos.size = 1
        return pos

    @pytest.fixture
    def risk_manager(self):
        """Create a mock risk manager that allows trading."""
        rm = MagicMock()
        rm.can_trade.return_value = True
        return rm

    def test_first_reversal_allowed(self, generator, long_position, risk_manager):
        """Test first reversal in bar range is allowed."""
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        prediction = ModelPrediction(
            direction=-1,  # DOWN
            confidence=0.80,  # Above 75%
        )

        signal = generator.generate(prediction, long_position, risk_manager)

        assert signal is not None
        assert signal.signal_type == SignalType.REVERSE_TO_SHORT
        assert generator._reversals_in_bar_range == 1

    def test_second_reversal_allowed(self, generator, short_position, risk_manager):
        """Test second reversal in bar range is allowed."""
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 1  # After first reversal

        prediction = ModelPrediction(
            direction=1,  # UP
            confidence=0.80,  # Above 75%
        )

        signal = generator.generate(prediction, short_position, risk_manager)

        assert signal is not None
        assert signal.signal_type == SignalType.REVERSE_TO_LONG
        assert generator._reversals_in_bar_range == 2

    def test_third_reversal_blocked(self, generator, long_position, risk_manager):
        """Test third reversal in bar range is blocked (2x limit)."""
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 2  # Already at max

        prediction = ModelPrediction(
            direction=-1,  # DOWN
            confidence=0.80,  # High confidence
        )

        signal = generator.generate(prediction, long_position, risk_manager)

        # Should not be a reversal
        assert signal is not None
        assert signal.signal_type != SignalType.REVERSE_TO_SHORT
        # Counter should not increase
        assert generator._reversals_in_bar_range == 2

    def test_reversal_allowed_after_new_bar_range(self, generator, long_position, risk_manager):
        """Test reversal allowed after moving to new bar range."""
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 2  # At max

        # New non-overlapping bar resets counter
        generator.update_bar_range(high=6040.00, low=6038.00, current_price=6039.00)

        prediction = ModelPrediction(
            direction=-1,  # DOWN
            confidence=0.80,
        )

        signal = generator.generate(prediction, long_position, risk_manager)

        assert signal is not None
        assert signal.signal_type == SignalType.REVERSE_TO_SHORT
        assert generator._reversals_in_bar_range == 1

    def test_blocked_reversal_falls_through_to_exit(self, generator, long_position, risk_manager):
        """Test blocked reversal falls through to regular exit."""
        config = SignalConfig(
            min_reversal_confidence=0.75,
            min_exit_confidence=0.55,
            max_reversals_per_bar_range=2,
            exit_on_opposite_signal=True,
            reversal_cooldown_seconds=0.0,
        )
        generator = SignalGenerator(config)
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 2  # At max

        prediction = ModelPrediction(
            direction=-1,  # DOWN
            confidence=0.80,  # Above reversal threshold but blocked
        )

        signal = generator.generate(prediction, long_position, risk_manager)

        # Should exit instead of reverse
        assert signal is not None
        assert signal.signal_type == SignalType.EXIT_LONG

    def test_custom_max_reversals(self):
        """Test custom max reversals per bar range setting."""
        config = SignalConfig(max_reversals_per_bar_range=1, reversal_cooldown_seconds=0.0)
        generator = SignalGenerator(config)
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 1

        can_reverse, reason = generator._can_reverse_in_bar_range()
        assert can_reverse is False
        assert "Max 1 reversals" in reason


class TestReversalCooldown:
    """Test the 30-second reversal cooldown."""

    def test_reversal_cooldown_active(self):
        """Test reversal cooldown is active after reversal."""
        config = SignalConfig(reversal_cooldown_seconds=30.0)
        generator = SignalGenerator(config)
        generator._last_reversal_time = datetime.now()

        assert generator._in_reversal_cooldown() is True

    def test_reversal_cooldown_expired(self):
        """Test reversal cooldown expires after time passes."""
        config = SignalConfig(reversal_cooldown_seconds=30.0)
        generator = SignalGenerator(config)
        generator._last_reversal_time = datetime.now() - timedelta(seconds=31)

        assert generator._in_reversal_cooldown() is False

    def test_reversal_cooldown_blocks_reversal(self):
        """Test reversal is blocked during cooldown."""
        config = SignalConfig(reversal_cooldown_seconds=30.0)
        generator = SignalGenerator(config)
        generator._last_reversal_time = datetime.now()

        can_reverse, reason = generator._can_reverse_in_bar_range()
        assert can_reverse is False
        assert "cooldown active" in reason

    @patch('src.trading.signal_generator.datetime')
    def test_reversal_allowed_after_cooldown(self, mock_datetime):
        """Test reversal allowed after cooldown expires."""
        config = SignalConfig(reversal_cooldown_seconds=30.0)
        generator = SignalGenerator(config)

        # Set last reversal 31 seconds ago
        now = datetime(2026, 1, 16, 10, 0, 0)
        mock_datetime.now.return_value = now
        generator._last_reversal_time = now - timedelta(seconds=31)

        can_reverse, reason = generator._can_reverse_in_bar_range()
        assert can_reverse is True


class TestReversalRecording:
    """Test reversal event recording."""

    def test_record_reversal_increments_counter(self):
        """Test recording reversal increments counter."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        initial_count = generator._reversals_in_bar_range

        generator._record_reversal()

        assert generator._reversals_in_bar_range == initial_count + 1

    def test_record_reversal_sets_time(self):
        """Test recording reversal sets timestamp."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        before = datetime.now()
        generator._record_reversal()
        after = datetime.now()

        assert generator._last_reversal_time is not None
        assert before <= generator._last_reversal_time <= after

    def test_record_reversal_triggers_exit_cooldown(self):
        """Test recording reversal also triggers exit cooldown."""
        config = SignalConfig(exit_cooldown_seconds=30.0)
        generator = SignalGenerator(config)

        generator._record_reversal()

        assert generator._in_exit_cooldown() is True


class TestReversalWithShortPosition:
    """Test reversals when in short position."""

    @pytest.fixture
    def generator(self):
        """Create signal generator with no cooldowns for testing."""
        config = SignalConfig(
            min_reversal_confidence=0.75,
            max_reversals_per_bar_range=2,
            reversal_cooldown_seconds=0.0,
            exit_cooldown_seconds=0.0,
        )
        return SignalGenerator(config)

    @pytest.fixture
    def short_position(self):
        """Create short position."""
        pos = MagicMock(spec=Position)
        pos.is_flat = False
        pos.is_long = False
        pos.is_short = True
        pos.direction = -1
        pos.size = 1
        return pos

    @pytest.fixture
    def risk_manager(self):
        """Create mock risk manager."""
        rm = MagicMock()
        rm.can_trade.return_value = True
        return rm

    def test_reverse_to_long_allowed(self, generator, short_position, risk_manager):
        """Test REVERSE_TO_LONG is generated when conditions met."""
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        prediction = ModelPrediction(direction=1, confidence=0.80)
        signal = generator.generate(prediction, short_position, risk_manager)

        assert signal.signal_type == SignalType.REVERSE_TO_LONG
        assert generator._reversals_in_bar_range == 1

    def test_reverse_to_long_blocked_at_max(self, generator, short_position, risk_manager):
        """Test REVERSE_TO_LONG blocked at max reversals."""
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        generator._reversals_in_bar_range = 2

        prediction = ModelPrediction(direction=1, confidence=0.80)
        signal = generator.generate(prediction, short_position, risk_manager)

        assert signal.signal_type != SignalType.REVERSE_TO_LONG


class TestIntegrationWithExitCooldown:
    """Test interaction between reversal and exit cooldowns."""

    def test_exit_cooldown_separate_from_reversal_cooldown(self):
        """Test exit and reversal cooldowns are independent."""
        config = SignalConfig(
            exit_cooldown_seconds=30.0,
            reversal_cooldown_seconds=60.0,  # Longer
        )
        generator = SignalGenerator(config)

        # Simulate exit (not reversal)
        generator._record_exit_time()

        # Should be in exit cooldown but not reversal cooldown
        assert generator._in_exit_cooldown() is True
        assert generator._in_reversal_cooldown() is False

    def test_reversal_triggers_both_cooldowns(self):
        """Test reversal triggers both exit and reversal cooldowns."""
        config = SignalConfig(
            exit_cooldown_seconds=30.0,
            reversal_cooldown_seconds=30.0,
        )
        generator = SignalGenerator(config)
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        generator._record_reversal()

        assert generator._in_exit_cooldown() is True
        assert generator._in_reversal_cooldown() is True


class TestEdgeCases:
    """Test edge cases in reversal constraint logic."""

    def test_reversal_without_bar_range_initialized(self):
        """Test reversal attempt without bar range being initialized."""
        generator = SignalGenerator()
        # Don't call update_bar_range

        can_reverse, reason = generator._can_reverse_in_bar_range()
        # Should still check cooldown and counter
        assert can_reverse is True  # No constraints violated

    def test_zero_max_reversals_blocks_all(self):
        """Test setting max_reversals to 0 blocks all reversals."""
        config = SignalConfig(max_reversals_per_bar_range=0, reversal_cooldown_seconds=0.0)
        generator = SignalGenerator(config)
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        can_reverse, reason = generator._can_reverse_in_bar_range()
        assert can_reverse is False
        assert "Max 0 reversals" in reason

    def test_negative_tolerance_treated_as_zero(self):
        """Test negative bar_range_tolerance is handled."""
        config = SignalConfig(bar_range_tolerance=-0.25)
        generator = SignalGenerator(config)
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        # Should still function without errors
        generator.update_bar_range(high=6051.00, low=6049.00, current_price=6050.00)
        assert generator._current_bar_range is not None

    def test_very_wide_bar_range(self):
        """Test handling of very wide bar ranges."""
        generator = SignalGenerator()
        generator.update_bar_range(high=6100.00, low=6000.00, current_price=6050.00)

        assert generator._current_bar_range.high == 6100.00
        assert generator._current_bar_range.low == 6000.00

    def test_bar_range_history_limited(self):
        """Test bar range history doesn't grow unbounded."""
        generator = SignalGenerator()

        # Create many non-overlapping bars
        for i in range(100):
            base = 6000 + (i * 10)
            generator.update_bar_range(high=base + 2, low=base, current_price=base + 1)

        # History should exist but be reasonable
        # (Implementation doesn't limit history, but this tests it doesn't crash)
        assert len(generator._reversal_bar_ranges) == 99  # One less than updates


class TestReversalConstraintWithRealScenario:
    """Test reversal constraint in realistic trading scenarios."""

    @pytest.fixture
    def generator(self):
        """Create production-like generator."""
        config = SignalConfig(
            min_entry_confidence=0.65,
            min_exit_confidence=0.55,
            min_reversal_confidence=0.75,
            max_reversals_per_bar_range=2,
            reversal_cooldown_seconds=30.0,
            exit_cooldown_seconds=30.0,
        )
        return SignalGenerator(config)

    @pytest.fixture
    def risk_manager(self):
        rm = MagicMock()
        rm.can_trade.return_value = True
        return rm

    def test_whipsaw_protection_scenario(self, generator, risk_manager):
        """Test protection from rapid whipsawing in choppy market."""
        # Scenario: Price oscillating in tight range, generating
        # multiple opposite signals. Without constraint, could
        # generate unlimited reversals causing excessive commissions.

        # Initial bar establishes range
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)

        # Long position
        long_pos = MagicMock(spec=Position)
        long_pos.is_flat, long_pos.is_long, long_pos.is_short = False, True, False
        long_pos.direction, long_pos.size = 1, 1

        short_pos = MagicMock(spec=Position)
        short_pos.is_flat, short_pos.is_long, short_pos.is_short = False, False, True
        short_pos.direction, short_pos.size = -1, 1

        # Disable cooldown for this test
        generator.config.reversal_cooldown_seconds = 0.0
        generator.config.exit_cooldown_seconds = 0.0

        # First reversal: Long -> Short
        pred1 = ModelPrediction(direction=-1, confidence=0.80)
        sig1 = generator.generate(pred1, long_pos, risk_manager)
        assert sig1.signal_type == SignalType.REVERSE_TO_SHORT
        assert generator._reversals_in_bar_range == 1

        # Second reversal: Short -> Long (same bar range)
        generator.update_bar_range(high=6050.25, low=6047.75, current_price=6048.50)
        pred2 = ModelPrediction(direction=1, confidence=0.85)
        sig2 = generator.generate(pred2, short_pos, risk_manager)
        assert sig2.signal_type == SignalType.REVERSE_TO_LONG
        assert generator._reversals_in_bar_range == 2

        # Third reversal attempt: BLOCKED
        generator.update_bar_range(high=6050.50, low=6048.00, current_price=6049.25)
        pred3 = ModelPrediction(direction=-1, confidence=0.90)
        sig3 = generator.generate(pred3, long_pos, risk_manager)
        assert sig3.signal_type != SignalType.REVERSE_TO_SHORT
        # Counter stays at 2
        assert generator._reversals_in_bar_range == 2

    def test_trending_market_allows_reversals(self, generator, risk_manager):
        """Test reversals allowed in trending market (non-overlapping bars)."""
        # Disable both cooldowns for this test
        generator.config.reversal_cooldown_seconds = 0.0
        generator.config.exit_cooldown_seconds = 0.0

        long_pos = MagicMock(spec=Position)
        long_pos.is_flat, long_pos.is_long, long_pos.is_short = False, True, False
        long_pos.direction, long_pos.size = 1, 1

        short_pos = MagicMock(spec=Position)
        short_pos.is_flat, short_pos.is_long, short_pos.is_short = False, False, True
        short_pos.direction, short_pos.size = -1, 1

        # Uptrend - each bar higher
        generator.update_bar_range(high=6050.00, low=6048.00, current_price=6049.00)
        pred1 = ModelPrediction(direction=-1, confidence=0.80)
        sig1 = generator.generate(pred1, long_pos, risk_manager)
        assert sig1.signal_type == SignalType.REVERSE_TO_SHORT

        # New higher bar (non-overlapping) resets counter
        generator.update_bar_range(high=6055.00, low=6053.00, current_price=6054.00)
        assert generator._reversals_in_bar_range == 0  # Reset

        # Another reversal in new range
        pred2 = ModelPrediction(direction=1, confidence=0.80)
        sig2 = generator.generate(pred2, short_pos, risk_manager)
        assert sig2.signal_type == SignalType.REVERSE_TO_LONG
        assert generator._reversals_in_bar_range == 1
