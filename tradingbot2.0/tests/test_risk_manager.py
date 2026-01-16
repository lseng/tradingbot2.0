"""
Comprehensive unit tests for the Risk Management module.

Tests cover:
- RiskManager: Daily limits, kill switch, state persistence
- PositionSizer: Balance tiers, confidence scaling
- StopLossManager: ATR/fixed/structure/trailing stops
- EODManager: Time-based phases, flatten logic
- CircuitBreakers: Consecutive losses, market conditions

These tests ensure the risk management system protects the $1,000 starting capital.
"""

import pytest
import tempfile
import json
from datetime import datetime, date, time, timedelta
from pathlib import Path
from unittest.mock import patch

# Import risk management modules
from src.risk.risk_manager import (
    RiskManager, RiskState, RiskLimits, TradingStatus
)
from src.risk.position_sizing import (
    PositionSizer, PositionSizeResult, PositionSizingConfig, calculate_position_size
)
from src.risk.stops import (
    StopLossManager, StopConfig, StopType, StopResult, calculate_atr
)
from src.risk.eod_manager import (
    EODManager, EODConfig, EODPhase, EODStatus, NY_TZ, time_to_ny, get_ny_time, is_market_open
)
from src.risk.circuit_breakers import (
    CircuitBreakers, CircuitBreakerConfig, CircuitBreakerState, BreakerType, check_market_conditions
)

import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def risk_limits():
    """Default risk limits for testing."""
    return RiskLimits()


@pytest.fixture
def risk_manager(risk_limits):
    """Risk manager with default limits."""
    return RiskManager(limits=risk_limits, auto_persist=False)


@pytest.fixture
def position_sizer():
    """Position sizer with default config."""
    return PositionSizer()


@pytest.fixture
def stop_manager():
    """Stop loss manager with default config."""
    return StopLossManager()


@pytest.fixture
def eod_manager():
    """EOD manager with default config."""
    return EODManager()


@pytest.fixture
def circuit_breakers():
    """Circuit breakers with default config."""
    return CircuitBreakers()


# =============================================================================
# RiskLimits Tests
# =============================================================================

class TestRiskLimits:
    """Tests for RiskLimits configuration."""

    def test_default_limits_match_spec(self):
        """Verify default limits match specification values."""
        limits = RiskLimits()

        assert limits.starting_capital == 1000.0
        assert limits.min_account_balance == 700.0
        assert limits.max_daily_loss == 50.0  # 5%
        assert limits.max_daily_drawdown == 75.0  # 7.5%
        assert limits.max_per_trade_risk == 25.0  # 2.5%
        assert limits.kill_switch_loss == 300.0  # 30%
        assert limits.max_account_drawdown == 200.0  # 20%
        assert limits.min_confidence == 0.60

    def test_mes_contract_specs(self):
        """Verify MES contract specifications are correct."""
        limits = RiskLimits()

        assert limits.tick_size == 0.25
        assert limits.tick_value == 1.25
        assert limits.point_value == 5.0
        assert limits.round_trip_commission == 0.84


# =============================================================================
# RiskManager Tests
# =============================================================================

class TestRiskManager:
    """Tests for core RiskManager functionality."""

    def test_initialization(self, risk_manager):
        """Test risk manager initializes with correct state."""
        assert risk_manager.state.account_balance == 1000.0
        assert risk_manager.state.peak_balance == 1000.0
        assert risk_manager.state.status == TradingStatus.ACTIVE
        assert risk_manager.state.daily_pnl == 0.0
        assert risk_manager.state.consecutive_losses == 0

    def test_can_trade_initial(self, risk_manager):
        """Test trading is allowed initially."""
        assert risk_manager.can_trade() is True

    def test_approve_trade_valid(self, risk_manager):
        """Test valid trade is approved."""
        assert risk_manager.approve_trade(
            risk_amount=20.0,
            confidence=0.75
        ) is True

    def test_reject_low_confidence(self, risk_manager):
        """Test trade rejected for low confidence."""
        assert risk_manager.approve_trade(
            risk_amount=20.0,
            confidence=0.55  # Below 60% threshold
        ) is False

    def test_reject_excessive_risk(self, risk_manager):
        """Test trade rejected for excessive risk."""
        assert risk_manager.approve_trade(
            risk_amount=30.0,  # Above $25 max
            confidence=0.75
        ) is False

    def test_record_winning_trade(self, risk_manager):
        """Test recording a winning trade."""
        risk_manager.record_trade_result(pnl=15.0)

        assert risk_manager.state.account_balance == 1015.0
        assert risk_manager.state.daily_pnl == 15.0
        assert risk_manager.state.wins_today == 1
        assert risk_manager.state.consecutive_losses == 0

    def test_record_losing_trade(self, risk_manager):
        """Test recording a losing trade."""
        risk_manager.record_trade_result(pnl=-10.0)

        assert risk_manager.state.account_balance == 990.0
        assert risk_manager.state.daily_pnl == -10.0
        assert risk_manager.state.losses_today == 1
        assert risk_manager.state.consecutive_losses == 1
        assert risk_manager.state.cumulative_loss == 10.0

    def test_daily_loss_limit(self, risk_manager):
        """Test daily loss limit triggers stop for day."""
        # Lose $50 (5% of $1000)
        risk_manager.record_trade_result(pnl=-50.0)

        assert risk_manager.state.status == TradingStatus.STOPPED_FOR_DAY
        assert risk_manager.can_trade() is False

    def test_consecutive_losses_pause(self, risk_manager):
        """Test consecutive losses trigger pause."""
        # Record 5 consecutive losses
        for _ in range(5):
            risk_manager.record_trade_result(pnl=-5.0)

        assert risk_manager.state.status == TradingStatus.PAUSED
        assert risk_manager.state.consecutive_losses == 5
        assert risk_manager.can_trade() is False

    def test_kill_switch(self, risk_manager):
        """Test kill switch halts trading permanently."""
        # Simulate cumulative loss of $300
        for _ in range(30):
            risk_manager.state.cumulative_loss += 10.0

        # Check limits
        risk_manager._check_risk_limits()

        assert risk_manager.state.status == TradingStatus.HALTED
        assert risk_manager.can_trade() is False

    def test_min_balance_check(self, risk_manager):
        """Test trading blocked when below minimum balance."""
        risk_manager.state.account_balance = 650.0  # Below $700 min

        assert risk_manager.can_trade() is False

    def test_reset_daily_state(self, risk_manager):
        """Test daily state reset."""
        # Simulate a trading day
        risk_manager.record_trade_result(pnl=10.0)
        risk_manager.record_trade_result(pnl=-5.0)

        # Reset for new day
        risk_manager.reset_daily_state()

        assert risk_manager.state.daily_pnl == 0.0
        assert risk_manager.state.trades_today == 0
        assert risk_manager.state.wins_today == 0
        assert risk_manager.state.losses_today == 0

    def test_remaining_daily_risk(self, risk_manager):
        """Test remaining daily risk calculation."""
        assert risk_manager.get_remaining_daily_risk() == 50.0  # Full budget

        risk_manager.record_trade_result(pnl=-20.0)
        assert risk_manager.get_remaining_daily_risk() == 30.0  # $50 - $20

    def test_state_persistence(self, risk_limits):
        """Test state persists to and loads from file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_file = Path(f.name)

        try:
            # Create manager and make some trades
            manager1 = RiskManager(limits=risk_limits, state_file=state_file)
            manager1.record_trade_result(pnl=15.0)
            manager1.record_trade_result(pnl=-10.0)

            # Create new manager that should load state
            manager2 = RiskManager(limits=risk_limits, state_file=state_file)

            assert manager2.state.account_balance == manager1.state.account_balance
            assert manager2.state.total_trades == 2
        finally:
            state_file.unlink()

    def test_manual_resume(self, risk_manager):
        """Test manual resume after drawdown review."""
        # Trigger manual review
        risk_manager.state.status = TradingStatus.MANUAL_REVIEW

        # Resume trading
        result = risk_manager.manual_resume(new_balance=850.0)

        assert result is True
        assert risk_manager.state.status == TradingStatus.ACTIVE
        assert risk_manager.state.account_balance == 850.0

    def test_cannot_resume_after_kill_switch(self, risk_manager):
        """Test cannot resume after kill switch."""
        risk_manager.state.status = TradingStatus.HALTED
        risk_manager.state.halt_reason = "Kill switch triggered"

        result = risk_manager.manual_resume()

        assert result is False
        assert risk_manager.state.status == TradingStatus.HALTED

    def test_metrics(self, risk_manager):
        """Test metrics dictionary contains expected keys."""
        metrics = risk_manager.get_metrics()

        assert 'account_balance' in metrics
        assert 'daily_pnl' in metrics
        assert 'daily_drawdown' in metrics
        assert 'consecutive_losses' in metrics
        assert 'remaining_daily_risk' in metrics
        assert 'status' in metrics


# =============================================================================
# PositionSizer Tests
# =============================================================================

class TestPositionSizer:
    """Tests for position sizing calculations."""

    def test_default_config_matches_spec(self):
        """Verify default config matches specification."""
        config = PositionSizingConfig()

        assert config.tick_value == 1.25
        assert config.min_balance == 700.0
        assert config.default_risk_pct == 0.02  # 2%

    def test_calculate_for_1000_balance(self, position_sizer):
        """Test position size for $1000 balance tier."""
        result = position_sizer.calculate(
            account_balance=1000.0,
            stop_ticks=8,
            confidence=0.75
        )

        # $1000 is the END of the $700-$1000 tier (1 contract max, inclusive)
        # Per spec: "$700-$1,000: 1 contract" - boundary is inclusive
        assert result.max_contracts_for_tier == 1  # $700-$1000 tier
        assert result.confidence_multiplier == 1.0  # 70-80% = 1x
        assert result.risk_per_contract == 10.0  # 8 ticks * $1.25

    def test_calculate_for_1500_balance(self, position_sizer):
        """Test position size for $1500 balance tier."""
        result = position_sizer.calculate(
            account_balance=1500.0,
            stop_ticks=8,
            confidence=0.85
        )

        # $1500 is the END of the $1000-$1500 tier (2 contracts max, inclusive)
        # Per spec: "$1,000-$1,500: 2 contracts" - boundary is inclusive
        # $1500 * 2% = $30 risk budget
        # $30 / $10 per contract = 3 contracts base
        # 85% confidence = 1.5x multiplier, so 3 * 1.5 = 4.5 -> 4
        # Capped at tier max of 2
        assert result.max_contracts_for_tier == 2  # $1000-$1500 tier
        assert result.contracts <= 2  # Capped at tier max

    def test_reject_below_min_balance(self, position_sizer):
        """Test zero contracts when below minimum balance."""
        result = position_sizer.calculate(
            account_balance=650.0,  # Below $700
            stop_ticks=8,
            confidence=0.75
        )

        assert result.contracts == 0
        assert "below minimum" in result.reason.lower()

    def test_reject_low_confidence(self, position_sizer):
        """Test zero contracts when confidence too low."""
        result = position_sizer.calculate(
            account_balance=1000.0,
            stop_ticks=8,
            confidence=0.55  # Below 60%
        )

        assert result.contracts == 0
        assert result.confidence_multiplier == 0

    def test_confidence_multipliers(self, position_sizer):
        """Test confidence multipliers are applied correctly."""
        # Below 60%: 0x (no trade)
        result = position_sizer.calculate(1000.0, 8, 0.55)
        assert result.confidence_multiplier == 0.0
        assert result.contracts == 0

        # At 60%: 0.5x (exactly at threshold)
        result = position_sizer.calculate(1000.0, 8, 0.60)
        assert result.confidence_multiplier == 0.5

        # 60-70%: 0.5x
        result = position_sizer.calculate(1000.0, 8, 0.65)
        assert result.confidence_multiplier == 0.5

        # 70-80%: 1.0x
        result = position_sizer.calculate(1000.0, 8, 0.75)
        assert result.confidence_multiplier == 1.0

        # 80-90%: 1.5x
        result = position_sizer.calculate(1000.0, 8, 0.85)
        assert result.confidence_multiplier == 1.5

        # >90%: 2.0x
        result = position_sizer.calculate(1000.0, 8, 0.95)
        assert result.confidence_multiplier == 2.0

    def test_balance_tiers(self, position_sizer):
        """Test balance tier max contracts."""
        # $700-$1000: 1 contract
        info = position_sizer.get_tier_info(800.0)
        assert info['max_contracts'] == 1

        # $1000-$1500: 2 contracts
        info = position_sizer.get_tier_info(1200.0)
        assert info['max_contracts'] == 2

        # $1500-$2000: 3 contracts
        info = position_sizer.get_tier_info(1800.0)
        assert info['max_contracts'] == 3

        # $2000-$3000: 4 contracts
        info = position_sizer.get_tier_info(2500.0)
        assert info['max_contracts'] == 4

        # $3000+: more contracts, lower risk %
        info = position_sizer.get_tier_info(4000.0)
        assert info['max_contracts'] >= 5
        assert info['risk_pct'] == 0.015  # 1.5%

    def test_calculate_max_stop_for_risk(self, position_sizer):
        """Test max stop calculation for risk budget."""
        max_ticks = position_sizer.calculate_max_stop_for_risk(
            account_balance=1000.0,
            target_risk_dollars=20.0,
            contracts=1
        )

        # $20 / $1.25 per tick = 16 ticks max
        assert max_ticks == 16.0

    def test_calculate_contracts_for_risk(self, position_sizer):
        """Test contracts calculation for target risk."""
        contracts = position_sizer.calculate_contracts_for_risk(
            target_risk_dollars=20.0,
            stop_ticks=8
        )

        # $20 / ($8 ticks * $1.25) = 2 contracts
        assert contracts == 2

    def test_simple_position_size_function(self):
        """Test standalone calculate_position_size function."""
        size = calculate_position_size(
            account_balance=1000.0,
            stop_ticks=8,
            confidence=0.75,
            max_contracts=1
        )

        assert size == 1


# =============================================================================
# StopLossManager Tests
# =============================================================================

class TestStopLossManager:
    """Tests for stop loss calculations."""

    def test_atr_stop_long(self, stop_manager):
        """Test ATR-based stop for long position."""
        result = stop_manager.calculate_atr_stop(
            entry_price=6050.00,
            direction=1,  # Long
            atr=2.0  # 2 point ATR
        )

        assert result.stop_type == StopType.ATR
        assert result.stop_price < 6050.00  # Stop below entry for long
        # ATR 2.0 * 1.5 mult = 3 points = 12 ticks
        assert result.stop_ticks == 12
        assert result.stop_price == 6047.00  # 6050 - 3

    def test_atr_stop_short(self, stop_manager):
        """Test ATR-based stop for short position."""
        result = stop_manager.calculate_atr_stop(
            entry_price=6050.00,
            direction=-1,  # Short
            atr=2.0
        )

        assert result.stop_price > 6050.00  # Stop above entry for short
        assert result.stop_price == 6053.00  # 6050 + 3

    def test_atr_stop_clamped_min(self, stop_manager):
        """Test ATR stop clamped to minimum."""
        result = stop_manager.calculate_atr_stop(
            entry_price=6050.00,
            direction=1,
            atr=0.5  # Very low ATR -> would be 3 ticks
        )

        # Should be clamped to min 4 ticks
        assert result.stop_ticks >= stop_manager.config.min_atr_ticks

    def test_atr_stop_clamped_max(self, stop_manager):
        """Test ATR stop clamped to maximum."""
        result = stop_manager.calculate_atr_stop(
            entry_price=6050.00,
            direction=1,
            atr=10.0  # Very high ATR -> would be 60 ticks
        )

        # Should be clamped to max 16 ticks
        assert result.stop_ticks <= stop_manager.config.max_atr_ticks

    def test_fixed_stop_long(self, stop_manager):
        """Test fixed tick stop for long position."""
        result = stop_manager.calculate_fixed_stop(
            entry_price=6050.00,
            direction=1,
            stop_ticks=8
        )

        assert result.stop_type == StopType.FIXED
        assert result.stop_ticks == 8
        assert result.stop_price == 6048.00  # 6050 - 2 points
        assert result.stop_dollars == 10.0  # 8 * $1.25

    def test_fixed_stop_short(self, stop_manager):
        """Test fixed tick stop for short position."""
        result = stop_manager.calculate_fixed_stop(
            entry_price=6050.00,
            direction=-1,
            stop_ticks=8
        )

        assert result.stop_price == 6052.00  # 6050 + 2 points

    def test_structure_stop_long(self, stop_manager):
        """Test structure-based stop for long position."""
        swing_lows = [6045.00, 6046.50, 6044.75]

        result = stop_manager.calculate_structure_stop(
            entry_price=6050.00,
            direction=1,
            swing_prices=swing_lows,
            buffer_ticks=2
        )

        assert result.stop_type == StopType.STRUCTURE
        # Min swing low is 6044.75, minus 2 tick buffer = 6044.25
        assert result.stop_price == 6044.25

    def test_trailing_stop_not_triggered(self, stop_manager):
        """Test trailing stop not triggered when profit insufficient."""
        new_stop = stop_manager.calculate_trailing_stop(
            entry_price=6050.00,
            current_price=6050.50,  # Only 2 ticks profit
            current_stop=6048.00,
            direction=1,
            trail_trigger_ticks=4  # Need 4 ticks to trigger
        )

        assert new_stop is None  # Not enough profit

    def test_trailing_stop_moves_to_breakeven(self, stop_manager):
        """Test trailing stop moves to breakeven when triggered."""
        new_stop = stop_manager.calculate_trailing_stop(
            entry_price=6050.00,
            current_price=6052.00,  # 8 ticks profit
            current_stop=6048.00,  # Original stop
            direction=1,
            trail_trigger_ticks=4,
            trail_distance_ticks=4
        )

        # Stop should be at 6052 - 1 point = 6051, but min is entry (6050)
        assert new_stop is not None
        assert new_stop >= 6050.00  # At least breakeven

    def test_trailing_stop_only_tightens(self, stop_manager):
        """Test trailing stop only moves in favorable direction."""
        # Stop already at breakeven
        new_stop = stop_manager.calculate_trailing_stop(
            entry_price=6050.00,
            current_price=6051.00,  # Price pulled back
            current_stop=6050.00,  # Already at breakeven
            direction=1,
            trail_trigger_ticks=4,
            trail_distance_ticks=4
        )

        # Should not loosen the stop
        assert new_stop is None

    def test_eod_tightening(self, stop_manager):
        """Test EOD stop tightening."""
        original = stop_manager.calculate_fixed_stop(6050.00, 1, 12)

        tightened = stop_manager.apply_eod_tightening(
            stop_result=original,
            entry_price=6050.00,
            direction=1,
            tighten_factor=0.75  # 25% tighter
        )

        # 12 ticks * 0.75 = 9 ticks
        assert tightened.stop_ticks == 9
        assert tightened.stop_price > original.stop_price  # Tighter = closer

    def test_calculate_target_price(self, stop_manager):
        """Test target price calculation with R:R ratio."""
        target_price, target_ticks, target_dollars = stop_manager.calculate_target_price(
            entry_price=6050.00,
            stop_price=6048.00,  # 8 tick stop
            direction=1,
            rr_ratio=2.0  # 1:2 R:R
        )

        # Stop distance = 2 points = 8 ticks
        # Target = 2x = 4 points = 16 ticks
        assert target_price == 6054.00  # 6050 + 4
        assert target_ticks == 16
        assert target_dollars == 20.0  # 16 * $1.25

    def test_calculate_atr_function(self):
        """Test standalone ATR calculation function."""
        highs = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                         111, 112, 113, 114, 115])
        lows = np.array([99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                        109, 110, 111, 112, 113])
        closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114])

        atr = calculate_atr(highs, lows, closes, period=14)

        assert len(atr) == 15
        assert np.isnan(atr[12])  # Before period-1
        assert not np.isnan(atr[13])  # First valid ATR
        assert atr[14] > 0  # Positive ATR


# =============================================================================
# EODManager Tests
# =============================================================================

class TestEODManager:
    """Tests for end-of-day management."""

    def test_normal_trading_hours(self, eod_manager):
        """Test normal phase during regular hours."""
        # 10:00 AM NY
        test_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.NORMAL
        assert status.can_open_new_positions is True
        assert status.position_size_multiplier == 1.0
        assert status.should_flatten is False

    def test_reduced_size_phase(self, eod_manager):
        """Test reduced size phase after 4:00 PM."""
        # 4:05 PM NY
        test_time = datetime(2025, 1, 15, 16, 5, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.REDUCED_SIZE
        assert status.can_open_new_positions is True
        assert status.position_size_multiplier == 0.5
        assert status.should_flatten is False

    def test_close_only_phase(self, eod_manager):
        """Test close only phase after 4:15 PM."""
        # 4:20 PM NY
        test_time = datetime(2025, 1, 15, 16, 20, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.CLOSE_ONLY
        assert status.can_open_new_positions is False
        assert status.position_size_multiplier == 0.0
        assert status.should_flatten is False

    def test_aggressive_exit_phase(self, eod_manager):
        """Test aggressive exit phase after 4:25 PM."""
        # 4:27 PM NY
        test_time = datetime(2025, 1, 15, 16, 27, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.AGGRESSIVE_EXIT
        assert status.can_open_new_positions is False
        assert status.should_flatten is True  # Must flatten!

    def test_must_be_flat_phase(self, eod_manager):
        """Test must be flat at 4:30 PM."""
        # 4:30 PM NY
        test_time = datetime(2025, 1, 15, 16, 30, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.MUST_BE_FLAT
        assert status.can_open_new_positions is False
        assert status.should_flatten is True

    def test_after_hours(self, eod_manager):
        """Test after hours phase."""
        # 5:00 PM NY (after market)
        test_time = datetime(2025, 1, 15, 17, 0, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        # Should still be MUST_BE_FLAT as it's after 4:30
        assert status.phase == EODPhase.MUST_BE_FLAT

    def test_pre_market(self, eod_manager):
        """Test pre-market phase."""
        # 9:00 AM NY (before 9:30 + buffer)
        test_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.phase == EODPhase.AFTER_HOURS
        assert status.can_open_new_positions is False

    def test_minutes_to_close_calculation(self, eod_manager):
        """Test minutes to close calculation."""
        # 4:00 PM = 30 mins to 4:30
        test_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TZ)
        mins = eod_manager.get_minutes_to_close(test_time)
        assert mins == 30

        # 10:00 AM = 390 mins to 4:30
        test_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=NY_TZ)
        mins = eod_manager.get_minutes_to_close(test_time)
        assert mins == 390

    def test_is_trading_session(self, eod_manager):
        """Test trading session check."""
        # During session
        assert eod_manager.is_trading_session(
            datetime(2025, 1, 15, 12, 0, 0, tzinfo=NY_TZ)
        ) is True

        # After session
        assert eod_manager.is_trading_session(
            datetime(2025, 1, 15, 17, 0, 0, tzinfo=NY_TZ)
        ) is False

    def test_next_session_start(self, eod_manager):
        """Test next session start calculation."""
        # During session - should return next day
        test_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=NY_TZ)  # Wednesday
        next_start = eod_manager.get_next_trading_session_start(test_time)

        assert next_start.date() == date(2025, 1, 16)  # Thursday
        assert next_start.time() == time(9, 30)

    def test_skip_weekend(self, eod_manager):
        """Test next session skips weekends."""
        # Friday after close
        test_time = datetime(2025, 1, 17, 17, 0, 0, tzinfo=NY_TZ)  # Friday
        next_start = eod_manager.get_next_trading_session_start(test_time)

        # Should skip Saturday (18) and Sunday (19), return Monday (20)
        assert next_start.weekday() == 0  # Monday

    def test_utility_functions(self):
        """Test utility functions."""
        # time_to_ny converts to NY timezone
        utc_time = datetime(2025, 1, 15, 15, 0, 0)  # 3 PM UTC
        ny_time = time_to_ny(utc_time)
        assert ny_time.tzinfo == NY_TZ

        # get_ny_time returns current NY time
        now_ny = get_ny_time()
        assert now_ny.tzinfo == NY_TZ


# =============================================================================
# CircuitBreakers Tests
# =============================================================================

class TestCircuitBreakers:
    """Tests for circuit breaker functionality."""

    def test_initial_state(self, circuit_breakers):
        """Test initial state allows trading."""
        assert circuit_breakers.can_trade() is True
        assert circuit_breakers.get_size_multiplier() == 1.0

    def test_record_win_resets_losses(self, circuit_breakers):
        """Test winning trade resets consecutive losses."""
        circuit_breakers.record_loss()
        circuit_breakers.record_loss()
        assert circuit_breakers._consecutive_losses == 2

        circuit_breakers.record_win()
        assert circuit_breakers._consecutive_losses == 0

    def test_three_losses_trigger_pause(self, circuit_breakers):
        """Test 3 consecutive losses trigger 15-min pause."""
        for _ in range(3):
            circuit_breakers.record_loss()

        assert circuit_breakers.state.is_paused is True
        assert circuit_breakers.can_trade() is False

    def test_five_losses_trigger_longer_pause(self, circuit_breakers):
        """Test 5 consecutive losses trigger 30-min pause."""
        for _ in range(5):
            circuit_breakers.record_loss()

        assert circuit_breakers.state.is_paused is True
        assert circuit_breakers._consecutive_losses == 5

    def test_daily_loss_stop(self, circuit_breakers):
        """Test daily loss triggers stop for day."""
        circuit_breakers.trigger_daily_loss_stop(
            daily_loss=50.0,
            limit=50.0
        )

        assert circuit_breakers.state.is_halted is True
        assert circuit_breakers.can_trade() is False
        assert BreakerType.DAILY_LOSS in circuit_breakers.state.active_breakers

    def test_max_drawdown_halt(self, circuit_breakers):
        """Test max drawdown triggers halt with manual review."""
        circuit_breakers.trigger_max_drawdown_halt(
            drawdown=200.0,
            limit=200.0
        )

        assert circuit_breakers.state.is_halted is True
        assert circuit_breakers.state.requires_manual_review is True

    def test_high_volatility_reduces_size(self, circuit_breakers):
        """Test high volatility reduces position size."""
        circuit_breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,  # 4x volatility
            spread_ticks=1,
            volume_pct=0.85
        )

        assert circuit_breakers.get_size_multiplier() == 0.5
        assert BreakerType.HIGH_VOLATILITY in circuit_breakers.state.active_breakers

    def test_wide_spread_triggers_pause(self, circuit_breakers):
        """Test wide spread triggers pause."""
        circuit_breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=3,  # > 2 tick threshold
            volume_pct=0.85
        )

        assert circuit_breakers.state.is_paused is True
        assert BreakerType.WIDE_SPREAD in circuit_breakers.state.active_breakers

    def test_low_volume_reduces_size(self, circuit_breakers):
        """Test low volume reduces position size."""
        circuit_breakers.update_market_conditions(
            current_atr=1.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.05  # < 10% threshold
        )

        assert circuit_breakers.get_size_multiplier() == 0.5
        assert BreakerType.LOW_VOLUME in circuit_breakers.state.active_breakers

    def test_conditions_clear(self, circuit_breakers):
        """Test conditions clear when market normalizes."""
        # First trigger high volatility
        circuit_breakers.update_market_conditions(
            current_atr=4.0,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85
        )
        assert circuit_breakers.get_size_multiplier() == 0.5

        # Then normalize
        circuit_breakers.update_market_conditions(
            current_atr=1.5,  # Normal range
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85
        )
        assert circuit_breakers.get_size_multiplier() == 1.0
        assert BreakerType.HIGH_VOLATILITY not in circuit_breakers.state.active_breakers

    def test_daily_reset(self, circuit_breakers):
        """Test daily reset clears breakers."""
        # Trigger some breakers
        circuit_breakers.trigger_daily_loss_stop(50.0, 50.0)

        # Reset for new day
        circuit_breakers.reset_daily()

        assert circuit_breakers.can_trade() is True
        assert circuit_breakers._consecutive_losses == 0
        assert BreakerType.DAILY_LOSS not in circuit_breakers.state.active_breakers

    def test_manual_reset(self, circuit_breakers):
        """Test manual reset clears all breakers."""
        # Trigger max drawdown (requires manual review)
        circuit_breakers.trigger_max_drawdown_halt(200.0, 200.0)

        # Manual reset
        result = circuit_breakers.manual_reset()

        assert result is True
        assert circuit_breakers.can_trade() is True
        assert circuit_breakers.state.is_halted is False
        assert circuit_breakers.state.requires_manual_review is False

    def test_status_report(self, circuit_breakers):
        """Test status report contains expected keys."""
        status = circuit_breakers.get_status()

        assert 'can_trade' in status
        assert 'is_paused' in status
        assert 'is_halted' in status
        assert 'size_multiplier' in status
        assert 'consecutive_losses' in status
        assert 'active_breakers' in status

    def test_check_market_conditions_function(self):
        """Test standalone market conditions check."""
        result = check_market_conditions(
            atr=1.5,
            baseline_atr=1.0,
            spread_ticks=1,
            volume_ratio=0.85
        )

        assert result['volatility_normal'] is True
        assert result['spread_acceptable'] is True
        assert result['volume_adequate'] is True
        assert result['tradeable'] is True

        # Test with bad conditions
        result = check_market_conditions(
            atr=5.0,  # 5x volatility
            baseline_atr=1.0,
            spread_ticks=3,  # Wide spread
            volume_ratio=0.05  # Low volume
        )

        assert result['tradeable'] is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestRiskModuleIntegration:
    """Integration tests for the risk module components working together."""

    def test_full_trade_flow(self):
        """Test complete trade flow through risk system."""
        # Initialize all components
        risk_manager = RiskManager(auto_persist=False)
        position_sizer = PositionSizer()
        stop_manager = StopLossManager()
        eod_manager = EODManager()
        circuit_breakers = CircuitBreakers()

        # Simulate 10:00 AM NY
        test_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=NY_TZ)

        # 1. Check if we can trade
        assert risk_manager.can_trade() is True
        assert circuit_breakers.can_trade() is True
        eod_status = eod_manager.get_status(test_time)
        assert eod_status.can_open_new_positions is True

        # 2. Calculate position size
        confidence = 0.75
        position_result = position_sizer.calculate(
            account_balance=risk_manager.state.account_balance,
            stop_ticks=8,
            confidence=confidence
        )
        # $1000 balance = tier 1 (max 1 contract per spec "$700-$1,000: 1 contract")
        # 75% confidence = 1.0x, $1000 * 2% = $20 risk budget / $10 per contract = 2 base
        # But capped at tier max of 1
        assert position_result.contracts == 1
        assert position_result.max_contracts_for_tier == 1

        # 3. Calculate stop price
        entry_price = 6050.00
        stop_result = stop_manager.calculate_fixed_stop(
            entry_price=entry_price,
            direction=1,
            stop_ticks=8
        )
        assert stop_result.stop_price == 6048.00

        # 4. Approve trade with risk manager
        assert risk_manager.approve_trade(
            risk_amount=stop_result.stop_dollars,
            confidence=confidence
        ) is True

        # 5. Simulate trade result (win)
        pnl = 15.0  # $15 profit
        risk_manager.record_trade_result(pnl=pnl)
        circuit_breakers.record_win()

        assert risk_manager.state.account_balance == 1015.0
        assert risk_manager.state.daily_pnl == 15.0

    def test_risk_limits_enforced_through_flow(self):
        """Test that risk limits are enforced at each step."""
        risk_manager = RiskManager(auto_persist=False)
        position_sizer = PositionSizer()

        # Simulate losses until daily limit
        for i in range(5):
            risk_manager.record_trade_result(pnl=-10.0)

        # Total loss = $50, hits daily limit
        assert risk_manager.state.status == TradingStatus.STOPPED_FOR_DAY
        assert risk_manager.can_trade() is False

        # Position sizer should still work (doesn't check daily limits)
        result = position_sizer.calculate(
            account_balance=950.0,  # Reduced balance
            stop_ticks=8,
            confidence=0.75
        )
        # But risk manager would reject the trade
        assert risk_manager.approve_trade(
            risk_amount=result.dollar_risk,
            confidence=0.75
        ) is False

    def test_eod_flatten_enforced(self):
        """Test EOD flatten is enforced."""
        eod_manager = EODManager()

        # At 4:27 PM, should_flatten is True
        test_time = datetime(2025, 1, 15, 16, 27, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(test_time)

        assert status.should_flatten is True
        assert status.can_open_new_positions is False

        # Any system integrating this should flatten positions immediately
        # when should_flatten is True


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_stop_distance(self, position_sizer):
        """Test handling of zero stop distance."""
        result = position_sizer.calculate(
            account_balance=1000.0,
            stop_ticks=0,
            confidence=0.75
        )

        assert result.contracts == 0

    def test_negative_pnl_tracking(self):
        """Test cumulative loss tracks only losses."""
        manager = RiskManager(auto_persist=False)

        # Win $20
        manager.record_trade_result(pnl=20.0)
        assert manager.state.cumulative_loss == 0.0

        # Lose $10
        manager.record_trade_result(pnl=-10.0)
        assert manager.state.cumulative_loss == 10.0

        # Win $30
        manager.record_trade_result(pnl=30.0)
        assert manager.state.cumulative_loss == 10.0  # Still 10, not reduced

    def test_pause_expiry(self):
        """Test pause automatically expires."""
        breakers = CircuitBreakers()

        # Trigger pause
        for _ in range(3):
            breakers.record_loss()

        assert breakers.can_trade() is False

        # Simulate time passing (mock the pause_until)
        breakers.state.pause_until = datetime.now() - timedelta(seconds=1)

        # Now should be able to trade
        assert breakers.can_trade() is True

    def test_boundary_confidence_values(self, position_sizer):
        """Test confidence at exact boundary values."""
        # Just below 60% - no trade
        result = position_sizer.calculate(1000.0, 8, 0.599)
        assert result.confidence_multiplier == 0.0

        # Exactly 60% - should be 0.5x (first tier above threshold)
        result = position_sizer.calculate(1000.0, 8, 0.60)
        assert result.confidence_multiplier == 0.5

        # Exactly 70%
        result = position_sizer.calculate(1000.0, 8, 0.70)
        assert result.confidence_multiplier == 1.0

        # Exactly 80%
        result = position_sizer.calculate(1000.0, 8, 0.80)
        assert result.confidence_multiplier == 1.5

        # Exactly 90%
        result = position_sizer.calculate(1000.0, 8, 0.90)
        assert result.confidence_multiplier == 2.0

    def test_boundary_balance_values(self, position_sizer):
        """Test balance at exact tier boundaries."""
        # Exactly $1000 - should be tier 1 (1 contract) per spec "$700-$1,000: 1 contract"
        # The boundary is inclusive, so $1000 is still in tier 1
        info = position_sizer.get_tier_info(1000.0)
        assert info['max_contracts'] == 1

        # Just below $1000
        info = position_sizer.get_tier_info(999.99)
        assert info['max_contracts'] == 1

        # Just above $1000 - should be tier 2 (2 contracts)
        info = position_sizer.get_tier_info(1000.01)
        assert info['max_contracts'] == 2

    def test_dst_transition(self, eod_manager):
        """Test EOD manager handles DST transitions."""
        # During DST (summer)
        summer_time = datetime(2025, 7, 15, 16, 0, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(summer_time)
        assert status.phase == EODPhase.REDUCED_SIZE

        # During standard time (winter)
        winter_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=NY_TZ)
        status = eod_manager.get_status(winter_time)
        assert status.phase == EODPhase.REDUCED_SIZE

        # Both should behave the same (NY time is NY time regardless of DST)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
