"""
Unit Tests for Partial Profit Taking (1.10).

Tests cover:
- PartialProfitLevel and PartialProfitConfig dataclasses
- StopLossManager.calculate_partial_profit_targets()
- StopLossManager.check_partial_profit()
- StopLossManager.get_breakeven_stop()
- Position multi-target fields
- PositionManager partial profit methods
- 2-level and 3-level partial profit scenarios

Per specs/risk-management.md:
- Position: 2 contracts
- TP1: Close 1 contract at 1:1 R:R, move stop to breakeven
- TP2: Close remaining at 1:2 R:R or trail
"""

import pytest
from datetime import datetime

from src.risk.stops import (
    StopLossManager,
    StopConfig,
    PartialProfitLevel,
    PartialProfitConfig,
    PartialProfitResult,
)
from src.trading.position_manager import (
    Position,
    PositionManager,
    Fill,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def stop_manager():
    """Stop loss manager with default config."""
    return StopLossManager()


@pytest.fixture
def two_level_config():
    """Standard 2-level partial profit config."""
    return PartialProfitConfig.default_two_level()


@pytest.fixture
def three_level_config():
    """Standard 3-level partial profit config."""
    return PartialProfitConfig.default_three_level()


@pytest.fixture
def position_manager():
    """Position manager for testing."""
    return PositionManager("CON.F.US.MES.H26")


# =============================================================================
# PartialProfitLevel Tests
# =============================================================================

class TestPartialProfitLevel:
    """Tests for PartialProfitLevel dataclass."""

    def test_create_level(self):
        """Create a partial profit level."""
        level = PartialProfitLevel(
            rr_ratio=1.0,
            percentage=0.5,
            move_stop_to_breakeven=True,
        )
        assert level.rr_ratio == 1.0
        assert level.percentage == 0.5
        assert level.move_stop_to_breakeven is True

    def test_default_move_stop(self):
        """Default move_stop_to_breakeven is False."""
        level = PartialProfitLevel(rr_ratio=2.0, percentage=0.5)
        assert level.move_stop_to_breakeven is False


# =============================================================================
# PartialProfitConfig Tests
# =============================================================================

class TestPartialProfitConfig:
    """Tests for PartialProfitConfig dataclass."""

    def test_default_two_level(self):
        """Default 2-level config."""
        config = PartialProfitConfig.default_two_level()
        assert len(config.levels) == 2
        assert config.enabled is True

        # TP1: 50% at 1:1, move stop to BE
        assert config.levels[0].rr_ratio == 1.0
        assert config.levels[0].percentage == 0.5
        assert config.levels[0].move_stop_to_breakeven is True

        # TP2: 50% at 2:1
        assert config.levels[1].rr_ratio == 2.0
        assert config.levels[1].percentage == 0.5
        assert config.levels[1].move_stop_to_breakeven is False

    def test_default_three_level(self):
        """Default 3-level config."""
        config = PartialProfitConfig.default_three_level()
        assert len(config.levels) == 3
        assert config.enabled is True

        # Verify percentages sum to ~1.0
        total_pct = sum(level.percentage for level in config.levels)
        assert 0.99 <= total_pct <= 1.01

    def test_custom_config(self):
        """Create custom partial profit config."""
        config = PartialProfitConfig(
            levels=[
                PartialProfitLevel(rr_ratio=0.5, percentage=0.25),
                PartialProfitLevel(rr_ratio=1.0, percentage=0.25, move_stop_to_breakeven=True),
                PartialProfitLevel(rr_ratio=1.5, percentage=0.25),
                PartialProfitLevel(rr_ratio=2.0, percentage=0.25),
            ]
        )
        assert len(config.levels) == 4

    def test_disabled_config(self):
        """Disabled partial profit config."""
        config = PartialProfitConfig(enabled=False)
        assert config.enabled is False
        assert len(config.levels) == 0


# =============================================================================
# StopLossManager Partial Profit Tests
# =============================================================================

class TestCalculatePartialProfitTargets:
    """Tests for calculate_partial_profit_targets method."""

    def test_two_level_long_position(self, stop_manager, two_level_config):
        """Calculate 2-level targets for long position."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,  # 8 ticks below entry
            direction=1,  # Long
            total_contracts=2,
            config=two_level_config,
        )

        assert isinstance(result, PartialProfitResult)
        assert len(result.target_prices) == 2
        assert len(result.target_quantities) == 2
        assert result.total_contracts == 2
        assert result.direction == 1

        # TP1 at 1:1 R:R (8 ticks above entry = 6002.0)
        assert result.target_prices[0] == pytest.approx(6002.0)
        assert result.target_quantities[0] == 1  # 50% of 2 = 1

        # TP2 at 2:1 R:R (16 ticks above entry = 6004.0)
        assert result.target_prices[1] == pytest.approx(6004.0)
        assert result.target_quantities[1] == 1  # Remaining

        # TP1 triggers breakeven stop
        assert 0 in result.move_stop_indices

    def test_two_level_short_position(self, stop_manager, two_level_config):
        """Calculate 2-level targets for short position."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=6002.0,  # 8 ticks above entry (stop for short)
            direction=-1,  # Short
            total_contracts=2,
            config=two_level_config,
        )

        assert len(result.target_prices) == 2
        assert result.direction == -1

        # TP1 at 1:1 R:R (8 ticks below entry = 5998.0)
        assert result.target_prices[0] == pytest.approx(5998.0)

        # TP2 at 2:1 R:R (16 ticks below entry = 5996.0)
        assert result.target_prices[1] == pytest.approx(5996.0)

    def test_three_level_long_position(self, stop_manager, three_level_config):
        """Calculate 3-level targets for long position."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,  # 8 ticks below entry
            direction=1,  # Long
            total_contracts=3,
            config=three_level_config,
        )

        assert len(result.target_prices) == 3
        assert len(result.target_quantities) == 3

        # Total quantities should equal total contracts
        assert sum(result.target_quantities) == 3

        # TP1 at 1:1 R:R
        assert result.target_prices[0] == pytest.approx(6002.0)

        # TP2 at 1.5:1 R:R (12 ticks = 6003.0)
        assert result.target_prices[1] == pytest.approx(6003.0)

        # TP3 at 2:1 R:R
        assert result.target_prices[2] == pytest.approx(6004.0)

    def test_single_contract_two_level(self, stop_manager, two_level_config):
        """Handle single contract with 2-level config."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,
            direction=1,
            total_contracts=1,
            config=two_level_config,
        )

        # With 1 contract, min 1 per level means one level gets all
        total_qty = sum(result.target_quantities)
        assert total_qty == 1

    def test_disabled_config_fallback(self, stop_manager):
        """Disabled config falls back to single target."""
        config = PartialProfitConfig(enabled=False)
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,
            direction=1,
            total_contracts=2,
            config=config,
        )

        # Falls back to single target at 2:1 R:R
        assert len(result.target_prices) == 1
        assert result.target_quantities[0] == 2  # All contracts

    def test_none_config_uses_default(self, stop_manager):
        """None config uses default 2-level."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,
            direction=1,
            total_contracts=2,
            config=None,
        )

        assert len(result.target_prices) == 2


class TestCheckPartialProfit:
    """Tests for check_partial_profit method."""

    def test_tp1_hit_long(self, stop_manager, two_level_config):
        """Check TP1 hit for long position."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,
            direction=1,
            total_contracts=2,
            config=two_level_config,
        )

        filled_levels = [False, False]

        # Price reaches TP1 (6002.0)
        level_hit, should_move_stop = stop_manager.check_partial_profit(
            current_price=6002.0,
            partial_profit_result=result,
            filled_levels=filled_levels,
        )

        assert level_hit == 0  # TP1
        assert should_move_stop is True  # TP1 triggers breakeven

    def test_tp2_hit_long(self, stop_manager, two_level_config):
        """Check TP2 hit for long position after TP1 filled."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,
            direction=1,
            total_contracts=2,
            config=two_level_config,
        )

        filled_levels = [True, False]  # TP1 already filled

        # Price reaches TP2 (6004.0)
        level_hit, should_move_stop = stop_manager.check_partial_profit(
            current_price=6004.0,
            partial_profit_result=result,
            filled_levels=filled_levels,
        )

        assert level_hit == 1  # TP2
        assert should_move_stop is False  # TP2 doesn't trigger breakeven

    def test_no_level_hit(self, stop_manager, two_level_config):
        """No level hit when price not reached."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=5998.0,
            direction=1,
            total_contracts=2,
            config=two_level_config,
        )

        filled_levels = [False, False]

        # Price below TP1
        level_hit, should_move_stop = stop_manager.check_partial_profit(
            current_price=6001.0,
            partial_profit_result=result,
            filled_levels=filled_levels,
        )

        assert level_hit is None
        assert should_move_stop is False

    def test_short_tp1_hit(self, stop_manager, two_level_config):
        """Check TP1 hit for short position."""
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=6000.0,
            stop_price=6002.0,  # Stop above for short
            direction=-1,
            total_contracts=2,
            config=two_level_config,
        )

        filled_levels = [False, False]

        # Price falls to TP1 (5998.0)
        level_hit, should_move_stop = stop_manager.check_partial_profit(
            current_price=5998.0,
            partial_profit_result=result,
            filled_levels=filled_levels,
        )

        assert level_hit == 0  # TP1
        assert should_move_stop is True


class TestGetBreakevenStop:
    """Tests for get_breakeven_stop method."""

    def test_breakeven_stop_long(self, stop_manager):
        """Calculate breakeven stop for long position."""
        stop_price = stop_manager.get_breakeven_stop(
            entry_price=6000.0,
            direction=1,
            buffer_ticks=1,
        )

        # Stop should be 1 tick below entry
        assert stop_price == pytest.approx(5999.75)

    def test_breakeven_stop_short(self, stop_manager):
        """Calculate breakeven stop for short position."""
        stop_price = stop_manager.get_breakeven_stop(
            entry_price=6000.0,
            direction=-1,
            buffer_ticks=1,
        )

        # Stop should be 1 tick above entry
        assert stop_price == pytest.approx(6000.25)

    def test_breakeven_stop_no_buffer(self, stop_manager):
        """Calculate breakeven stop with no buffer."""
        stop_price = stop_manager.get_breakeven_stop(
            entry_price=6000.0,
            direction=1,
            buffer_ticks=0,
        )

        assert stop_price == pytest.approx(6000.0)


# =============================================================================
# Position Multi-Target Tests
# =============================================================================

class TestPositionMultiTarget:
    """Tests for Position multi-target fields."""

    def test_position_has_partial_profit_levels(self):
        """Position correctly identifies multi-level targets."""
        pos = Position(
            contract_id="MES",
            direction=1,
            size=2,
            entry_price=6000.0,
            target_prices=[6002.0, 6004.0],
            target_quantities=[1, 1],
        )

        assert pos.has_partial_profit_levels is True

    def test_position_single_target(self):
        """Position with single target is not multi-level."""
        pos = Position(
            contract_id="MES",
            direction=1,
            size=2,
            entry_price=6000.0,
            target_prices=[6004.0],
            target_quantities=[2],
        )

        assert pos.has_partial_profit_levels is False

    def test_position_remaining_unfilled(self):
        """Count remaining unfilled levels."""
        pos = Position(
            contract_id="MES",
            target_prices=[6002.0, 6004.0],
            filled_target_levels=[True, False],
        )

        assert pos.remaining_unfilled_levels == 1

    def test_position_get_next_unfilled(self):
        """Get next unfilled level index."""
        pos = Position(
            contract_id="MES",
            target_prices=[6002.0, 6003.0, 6004.0],
            filled_target_levels=[True, False, False],
        )

        assert pos.get_next_unfilled_level() == 1

    def test_position_all_filled(self):
        """All levels filled returns None."""
        pos = Position(
            contract_id="MES",
            target_prices=[6002.0, 6004.0],
            filled_target_levels=[True, True],
        )

        assert pos.get_next_unfilled_level() is None
        assert pos.remaining_unfilled_levels == 0


# =============================================================================
# PositionManager Partial Profit Tests
# =============================================================================

class TestPositionManagerPartialProfit:
    """Tests for PositionManager partial profit methods."""

    def test_set_partial_profit_targets(self, position_manager):
        """Set partial profit targets."""
        # First open a position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)

        # Set partial profit targets
        position_manager.set_partial_profit_targets(
            target_prices=[6002.0, 6004.0],
            target_quantities=[1, 1],
            target_order_ids=["tp1", "tp2"],
            move_stop_to_breakeven_at=[0],
        )

        pos = position_manager.position
        assert pos.target_prices == [6002.0, 6004.0]
        assert pos.target_quantities == [1, 1]
        assert pos.target_order_ids == ["tp1", "tp2"]
        assert pos.filled_target_levels == [False, False]
        assert pos.move_stop_to_breakeven_at == [0]

        # Legacy single target should be first level
        assert pos.target_price == 6002.0

    def test_mark_target_level_filled(self, position_manager):
        """Mark a target level as filled."""
        # Open position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)

        # Set partial profit targets
        position_manager.set_partial_profit_targets(
            target_prices=[6002.0, 6004.0],
            target_quantities=[1, 1],
            target_order_ids=["tp1", "tp2"],
            move_stop_to_breakeven_at=[0],
        )

        # Fill TP1
        should_move_stop = position_manager.mark_target_level_filled(
            level_index=0,
            fill_price=6002.0,
        )

        assert should_move_stop is True  # Level 0 triggers breakeven

        pos = position_manager.position
        assert pos.filled_target_levels == [True, False]
        assert pos.size == 1  # Reduced by 1 contract
        assert pos.realized_pnl > 0  # Profit from partial close

    def test_mark_target_level_filled_tp2(self, position_manager):
        """Mark TP2 as filled (doesn't trigger breakeven)."""
        # Open position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=1,
            size=2,
            price=6000.0,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)

        # Set partial profit targets
        position_manager.set_partial_profit_targets(
            target_prices=[6002.0, 6004.0],
            target_quantities=[1, 1],
            target_order_ids=["tp1", "tp2"],
            move_stop_to_breakeven_at=[0],  # Only level 0
        )

        # Simulate TP1 already filled
        position_manager.mark_target_level_filled(0, 6002.0)

        # Now fill TP2
        should_move_stop = position_manager.mark_target_level_filled(
            level_index=1,
            fill_price=6004.0,
        )

        assert should_move_stop is False  # Level 1 doesn't trigger breakeven

        pos = position_manager.position
        assert pos.is_flat  # Position fully closed

    def test_get_unfilled_target_order_ids(self, position_manager):
        """Get order IDs for unfilled targets."""
        # Open position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=1,
            size=3,
            price=6000.0,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)

        # Set 3-level targets
        position_manager.set_partial_profit_targets(
            target_prices=[6002.0, 6003.0, 6004.0],
            target_quantities=[1, 1, 1],
            target_order_ids=["tp1", "tp2", "tp3"],
            move_stop_to_breakeven_at=[0],
        )

        # Fill TP1
        position_manager.mark_target_level_filled(0, 6002.0)

        # Get unfilled order IDs
        unfilled = position_manager.get_unfilled_target_order_ids()
        assert unfilled == ["tp2", "tp3"]


# =============================================================================
# Integration Scenario Tests
# =============================================================================

class TestPartialProfitScenarios:
    """Integration tests for complete partial profit scenarios."""

    def test_two_level_long_scenario(self, stop_manager, two_level_config, position_manager):
        """Complete 2-level partial profit scenario for long position."""
        # 1. Entry at 6000 with 8-tick stop at 5998
        entry_price = 6000.0
        stop_price = 5998.0
        size = 2

        # Open position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=1,
            size=size,
            price=entry_price,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)
        position_manager.set_stop_price(stop_price)

        # 2. Calculate partial profit targets
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=entry_price,
            stop_price=stop_price,
            direction=1,
            total_contracts=size,
            config=two_level_config,
        )

        # Set targets in position manager
        position_manager.set_partial_profit_targets(
            target_prices=result.target_prices,
            target_quantities=result.target_quantities,
            target_order_ids=["tp1", "tp2"],
            move_stop_to_breakeven_at=result.move_stop_indices,
        )

        # 3. Price reaches TP1 (6002.0) - check and fill
        filled_levels = [False, False]
        level_hit, should_move_stop = stop_manager.check_partial_profit(
            current_price=6002.0,
            partial_profit_result=result,
            filled_levels=filled_levels,
        )

        assert level_hit == 0
        assert should_move_stop is True

        # Fill TP1
        filled_levels[level_hit] = True
        should_be = position_manager.mark_target_level_filled(level_hit, 6002.0)
        assert should_be is True

        # 4. Position should be reduced to 1 contract
        pos = position_manager.position
        assert pos.size == 1

        # 5. Move stop to breakeven
        breakeven = stop_manager.get_breakeven_stop(entry_price, 1, buffer_ticks=1)
        position_manager.set_stop_price(breakeven)
        assert position_manager.position.stop_price == pytest.approx(5999.75)

        # 6. Price reaches TP2 (6004.0) - check and fill
        level_hit, should_move_stop = stop_manager.check_partial_profit(
            current_price=6004.0,
            partial_profit_result=result,
            filled_levels=filled_levels,
        )

        assert level_hit == 1
        assert should_move_stop is False

        # Fill TP2
        position_manager.mark_target_level_filled(level_hit, 6004.0)

        # 7. Position should be fully closed
        assert position_manager.position.is_flat
        assert position_manager.position.realized_pnl > 0

    def test_three_level_short_scenario(self, stop_manager, three_level_config, position_manager):
        """Complete 3-level partial profit scenario for short position."""
        # 1. Entry at 6000 with 8-tick stop at 6002
        entry_price = 6000.0
        stop_price = 6002.0
        size = 3

        # Open short position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL
            size=size,
            price=entry_price,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)
        position_manager.set_stop_price(stop_price)

        # 2. Calculate partial profit targets
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=entry_price,
            stop_price=stop_price,
            direction=-1,  # Short
            total_contracts=size,
            config=three_level_config,
        )

        assert len(result.target_prices) == 3
        # Short targets should be below entry
        assert all(tp < entry_price for tp in result.target_prices)

        # Set targets
        position_manager.set_partial_profit_targets(
            target_prices=result.target_prices,
            target_quantities=result.target_quantities,
            target_order_ids=["tp1", "tp2", "tp3"],
            move_stop_to_breakeven_at=result.move_stop_indices,
        )

        # 3. Fill all levels sequentially
        filled_levels = [False, False, False]

        for i in range(3):
            level_hit, should_move_stop = stop_manager.check_partial_profit(
                current_price=result.target_prices[i],
                partial_profit_result=result,
                filled_levels=filled_levels,
            )

            assert level_hit == i
            filled_levels[level_hit] = True
            position_manager.mark_target_level_filled(level_hit, result.target_prices[i])

        # 4. Position should be fully closed
        assert position_manager.position.is_flat

    def test_stop_hit_scenario(self, stop_manager, two_level_config, position_manager):
        """Scenario where stop is hit before all targets."""
        # 1. Entry at 6000 with 8-tick stop at 5998
        entry_price = 6000.0
        stop_price = 5998.0
        size = 2

        # Open position
        fill = Fill(
            order_id="entry1",
            contract_id="CON.F.US.MES.H26",
            side=1,
            size=size,
            price=entry_price,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(fill)

        # 2. Calculate partial profit targets
        result = stop_manager.calculate_partial_profit_targets(
            entry_price=entry_price,
            stop_price=stop_price,
            direction=1,
            total_contracts=size,
            config=two_level_config,
        )

        position_manager.set_partial_profit_targets(
            target_prices=result.target_prices,
            target_quantities=result.target_quantities,
            target_order_ids=["tp1", "tp2"],
            move_stop_to_breakeven_at=result.move_stop_indices,
        )

        # 3. TP1 fills
        position_manager.mark_target_level_filled(0, 6002.0)
        assert position_manager.position.size == 1

        # 4. Stop is hit for remaining position
        stop_fill = Fill(
            order_id="stop1",
            contract_id="CON.F.US.MES.H26",
            side=2,  # SELL (closing long)
            size=1,
            price=5998.0,
            timestamp=datetime.now(),
        )
        position_manager.update_from_fill(stop_fill)

        # 5. Position should be flat with mixed P&L
        # (profit from TP1, loss from stop)
        assert position_manager.position.is_flat
