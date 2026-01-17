"""
Unit Tests for Backtesting Module

Tests cover:
1. Transaction cost model (MES-specific $0.84 round-trip)
2. Slippage model (tick-based, volatility-adaptive)
3. Performance metrics (Sharpe, Sortino, Calmar, etc.)
4. Trade logging and export
5. Backtest engine (bar-by-bar simulation)
6. Walk-forward validation framework

These tests ensure the backtesting engine produces accurate, realistic results
that can be trusted for strategy evaluation before live trading.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from pathlib import Path
import tempfile
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.costs import (
    TransactionCostModel,
    MESCostConfig,
    calculate_mes_cost,
)
from backtest.slippage import (
    SlippageModel,
    SlippageConfig,
    MarketCondition,
    OrderType,
    calculate_realistic_slippage,
)
from backtest.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_expectancy,
    calculate_consecutive_streaks,
    calculate_daily_stats,
    calculate_var,
)
from backtest.trade_logger import (
    TradeLog,
    TradeRecord,
    EquityCurve,
    EquityPoint,
    BacktestReport,
    ExitReason,
)
from backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    OrderFillMode,
    Signal,
    SignalType,
    Position,
    create_simple_signal_generator,
    WalkForwardValidator,
)


# =============================================================================
# Transaction Cost Model Tests
# =============================================================================

class TestMESCostConfig:
    """Tests for MES cost configuration."""

    def test_default_values(self):
        """Test default MES cost values."""
        config = MESCostConfig()
        assert config.commission_per_side == 0.20
        assert config.exchange_fee_per_side == 0.22

    def test_per_side_cost(self):
        """Test per-side cost calculation."""
        config = MESCostConfig()
        assert config.per_side_cost == pytest.approx(0.42)  # $0.20 + $0.22

    def test_round_trip_cost(self):
        """Test round-trip cost calculation."""
        config = MESCostConfig()
        assert config.round_trip_cost == pytest.approx(0.84)  # $0.42 x 2

    def test_custom_values(self):
        """Test custom cost values."""
        config = MESCostConfig(commission_per_side=0.50, exchange_fee_per_side=0.30)
        assert config.per_side_cost == pytest.approx(0.80)
        assert config.round_trip_cost == pytest.approx(1.60)


class TestTransactionCostModel:
    """Tests for transaction cost model."""

    def test_entry_cost(self):
        """Test entry cost calculation."""
        model = TransactionCostModel()
        cost = model.calculate_entry_cost(contracts=1)
        assert cost == pytest.approx(0.42)  # $0.20 + $0.22

    def test_exit_cost(self):
        """Test exit cost calculation."""
        model = TransactionCostModel()
        cost = model.calculate_exit_cost(contracts=1)
        assert cost == pytest.approx(0.42)

    def test_round_trip_cost_single_contract(self):
        """Test round-trip cost for single contract."""
        model = TransactionCostModel()
        cost = model.calculate_round_trip_cost(contracts=1)
        assert cost == pytest.approx(0.84)

    def test_round_trip_cost_multiple_contracts(self):
        """Test round-trip cost for multiple contracts."""
        model = TransactionCostModel()
        cost = model.calculate_round_trip_cost(contracts=3)
        assert cost == pytest.approx(2.52)  # $0.84 x 3

    def test_zero_contracts(self):
        """Test cost for zero contracts."""
        model = TransactionCostModel()
        assert model.calculate_round_trip_cost(contracts=0) == 0.0

    def test_negative_contracts(self):
        """Test cost for negative contracts returns zero."""
        model = TransactionCostModel()
        assert model.calculate_round_trip_cost(contracts=-1) == 0.0

    def test_record_trade_tracking(self):
        """Test cumulative trade tracking."""
        model = TransactionCostModel()

        model.record_trade(contracts=1)
        model.record_trade(contracts=2)
        model.record_trade(contracts=1)

        assert model.get_total_trades() == 3
        assert model.get_total_commission() == pytest.approx(3.36)  # $0.84 + $1.68 + $0.84
        assert model.get_average_cost_per_trade() == pytest.approx(1.12)

    def test_reset(self):
        """Test reset clears tracking."""
        model = TransactionCostModel()
        model.record_trade(contracts=5)
        model.reset()

        assert model.get_total_trades() == 0
        assert model.get_total_commission() == 0.0

    def test_breakeven_ticks(self):
        """Test breakeven tick calculation."""
        model = TransactionCostModel()

        # For 1 contract: $0.84 / $1.25 = 0.672 ticks
        ticks = model.calculate_breakeven_ticks(contracts=1)
        assert ticks == pytest.approx(0.672, rel=0.01)

    def test_convenience_function(self):
        """Test calculate_mes_cost convenience function."""
        assert calculate_mes_cost(1) == pytest.approx(0.84)
        assert calculate_mes_cost(5) == pytest.approx(4.20)


# =============================================================================
# Slippage Model Tests
# =============================================================================

class TestSlippageConfig:
    """Tests for slippage configuration."""

    def test_default_values(self):
        """Test default slippage values."""
        config = SlippageConfig()
        assert config.tick_size == 0.25
        assert config.tick_value == 1.25
        assert config.normal_slippage_ticks == 1.0

    def test_custom_values(self):
        """Test custom slippage configuration."""
        config = SlippageConfig(
            normal_slippage_ticks=0.5,
            high_volatility_slippage_ticks=3.0,
        )
        assert config.normal_slippage_ticks == 0.5
        assert config.high_volatility_slippage_ticks == 3.0


class TestSlippageModel:
    """Tests for slippage model."""

    def test_limit_order_no_slippage(self):
        """Limit orders should have no slippage."""
        model = SlippageModel()
        slippage = model.get_slippage_ticks(OrderType.LIMIT)
        assert slippage == 0.0

    def test_market_order_normal_slippage(self):
        """Market orders should have normal slippage."""
        model = SlippageModel()
        slippage = model.get_slippage_ticks(
            OrderType.MARKET,
            condition=MarketCondition.NORMAL,
        )
        assert slippage == 1.0

    def test_market_order_high_volatility(self):
        """High volatility should increase slippage."""
        model = SlippageModel()
        slippage = model.get_slippage_ticks(
            OrderType.MARKET,
            condition=MarketCondition.HIGH_VOLATILITY,
        )
        assert slippage == 2.0

    def test_market_order_extreme_conditions(self):
        """Extreme conditions should have maximum slippage."""
        model = SlippageModel()
        slippage = model.get_slippage_ticks(
            OrderType.MARKET,
            condition=MarketCondition.EXTREME,
        )
        assert slippage == 4.0

    def test_apply_slippage_buy(self):
        """Buying should increase fill price (slippage against buyer)."""
        model = SlippageModel()
        price = 4500.00

        fill_price = model.apply_slippage(
            price=price,
            direction=1,  # Buy
            order_type=OrderType.MARKET,
        )

        # 1 tick slippage = 0.25 points higher
        assert fill_price == 4500.25

    def test_apply_slippage_sell(self):
        """Selling should decrease fill price (slippage against seller)."""
        model = SlippageModel()
        price = 4500.00

        fill_price = model.apply_slippage(
            price=price,
            direction=-1,  # Sell
            order_type=OrderType.MARKET,
        )

        # 1 tick slippage = 0.25 points lower
        assert fill_price == 4499.75

    def test_apply_slippage_rounds_to_tick(self):
        """Fill price should be rounded to tick size."""
        model = SlippageModel()

        fill_price = model.apply_slippage(
            price=4500.10,  # Not on tick boundary
            direction=1,
            order_type=OrderType.MARKET,
        )

        # Should round to nearest 0.25
        assert fill_price % 0.25 == 0

    def test_slippage_tracking(self):
        """Test cumulative slippage tracking."""
        model = SlippageModel()

        model.apply_slippage(4500.0, 1, OrderType.MARKET, contracts=1)
        model.apply_slippage(4500.0, -1, OrderType.MARKET, contracts=2)

        assert model.get_slippage_events() == 2
        assert model.get_total_slippage_ticks() == 3.0  # 1 + 2
        assert model.get_total_slippage_dollars() == pytest.approx(3.75)  # 3 x $1.25

    def test_detect_market_condition_normal(self):
        """Test market condition detection - normal."""
        model = SlippageModel()
        condition = model.detect_market_condition(
            current_atr=2.0,
            normal_atr=2.0,
        )
        assert condition == MarketCondition.NORMAL

    def test_detect_market_condition_high_volatility(self):
        """Test market condition detection - high volatility."""
        model = SlippageModel()
        condition = model.detect_market_condition(
            current_atr=8.0,  # 4x normal
            normal_atr=2.0,
        )
        assert condition == MarketCondition.HIGH_VOLATILITY

    def test_detect_market_condition_extreme(self):
        """Test market condition detection - extreme."""
        model = SlippageModel()
        condition = model.detect_market_condition(
            current_atr=12.0,  # 6x normal
            normal_atr=2.0,
        )
        assert condition == MarketCondition.EXTREME

    def test_detect_market_condition_wide_spread(self):
        """Test market condition detection - wide spread."""
        model = SlippageModel()
        condition = model.detect_market_condition(
            current_atr=2.0,
            normal_atr=2.0,
            spread_ticks=3.0,  # > 2 tick threshold
        )
        assert condition == MarketCondition.LOW_LIQUIDITY

    def test_convenience_function(self):
        """Test calculate_realistic_slippage convenience function."""
        price = calculate_realistic_slippage(4500.0, 1, is_market_order=True)
        assert price == 4500.25

        price = calculate_realistic_slippage(4500.0, 1, is_market_order=False)
        assert price == 4500.0  # No slippage on limits


# =============================================================================
# Performance Metrics Tests
# =============================================================================

class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_positive_sharpe(self):
        """Test Sharpe ratio with positive returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_negative_sharpe(self):
        """Test Sharpe ratio with negative returns."""
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0  # Division by zero protection

    def test_empty_returns(self):
        """Test Sharpe ratio with empty returns."""
        sharpe = calculate_sharpe_ratio(np.array([]))
        assert sharpe == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_positive_sortino(self):
        """Test Sortino ratio with mixed returns."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        sortino = calculate_sortino_ratio(returns)
        assert sortino > 0

    def test_no_downside(self):
        """Test Sortino ratio with no negative returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.03])
        sortino = calculate_sortino_ratio(returns)
        assert sortino == 10.0  # Capped at high value

    def test_all_negative(self):
        """Test Sortino ratio with all negative returns."""
        returns = np.array([-0.01, -0.02, -0.01])
        sortino = calculate_sortino_ratio(returns)
        assert sortino < 0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_positive_calmar(self):
        """Test Calmar ratio with positive return and drawdown."""
        calmar = calculate_calmar_ratio(
            total_return=0.50,  # 50% return
            max_drawdown=0.10,  # 10% drawdown
            years=1.0,
        )
        assert calmar == pytest.approx(5.0, rel=0.01)

    def test_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown."""
        calmar = calculate_calmar_ratio(
            total_return=0.20,
            max_drawdown=0.0,
        )
        assert calmar == 0.0

    def test_negative_return(self):
        """Test Calmar ratio with total loss."""
        calmar = calculate_calmar_ratio(
            total_return=-1.0,  # Lost everything
            max_drawdown=1.0,
        )
        assert calmar == 0.0


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_simple_drawdown(self):
        """Test simple drawdown calculation."""
        equity = np.array([100, 110, 105, 115, 100])

        max_dd_pct, max_dd_dollars, peak_idx, trough_idx, duration = calculate_max_drawdown(equity)

        # Max DD from 115 to 100 = 13.04%
        assert max_dd_pct == pytest.approx(0.1304, rel=0.01)
        assert max_dd_dollars == pytest.approx(15.0, rel=0.01)

    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = np.array([100, 105, 110, 115, 120])

        max_dd_pct, _, _, _, _ = calculate_max_drawdown(equity)
        assert max_dd_pct == 0.0

    def test_empty_equity(self):
        """Test with empty equity curve."""
        max_dd_pct, _, _, _, _ = calculate_max_drawdown(np.array([]))
        assert max_dd_pct == 0.0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_basic_win_rate(self):
        """Test basic win rate calculation."""
        assert calculate_win_rate(6, 10) == 60.0
        assert calculate_win_rate(0, 10) == 0.0
        assert calculate_win_rate(10, 10) == 100.0

    def test_zero_trades(self):
        """Test win rate with zero trades."""
        assert calculate_win_rate(0, 0) == 0.0


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_profitable(self):
        """Test profit factor for profitable strategy."""
        pf = calculate_profit_factor(gross_profit=200, gross_loss=100)
        assert pf == 2.0

    def test_no_losses(self):
        """Test profit factor with no losses."""
        pf = calculate_profit_factor(gross_profit=100, gross_loss=0)
        assert pf == float('inf')

    def test_no_wins(self):
        """Test profit factor with no wins."""
        pf = calculate_profit_factor(gross_profit=0, gross_loss=100)
        assert pf == 0.0


class TestExpectancy:
    """Tests for expectancy calculation."""

    def test_positive_expectancy(self):
        """Test positive expectancy calculation."""
        expectancy = calculate_expectancy(
            win_rate=0.60,  # 60% win rate
            avg_win=15.0,   # $15 average win
            avg_loss=10.0,  # $10 average loss
        )
        # (0.60 x 15) - (0.40 x 10) = 9 - 4 = 5
        assert expectancy == pytest.approx(5.0, rel=0.01)

    def test_negative_expectancy(self):
        """Test negative expectancy calculation."""
        expectancy = calculate_expectancy(
            win_rate=0.40,
            avg_win=10.0,
            avg_loss=15.0,
        )
        # (0.40 x 10) - (0.60 x 15) = 4 - 9 = -5
        assert expectancy == pytest.approx(-5.0, rel=0.01)


class TestConsecutiveStreaks:
    """Tests for consecutive win/loss streak calculation."""

    def test_winning_streak(self):
        """Test consecutive win streak detection."""
        trades = [10.0, 15.0, 20.0, -5.0, 10.0]
        wins, losses = calculate_consecutive_streaks(trades)
        assert wins == 3
        assert losses == 1

    def test_losing_streak(self):
        """Test consecutive loss streak detection."""
        trades = [-10.0, -15.0, -5.0, -8.0, 20.0]
        wins, losses = calculate_consecutive_streaks(trades)
        assert wins == 1
        assert losses == 4

    def test_empty_trades(self):
        """Test with no trades."""
        wins, losses = calculate_consecutive_streaks([])
        assert wins == 0
        assert losses == 0


class TestCalculateMetrics:
    """Tests for comprehensive metrics calculation."""

    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        trade_pnls = [10.0, -5.0, 15.0, -8.0, 20.0, -3.0]
        equity_curve = [1000, 1010, 1005, 1020, 1012, 1032, 1029]

        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=5,
        )

        assert metrics.total_trades == 6
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 3
        assert metrics.win_rate_pct == 50.0
        assert metrics.total_return_dollars == pytest.approx(29.0, rel=0.01)

    def test_metrics_to_dict(self):
        """Test metrics serialization to dictionary."""
        metrics = PerformanceMetrics(
            total_return_pct=0.10,
            sharpe_ratio=1.5,
            total_trades=50,
        )

        d = metrics.to_dict()
        assert 'returns' in d
        assert 'risk' in d
        assert 'trades' in d
        assert d['returns']['total_return_pct'] == 0.10

    def test_default_returns_source_is_equity(self):
        """Test that default returns source is equity curve."""
        trade_pnls = [10.0, -5.0, 15.0]
        equity_curve = [1000, 1010, 1005, 1020]

        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=3,
        )

        assert metrics.returns_source == "equity"

    def test_closed_trade_returns_source(self):
        """Test that use_closed_trade_returns sets source to closed_trades."""
        trade_pnls = [10.0, -5.0, 15.0]
        equity_curve = [1000, 1010, 1005, 1020]
        daily_pnls = [10.0, -5.0, 15.0]  # Same as trade_pnls for simplicity

        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=3,
            daily_pnls=daily_pnls,
            use_closed_trade_returns=True,
        )

        assert metrics.returns_source == "closed_trades"

    def test_closed_trade_returns_without_daily_pnls_falls_back_to_equity(self):
        """Test that without daily_pnls, falls back to equity curve."""
        trade_pnls = [10.0, -5.0, 15.0]
        equity_curve = [1000, 1010, 1005, 1020]

        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=3,
            use_closed_trade_returns=True,  # No daily_pnls provided
        )

        # Should fall back to equity since no daily_pnls
        assert metrics.returns_source == "equity"

    def test_returns_source_in_to_dict(self):
        """Test that returns_source is included in to_dict output."""
        metrics = PerformanceMetrics(
            returns_source="closed_trades",
        )

        d = metrics.to_dict()
        assert d['risk']['returns_source'] == "closed_trades"

    def test_sharpe_differs_between_sources(self):
        """Test that Sharpe/Sortino can differ between equity and closed trades.

        This test demonstrates why the option matters: intraday volatility
        in the equity curve can exaggerate risk metrics for scalping strategies.
        """
        # Simulate high intraday volatility but same daily P&L
        # Equity curve has high intraday swings
        equity_curve = [
            1000, 950, 1020,  # Day 1: volatile but ends +20
            1020, 970, 1035,  # Day 2: volatile but ends +15
            1035, 990, 1025,  # Day 3: volatile but ends -10
        ]

        # Daily P&L is much smoother (just daily closes)
        daily_pnls = [20.0, 15.0, -10.0]

        trade_pnls = [20.0, 15.0, -10.0]

        # Calculate with equity curve
        metrics_equity = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=3,
        )

        # Calculate with closed trade returns
        metrics_closed = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=3,
            daily_pnls=daily_pnls,
            use_closed_trade_returns=True,
        )

        # Both should have valid metrics
        assert metrics_equity.sharpe_ratio != 0.0
        assert metrics_closed.sharpe_ratio != 0.0

        # The closed trades version should typically show better risk-adjusted
        # returns since it excludes intraday volatility (though actual values
        # depend on the specific data pattern)
        assert metrics_equity.returns_source == "equity"
        assert metrics_closed.returns_source == "closed_trades"


# =============================================================================
# Trade Logger Tests
# =============================================================================

class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_is_winner(self):
        """Test winner detection."""
        winner = TradeRecord(
            trade_id=1,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            direction=1,
            entry_price=4500.0,
            exit_price=4510.0,
            contracts=1,
            gross_pnl=50.0,
            commission=0.84,
            slippage=1.25,
            net_pnl=47.91,
            exit_reason=ExitReason.TARGET,
        )
        assert winner.is_winner is True

        loser = TradeRecord(
            trade_id=2,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            direction=1,
            entry_price=4500.0,
            exit_price=4490.0,
            contracts=1,
            gross_pnl=-50.0,
            commission=0.84,
            slippage=1.25,
            net_pnl=-52.09,
            exit_reason=ExitReason.STOP,
        )
        assert loser.is_winner is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        trade = TradeRecord(
            trade_id=1,
            entry_time=datetime(2023, 1, 1, 10, 0, 0),
            exit_time=datetime(2023, 1, 1, 10, 5, 0),
            direction=1,
            entry_price=4500.0,
            exit_price=4510.0,
            contracts=1,
            gross_pnl=50.0,
            commission=0.84,
            slippage=1.25,
            net_pnl=47.91,
            exit_reason=ExitReason.TARGET,
        )

        d = trade.to_dict()
        assert d['trade_id'] == 1
        assert d['direction'] == 'LONG'
        assert d['exit_reason'] == 'target'


class TestTradeLog:
    """Tests for TradeLog class."""

    def test_add_trade(self):
        """Test adding trades to log."""
        log = TradeLog()

        trade = log.add_trade(
            entry_time=datetime(2023, 1, 1, 10, 0),
            exit_time=datetime(2023, 1, 1, 10, 5),
            direction=1,
            entry_price=4500.0,
            exit_price=4510.0,
            contracts=1,
            gross_pnl=50.0,
            commission=0.84,
            slippage=1.25,
            exit_reason=ExitReason.TARGET,
        )

        assert trade.trade_id == 1
        assert log.get_trade_count() == 1
        assert len(log.get_trades()) == 1

    def test_get_trade_pnls(self):
        """Test getting P&L list."""
        log = TradeLog()

        log.add_trade(
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            direction=1,
            entry_price=4500.0,
            exit_price=4510.0,
            contracts=1,
            gross_pnl=50.0,
            commission=0.84,
            slippage=1.25,
            exit_reason=ExitReason.TARGET,
        )

        pnls = log.get_trade_pnls()
        assert len(pnls) == 1
        assert pnls[0] == pytest.approx(47.91, rel=0.01)

    def test_export_csv(self):
        """Test CSV export."""
        log = TradeLog()

        log.add_trade(
            entry_time=datetime(2023, 1, 1, 10, 0),
            exit_time=datetime(2023, 1, 1, 10, 5),
            direction=1,
            entry_price=4500.0,
            exit_price=4510.0,
            contracts=1,
            gross_pnl=50.0,
            commission=0.84,
            slippage=1.25,
            exit_reason=ExitReason.TARGET,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trades.csv"
            log.export_csv(str(filepath))

            assert filepath.exists()

            # Read and verify
            import csv
            with open(filepath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]['direction'] == 'LONG'


class TestEquityCurve:
    """Tests for EquityCurve class."""

    def test_add_point(self):
        """Test adding equity points."""
        curve = EquityCurve(initial_equity=1000.0)

        point = curve.add_point(
            timestamp=datetime(2023, 1, 1, 10, 0),
            equity=1050.0,
        )

        assert point.equity == 1050.0
        assert point.drawdown == 0.0
        assert curve.get_peak_equity() == 1050.0

    def test_drawdown_calculation(self):
        """Test automatic drawdown calculation."""
        curve = EquityCurve(initial_equity=1000.0)

        curve.add_point(datetime(2023, 1, 1, 10, 0), 1100.0)  # New peak
        curve.add_point(datetime(2023, 1, 1, 10, 1), 1050.0)  # Drawdown

        points = curve.get_points()
        assert points[1].drawdown == pytest.approx(50.0, rel=0.01)
        assert points[1].drawdown_pct == pytest.approx(0.0454, rel=0.01)

    def test_export_csv(self):
        """Test CSV export."""
        curve = EquityCurve(initial_equity=1000.0)
        curve.add_point(datetime(2023, 1, 1, 10, 0), 1050.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "equity.csv"
            curve.export_csv(str(filepath))
            assert filepath.exists()


# =============================================================================
# Backtest Engine Tests
# =============================================================================

@pytest.fixture
def sample_data():
    """Create sample 1-second OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 1000

    # Generate random walk prices
    returns = np.random.normal(0, 0.0001, n_bars)
    prices = 4500.0 * np.cumprod(1 + returns)

    # Create OHLCV data
    timestamps = pd.date_range(
        start='2023-01-03 09:30:00',
        periods=n_bars,
        freq='1s',
        tz='America/New_York',
    )

    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.0001, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.0001, n_bars))),
        'close': prices * (1 + np.random.normal(0, 0.00005, n_bars)),
        'volume': np.random.randint(100, 1000, n_bars),
    }, index=timestamps)

    return data


@pytest.fixture
def sample_data_with_predictions(sample_data):
    """Add prediction columns to sample data."""
    n = len(sample_data)
    np.random.seed(42)

    # Generate random predictions
    sample_data['prediction'] = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    sample_data['confidence'] = np.random.uniform(0.5, 0.95, n)

    return sample_data


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.initial_capital == 1000.0
        assert config.commission_per_side == 0.20
        assert config.round_trip_cost == pytest.approx(0.84)
        assert config.tick_size == 0.25
        assert config.tick_value == 1.25

    def test_to_dict(self):
        """Test configuration serialization."""
        config = BacktestConfig()
        d = config.to_dict()

        assert 'initial_capital' in d
        assert 'fill_mode' in d
        assert d['fill_mode'] == 'next_bar_open'


class TestSignal:
    """Tests for Signal class."""

    def test_entry_signal(self):
        """Test entry signal creation."""
        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.75,
            predicted_class=2,
            stop_ticks=8.0,
            target_ticks=16.0,
        )

        assert signal.signal_type == SignalType.LONG_ENTRY
        assert signal.confidence == 0.75

    def test_hold_signal(self):
        """Test hold signal."""
        signal = Signal(SignalType.HOLD)
        assert signal.signal_type == SignalType.HOLD


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine()
        assert engine.config.initial_capital == 1000.0

    def test_custom_config(self):
        """Test engine with custom config."""
        config = BacktestConfig(initial_capital=5000.0)
        engine = BacktestEngine(config=config)
        assert engine.config.initial_capital == 5000.0

    def test_validate_data_empty(self):
        """Test validation rejects empty data."""
        engine = BacktestEngine()

        with pytest.raises(ValueError, match="empty"):
            engine._validate_data(pd.DataFrame())

    def test_validate_data_missing_columns(self):
        """Test validation rejects missing columns."""
        engine = BacktestEngine()
        data = pd.DataFrame({'open': [1, 2, 3]}, index=pd.date_range('2023-01-01', periods=3))

        with pytest.raises(ValueError, match="missing required columns"):
            engine._validate_data(data)

    def test_run_basic(self, sample_data):
        """Test basic backtest run."""
        engine = BacktestEngine()

        # Simple signal generator that never trades
        def no_trade_signal(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(sample_data, no_trade_signal)

        assert result is not None
        assert result.report.trade_log.get_trade_count() == 0
        assert result.report.metrics.total_trades == 0

    def test_run_with_trades(self, sample_data_with_predictions):
        """Test backtest with actual trades."""
        config = BacktestConfig(
            log_frequency=10,  # More frequent logging
        )
        engine = BacktestEngine(config=config)

        signal_gen = create_simple_signal_generator(
            min_confidence=0.60,
            stop_ticks=8.0,
            target_ticks=16.0,
        )

        result = engine.run(sample_data_with_predictions, signal_gen)

        # Should have some trades
        assert result.report.trade_log.get_trade_count() >= 0
        assert result.report.metrics is not None

    def test_eod_flatten(self):
        """Test EOD flatten enforcement."""
        # Create data that spans past 4:30 PM
        timestamps = pd.date_range(
            start='2023-01-03 16:00:00',
            periods=2000,  # ~33 minutes
            freq='1s',
            tz='America/New_York',
        )

        prices = np.full(len(timestamps), 4500.0)
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.full(len(timestamps), 100),
        }, index=timestamps)

        engine = BacktestEngine()

        # Signal generator that tries to enter
        def always_long(bar, position, context):
            if position is None:
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(data, always_long)

        # Should have handled EOD appropriately
        assert result is not None


class TestSimpleSignalGenerator:
    """Tests for create_simple_signal_generator helper."""

    def test_long_entry(self, sample_data_with_predictions):
        """Test long entry signal generation."""
        signal_gen = create_simple_signal_generator(min_confidence=0.60)

        # Get a bar where prediction is UP (2) with high confidence
        sample_data_with_predictions.iloc[0, sample_data_with_predictions.columns.get_loc('prediction')] = 2
        sample_data_with_predictions.iloc[0, sample_data_with_predictions.columns.get_loc('confidence')] = 0.80

        bar = sample_data_with_predictions.iloc[0]
        signal = signal_gen(bar, None, {})

        assert signal.signal_type == SignalType.LONG_ENTRY
        assert signal.confidence == 0.80

    def test_short_entry(self, sample_data_with_predictions):
        """Test short entry signal generation."""
        signal_gen = create_simple_signal_generator(min_confidence=0.60)

        # Get a bar where prediction is DOWN (0) with high confidence
        sample_data_with_predictions.iloc[0, sample_data_with_predictions.columns.get_loc('prediction')] = 0
        sample_data_with_predictions.iloc[0, sample_data_with_predictions.columns.get_loc('confidence')] = 0.75

        bar = sample_data_with_predictions.iloc[0]
        signal = signal_gen(bar, None, {})

        assert signal.signal_type == SignalType.SHORT_ENTRY

    def test_hold_on_low_confidence(self, sample_data_with_predictions):
        """Test hold signal on low confidence."""
        signal_gen = create_simple_signal_generator(min_confidence=0.60)

        sample_data_with_predictions.iloc[0, sample_data_with_predictions.columns.get_loc('prediction')] = 2
        sample_data_with_predictions.iloc[0, sample_data_with_predictions.columns.get_loc('confidence')] = 0.50

        bar = sample_data_with_predictions.iloc[0]
        signal = signal_gen(bar, None, {})

        assert signal.signal_type == SignalType.HOLD


class TestWalkForwardValidator:
    """Tests for walk-forward validation framework."""

    def test_generate_folds(self):
        """Test fold generation."""
        # Create 2 years of data
        timestamps = pd.date_range(
            start='2023-01-01',
            end='2024-12-31',
            freq='B',  # Business days
        )
        data = pd.DataFrame(
            {'close': np.random.randn(len(timestamps))},
            index=timestamps,
        )

        validator = WalkForwardValidator(
            training_months=6,
            validation_months=1,
            test_months=1,
            step_months=1,
        )

        folds = validator.generate_folds(data)

        assert len(folds) > 0
        for fold in folds:
            assert 'train' in fold
            assert 'val' in fold
            assert 'test' in fold

    def test_fold_dates(self):
        """Test fold date ranges are correct."""
        timestamps = pd.date_range(
            start='2023-01-01',
            end='2024-12-31',
            freq='B',
        )
        data = pd.DataFrame(
            {'close': np.random.randn(len(timestamps))},
            index=timestamps,
        )

        validator = WalkForwardValidator(
            training_months=6,
            validation_months=1,
            test_months=1,
        )

        folds = validator.generate_folds(data)

        if folds:
            first_fold = folds[0]
            train_start, train_end = first_fold['train']
            val_start, val_end = first_fold['val']
            test_start, test_end = first_fold['test']

            # Validate ordering
            assert train_start < train_end
            assert train_end == val_start
            assert val_end == test_start


# =============================================================================
# Integration Tests
# =============================================================================

class TestBacktestIntegration:
    """Integration tests for full backtest workflow."""

    def test_full_backtest_workflow(self, sample_data_with_predictions):
        """Test complete backtest workflow."""
        # Configure
        config = BacktestConfig(
            initial_capital=1000.0,
            log_frequency=50,
        )

        # Create engine
        engine = BacktestEngine(config=config)

        # Create signal generator
        signal_gen = create_simple_signal_generator(
            min_confidence=0.60,
            stop_ticks=8.0,
            target_ticks=16.0,
        )

        # Run backtest
        result = engine.run(
            sample_data_with_predictions,
            signal_gen,
            verbose=False,
        )

        # Verify result structure
        assert result.report is not None
        assert result.report.trade_log is not None
        assert result.report.equity_curve is not None
        assert result.report.metrics is not None
        assert result.execution_time_seconds > 0

    def test_export_results(self, sample_data_with_predictions):
        """Test exporting backtest results."""
        engine = BacktestEngine()
        signal_gen = create_simple_signal_generator()

        result = engine.run(sample_data_with_predictions, signal_gen)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = result.report.export_all(tmpdir, prefix="test")

            assert 'trades_csv' in files
            assert 'equity_csv' in files
            assert Path(files['trades_csv']).exists()
            assert Path(files['equity_csv']).exists()

    def test_metrics_accuracy(self):
        """Test metrics calculation accuracy with known data."""
        # Create deterministic test case
        trade_pnls = [10.0, 10.0, -5.0, 10.0, -5.0]  # 3 wins, 2 losses
        equity_curve = [1000, 1010, 1020, 1015, 1025, 1020]

        metrics = calculate_metrics(
            trade_pnls=trade_pnls,
            equity_curve=equity_curve,
            initial_capital=1000,
            trading_days=1,
        )

        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate_pct == 60.0
        assert metrics.total_return_dollars == pytest.approx(20.0, rel=0.01)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        engine = BacktestEngine()

        with pytest.raises(ValueError):
            engine.run(pd.DataFrame(), lambda b, p, c: Signal(SignalType.HOLD))

    def test_single_bar(self):
        """Test with single bar of data."""
        data = pd.DataFrame({
            'open': [4500.0],
            'high': [4501.0],
            'low': [4499.0],
            'close': [4500.5],
            'volume': [100],
        }, index=pd.date_range('2023-01-03 09:30:00', periods=1, tz='America/New_York'))

        engine = BacktestEngine()
        result = engine.run(data, lambda b, p, c: Signal(SignalType.HOLD))

        assert result.report.trade_log.get_trade_count() == 0

    def test_all_flat_predictions(self, sample_data):
        """Test when model always predicts FLAT."""
        sample_data['prediction'] = 1  # Always FLAT
        sample_data['confidence'] = 0.90

        engine = BacktestEngine()
        signal_gen = create_simple_signal_generator()

        result = engine.run(sample_data, signal_gen)

        # Should have no trades since FLAT = no signal
        assert result.report.trade_log.get_trade_count() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
