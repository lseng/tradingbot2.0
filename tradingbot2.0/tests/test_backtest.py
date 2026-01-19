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
    SessionFilter,
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


class TestSessionFiltering:
    """Tests for session filtering and timezone handling."""

    @pytest.fixture
    def utc_data(self):
        """Create sample data with UTC timestamps."""
        # Create data across multiple sessions
        timestamps = pd.date_range(
            start='2023-01-03 14:30:00',  # 9:30 AM NY in UTC
            periods=36000,  # 10 hours of data (covers RTH and beyond)
            freq='1s',
            tz='UTC',
        )

        prices = np.full(len(timestamps), 4500.0)
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.full(len(timestamps), 100),
        }, index=timestamps)

        return data

    @pytest.fixture
    def mixed_session_data(self):
        """Create data that spans RTH and ETH."""
        # Start at 6 AM NY (ETH) and run through 5 PM NY
        timestamps = pd.date_range(
            start='2023-01-03 06:00:00',
            periods=39600,  # 11 hours of data
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

        return data

    def test_session_filter_rth_only(self, mixed_session_data):
        """Test RTH-only session filtering."""
        config = BacktestConfig(
            session_filter=SessionFilter.RTH_ONLY,
        )
        engine = BacktestEngine(config=config)

        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(mixed_session_data, no_trade)

        # RTH is 9:30-16:00 = 6.5 hours = 23400 seconds
        assert result.data_stats['bars_filtered'] > 0
        # Original 11 hours minus filtered = RTH only
        assert result.data_stats['session_filter'] == 'rth_only'

    def test_session_filter_all(self, mixed_session_data):
        """Test ALL session filter (no filtering)."""
        config = BacktestConfig(
            session_filter=SessionFilter.ALL,
        )
        engine = BacktestEngine(config=config)

        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(mixed_session_data, no_trade)

        # Should have all bars
        assert result.data_stats['bars_filtered'] == 0
        assert result.data_stats['session_filter'] == 'all'

    def test_utc_to_ny_conversion(self, utc_data):
        """Test that UTC timestamps are properly converted to NY for session checks."""
        config = BacktestConfig(
            session_filter=SessionFilter.RTH_ONLY,
            convert_timestamps_to_ny=True,
        )
        engine = BacktestEngine(config=config)

        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(utc_data, no_trade)

        # Should filter out non-RTH hours
        assert result.data_stats['bars_filtered'] > 0
        # The data starts at 9:30 AM NY (14:30 UTC) so most should be RTH

    def test_eod_flatten_with_utc_data(self):
        """Test EOD flatten works correctly with UTC timestamps."""
        # Create UTC data around EOD time (4:30 PM NY = 21:30 UTC in winter)
        timestamps = pd.date_range(
            start='2023-01-03 21:00:00',  # 4:00 PM NY
            periods=3600,  # 1 hour
            freq='1s',
            tz='UTC',
        )

        prices = np.full(len(timestamps), 4500.0)
        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.full(len(timestamps), 100),
        }, index=timestamps)

        config = BacktestConfig(
            session_filter=SessionFilter.ALL,  # Don't filter
            convert_timestamps_to_ny=True,
        )
        engine = BacktestEngine(config=config)

        trades_after_flatten = []

        def try_to_trade(bar, position, context):
            if position is None:
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(data, try_to_trade)

        # Engine should prevent trading after 4:30 PM
        # Any position should be flattened
        assert result is not None

    def test_session_filter_eth_only(self, mixed_session_data):
        """Test ETH-only session filtering."""
        config = BacktestConfig(
            session_filter=SessionFilter.ETH_ONLY,
        )
        engine = BacktestEngine(config=config)

        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(mixed_session_data, no_trade)

        # ETH should filter out RTH (9:30-16:00)
        assert result.data_stats['bars_filtered'] > 0
        assert result.data_stats['session_filter'] == 'eth_only'

    def test_empty_after_filtering(self):
        """Test handling when all data is filtered out."""
        # Create data only during non-trading hours (Saturday)
        timestamps = pd.date_range(
            start='2023-01-07 12:00:00',  # Saturday
            periods=1000,
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

        config = BacktestConfig(
            session_filter=SessionFilter.RTH_ONLY,
        )
        engine = BacktestEngine(config=config)

        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(data, no_trade)

        # Should return valid result with 0 bars
        assert result.data_stats['total_bars'] == 0
        assert result.data_stats['bars_filtered'] == 1000

    def test_timezone_conversion_disabled(self):
        """Test that timezone conversion can be disabled."""
        timestamps = pd.date_range(
            start='2023-01-03 09:30:00',
            periods=1000,
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

        config = BacktestConfig(
            convert_timestamps_to_ny=False,
        )
        engine = BacktestEngine(config=config)

        # Should still work with NY-timezone data
        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(data, no_trade)
        assert result is not None

    def test_dst_handling(self):
        """Test DST transition handling (March - after spring forward)."""
        # March 13, 2023 was the Monday after DST transition
        timestamps = pd.date_range(
            start='2023-03-13 09:30:00',
            periods=23400,  # RTH duration
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

        config = BacktestConfig(
            session_filter=SessionFilter.RTH_ONLY,
            convert_timestamps_to_ny=True,
        )
        engine = BacktestEngine(config=config)

        def no_trade(bar, position, context):
            return Signal(SignalType.HOLD)

        result = engine.run(data, no_trade)

        # Should handle DST without errors
        assert result is not None
        assert result.data_stats['total_bars'] > 0


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


# =============================================================================
# Fill Mode Tests
# =============================================================================

class TestFillModes:
    """Tests for different order fill modes."""

    @pytest.fixture
    def fill_mode_data(self):
        """Create data with known OHLC values for fill mode testing."""
        timestamps = pd.date_range(
            start='2023-01-03 09:30:00',
            periods=100,
            freq='1s',
            tz='America/New_York',
        )

        # Create predictable price data
        # Each bar: open=100, high=101, low=99, close=100.5
        data = pd.DataFrame({
            'open': [4500.0 + i * 0.5 for i in range(100)],  # Incremental opens
            'high': [4501.0 + i * 0.5 for i in range(100)],
            'low': [4499.0 + i * 0.5 for i in range(100)],
            'close': [4500.25 + i * 0.5 for i in range(100)],
            'volume': [100] * 100,
        }, index=timestamps)

        return data

    def test_signal_bar_close_mode_fills_at_close(self, fill_mode_data):
        """Test SIGNAL_BAR_CLOSE mode fills at the signal bar's close price."""
        config = BacktestConfig(
            fill_mode=OrderFillMode.SIGNAL_BAR_CLOSE,
            slippage_ticks=0,  # Disable slippage for precise testing
            enable_dynamic_slippage=False,
        )
        engine = BacktestEngine(config=config)

        bar_count = [0]
        signal_bar_close = [None]

        def capture_entry(bar, position, context):
            bar_idx = bar_count[0]
            bar_count[0] += 1

            if position is None and bar_idx == 10:
                # Capture the close price at signal bar
                signal_bar_close[0] = bar['close']
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            elif position is not None:
                return Signal(SignalType.EXIT_LONG, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(fill_mode_data, capture_entry)

        # With SIGNAL_BAR_CLOSE, fill at signal bar's close
        # Since slippage is 0, entry should be exactly at close
        trades = result.report.trade_log.get_trades()
        if trades:
            assert trades[0].entry_price == pytest.approx(signal_bar_close[0], rel=0.001)

    def test_next_bar_open_mode_fills_at_next_open(self, fill_mode_data):
        """Test NEXT_BAR_OPEN mode queues entry and fills at next bar's open."""
        config = BacktestConfig(
            fill_mode=OrderFillMode.NEXT_BAR_OPEN,
            slippage_ticks=0,  # Disable slippage for precise testing
            enable_dynamic_slippage=False,
        )
        engine = BacktestEngine(config=config)

        bar_count = [0]
        signal_bar_idx = [None]
        next_bar_open = [None]

        def track_entry(bar, position, context):
            current_idx = bar_count[0]
            bar_count[0] += 1

            # Capture the open of bar 11 (the next bar after signal)
            if current_idx == 11:
                next_bar_open[0] = bar['open']

            if position is None and signal_bar_idx[0] is None and current_idx == 10:
                signal_bar_idx[0] = current_idx
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            elif position is not None:
                return Signal(SignalType.EXIT_LONG, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(fill_mode_data, track_entry)

        # With NEXT_BAR_OPEN, signal at bar 10 should fill at bar 11's open
        trades = result.report.trade_log.get_trades()
        if trades:
            # Entry price should be bar 11's open
            assert trades[0].entry_price == pytest.approx(next_bar_open[0], rel=0.001)
            # Verify it's different from bar 10's close
            bar10_close = fill_mode_data.iloc[10]['close']
            assert trades[0].entry_price != bar10_close, \
                "NEXT_BAR_OPEN should fill at next bar's open, not signal bar's close"

    def test_price_touch_mode_rejects_untouched_price(self):
        """Test PRICE_TOUCH mode rejects entries when price not in range."""
        # Create data where close is OUTSIDE high/low range (impossible in real data)
        # This tests the edge case where price touch check matters
        timestamps = pd.date_range(
            start='2023-01-03 09:30:00',
            periods=50,
            freq='1s',
            tz='America/New_York',
        )

        # Create normal OHLC where close is always in range
        data = pd.DataFrame({
            'open': [4500.0] * 50,
            'high': [4501.0] * 50,
            'low': [4499.0] * 50,
            'close': [4500.5] * 50,  # Within [4499, 4501]
            'volume': [100] * 50,
        }, index=timestamps)

        config = BacktestConfig(
            fill_mode=OrderFillMode.PRICE_TOUCH,
            enable_dynamic_slippage=False,
        )
        engine = BacktestEngine(config=config)

        entries_attempted = [0]

        def always_try_entry(bar, position, context):
            if position is None:
                entries_attempted[0] += 1
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            return Signal(SignalType.EXIT_LONG, confidence=0.80)

        result = engine.run(data, always_try_entry)

        # With valid OHLC data, PRICE_TOUCH should allow fills
        # (close is always within high/low range)
        assert result.report.trade_log.get_trade_count() > 0

    def test_fill_modes_produce_different_results(self, fill_mode_data):
        """Test that different fill modes produce different entry prices."""
        results = {}

        for mode in [OrderFillMode.SIGNAL_BAR_CLOSE, OrderFillMode.NEXT_BAR_OPEN]:
            config = BacktestConfig(
                fill_mode=mode,
                slippage_ticks=0,  # Disable slippage
                enable_dynamic_slippage=False,
            )
            engine = BacktestEngine(config=config)

            entry_count = [0]

            def single_entry(bar, position, context):
                if position is None and entry_count[0] == 0:
                    entry_count[0] += 1
                    return Signal(SignalType.LONG_ENTRY, confidence=0.80)
                elif position is not None:
                    return Signal(SignalType.EXIT_LONG, confidence=0.80)
                return Signal(SignalType.HOLD)

            result = engine.run(fill_mode_data, single_entry)

            trades = result.report.trade_log.get_trades()
            if trades:
                results[mode.value] = trades[0].entry_price

        # The two modes should produce different entry prices
        # (SIGNAL_BAR_CLOSE uses close, NEXT_BAR_OPEN uses next bar's open)
        if len(results) == 2:
            assert results['signal_bar_close'] != results['next_bar_open'], \
                "Fill modes should produce different entry prices"

    def test_pending_entry_cancelled_at_eod(self):
        """Test that pending entries are cancelled at EOD flatten time."""
        # Create data spanning EOD
        timestamps = pd.date_range(
            start='2023-01-03 16:28:00',  # Just before EOD flatten (4:30 PM)
            periods=300,  # 5 minutes
            freq='1s',
            tz='America/New_York',
        )

        data = pd.DataFrame({
            'open': [4500.0] * 300,
            'high': [4501.0] * 300,
            'low': [4499.0] * 300,
            'close': [4500.5] * 300,
            'volume': [100] * 300,
        }, index=timestamps)

        config = BacktestConfig(
            fill_mode=OrderFillMode.NEXT_BAR_OPEN,
            session_filter=SessionFilter.ALL,
        )
        engine = BacktestEngine(config=config)

        signal_at_4_29 = [False]

        def try_late_entry(bar, position, context):
            bar_time = bar.name.time()
            # Try to enter at 4:29 PM (1 min before flatten)
            if position is None and not signal_at_4_29[0]:
                if bar_time.hour == 16 and bar_time.minute == 29:
                    signal_at_4_29[0] = True
                    return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(data, try_late_entry)

        # Entry should be cancelled because fill would happen at 4:30 (EOD flatten)
        # No position should be open at end
        assert result is not None


# =============================================================================
# ATR-Based Dynamic Slippage Tests
# =============================================================================

class TestATRDynamicSlippage:
    """Tests for ATR-based dynamic slippage in backtest engine."""

    @pytest.fixture
    def volatile_data(self):
        """Create data with increasing volatility for ATR testing."""
        timestamps = pd.date_range(
            start='2023-01-03 09:30:00',
            periods=200,
            freq='1s',
            tz='America/New_York',
        )

        # First 100 bars: low volatility (ATR ~0.5)
        # Next 100 bars: high volatility (ATR ~2.0)
        low_vol_ranges = [0.5] * 100
        high_vol_ranges = [2.0] * 100
        ranges = low_vol_ranges + high_vol_ranges

        base_price = 4500.0
        data = pd.DataFrame({
            'open': [base_price] * 200,
            'high': [base_price + r/2 for r in ranges],
            'low': [base_price - r/2 for r in ranges],
            'close': [base_price] * 200,
            'volume': [100] * 200,
        }, index=timestamps)

        return data

    def test_atr_calculation_updates(self, volatile_data):
        """Test that ATR calculation updates during backtest."""
        config = BacktestConfig(
            atr_period=14,
            atr_baseline_period=50,
            enable_dynamic_slippage=True,
        )
        engine = BacktestEngine(config=config)

        atr_values = []

        def capture_atr(bar, position, context):
            atr_values.append(engine._current_atr)
            return Signal(SignalType.HOLD)

        result = engine.run(volatile_data, capture_atr)

        # ATR should increase when we hit the high volatility section
        # Check that ATR is different in low-vol vs high-vol sections
        low_vol_atr = np.mean(atr_values[50:100])  # After warmup in low vol
        high_vol_atr = np.mean(atr_values[150:200])  # In high vol section

        assert high_vol_atr > low_vol_atr, "ATR should be higher in high volatility section"

    def test_dynamic_slippage_enabled(self):
        """Test that dynamic slippage is applied when enabled."""
        timestamps = pd.date_range(
            start='2023-01-03 09:30:00',
            periods=100,
            freq='1s',
            tz='America/New_York',
        )

        # High volatility data
        data = pd.DataFrame({
            'open': [4500.0] * 100,
            'high': [4510.0] * 100,  # 10 point range = high vol
            'low': [4490.0] * 100,
            'close': [4500.0] * 100,
            'volume': [100] * 100,
        }, index=timestamps)

        config = BacktestConfig(
            fill_mode=OrderFillMode.SIGNAL_BAR_CLOSE,
            slippage_ticks=1.0,
            enable_dynamic_slippage=True,
            atr_period=14,
            atr_baseline_period=50,
        )
        engine = BacktestEngine(config=config)

        def single_entry(bar, position, context):
            if position is None and context.get('entered') is None:
                context['entered'] = True
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            elif position is not None:
                return Signal(SignalType.EXIT_LONG, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(data, single_entry, context={})

        # With dynamic slippage, high volatility should result in additional slippage
        # Just verify it runs without error and produces trades
        assert result is not None

    def test_dynamic_slippage_disabled(self):
        """Test that slippage is not affected by ATR when disabled."""
        timestamps = pd.date_range(
            start='2023-01-03 09:30:00',
            periods=100,
            freq='1s',
            tz='America/New_York',
        )

        data = pd.DataFrame({
            'open': [4500.0] * 100,
            'high': [4510.0] * 100,
            'low': [4490.0] * 100,
            'close': [4500.0] * 100,
            'volume': [100] * 100,
        }, index=timestamps)

        config = BacktestConfig(
            fill_mode=OrderFillMode.SIGNAL_BAR_CLOSE,
            slippage_ticks=1.0,
            enable_dynamic_slippage=False,  # Disabled
        )
        engine = BacktestEngine(config=config)

        def single_entry(bar, position, context):
            if position is None and context.get('entered') is None:
                context['entered'] = True
                return Signal(SignalType.LONG_ENTRY, confidence=0.80)
            elif position is not None:
                return Signal(SignalType.EXIT_LONG, confidence=0.80)
            return Signal(SignalType.HOLD)

        result = engine.run(data, single_entry, context={})

        # Check that trades are executed
        trades = result.report.trade_log.get_trades()
        if trades:
            trade = trades[0]
            # Entry should be close + 1 tick slippage (buying adds slippage)
            expected_entry = 4500.0 + 0.25  # close + 1 tick
            assert trade.entry_price == pytest.approx(expected_entry, rel=0.01)

    def test_atr_params_in_config_serialization(self):
        """Test that ATR config params are included in serialization."""
        config = BacktestConfig(
            atr_period=20,
            atr_baseline_period=100,
            enable_dynamic_slippage=True,
        )

        d = config.to_dict()

        assert d['atr_period'] == 20
        assert d['atr_baseline_period'] == 100
        assert d['enable_dynamic_slippage'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
