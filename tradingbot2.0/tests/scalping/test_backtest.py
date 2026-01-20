"""
Tests for the 5-Minute Scalping Backtest Engine

These tests verify that the backtest engine correctly:
1. Processes bar-by-bar data
2. Applies entry/exit rules (stop loss, take profit, time stop, EOD)
3. Calculates P&L with slippage and commission
4. Generates accurate metrics

Why these tests matter:
- Backtest engines are the foundation of strategy validation
- Bugs in backtest code lead to false confidence in unprofitable strategies
- These tests ensure realistic execution modeling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

from src.scalping.backtest import (
    ScalpingBacktest,
    BacktestConfig,
    BacktestResult,
    Trade,
    Position,
    ExitReason,
    run_backtest,
    analyze_results,
    export_trades_csv,
    export_summary_json,
    TICK_SIZE,
    TICK_VALUE,
    POINT_VALUE,
)


NY_TZ = ZoneInfo("America/New_York")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_5min_bars():
    """Create sample 5-minute OHLCV data for testing."""
    # Create 2 days of data (78 bars per day RTH)
    dates = []
    base_date = datetime(2024, 1, 2, 9, 30, tzinfo=NY_TZ)

    # Day 1
    for i in range(78):  # 9:30 to 4:00 = 78 5-min bars
        dates.append(base_date + timedelta(minutes=i * 5))

    # Day 2
    base_date = datetime(2024, 1, 3, 9, 30, tzinfo=NY_TZ)
    for i in range(78):
        dates.append(base_date + timedelta(minutes=i * 5))

    # Create price data with some movement
    n = len(dates)
    base_price = 5000.0
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close - np.random.randn(n) * 0.5

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
    }, index=pd.DatetimeIndex(dates, tz=NY_TZ))

    # Add dummy features
    df["return_1bar"] = df["close"].pct_change()
    df["return_3bar"] = df["close"].pct_change(3)
    df["close_vs_ema8"] = (df["close"] / df["close"].ewm(span=8).mean()) - 1
    df["rsi_14"] = 50 + np.random.randn(n) * 10  # Mock RSI
    df["atr_14"] = np.abs(df["high"] - df["low"]).rolling(14).mean()

    # Fill NaN with 0 for simplicity
    df = df.fillna(0)

    return df


@pytest.fixture
def mock_model():
    """Create a mock ScalpingModel for testing."""
    model = MagicMock()

    def mock_get_trading_signals(X, min_confidence=0.60):
        n = len(X)
        # Generate deterministic signals for testing
        signals = np.zeros(n, dtype=int)
        confidences = np.full(n, 0.55)  # Below threshold by default
        should_trade = np.full(n, False)

        # Every 10th bar, generate a signal
        for i in range(0, n, 10):
            if i % 20 == 0:
                signals[i] = 1  # LONG
            else:
                signals[i] = -1  # SHORT
            confidences[i] = 0.65
            should_trade[i] = True

        return signals, confidences, should_trade

    model.get_trading_signals = mock_get_trading_signals
    return model


@pytest.fixture
def winning_mock_model():
    """Create a mock model that generates consistently winning signals."""
    model = MagicMock()

    def mock_get_trading_signals(X, min_confidence=0.60):
        n = len(X)
        signals = np.zeros(n, dtype=int)
        confidences = np.full(n, 0.55)
        should_trade = np.full(n, False)

        # Generate LONG signals when price is likely to go up
        for i in range(5, n - 10, 20):
            signals[i] = 1
            confidences[i] = 0.70
            should_trade[i] = True

        return signals, confidences, should_trade

    model.get_trading_signals = mock_get_trading_signals
    return model


@pytest.fixture
def default_config():
    """Default backtest configuration."""
    return BacktestConfig(
        initial_capital=1000.0,
        profit_target_ticks=6,
        stop_loss_ticks=8,
        time_stop_bars=6,
        min_confidence=0.60,
        max_position=1,
        commission_per_side=0.42,
        slippage_ticks=1.0,
        max_daily_loss=100.0,
        verbose=False,
    )


# ============================================================================
# BacktestConfig Tests
# ============================================================================


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values from spec."""
        config = BacktestConfig()

        assert config.initial_capital == 1000.0
        assert config.profit_target_ticks == 6  # $7.50
        assert config.stop_loss_ticks == 8  # $10.00
        assert config.time_stop_bars == 6  # 30 minutes
        assert config.min_confidence == 0.60
        assert config.max_position == 1
        assert config.max_daily_loss == 100.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=5000.0,
            profit_target_ticks=8,
            stop_loss_ticks=6,
            time_stop_bars=4,
        )

        assert config.initial_capital == 5000.0
        assert config.profit_target_ticks == 8
        assert config.stop_loss_ticks == 6
        assert config.time_stop_bars == 4

    def test_time_constraints(self):
        """Test EOD time constraints."""
        config = BacktestConfig()

        assert config.no_new_entries_time == time(15, 45)
        assert config.flatten_time == time(15, 55)


# ============================================================================
# ScalpingBacktest Basic Tests
# ============================================================================


class TestScalpingBacktestInit:
    """Tests for ScalpingBacktest initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        backtest = ScalpingBacktest()

        assert backtest.config is not None
        assert backtest.position is None
        assert backtest.equity == backtest.config.initial_capital
        assert backtest.trades == []
        assert backtest.equity_curve == []

    def test_custom_config_initialization(self, default_config):
        """Test initialization with custom config."""
        backtest = ScalpingBacktest(config=default_config)

        assert backtest.config == default_config
        assert backtest.equity == 1000.0


class TestScalpingBacktestRun:
    """Tests for the main backtest run method."""

    def test_run_completes_without_error(self, sample_5min_bars, mock_model, default_config):
        """Test that backtest runs to completion."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == len(sample_5min_bars)

    def test_run_generates_trades(self, sample_5min_bars, mock_model, default_config):
        """Test that backtest generates trades from signals."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        # Should have some trades
        assert result.total_trades > 0
        assert len(result.trades) > 0

    def test_run_tracks_daily_pnl(self, sample_5min_bars, mock_model, default_config):
        """Test that daily P&L is tracked."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        # Should have 2 days of data
        assert len(result.daily_pnls) == 2

    def test_empty_dataframe_raises_error(self, mock_model, default_config):
        """Test that empty DataFrame raises error."""
        backtest = ScalpingBacktest(config=default_config)

        with pytest.raises(ValueError, match="empty"):
            backtest.run(pd.DataFrame(), mock_model)

    def test_missing_features_raises_error(self, sample_5min_bars, mock_model, default_config):
        """Test that missing features raises error."""
        backtest = ScalpingBacktest(config=default_config)

        # Request features that don't exist
        with pytest.raises(ValueError, match="Missing feature columns"):
            backtest.run(sample_5min_bars, mock_model, feature_cols=["nonexistent_feature"])


# ============================================================================
# Exit Condition Tests
# ============================================================================


class TestExitConditions:
    """Tests for position exit logic."""

    def test_profit_target_exit_long(self, default_config):
        """Test that LONG position exits at profit target."""
        backtest = ScalpingBacktest(config=default_config)

        # Create a position
        entry_price = 5000.0
        target_price = entry_price + (default_config.profit_target_ticks * TICK_SIZE)

        backtest.position = Position(
            entry_time=datetime.now(NY_TZ),
            entry_price=entry_price,
            direction=1,  # LONG
            contracts=1,
            confidence=0.65,
            stop_price=entry_price - (default_config.stop_loss_ticks * TICK_SIZE),
            target_price=target_price,
            entry_bar_idx=0,
        )

        # Create a bar that hits target
        bar = pd.Series({
            "open": entry_price + 0.5,
            "high": target_price + 1.0,  # Hits target
            "low": entry_price,
            "close": target_price,
        })

        should_exit, reason, exit_price = backtest._check_exits(
            bar_idx=1,
            bar_time=datetime.now(NY_TZ),
            bar=bar,
        )

        assert should_exit is True
        assert reason == ExitReason.PROFIT_TARGET
        assert exit_price == target_price

    def test_profit_target_exit_short(self, default_config):
        """Test that SHORT position exits at profit target."""
        backtest = ScalpingBacktest(config=default_config)

        entry_price = 5000.0
        target_price = entry_price - (default_config.profit_target_ticks * TICK_SIZE)

        backtest.position = Position(
            entry_time=datetime.now(NY_TZ),
            entry_price=entry_price,
            direction=-1,  # SHORT
            contracts=1,
            confidence=0.65,
            stop_price=entry_price + (default_config.stop_loss_ticks * TICK_SIZE),
            target_price=target_price,
            entry_bar_idx=0,
        )

        # Create a bar that hits target
        bar = pd.Series({
            "open": entry_price - 0.5,
            "high": entry_price,
            "low": target_price - 1.0,  # Hits target
            "close": target_price,
        })

        should_exit, reason, exit_price = backtest._check_exits(
            bar_idx=1,
            bar_time=datetime.now(NY_TZ),
            bar=bar,
        )

        assert should_exit is True
        assert reason == ExitReason.PROFIT_TARGET
        assert exit_price == target_price

    def test_stop_loss_exit_long(self, default_config):
        """Test that LONG position exits at stop loss."""
        backtest = ScalpingBacktest(config=default_config)

        entry_price = 5000.0
        stop_price = entry_price - (default_config.stop_loss_ticks * TICK_SIZE)

        backtest.position = Position(
            entry_time=datetime.now(NY_TZ),
            entry_price=entry_price,
            direction=1,  # LONG
            contracts=1,
            confidence=0.65,
            stop_price=stop_price,
            target_price=entry_price + (default_config.profit_target_ticks * TICK_SIZE),
            entry_bar_idx=0,
        )

        # Create a bar that hits stop
        bar = pd.Series({
            "open": entry_price - 0.5,
            "high": entry_price,
            "low": stop_price - 1.0,  # Hits stop
            "close": stop_price,
        })

        should_exit, reason, exit_price = backtest._check_exits(
            bar_idx=1,
            bar_time=datetime.now(NY_TZ),
            bar=bar,
        )

        assert should_exit is True
        assert reason == ExitReason.STOP_LOSS
        assert exit_price == stop_price

    def test_stop_loss_exit_short(self, default_config):
        """Test that SHORT position exits at stop loss."""
        backtest = ScalpingBacktest(config=default_config)

        entry_price = 5000.0
        stop_price = entry_price + (default_config.stop_loss_ticks * TICK_SIZE)

        backtest.position = Position(
            entry_time=datetime.now(NY_TZ),
            entry_price=entry_price,
            direction=-1,  # SHORT
            contracts=1,
            confidence=0.65,
            stop_price=stop_price,
            target_price=entry_price - (default_config.profit_target_ticks * TICK_SIZE),
            entry_bar_idx=0,
        )

        # Create a bar that hits stop
        bar = pd.Series({
            "open": entry_price + 0.5,
            "high": stop_price + 1.0,  # Hits stop
            "low": entry_price,
            "close": stop_price,
        })

        should_exit, reason, exit_price = backtest._check_exits(
            bar_idx=1,
            bar_time=datetime.now(NY_TZ),
            bar=bar,
        )

        assert should_exit is True
        assert reason == ExitReason.STOP_LOSS
        assert exit_price == stop_price

    def test_time_stop_exit(self, default_config):
        """Test that position exits after time stop (6 bars = 30 min)."""
        backtest = ScalpingBacktest(config=default_config)

        entry_price = 5000.0

        backtest.position = Position(
            entry_time=datetime.now(NY_TZ),
            entry_price=entry_price,
            direction=1,
            contracts=1,
            confidence=0.65,
            stop_price=entry_price - (default_config.stop_loss_ticks * TICK_SIZE),
            target_price=entry_price + (default_config.profit_target_ticks * TICK_SIZE),
            entry_bar_idx=0,
        )

        # Create a neutral bar (no stop/target hit)
        bar = pd.Series({
            "open": entry_price,
            "high": entry_price + 0.5,
            "low": entry_price - 0.5,
            "close": entry_price + 0.25,
        })

        # After 6 bars, should trigger time stop
        should_exit, reason, exit_price = backtest._check_exits(
            bar_idx=6,  # 6 bars later
            bar_time=datetime.now(NY_TZ),
            bar=bar,
        )

        assert should_exit is True
        assert reason == ExitReason.TIME_STOP
        assert exit_price == bar["close"]

    def test_eod_flatten_exit(self, default_config):
        """Test that position is flattened at EOD (3:55 PM)."""
        backtest = ScalpingBacktest(config=default_config)

        entry_price = 5000.0

        backtest.position = Position(
            entry_time=datetime.now(NY_TZ),
            entry_price=entry_price,
            direction=1,
            contracts=1,
            confidence=0.65,
            stop_price=entry_price - (default_config.stop_loss_ticks * TICK_SIZE),
            target_price=entry_price + (default_config.profit_target_ticks * TICK_SIZE),
            entry_bar_idx=0,
        )

        # Create a bar at 3:55 PM
        bar = pd.Series({
            "open": entry_price,
            "high": entry_price + 0.5,
            "low": entry_price - 0.5,
            "close": entry_price,
        })

        bar_time = datetime(2024, 1, 2, 15, 55, tzinfo=NY_TZ)

        should_exit, reason, exit_price = backtest._check_exits(
            bar_idx=1,
            bar_time=bar_time,
            bar=bar,
        )

        assert should_exit is True
        assert reason == ExitReason.EOD_FLATTEN
        assert exit_price == bar["close"]


# ============================================================================
# Entry Condition Tests
# ============================================================================


class TestEntryConditions:
    """Tests for position entry logic."""

    def test_no_entry_after_345pm(self, default_config):
        """Test that no new entries after 3:45 PM."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()

        # Create bar at 3:46 PM
        bar_time = datetime(2024, 1, 2, 15, 46, tzinfo=NY_TZ)
        bar = pd.Series({
            "open": 5000.0,
            "high": 5001.0,
            "low": 4999.0,
            "close": 5000.5,
        })

        # Process bar with valid signal
        backtest._process_bar(
            bar_idx=0,
            bar_time=bar_time,
            bar=bar,
            signal=1,
            confidence=0.70,
            should_trade=True,
        )

        # Should not have opened a position
        assert backtest.position is None

    def test_no_entry_when_daily_loss_exceeded(self, default_config):
        """Test that no entries when daily loss limit exceeded."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()
        backtest.current_day = datetime(2024, 1, 2).date()
        backtest.current_day_pnl = -100.0  # At limit

        bar_time = datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ)
        bar = pd.Series({
            "open": 5000.0,
            "high": 5001.0,
            "low": 4999.0,
            "close": 5000.5,
        })

        backtest._process_bar(
            bar_idx=0,
            bar_time=bar_time,
            bar=bar,
            signal=1,
            confidence=0.70,
            should_trade=True,
        )

        # Should not have opened a position
        assert backtest.position is None

    def test_entry_applies_slippage(self, default_config):
        """Test that entry applies slippage correctly."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()

        bar_time = datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ)
        bar = pd.Series({
            "open": 5000.0,
            "high": 5001.0,
            "low": 4999.0,
            "close": 5000.5,
        })

        backtest._open_position(
            bar_idx=0,
            bar_time=bar_time,
            bar=bar,
            direction=1,  # LONG
            confidence=0.70,
        )

        # Entry price should be bar open + 1 tick slippage
        expected_entry = 5000.0 + TICK_SIZE  # 5000.25
        assert backtest.position.entry_price == expected_entry


# ============================================================================
# P&L Calculation Tests
# ============================================================================


class TestPnLCalculation:
    """Tests for P&L calculations."""

    def test_pnl_includes_commission(self, sample_5min_bars, mock_model, default_config):
        """Test that P&L includes commission."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        # Commission should be tracked
        assert result.metrics.total_commission > 0

        # Each trade should have commission
        for trade in result.trades:
            assert trade.commission > 0

    def test_winning_trade_pnl(self, default_config):
        """Test P&L calculation for a winning trade."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()

        entry_price = 5000.0
        direction = 1  # LONG
        target_price = entry_price + (default_config.profit_target_ticks * TICK_SIZE)

        backtest.position = Position(
            entry_time=datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ),
            entry_price=entry_price,
            direction=direction,
            contracts=1,
            confidence=0.70,
            stop_price=entry_price - (default_config.stop_loss_ticks * TICK_SIZE),
            target_price=target_price,
            entry_bar_idx=0,
        )

        # Close at target
        backtest._close_position(
            exit_time=datetime(2024, 1, 2, 10, 15, tzinfo=NY_TZ),
            exit_price=target_price,
            exit_reason=ExitReason.PROFIT_TARGET,
            bar_idx=3,
        )

        # Should have 1 trade
        assert len(backtest.trades) == 1
        trade = backtest.trades[0]

        # Gross PnL should be positive (6 ticks = 1.5 points = $7.50)
        # But exit gets slippage too, so actual exit is 1 tick lower
        # Expected gross = ((target - 1 tick) - entry) * 5 = (1.5 - 0.25) * 5 = $6.25
        # Commission = $0.84
        # Net = $6.25 - $0.84 = $5.41
        assert trade.gross_pnl > 0
        assert trade.net_pnl < trade.gross_pnl  # Commission reduces net

    def test_losing_trade_pnl(self, default_config):
        """Test P&L calculation for a losing trade."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()

        entry_price = 5000.0
        direction = 1  # LONG
        stop_price = entry_price - (default_config.stop_loss_ticks * TICK_SIZE)

        backtest.position = Position(
            entry_time=datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ),
            entry_price=entry_price,
            direction=direction,
            contracts=1,
            confidence=0.70,
            stop_price=stop_price,
            target_price=entry_price + (default_config.profit_target_ticks * TICK_SIZE),
            entry_bar_idx=0,
        )

        # Close at stop
        backtest._close_position(
            exit_time=datetime(2024, 1, 2, 10, 15, tzinfo=NY_TZ),
            exit_price=stop_price,
            exit_reason=ExitReason.STOP_LOSS,
            bar_idx=3,
        )

        trade = backtest.trades[0]

        # Gross PnL should be negative
        assert trade.gross_pnl < 0
        # Net PnL should be even more negative (commission)
        assert trade.net_pnl < trade.gross_pnl


# ============================================================================
# Trade Recording Tests
# ============================================================================


class TestTradeRecording:
    """Tests for trade record creation."""

    def test_trade_record_fields(self, sample_5min_bars, mock_model, default_config):
        """Test that trades have all required fields."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        if result.trades:
            trade = result.trades[0]

            # Check all fields are set
            assert trade.entry_time is not None
            assert trade.exit_time is not None
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.direction in [1, -1]
            assert trade.contracts > 0
            assert isinstance(trade.gross_pnl, (int, float))
            assert trade.commission >= 0
            assert trade.exit_reason is not None
            assert trade.confidence >= 0
            assert trade.bars_held >= 0

    def test_trade_to_dict(self, sample_5min_bars, mock_model, default_config):
        """Test trade conversion to dictionary."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        if result.trades:
            trade = result.trades[0]
            trade_dict = trade.to_dict()

            assert "entry_time" in trade_dict
            assert "exit_time" in trade_dict
            assert "direction" in trade_dict
            assert "net_pnl" in trade_dict
            assert "exit_reason" in trade_dict


# ============================================================================
# Results and Metrics Tests
# ============================================================================


class TestBacktestResult:
    """Tests for BacktestResult structure."""

    def test_result_structure(self, sample_5min_bars, mock_model, default_config):
        """Test that result has expected structure."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.daily_pnls, dict)
        assert isinstance(result.metrics, object)  # PerformanceMetrics
        assert result.config == default_config

    def test_exit_reason_breakdown(self, sample_5min_bars, mock_model, default_config):
        """Test that exit reasons are tracked."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        assert "profit_target" in result.exits_by_reason
        assert "stop_loss" in result.exits_by_reason
        assert "time_stop" in result.exits_by_reason
        assert "eod_flatten" in result.exits_by_reason

    def test_trades_by_confidence_tier(self, sample_5min_bars, mock_model, default_config):
        """Test that trades are categorized by confidence."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        assert "high" in result.trades_by_confidence
        assert "medium" in result.trades_by_confidence
        assert "low" in result.trades_by_confidence

    def test_trades_by_hour(self, sample_5min_bars, mock_model, default_config):
        """Test that trades are categorized by hour."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        # Should have trades during RTH hours (9-16)
        for hour in result.trades_by_hour.keys():
            assert 9 <= hour <= 16


# ============================================================================
# Analysis Function Tests
# ============================================================================


class TestAnalyzeResults:
    """Tests for the analyze_results function."""

    def test_analyze_returns_dict(self, sample_5min_bars, mock_model, default_config):
        """Test that analyze_results returns a dictionary."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)
        analysis = analyze_results(result)

        assert isinstance(analysis, dict)
        assert "summary" in analysis
        assert "exit_reasons" in analysis
        assert "by_confidence" in analysis
        assert "by_hour" in analysis

    def test_analyze_best_worst_days(self, sample_5min_bars, mock_model, default_config):
        """Test that best/worst days are identified."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)
        analysis = analyze_results(result)

        assert "worst_days" in analysis
        assert "best_days" in analysis


# ============================================================================
# Export Function Tests
# ============================================================================


class TestExportFunctions:
    """Tests for export functions."""

    def test_export_trades_csv(self, sample_5min_bars, mock_model, default_config, tmp_path):
        """Test CSV export."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        csv_path = tmp_path / "trades.csv"
        export_trades_csv(result, str(csv_path))

        assert csv_path.exists()

        # Read back and verify
        df = pd.read_csv(csv_path)
        assert len(df) == len(result.trades)

    def test_export_summary_json(self, sample_5min_bars, mock_model, default_config, tmp_path):
        """Test JSON export."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        json_path = tmp_path / "summary.json"
        export_summary_json(result, str(json_path))

        assert json_path.exists()

        # Read back and verify
        import json
        with open(json_path) as f:
            data = json.load(f)
        assert "summary" in data


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestRunBacktest:
    """Tests for the run_backtest convenience function."""

    def test_run_backtest_function(self, sample_5min_bars, mock_model):
        """Test the convenience function."""
        result = run_backtest(sample_5min_bars, mock_model)

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0

    def test_run_backtest_with_custom_config(self, sample_5min_bars, mock_model, default_config):
        """Test convenience function with custom config."""
        result = run_backtest(sample_5min_bars, mock_model, config=default_config)

        assert result.config == default_config


# ============================================================================
# MFE/MAE Tracking Tests
# ============================================================================


class TestMfeMaeTracking:
    """Tests for Maximum Favorable/Adverse Excursion tracking."""

    def test_mfe_tracked_during_trade(self, default_config):
        """Test that MFE is tracked during trade."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()

        entry_price = 5000.0
        backtest.position = Position(
            entry_time=datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ),
            entry_price=entry_price,
            direction=1,  # LONG
            contracts=1,
            confidence=0.70,
            stop_price=entry_price - 2.0,
            target_price=entry_price + 1.5,
            entry_bar_idx=0,
        )

        # Process bar with favorable move
        bar = pd.Series({
            "open": entry_price,
            "high": entry_price + 1.0,  # Favorable
            "low": entry_price - 0.5,
            "close": entry_price + 0.5,
        })

        backtest._check_exits(
            bar_idx=1,
            bar_time=datetime(2024, 1, 2, 10, 5, tzinfo=NY_TZ),
            bar=bar,
        )

        # MFE should be updated
        assert backtest.position.mfe > 0

    def test_mae_tracked_during_trade(self, default_config):
        """Test that MAE is tracked during trade."""
        backtest = ScalpingBacktest(config=default_config)
        backtest._reset()

        entry_price = 5000.0
        backtest.position = Position(
            entry_time=datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ),
            entry_price=entry_price,
            direction=1,  # LONG
            contracts=1,
            confidence=0.70,
            stop_price=entry_price - 3.0,
            target_price=entry_price + 1.5,
            entry_bar_idx=0,
        )

        # Process bar with adverse move
        bar = pd.Series({
            "open": entry_price,
            "high": entry_price + 0.25,
            "low": entry_price - 1.0,  # Adverse
            "close": entry_price - 0.5,
        })

        backtest._check_exits(
            bar_idx=1,
            bar_time=datetime(2024, 1, 2, 10, 5, tzinfo=NY_TZ),
            bar=bar,
        )

        # MAE should be negative for long position
        assert backtest.position.mae < 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_bar_data(self, mock_model, default_config):
        """Test with minimal data (1 bar)."""
        dates = [datetime(2024, 1, 2, 10, 0, tzinfo=NY_TZ)]
        df = pd.DataFrame({
            "open": [5000.0],
            "high": [5001.0],
            "low": [4999.0],
            "close": [5000.5],
            "volume": [100],
            "return_1bar": [0.0],
        }, index=pd.DatetimeIndex(dates, tz=NY_TZ))

        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(df, mock_model)

        assert len(result.equity_curve) == 1

    def test_no_signals_generated(self, sample_5min_bars, default_config):
        """Test when model generates no signals."""
        no_signal_model = MagicMock()
        no_signal_model.get_trading_signals = lambda X, **kwargs: (
            np.zeros(len(X), dtype=int),
            np.full(len(X), 0.50),
            np.full(len(X), False),
        )

        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, no_signal_model)

        assert result.total_trades == 0
        assert len(result.trades) == 0

    def test_all_signals_below_threshold(self, sample_5min_bars, default_config):
        """Test when all signals are below confidence threshold."""
        low_conf_model = MagicMock()
        low_conf_model.get_trading_signals = lambda X, **kwargs: (
            np.ones(len(X), dtype=int),  # All LONG signals
            np.full(len(X), 0.55),  # Below 0.60 threshold
            np.full(len(X), False),  # No trades
        )

        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, low_conf_model)

        assert result.total_trades == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real model (if available)."""

    @pytest.mark.slow
    def test_with_real_model_training(self):
        """Test backtest with a real trained model (slow test)."""
        pytest.skip("Requires real model training - run manually")

    def test_equity_curve_integrity(self, sample_5min_bars, mock_model, default_config):
        """Test that equity curve reflects all trades."""
        backtest = ScalpingBacktest(config=default_config)
        result = backtest.run(sample_5min_bars, mock_model)

        # Final equity should equal initial + sum of net P&Ls
        expected_final = default_config.initial_capital + sum(t.net_pnl for t in result.trades)
        actual_final = result.equity_curve[-1] if result.equity_curve else default_config.initial_capital

        assert abs(actual_final - expected_final) < 0.01  # Allow small float error
