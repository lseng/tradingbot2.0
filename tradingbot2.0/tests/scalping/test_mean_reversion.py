"""
Tests for Mean-Reversion Strategy

Tests cover:
1. Configuration validation
2. Feature generation
3. Setup identification
4. Trader entry/exit logic
5. Backtest execution
6. No lookahead bias
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.scalping.mean_reversion import (
    MeanReversionConfig,
    MeanReversionTrader,
    add_mean_reversion_features,
    identify_mean_reversion_setups,
    create_mean_reversion_target,
    run_mean_reversion_backtest,
)


# Fixtures
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 500

    # Create timestamps (RTH hours)
    base = datetime(2023, 1, 3, 9, 30)
    timestamps = [base + timedelta(minutes=5 * i) for i in range(n)]

    # Generate realistic price data
    close = 4000 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 5000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=pd.DatetimeIndex(timestamps, tz="America/New_York"))

    return df


@pytest.fixture
def df_with_features(sample_df):
    """Sample DataFrame with mean-reversion features added."""
    return add_mean_reversion_features(sample_df)


@pytest.fixture
def default_config():
    """Default configuration."""
    return MeanReversionConfig()


# Test Configuration
class TestMeanReversionConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MeanReversionConfig()

        assert config.low_vol_threshold == 0.40
        assert config.rsi_period == 7
        assert config.rsi_oversold == 30.0
        assert config.rsi_overbought == 70.0
        assert config.profit_target_ticks == 4.0
        assert config.stop_loss_ticks == 4.0
        assert config.time_stop_bars == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MeanReversionConfig(
            low_vol_threshold=0.35,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
            profit_target_ticks=3.0,
        )

        assert config.low_vol_threshold == 0.35
        assert config.rsi_oversold == 25.0
        assert config.rsi_overbought == 75.0
        assert config.profit_target_ticks == 3.0


# Test Feature Generation
class TestMeanReversionFeatures:
    """Test mean-reversion feature generation."""

    def test_add_features(self, sample_df):
        """Test that features are added correctly."""
        result = add_mean_reversion_features(sample_df)

        assert "rsi_7_raw" in result.columns
        assert "ema_21" in result.columns
        assert "ema_deviation" in result.columns
        assert "price_zscore" in result.columns
        assert "vwap_deviation_mr" in result.columns
        assert "up_streak" in result.columns
        assert "down_streak" in result.columns
        assert "reversion_rate" in result.columns

    def test_rsi_raw_range(self, df_with_features):
        """Test that raw RSI is in valid range."""
        rsi = df_with_features["rsi_7_raw"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_ema_deviation_range(self, df_with_features):
        """Test EMA deviation is reasonable."""
        dev = df_with_features["ema_deviation"].dropna()
        # Deviation should be small (typically < 5%)
        assert abs(dev.mean()) < 0.05
        assert dev.std() < 0.05

    def test_zscore_distribution(self, df_with_features):
        """Test price z-score has reasonable distribution."""
        zscore = df_with_features["price_zscore"].dropna()
        # Z-score should have mean ~0 and std ~1
        assert abs(zscore.mean()) < 0.5
        assert 0.5 < zscore.std() < 2.0


# Test Setup Identification
class TestSetupIdentification:
    """Test mean-reversion setup identification."""

    def test_identifies_setups(self, sample_df):
        """Test that setups are identified."""
        df = add_mean_reversion_features(sample_df)
        vol_predictions = np.random.rand(len(df))

        result = identify_mean_reversion_setups(df, vol_predictions)

        assert "is_low_vol" in result.columns
        assert "is_oversold" in result.columns
        assert "is_overbought" in result.columns
        assert "is_long_setup" in result.columns
        assert "is_short_setup" in result.columns
        assert "setup_direction" in result.columns

    def test_low_vol_filter(self, sample_df):
        """Test low volatility filter works."""
        df = add_mean_reversion_features(sample_df)

        # All low vol predictions
        vol_predictions = np.zeros(len(df))
        result = identify_mean_reversion_setups(df, vol_predictions)
        assert result["is_low_vol"].sum() == len(df)

        # All high vol predictions
        vol_predictions = np.ones(len(df))
        result = identify_mean_reversion_setups(df, vol_predictions)
        assert result["is_low_vol"].sum() == 0

    def test_setup_requires_low_vol(self, sample_df):
        """Test that setups require low volatility."""
        df = add_mean_reversion_features(sample_df)

        # Force RSI extreme but high vol
        df["rsi_7_raw"] = 20  # Oversold
        vol_predictions = np.ones(len(df)) * 0.9  # High vol

        result = identify_mean_reversion_setups(df, vol_predictions)
        # No long setups because vol is high
        assert result["is_long_setup"].sum() == 0

    def test_setup_direction_values(self, sample_df):
        """Test setup direction values are valid."""
        df = add_mean_reversion_features(sample_df)
        vol_predictions = np.random.rand(len(df))

        result = identify_mean_reversion_setups(df, vol_predictions)

        directions = result["setup_direction"].unique()
        assert all(d in [-1, 0, 1] for d in directions)


# Test Trader Logic
class TestMeanReversionTrader:
    """Test trading logic."""

    def test_initialization(self, default_config):
        """Test trader initialization."""
        trader = MeanReversionTrader(default_config)

        assert trader.position == 0
        assert trader.entry_price == 0.0
        assert trader.entry_bar == 0
        assert len(trader.trades) == 0

    def test_no_entry_high_vol(self, default_config):
        """Test no entry when volatility is high."""
        trader = MeanReversionTrader(default_config)

        direction = trader.should_enter(
            vol_prediction=0.80,  # High vol
            rsi=20,  # Oversold
            ema_deviation=-0.01,  # Extended below
        )

        assert direction == 0

    def test_long_entry_oversold(self, default_config):
        """Test long entry on oversold conditions."""
        trader = MeanReversionTrader(default_config)

        direction = trader.should_enter(
            vol_prediction=0.20,  # Low vol
            rsi=25,  # Oversold
            ema_deviation=-0.01,  # Extended below
            is_first_hour=False,
            is_last_hour=False,
        )

        assert direction == 1  # Long

    def test_short_entry_overbought(self, default_config):
        """Test short entry on overbought conditions."""
        trader = MeanReversionTrader(default_config)

        direction = trader.should_enter(
            vol_prediction=0.20,  # Low vol
            rsi=75,  # Overbought
            ema_deviation=0.01,  # Extended above
            is_first_hour=False,
            is_last_hour=False,
        )

        assert direction == -1  # Short

    def test_no_entry_first_hour(self, default_config):
        """Test no entry during first hour."""
        trader = MeanReversionTrader(default_config)

        direction = trader.should_enter(
            vol_prediction=0.20,
            rsi=25,
            ema_deviation=-0.01,
            is_first_hour=True,
            is_last_hour=False,
        )

        assert direction == 0

    def test_no_entry_last_hour(self, default_config):
        """Test no entry during last hour."""
        trader = MeanReversionTrader(default_config)

        direction = trader.should_enter(
            vol_prediction=0.20,
            rsi=75,
            ema_deviation=0.01,
            is_first_hour=False,
            is_last_hour=True,
        )

        assert direction == 0

    def test_no_entry_neutral_rsi(self, default_config):
        """Test no entry when RSI is neutral."""
        trader = MeanReversionTrader(default_config)

        direction = trader.should_enter(
            vol_prediction=0.20,  # Low vol
            rsi=50,  # Neutral
            ema_deviation=-0.01,  # Extended below
        )

        assert direction == 0

    def test_stop_loss_exit_long(self, default_config):
        """Test stop loss exit for long position."""
        trader = MeanReversionTrader(default_config)
        trader.position = 1
        trader.entry_price = 4000.0
        trader.entry_bar = 0

        # Price dropped 4+ ticks
        should_exit, reason = trader.should_exit(
            current_price=3999.0,
            bar_idx=1,
            high=4000.0,
            low=3998.0,  # 8 ticks below entry
        )

        assert should_exit
        assert reason == "stop_loss"

    def test_profit_target_exit_long(self, default_config):
        """Test profit target exit for long position."""
        trader = MeanReversionTrader(default_config)
        trader.position = 1
        trader.entry_price = 4000.0
        trader.entry_bar = 0

        # Price rose 4+ ticks
        should_exit, reason = trader.should_exit(
            current_price=4001.0,
            bar_idx=1,
            high=4002.0,  # 8 ticks above entry
            low=4000.0,
        )

        assert should_exit
        assert reason == "profit_target"

    def test_time_stop_exit(self, default_config):
        """Test time stop exit."""
        trader = MeanReversionTrader(default_config)
        trader.position = 1
        trader.entry_price = 4000.0
        trader.entry_bar = 0

        # 3+ bars held
        should_exit, reason = trader.should_exit(
            current_price=4000.5,
            bar_idx=3,  # 3 bars held
            high=4000.5,
            low=4000.0,
        )

        assert should_exit
        assert reason == "time_stop"

    def test_vol_regime_change_exit(self, default_config):
        """Test exit on volatility regime change."""
        trader = MeanReversionTrader(default_config)
        trader.position = 1
        trader.entry_price = 4000.0
        trader.entry_bar = 0

        # High vol predicted
        should_exit, reason = trader.should_exit(
            current_price=4000.5,
            bar_idx=1,
            high=4000.5,
            low=4000.0,
            vol_prediction=0.70,  # High vol
        )

        assert should_exit
        assert reason == "vol_regime_change"

    def test_enter_trade_slippage(self, default_config):
        """Test entry applies slippage."""
        trader = MeanReversionTrader(default_config)
        timestamp = pd.Timestamp("2023-01-03 10:00:00", tz="America/New_York")

        # Long entry
        trade = trader.enter_trade(1, 4000.0, 0, timestamp)
        assert trade["entry_price"] == 4000.25  # +1 tick slippage

        trader.reset()

        # Short entry
        trade = trader.enter_trade(-1, 4000.0, 0, timestamp)
        assert trade["entry_price"] == 3999.75  # -1 tick slippage

    def test_exit_trade_slippage(self, default_config):
        """Test exit applies slippage."""
        trader = MeanReversionTrader(default_config)
        timestamp = pd.Timestamp("2023-01-03 10:00:00", tz="America/New_York")

        # Long position exit
        trader.position = 1
        trader.entry_price = 4000.25
        trader.entry_bar = 0

        trade = trader.exit_trade(4001.0, 1, timestamp, "profit_target")
        assert trade["exit_price"] == 4000.75  # -1 tick slippage

        trader.reset()

        # Short position exit
        trader.position = -1
        trader.entry_price = 3999.75
        trader.entry_bar = 0

        trade = trader.exit_trade(3999.0, 1, timestamp, "profit_target")
        assert trade["exit_price"] == 3999.25  # +1 tick slippage

    def test_reset(self, default_config):
        """Test trader reset."""
        trader = MeanReversionTrader(default_config)
        trader.position = 1
        trader.entry_price = 4000.0
        trader.entry_bar = 5
        trader.trades = [{"test": 1}]

        trader.reset()

        assert trader.position == 0
        assert trader.entry_price == 0.0
        assert trader.entry_bar == 0
        assert len(trader.trades) == 0


# Test Target Creation
class TestMeanReversionTarget:
    """Test mean-reversion target creation."""

    def test_target_creation(self, sample_df):
        """Test target creation succeeds."""
        df = add_mean_reversion_features(sample_df)
        result, stats = create_mean_reversion_target(df)

        assert "target_reversion" in result.columns
        assert "total_samples" in stats
        assert "reversion_rate" in stats

    def test_target_values(self, sample_df):
        """Test target values are valid."""
        df = add_mean_reversion_features(sample_df)
        result, _ = create_mean_reversion_target(df)

        valid_targets = result["target_reversion"].dropna()
        assert all(t in [0, 1] for t in valid_targets)

    def test_no_lookahead_bias(self, sample_df):
        """Test that target doesn't use future data at point t."""
        df = add_mean_reversion_features(sample_df)
        result, _ = create_mean_reversion_target(df)

        # Last horizon_bars should be NaN (no future data)
        horizon = 3  # default
        assert result["target_reversion"].iloc[-1] is np.nan or np.isnan(result["target_reversion"].iloc[-1])


# Test Backtest
class TestMeanReversionBacktest:
    """Test backtest execution."""

    def test_backtest_runs(self, sample_df, default_config):
        """Test backtest completes without error."""
        df = add_mean_reversion_features(sample_df)
        vol_predictions = np.random.rand(len(df))

        trades, summary = run_mean_reversion_backtest(df, vol_predictions, default_config)

        assert isinstance(trades, list)
        assert isinstance(summary, dict)
        assert "total_trades" in summary
        assert "total_pnl" in summary
        assert "win_rate" in summary
        assert "profit_factor" in summary

    def test_backtest_no_trades_high_vol(self, sample_df, default_config):
        """Test no trades when all high volatility."""
        df = add_mean_reversion_features(sample_df)
        vol_predictions = np.ones(len(df))  # All high vol

        trades, summary = run_mean_reversion_backtest(df, vol_predictions, default_config)

        assert summary["total_trades"] == 0

    def test_backtest_trade_structure(self, sample_df, default_config):
        """Test trade structure has required fields."""
        df = add_mean_reversion_features(sample_df)
        # Force some low vol to generate trades
        vol_predictions = np.random.rand(len(df)) * 0.3  # Mostly low vol

        trades, _ = run_mean_reversion_backtest(df, vol_predictions, default_config)

        if trades:
            trade = trades[0]
            assert "entry_time" in trade
            assert "entry_bar" in trade
            assert "direction" in trade
            assert "entry_price" in trade
            assert "exit_time" in trade
            assert "exit_price" in trade
            assert "exit_reason" in trade
            assert "pnl_dollars" in trade

    def test_backtest_exit_reasons(self, sample_df, default_config):
        """Test backtest tracks exit reasons."""
        df = add_mean_reversion_features(sample_df)
        vol_predictions = np.random.rand(len(df)) * 0.3

        _, summary = run_mean_reversion_backtest(df, vol_predictions, default_config)

        assert "exit_reasons" in summary


# Test No Lookahead Bias
class TestNoLookaheadBias:
    """Test that features don't use future data."""

    def test_ema_no_lookahead(self, sample_df):
        """Test EMA uses only past data."""
        df = add_mean_reversion_features(sample_df)

        # EMA at time t should only use data up to t
        for i in range(50, 100):
            # Calculate EMA manually for verification
            past_data = sample_df["close"].iloc[:i+1]
            ema_manual = past_data.ewm(span=21, adjust=False).mean().iloc[-1]
            ema_feature = df["ema_21"].iloc[i]

            # Should be close (may have small numerical differences)
            assert abs(ema_manual - ema_feature) < 0.01

    def test_rsi_no_lookahead(self, sample_df):
        """Test RSI uses only past data."""
        df = add_mean_reversion_features(sample_df)

        # RSI should be same when calculated on truncated data
        for i in range(50, 100):
            truncated = sample_df.iloc[:i+1]
            from src.scalping.features import _calculate_rsi
            rsi_truncated = _calculate_rsi(truncated["close"], 7).iloc[-1]
            rsi_full = df["rsi_7_raw"].iloc[i]

            assert abs(rsi_truncated - rsi_full) < 0.01

    def test_features_independent_of_future(self, sample_df):
        """Test all features are same regardless of future data."""
        df_full = add_mean_reversion_features(sample_df)

        # Truncate to first 200 rows
        df_truncated = add_mean_reversion_features(sample_df.iloc[:200])

        # Features at row 100 should be same in both
        test_idx = 100
        feature_cols = ["ema_deviation", "price_zscore", "rsi_7_raw"]

        for col in feature_cols:
            if col in df_full.columns and col in df_truncated.columns:
                val_full = df_full[col].iloc[test_idx]
                val_truncated = df_truncated[col].iloc[test_idx]

                if not np.isnan(val_full) and not np.isnan(val_truncated):
                    assert abs(val_full - val_truncated) < 0.01, f"Feature {col} differs"
