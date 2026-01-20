"""
Tests for the breakout detection module.

These tests verify:
1. Consolidation detection features work correctly
2. Breakout direction features are computed without lookahead bias
3. Breakout target creation is correct
4. Trading logic works as expected
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, time

from src.scalping.breakout import (
    BreakoutConfig,
    BreakoutFeatureGenerator,
    BreakoutTrader,
    ConsolidationType,
    _calculate_range_position,
    _detect_squeeze,
    _calculate_consolidation_score,
    _calculate_momentum_divergence,
    create_breakout_target,
    identify_breakout_setups,
    run_breakout_backtest,
)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 500

    # Generate realistic price data with some consolidation and breakout patterns
    dates = pd.date_range(
        start="2023-01-03 09:30:00",
        periods=n,
        freq="5min",
        tz="America/New_York"
    )

    # Start with base price
    base_price = 4000.0
    prices = [base_price]

    # Generate price movements with consolidation periods
    for i in range(1, n):
        # Alternate between trending and consolidating
        if (i // 50) % 2 == 0:
            # Consolidation period (small moves)
            change = np.random.normal(0, 0.5)
        else:
            # Trending period (larger moves)
            change = np.random.normal(0.2, 2.0)
        prices.append(prices[-1] + change)

    prices = np.array(prices)

    # Create OHLC from close prices
    high = prices + np.abs(np.random.normal(0, 1, n))
    low = prices - np.abs(np.random.normal(0, 1, n))
    open_prices = prices + np.random.normal(0, 0.5, n)

    # Ensure OHLC relationships
    high = np.maximum(high, np.maximum(open_prices, prices))
    low = np.minimum(low, np.minimum(open_prices, prices))

    volume = np.random.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def sample_features_df(sample_ohlcv_df):
    """Create sample DataFrame with features already generated."""
    gen = BreakoutFeatureGenerator()
    return gen.generate_all(sample_ohlcv_df)


class TestBreakoutConfig:
    """Tests for BreakoutConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BreakoutConfig()
        assert config.lookback_bars == 12
        assert config.vol_prediction_threshold == 0.60
        assert config.horizon_bars == 6
        assert config.tick_size == 0.25

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BreakoutConfig(
            lookback_bars=20,
            vol_prediction_threshold=0.70,
            horizon_bars=12,
        )
        assert config.lookback_bars == 20
        assert config.vol_prediction_threshold == 0.70
        assert config.horizon_bars == 12


class TestRangePosition:
    """Tests for range position calculation."""

    def test_at_range_low(self, sample_ohlcv_df):
        """Test range position at range low."""
        position = _calculate_range_position(sample_ohlcv_df, lookback=20)

        # Should be between 0 and 1
        valid = position.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_range_position_bounds(self, sample_ohlcv_df):
        """Test range position stays within [0, 1]."""
        position = _calculate_range_position(sample_ohlcv_df, lookback=20)
        valid = position.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1


class TestSqueezeDetection:
    """Tests for Bollinger Band squeeze detection."""

    def test_squeeze_binary(self, sample_ohlcv_df):
        """Test squeeze indicator is binary."""
        squeeze = _detect_squeeze(sample_ohlcv_df)
        valid = squeeze.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_squeeze_detects_low_vol(self, sample_ohlcv_df):
        """Test that squeeze is detected in low volatility periods."""
        squeeze = _detect_squeeze(sample_ohlcv_df)
        # Should have some squeeze periods
        assert squeeze.sum() > 0


class TestConsolidationScore:
    """Tests for consolidation score calculation."""

    def test_score_range(self, sample_ohlcv_df):
        """Test consolidation score is in valid range."""
        score = _calculate_consolidation_score(sample_ohlcv_df, lookback=12)
        valid = score.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_score_variability(self, sample_ohlcv_df):
        """Test consolidation score has variability."""
        score = _calculate_consolidation_score(sample_ohlcv_df, lookback=12)
        valid = score.dropna()
        # Should have some variance
        assert valid.std() > 0


class TestMomentumDivergence:
    """Tests for momentum divergence calculation."""

    def test_divergence_range(self, sample_ohlcv_df):
        """Test divergence is in expected range."""
        divergence = _calculate_momentum_divergence(sample_ohlcv_df)
        valid = divergence.dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()


class TestBreakoutFeatureGenerator:
    """Tests for BreakoutFeatureGenerator."""

    def test_generates_all_features(self, sample_ohlcv_df):
        """Test that all expected features are generated."""
        gen = BreakoutFeatureGenerator()
        df = gen.generate_all(sample_ohlcv_df)
        feature_names = gen.get_feature_names()

        # Check all features are present
        missing = [f for f in feature_names if f not in df.columns]
        assert len(missing) == 0, f"Missing features: {missing}"

    def test_no_nan_in_features(self, sample_ohlcv_df):
        """Test that features don't have NaN after generation."""
        gen = BreakoutFeatureGenerator()
        df = gen.generate_all(sample_ohlcv_df)
        feature_names = gen.get_feature_names()

        for f in feature_names:
            nan_count = df[f].isna().sum()
            assert nan_count == 0, f"Feature {f} has {nan_count} NaN values"

    def test_consolidation_features_present(self, sample_ohlcv_df):
        """Test consolidation features are generated."""
        gen = BreakoutFeatureGenerator()
        df = gen.generate_all(sample_ohlcv_df)

        assert "bb_squeeze" in df.columns
        assert "consolidation_score" in df.columns
        assert "squeeze_duration" in df.columns
        assert "range_contraction" in df.columns

    def test_direction_features_present(self, sample_ohlcv_df):
        """Test direction hint features are generated."""
        gen = BreakoutFeatureGenerator()
        df = gen.generate_all(sample_ohlcv_df)

        assert "range_position" in df.columns
        assert "dist_from_high" in df.columns
        assert "dist_from_low" in df.columns
        assert "momentum_divergence" in df.columns

    def test_timing_features_present(self, sample_ohlcv_df):
        """Test timing features are generated."""
        gen = BreakoutFeatureGenerator()
        df = gen.generate_all(sample_ohlcv_df)

        assert "cumulative_consolidation" in df.columns
        assert "is_opening_range" in df.columns
        assert "is_pre_close" in df.columns


class TestBreakoutTargetCreation:
    """Tests for breakout target creation."""

    def test_target_creation(self, sample_features_df):
        """Test breakout target is created correctly."""
        df, stats = create_breakout_target(
            sample_features_df,
            horizon_bars=6,
            breakout_threshold_ticks=4.0,
        )

        assert "target_breakout" in df.columns
        assert "target_breakout_up" in df.columns
        assert "target_breakout_down" in df.columns

    def test_target_values(self, sample_features_df):
        """Test target values are valid."""
        df, stats = create_breakout_target(
            sample_features_df,
            horizon_bars=6,
            breakout_threshold_ticks=4.0,
        )

        valid = df["target_breakout"].dropna()
        assert set(valid.unique()).issubset({0, 1, 2})

    def test_binary_targets(self, sample_features_df):
        """Test binary up/down targets are valid."""
        df, stats = create_breakout_target(
            sample_features_df,
            horizon_bars=6,
            breakout_threshold_ticks=4.0,
        )

        valid_up = df["target_breakout_up"].dropna()
        valid_down = df["target_breakout_down"].dropna()

        assert set(valid_up.unique()).issubset({0, 1})
        assert set(valid_down.unique()).issubset({0, 1})

    def test_stats_returned(self, sample_features_df):
        """Test statistics are returned."""
        df, stats = create_breakout_target(
            sample_features_df,
            horizon_bars=6,
            breakout_threshold_ticks=4.0,
        )

        assert "total_samples" in stats
        assert "up_breakout_pct" in stats
        assert "down_breakout_pct" in stats
        assert "no_breakout_pct" in stats

    def test_no_lookahead_bias(self, sample_features_df):
        """Test that target uses only future data."""
        df, stats = create_breakout_target(
            sample_features_df,
            horizon_bars=6,
            breakout_threshold_ticks=4.0,
        )

        # Last few rows should have NaN target (no future data)
        assert df["target_breakout"].iloc[-1] is np.nan or pd.isna(df["target_breakout"].iloc[-1])


class TestBreakoutTrader:
    """Tests for BreakoutTrader class."""

    def test_initialization(self):
        """Test trader initialization."""
        trader = BreakoutTrader()
        assert trader.position == 0
        assert trader.profit_target_ticks == 6.0
        assert trader.stop_loss_ticks == 8.0

    def test_should_enter_no_consolidation(self):
        """Test no entry when not consolidated."""
        trader = BreakoutTrader(min_consolidation_score=0.6)
        direction = trader.should_enter(
            consolidation_score=0.3,  # Below threshold
            vol_prediction=0.8,
            range_position=0.2,
            bar_idx=0,
        )
        assert direction == 0

    def test_should_enter_no_vol_signal(self):
        """Test no entry when volatility prediction is low."""
        trader = BreakoutTrader(min_vol_confidence=0.6)
        direction = trader.should_enter(
            consolidation_score=0.8,
            vol_prediction=0.4,  # Below threshold
            range_position=0.2,
            bar_idx=0,
        )
        assert direction == 0

    def test_should_enter_long(self):
        """Test long entry when near range bottom."""
        trader = BreakoutTrader(min_consolidation_score=0.6, min_vol_confidence=0.6)
        direction = trader.should_enter(
            consolidation_score=0.8,
            vol_prediction=0.8,
            range_position=0.2,  # Near bottom
            bar_idx=0,
        )
        assert direction == 1  # Long

    def test_should_enter_short(self):
        """Test short entry when near range top."""
        trader = BreakoutTrader(min_consolidation_score=0.6, min_vol_confidence=0.6)
        direction = trader.should_enter(
            consolidation_score=0.8,
            vol_prediction=0.8,
            range_position=0.8,  # Near top
            bar_idx=0,
        )
        assert direction == -1  # Short

    def test_should_enter_no_direction(self):
        """Test no entry when in middle of range."""
        trader = BreakoutTrader(min_consolidation_score=0.6, min_vol_confidence=0.6)
        direction = trader.should_enter(
            consolidation_score=0.8,
            vol_prediction=0.8,
            range_position=0.5,  # Middle - no clear direction
            bar_idx=0,
        )
        assert direction == 0

    def test_stop_loss_exit(self):
        """Test stop loss exit."""
        trader = BreakoutTrader(stop_loss_ticks=8.0, tick_size=0.25)

        # Enter long
        trader.enter_trade(1, 4000.0, 0, pd.Timestamp.now(tz="America/New_York"))

        # Price drops below stop
        should_exit, reason = trader.should_exit(
            current_price=3996.0,
            bar_idx=1,
            high=4001.0,
            low=3996.0,  # 16+ ticks below entry (after slippage)
        )

        assert should_exit
        assert reason == "stop_loss"

    def test_profit_target_exit(self):
        """Test profit target exit."""
        trader = BreakoutTrader(profit_target_ticks=6.0, tick_size=0.25)

        # Enter long at 4000
        trader.enter_trade(1, 4000.0, 0, pd.Timestamp.now(tz="America/New_York"))
        # Entry with slippage = 4000.25

        # Price rises to hit target (entry + 6 ticks = 4000.25 + 1.50 = 4001.75)
        should_exit, reason = trader.should_exit(
            current_price=4002.0,
            bar_idx=1,
            high=4003.0,  # High hits target
            low=4001.0,
        )

        assert should_exit
        assert reason == "profit_target"

    def test_time_stop_exit(self):
        """Test time stop exit."""
        trader = BreakoutTrader(time_stop_bars=6)

        # Enter trade
        trader.enter_trade(1, 4000.0, 0, pd.Timestamp.now(tz="America/New_York"))

        # After 6 bars, check exit
        should_exit, reason = trader.should_exit(
            current_price=4000.5,
            bar_idx=6,  # 6 bars held
            high=4001.0,
            low=3999.0,
        )

        assert should_exit
        assert reason == "time_stop"

    def test_enter_trade_applies_slippage(self):
        """Test that entry applies 1 tick slippage."""
        trader = BreakoutTrader(tick_size=0.25)

        trade = trader.enter_trade(1, 4000.0, 0, pd.Timestamp.now(tz="America/New_York"))

        # Long entry should be 1 tick higher
        assert trade["entry_price"] == 4000.25

    def test_exit_trade_applies_slippage(self):
        """Test that exit applies 1 tick slippage."""
        trader = BreakoutTrader(tick_size=0.25)

        # Enter long
        trader.enter_trade(1, 4000.0, 0, pd.Timestamp.now(tz="America/New_York"))

        # Exit
        exit_info = trader.exit_trade(
            4002.0, 1, pd.Timestamp.now(tz="America/New_York"), "profit_target"
        )

        # Long exit should be 1 tick lower
        assert exit_info["exit_price"] == 4001.75

    def test_reset(self):
        """Test trader reset."""
        trader = BreakoutTrader()
        trader.enter_trade(1, 4000.0, 0, pd.Timestamp.now(tz="America/New_York"))

        trader.reset()

        assert trader.position == 0
        assert trader.entry_price == 0.0
        assert trader.entry_bar == 0


class TestIdentifyBreakoutSetups:
    """Tests for setup identification."""

    def test_identifies_setups(self, sample_features_df):
        """Test that setups are identified correctly."""
        vol_predictions = np.random.uniform(0.3, 0.9, len(sample_features_df))

        result = identify_breakout_setups(
            sample_features_df,
            vol_predictions,
            vol_threshold=0.6,
            consolidation_threshold=0.5,
        )

        assert "is_breakout_setup" in result.columns
        assert "setup_direction" in result.columns
        assert "vol_prediction" in result.columns

    def test_setup_requires_consolidation(self, sample_features_df):
        """Test that setups require consolidation."""
        # All high volatility predictions
        vol_predictions = np.ones(len(sample_features_df)) * 0.9

        result = identify_breakout_setups(
            sample_features_df,
            vol_predictions,
            vol_threshold=0.6,
            consolidation_threshold=0.99,  # Very high threshold
        )

        # Should have few or no setups due to high consolidation threshold
        assert result["is_breakout_setup"].sum() < len(result) * 0.5

    def test_setup_requires_vol_prediction(self, sample_features_df):
        """Test that setups require volatility prediction."""
        # All low volatility predictions
        vol_predictions = np.ones(len(sample_features_df)) * 0.3

        result = identify_breakout_setups(
            sample_features_df,
            vol_predictions,
            vol_threshold=0.6,
            consolidation_threshold=0.1,
        )

        # Should have no setups due to low vol predictions
        assert result["is_breakout_setup"].sum() == 0


class TestBreakoutBacktest:
    """Tests for breakout backtest."""

    def test_backtest_runs(self, sample_features_df):
        """Test that backtest completes without error."""
        vol_predictions = np.random.uniform(0.4, 0.8, len(sample_features_df))
        feature_names = BreakoutFeatureGenerator().get_feature_names()

        trades, summary = run_breakout_backtest(
            sample_features_df,
            vol_predictions,
            feature_names,
        )

        assert isinstance(trades, list)
        assert isinstance(summary, dict)
        assert "total_trades" in summary
        assert "total_pnl" in summary
        assert "win_rate" in summary

    def test_backtest_no_trades_with_strict_thresholds(self, sample_features_df):
        """Test that no trades occur with very strict thresholds."""
        vol_predictions = np.ones(len(sample_features_df)) * 0.3  # Low vol

        config = BreakoutConfig(
            vol_prediction_threshold=0.95,  # Very high
            consolidation_threshold=0.95,  # Very high
        )

        trades, summary = run_breakout_backtest(
            sample_features_df,
            vol_predictions,
            [],
            config=config,
        )

        assert summary["total_trades"] == 0

    def test_backtest_trade_structure(self, sample_features_df):
        """Test that trades have required fields."""
        # Force some trades
        vol_predictions = np.ones(len(sample_features_df)) * 0.8
        sample_features_df["consolidation_score"] = 0.8
        sample_features_df["range_position"] = 0.2  # Near bottom

        config = BreakoutConfig(
            vol_prediction_threshold=0.6,
            consolidation_threshold=0.5,
        )

        trades, summary = run_breakout_backtest(
            sample_features_df,
            vol_predictions,
            [],
            config=config,
        )

        if trades:
            trade = trades[0]
            assert "entry_time" in trade
            assert "entry_price" in trade
            assert "direction" in trade
            assert "exit_time" in trade
            assert "exit_price" in trade
            assert "pnl_dollars" in trade
            assert "exit_reason" in trade


class TestNoLookaheadBias:
    """Tests to verify no lookahead bias in features or targets."""

    def test_consolidation_score_no_lookahead(self, sample_ohlcv_df):
        """Test consolidation score doesn't use future data."""
        score = _calculate_consolidation_score(sample_ohlcv_df, lookback=12)

        # Changing future values shouldn't affect current score
        df_modified = sample_ohlcv_df.copy()
        df_modified.iloc[-10:, df_modified.columns.get_loc("close")] *= 2

        score_modified = _calculate_consolidation_score(df_modified, lookback=12)

        # Scores before last 10 bars should be the same
        pd.testing.assert_series_equal(
            score.iloc[:-15],  # Extra buffer for lookback
            score_modified.iloc[:-15],
            check_names=False,
        )

    def test_range_position_no_lookahead(self, sample_ohlcv_df):
        """Test range position doesn't use future data."""
        position = _calculate_range_position(sample_ohlcv_df, lookback=20)

        # Changing future values shouldn't affect current position
        df_modified = sample_ohlcv_df.copy()
        df_modified.iloc[-10:, df_modified.columns.get_loc("close")] *= 2

        position_modified = _calculate_range_position(df_modified, lookback=20)

        # Positions before last 10 bars should be the same
        pd.testing.assert_series_equal(
            position.iloc[:-15],  # Extra buffer
            position_modified.iloc[:-15],
            check_names=False,
        )

    def test_features_independent_of_future(self, sample_ohlcv_df):
        """Test all features are independent of future data."""
        gen = BreakoutFeatureGenerator()

        # Generate features on original data (400 bars)
        df_original = sample_ohlcv_df.iloc[:400].copy()
        df_features = gen.generate_all(df_original)

        # Generate features on data with different future values
        # Only modify the last 50 bars (after warmup period)
        df_modified = sample_ohlcv_df.iloc[:400].copy()
        df_modified.iloc[350:] *= 10  # Drastically change last 50 bars

        df_features_mod = gen.generate_all(df_modified)

        # After warmup (200), features at indices 0-149 (original indices 200-349)
        # should be unchanged since the modified data is at indices 350+
        # The feature output starts at warmup period (200), so index 100 in features
        # corresponds to index 300 in original data, which is before the modification
        feature_names = gen.get_feature_names()
        compare_end = min(100, len(df_features) - 1, len(df_features_mod) - 1)

        for f in feature_names:
            if f in df_features.columns and f in df_features_mod.columns:
                # Compare only the first 100 rows (well before the modification point)
                original_vals = df_features[f].iloc[:compare_end].values
                modified_vals = df_features_mod[f].iloc[:compare_end].values

                # Use approximate equality due to floating point
                np.testing.assert_allclose(
                    original_vals,
                    modified_vals,
                    rtol=1e-5,
                    err_msg=f"Feature {f} depends on future data"
                )
