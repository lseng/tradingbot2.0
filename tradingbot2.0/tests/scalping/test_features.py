"""
Tests for the 5-minute scalping feature generator.

Tests cover:
1. Individual feature calculations (returns, EMAs, momentum, volatility, volume, time)
2. Feature generation pipeline
3. Lookahead bias validation
4. Feature normalization
5. Target variable creation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from scalping.features import (
    ScalpingFeatureGenerator,
    FeatureConfig,
    create_target_variable,
    _calculate_returns,
    _calculate_emas,
    _calculate_rsi,
    _calculate_momentum,
    _calculate_atr,
    _calculate_volatility,
    _calculate_volume,
    _calculate_time_features,
)


class TestReturnFeatures:
    """Tests for return feature calculations."""

    def test_return_1bar(self, full_day_5min_data):
        """Test 1-bar return calculation."""
        returns = _calculate_returns(full_day_5min_data, [1])

        assert "return_1bar" in returns.columns
        # First return should be NaN
        assert pd.isna(returns.iloc[0]["return_1bar"])
        # Subsequent returns should be calculated
        assert not pd.isna(returns.iloc[1]["return_1bar"])

    def test_return_multiple_periods(self, full_day_5min_data):
        """Test returns for multiple lookback periods."""
        periods = [1, 3, 6, 12, 24]
        returns = _calculate_returns(full_day_5min_data, periods)

        for period in periods:
            assert f"return_{period}bar" in returns.columns

    def test_return_calculation_correct(self, full_day_5min_data):
        """Test that return calculation is mathematically correct."""
        returns = _calculate_returns(full_day_5min_data, [1])

        # Manual calculation
        close = full_day_5min_data["close"]
        expected_return = (close.iloc[5] - close.iloc[4]) / close.iloc[4]

        assert returns.iloc[5]["return_1bar"] == pytest.approx(expected_return, rel=1e-6)


class TestEMAFeatures:
    """Tests for EMA deviation features."""

    def test_ema_features_created(self, extended_5min_data):
        """Test that EMA features are created."""
        emas = _calculate_emas(extended_5min_data, [8, 21, 50, 200])

        assert "close_vs_ema8" in emas.columns
        assert "close_vs_ema21" in emas.columns
        assert "close_vs_ema50" in emas.columns
        assert "close_vs_ema200" in emas.columns

    def test_ema_normalized(self, extended_5min_data):
        """Test that EMA deviations are normalized by price."""
        emas = _calculate_emas(extended_5min_data, [8])

        # Values should be small percentages (< 1% typically)
        valid_values = emas["close_vs_ema8"].dropna()
        assert all(abs(v) < 0.1 for v in valid_values)  # Within 10%


class TestRSIFeatures:
    """Tests for RSI calculation."""

    def test_rsi_range(self, extended_5min_data):
        """Test that RSI is in valid range [0, 100]."""
        rsi = _calculate_rsi(extended_5min_data["close"], 14)

        valid_rsi = rsi.dropna()
        assert all(0 <= v <= 100 for v in valid_rsi)

    def test_rsi_neutral_handling(self, extended_5min_data):
        """Test RSI handles flat periods (no gain/loss)."""
        rsi = _calculate_rsi(extended_5min_data["close"], 14)

        # Should have valid values (no NaN except warmup)
        nan_count = rsi.isna().sum()
        assert nan_count < 20  # Some warmup NaN expected


class TestMomentumFeatures:
    """Tests for momentum indicators (RSI, MACD)."""

    def test_momentum_features_created(self, extended_5min_data):
        """Test that all momentum features are created."""
        momentum = _calculate_momentum(
            extended_5min_data,
            rsi_periods=[7, 14],
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
        )

        assert "rsi_7" in momentum.columns
        assert "rsi_14" in momentum.columns
        assert "macd" in momentum.columns
        assert "macd_signal" in momentum.columns
        assert "macd_hist" in momentum.columns

    def test_rsi_normalized_range(self, extended_5min_data):
        """Test that RSI is normalized to [-1, 1] range."""
        momentum = _calculate_momentum(
            extended_5min_data,
            rsi_periods=[14],
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
        )

        valid_rsi = momentum["rsi_14"].dropna()
        assert all(-1 <= v <= 1 for v in valid_rsi)


class TestATRCalculation:
    """Tests for ATR calculation."""

    def test_atr_positive(self, extended_5min_data):
        """Test that ATR is always positive."""
        atr = _calculate_atr(extended_5min_data, 14)

        valid_atr = atr.dropna()
        assert all(v > 0 for v in valid_atr)

    def test_atr_reasonable_range(self, extended_5min_data):
        """Test that ATR is in reasonable range relative to price."""
        atr = _calculate_atr(extended_5min_data, 14)
        close = extended_5min_data["close"]

        # ATR should typically be < 1% of price for MES
        atr_pct = atr / close
        valid_pct = atr_pct.dropna()
        assert all(v < 0.02 for v in valid_pct)  # < 2% of price


class TestVolatilityFeatures:
    """Tests for volatility features."""

    def test_volatility_features_created(self, extended_5min_data):
        """Test that volatility features are created."""
        vol = _calculate_volatility(extended_5min_data, atr_period=14, bb_period=20, bb_std=2.0)

        assert "atr_14" in vol.columns
        assert "bb_width" in vol.columns
        assert "bar_range" in vol.columns

    def test_bb_width_positive(self, extended_5min_data):
        """Test that Bollinger Band width is positive."""
        vol = _calculate_volatility(extended_5min_data, atr_period=14, bb_period=20, bb_std=2.0)

        valid_bb = vol["bb_width"].dropna()
        assert all(v > 0 for v in valid_bb)

    def test_bar_range_positive(self, extended_5min_data):
        """Test that bar range is positive."""
        vol = _calculate_volatility(extended_5min_data, atr_period=14, bb_period=20, bb_std=2.0)

        valid_range = vol["bar_range"].dropna()
        assert all(v >= 0 for v in valid_range)


class TestVolumeFeatures:
    """Tests for volume features."""

    def test_volume_features_created(self, extended_5min_data):
        """Test that volume features are created."""
        vol = _calculate_volume(extended_5min_data, ma_period=20)

        assert "volume_ratio_20" in vol.columns
        assert "volume_trend" in vol.columns
        assert "vwap_deviation" in vol.columns

    def test_volume_ratio_around_one(self, extended_5min_data):
        """Test that volume ratio centers around 1.0."""
        vol = _calculate_volume(extended_5min_data, ma_period=20)

        valid_ratio = vol["volume_ratio_20"].dropna()
        mean_ratio = valid_ratio.mean()
        # Should be approximately 1.0 (actual volume / average volume)
        assert 0.5 < mean_ratio < 2.0


class TestTimeFeatures:
    """Tests for time-of-day features."""

    def test_time_features_created(self, full_day_5min_data):
        """Test that time features are created."""
        time_feats = _calculate_time_features(full_day_5min_data)

        assert "time_of_day" in time_feats.columns
        assert "minutes_since_open" in time_feats.columns
        assert "is_first_hour" in time_feats.columns
        assert "is_last_hour" in time_feats.columns

    def test_time_of_day_range(self, full_day_5min_data):
        """Test that time_of_day is in [0, 1] range."""
        time_feats = _calculate_time_features(full_day_5min_data)

        # For RTH data, should be between 0 and 1
        assert all(0 <= v <= 1 for v in time_feats["time_of_day"])

    def test_first_hour_flag(self, full_day_5min_data):
        """Test that first hour flag is set correctly."""
        time_feats = _calculate_time_features(full_day_5min_data)

        # First 12 bars (60 min / 5 min = 12 bars) should have is_first_hour = 1
        first_12_bars = time_feats.iloc[:12]["is_first_hour"]
        assert all(v == 1 for v in first_12_bars)

    def test_last_hour_flag(self, full_day_5min_data):
        """Test that last hour flag is set correctly."""
        time_feats = _calculate_time_features(full_day_5min_data)

        # Last 12 bars should have is_last_hour = 1
        last_12_bars = time_feats.iloc[-12:]["is_last_hour"]
        assert all(v == 1 for v in last_12_bars)

    def test_flags_binary(self, full_day_5min_data):
        """Test that flags are binary (0 or 1)."""
        time_feats = _calculate_time_features(full_day_5min_data)

        assert all(v in [0, 1] for v in time_feats["is_first_hour"])
        assert all(v in [0, 1] for v in time_feats["is_last_hour"])


class TestScalpingFeatureGenerator:
    """Tests for the ScalpingFeatureGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = ScalpingFeatureGenerator()

        assert generator.config is not None
        assert generator.config.return_periods == [1, 3, 6, 12, 24]

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = FeatureConfig(return_periods=[1, 5, 10])
        generator = ScalpingFeatureGenerator(config=config)

        assert generator.config.return_periods == [1, 5, 10]

    def test_generate_all_features(self, extended_5min_data):
        """Test generating all 24 features."""
        generator = ScalpingFeatureGenerator()
        df_with_features = generator.generate_all(extended_5min_data)

        feature_names = generator.get_feature_names()
        assert len(feature_names) == 24

        for name in feature_names:
            assert name in df_with_features.columns

    def test_warmup_dropped(self, extended_5min_data):
        """Test that warmup period rows are dropped."""
        generator = ScalpingFeatureGenerator()
        df_with_features = generator.generate_all(extended_5min_data, drop_warmup=True)

        # Should have fewer rows than original
        expected_len = len(extended_5min_data) - generator.config.warmup_period
        assert len(df_with_features) == expected_len

    def test_no_nan_after_warmup(self, extended_5min_data):
        """Test that there are no NaN in features after warmup."""
        generator = ScalpingFeatureGenerator()
        df_with_features = generator.generate_all(extended_5min_data, drop_warmup=True)

        feature_names = generator.get_feature_names()
        nan_count = df_with_features[feature_names].isna().sum().sum()

        # Should be 0 NaN (filled with 0 if any)
        assert nan_count == 0

    def test_feature_matrix(self, extended_5min_data):
        """Test getting feature matrix as numpy array."""
        generator = ScalpingFeatureGenerator()
        df_with_features = generator.generate_all(extended_5min_data)

        feature_matrix = generator.get_feature_matrix(df_with_features)

        assert isinstance(feature_matrix, np.ndarray)
        assert feature_matrix.shape[1] == 24

    def test_get_feature_names(self):
        """Test getting feature names."""
        generator = ScalpingFeatureGenerator()
        names = generator.get_feature_names()

        assert len(names) == 24
        assert "return_1bar" in names
        assert "close_vs_ema8" in names
        assert "rsi_14" in names
        assert "atr_14" in names
        assert "time_of_day" in names

    def test_validate_no_lookahead(self, extended_5min_data):
        """Test lookahead bias validation."""
        generator = ScalpingFeatureGenerator()
        df_with_features = generator.generate_all(extended_5min_data, drop_warmup=False)

        # Should pass validation (features use past data only)
        assert generator.validate_no_lookahead(df_with_features)

    def test_chainable_feature_methods(self, extended_5min_data):
        """Test that feature methods can be chained."""
        generator = ScalpingFeatureGenerator()

        df = generator.add_returns(extended_5min_data)
        df = generator.add_emas(df)
        df = generator.add_momentum(df)
        df = generator.add_volatility(df)
        df = generator.add_volume(df)
        df = generator.add_time_features(df)

        # Should have all feature columns
        assert "return_1bar" in df.columns
        assert "close_vs_ema8" in df.columns
        assert "rsi_14" in df.columns
        assert "atr_14" in df.columns
        assert "volume_ratio_20" in df.columns
        assert "time_of_day" in df.columns


class TestCreateTargetVariable:
    """Tests for target variable creation."""

    def test_target_created(self, extended_5min_data):
        """Test that target variable is created."""
        df_with_target = create_target_variable(extended_5min_data, horizon_bars=6)

        assert "target_6bar" in df_with_target.columns

    def test_target_binary(self, extended_5min_data):
        """Test that target is binary (0 or 1)."""
        df_with_target = create_target_variable(extended_5min_data, horizon_bars=6)

        valid_targets = df_with_target["target_6bar"].dropna()
        assert all(v in [0, 1] for v in valid_targets)

    def test_target_nan_at_end(self, extended_5min_data):
        """Test that target has NaN at the end (no future data)."""
        df_with_target = create_target_variable(extended_5min_data, horizon_bars=6)

        # Last 6 rows should have NaN target
        last_targets = df_with_target["target_6bar"].tail(6)
        assert all(pd.isna(v) for v in last_targets)

    def test_target_class_balance(self, extended_5min_data):
        """Test that target has reasonable class balance (roughly 50/50)."""
        df_with_target = create_target_variable(
            extended_5min_data,
            horizon_bars=6,
            min_move_ticks=0,  # No minimum move for this test
        )

        valid_targets = df_with_target["target_6bar"].dropna()
        class_1_pct = valid_targets.mean()

        # Should be roughly balanced (between 30% and 70%)
        assert 0.3 < class_1_pct < 0.7

    def test_multiple_horizons(self, extended_5min_data):
        """Test creating targets for multiple horizons."""
        df_with_target = create_target_variable(extended_5min_data, horizon_bars=6)

        # Should also create 3-bar and 12-bar targets
        assert "target_3bar" in df_with_target.columns
        assert "target_6bar" in df_with_target.columns
        assert "target_12bar" in df_with_target.columns

    def test_min_move_filter(self, extended_5min_data):
        """Test that minimum move filter reduces positive targets."""
        df_no_filter = create_target_variable(
            extended_5min_data,
            horizon_bars=6,
            min_move_ticks=0,
        )
        df_with_filter = create_target_variable(
            extended_5min_data,
            horizon_bars=6,
            min_move_ticks=2.0,
        )

        # With filter, should have fewer positive targets
        no_filter_pct = df_no_filter["target_6bar"].dropna().mean()
        with_filter_pct = df_with_filter["target_6bar"].dropna().mean()

        # The filter should reduce positive class percentage
        # (requires larger upward move to count as 1)
        assert with_filter_pct <= no_filter_pct
