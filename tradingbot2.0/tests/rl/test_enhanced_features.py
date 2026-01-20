"""
Tests for RL enhanced feature engineering module.

Tests cover:
- add_volume_profile_features function
- add_advanced_momentum_features function
- add_regime_features function
- add_price_action_features function
- add_time_features function
- generate_enhanced_features function
- combine_with_base_features function
"""

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.rl.enhanced_features import (
    add_volume_profile_features,
    add_advanced_momentum_features,
    add_regime_features,
    add_price_action_features,
    add_time_features,
    generate_enhanced_features,
    combine_with_base_features,
)


NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing with proper OHLCV semantics."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:30:00", periods=500, freq="1min", tz=NY_TZ)

    base_price = 4800.0
    returns = np.random.normal(0, 0.0002, 500)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate open prices (previous close)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Ensure proper OHLCV semantics: high >= max(open,close), low <= min(open,close)
    high_extra = np.random.uniform(0.0001, 0.0005, 500)
    low_extra = np.random.uniform(0.0001, 0.0005, 500)

    high_prices = np.maximum(open_prices, close_prices) * (1 + high_extra)
    low_prices = np.minimum(open_prices, close_prices) * (1 - low_extra)

    df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": np.random.randint(100, 1000, 500),
    }, index=dates)

    return df


class TestAddVolumeProfileFeatures:
    """Tests for add_volume_profile_features function."""

    def test_returns_dataframe(self, sample_ohlcv_data):
        """Function returns a DataFrame."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_adds_vwap(self, sample_ohlcv_data):
        """Function adds VWAP column."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert "vwap" in result.columns

    def test_adds_vwap_deviation(self, sample_ohlcv_data):
        """Function adds VWAP deviation column."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert "vwap_deviation" in result.columns

    def test_adds_volume_ratio(self, sample_ohlcv_data):
        """Function adds volume ratio column."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert "volume_ratio" in result.columns

    def test_adds_volume_trend(self, sample_ohlcv_data):
        """Function adds volume trend column."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert "volume_trend" in result.columns

    def test_adds_obv(self, sample_ohlcv_data):
        """Function adds OBV columns."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert "obv" in result.columns
        assert "obv_sma" in result.columns
        assert "obv_signal" in result.columns

    def test_adds_mfi(self, sample_ohlcv_data):
        """Function adds Money Flow Index column."""
        result = add_volume_profile_features(sample_ohlcv_data)
        assert "mfi" in result.columns

    def test_mfi_normalized(self, sample_ohlcv_data):
        """MFI is normalized to approximately [-0.5, 0.5]."""
        result = add_volume_profile_features(sample_ohlcv_data)
        valid_mfi = result["mfi"].dropna()
        # Allow for some numerical variation
        assert valid_mfi.min() >= -0.6
        assert valid_mfi.max() <= 0.6

    def test_does_not_modify_original(self, sample_ohlcv_data):
        """Function returns copy, doesn't modify original."""
        original_cols = set(sample_ohlcv_data.columns)
        result = add_volume_profile_features(sample_ohlcv_data)
        assert set(sample_ohlcv_data.columns) == original_cols


class TestAddAdvancedMomentumFeatures:
    """Tests for add_advanced_momentum_features function."""

    def test_returns_dataframe(self, sample_ohlcv_data):
        """Function returns a DataFrame."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_adds_roc_features(self, sample_ohlcv_data):
        """Function adds Rate of Change features."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert "roc_5" in result.columns
        assert "roc_10" in result.columns
        assert "roc_20" in result.columns
        assert "roc_60" in result.columns

    def test_adds_williams_r(self, sample_ohlcv_data):
        """Function adds Williams %R features."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert "williams_r_14" in result.columns
        assert "williams_r_28" in result.columns

    def test_williams_r_centered(self, sample_ohlcv_data):
        """Williams %R is centered at 0."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        valid_wr = result["williams_r_14"].dropna()
        # Centered at 0 means range is approximately [-0.5, 0.5]
        assert valid_wr.min() >= -0.6
        assert valid_wr.max() <= 0.6

    def test_adds_cci(self, sample_ohlcv_data):
        """Function adds Commodity Channel Index."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert "cci" in result.columns

    def test_adds_stoch_rsi(self, sample_ohlcv_data):
        """Function adds Stochastic RSI."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert "stoch_rsi" in result.columns

    def test_stoch_rsi_centered(self, sample_ohlcv_data):
        """Stochastic RSI is centered at 0."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        valid_srsi = result["stoch_rsi"].dropna()
        assert valid_srsi.min() >= -0.6
        assert valid_srsi.max() <= 0.6

    def test_adds_macd_hist_slope(self, sample_ohlcv_data):
        """Function adds MACD histogram slope."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert "macd_hist_slope" in result.columns

    def test_adds_momentum_divergence(self, sample_ohlcv_data):
        """Function adds momentum divergence feature."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        assert "momentum_divergence" in result.columns

    def test_momentum_divergence_binary(self, sample_ohlcv_data):
        """Momentum divergence is 0 or 1."""
        result = add_advanced_momentum_features(sample_ohlcv_data)
        valid_div = result["momentum_divergence"].dropna()
        assert set(valid_div.unique()).issubset({0.0, 1.0})


class TestAddRegimeFeatures:
    """Tests for add_regime_features function."""

    def test_returns_dataframe(self, sample_ohlcv_data):
        """Function returns a DataFrame."""
        result = add_regime_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_adds_atr_percentile(self, sample_ohlcv_data):
        """Function adds ATR percentile feature."""
        result = add_regime_features(sample_ohlcv_data)
        assert "atr_percentile" in result.columns

    def test_adds_adx(self, sample_ohlcv_data):
        """Function adds ADX feature."""
        result = add_regime_features(sample_ohlcv_data)
        assert "adx" in result.columns

    def test_adx_normalized(self, sample_ohlcv_data):
        """ADX is normalized to [0, 1]."""
        result = add_regime_features(sample_ohlcv_data)
        valid_adx = result["adx"].dropna()
        assert valid_adx.min() >= 0
        assert valid_adx.max() <= 1.5  # Allow some tolerance

    def test_adds_trend_direction(self, sample_ohlcv_data):
        """Function adds trend direction feature."""
        result = add_regime_features(sample_ohlcv_data)
        assert "trend_direction" in result.columns

    def test_trend_direction_values(self, sample_ohlcv_data):
        """Trend direction is -1, 0, or 1."""
        result = add_regime_features(sample_ohlcv_data)
        valid_td = result["trend_direction"].dropna()
        assert set(valid_td.unique()).issubset({-1.0, 0.0, 1.0})

    def test_adds_bb_width(self, sample_ohlcv_data):
        """Function adds Bollinger Band width features."""
        result = add_regime_features(sample_ohlcv_data)
        assert "bb_width" in result.columns
        assert "bb_width_change" in result.columns

    def test_adds_regime_autocorr(self, sample_ohlcv_data):
        """Function adds regime autocorrelation feature."""
        result = add_regime_features(sample_ohlcv_data)
        assert "regime_autocorr" in result.columns

    def test_adds_hurst(self, sample_ohlcv_data):
        """Function adds Hurst exponent approximation."""
        result = add_regime_features(sample_ohlcv_data)
        assert "hurst" in result.columns


class TestAddPriceActionFeatures:
    """Tests for add_price_action_features function."""

    def test_returns_dataframe(self, sample_ohlcv_data):
        """Function returns a DataFrame."""
        result = add_price_action_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_adds_body_ratio(self, sample_ohlcv_data):
        """Function adds candle body ratio."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "body_ratio" in result.columns

    def test_body_ratio_bounded(self, sample_ohlcv_data):
        """Body ratio is bounded between -1 and 1."""
        result = add_price_action_features(sample_ohlcv_data)
        valid_br = result["body_ratio"].dropna()
        assert valid_br.min() >= -1.1
        assert valid_br.max() <= 1.1

    def test_adds_wick_ratios(self, sample_ohlcv_data):
        """Function adds upper and lower wick ratios."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "upper_wick_ratio" in result.columns
        assert "lower_wick_ratio" in result.columns

    def test_wick_ratios_non_negative(self, sample_ohlcv_data):
        """Wick ratios are non-negative."""
        result = add_price_action_features(sample_ohlcv_data)
        valid_upper = result["upper_wick_ratio"].dropna()
        valid_lower = result["lower_wick_ratio"].dropna()
        assert (valid_upper >= 0).all()
        assert (valid_lower >= 0).all()

    def test_adds_support_resistance(self, sample_ohlcv_data):
        """Function adds support/resistance features."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "resistance_20" in result.columns
        assert "support_20" in result.columns
        assert "resistance_60" in result.columns
        assert "support_60" in result.columns

    def test_adds_dist_to_levels(self, sample_ohlcv_data):
        """Function adds distance to support/resistance."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "dist_to_resistance_20" in result.columns
        assert "dist_to_support_20" in result.columns
        assert "dist_to_resistance_60" in result.columns
        assert "dist_to_support_60" in result.columns

    def test_adds_breakout_signals(self, sample_ohlcv_data):
        """Function adds breakout detection signals."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "breakout_up" in result.columns
        assert "breakout_down" in result.columns

    def test_breakout_binary(self, sample_ohlcv_data):
        """Breakout signals are 0 or 1."""
        result = add_price_action_features(sample_ohlcv_data)
        valid_up = result["breakout_up"].dropna()
        valid_down = result["breakout_down"].dropna()
        assert set(valid_up.unique()).issubset({0.0, 1.0})
        assert set(valid_down.unique()).issubset({0.0, 1.0})

    def test_adds_gap_features(self, sample_ohlcv_data):
        """Function adds gap detection features."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "gap" in result.columns
        assert "gap_filled" in result.columns

    def test_adds_inside_outside_bars(self, sample_ohlcv_data):
        """Function adds inside/outside bar detection."""
        result = add_price_action_features(sample_ohlcv_data)
        assert "inside_bar" in result.columns
        assert "outside_bar" in result.columns


class TestAddTimeFeatures:
    """Tests for add_time_features function."""

    def test_returns_dataframe(self, sample_ohlcv_data):
        """Function returns a DataFrame."""
        result = add_time_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)

    def test_adds_time_cyclical(self, sample_ohlcv_data):
        """Function adds cyclical time encoding."""
        result = add_time_features(sample_ohlcv_data)
        assert "time_sin" in result.columns
        assert "time_cos" in result.columns

    def test_time_sin_cos_bounded(self, sample_ohlcv_data):
        """Sine and cosine time features are bounded [-1, 1]."""
        result = add_time_features(sample_ohlcv_data)
        assert result["time_sin"].min() >= -1
        assert result["time_sin"].max() <= 1
        assert result["time_cos"].min() >= -1
        assert result["time_cos"].max() <= 1

    def test_adds_session_periods(self, sample_ohlcv_data):
        """Function adds session period flags."""
        result = add_time_features(sample_ohlcv_data)
        assert "opening_30min" in result.columns
        assert "closing_30min" in result.columns
        assert "lunch_hour" in result.columns

    def test_session_periods_binary(self, sample_ohlcv_data):
        """Session period flags are 0 or 1."""
        result = add_time_features(sample_ohlcv_data)
        assert set(result["opening_30min"].unique()).issubset({0.0, 1.0})
        assert set(result["closing_30min"].unique()).issubset({0.0, 1.0})
        assert set(result["lunch_hour"].unique()).issubset({0.0, 1.0})

    def test_adds_dow_cyclical(self, sample_ohlcv_data):
        """Function adds cyclical day-of-week encoding."""
        result = add_time_features(sample_ohlcv_data)
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_adds_day_effects(self, sample_ohlcv_data):
        """Function adds Monday/Friday effect flags."""
        result = add_time_features(sample_ohlcv_data)
        assert "is_monday" in result.columns
        assert "is_friday" in result.columns

    def test_handles_non_datetime_index(self):
        """Function handles non-datetime index gracefully."""
        df = pd.DataFrame({
            "open": [100] * 5,
            "high": [101] * 5,
            "low": [99] * 5,
            "close": [100] * 5,
            "volume": [100] * 5,
        })
        result = add_time_features(df)
        # Should return without adding time features
        assert "time_sin" not in result.columns


class TestGenerateEnhancedFeatures:
    """Tests for generate_enhanced_features function."""

    def test_returns_tuple(self, sample_ohlcv_data):
        """Function returns (DataFrame, feature_list) tuple."""
        result = generate_enhanced_features(sample_ohlcv_data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_feature_list_not_empty(self, sample_ohlcv_data):
        """Feature list contains items."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        assert len(feature_cols) > 0

    def test_all_feature_cols_exist(self, sample_ohlcv_data):
        """All listed feature columns exist in DataFrame."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        for col in feature_cols:
            assert col in df.columns, f"Feature {col} not in DataFrame"

    def test_includes_volume_features(self, sample_ohlcv_data):
        """Generated features include volume profile features."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        assert "vwap_deviation" in feature_cols
        assert "volume_ratio" in feature_cols

    def test_includes_momentum_features(self, sample_ohlcv_data):
        """Generated features include momentum features."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        assert "roc_5" in feature_cols
        assert "stoch_rsi" in feature_cols

    def test_includes_regime_features(self, sample_ohlcv_data):
        """Generated features include regime features."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        assert "adx" in feature_cols
        assert "bb_width" in feature_cols

    def test_includes_price_action_features(self, sample_ohlcv_data):
        """Generated features include price action features."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        assert "body_ratio" in feature_cols
        assert "breakout_up" in feature_cols

    def test_includes_time_features(self, sample_ohlcv_data):
        """Generated features include time features."""
        df, feature_cols = generate_enhanced_features(sample_ohlcv_data)
        assert "time_sin" in feature_cols
        assert "dow_cos" in feature_cols


class TestCombineWithBaseFeatures:
    """Tests for combine_with_base_features function."""

    def test_returns_tuple(self, sample_ohlcv_data):
        """Function returns (DataFrame, feature_list) tuple."""
        base_cols = ["close", "volume"]
        result = combine_with_base_features(sample_ohlcv_data, base_cols)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_includes_base_features(self, sample_ohlcv_data):
        """Combined features include base features."""
        base_cols = ["close", "volume"]
        df, all_features = combine_with_base_features(sample_ohlcv_data, base_cols)
        assert "close" in all_features
        assert "volume" in all_features

    def test_includes_enhanced_features(self, sample_ohlcv_data):
        """Combined features include enhanced features."""
        base_cols = ["close", "volume"]
        df, all_features = combine_with_base_features(sample_ohlcv_data, base_cols)
        # Check some enhanced features are included
        assert "vwap_deviation" in all_features
        assert "roc_5" in all_features

    def test_no_duplicate_features(self, sample_ohlcv_data):
        """Combined features have no duplicates."""
        base_cols = ["close", "volume", "volume_ratio"]  # volume_ratio might be in enhanced
        df, all_features = combine_with_base_features(sample_ohlcv_data, base_cols)
        assert len(all_features) == len(set(all_features))

    def test_empty_base_features(self, sample_ohlcv_data):
        """Function works with empty base feature list."""
        df, all_features = combine_with_base_features(sample_ohlcv_data, [])
        assert len(all_features) > 0  # Should still have enhanced features


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self):
        """Functions handle small datasets (may have NaN)."""
        dates = pd.date_range("2024-01-02 09:30:00", periods=20, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [100] * 20,
        }, index=dates)

        # These should not raise errors
        result1 = add_volume_profile_features(df)
        result2 = add_advanced_momentum_features(df)
        result3 = add_time_features(df)

        assert len(result1) == 20
        assert len(result2) == 20
        assert len(result3) == 20

    def test_constant_prices(self):
        """Functions handle constant prices gracefully."""
        dates = pd.date_range("2024-01-02 09:30:00", periods=100, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [100.0] * 100,
            "high": [100.0] * 100,
            "low": [100.0] * 100,
            "close": [100.0] * 100,
            "volume": [100] * 100,
        }, index=dates)

        # Should not raise errors (may produce NaN/inf)
        result = add_volume_profile_features(df)
        assert len(result) == 100

    def test_zero_volume(self):
        """Functions handle zero volume gracefully."""
        dates = pd.date_range("2024-01-02 09:30:00", periods=50, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.0] * 50,
            "volume": [0] * 50,  # Zero volume
        }, index=dates)

        # Should not raise errors
        result = add_volume_profile_features(df)
        assert len(result) == 50
