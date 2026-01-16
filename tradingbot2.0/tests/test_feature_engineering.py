"""
Tests for FeatureEngineer and feature engineering utilities.

Tests cover:
- Return calculations (simple and log returns)
- Moving averages (SMA, EMA, crossovers)
- Volatility features (ATR, Bollinger Bands, realized volatility)
- Momentum indicators (RSI, MACD, Stochastic)
- Volume features (volume ratio, OBV, VPT)
- Candlestick features (body, wicks, gaps)
- Time features (day of week, month)
- Feature normalization
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))

from data.feature_engineering import FeatureEngineer, prepare_features_for_training


class TestFeatureEngineerInit:
    """Tests for FeatureEngineer initialization."""

    def test_init_copies_dataframe(self, sample_daily_ohlcv):
        """Test that initialization creates a copy of the DataFrame."""
        engineer = FeatureEngineer(sample_daily_ohlcv)

        # Modify original
        original_close = sample_daily_ohlcv.iloc[0]['close']
        sample_daily_ohlcv.iloc[0, sample_daily_ohlcv.columns.get_loc('close')] = 99999

        # Engineer's copy should be unchanged
        assert engineer.df.iloc[0]['close'] == original_close

    def test_init_empty_feature_names(self, sample_daily_ohlcv):
        """Test that feature_names list is initially empty."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        assert engineer.feature_names == []


class TestReturnFeatures:
    """Tests for return feature calculations."""

    def test_add_returns_creates_features(self, sample_daily_ohlcv):
        """Test that add_returns creates the expected features."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_returns(periods=[1, 5])

        assert 'return_1d' in engineer.df.columns
        assert 'return_5d' in engineer.df.columns
        assert 'log_return_1d' in engineer.df.columns
        assert 'log_return_5d' in engineer.df.columns

    def test_returns_calculation_correctness(self, sample_daily_ohlcv):
        """Test that return calculations are correct."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_returns(periods=[1])

        # Manually calculate expected 1-day return
        for i in range(1, min(10, len(engineer.df))):
            expected = (sample_daily_ohlcv.iloc[i]['close'] - sample_daily_ohlcv.iloc[i-1]['close']) / sample_daily_ohlcv.iloc[i-1]['close']
            actual = engineer.df.iloc[i]['return_1d']
            np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_log_returns_calculation(self, sample_daily_ohlcv):
        """Test that log return calculations are correct."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_returns(periods=[1])

        # Log return should be log(current / previous)
        for i in range(1, min(10, len(engineer.df))):
            expected = np.log(sample_daily_ohlcv.iloc[i]['close'] / sample_daily_ohlcv.iloc[i-1]['close'])
            actual = engineer.df.iloc[i]['log_return_1d']
            np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_returns_method_chaining(self, sample_daily_ohlcv):
        """Test that add_returns supports method chaining."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        result = engineer.add_returns()

        assert result is engineer

    def test_returns_updates_feature_names(self, sample_daily_ohlcv):
        """Test that add_returns updates feature_names list."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_returns(periods=[1, 5])

        assert 'return_1d' in engineer.feature_names
        assert 'log_return_1d' in engineer.feature_names


class TestMovingAverageFeatures:
    """Tests for moving average features."""

    def test_sma_calculation(self, sample_daily_ohlcv):
        """Test SMA calculation correctness."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_moving_averages(sma_periods=[5], ema_periods=[])

        # Calculate expected SMA
        expected_sma_5 = sample_daily_ohlcv['close'].rolling(window=5).mean()

        # Compare (skip first 4 NaN values)
        for i in range(5, min(15, len(engineer.df))):
            np.testing.assert_almost_equal(
                engineer.df.iloc[i]['sma_5'],
                expected_sma_5.iloc[i],
                decimal=10
            )

    def test_ema_calculation(self, sample_daily_ohlcv):
        """Test EMA calculation correctness."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_moving_averages(sma_periods=[], ema_periods=[5])

        # Calculate expected EMA
        expected_ema_5 = sample_daily_ohlcv['close'].ewm(span=5, adjust=False).mean()

        for i in range(10, min(20, len(engineer.df))):
            np.testing.assert_almost_equal(
                engineer.df.iloc[i]['ema_5'],
                expected_ema_5.iloc[i],
                decimal=10
            )

    def test_close_to_sma_ratio(self, sample_daily_ohlcv):
        """Test close_to_sma ratio calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_moving_averages(sma_periods=[5], ema_periods=[])

        # close_to_sma = (close - sma) / sma
        for i in range(10, min(20, len(engineer.df))):
            expected = (engineer.df.iloc[i]['close'] - engineer.df.iloc[i]['sma_5']) / engineer.df.iloc[i]['sma_5']
            np.testing.assert_almost_equal(
                engineer.df.iloc[i]['close_to_sma_5'],
                expected,
                decimal=10
            )

    def test_ma_crossover_feature(self, sample_daily_ohlcv):
        """Test MA crossover feature creation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_moving_averages(sma_periods=[5, 20], ema_periods=[])

        assert 'sma_5_20_cross' in engineer.df.columns


class TestVolatilityFeatures:
    """Tests for volatility features."""

    def test_atr_calculation(self, sample_daily_ohlcv):
        """Test ATR calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volatility_features(atr_period=14)

        assert 'atr' in engineer.df.columns
        assert 'atr_pct' in engineer.df.columns

        # ATR should be positive
        valid_atr = engineer.df['atr'].dropna()
        assert (valid_atr > 0).all()

    def test_bollinger_bands(self, sample_daily_ohlcv):
        """Test Bollinger Band features."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volatility_features(bb_period=20, bb_std=2.0)

        assert 'bb_upper' in engineer.df.columns
        assert 'bb_lower' in engineer.df.columns
        assert 'bb_width' in engineer.df.columns
        assert 'bb_position' in engineer.df.columns

        # Upper band should be above lower band
        valid_idx = engineer.df['bb_upper'].notna()
        assert (engineer.df.loc[valid_idx, 'bb_upper'] >= engineer.df.loc[valid_idx, 'bb_lower']).all()

    def test_bb_position_range(self, sample_daily_ohlcv):
        """Test that BB position is roughly in expected range."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volatility_features()

        valid_pos = engineer.df['bb_position'].dropna()
        # Most values should be between 0 and 1 (price within bands)
        within_range = ((valid_pos >= -0.5) & (valid_pos <= 1.5)).mean()
        assert within_range > 0.8  # At least 80% within reasonable range

    def test_realized_volatility(self, sample_daily_ohlcv):
        """Test realized volatility calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volatility_features(vol_windows=[5, 10])

        assert 'volatility_5d' in engineer.df.columns
        assert 'volatility_10d' in engineer.df.columns

        # Volatility should be non-negative
        valid_vol = engineer.df['volatility_5d'].dropna()
        assert (valid_vol >= 0).all()


class TestMomentumIndicators:
    """Tests for momentum indicators."""

    def test_rsi_calculation(self, sample_daily_ohlcv):
        """Test RSI calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_momentum_indicators(rsi_period=14)

        assert 'rsi' in engineer.df.columns
        assert 'rsi_normalized' in engineer.df.columns

        # RSI should be between 0 and 100
        valid_rsi = engineer.df['rsi'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_normalized_range(self, sample_daily_ohlcv):
        """Test that normalized RSI is in [-1, 1] range."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_momentum_indicators()

        valid_rsi = engineer.df['rsi_normalized'].dropna()
        assert (valid_rsi >= -1).all()
        assert (valid_rsi <= 1).all()

    def test_macd_features(self, sample_daily_ohlcv):
        """Test MACD feature creation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_momentum_indicators()

        assert 'macd' in engineer.df.columns
        assert 'macd_signal' in engineer.df.columns
        assert 'macd_histogram' in engineer.df.columns
        assert 'macd_normalized' in engineer.df.columns

    def test_stochastic_oscillator(self, sample_daily_ohlcv):
        """Test Stochastic Oscillator features."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_momentum_indicators(stoch_period=14)

        assert 'stoch_k' in engineer.df.columns
        assert 'stoch_d' in engineer.df.columns
        assert 'stoch_k_normalized' in engineer.df.columns

        # Stochastic should be between 0 and 100
        valid_k = engineer.df['stoch_k'].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


class TestVolumeFeatures:
    """Tests for volume features."""

    def test_volume_ratio(self, sample_daily_ohlcv):
        """Test volume ratio calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volume_features(periods=[5])

        assert 'volume_ratio_5d' in engineer.df.columns

        # Volume ratio should be positive
        valid_ratio = engineer.df['volume_ratio_5d'].dropna()
        assert (valid_ratio > 0).all()

    def test_obv_rate_of_change(self, sample_daily_ohlcv):
        """Test OBV rate of change calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volume_features()

        assert 'obv_roc' in engineer.df.columns

    def test_vpt_feature(self, sample_daily_ohlcv):
        """Test VPT feature creation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_volume_features()

        assert 'vpt' in engineer.df.columns
        assert 'vpt_roc' in engineer.df.columns


class TestCandlestickFeatures:
    """Tests for candlestick pattern features."""

    def test_body_percentage(self, sample_daily_ohlcv):
        """Test body percentage calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_candlestick_features()

        assert 'body_pct' in engineer.df.columns

        # Body pct should be reasonable (< 10% for most normal days)
        valid_body = engineer.df['body_pct'].dropna().abs()
        assert valid_body.median() < 0.05  # Median should be < 5%

    def test_wick_ratios(self, sample_daily_ohlcv):
        """Test wick ratio calculations."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_candlestick_features()

        assert 'upper_wick_pct' in engineer.df.columns
        assert 'lower_wick_pct' in engineer.df.columns

        # Wick percentages should be between 0 and 1
        valid_upper = engineer.df['upper_wick_pct'].dropna()
        valid_lower = engineer.df['lower_wick_pct'].dropna()

        assert (valid_upper >= 0).all()
        assert (valid_upper <= 1).all()
        assert (valid_lower >= 0).all()
        assert (valid_lower <= 1).all()

    def test_body_range_ratio(self, sample_daily_ohlcv):
        """Test body/range ratio."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_candlestick_features()

        assert 'body_range_ratio' in engineer.df.columns

        # Body/range ratio should be between 0 and 1
        valid_ratio = engineer.df['body_range_ratio'].dropna()
        assert (valid_ratio >= 0).all()
        assert (valid_ratio <= 1).all()

    def test_gap_percentage(self, sample_daily_ohlcv):
        """Test gap percentage calculation."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_candlestick_features()

        assert 'gap_pct' in engineer.df.columns


class TestTimeFeatures:
    """Tests for time-based features."""

    def test_day_of_week_encoding(self, sample_daily_ohlcv):
        """Test day of week cyclical encoding."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_time_features()

        assert 'day_of_week_sin' in engineer.df.columns
        assert 'day_of_week_cos' in engineer.df.columns

        # Sin/cos should be in [-1, 1]
        assert (engineer.df['day_of_week_sin'] >= -1).all()
        assert (engineer.df['day_of_week_sin'] <= 1).all()
        assert (engineer.df['day_of_week_cos'] >= -1).all()
        assert (engineer.df['day_of_week_cos'] <= 1).all()

    def test_month_encoding(self, sample_daily_ohlcv):
        """Test month cyclical encoding."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_time_features()

        assert 'month_sin' in engineer.df.columns
        assert 'month_cos' in engineer.df.columns


class TestGenerateAllFeatures:
    """Tests for generate_all_features method."""

    def test_generates_all_feature_categories(self, sample_daily_ohlcv):
        """Test that all feature categories are generated."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        df = engineer.generate_all_features()

        # Check representatives from each category
        assert any('return' in f for f in engineer.feature_names)
        assert any('sma' in f or 'ema' in f for f in engineer.feature_names)
        assert any('atr' in f or 'bb' in f or 'volatility' in f for f in engineer.feature_names)
        assert any('rsi' in f or 'macd' in f or 'stoch' in f for f in engineer.feature_names)
        assert any('volume' in f or 'obv' in f for f in engineer.feature_names)
        assert any('body' in f or 'wick' in f for f in engineer.feature_names)

    def test_drops_nan_rows(self, sample_daily_ohlcv):
        """Test that NaN rows are dropped."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        df = engineer.generate_all_features()

        # Should have fewer rows than original due to lookback windows
        assert len(df) < len(sample_daily_ohlcv)

        # No NaN in feature columns
        for feature in engineer.feature_names:
            assert not df[feature].isna().any()

    def test_feature_count(self, sample_daily_ohlcv):
        """Test that reasonable number of features are generated."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.generate_all_features()

        # Should have 30+ features
        assert len(engineer.feature_names) >= 30


class TestGetFeatureMethods:
    """Tests for feature accessor methods."""

    def test_get_feature_names_returns_copy(self, sample_daily_ohlcv):
        """Test that get_feature_names returns a copy."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.add_returns()

        names = engineer.get_feature_names()
        names.append('fake_feature')

        # Original should be unchanged
        assert 'fake_feature' not in engineer.feature_names

    def test_get_feature_matrix_shape(self, sample_daily_ohlcv):
        """Test feature matrix shape."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.generate_all_features()

        matrix = engineer.get_feature_matrix()
        assert matrix.shape[0] == len(engineer.df)
        assert matrix.shape[1] == len(engineer.feature_names)

    def test_get_feature_matrix_dtype(self, sample_daily_ohlcv):
        """Test feature matrix data type."""
        engineer = FeatureEngineer(sample_daily_ohlcv)
        engineer.generate_all_features()

        matrix = engineer.get_feature_matrix()
        assert matrix.dtype == np.float64


class TestPrepareFeatures:
    """Tests for prepare_features_for_training function."""

    def test_returns_three_items(self, sample_daily_ohlcv):
        """Test that function returns df, names, scaler."""
        df, names, scaler = prepare_features_for_training(sample_daily_ohlcv)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(names, list)
        assert scaler is not None

    def test_normalization_applied(self, sample_daily_ohlcv):
        """Test that features are normalized when requested."""
        df, names, scaler = prepare_features_for_training(sample_daily_ohlcv, normalize=True)

        # Normalized features should have mean ~0 and std ~1
        for feature in names[:5]:  # Check first 5 features
            mean = df[feature].mean()
            std = df[feature].std()
            assert abs(mean) < 0.5  # Should be close to 0
            assert 0.5 < std < 1.5  # Should be close to 1

    def test_no_normalization(self, sample_daily_ohlcv):
        """Test that normalization can be skipped."""
        df, names, scaler = prepare_features_for_training(sample_daily_ohlcv, normalize=False)

        assert scaler is None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_dataset(self):
        """Test with minimal data (just enough for lookback windows)."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-02', periods=60, freq='B')
        df = pd.DataFrame({
            'open': np.random.randn(60) + 5000,
            'high': np.random.randn(60) + 5001,
            'low': np.random.randn(60) + 4999,
            'close': np.random.randn(60) + 5000,
            'volume': np.random.randint(1000, 10000, 60)
        }, index=dates)

        # Ensure OHLC validity
        df['high'] = df[['open', 'high', 'close']].max(axis=1) + 1
        df['low'] = df[['open', 'low', 'close']].min(axis=1) - 1

        engineer = FeatureEngineer(df)
        result = engineer.generate_all_features()

        assert len(result) > 0

    def test_zero_volume_handling(self, sample_daily_ohlcv):
        """Test handling of zero volume."""
        df = sample_daily_ohlcv.copy()
        df.iloc[50:55, df.columns.get_loc('volume')] = 0

        engineer = FeatureEngineer(df)
        result = engineer.generate_all_features()

        # Should complete without errors
        assert result is not None

    def test_constant_price(self):
        """Test with constant prices (edge case for many indicators)."""
        dates = pd.date_range(start='2024-01-02', periods=100, freq='B')
        df = pd.DataFrame({
            'open': [5000.0] * 100,
            'high': [5001.0] * 100,
            'low': [4999.0] * 100,
            'close': [5000.0] * 100,
            'volume': [1000] * 100
        }, index=dates)

        engineer = FeatureEngineer(df)
        # Should not raise errors
        engineer.add_returns()
        engineer.add_moving_averages()

        # Returns should be 0 for constant prices
        valid_returns = engineer.df['return_1d'].dropna()
        assert (valid_returns == 0).all()


class TestMethodChaining:
    """Tests for method chaining."""

    def test_all_methods_return_self(self, sample_daily_ohlcv):
        """Test that all methods support chaining."""
        engineer = FeatureEngineer(sample_daily_ohlcv)

        result = (engineer
            .add_returns()
            .add_moving_averages()
            .add_volatility_features()
            .add_momentum_indicators()
            .add_volume_features()
            .add_candlestick_features()
            .add_time_features())

        assert result is engineer
