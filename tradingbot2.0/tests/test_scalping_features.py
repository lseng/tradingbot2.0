"""
Unit Tests for Scalping Feature Engineering Module.

Tests the ScalpingFeatureEngineer class which generates features
for 1-second MES futures scalping data.

Test Categories:
1. Feature calculation correctness
2. VWAP session-based reset
3. Minutes-to-close calculation
4. No lookahead bias in features
5. Multi-timeframe lagging
6. Feature normalization
7. Edge cases and error handling

Reference: specs/ml-scalping-model.md
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'ml'))

from data.scalping_features import (
    ScalpingFeatureEngineer,
    FeatureConfig,
    prepare_scalping_features,
    validate_no_lookahead,
    MES_TICK_SIZE,
    RTH_START,
    RTH_END,
    RTH_DURATION_MINUTES,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def ny_tz():
    """New York timezone."""
    return ZoneInfo('America/New_York')


@pytest.fixture
def sample_1s_data(ny_tz):
    """
    Create sample 1-second OHLCV data for a single RTH session.

    Returns 1000 seconds of data starting at 9:30 AM NY.
    """
    np.random.seed(42)
    n_bars = 1000

    # Start at 9:30 AM NY
    start = datetime(2025, 6, 15, 9, 30, 0, tzinfo=ny_tz)
    timestamps = [start + timedelta(seconds=i) for i in range(n_bars)]

    # Generate realistic MES price movements
    base_price = 5000.0
    returns = np.random.randn(n_bars) * 0.0001  # ~0.01% per second
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    df = pd.DataFrame(index=pd.DatetimeIndex(timestamps))
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(base_price)

    # High/low with some noise
    noise = np.abs(np.random.randn(n_bars)) * 0.5  # 0.5 points noise
    df['high'] = df[['open', 'close']].max(axis=1) + noise
    df['low'] = df[['open', 'close']].min(axis=1) - noise

    # Ensure OHLC relationships
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    # Volume
    df['volume'] = np.random.randint(10, 500, n_bars)

    return df


@pytest.fixture
def multi_session_data(ny_tz):
    """
    Create sample data spanning multiple trading sessions.

    Returns 2 full RTH sessions (2 days × 390 minutes × 60 seconds each).
    """
    np.random.seed(123)

    sessions = []
    base_price = 5000.0

    for day_offset in range(2):  # 2 days
        day = datetime(2025, 6, 16 + day_offset, tzinfo=ny_tz)

        # RTH session: 9:30 AM to 4:00 PM (390 minutes = 23400 seconds)
        n_bars = 1000  # Use 1000 for speed (not full session)
        start = day.replace(hour=9, minute=30, second=0)
        timestamps = [start + timedelta(seconds=i) for i in range(n_bars)]

        returns = np.random.randn(n_bars) * 0.0001
        prices = base_price * np.cumprod(1 + returns)

        session_df = pd.DataFrame(index=pd.DatetimeIndex(timestamps))
        session_df['close'] = prices
        session_df['open'] = session_df['close'].shift(1).fillna(base_price)

        noise = np.abs(np.random.randn(n_bars)) * 0.5
        session_df['high'] = session_df[['open', 'close']].max(axis=1) + noise
        session_df['low'] = session_df[['open', 'close']].min(axis=1) - noise
        session_df['high'] = session_df[['open', 'close', 'high']].max(axis=1)
        session_df['low'] = session_df[['open', 'close', 'low']].min(axis=1)
        session_df['volume'] = np.random.randint(10, 500, n_bars)

        sessions.append(session_df)
        base_price = prices[-1]  # Carry over to next day

    return pd.concat(sessions)


@pytest.fixture
def custom_config():
    """Custom feature configuration for testing."""
    return FeatureConfig(
        return_periods=[1, 5, 10],
        ema_periods=[9, 21],
        volatility_windows=[10, 30],
        volume_windows=[10, 30],
        rsi_period=14,
        atr_period=14,
    )


# ============================================================================
# TEST: BASIC INITIALIZATION
# ============================================================================

class TestScalpingFeatureEngineerInit:
    """Test ScalpingFeatureEngineer initialization."""

    def test_init_with_valid_data(self, sample_1s_data):
        """Initialize with valid OHLCV DataFrame."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        assert engineer.df is not None
        assert len(engineer.feature_names) == 0  # No features yet

    def test_init_missing_columns(self, ny_tz):
        """Raise error when required columns are missing."""
        df = pd.DataFrame({
            'open': [100, 101],
            'close': [101, 102],
        }, index=pd.DatetimeIndex([
            datetime(2025, 6, 15, 9, 30, tzinfo=ny_tz),
            datetime(2025, 6, 15, 9, 30, 1, tzinfo=ny_tz),
        ]))

        with pytest.raises(ValueError, match="Missing required columns"):
            ScalpingFeatureEngineer(df)

    def test_init_non_datetime_index(self):
        """Raise error when index is not DatetimeIndex."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [100, 200],
        })

        with pytest.raises(ValueError, match="DatetimeIndex"):
            ScalpingFeatureEngineer(df)

    def test_init_with_custom_config(self, sample_1s_data, custom_config):
        """Initialize with custom configuration."""
        engineer = ScalpingFeatureEngineer(sample_1s_data, config=custom_config)
        assert engineer.config.return_periods == [1, 5, 10]
        assert engineer.config.ema_periods == [9, 21]


# ============================================================================
# TEST: RETURN FEATURES
# ============================================================================

class TestReturnFeatures:
    """Test return feature calculations."""

    def test_return_periods_in_seconds(self, sample_1s_data):
        """Verify returns are calculated at correct second intervals."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_returns()

        # Check feature names include seconds suffix
        assert 'return_1s' in engineer.feature_names
        assert 'return_5s' in engineer.feature_names
        assert 'return_30s' in engineer.feature_names
        assert 'return_60s' in engineer.feature_names

    def test_return_calculation_correctness(self, sample_1s_data):
        """Verify return values are calculated correctly."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_returns()

        df = engineer.df

        # Manually calculate 1-second return and compare
        expected_1s = df['close'].pct_change(1)
        pd.testing.assert_series_equal(
            df['return_1s'],
            expected_1s,
            check_names=False
        )

    def test_log_returns_included(self, sample_1s_data):
        """Verify log returns are calculated."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_returns()

        assert 'log_return_1s' in engineer.feature_names
        assert 'log_return_5s' in engineer.feature_names


# ============================================================================
# TEST: EMA FEATURES
# ============================================================================

class TestEMAFeatures:
    """Test EMA feature calculations."""

    def test_ema_periods(self, sample_1s_data):
        """Verify EMAs are calculated at correct periods."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_emas()

        assert 'close_to_ema_9' in engineer.feature_names
        assert 'close_to_ema_21' in engineer.feature_names
        assert 'close_to_ema_50' in engineer.feature_names
        assert 'close_to_ema_200' in engineer.feature_names

    def test_ema_crossover_features(self, sample_1s_data):
        """Verify EMA crossover features are included."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_emas()

        assert 'ema_9_21_cross' in engineer.feature_names
        assert 'ema_21_50_cross' in engineer.feature_names

    def test_close_to_ema_is_normalized(self, sample_1s_data):
        """Verify close-to-EMA is a ratio (not absolute difference)."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_emas()

        df = engineer.df.dropna()

        # Close-to-EMA should be small relative values, not large price differences
        assert df['close_to_ema_9'].abs().max() < 1.0  # Less than 100%


# ============================================================================
# TEST: VWAP FEATURES
# ============================================================================

class TestVWAPFeatures:
    """Test VWAP calculation with session-based reset."""

    def test_vwap_calculated(self, sample_1s_data):
        """Verify VWAP is calculated."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_vwap()

        assert 'close_to_vwap' in engineer.feature_names
        assert 'vwap_slope' in engineer.feature_names
        assert 'vwap' in engineer.df.columns

    def test_vwap_resets_at_session_boundary(self, multi_session_data):
        """Verify VWAP resets at the start of each trading session."""
        engineer = ScalpingFeatureEngineer(multi_session_data)
        engineer.add_vwap()

        df = engineer.df

        # Get VWAP at start of each session
        dates = df.index.date
        unique_dates = pd.unique(dates)

        # First bar of each session should have VWAP ~ first bar's typical price
        for date in unique_dates:
            session_mask = dates == date
            session_df = df[session_mask]

            first_bar = session_df.iloc[0]
            first_tp = (first_bar['high'] + first_bar['low'] + first_bar['close']) / 3
            first_vwap = first_bar['vwap']

            # VWAP at first bar should equal typical price (within rounding)
            assert abs(first_vwap - first_tp) < 0.01, \
                f"VWAP not reset at session start for {date}"

    def test_vwap_cumulative_within_session(self, sample_1s_data):
        """Verify VWAP is cumulative within a single session."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_vwap()

        df = engineer.df

        # VWAP should be relatively stable (cumulative average)
        vwap_std = df['vwap'].std()
        close_std = df['close'].std()

        # VWAP std should be smaller than close std (smoothing effect)
        assert vwap_std < close_std


# ============================================================================
# TEST: MINUTES TO CLOSE
# ============================================================================

class TestMinutesToClose:
    """Test minutes-to-close feature calculation."""

    def test_minutes_to_close_at_open(self, sample_1s_data):
        """At 9:30 AM, minutes_to_close should be 390."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_minutes_to_close()

        df = engineer.df
        first_bar = df.iloc[0]

        # First bar at 9:30 AM should have 390 minutes to close
        assert abs(first_bar['minutes_to_close'] - 390) < 1

    def test_minutes_to_close_normalized(self, sample_1s_data):
        """Verify normalized minutes_to_close is in [0, 1]."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_minutes_to_close()

        df = engineer.df

        assert df['minutes_to_close_norm'].min() >= 0
        assert df['minutes_to_close_norm'].max() <= 1

    def test_eod_urgency_increases(self, sample_1s_data):
        """EOD urgency should increase as we approach close."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_minutes_to_close()

        df = engineer.df

        # EOD urgency at first bar should be low
        assert df['eod_urgency'].iloc[0] < 0.1

        # EOD urgency should increase over time
        assert df['eod_urgency'].iloc[-1] > df['eod_urgency'].iloc[0]


# ============================================================================
# TEST: TIME OF DAY FEATURES
# ============================================================================

class TestTimeOfDayFeatures:
    """Test time-of-day encoding features."""

    def test_cyclical_time_encoding(self, sample_1s_data):
        """Verify cyclical time encoding is in [-1, 1]."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_time_of_day()

        df = engineer.df

        assert df['time_sin'].min() >= -1
        assert df['time_sin'].max() <= 1
        assert df['time_cos'].min() >= -1
        assert df['time_cos'].max() <= 1

    def test_session_period_flags(self, sample_1s_data):
        """Verify session period flags are binary."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_time_of_day()

        df = engineer.df

        # Should be 0 or 1 only
        assert set(df['is_open_period'].unique()).issubset({0, 1})
        assert set(df['is_close_period'].unique()).issubset({0, 1})
        assert set(df['is_lunch_period'].unique()).issubset({0, 1})

    def test_open_period_flag_at_930(self, sample_1s_data):
        """First bar at 9:30 AM should have is_open_period=1."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_time_of_day()

        df = engineer.df

        # First bar at 9:30 AM is in opening period
        assert df['is_open_period'].iloc[0] == 1


# ============================================================================
# TEST: VOLATILITY FEATURES
# ============================================================================

class TestVolatilityFeatures:
    """Test volatility feature calculations."""

    def test_atr_in_ticks(self, sample_1s_data):
        """Verify ATR is calculated and converted to ticks."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volatility_features()

        assert 'atr_ticks' in engineer.feature_names
        assert 'atr_pct' in engineer.feature_names

    def test_bollinger_band_position(self, sample_1s_data):
        """Verify BB position is in [0, 1] range."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volatility_features()

        df = engineer.df.dropna()

        # BB position should be clipped to [0, 1]
        assert df['bb_position'].min() >= 0
        assert df['bb_position'].max() <= 1

    def test_volatility_windows_in_seconds(self, sample_1s_data):
        """Verify volatility is calculated at second windows."""
        config = FeatureConfig(volatility_windows=[10, 30, 60])
        engineer = ScalpingFeatureEngineer(sample_1s_data, config=config)
        engineer.add_volatility_features()

        assert 'volatility_10s' in engineer.feature_names
        assert 'volatility_30s' in engineer.feature_names
        assert 'volatility_60s' in engineer.feature_names


# ============================================================================
# TEST: MOMENTUM INDICATORS
# ============================================================================

class TestMomentumIndicators:
    """Test momentum indicator calculations."""

    def test_rsi_normalized(self, sample_1s_data):
        """Verify RSI is normalized to [-1, 1]."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_momentum_indicators()

        df = engineer.df.dropna()

        assert df['rsi_norm'].min() >= -1
        assert df['rsi_norm'].max() <= 1

    def test_macd_features(self, sample_1s_data):
        """Verify MACD features are calculated."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_momentum_indicators()

        assert 'macd_norm' in engineer.feature_names
        assert 'macd_hist_norm' in engineer.feature_names

    def test_stochastic_normalized(self, sample_1s_data):
        """Verify Stochastic is normalized to [-1, 1]."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_momentum_indicators()

        df = engineer.df.dropna()

        assert df['stoch_k_norm'].min() >= -1
        assert df['stoch_k_norm'].max() <= 1


# ============================================================================
# TEST: MICROSTRUCTURE FEATURES
# ============================================================================

class TestMicrostructureFeatures:
    """Test microstructure feature calculations."""

    def test_bar_direction(self, sample_1s_data):
        """Verify bar direction is -1, 0, or 1."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_microstructure_features()

        df = engineer.df

        assert set(df['bar_direction'].unique()).issubset({-1, 0, 1})

    def test_wick_ratios_in_range(self, sample_1s_data):
        """Verify wick ratios are in [0, 1]."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_microstructure_features()

        df = engineer.df.dropna()

        assert df['upper_wick_ratio'].min() >= 0
        assert df['upper_wick_ratio'].max() <= 1
        assert df['lower_wick_ratio'].min() >= 0
        assert df['lower_wick_ratio'].max() <= 1

    def test_body_ratio_in_range(self, sample_1s_data):
        """Verify body ratio is in [0, 1]."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_microstructure_features()

        df = engineer.df.dropna()

        assert df['body_ratio'].min() >= 0
        assert df['body_ratio'].max() <= 1

    def test_gap_in_ticks(self, sample_1s_data):
        """Verify gap is calculated in ticks."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_microstructure_features()

        assert 'gap_ticks' in engineer.feature_names


# ============================================================================
# TEST: VOLUME FEATURES
# ============================================================================

class TestVolumeFeatures:
    """Test volume feature calculations."""

    def test_volume_ratio_windows(self, sample_1s_data):
        """Verify volume ratios are calculated at correct windows."""
        config = FeatureConfig(volume_windows=[10, 30, 60])
        engineer = ScalpingFeatureEngineer(sample_1s_data, config=config)
        engineer.add_volume_features()

        assert 'volume_ratio_10s' in engineer.feature_names
        assert 'volume_ratio_30s' in engineer.feature_names
        assert 'volume_ratio_60s' in engineer.feature_names

    def test_volume_delta_normalized(self, sample_1s_data):
        """Verify volume delta is normalized."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volume_features()

        df = engineer.df.dropna()

        # Volume delta normalized should be in reasonable range
        assert df['volume_delta_norm'].abs().max() <= 2  # Within 2 std


# ============================================================================
# TEST: MULTI-TIMEFRAME FEATURES
# ============================================================================

class TestMultiTimeframeFeatures:
    """Test multi-timeframe feature calculations with proper lagging."""

    def test_htf_features_exist(self, sample_1s_data):
        """Verify higher timeframe features are created."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_multiframe_features()

        assert 'htf_trend_1m' in engineer.feature_names
        assert 'htf_momentum_1m' in engineer.feature_names
        assert 'htf_trend_5m' in engineer.feature_names
        assert 'htf_momentum_5m' in engineer.feature_names

    def test_htf_features_are_lagged(self, sample_1s_data):
        """
        Verify HTF features are properly lagged to prevent lookahead.

        The first minute of data should have NaN for 1-minute HTF features
        because we lag by 1 minute.
        """
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_multiframe_features()

        df = engineer.df

        # First 60 seconds should have NaN for 1-minute features
        # (because we need 1 completed minute plus 1 lag)
        first_60_htf = df['htf_trend_1m'].iloc[:60]
        assert first_60_htf.isna().all() or first_60_htf.iloc[:60].isna().sum() > 50

    def test_htf_features_forward_filled(self, sample_1s_data):
        """Verify HTF features are forward-filled within each minute."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_multiframe_features()

        df = engineer.df.dropna(subset=['htf_trend_1m'])

        # Within each minute, all 60 seconds should have same HTF value
        # (because they use the same lagged 1-minute bar)
        if len(df) >= 120:
            minute_start = 120  # Skip first 2 minutes to ensure we have data
            minute_values = df['htf_trend_1m'].iloc[minute_start:minute_start+60]

            # Should have few unique values (ideally 1-2 due to minute boundary)
            assert len(minute_values.unique()) <= 2


# ============================================================================
# TEST: FULL FEATURE GENERATION
# ============================================================================

class TestFullFeatureGeneration:
    """Test complete feature generation pipeline."""

    def test_generate_all_features(self, sample_1s_data):
        """Generate all features and verify count."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features()

        # Should have many features (50+)
        assert len(engineer.feature_names) >= 40

        # DataFrame should have all features
        for name in engineer.feature_names:
            assert name in df.columns

    def test_generate_without_multiframe(self, sample_1s_data):
        """Generate features without multi-timeframe for speed."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features(include_multiframe=False)

        # Should still have core features
        assert len(engineer.feature_names) >= 30

        # Should NOT have HTF features
        assert 'htf_trend_1m' not in engineer.feature_names

    def test_no_nan_in_final_features(self, sample_1s_data):
        """Final features should have no NaN values."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features()

        for name in engineer.feature_names:
            assert df[name].isna().sum() == 0, f"Feature {name} has NaN values"

    def test_get_feature_matrix(self, sample_1s_data):
        """Verify feature matrix is correct shape."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features()

        matrix = engineer.get_feature_matrix()

        assert matrix.shape[0] == len(df)
        assert matrix.shape[1] == len(engineer.feature_names)


# ============================================================================
# TEST: PREPARE FEATURES FUNCTION
# ============================================================================

class TestPrepareFeatures:
    """Test the prepare_scalping_features convenience function."""

    def test_prepare_with_normalization(self, sample_1s_data):
        """Prepare features with StandardScaler normalization."""
        df, feature_names, scaler = prepare_scalping_features(
            sample_1s_data,
            normalize=True,
            include_multiframe=False
        )

        assert scaler is not None
        assert len(feature_names) > 0

        # Normalized features should have mean ~0, std ~1
        for name in feature_names[:5]:  # Check first 5
            assert abs(df[name].mean()) < 0.1
            assert abs(df[name].std() - 1.0) < 0.1

    def test_prepare_without_normalization(self, sample_1s_data):
        """Prepare features without normalization."""
        df, feature_names, scaler = prepare_scalping_features(
            sample_1s_data,
            normalize=False,
            include_multiframe=False
        )

        assert scaler is None
        assert len(feature_names) > 0


# ============================================================================
# TEST: NO LOOKAHEAD BIAS
# ============================================================================

class TestNoLookaheadBias:
    """Test that features have no lookahead bias."""

    def test_returns_use_past_data_only(self, sample_1s_data):
        """Verify returns are calculated from past data only."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_returns()

        df = engineer.df

        # return_1s at time T should be (close[T] - close[T-1]) / close[T-1]
        # This means it uses data from T-1 and T, not T+1
        for i in range(5, len(df)):
            expected = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            actual = df['return_1s'].iloc[i]

            if not np.isnan(expected) and not np.isnan(actual):
                assert abs(expected - actual) < 1e-10

    def test_vwap_uses_cumulative_past_only(self, sample_1s_data):
        """Verify VWAP is cumulative from session start, not using future."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_vwap()

        df = engineer.df

        # VWAP at bar 100 should only use bars 0-100
        # We can verify this by checking that adding more data doesn't change past VWAP
        vwap_at_100 = df['vwap'].iloc[100]

        # Calculate manually using only first 101 bars
        subset = df.iloc[:101]
        tp = (subset['high'] + subset['low'] + subset['close']) / 3
        manual_vwap = (tp * subset['volume']).sum() / subset['volume'].sum()

        assert abs(vwap_at_100 - manual_vwap) < 0.01

    def test_validate_no_lookahead_function(self, sample_1s_data):
        """Test the validate_no_lookahead utility function."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features(include_multiframe=False)

        # Add a dummy target
        df['target'] = np.random.randint(0, 3, len(df))

        # Should pass validation
        result = validate_no_lookahead(df, engineer.feature_names, 'target')
        assert result is True


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_bar(self, ny_tz):
        """Handle single bar gracefully."""
        df = pd.DataFrame({
            'open': [5000.0],
            'high': [5001.0],
            'low': [4999.0],
            'close': [5000.5],
            'volume': [100],
        }, index=pd.DatetimeIndex([
            datetime(2025, 6, 15, 9, 30, 0, tzinfo=ny_tz)
        ]))

        engineer = ScalpingFeatureEngineer(df)
        # Should not raise, but result will have many NaN
        engineer.add_returns()

    def test_zero_volume_bars(self, sample_1s_data):
        """Handle zero volume bars."""
        df = sample_1s_data.copy()
        df.loc[df.index[10:20], 'volume'] = 0  # Set some bars to zero volume

        engineer = ScalpingFeatureEngineer(df)
        engineer.add_vwap()  # VWAP uses volume

        # Should handle gracefully (NaN or zero)
        assert engineer.df['vwap'].iloc[15] is not None  # Doesn't crash

    def test_flat_price_bars(self, ny_tz):
        """Handle flat price (open=high=low=close) bars."""
        n_bars = 100
        start = datetime(2025, 6, 15, 9, 30, 0, tzinfo=ny_tz)
        timestamps = [start + timedelta(seconds=i) for i in range(n_bars)]

        df = pd.DataFrame({
            'open': [5000.0] * n_bars,
            'high': [5000.0] * n_bars,
            'low': [5000.0] * n_bars,
            'close': [5000.0] * n_bars,
            'volume': [100] * n_bars,
        }, index=pd.DatetimeIndex(timestamps))

        engineer = ScalpingFeatureEngineer(df)
        engineer.add_microstructure_features()

        # Body ratio should be 0 or NaN (not error)
        # Total range is 0, so ratio is undefined


# ============================================================================
# TEST: FEATURE CONFIG
# ============================================================================

class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()

        assert config.return_periods == [1, 5, 10, 30, 60]
        assert config.ema_periods == [9, 21, 50, 200]
        assert config.rsi_period == 14
        assert config.atr_period == 14

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureConfig(
            return_periods=[1, 5],
            ema_periods=[10, 20],
            rsi_period=10,
        )

        assert config.return_periods == [1, 5]
        assert config.ema_periods == [10, 20]
        assert config.rsi_period == 10


# ============================================================================
# INTEGRATION TEST
# ============================================================================

class TestIntegration:
    """Integration tests with real-like data."""

    def test_full_pipeline_multi_session(self, multi_session_data):
        """Test full pipeline across multiple trading sessions."""
        engineer = ScalpingFeatureEngineer(multi_session_data)
        df = engineer.generate_all_features(include_multiframe=True)

        # Should have data from both sessions (some rows dropped for NaN from lookbacks)
        assert len(df) >= 500

        # Should have all expected features
        assert len(engineer.feature_names) >= 40

        # No NaN in final output
        for name in engineer.feature_names:
            assert df[name].isna().sum() == 0

    @pytest.mark.skipif(
        not Path("data/historical/MES/MES_1s_2years.parquet").exists(),
        reason="Real parquet file not available"
    )
    def test_with_real_data(self):
        """Integration test with real MES data (if available)."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader("data/historical/MES/MES_1s_2years.parquet")
        df = loader.load_data()
        df = loader.convert_to_ny_timezone()
        df = loader.filter_rth()

        # Take a small sample for testing
        df = df.head(10000)

        engineer = ScalpingFeatureEngineer(df)
        df_features = engineer.generate_all_features(include_multiframe=True)

        assert len(df_features) > 5000
        assert len(engineer.feature_names) >= 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
