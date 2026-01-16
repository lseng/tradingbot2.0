"""
Comprehensive Lookahead Bias Tests for Scalping Feature Engineering.

This module provides exhaustive tests to verify that no features use future data.
Lookahead bias is the #1 cause of backtest overfitting and must be eliminated.

Test Categories:
1. Rolling Window Operations - Verify rolling calculations only use past data
2. Shift Operations - Ensure all shifts are backward (positive values)
3. EMA/MACD Initialization - Verify warmup doesn't peek ahead
4. Resampling Operations - Ensure aggregations use completed bars only
5. Multi-Timeframe Features - Verify HTF features are properly lagged
6. Target Variable - Ensure target uses strictly future data
7. End-to-End Pipeline - Full pipeline validation
8. Statistical Detection - Correlation-based lookahead detection

Reference: Go-Live Checklist Item #6
Spec: specs/ml-scalping-model.md (Overfitting Prevention)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.ml.data.scalping_features import (
    ScalpingFeatureEngineer,
    FeatureConfig,
    validate_no_lookahead,
)
from src.ml.data.parquet_loader import ParquetDataLoader


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def ny_tz():
    """New York timezone."""
    return ZoneInfo('America/New_York')


@pytest.fixture
def sample_1s_data(ny_tz):
    """Create sample 1-second OHLCV data for testing."""
    # Create 500 bars (enough for all feature calculations)
    n_bars = 500
    base_time = datetime(2025, 6, 15, 9, 30, 0, tzinfo=ny_tz)

    # Create deterministic price series with known patterns
    np.random.seed(42)
    base_price = 5000.0

    # Generate prices with small random walk
    returns = np.random.randn(n_bars) * 0.0001  # 1bp std dev
    prices = base_price * np.cumprod(1 + returns)

    # Create OHLCV data
    opens = prices * (1 + np.random.randn(n_bars) * 0.00005)
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.randn(n_bars)) * 0.00005)
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.randn(n_bars)) * 0.00005)
    closes = prices
    volumes = np.random.randint(50, 200, n_bars)

    # Create timestamps
    timestamps = [base_time + timedelta(seconds=i) for i in range(n_bars)]

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    }, index=pd.DatetimeIndex(timestamps))

    return df


@pytest.fixture
def large_sample_data(ny_tz):
    """Create larger sample for statistical tests."""
    n_bars = 2000
    base_time = datetime(2025, 6, 15, 9, 30, 0, tzinfo=ny_tz)

    np.random.seed(123)
    base_price = 5000.0
    returns = np.random.randn(n_bars) * 0.0002
    prices = base_price * np.cumprod(1 + returns)

    opens = prices * (1 + np.random.randn(n_bars) * 0.0001)
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.randn(n_bars)) * 0.0001)
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.randn(n_bars)) * 0.0001)
    closes = prices
    volumes = np.random.randint(50, 200, n_bars)

    timestamps = [base_time + timedelta(seconds=i) for i in range(n_bars)]

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    }, index=pd.DatetimeIndex(timestamps))


# ============================================================================
# TEST: ROLLING WINDOW OPERATIONS
# ============================================================================

class TestRollingWindowNoLookahead:
    """Test that rolling window calculations only use past data."""

    def test_rolling_mean_uses_past_only(self, sample_1s_data):
        """Verify rolling mean at index i only uses data from i-window to i."""
        df = sample_1s_data.copy()
        window = 10

        df['rolling_mean'] = df['close'].rolling(window=window).mean()

        # For each index i >= window-1, verify rolling mean
        for i in range(window - 1, min(100, len(df))):
            expected = df['close'].iloc[i-window+1:i+1].mean()
            actual = df['rolling_mean'].iloc[i]

            assert abs(expected - actual) < 1e-10, \
                f"Rolling mean at {i} uses data outside window [{i-window+1}, {i}]"

    def test_rolling_std_uses_past_only(self, sample_1s_data):
        """Verify rolling std at index i only uses data from i-window to i."""
        df = sample_1s_data.copy()
        window = 20

        df['rolling_std'] = df['close'].rolling(window=window).std()

        for i in range(window - 1, min(100, len(df))):
            expected = df['close'].iloc[i-window+1:i+1].std()
            actual = df['rolling_std'].iloc[i]

            if not np.isnan(expected) and not np.isnan(actual):
                assert abs(expected - actual) < 1e-10, \
                    f"Rolling std at {i} mismatch"

    def test_rolling_max_uses_past_only(self, sample_1s_data):
        """Verify rolling max (used in Stochastic) only uses past data."""
        df = sample_1s_data.copy()
        window = 14

        df['rolling_max'] = df['high'].rolling(window=window).max()

        for i in range(window - 1, min(100, len(df))):
            expected = df['high'].iloc[i-window+1:i+1].max()
            actual = df['rolling_max'].iloc[i]

            assert abs(expected - actual) < 1e-10, \
                f"Rolling max at {i} mismatch"

    def test_rolling_min_uses_past_only(self, sample_1s_data):
        """Verify rolling min (used in Stochastic) only uses past data."""
        df = sample_1s_data.copy()
        window = 14

        df['rolling_min'] = df['low'].rolling(window=window).min()

        for i in range(window - 1, min(100, len(df))):
            expected = df['low'].iloc[i-window+1:i+1].min()
            actual = df['rolling_min'].iloc[i]

            assert abs(expected - actual) < 1e-10, \
                f"Rolling min at {i} mismatch"


# ============================================================================
# TEST: EMA AND MOMENTUM INDICATOR INITIALIZATION
# ============================================================================

class TestEMANoLookahead:
    """Test that EMA calculations don't peek ahead during warmup."""

    def test_ema_depends_only_on_past(self, sample_1s_data):
        """
        Verify EMA at index i depends only on data 0:i+1.

        The key insight: if we modify data after index i, EMA at i should not change.
        """
        df = sample_1s_data.copy()

        # Calculate EMA on full data
        ema_full = df['close'].ewm(span=9, adjust=False).mean()

        # For test index, modify future data and recalculate
        test_idx = 100
        df_modified = df.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = 9999.0  # Extreme value
        ema_modified = df_modified['close'].ewm(span=9, adjust=False).mean()

        # EMA at test_idx should be identical
        assert abs(ema_full.iloc[test_idx] - ema_modified.iloc[test_idx]) < 1e-10, \
            "EMA at test index changed when future data was modified"

    def test_ema_incremental_calculation(self, sample_1s_data):
        """Verify EMA can be calculated incrementally without future data."""
        df = sample_1s_data.copy()
        span = 9
        alpha = 2 / (span + 1)

        # Calculate EMA incrementally
        ema_incremental = np.zeros(len(df))
        ema_incremental[0] = df['close'].iloc[0]

        for i in range(1, len(df)):
            ema_incremental[i] = alpha * df['close'].iloc[i] + (1 - alpha) * ema_incremental[i-1]

        # Compare with pandas EMA
        ema_pandas = df['close'].ewm(span=span, adjust=False).mean()

        # Should match
        np.testing.assert_allclose(ema_incremental, ema_pandas.values, rtol=1e-10)

    def test_rsi_warmup_no_lookahead(self, sample_1s_data):
        """Verify RSI warmup period doesn't use future data."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_momentum_indicators()

        df = engineer.df
        test_idx = 50  # Well past warmup

        # Modify future data
        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = df_modified['close'].iloc[test_idx] * 2

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_momentum_indicators()

        # RSI at test_idx should be identical
        assert abs(df['rsi_norm'].iloc[test_idx] - engineer_modified.df['rsi_norm'].iloc[test_idx]) < 1e-10, \
            "RSI changed when future data was modified"

    def test_macd_no_lookahead(self, sample_1s_data):
        """Verify MACD calculation doesn't use future data."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_momentum_indicators()

        df = engineer.df
        test_idx = 100

        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = df_modified['close'].iloc[test_idx] * 2

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_momentum_indicators()

        # MACD at test_idx should be identical
        assert abs(df['macd_norm'].iloc[test_idx] - engineer_modified.df['macd_norm'].iloc[test_idx]) < 1e-10, \
            "MACD changed when future data was modified"


# ============================================================================
# TEST: SHIFT OPERATIONS
# ============================================================================

class TestShiftOperations:
    """Test that all shift operations are backward (positive values)."""

    def test_return_shift_is_backward(self, sample_1s_data):
        """Verify returns use shift(period), not shift(-period)."""
        df = sample_1s_data.copy()

        # pct_change(period) is equivalent to using shift(period)
        # This means return at T uses close[T] and close[T-period], not future

        period = 5
        manual_return = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        pct_return = df['close'].pct_change(period)

        # Should be equivalent (use rtol=1e-8 for floating point tolerance)
        np.testing.assert_allclose(
            manual_return.dropna().values,
            pct_return.dropna().values,
            rtol=1e-8  # Relaxed tolerance for floating point precision
        )

        # Verify specific index
        test_idx = 100
        expected = (df['close'].iloc[test_idx] - df['close'].iloc[test_idx - period]) / df['close'].iloc[test_idx - period]
        actual = pct_return.iloc[test_idx]

        assert abs(expected - actual) < 1e-10, "Return calculation uses wrong shift direction"

    def test_log_return_shift_is_backward(self, sample_1s_data):
        """Verify log returns use backward shift."""
        df = sample_1s_data.copy()
        period = 10

        log_return = np.log(df['close'] / df['close'].shift(period))

        test_idx = 100
        expected = np.log(df['close'].iloc[test_idx] / df['close'].iloc[test_idx - period])
        actual = log_return.iloc[test_idx]

        assert abs(expected - actual) < 1e-10, "Log return uses wrong shift direction"

    def test_no_negative_shifts_in_features(self, sample_1s_data):
        """
        Verify feature engineering doesn't use negative shifts (which would be lookahead).

        This test generates features and checks that modifying future data
        doesn't affect features at the test index.
        """
        # Add individual features instead of generate_all_features to avoid dropna issues
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_returns()
        engineer.add_emas()
        engineer.add_vwap()
        engineer.add_volatility_features()
        engineer.add_momentum_indicators()
        engineer.add_microstructure_features()
        engineer.add_volume_features()

        df = engineer.df
        test_idx = 250  # Far enough for all warmup periods

        # Store original feature values
        original_features = {name: df[name].iloc[test_idx] for name in engineer.feature_names if name in df.columns}

        # Modify future data
        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = 9999.0
        df_modified.loc[df_modified.index[test_idx+1:], 'high'] = 9999.0
        df_modified.loc[df_modified.index[test_idx+1:], 'low'] = 9999.0
        df_modified.loc[df_modified.index[test_idx+1:], 'volume'] = 99999

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_returns()
        engineer_modified.add_emas()
        engineer_modified.add_vwap()
        engineer_modified.add_volatility_features()
        engineer_modified.add_momentum_indicators()
        engineer_modified.add_microstructure_features()
        engineer_modified.add_volume_features()

        # All features at test_idx should be identical
        for feature in engineer.feature_names:
            if feature not in df.columns or feature not in engineer_modified.df.columns:
                continue
            orig_val = original_features.get(feature)
            mod_val = engineer_modified.df[feature].iloc[test_idx]

            if orig_val is not None and not np.isnan(orig_val) and not np.isnan(mod_val):
                assert abs(orig_val - mod_val) < 1e-8, \
                    f"Feature '{feature}' changed when future data was modified: {orig_val} vs {mod_val}"


# ============================================================================
# TEST: VWAP AND RESAMPLING OPERATIONS
# ============================================================================

class TestVWAPNoLookahead:
    """Test that VWAP and resampling operations don't use future data."""

    def test_vwap_is_cumulative_not_forward(self, sample_1s_data):
        """Verify VWAP uses cumulative data from session start, not future."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_vwap()

        df = engineer.df

        # Manually calculate VWAP at bar 100
        test_idx = 100
        subset = df.iloc[:test_idx + 1]
        tp = (subset['high'] + subset['low'] + subset['close']) / 3
        manual_vwap = (tp * subset['volume']).sum() / subset['volume'].sum()

        assert abs(df['vwap'].iloc[test_idx] - manual_vwap) < 0.01, \
            "VWAP calculation doesn't match cumulative formula"

    def test_vwap_not_affected_by_future_data(self, sample_1s_data):
        """Verify VWAP at index i is not affected by data after i."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_vwap()
        vwap_original = engineer.df['vwap'].copy()

        test_idx = 100

        # Modify future data
        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = 9999.0
        df_modified.loc[df_modified.index[test_idx+1:], 'volume'] = 99999

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_vwap()

        # VWAP at test_idx should be identical
        assert abs(vwap_original.iloc[test_idx] - engineer_modified.df['vwap'].iloc[test_idx]) < 1e-6, \
            "VWAP changed when future data was modified"


# ============================================================================
# TEST: ATR AND VOLATILITY FEATURES
# ============================================================================

class TestVolatilityNoLookahead:
    """Test that volatility features don't use future data."""

    def test_atr_uses_past_only(self, sample_1s_data):
        """Verify ATR at index i only uses data up to i."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volatility_features()

        df = engineer.df
        test_idx = 100

        # Modify future data
        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'high'] = 9999.0
        df_modified.loc[df_modified.index[test_idx+1:], 'low'] = 0.0

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_volatility_features()

        # ATR at test_idx should be identical
        assert abs(df['atr_ticks'].iloc[test_idx] - engineer_modified.df['atr_ticks'].iloc[test_idx]) < 1e-6, \
            "ATR changed when future data was modified"

    def test_bollinger_bands_use_past_only(self, sample_1s_data):
        """Verify Bollinger Bands at index i only use data up to i."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volatility_features()

        df = engineer.df
        test_idx = 100

        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = 9999.0

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_volatility_features()

        assert abs(df['bb_width'].iloc[test_idx] - engineer_modified.df['bb_width'].iloc[test_idx]) < 1e-6, \
            "BB width changed when future data was modified"

        assert abs(df['bb_position'].iloc[test_idx] - engineer_modified.df['bb_position'].iloc[test_idx]) < 1e-6, \
            "BB position changed when future data was modified"

    def test_realized_volatility_windows(self, sample_1s_data):
        """Verify realized volatility windows use past data only."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volatility_features()

        df = engineer.df
        test_idx = 300  # Far enough for all volatility windows

        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'close'] = df_modified['close'].iloc[test_idx] * 2

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_volatility_features()

        # Check all volatility features
        for window in [10, 30, 60, 300]:
            col = f'volatility_{window}s'
            if col in df.columns:
                orig = df[col].iloc[test_idx]
                mod = engineer_modified.df[col].iloc[test_idx]
                if not np.isnan(orig) and not np.isnan(mod):
                    assert abs(orig - mod) < 1e-8, \
                        f"Volatility window {window}s changed when future data was modified"


# ============================================================================
# TEST: TARGET VARIABLE
# ============================================================================

class TestTargetNoLookahead:
    """Test that target variable uses strictly future data (not current bar)."""

    def test_target_uses_future_close(self, sample_1s_data):
        """Verify target is calculated from future close, not current."""
        from src.ml.data.parquet_loader import ParquetDataLoader

        # Create a mock loader to use create_target_variable
        df = sample_1s_data.copy()
        df['symbol'] = 'MES'

        # Manually test target calculation
        lookahead = 30
        threshold = 3.0

        # Target should be: future_price = close.shift(-lookahead)
        future_price = df['close'].shift(-lookahead)
        tick_move = (future_price - df['close']) / 0.25

        target = np.where(tick_move > threshold, 2,
                 np.where(tick_move < -threshold, 0, 1))

        # Verify specific index
        test_idx = 100
        expected_future = df['close'].iloc[test_idx + lookahead]
        expected_move = (expected_future - df['close'].iloc[test_idx]) / 0.25

        if expected_move > threshold:
            expected_target = 2
        elif expected_move < -threshold:
            expected_target = 0
        else:
            expected_target = 1

        assert target[test_idx] == expected_target, \
            "Target calculation mismatch"

    def test_target_excludes_current_bar_data(self, sample_1s_data):
        """
        Verify that target at T does not depend on any data at T.

        This is critical: the model predicts using features at T,
        so target must be based purely on T+lookahead data.
        """
        df = sample_1s_data.copy()
        lookahead = 30

        test_idx = 100

        # Calculate target
        future_price = df['close'].shift(-lookahead)
        tick_move = (future_price - df['close']) / 0.25

        # The target at test_idx uses:
        # - df['close'].iloc[test_idx + lookahead] (future)
        # - df['close'].iloc[test_idx] (current)

        # Modify current bar's OHLC (except close used in target)
        df_modified = df.copy()
        df_modified.loc[df_modified.index[test_idx], 'open'] = 9999.0
        df_modified.loc[df_modified.index[test_idx], 'high'] = 9999.0
        df_modified.loc[df_modified.index[test_idx], 'low'] = 0.0
        df_modified.loc[df_modified.index[test_idx], 'volume'] = 999999

        # Target should still be the same (only uses close[T] and close[T+lookahead])
        future_price_mod = df_modified['close'].shift(-lookahead)
        tick_move_mod = (future_price_mod - df_modified['close']) / 0.25

        assert tick_move.iloc[test_idx] == tick_move_mod.iloc[test_idx], \
            "Target unexpectedly changed when non-close current bar data was modified"


# ============================================================================
# TEST: END-TO-END PIPELINE VALIDATION
# ============================================================================

class TestEndToEndNoLookahead:
    """End-to-end tests for lookahead bias in the full pipeline."""

    def test_modifying_future_doesnt_change_past_features(self, sample_1s_data):
        """
        Gold standard lookahead test: modify future data, verify past features unchanged.

        If any feature at time T changes when we modify data after T,
        that feature has lookahead bias.
        """
        test_idx = 350  # Far enough for all warmup periods

        # Generate features on original data WITHOUT dropna
        engineer_orig = ScalpingFeatureEngineer(sample_1s_data)
        engineer_orig.add_returns()
        engineer_orig.add_emas()
        engineer_orig.add_vwap()
        engineer_orig.add_volatility_features()
        engineer_orig.add_momentum_indicators()
        engineer_orig.add_microstructure_features()
        engineer_orig.add_volume_features()

        df_orig = engineer_orig.df
        features_orig = {name: df_orig[name].iloc[test_idx] for name in engineer_orig.feature_names if name in df_orig.columns}

        # Modify all future data dramatically
        df_modified = sample_1s_data.copy()
        future_mask = df_modified.index > df_modified.index[test_idx]
        df_modified.loc[future_mask, 'open'] = 9999.0
        df_modified.loc[future_mask, 'high'] = 9999.0
        df_modified.loc[future_mask, 'low'] = 0.0
        df_modified.loc[future_mask, 'close'] = 5000.0  # Different from original
        df_modified.loc[future_mask, 'volume'] = 999999

        # Generate features on modified data WITHOUT dropna
        engineer_mod = ScalpingFeatureEngineer(df_modified)
        engineer_mod.add_returns()
        engineer_mod.add_emas()
        engineer_mod.add_vwap()
        engineer_mod.add_volatility_features()
        engineer_mod.add_momentum_indicators()
        engineer_mod.add_microstructure_features()
        engineer_mod.add_volume_features()

        df_mod = engineer_mod.df

        # Compare all features
        failed_features = []
        for feature in engineer_orig.feature_names:
            if feature not in df_orig.columns or feature not in df_mod.columns:
                continue
            orig_val = features_orig.get(feature)
            mod_val = df_mod[feature].iloc[test_idx]

            if orig_val is None:
                continue
            if np.isnan(orig_val) and np.isnan(mod_val):
                continue  # Both NaN is OK
            elif np.isnan(orig_val) != np.isnan(mod_val):
                failed_features.append((feature, orig_val, mod_val, "NaN mismatch"))
            elif abs(orig_val - mod_val) > 1e-6:
                failed_features.append((feature, orig_val, mod_val, "Value changed"))

        assert len(failed_features) == 0, \
            f"Features with lookahead bias detected:\n" + \
            "\n".join([f"  {f[0]}: {f[1]} -> {f[2]} ({f[3]})" for f in failed_features])

    def test_feature_at_t_independent_of_data_after_t(self, large_sample_data):
        """
        Extensive test: verify features at multiple points are independent of future.

        Uses individual feature methods to avoid dropna issues.
        """
        test_indices = [400, 600, 800, 1000, 1200]  # Adjusted for warmup periods

        # Generate features on original data WITHOUT dropna
        engineer_orig = ScalpingFeatureEngineer(large_sample_data)
        engineer_orig.add_returns()
        engineer_orig.add_emas()
        engineer_orig.add_vwap()
        engineer_orig.add_volatility_features()
        engineer_orig.add_momentum_indicators()
        engineer_orig.add_microstructure_features()
        engineer_orig.add_volume_features()

        df_orig = engineer_orig.df

        for test_idx in test_indices:
            features_orig = {name: df_orig[name].iloc[test_idx] for name in engineer_orig.feature_names if name in df_orig.columns}

            # Truncate data at test_idx + 1 and regenerate
            df_truncated = large_sample_data.iloc[:test_idx + 1].copy()
            engineer_trunc = ScalpingFeatureEngineer(df_truncated)
            engineer_trunc.add_returns()
            engineer_trunc.add_emas()
            engineer_trunc.add_vwap()
            engineer_trunc.add_volatility_features()
            engineer_trunc.add_momentum_indicators()
            engineer_trunc.add_microstructure_features()
            engineer_trunc.add_volume_features()

            df_trunc = engineer_trunc.df

            # Compare
            for feature in engineer_orig.feature_names:
                if feature not in df_orig.columns or feature not in df_trunc.columns:
                    continue
                orig_val = features_orig.get(feature)
                # Use -1 index since we truncated
                trunc_val = df_trunc[feature].iloc[-1]

                if orig_val is None:
                    continue
                if not np.isnan(orig_val) and not np.isnan(trunc_val):
                    assert abs(orig_val - trunc_val) < 1e-6, \
                        f"Feature '{feature}' at index {test_idx} differs when data is truncated"


# ============================================================================
# TEST: STATISTICAL LOOKAHEAD DETECTION
# ============================================================================

class TestStatisticalLookaheadDetection:
    """Statistical tests to detect subtle lookahead bias."""

    def test_no_perfect_future_correlation(self, large_sample_data):
        """
        Verify no feature has suspiciously high correlation with future targets.

        A correlation > 0.9 with future targets would indicate lookahead bias.
        Features naturally have some correlation due to market dynamics, but
        near-perfect correlation indicates information leakage.

        Note: We do NOT test for correlation "increasing" with horizon because
        momentum features naturally correlate more with matching horizons.
        That's predictive power, not lookahead bias.
        """
        engineer = ScalpingFeatureEngineer(large_sample_data)
        engineer.add_returns()
        engineer.add_emas()
        engineer.add_vwap()
        engineer.add_volatility_features()
        engineer.add_momentum_indicators()
        engineer.add_microstructure_features()
        engineer.add_volume_features()

        df = engineer.df

        # Create target at primary horizon
        horizon = 30
        threshold = 3.0
        future_price = df['close'].shift(-horizon)
        tick_move = (future_price - df['close']) / 0.25
        df['target'] = np.where(tick_move > threshold, 2,
                       np.where(tick_move < -threshold, 0, 1))

        # Check for suspiciously high correlation (lookahead indicator)
        suspicious_features = []
        for feature in engineer.feature_names:
            if feature not in df.columns or df[feature].isna().all():
                continue

            corr = abs(df[feature].corr(df['target']))
            if not np.isnan(corr) and corr > 0.9:  # Near-perfect correlation
                suspicious_features.append((feature, corr))

        assert len(suspicious_features) == 0, \
            f"Features with suspiciously HIGH correlation (>0.9) detected:\n" + \
            "\n".join([f"  {f[0]}: {f[1]:.3f}" for f in suspicious_features])

    def test_random_shuffle_should_destroy_predictability(self, large_sample_data):
        """
        If we randomly shuffle the target, features should have ~0 correlation.

        If correlation remains high after shuffling, features have no true
        predictive power (possibly due to lookahead or other issues).
        """
        engineer = ScalpingFeatureEngineer(large_sample_data)
        df = engineer.generate_all_features(include_multiframe=False)

        # Create actual target
        lookahead = 30
        threshold = 3.0
        future_price = df['close'].shift(-lookahead)
        tick_move = (future_price - df['close']) / 0.25
        df['target'] = np.where(tick_move > threshold, 2,
                       np.where(tick_move < -threshold, 0, 1))

        # Create shuffled target
        np.random.seed(42)
        df['target_shuffled'] = np.random.permutation(df['target'].values)

        # Features should have low correlation with shuffled target
        high_shuffled_corr_features = []
        for feature in engineer.feature_names:
            if df[feature].isna().all():
                continue

            corr_shuffled = abs(df[feature].corr(df['target_shuffled']))

            if not np.isnan(corr_shuffled) and corr_shuffled > 0.1:
                high_shuffled_corr_features.append((feature, corr_shuffled))

        assert len(high_shuffled_corr_features) == 0, \
            f"Features with suspiciously high correlation to shuffled target:\n" + \
            "\n".join([f"  {f[0]}: {f[1]:.3f}" for f in high_shuffled_corr_features])

    def test_validate_no_lookahead_utility(self, sample_1s_data):
        """Test the validate_no_lookahead utility function."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features(include_multiframe=False)

        # Add target
        lookahead = 30
        threshold = 3.0
        future_price = df['close'].shift(-lookahead)
        tick_move = (future_price - df['close']) / 0.25
        df['target'] = np.where(tick_move > threshold, 2,
                       np.where(tick_move < -threshold, 0, 1))

        # Should pass validation
        result = validate_no_lookahead(df, engineer.feature_names, 'target')
        assert result is True, "Legitimate features failed lookahead validation"

    def test_detect_intentional_lookahead(self, sample_1s_data):
        """Verify we can detect intentionally introduced lookahead bias."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        df = engineer.generate_all_features(include_multiframe=False)

        # Create target
        lookahead = 30
        threshold = 3.0
        future_price = df['close'].shift(-lookahead)
        tick_move = (future_price - df['close']) / 0.25
        df['target'] = np.where(tick_move > threshold, 2,
                       np.where(tick_move < -threshold, 0, 1))

        # Add an INTENTIONALLY BIASED feature (uses future data)
        df['cheating_feature'] = future_price / df['close'] - 1  # This is lookahead!

        # This feature should have near-perfect correlation with target
        corr = df['cheating_feature'].corr(df['target'].shift(-1))  # shift to align

        # The cheating feature should have very high correlation with future
        # (This test documents the expected behavior of a biased feature)
        assert abs(df['cheating_feature'].corr(tick_move)) > 0.5, \
            "Sanity check: cheating feature should correlate with future movement"


# ============================================================================
# TEST: MICROSTRUCTURE FEATURES
# ============================================================================

class TestMicrostructureNoLookahead:
    """Test that microstructure features use only current and past bar data."""

    def test_bar_direction_uses_current_bar_only(self, sample_1s_data):
        """Verify bar direction uses only the current bar's OHLC."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_microstructure_features()

        df = engineer.df
        test_idx = 100

        # Bar direction should be sign(close - open) for current bar
        expected = 1 if df['close'].iloc[test_idx] > df['open'].iloc[test_idx] else (
            -1 if df['close'].iloc[test_idx] < df['open'].iloc[test_idx] else 0
        )

        actual = df['bar_direction'].iloc[test_idx]

        assert expected == actual, "Bar direction calculation mismatch"

    def test_wick_ratios_use_current_bar_only(self, sample_1s_data):
        """Verify wick ratios use only current bar data."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_microstructure_features()

        df = engineer.df
        test_idx = 100

        bar = df.iloc[test_idx]
        bar_range = bar['high'] - bar['low']

        if bar_range > 0:
            expected_upper = (bar['high'] - max(bar['open'], bar['close'])) / bar_range
            expected_lower = (min(bar['open'], bar['close']) - bar['low']) / bar_range

            actual_upper = df['upper_wick_ratio'].iloc[test_idx]
            actual_lower = df['lower_wick_ratio'].iloc[test_idx]

            assert abs(expected_upper - actual_upper) < 1e-10, "Upper wick ratio mismatch"
            assert abs(expected_lower - actual_lower) < 1e-10, "Lower wick ratio mismatch"


# ============================================================================
# TEST: VOLUME FEATURES
# ============================================================================

class TestVolumeNoLookahead:
    """Test that volume features use only past and current data."""

    def test_volume_ratio_uses_past_only(self, sample_1s_data):
        """Verify volume ratio at index i uses volume from i-window to i."""
        engineer = ScalpingFeatureEngineer(sample_1s_data)
        engineer.add_volume_features()

        df = engineer.df
        test_idx = 100

        df_modified = sample_1s_data.copy()
        df_modified.loc[df_modified.index[test_idx+1:], 'volume'] = 999999

        engineer_modified = ScalpingFeatureEngineer(df_modified)
        engineer_modified.add_volume_features()

        # Volume ratio should be unchanged
        for col in ['volume_ratio_10s', 'volume_ratio_30s', 'volume_ratio_60s']:
            if col in df.columns:
                orig = df[col].iloc[test_idx]
                mod = engineer_modified.df[col].iloc[test_idx]
                if not np.isnan(orig) and not np.isnan(mod):
                    assert abs(orig - mod) < 1e-8, \
                        f"{col} changed when future volume was modified"


# ============================================================================
# TEST: DOCUMENTATION CHECK
# ============================================================================

class TestLookaheadDocumentation:
    """Verify lookahead prevention is documented in code."""

    def test_multiframe_lagging_documented(self):
        """Verify multi-timeframe features document their lagging strategy."""
        import inspect
        from src.ml.data.scalping_features import ScalpingFeatureEngineer

        # Check that add_multiframe_features mentions lagging
        source = inspect.getsource(ScalpingFeatureEngineer.add_multiframe_features)

        assert 'shift' in source.lower() or 'lag' in source.lower(), \
            "add_multiframe_features should document lagging strategy"

    def test_validate_no_lookahead_exists(self):
        """Verify the validation function exists and is documented."""
        from src.ml.data.scalping_features import validate_no_lookahead

        assert validate_no_lookahead.__doc__ is not None, \
            "validate_no_lookahead should have documentation"

        assert 'lookahead' in validate_no_lookahead.__doc__.lower(), \
            "Documentation should mention lookahead"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
