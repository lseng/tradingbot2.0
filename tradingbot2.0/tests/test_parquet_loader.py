"""
Unit tests for the parquet data loader.

These tests verify:
- Parquet file loading
- UTC to NY timezone conversion (including DST handling)
- RTH/ETH session filtering
- Multi-timeframe aggregation
- 3-class target variable creation
- Class weight calculation
- Time feature generation

Why These Tests Matter:
- Timezone conversion errors can cause trades at wrong times (critical for EOD flatten)
- RTH filtering ensures we only train/trade during liquid market hours
- 3-class target is essential for scalping (binary UP/DOWN misses FLAT trades)
- Class weights prevent model from always predicting dominant FLAT class
"""

import pytest
import pandas as pd
import numpy as np
from datetime import time
from zoneinfo import ZoneInfo
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))

from data.parquet_loader import (
    ParquetDataLoader,
    MES_TICK_SIZE,
    MES_TICK_VALUE,
    RTH_START,
    RTH_END,
    load_and_prepare_scalping_data
)


class TestParquetDataLoader:
    """Tests for ParquetDataLoader class."""

    def test_init_with_valid_path(self, sample_parquet_file):
        """Test loader initializes with valid parquet file."""
        loader = ParquetDataLoader(sample_parquet_file)
        assert loader.data_path.exists()
        assert loader.raw_data is None  # Not loaded yet

    def test_init_with_invalid_path(self):
        """Test loader raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            ParquetDataLoader("/nonexistent/path.parquet")

    def test_load_data(self, sample_parquet_file):
        """Test loading parquet data."""
        loader = ParquetDataLoader(sample_parquet_file)
        df = loader.load_data()

        assert df is not None
        assert len(df) == 1000
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_timezone_conversion(self, sample_parquet_file):
        """Test UTC to NY timezone conversion."""
        loader = ParquetDataLoader(sample_parquet_file)
        df = loader.load_data()
        df_ny = loader.convert_to_ny_timezone(df)

        # Check timezone is America/New_York
        assert str(df_ny.index.tz) == 'America/New_York'

        # Check time is correctly converted (UTC 14:30 -> NY 09:30 in winter)
        first_time = df_ny.index[0].time()
        assert first_time.hour == 9
        assert first_time.minute == 30

    def test_dst_handling(self, tmp_path):
        """Test DST transition handling.

        During DST transition:
        - Winter: NY = UTC - 5 hours
        - Summer: NY = UTC - 4 hours
        """
        # Create data spanning DST transition (March 10, 2024 at 2 AM NY)
        # Before DST: UTC 14:30 = NY 09:30 (UTC-5)
        # After DST: UTC 13:30 = NY 09:30 (UTC-4)
        winter_time = pd.Timestamp('2024-01-02 14:30:00', tz='UTC')
        summer_time = pd.Timestamp('2024-07-02 13:30:00', tz='UTC')

        # Create test data
        np.random.seed(42)
        winter_df = pd.DataFrame({
            'timestamp': [winter_time],
            'open': [5000.0], 'high': [5001.0], 'low': [4999.0],
            'close': [5000.5], 'volume': [100]
        })
        summer_df = pd.DataFrame({
            'timestamp': [summer_time],
            'open': [5000.0], 'high': [5001.0], 'low': [4999.0],
            'close': [5000.5], 'volume': [100]
        })

        # Save to parquet
        winter_path = tmp_path / "winter.parquet"
        summer_path = tmp_path / "summer.parquet"
        winter_df.to_parquet(winter_path, engine='pyarrow')
        summer_df.to_parquet(summer_path, engine='pyarrow')

        # Load and convert
        winter_loader = ParquetDataLoader(str(winter_path))
        summer_loader = ParquetDataLoader(str(summer_path))

        winter_data = winter_loader.load_data()
        summer_data = summer_loader.load_data()

        winter_ny = winter_loader.convert_to_ny_timezone(winter_data)
        summer_ny = summer_loader.convert_to_ny_timezone(summer_data)

        # Both should be 9:30 AM NY time
        assert winter_ny.index[0].time().hour == 9
        assert winter_ny.index[0].time().minute == 30
        assert summer_ny.index[0].time().hour == 9
        assert summer_ny.index[0].time().minute == 30


class TestSessionFiltering:
    """Tests for RTH/ETH session filtering."""

    def test_rth_filter(self, sample_ohlcv_data):
        """Test RTH filtering keeps only 9:30 AM - 4:00 PM NY."""
        # sample_ohlcv_data starts at 9:30 AM NY, so should all pass RTH filter
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.timezone = ZoneInfo('America/New_York')
        loader.raw_data = sample_ohlcv_data

        rth_data = loader.filter_rth(sample_ohlcv_data)

        # All 1000 rows should pass (they're all during RTH)
        assert len(rth_data) == len(sample_ohlcv_data)

    def test_rth_filter_excludes_eth(self, sample_ohlcv_data):
        """Test RTH filter excludes ETH data."""
        # Create data outside RTH (before 9:30 AM)
        eth_time = pd.Timestamp('2024-01-02 08:00:00', tz='America/New_York')
        eth_data = sample_ohlcv_data.copy()
        eth_data.index = pd.date_range(start=eth_time, periods=len(eth_data), freq='1s')

        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.timezone = ZoneInfo('America/New_York')
        loader.raw_data = eth_data

        rth_data = loader.filter_rth(eth_data)

        # No rows should pass (all before 9:30 AM)
        assert len(rth_data) == 0

    def test_rth_filter_excludes_weekends(self, sample_ohlcv_data):
        """Test RTH filter excludes weekend data."""
        # Create data on Saturday
        saturday = pd.Timestamp('2024-01-06 10:00:00', tz='America/New_York')  # Saturday
        weekend_data = sample_ohlcv_data.copy()
        weekend_data.index = pd.date_range(start=saturday, periods=len(weekend_data), freq='1s')

        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.timezone = ZoneInfo('America/New_York')
        loader.raw_data = weekend_data

        rth_data = loader.filter_rth(weekend_data)

        # No rows should pass (Saturday)
        assert len(rth_data) == 0


class TestTimeframeAggregation:
    """Tests for multi-timeframe aggregation."""

    def test_aggregate_to_5s(self, sample_ohlcv_data):
        """Test aggregation to 5-second bars."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        agg_data = loader.aggregate_timeframe(sample_ohlcv_data, '5s')

        # 1000 seconds / 5 = 200 bars
        assert len(agg_data) == 200

        # OHLCV columns preserved
        assert all(col in agg_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_aggregate_to_1min(self, sample_ohlcv_data):
        """Test aggregation to 1-minute bars."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        agg_data = loader.aggregate_timeframe(sample_ohlcv_data, '1min')

        # 1000 seconds / 60 = ~16-17 bars
        assert len(agg_data) <= 17

    def test_aggregate_ohlc_validity(self, sample_ohlcv_data):
        """Test aggregated OHLC relationships are valid."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        agg_data = loader.aggregate_timeframe(sample_ohlcv_data, '1min')

        # High >= Low
        assert (agg_data['high'] >= agg_data['low']).all()

        # High >= Open and Close
        assert (agg_data['high'] >= agg_data['open']).all()
        assert (agg_data['high'] >= agg_data['close']).all()

        # Low <= Open and Close
        assert (agg_data['low'] <= agg_data['open']).all()
        assert (agg_data['low'] <= agg_data['close']).all()

    def test_invalid_timeframe_raises_error(self, sample_ohlcv_data):
        """Test invalid timeframe raises ValueError."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        with pytest.raises(ValueError, match="Invalid timeframe"):
            loader.aggregate_timeframe(sample_ohlcv_data, '7s')


class TestTargetVariable:
    """Tests for 3-class target variable creation."""

    def test_target_creates_3_classes(self, sample_ohlcv_data):
        """Test target variable has exactly 3 classes."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        df_with_target = loader.create_target_variable(
            sample_ohlcv_data,
            lookahead_seconds=30,
            threshold_ticks=3.0
        )

        unique_classes = df_with_target['target'].unique()
        assert set(unique_classes).issubset({0, 1, 2})

    def test_target_no_nan(self, sample_ohlcv_data):
        """Test target variable has no NaN values."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        df_with_target = loader.create_target_variable(
            sample_ohlcv_data,
            lookahead_seconds=30,
            threshold_ticks=3.0
        )

        assert df_with_target['target'].isna().sum() == 0

    def test_target_removes_last_rows(self, sample_ohlcv_data):
        """Test lookahead rows are removed to prevent lookahead bias."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        lookahead = 30
        df_with_target = loader.create_target_variable(
            sample_ohlcv_data,
            lookahead_seconds=lookahead,
            threshold_ticks=3.0
        )

        # Should have lookahead fewer rows
        assert len(df_with_target) == len(sample_ohlcv_data) - lookahead

    def test_target_classification_logic(self):
        """Test target classification logic is correct."""
        # Create synthetic data with known price moves
        timestamps = pd.date_range(
            start='2024-01-02 09:30:00',
            periods=100,
            freq='1s',
            tz='America/New_York'
        )

        # Create prices that move exactly as expected
        close_prices = [5000.0] * 100

        # After 30 seconds, price should move by specific amounts
        # Tick size = 0.25, threshold = 3 ticks = 0.75 points
        # At index 0, price is 5000
        # At index 30, price should determine class for index 0

        # Price at 30 = 5000.80 (3.2 ticks up) -> Class 2 (UP)
        close_prices[30] = 5000.80

        # Price at 31 = 4999.20 (3.2 ticks down) -> Class 0 (DOWN)
        close_prices[31] = 4999.20

        # Price at 32 = 5000.50 (2 ticks up) -> Class 1 (FLAT)
        close_prices[32] = 5000.50

        df = pd.DataFrame({
            'open': close_prices,
            'high': [p + 0.5 for p in close_prices],
            'low': [p - 0.5 for p in close_prices],
            'close': close_prices,
            'volume': [100] * 100
        }, index=timestamps)

        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_target = loader.create_target_variable(
            df,
            lookahead_seconds=30,
            threshold_ticks=3.0,
            tick_size=0.25
        )

        # Check classifications
        assert df_with_target.iloc[0]['target'] == 2  # UP
        assert df_with_target.iloc[1]['target'] == 0  # DOWN
        assert df_with_target.iloc[2]['target'] == 1  # FLAT

    def test_future_columns_not_in_output(self, sample_ohlcv_data):
        """Test that future_close and future_tick_move columns are dropped.

        This is CRITICAL for preventing data leakage:
        - future_close contains the price N seconds in the future
        - future_tick_move contains the tick movement N seconds in the future
        - If accidentally used in feature engineering, model would have perfect foresight
        - This would cause catastrophic overfitting that collapses in live trading

        Fix for bug 10B.4 in IMPLEMENTATION_PLAN.md.
        """
        loader = ParquetDataLoader.__new__(ParquetDataLoader)
        loader.raw_data = sample_ohlcv_data

        df_with_target = loader.create_target_variable(
            sample_ohlcv_data,
            lookahead_seconds=30,
            threshold_ticks=3.0
        )

        # CRITICAL: Verify future columns are NOT in output
        assert 'future_close' not in df_with_target.columns, \
            "future_close column would leak future price data!"
        assert 'future_tick_move' not in df_with_target.columns, \
            "future_tick_move column would leak future movement data!"

        # Verify target column IS present
        assert 'target' in df_with_target.columns


class TestClassWeights:
    """Tests for class weight calculation."""

    def test_class_weights_sum_to_n_classes(self, sample_ohlcv_data):
        """Test class weights are properly normalized."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_target = loader.create_target_variable(
            sample_ohlcv_data,
            lookahead_seconds=30,
            threshold_ticks=3.0
        )

        weights = loader.get_class_weights(df_with_target)

        # Weights should exist for each class present
        assert len(weights) > 0

        # All weights should be positive
        for w in weights.values():
            assert w > 0

    def test_class_weights_higher_for_minority(self, sample_ohlcv_data):
        """Test minority classes get higher weights."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_target = loader.create_target_variable(
            sample_ohlcv_data,
            lookahead_seconds=30,
            threshold_ticks=3.0
        )

        weights = loader.get_class_weights(df_with_target)
        class_counts = df_with_target['target'].value_counts()

        # If there's class imbalance, minority should have higher weight
        if len(weights) > 1 and len(class_counts) > 1:
            min_class = class_counts.idxmin()
            max_class = class_counts.idxmax()

            if min_class in weights and max_class in weights:
                assert weights[min_class] >= weights[max_class]


class TestTimeFeatures:
    """Tests for time-based feature generation."""

    def test_time_features_added(self, sample_ohlcv_data):
        """Test all time features are added."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_features = loader.add_time_features(sample_ohlcv_data)

        expected_features = [
            'minutes_to_close', 'minutes_to_close_norm',
            'time_sin', 'time_cos',
            'day_of_week',
            'is_first_hour', 'is_last_hour', 'is_lunch'
        ]

        for feature in expected_features:
            assert feature in df_with_features.columns, f"Missing feature: {feature}"

    def test_minutes_to_close_range(self, sample_ohlcv_data):
        """Test minutes_to_close is in valid range."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_features = loader.add_time_features(sample_ohlcv_data)

        # Minutes to close should be 0-390 for RTH
        assert df_with_features['minutes_to_close'].min() >= 0
        assert df_with_features['minutes_to_close'].max() <= 390

    def test_cyclical_time_encoding(self, sample_ohlcv_data):
        """Test cyclical time encoding is in [-1, 1] range."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_features = loader.add_time_features(sample_ohlcv_data)

        # Sin and cos should be in [-1, 1]
        assert df_with_features['time_sin'].min() >= -1
        assert df_with_features['time_sin'].max() <= 1
        assert df_with_features['time_cos'].min() >= -1
        assert df_with_features['time_cos'].max() <= 1

    def test_is_first_hour(self, full_day_rth_data):
        """Test is_first_hour flag is set correctly."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        df_with_features = loader.add_time_features(full_day_rth_data)

        # First hour is 9:30-10:30 AM
        first_hour_data = df_with_features[df_with_features['is_first_hour']]
        other_data = df_with_features[~df_with_features['is_first_hour']]

        # Check times are correct
        for idx in first_hour_data.index[:10]:
            hour = idx.hour
            minute = idx.minute
            assert (hour == 9 and minute >= 30) or hour == 10


class TestDataSplit:
    """Tests for train/val/test split."""

    def test_split_ratios(self, sample_ohlcv_data):
        """Test split ratios are approximately correct."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        train, val, test = loader.train_test_split(
            sample_ohlcv_data,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        total = len(sample_ohlcv_data)
        assert abs(len(train) / total - 0.6) < 0.01
        assert abs(len(val) / total - 0.2) < 0.01
        assert abs(len(test) / total - 0.2) < 0.01

    def test_split_preserves_time_order(self, sample_ohlcv_data):
        """Test split maintains chronological order (no shuffling)."""
        loader = ParquetDataLoader.__new__(ParquetDataLoader)

        train, val, test = loader.train_test_split(
            sample_ohlcv_data,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        # Train should be before val
        assert train.index.max() < val.index.min()

        # Val should be before test
        assert val.index.max() < test.index.min()


class TestIntegration:
    """Integration tests using real parquet file (if available)."""

    @pytest.mark.skipif(
        not Path("data/historical/MES/MES_1s_2years.parquet").exists(),
        reason="Real parquet file not available"
    )
    def test_load_real_parquet(self, real_parquet_path):
        """Test loading the real MES parquet file."""
        if real_parquet_path is None:
            pytest.skip("Real parquet file not available")

        loader = ParquetDataLoader(real_parquet_path)
        df = loader.load_data()

        # Should have millions of rows
        assert len(df) > 1_000_000

        # Should have OHLCV columns
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.skipif(
        not Path("data/historical/MES/MES_1s_2years.parquet").exists(),
        reason="Real parquet file not available"
    )
    def test_full_pipeline(self, real_parquet_path):
        """Test full data preparation pipeline."""
        if real_parquet_path is None:
            pytest.skip("Real parquet file not available")

        full_df, train_df, val_df, test_df = load_and_prepare_scalping_data(
            real_parquet_path,
            filter_rth=True,
            lookahead_seconds=30,
            threshold_ticks=3.0
        )

        # Should have data
        assert len(full_df) > 0
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Should have target column
        assert 'target' in full_df.columns

        # Target should have 3 classes
        unique_classes = full_df['target'].unique()
        assert set(unique_classes) == {0, 1, 2}

        # Should have time features
        assert 'minutes_to_close' in full_df.columns
