"""
Tests for the 5-minute scalping data pipeline.

Tests cover:
1. Data loading from CSV/TXT files
2. Aggregation from 1-minute to 5-minute bars
3. RTH filtering
4. Temporal train/val/test splits
5. Data validation and integrity checks
"""

import numpy as np
import pandas as pd
import pytest
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from scalping.data_pipeline import (
    load_1min_data,
    aggregate_to_5min,
    filter_rth,
    create_temporal_splits,
    ScalpingDataPipeline,
    DataConfig,
    _validate_ohlcv,
)

NY_TZ = ZoneInfo("America/New_York")


class TestLoad1MinData:
    """Tests for load_1min_data function."""

    def test_load_from_csv(self, sample_csv_file):
        """Test loading data from CSV file."""
        df = load_1min_data(sample_csv_file)

        assert df is not None
        assert len(df) > 0
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])

    def test_datetime_index(self, sample_csv_file):
        """Test that datetime is set as index with NY timezone."""
        df = load_1min_data(sample_csv_file)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None
        assert str(df.index.tz) == "America/New_York"

    def test_sorted_by_datetime(self, sample_csv_file):
        """Test that data is sorted by datetime."""
        df = load_1min_data(sample_csv_file)

        assert df.index.is_monotonic_increasing

    def test_file_not_found(self, tmp_path):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_1min_data(tmp_path / "nonexistent.txt")


class TestValidateOHLCV:
    """Tests for OHLCV validation."""

    def test_valid_data(self, sample_1min_data):
        """Test validation passes for valid data."""
        # Should not raise
        _validate_ohlcv(sample_1min_data)

    def test_detects_invalid_high(self, sample_1min_data):
        """Test detection of invalid high < close."""
        df = sample_1min_data.copy()
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["close"] - 1

        # Should log warning but not raise
        _validate_ohlcv(df)

    def test_detects_invalid_low(self, sample_1min_data):
        """Test detection of invalid low > close."""
        df = sample_1min_data.copy()
        df.iloc[5, df.columns.get_loc("low")] = df.iloc[5]["close"] + 1

        # Should log warning but not raise
        _validate_ohlcv(df)


class TestAggregateTo5Min:
    """Tests for aggregate_to_5min function."""

    def test_aggregation_reduces_rows(self, sample_1min_data):
        """Test that aggregation reduces row count by ~5x."""
        df_5min = aggregate_to_5min(sample_1min_data)

        # 100 1-min bars should become ~20 5-min bars
        expected_max = len(sample_1min_data) // 5 + 1
        assert len(df_5min) <= expected_max

    def test_ohlc_aggregation_correct(self, sample_1min_data):
        """Test OHLC aggregation rules: open=first, high=max, low=min, close=last."""
        df_5min = aggregate_to_5min(sample_1min_data)

        # Get first 5-min bar
        first_bar = df_5min.iloc[0]

        # Get corresponding 1-min bars
        start_time = df_5min.index[0]
        end_time = start_time + pd.Timedelta(minutes=5)
        mask = (sample_1min_data.index >= start_time) & (sample_1min_data.index < end_time)
        corresponding_1min = sample_1min_data[mask]

        if len(corresponding_1min) > 0:
            assert first_bar["open"] == corresponding_1min.iloc[0]["open"]
            assert first_bar["high"] == corresponding_1min["high"].max()
            assert first_bar["low"] == corresponding_1min["low"].min()
            assert first_bar["close"] == corresponding_1min.iloc[-1]["close"]

    def test_volume_sum(self, sample_1min_data):
        """Test that volume is summed correctly."""
        df_5min = aggregate_to_5min(sample_1min_data)

        # Get first 5-min bar
        first_bar = df_5min.iloc[0]
        start_time = df_5min.index[0]
        end_time = start_time + pd.Timedelta(minutes=5)
        mask = (sample_1min_data.index >= start_time) & (sample_1min_data.index < end_time)
        corresponding_1min = sample_1min_data[mask]

        if len(corresponding_1min) > 0:
            assert first_bar["volume"] == corresponding_1min["volume"].sum()

    def test_preserves_timezone(self, sample_1min_data):
        """Test that timezone is preserved after aggregation."""
        df_5min = aggregate_to_5min(sample_1min_data)

        assert df_5min.index.tz is not None


class TestFilterRTH:
    """Tests for filter_rth function."""

    def test_filters_to_rth_hours(self, multi_day_1min_data):
        """Test that data is filtered to 9:30 AM - 4:00 PM."""
        df_rth = filter_rth(multi_day_1min_data)

        for dt in df_rth.index:
            t = dt.time()
            assert t >= time(9, 30)
            assert t < time(16, 0)

    def test_removes_weekend(self):
        """Test that weekend data is removed."""
        # Create data with weekend
        dates = pd.date_range(
            start="2024-01-05 10:00:00",  # Friday
            periods=100,
            freq="1h",
            tz=NY_TZ,
        )
        df = pd.DataFrame({
            "open": np.random.random(100) * 100 + 4800,
            "high": np.random.random(100) * 100 + 4800,
            "low": np.random.random(100) * 100 + 4800,
            "close": np.random.random(100) * 100 + 4800,
            "volume": np.random.randint(100, 1000, 100),
        }, index=dates)

        df_rth = filter_rth(df)

        # Check no weekend days
        for dt in df_rth.index:
            assert dt.dayofweek < 5  # 0-4 = Mon-Fri

    def test_custom_rth_hours(self, multi_day_1min_data):
        """Test filtering with custom RTH hours."""
        df_rth = filter_rth(
            multi_day_1min_data,
            start_time=time(10, 0),
            end_time=time(15, 0),
        )

        for dt in df_rth.index:
            t = dt.time()
            assert t >= time(10, 0)
            assert t < time(15, 0)


class TestCreateTemporalSplits:
    """Tests for create_temporal_splits function."""

    def test_no_overlap(self, multi_day_1min_data):
        """Test that splits have no overlap."""
        # Use smaller date ranges for test data
        train, val, test = create_temporal_splits(
            multi_day_1min_data,
            train_start="2024-01-02",
            train_end="2024-01-02",
            val_start="2024-01-03",
            val_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-04",
        )

        if len(train) > 0 and len(val) > 0:
            assert train.index.max() < val.index.min()
        if len(val) > 0 and len(test) > 0:
            assert val.index.max() < test.index.min()

    def test_chronological_order(self, multi_day_1min_data):
        """Test that splits are in chronological order."""
        train, val, test = create_temporal_splits(
            multi_day_1min_data,
            train_start="2024-01-02",
            train_end="2024-01-02",
            val_start="2024-01-03",
            val_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-04",
        )

        # Each split should be sorted
        if len(train) > 0:
            assert train.index.is_monotonic_increasing
        if len(val) > 0:
            assert val.index.is_monotonic_increasing
        if len(test) > 0:
            assert test.index.is_monotonic_increasing

    def test_empty_splits_handled(self, sample_1min_data):
        """Test that empty date ranges are handled gracefully."""
        train, val, test = create_temporal_splits(
            sample_1min_data,
            train_start="2020-01-01",
            train_end="2020-01-01",
            val_start="2021-01-01",
            val_end="2021-01-01",
            test_start="2022-01-01",
            test_end="2022-01-01",
        )

        # Should return empty DataFrames without error
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


class TestScalpingDataPipeline:
    """Tests for ScalpingDataPipeline class."""

    def test_init_with_path(self, sample_csv_file):
        """Test initialization with data path."""
        pipeline = ScalpingDataPipeline(data_path=sample_csv_file)

        assert pipeline.config.data_path == sample_csv_file

    def test_init_with_config(self, sample_csv_file):
        """Test initialization with DataConfig."""
        config = DataConfig(
            data_path=sample_csv_file,
            train_start="2024-01-02",
            train_end="2024-01-02",
        )
        pipeline = ScalpingDataPipeline(config=config)

        assert pipeline.config.train_start == "2024-01-02"

    def test_init_requires_path_or_config(self):
        """Test that either path or config is required."""
        with pytest.raises(ValueError):
            ScalpingDataPipeline()

    def test_load_raw_data(self, sample_csv_file):
        """Test loading raw 1-minute data."""
        pipeline = ScalpingDataPipeline(data_path=sample_csv_file)
        df = pipeline.load_raw_data()

        assert df is not None
        assert len(df) > 0

    def test_aggregate(self, sample_csv_file):
        """Test aggregation to 5-minute bars."""
        pipeline = ScalpingDataPipeline(data_path=sample_csv_file)
        df_5min = pipeline.aggregate()

        assert df_5min is not None
        # Should have fewer rows than raw data
        assert len(df_5min) < len(pipeline.load_raw_data())

    def test_filter_rth_data(self, sample_csv_file):
        """Test RTH filtering."""
        pipeline = ScalpingDataPipeline(data_path=sample_csv_file)
        df_rth = pipeline.filter_rth_data()

        assert df_rth is not None
        # All times should be within RTH
        for dt in df_rth.index:
            t = dt.time()
            assert t >= time(9, 30)
            assert t < time(16, 0)

    def test_load_and_split(self, sample_csv_file):
        """Test full pipeline: load, aggregate, filter, split."""
        config = DataConfig(
            data_path=sample_csv_file,
            train_start="2024-01-02",
            train_end="2024-01-02",
            val_start="2024-01-03",
            val_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-04",
        )
        pipeline = ScalpingDataPipeline(config=config)
        train, val, test = pipeline.load_and_split()

        # Should get three DataFrames
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_get_stats(self, sample_csv_file):
        """Test getting data statistics."""
        pipeline = ScalpingDataPipeline(data_path=sample_csv_file)
        pipeline.load_raw_data()
        pipeline.aggregate()
        pipeline.filter_rth_data()

        stats = pipeline.get_stats()

        assert "data_path" in stats
        assert "raw_1min_bars" in stats
        assert "aggregated_5min_bars" in stats
        assert "rth_5min_bars" in stats

    def test_caching(self, sample_csv_file):
        """Test that data is cached and not reloaded."""
        pipeline = ScalpingDataPipeline(data_path=sample_csv_file)

        # Load twice
        df1 = pipeline.load_raw_data()
        df2 = pipeline.load_raw_data()

        # Should be same object (cached)
        assert df1 is df2
