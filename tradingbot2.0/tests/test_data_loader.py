"""
Tests for FuturesDataLoader and data loading utilities.

Tests cover:
- File loading (CSV/TXT formats)
- Data validation (missing values, negative prices, OHLC validity)
- Resampling to daily bars
- Target variable creation
- Train/test splitting
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))

from data.data_loader import FuturesDataLoader, load_and_prepare_data


class TestFuturesDataLoaderInit:
    """Tests for FuturesDataLoader initialization."""

    def test_init_with_valid_path(self, sample_csv_data):
        """Test initialization with a valid file path."""
        loader = FuturesDataLoader(sample_csv_data)
        assert loader.data_path == Path(sample_csv_data)
        assert loader.raw_data is None
        assert loader.daily_data is None

    def test_init_stores_path_as_pathlib(self, sample_csv_data):
        """Test that path is stored as pathlib.Path."""
        loader = FuturesDataLoader(sample_csv_data)
        assert isinstance(loader.data_path, Path)


class TestLoadRawData:
    """Tests for loading raw OHLCV data."""

    def test_load_csv_file(self, sample_csv_data):
        """Test loading a CSV file."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        assert df is not None
        assert len(df) == 1000
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_data_is_sorted(self, sample_csv_data):
        """Test that loaded data is sorted by datetime index."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        # Check that index is sorted ascending
        assert df.index.is_monotonic_increasing

    def test_load_data_dtypes(self, sample_csv_data):
        """Test that loaded data has correct dtypes."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        assert df['open'].dtype == np.float64
        assert df['high'].dtype == np.float64
        assert df['low'].dtype == np.float64
        assert df['close'].dtype == np.float64
        assert df['volume'].dtype == np.int64

    def test_load_data_stores_in_instance(self, sample_csv_data):
        """Test that loaded data is stored in instance."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        assert loader.raw_data is not None
        assert len(loader.raw_data) == len(df)

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Test that loading nonexistent file raises an error."""
        loader = FuturesDataLoader(str(tmp_path / "nonexistent.csv"))

        with pytest.raises(Exception):
            loader.load_raw_data()


class TestDataValidation:
    """Tests for data validation and cleaning."""

    def test_validation_filters_negative_prices(self, sample_csv_with_issues):
        """Test that negative prices are filtered out."""
        loader = FuturesDataLoader(sample_csv_with_issues)
        df = loader.load_raw_data()

        # Should have filtered out the row with negative open price
        assert (df['open'] > 0).all()
        assert (df['high'] > 0).all()
        assert (df['low'] > 0).all()
        assert (df['close'] > 0).all()

    def test_validation_filters_invalid_ohlc(self, sample_csv_with_issues):
        """Test that invalid OHLC relationships are filtered."""
        loader = FuturesDataLoader(sample_csv_with_issues)
        df = loader.load_raw_data()

        # Check OHLC validity: high >= low, high >= open/close, low <= open/close
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()

    def test_validation_handles_negative_volume(self, sample_csv_with_issues):
        """Test that negative volumes are set to 0."""
        loader = FuturesDataLoader(sample_csv_with_issues)
        df = loader.load_raw_data()

        # Negative volumes should be set to 0
        assert (df['volume'] >= 0).all()

    def test_validation_result_size(self, sample_csv_with_issues):
        """Test that validation filters result in fewer rows."""
        loader = FuturesDataLoader(sample_csv_with_issues)
        df = loader.load_raw_data()

        # Original file has 5 rows, some should be filtered
        assert len(df) < 5  # At least negative price row should be filtered


class TestResampleToDaily:
    """Tests for daily resampling."""

    def test_resample_creates_daily_bars(self, sample_csv_data):
        """Test that resampling creates daily bars."""
        loader = FuturesDataLoader(sample_csv_data)
        loader.load_raw_data()
        daily = loader.resample_to_daily()

        assert daily is not None
        # All 1000 minutes are within one day, so should have 1 daily bar
        assert len(daily) >= 1

    def test_resample_aggregation_correctness(self, tmp_path):
        """Test that OHLCV aggregation is correct."""
        # Create data with known values across 2 days
        file_path = tmp_path / "test_multi_day.csv"
        with open(file_path, 'w') as f:
            # Day 1: open=100, high=110, low=90, close=105, volume=1000
            f.write("2024-01-02 09:30:00,100.00,105.00,95.00,102.00,300\n")
            f.write("2024-01-02 10:30:00,102.00,110.00,100.00,108.00,400\n")
            f.write("2024-01-02 11:30:00,108.00,109.00,90.00,105.00,300\n")
            # Day 2: open=105, high=120, low=100, close=115, volume=500
            f.write("2024-01-03 09:30:00,105.00,115.00,100.00,110.00,200\n")
            f.write("2024-01-03 10:30:00,110.00,120.00,105.00,115.00,300\n")

        loader = FuturesDataLoader(str(file_path))
        loader.load_raw_data()
        daily = loader.resample_to_daily()

        # Check day 1 aggregation
        day1 = daily.loc['2024-01-02']
        assert day1['open'] == 100.0  # First open of day
        assert day1['high'] == 110.0  # Max high of day
        assert day1['low'] == 90.0    # Min low of day
        assert day1['close'] == 105.0 # Last close of day
        assert day1['volume'] == 1000 # Sum of volumes

    def test_resample_loads_data_if_not_loaded(self, sample_csv_data):
        """Test that resample loads data if not already loaded."""
        loader = FuturesDataLoader(sample_csv_data)
        # Don't call load_raw_data first
        daily = loader.resample_to_daily()

        assert daily is not None
        assert loader.raw_data is not None

    def test_resample_stores_in_instance(self, sample_csv_data):
        """Test that daily data is stored in instance."""
        loader = FuturesDataLoader(sample_csv_data)
        loader.load_raw_data()
        daily = loader.resample_to_daily()

        assert loader.daily_data is not None
        assert len(loader.daily_data) == len(daily)


class TestCreateTargetVariable:
    """Tests for target variable creation."""

    def test_target_is_binary(self, sample_csv_data):
        """Test that target variable is binary (0 or 1)."""
        loader = FuturesDataLoader(sample_csv_data)
        loader.load_raw_data()
        daily = loader.resample_to_daily()
        df = loader.create_target_variable(daily)

        assert 'target' in df.columns
        assert set(df['target'].unique()).issubset({0, 1})

    def test_target_logic_correctness(self, sample_daily_ohlcv):
        """Test that target logic is correct (1 if next close > current close)."""
        loader = FuturesDataLoader("")  # Path doesn't matter here
        df = loader.create_target_variable(sample_daily_ohlcv.copy())

        # Manually verify a few targets
        for i in range(min(10, len(df) - 1)):
            expected = 1 if sample_daily_ohlcv.iloc[i + 1]['close'] > sample_daily_ohlcv.iloc[i]['close'] else 0
            assert df.iloc[i]['target'] == expected

    def test_target_removes_last_row(self, sample_daily_ohlcv):
        """Test that target creation removes last row (no future data)."""
        loader = FuturesDataLoader("")
        df = loader.create_target_variable(sample_daily_ohlcv.copy())

        # Should have one fewer row than original
        assert len(df) == len(sample_daily_ohlcv) - 1

    def test_target_with_lookahead(self, sample_daily_ohlcv):
        """Test target creation with different lookahead periods."""
        loader = FuturesDataLoader("")
        df = loader.create_target_variable(sample_daily_ohlcv.copy(), lookahead=2)

        # Should have two fewer rows than original
        assert len(df) == len(sample_daily_ohlcv) - 2

    def test_next_return_column_added(self, sample_daily_ohlcv):
        """Test that next_return column is added."""
        loader = FuturesDataLoader("")
        df = loader.create_target_variable(sample_daily_ohlcv.copy())

        assert 'next_return' in df.columns
        # next_return should be the actual return
        assert not df['next_return'].isna().any()

    def test_target_distribution_is_reasonable(self, sample_daily_ohlcv):
        """Test that target distribution is reasonable (not all 0 or all 1)."""
        loader = FuturesDataLoader("")
        df = loader.create_target_variable(sample_daily_ohlcv.copy())

        # With random data, should have roughly balanced targets
        target_mean = df['target'].mean()
        assert 0.1 < target_mean < 0.9  # Not too imbalanced


class TestTrainTestSplit:
    """Tests for train/test splitting."""

    def test_split_ratio_correctness(self, sample_daily_ohlcv):
        """Test that split respects the given ratio."""
        loader = FuturesDataLoader("")
        train_df, test_df, val_df = loader.train_test_split(sample_daily_ohlcv, train_ratio=0.8)

        expected_train_size = int(len(sample_daily_ohlcv) * 0.8)
        assert len(train_df) == expected_train_size
        assert len(test_df) == len(sample_daily_ohlcv) - expected_train_size
        assert val_df is None

    def test_split_preserves_temporal_order(self, sample_daily_ohlcv):
        """Test that split preserves temporal ordering."""
        loader = FuturesDataLoader("")
        train_df, test_df, _ = loader.train_test_split(sample_daily_ohlcv, train_ratio=0.8)

        # All training dates should be before all test dates
        assert train_df.index.max() < test_df.index.min()

    def test_split_with_validation(self, sample_daily_ohlcv):
        """Test split with validation set."""
        loader = FuturesDataLoader("")
        train_df, test_df, val_df = loader.train_test_split(
            sample_daily_ohlcv, train_ratio=0.6, validation_ratio=0.2
        )

        assert val_df is not None
        # Total should equal original
        assert len(train_df) + len(test_df) + len(val_df) == len(sample_daily_ohlcv)

    def test_split_temporal_order_with_validation(self, sample_daily_ohlcv):
        """Test temporal order: train < val < test."""
        loader = FuturesDataLoader("")
        train_df, test_df, val_df = loader.train_test_split(
            sample_daily_ohlcv, train_ratio=0.6, validation_ratio=0.2
        )

        # train < val < test
        assert train_df.index.max() < val_df.index.min()
        assert val_df.index.max() < test_df.index.min()

    def test_split_no_data_leakage(self, sample_daily_ohlcv):
        """Test that there's no overlap between train, val, and test."""
        loader = FuturesDataLoader("")
        train_df, test_df, val_df = loader.train_test_split(
            sample_daily_ohlcv, train_ratio=0.6, validation_ratio=0.2
        )

        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        val_indices = set(val_df.index)

        # No overlap
        assert len(train_indices & test_indices) == 0
        assert len(train_indices & val_indices) == 0
        assert len(test_indices & val_indices) == 0


class TestLoadAndPrepareData:
    """Tests for the convenience function."""

    def test_load_and_prepare_returns_three_dfs(self, sample_csv_data):
        """Test that function returns three DataFrames."""
        full_df, train_df, test_df = load_and_prepare_data(sample_csv_data)

        assert isinstance(full_df, pd.DataFrame)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_load_and_prepare_has_target(self, sample_csv_data):
        """Test that returned DataFrames have target column."""
        full_df, train_df, test_df = load_and_prepare_data(sample_csv_data)

        assert 'target' in full_df.columns
        assert 'target' in train_df.columns
        assert 'target' in test_df.columns

    def test_load_and_prepare_split_correctness(self, sample_csv_data):
        """Test that split sizes are correct."""
        full_df, train_df, test_df = load_and_prepare_data(sample_csv_data, train_ratio=0.8)

        expected_train_size = int(len(full_df) * 0.8)
        assert len(train_df) == expected_train_size


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_row_data(self, tmp_path):
        """Test handling of single row data."""
        file_path = tmp_path / "single_row.csv"
        with open(file_path, 'w') as f:
            f.write("2024-01-02 09:30:00,100.00,101.00,99.00,100.50,100\n")

        loader = FuturesDataLoader(str(file_path))
        df = loader.load_raw_data()
        assert len(df) == 1

    def test_empty_file_returns_empty_df(self, tmp_path):
        """Test that empty file returns empty DataFrame without error."""
        file_path = tmp_path / "empty.csv"
        file_path.touch()

        loader = FuturesDataLoader(str(file_path))
        df = loader.load_raw_data()
        # Pandas reads empty file as empty DataFrame
        assert len(df) == 0

    def test_high_train_ratio(self, sample_daily_ohlcv):
        """Test with very high train ratio."""
        loader = FuturesDataLoader("")
        train_df, test_df, _ = loader.train_test_split(sample_daily_ohlcv, train_ratio=0.99)

        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_low_train_ratio(self, sample_daily_ohlcv):
        """Test with low train ratio."""
        loader = FuturesDataLoader("")
        train_df, test_df, _ = loader.train_test_split(sample_daily_ohlcv, train_ratio=0.1)

        assert len(train_df) > 0
        assert len(test_df) > 0


class TestDataIntegrity:
    """Tests for data integrity after processing."""

    def test_no_nan_in_ohlcv(self, sample_csv_data):
        """Test that processed data has no NaN in OHLCV columns."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        assert not df['open'].isna().any()
        assert not df['high'].isna().any()
        assert not df['low'].isna().any()
        assert not df['close'].isna().any()
        assert not df['volume'].isna().any()

    def test_ohlc_relationships_preserved(self, sample_csv_data):
        """Test that OHLC relationships are valid after processing."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['high'] >= df['low']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()

    def test_volume_non_negative(self, sample_csv_data):
        """Test that volume is non-negative after processing."""
        loader = FuturesDataLoader(sample_csv_data)
        df = loader.load_raw_data()

        assert (df['volume'] >= 0).all()
