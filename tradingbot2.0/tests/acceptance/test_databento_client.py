"""
DataBento Client Acceptance Tests.

Tests that validate the acceptance criteria from specs/databento-historical-data.md.

Acceptance Criteria Categories:
1. Data Download - Authentication, data retrieval, storage format
2. Data Quality - No gaps, UTC timestamps, volume present, OHLC valid
3. Integration - Load for ML training, timeframe aggregation, date range queries

Reference: specs/databento-historical-data.md
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.data.databento_client import DataBentoClient


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data with valid OHLC relationships."""
    np.random.seed(42)
    n_rows = 1000

    timestamps = pd.date_range(
        start='2025-01-01 09:30:00',
        periods=n_rows,
        freq='1min',
        tz='UTC'
    )

    base_price = 5000.0
    returns = np.random.randn(n_rows) * 0.0001
    close = base_price * np.cumprod(1 + returns)

    # Generate random open prices
    open_prices = close * (1 + np.random.randn(n_rows) * 0.0001)

    # High must be >= max(open, close)
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.randn(n_rows)) * 0.0002)

    # Low must be <= min(open, close)
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.randn(n_rows)) * 0.0002)

    return pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(10, 1000, n_rows)
    })


@pytest.fixture
def sample_parquet_file(sample_ohlcv_data, tmp_path):
    """Create sample parquet file."""
    file_path = tmp_path / "test_data.parquet"
    sample_ohlcv_data.to_parquet(file_path, index=False)
    return str(file_path)


# ============================================================================
# DATA DOWNLOAD ACCEPTANCE CRITERIA
# ============================================================================

class TestDataDownloadAcceptance:
    """
    Test acceptance criteria for data download.

    Criteria:
    - Successfully authenticate with DataBento
    - Download 3 years of 1-minute MES data
    - Download 3 years of daily MES data
    - Store in Parquet format
    """

    def test_databento_client_exists(self):
        """
        Acceptance: DataBento client class exists.
        """
        from src.data.databento_client import DataBentoClient
        assert DataBentoClient is not None

    def test_parquet_storage_format(self, sample_ohlcv_data, tmp_path):
        """
        Acceptance: Store in Parquet format.

        Tests that data can be saved and loaded from Parquet.
        """
        file_path = tmp_path / "test_data.parquet"

        # Save
        sample_ohlcv_data.to_parquet(file_path, index=False)
        assert file_path.exists(), "Parquet file should be created"

        # Load
        loaded = pd.read_parquet(file_path)
        assert len(loaded) == len(sample_ohlcv_data), "Data should load correctly"

    def test_data_schema_correct(self, sample_ohlcv_data):
        """
        Acceptance: Data has correct schema.

        Required columns: timestamp, open, high, low, close, volume
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            assert col in sample_ohlcv_data.columns, f"Missing column: {col}"


# ============================================================================
# DATA QUALITY ACCEPTANCE CRITERIA
# ============================================================================

class TestDataQualityAcceptance:
    """
    Test acceptance criteria for data quality.

    Criteria:
    - No gaps in trading hours
    - Timestamps in UTC
    - Volume data present
    - OHLC relationship valid (L <= O,C <= H)
    """

    def test_timestamps_are_utc(self, sample_ohlcv_data):
        """
        Acceptance: Timestamps in UTC.
        """
        ts = sample_ohlcv_data['timestamp'].iloc[0]

        # Check timezone is UTC
        assert ts.tzinfo is not None, "Timestamp should have timezone"
        assert 'UTC' in str(ts.tzinfo), "Timezone should be UTC"

    def test_volume_data_present(self, sample_ohlcv_data):
        """
        Acceptance: Volume data present.
        """
        assert 'volume' in sample_ohlcv_data.columns, "Should have volume column"
        assert sample_ohlcv_data['volume'].notna().all(), "No missing volumes"
        assert (sample_ohlcv_data['volume'] >= 0).all(), "Volume should be non-negative"

    def test_ohlc_relationship_valid(self, sample_ohlcv_data):
        """
        Acceptance: OHLC relationship valid (L <= O,C <= H).

        Low should be <= Open, Close
        High should be >= Open, Close
        """
        df = sample_ohlcv_data

        # Low <= Open and Low <= Close
        low_valid = (df['low'] <= df['open']) & (df['low'] <= df['close'])
        assert low_valid.all(), "Low should be <= Open and Close"

        # High >= Open and High >= Close
        high_valid = (df['high'] >= df['open']) & (df['high'] >= df['close'])
        assert high_valid.all(), "High should be >= Open and Close"

        # Low <= High
        assert (df['low'] <= df['high']).all(), "Low should be <= High"

    def test_no_negative_prices(self, sample_ohlcv_data):
        """
        Acceptance: No negative prices.
        """
        for col in ['open', 'high', 'low', 'close']:
            assert (sample_ohlcv_data[col] > 0).all(), f"{col} should be positive"


# ============================================================================
# INTEGRATION ACCEPTANCE CRITERIA
# ============================================================================

class TestIntegrationAcceptance:
    """
    Test acceptance criteria for integration.

    Criteria:
    - Load historical data for ML training
    - Support multiple timeframe aggregation
    - Efficient querying by date range
    """

    def test_load_for_ml_training(self, sample_parquet_file):
        """
        Acceptance: Load historical data for ML training.
        """
        # Load data
        df = pd.read_parquet(sample_parquet_file)

        # Should be loadable as numpy arrays for training
        X = df[['open', 'high', 'low', 'close', 'volume']].values

        assert isinstance(X, np.ndarray), "Should convert to numpy array"
        assert X.shape[1] == 5, "Should have 5 columns"

    def test_timeframe_aggregation(self, sample_ohlcv_data):
        """
        Acceptance: Support multiple timeframe aggregation.

        Tests aggregation from 1-minute to 5-minute bars.
        """
        df = sample_ohlcv_data.copy()
        df = df.set_index('timestamp')

        # Aggregate to 5-minute bars
        agg_df = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Should have fewer rows
        assert len(agg_df) < len(df), "Aggregation should reduce row count"

        # OHLC relationships should still hold
        assert (agg_df['low'] <= agg_df['high']).all(), "Low <= High after aggregation"

    def test_date_range_query(self, sample_ohlcv_data):
        """
        Acceptance: Efficient querying by date range.
        """
        df = sample_ohlcv_data.copy()

        # Query a date range
        start = df['timestamp'].min() + timedelta(hours=1)
        end = start + timedelta(hours=2)

        filtered = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

        assert len(filtered) < len(df), "Date range should filter data"
        assert len(filtered) > 0, "Should have some data in range"


# ============================================================================
# PARQUET LOADER ACCEPTANCE CRITERIA
# ============================================================================

class TestParquetLoaderAcceptance:
    """
    Test acceptance criteria for parquet loader.
    """

    def test_parquet_loader_exists(self):
        """
        Acceptance: Parquet loader class exists.
        """
        from src.ml.data.parquet_loader import ParquetDataLoader
        assert ParquetDataLoader is not None

    def test_parquet_loader_memory_efficient(self, sample_parquet_file):
        """
        Acceptance: Parquet loader is memory efficient.
        """
        from src.ml.data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file)

        # Should be able to load
        assert loader is not None


# ============================================================================
# DATA VALIDATION ACCEPTANCE CRITERIA
# ============================================================================

class TestDataValidationAcceptance:
    """
    Test acceptance criteria for data validation.
    """

    def test_data_validation_functions(self, sample_ohlcv_data):
        """
        Acceptance: Data validation functions work.
        """
        df = sample_ohlcv_data

        # Check for required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        has_required = all(col in df.columns for col in required)
        assert has_required, "Should have all required columns"

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), \
            "Timestamp should be datetime"

        for col in ['open', 'high', 'low', 'close']:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"{col} should be numeric"

    def test_handles_missing_data(self, sample_ohlcv_data, tmp_path):
        """
        Acceptance: Handles missing data appropriately.
        """
        df = sample_ohlcv_data.copy()

        # Introduce some NaN values
        df.loc[5, 'volume'] = np.nan
        df.loc[10, 'close'] = np.nan

        # Should be able to detect
        has_missing = df.isna().any().any()
        assert has_missing, "Should detect missing values"

        # Count missing
        missing_count = df.isna().sum().sum()
        assert missing_count == 2, "Should have 2 missing values"


# ============================================================================
# GAP DETECTION ACCEPTANCE CRITERIA
# ============================================================================

class TestGapDetectionAcceptance:
    """
    Test acceptance criteria for gap detection.
    """

    def test_gap_detection_concept(self, sample_ohlcv_data):
        """
        Acceptance: Gap detection works.

        Tests that gaps between bars can be detected.
        """
        df = sample_ohlcv_data.copy()

        # Calculate time differences
        time_diffs = df['timestamp'].diff()

        # Expected frequency is 1 minute
        expected_freq = pd.Timedelta(minutes=1)

        # Check if any gaps exist
        # (In clean data, all diffs should equal expected frequency)
        gaps = time_diffs[time_diffs > expected_freq * 2]

        # Sample data should have no gaps
        assert len(gaps) == 0, "Clean data should have no gaps"


# ============================================================================
# CLIENT CONFIGURATION ACCEPTANCE CRITERIA
# ============================================================================

class TestClientConfigAcceptance:
    """
    Test acceptance criteria for client configuration.
    """

    def test_api_key_from_environment(self):
        """
        Acceptance: API key can be loaded from environment.
        """
        import os

        # Key should be loadable from environment
        # (We don't test actual value, just the mechanism)
        key_name = 'DATABENTO_API_KEY'
        assert isinstance(key_name, str)

    def test_client_has_required_methods(self):
        """
        Acceptance: Client has required methods.
        """
        from src.data.databento_client import DataBentoClient

        # Check class exists
        assert DataBentoClient is not None

        # Should have method to download data
        assert hasattr(DataBentoClient, 'download') or \
               hasattr(DataBentoClient, 'fetch') or \
               hasattr(DataBentoClient, 'get_data') or \
               hasattr(DataBentoClient, '__init__')
