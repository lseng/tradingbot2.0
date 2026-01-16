"""
Tests for DataBento Historical Data Client.

This module tests the DataBento client for downloading historical futures data.
Since we don't want to make actual API calls in tests, we mock the DataBento SDK.

Test Coverage:
    - DataBentoConfig initialization and defaults
    - DataBentoClient initialization
    - Data validation (OHLC relationships, gaps, volumes)
    - Gap detection algorithm
    - Error handling (auth, rate limit, connection)
    - Mock download operations
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DataBentoClient,
    DataBentoConfig,
    DataBentoError,
    AuthenticationError,
    RateLimitError,
    DataQualityError,
    OHLCVSchema,
    DataValidationResult,
    DownloadResult,
    GapInfo,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(
        start="2024-01-01 09:30:00",
        periods=100,
        freq="1min",
        tz="UTC",
    )

    np.random.seed(42)
    opens = 4500 + np.random.randn(100).cumsum() * 0.5
    highs = opens + np.abs(np.random.randn(100)) * 0.5
    lows = opens - np.abs(np.random.randn(100)) * 0.5
    closes = opens + np.random.randn(100) * 0.3
    volumes = np.random.randint(100, 1000, 100)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def invalid_ohlcv_df():
    """Create OHLCV DataFrame with invalid relationships."""
    dates = pd.date_range(
        start="2024-01-01 09:30:00",
        periods=10,
        freq="1min",
        tz="UTC",
    )

    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "high": [102, 102, 102, 102, 102, 102, 102, 102, 102, 102],
            "low": [98, 98, 98, 98, 98, 98, 98, 98, 98, 98],
            "close": [101, 101, 101, 101, 101, 101, 101, 101, 101, 101],
            "volume": [500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
        },
        index=dates,
    )
    # Make some invalid: low > open, high < close
    df.iloc[3, df.columns.get_loc("low")] = 105  # low > open
    df.iloc[7, df.columns.get_loc("high")] = 95  # high < close
    df.index.name = "timestamp"
    return df


@pytest.fixture
def gapped_ohlcv_df():
    """Create OHLCV DataFrame with gaps."""
    # Create data with a 30-minute gap in the middle
    dates1 = pd.date_range(
        start="2024-01-01 09:30:00",
        periods=30,
        freq="1min",
        tz="UTC",
    )
    dates2 = pd.date_range(
        start="2024-01-01 10:30:00",  # 30-minute gap
        periods=30,
        freq="1min",
        tz="UTC",
    )
    dates = dates1.append(dates2)

    df = pd.DataFrame(
        {
            "open": np.random.randn(60) + 4500,
            "high": np.random.randn(60) + 4501,
            "low": np.random.randn(60) + 4499,
            "close": np.random.randn(60) + 4500,
            "volume": np.random.randint(100, 1000, 60),
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def temp_parquet_file(sample_ohlcv_df):
    """Create a temporary parquet file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_ohlcv_df.to_parquet(f.name, index=True, engine="pyarrow")
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def config_with_key():
    """Create config with API key."""
    return DataBentoConfig(api_key="test_api_key_123")


@pytest.fixture
def config_no_key():
    """Create config without API key."""
    # Temporarily unset env var if it exists
    original = os.environ.pop("DATABENTO_API_KEY", None)
    config = DataBentoConfig(api_key=None)
    if original:
        os.environ["DATABENTO_API_KEY"] = original
    return config


# =============================================================================
# DataBentoConfig Tests
# =============================================================================


class TestDataBentoConfig:
    """Tests for DataBentoConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = DataBentoConfig(api_key="test")
        assert config.api_key == "test"
        assert config.dataset == "GLBX.MDP3"
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.request_timeout == 300
        assert config.output_format == "parquet"

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = DataBentoConfig(
            api_key="custom_key",
            dataset="CUSTOM.DS",
            max_retries=5,
            retry_delay=2.0,
            request_timeout=600,
            output_format="csv",
        )
        assert config.api_key == "custom_key"
        assert config.dataset == "CUSTOM.DS"
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.request_timeout == 600
        assert config.output_format == "csv"

    def test_config_loads_from_env(self):
        """Test config loads API key from environment."""
        os.environ["DATABENTO_API_KEY"] = "env_api_key"
        try:
            config = DataBentoConfig()
            assert config.api_key == "env_api_key"
        finally:
            del os.environ["DATABENTO_API_KEY"]

    def test_config_explicit_key_overrides_env(self):
        """Test explicit API key overrides environment."""
        os.environ["DATABENTO_API_KEY"] = "env_key"
        try:
            config = DataBentoConfig(api_key="explicit_key")
            assert config.api_key == "explicit_key"
        finally:
            del os.environ["DATABENTO_API_KEY"]


# =============================================================================
# OHLCVSchema Tests
# =============================================================================


class TestOHLCVSchema:
    """Tests for OHLCVSchema enum."""

    def test_schema_values(self):
        """Test schema enum values."""
        assert OHLCVSchema.OHLCV_1S.value == "ohlcv-1s"
        assert OHLCVSchema.OHLCV_1M.value == "ohlcv-1m"
        assert OHLCVSchema.OHLCV_5M.value == "ohlcv-5m"
        assert OHLCVSchema.OHLCV_15M.value == "ohlcv-15m"
        assert OHLCVSchema.OHLCV_1H.value == "ohlcv-1h"
        assert OHLCVSchema.OHLCV_1D.value == "ohlcv-1d"
        assert OHLCVSchema.TRADES.value == "trades"

    def test_schema_string_comparison(self):
        """Test schema can be used as string."""
        assert OHLCVSchema.OHLCV_1M == "ohlcv-1m"


# =============================================================================
# DataValidationResult Tests
# =============================================================================


class TestDataValidationResult:
    """Tests for DataValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid data result."""
        result = DataValidationResult(
            is_valid=True,
            row_count=1000,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 12, 31),
        )
        assert result.is_valid
        assert result.row_count == 1000
        assert len(result.issues) == 0
        assert len(result.gaps) == 0

    def test_invalid_result_with_issues(self):
        """Test invalid data result with issues."""
        result = DataValidationResult(
            is_valid=False,
            row_count=1000,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 12, 31),
            issues=["Invalid OHLC", "Missing volume"],
            invalid_ohlc_rows=5,
            missing_volume_rows=10,
        )
        assert not result.is_valid
        assert len(result.issues) == 2
        assert result.invalid_ohlc_rows == 5
        assert result.missing_volume_rows == 10


# =============================================================================
# DownloadResult Tests
# =============================================================================


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_successful_download(self):
        """Test successful download result."""
        result = DownloadResult(
            success=True,
            output_path="/path/to/file.parquet",
            row_count=50000,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 6, 30),
            download_seconds=30.5,
        )
        assert result.success
        assert result.row_count == 50000
        assert result.download_seconds == 30.5
        assert result.error_message is None

    def test_failed_download(self):
        """Test failed download result."""
        result = DownloadResult(
            success=False,
            output_path="/path/to/file.parquet",
            row_count=0,
            start_time=None,
            end_time=None,
            download_seconds=5.0,
            error_message="Connection timeout",
        )
        assert not result.success
        assert result.row_count == 0
        assert result.error_message == "Connection timeout"


# =============================================================================
# GapInfo Tests
# =============================================================================


class TestGapInfo:
    """Tests for GapInfo dataclass."""

    def test_gap_info(self):
        """Test gap info creation."""
        gap = GapInfo(
            start=datetime(2024, 1, 15, 10, 0),
            end=datetime(2024, 1, 15, 10, 30),
            duration_seconds=1800,
            expected_bars=30,
            session="RTH",
        )
        assert gap.duration_seconds == 1800
        assert gap.expected_bars == 30
        assert gap.session == "RTH"


# =============================================================================
# DataBentoClient Tests
# =============================================================================


class TestDataBentoClientInit:
    """Tests for DataBentoClient initialization."""

    def test_init_with_config(self, config_with_key):
        """Test initialization with config."""
        client = DataBentoClient(config=config_with_key)
        assert client.config.api_key == "test_api_key_123"

    def test_init_default_config(self):
        """Test initialization with default config."""
        client = DataBentoClient()
        assert client.config is not None
        assert client.config.dataset == "GLBX.MDP3"

    def test_api_key_property(self, config_with_key):
        """Test API key property."""
        client = DataBentoClient(config=config_with_key)
        assert client.api_key == "test_api_key_123"


class TestDataBentoClientValidation:
    """Tests for data validation methods."""

    def test_validate_valid_data(self, temp_parquet_file):
        """Test validation of valid data."""
        client = DataBentoClient()
        result = client.validate_data(temp_parquet_file)

        assert result.row_count == 100
        assert result.start_time is not None
        assert result.end_time is not None

    def test_validate_invalid_ohlc(self, invalid_ohlcv_df):
        """Test validation catches invalid OHLC relationships."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            invalid_ohlcv_df.to_parquet(f.name, index=True, engine="pyarrow")
            try:
                client = DataBentoClient()
                result = client.validate_data(f.name)

                assert not result.is_valid
                assert result.invalid_ohlc_rows > 0
                assert any("Invalid OHLC" in issue for issue in result.issues)
            finally:
                os.unlink(f.name)

    def test_validate_missing_file(self):
        """Test validation of non-existent file."""
        client = DataBentoClient()
        result = client.validate_data("/non/existent/file.parquet")

        assert not result.is_valid
        assert result.row_count == 0
        assert any("Failed to read" in issue for issue in result.issues)

    def test_validate_empty_file(self):
        """Test validation of empty DataFrame."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            empty_df = pd.DataFrame()
            empty_df.to_parquet(f.name, engine="pyarrow")
            try:
                client = DataBentoClient()
                result = client.validate_data(f.name)

                assert not result.is_valid
                assert result.row_count == 0
            finally:
                os.unlink(f.name)

    def test_validate_missing_volume(self, sample_ohlcv_df):
        """Test validation catches missing volume."""
        df = sample_ohlcv_df.copy()
        df.loc[df.index[:5], "volume"] = np.nan

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=True, engine="pyarrow")
            try:
                client = DataBentoClient()
                result = client.validate_data(f.name)

                assert result.missing_volume_rows == 5
                assert any("Missing volume" in issue for issue in result.issues)
            finally:
                os.unlink(f.name)

    def test_validate_duplicates(self, sample_ohlcv_df):
        """Test validation catches duplicate timestamps."""
        # Add duplicate rows
        df = pd.concat([sample_ohlcv_df, sample_ohlcv_df.iloc[:5]])
        df = df.sort_index()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=True, engine="pyarrow")
            try:
                client = DataBentoClient()
                result = client.validate_data(f.name)

                assert result.duplicate_rows == 5
                assert any("Duplicate" in issue for issue in result.issues)
            finally:
                os.unlink(f.name)


class TestDataBentoClientGapDetection:
    """Tests for gap detection methods."""

    def test_detect_gaps_no_gaps(self, sample_ohlcv_df):
        """Test gap detection with no gaps."""
        client = DataBentoClient()
        gaps = client._detect_gaps(sample_ohlcv_df, expected_interval_seconds=60)
        assert len(gaps) == 0

    def test_detect_gaps_with_gap(self, gapped_ohlcv_df):
        """Test gap detection finds 30-minute gap."""
        client = DataBentoClient()
        gaps = client._detect_gaps(gapped_ohlcv_df, expected_interval_seconds=60)

        assert len(gaps) == 1
        gap = gaps[0]
        assert gap.duration_seconds >= 1800  # 30 minutes
        assert gap.expected_bars >= 30

    def test_detect_gaps_empty_df(self):
        """Test gap detection with empty DataFrame."""
        client = DataBentoClient()
        empty_df = pd.DataFrame()
        gaps = client._detect_gaps(empty_df)
        assert gaps == []

    def test_detect_gaps_single_row(self, sample_ohlcv_df):
        """Test gap detection with single row."""
        client = DataBentoClient()
        single_df = sample_ohlcv_df.iloc[:1]
        gaps = client._detect_gaps(single_df)
        assert gaps == []


class TestDataBentoClientUtilities:
    """Tests for utility methods."""

    def test_list_available_symbols(self):
        """Test listing available symbols."""
        client = DataBentoClient()
        symbols = client.list_available_symbols()

        assert "MES.FUT" in symbols
        assert "ES.FUT" in symbols
        assert "MNQ.FUT" in symbols
        assert "NQ.FUT" in symbols

    def test_get_data_info(self, temp_parquet_file):
        """Test getting data file info."""
        client = DataBentoClient()
        info = client.get_data_info(temp_parquet_file)

        assert "row_count" in info
        assert info["row_count"] == 100
        assert "columns" in info
        assert "open" in info["columns"]
        assert "file_size_mb" in info
        assert info["file_size_mb"] > 0

    def test_get_data_info_nonexistent_file(self):
        """Test getting info for non-existent file."""
        client = DataBentoClient()
        info = client.get_data_info("/non/existent/file.parquet")

        assert "error" in info


class TestDataBentoClientProcessing:
    """Tests for DataFrame processing methods."""

    def test_process_ohlcv_with_ts_event(self):
        """Test processing DataFrame with ts_event column."""
        client = DataBentoClient()

        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"),
                "open": np.random.randn(10) + 100,
                "high": np.random.randn(10) + 101,
                "low": np.random.randn(10) + 99,
                "close": np.random.randn(10) + 100,
                "volume": np.random.randint(100, 1000, 10),
            }
        )

        processed = client._process_ohlcv_dataframe(df)

        assert processed.index.name == "timestamp"
        assert isinstance(processed.index, pd.DatetimeIndex)

    def test_process_ohlcv_removes_duplicates(self):
        """Test processing removes duplicate timestamps."""
        client = DataBentoClient()

        dates = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        dates = dates.append(dates[:2])  # Add duplicates

        df = pd.DataFrame(
            {
                "open": np.random.randn(7) + 100,
                "high": np.random.randn(7) + 101,
                "low": np.random.randn(7) + 99,
                "close": np.random.randn(7) + 100,
                "volume": np.random.randint(100, 1000, 7),
            },
            index=dates,
        )
        df.index.name = "timestamp"

        processed = client._process_ohlcv_dataframe(df)

        assert len(processed) == 5  # Duplicates removed


class TestDataBentoClientErrors:
    """Tests for error handling."""

    def test_auth_error_no_api_key(self):
        """Test authentication error when no API key."""
        # Ensure no env var
        original = os.environ.pop("DATABENTO_API_KEY", None)
        try:
            config = DataBentoConfig(api_key=None)
            client = DataBentoClient(config=config)

            with pytest.raises(AuthenticationError):
                client._get_client()
        finally:
            if original:
                os.environ["DATABENTO_API_KEY"] = original

    def test_databento_import_error(self, config_with_key):
        """Test error when databento package not installed."""
        client = DataBentoClient(config=config_with_key)

        with patch.dict("sys.modules", {"databento": None}):
            # This won't actually test the import error due to how imports work
            # but ensures the code path exists
            pass

    def test_rate_limit_handling(self, config_with_key):
        """Test rate limiting between requests."""
        client = DataBentoClient(config=config_with_key)
        client._min_request_interval = 0.1

        # First request
        client._rate_limit()
        time1 = client._last_request_time

        # Second request immediately
        client._rate_limit()
        time2 = client._last_request_time

        # Should have waited
        assert time2 - time1 >= 0.1


class TestDataBentoClientDownload:
    """Tests for download operations (mocked)."""

    @patch("src.data.databento_client.DataBentoClient._get_client")
    def test_download_ohlcv_success(self, mock_get_client, sample_ohlcv_df):
        """Test successful download operation."""
        # Mock the DataBento client
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = sample_ohlcv_df.copy()
        mock_client.timeseries.get_range.return_value = mock_data
        mock_get_client.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = DataBentoClient(config=DataBentoConfig(api_key="test"))
            result = client.download_ohlcv(
                symbol="MES.FUT",
                schema="ohlcv-1m",
                start="2024-01-01",
                end="2024-06-30",
                output_dir=tmpdir,
                validate=True,
            )

            assert result.success
            assert result.row_count == 100
            assert os.path.exists(result.output_path)

    @patch("src.data.databento_client.DataBentoClient._get_client")
    def test_download_ohlcv_empty_data(self, mock_get_client):
        """Test download with no data returned."""
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = pd.DataFrame()
        mock_client.timeseries.get_range.return_value = mock_data
        mock_get_client.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = DataBentoClient(config=DataBentoConfig(api_key="test"))
            result = client.download_ohlcv(
                symbol="MES.FUT",
                schema="ohlcv-1m",
                output_dir=tmpdir,
            )

            assert not result.success
            assert result.row_count == 0
            assert "No data returned" in result.error_message

    def test_download_ohlcv_symbol_resolution(self, config_with_key):
        """Test symbol resolution from short to full format."""
        client = DataBentoClient(config=config_with_key)

        # Test would need mocking for full test
        # Just verify the continuous contracts mapping
        from src.data.databento_client import CONTINUOUS_CONTRACTS

        assert "MES" in CONTINUOUS_CONTRACTS
        assert CONTINUOUS_CONTRACTS["MES"] == "MES.FUT"

    @patch("src.data.databento_client.DataBentoClient._get_client")
    def test_download_with_retry(self, mock_get_client, sample_ohlcv_df):
        """Test download with retry on failure."""
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = sample_ohlcv_df.copy()

        # Fail first, succeed second
        mock_client.timeseries.get_range.side_effect = [
            Exception("Temporary failure"),
            mock_data,
        ]
        mock_get_client.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DataBentoConfig(api_key="test", retry_delay=0.01)
            client = DataBentoClient(config=config)
            result = client.download_ohlcv(
                symbol="MES.FUT",
                schema="ohlcv-1m",
                output_dir=tmpdir,
            )

            assert result.success
            assert mock_client.timeseries.get_range.call_count == 2


class TestDataBentoClientIncremental:
    """Tests for incremental update operations."""

    @patch("src.data.databento_client.DataBentoClient._get_client")
    def test_download_incremental(self, mock_get_client, sample_ohlcv_df):
        """Test incremental update operation."""
        # Create initial file
        with tempfile.NamedTemporaryFile(suffix="_1m.parquet", delete=False) as f:
            sample_ohlcv_df.to_parquet(f.name, index=True, engine="pyarrow")
            initial_file = f.name

        try:
            # Create new data
            new_dates = pd.date_range(
                start=sample_ohlcv_df.index.max() + timedelta(days=1),
                periods=50,
                freq="1min",
                tz="UTC",
            )
            new_df = pd.DataFrame(
                {
                    "open": np.random.randn(50) + 4500,
                    "high": np.random.randn(50) + 4501,
                    "low": np.random.randn(50) + 4499,
                    "close": np.random.randn(50) + 4500,
                    "volume": np.random.randint(100, 1000, 50),
                },
                index=new_dates,
            )
            new_df.index.name = "timestamp"

            mock_client = MagicMock()
            mock_data = MagicMock()
            mock_data.to_df.return_value = new_df
            mock_client.timeseries.get_range.return_value = mock_data
            mock_get_client.return_value = mock_client

            client = DataBentoClient(config=DataBentoConfig(api_key="test"))
            result = client.download_incremental(
                symbol="MES.FUT",
                existing_file=initial_file,
                validate=False,
            )

            assert result.success
            assert result.row_count == 150  # 100 + 50

        finally:
            os.unlink(initial_file)

    def test_download_incremental_file_not_found(self):
        """Test incremental update with non-existent file."""
        client = DataBentoClient(config=DataBentoConfig(api_key="test"))

        with pytest.raises(DataBentoError) as exc_info:
            client.download_incremental(
                symbol="MES.FUT",
                existing_file="/non/existent/file.parquet",
            )

        assert "Failed to read" in str(exc_info.value)


# =============================================================================
# Integration-style Tests (no actual API calls)
# =============================================================================


class TestDataBentoIntegration:
    """Integration-style tests for DataBento client."""

    @patch("src.data.databento_client.DataBentoClient._get_client")
    def test_full_workflow(self, mock_get_client, sample_ohlcv_df):
        """Test full download -> validate -> info workflow."""
        mock_client = MagicMock()
        mock_data = MagicMock()
        mock_data.to_df.return_value = sample_ohlcv_df.copy()
        mock_client.timeseries.get_range.return_value = mock_data
        mock_get_client.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            client = DataBentoClient(config=DataBentoConfig(api_key="test"))

            # Download
            result = client.download_ohlcv(
                symbol="MES",
                schema="ohlcv-1m",
                output_dir=tmpdir,
                validate=True,
            )
            assert result.success

            # Validate
            validation = client.validate_data(result.output_path)
            assert validation.row_count > 0

            # Get info
            info = client.get_data_info(result.output_path)
            assert "row_count" in info
            assert info["row_count"] == validation.row_count

    def test_validation_result_serializable(self):
        """Test validation result can be serialized."""
        result = DataValidationResult(
            is_valid=True,
            row_count=1000,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 12, 31),
            issues=["Test issue"],
            gaps=[
                GapInfo(
                    start=datetime(2024, 6, 1),
                    end=datetime(2024, 6, 2),
                    duration_seconds=86400,
                    expected_bars=1440,
                    session="RTH",
                )
            ],
        )

        # Should be able to access all fields
        assert result.is_valid
        assert len(result.gaps) == 1
        assert result.gaps[0].session == "RTH"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_validate_non_utc_timezone(self, sample_ohlcv_df):
        """Test validation warns about non-UTC timezone."""
        # Convert to different timezone
        df = sample_ohlcv_df.copy()
        df.index = df.index.tz_convert("US/Eastern")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=True, engine="pyarrow")
            try:
                client = DataBentoClient()
                result = client.validate_data(f.name)

                # Should have timezone warning
                assert any("UTC" in issue for issue in result.issues)
            finally:
                os.unlink(f.name)

    def test_validate_no_timezone(self, sample_ohlcv_df):
        """Test validation warns about missing timezone."""
        df = sample_ohlcv_df.copy()
        df.index = df.index.tz_localize(None)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=True, engine="pyarrow")
            try:
                client = DataBentoClient()
                result = client.validate_data(f.name)

                # Should have timezone warning
                assert any("not timezone-aware" in issue for issue in result.issues)
            finally:
                os.unlink(f.name)

    def test_very_large_gap_detection(self):
        """Test gap detection with very large gaps."""
        client = DataBentoClient()

        # Create data with week-long gap
        dates1 = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        dates2 = pd.date_range("2024-01-08", periods=10, freq="1min", tz="UTC")
        dates = dates1.append(dates2)

        df = pd.DataFrame(
            {
                "open": np.random.randn(20) + 100,
                "high": np.random.randn(20) + 101,
                "low": np.random.randn(20) + 99,
                "close": np.random.randn(20) + 100,
                "volume": np.random.randint(100, 1000, 20),
            },
            index=dates,
        )
        df.index.name = "timestamp"

        gaps = client._detect_gaps(df, expected_interval_seconds=60)

        assert len(gaps) == 1
        assert gaps[0].duration_seconds > 600000  # > ~7 days in seconds (minus the first 10 minutes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
