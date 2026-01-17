"""
Unit tests for memory estimation utilities.

These tests verify:
- Pre-load size estimation for parquet/CSV files
- System memory availability checking
- Warning/blocking thresholds
- Chunked/streaming loading for large datasets

Why These Tests Matter:
- Memory estimation prevents OOM crashes during data loading
- Incorrect estimates could either:
  - Block valid operations (too conservative)
  - Allow OOM crashes (too optimistic)
- Chunked loading enables processing datasets larger than RAM

Reference: IMPLEMENTATION_PLAN.md Task 10.10
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))

from data.memory_utils import (
    MemoryEstimator,
    MemoryEstimationError,
    InsufficientMemoryError,
    MemoryCheckResult,
    FileMemoryEstimate,
    ChunkedParquetLoader,
    estimate_parquet_memory,
    estimate_csv_memory,
    check_memory_available,
    get_system_memory,
    load_with_memory_check,
    PANDAS_MEMORY_MULTIPLIER,
    DEFAULT_WARNING_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
)


class TestGetSystemMemory:
    """Tests for system memory detection."""

    def test_returns_tuple(self):
        """Test that get_system_memory returns a tuple of two floats."""
        available, total = get_system_memory()
        assert isinstance(available, float)
        assert isinstance(total, float)

    def test_available_less_than_total(self):
        """Test that available memory is less than or equal to total."""
        available, total = get_system_memory()
        assert available <= total

    def test_positive_values(self):
        """Test that memory values are positive."""
        available, total = get_system_memory()
        assert available > 0
        assert total > 0

    def test_reasonable_values(self):
        """Test that memory values are reasonable (>100MB, <1TB)."""
        available, total = get_system_memory()
        # Minimum 100MB available
        assert available >= 100
        # Maximum 1TB total (reasonable for any system)
        assert total < 1_000_000  # 1TB in MB


class TestEstimateParquetMemory:
    """Tests for parquet memory estimation."""

    def test_estimate_parquet(self, sample_parquet_file):
        """Test estimating memory for a parquet file."""
        estimate = estimate_parquet_memory(sample_parquet_file)

        assert isinstance(estimate, FileMemoryEstimate)
        assert estimate.file_path == sample_parquet_file
        assert estimate.file_size_mb > 0
        assert estimate.num_rows == 1000  # From sample fixture
        assert estimate.num_columns > 0
        assert estimate.estimated_memory_mb > 0
        assert estimate.estimation_method == 'metadata'

    def test_estimate_larger_than_file_size(self, sample_parquet_file):
        """Test that estimated memory is larger than compressed file size."""
        estimate = estimate_parquet_memory(sample_parquet_file)

        # In-memory should be larger than compressed parquet
        assert estimate.estimated_memory_mb >= estimate.file_size_mb

    def test_file_not_found(self):
        """Test error when parquet file doesn't exist."""
        with pytest.raises(MemoryEstimationError, match="not found"):
            estimate_parquet_memory("/nonexistent/file.parquet")

    def test_not_parquet_file(self, tmp_path):
        """Test error when file is not a parquet file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")

        with pytest.raises(MemoryEstimationError, match="Not a parquet"):
            estimate_parquet_memory(csv_file)


class TestEstimateCsvMemory:
    """Tests for CSV memory estimation."""

    def test_estimate_csv(self, sample_csv_data):
        """Test estimating memory for a CSV file."""
        estimate = estimate_csv_memory(sample_csv_data)

        assert isinstance(estimate, FileMemoryEstimate)
        assert estimate.file_path == sample_csv_data
        assert estimate.file_size_mb > 0
        assert estimate.num_rows == 1000  # From sample fixture
        assert estimate.num_columns == 6  # datetime, OHLCV
        assert estimate.estimated_memory_mb > 0
        assert estimate.estimation_method == 'sample'

    def test_file_not_found(self):
        """Test error when CSV file doesn't exist."""
        with pytest.raises(MemoryEstimationError, match="not found"):
            estimate_csv_memory("/nonexistent/file.csv")


class TestCheckMemoryAvailable:
    """Tests for memory availability checking."""

    def test_small_requirement_can_load(self):
        """Test that small memory requirements can be loaded."""
        # 10MB should always be loadable
        result = check_memory_available(10.0)

        assert isinstance(result, MemoryCheckResult)
        assert result.can_load is True
        assert result.estimated_mb == 10.0
        assert result.available_mb > 0
        assert result.total_mb > 0

    def test_huge_requirement_cannot_load(self):
        """Test that huge memory requirements are blocked."""
        # 1TB should never be loadable
        result = check_memory_available(1_000_000)

        assert result.can_load is False
        assert len(result.reason) > 0

    def test_warning_threshold(self):
        """Test warning is generated when approaching threshold."""
        available, _ = get_system_memory()
        # Request 80% of available memory (above 70% warning threshold)
        required = available * 0.8

        result = check_memory_available(required)

        # Should still be loadable but with warning
        assert result.can_load is True
        assert len(result.warning) > 0

    def test_custom_thresholds(self):
        """Test custom warning and block thresholds."""
        available, _ = get_system_memory()
        # Request 60% of available
        required = available * 0.6

        # With default threshold (0.7 warning), no warning
        result_default = check_memory_available(required)
        assert result_default.can_load is True

        # With stricter threshold (0.5 warning), should warn
        result_strict = check_memory_available(
            required,
            warning_threshold=0.5,
        )
        assert result_strict.can_load is True
        assert len(result_strict.warning) > 0

    def test_result_to_dict(self):
        """Test MemoryCheckResult.to_dict() works."""
        result = check_memory_available(100.0)
        result_dict = result.to_dict()

        assert 'can_load' in result_dict
        assert 'estimated_mb' in result_dict
        assert 'available_mb' in result_dict
        assert 'usage_ratio' in result_dict


class TestMemoryEstimator:
    """Tests for the MemoryEstimator class."""

    def test_estimate_parquet_file(self, sample_parquet_file):
        """Test estimating a parquet file."""
        estimator = MemoryEstimator()
        estimate = estimator.estimate_file(sample_parquet_file)

        assert estimate.estimation_method == 'metadata'
        assert estimate.num_rows > 0

    def test_estimate_csv_file(self, sample_csv_data):
        """Test estimating a CSV file."""
        estimator = MemoryEstimator()
        estimate = estimator.estimate_file(sample_csv_data)

        assert estimate.estimation_method == 'sample'
        assert estimate.num_rows > 0

    def test_check_can_load_small_file(self, sample_parquet_file):
        """Test checking a small file that should be loadable."""
        estimator = MemoryEstimator()
        result = estimator.check_can_load(sample_parquet_file)

        assert result.can_load is True

    def test_raise_on_block(self, sample_parquet_file):
        """Test raise_on_block parameter."""
        # Create estimator with extremely strict threshold
        estimator = MemoryEstimator(
            block_threshold=0.00001,  # Block if using more than 0.001%
            raise_on_block=True,
        )

        with pytest.raises(InsufficientMemoryError):
            estimator.check_can_load(sample_parquet_file)

    def test_get_recommended_chunk_size(self, sample_parquet_file):
        """Test getting recommended chunk size."""
        estimator = MemoryEstimator()
        chunk_size = estimator.get_recommended_chunk_size(sample_parquet_file)

        assert chunk_size > 0
        assert chunk_size >= 1000  # Minimum reasonable chunk

    def test_custom_target_memory(self, sample_parquet_file):
        """Test custom target memory for chunk size calculation."""
        estimator = MemoryEstimator()

        # Very small target (10MB) should give smaller chunks
        small_chunk = estimator.get_recommended_chunk_size(
            sample_parquet_file,
            target_memory_mb=10,
        )

        # Larger target (1000MB) should give larger chunks
        large_chunk = estimator.get_recommended_chunk_size(
            sample_parquet_file,
            target_memory_mb=1000,
        )

        # Large target should give larger chunks
        assert large_chunk >= small_chunk


class TestChunkedParquetLoader:
    """Tests for chunked parquet loading."""

    def test_init_with_file(self, sample_parquet_file):
        """Test initializing with a valid file."""
        loader = ChunkedParquetLoader(sample_parquet_file)

        assert loader.total_rows == 1000
        assert loader.num_chunks >= 1
        assert loader.chunk_rows > 0

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ChunkedParquetLoader("/nonexistent/file.parquet")

    def test_iterate_chunks(self, sample_parquet_file):
        """Test iterating over chunks."""
        loader = ChunkedParquetLoader(sample_parquet_file, chunk_rows=500)

        chunks = list(loader)

        # Should have 2 chunks for 1000 rows at 500 per chunk
        assert len(chunks) == 2

        # Each chunk should be a DataFrame
        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) <= 500

        # Total rows should match
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000

    def test_context_manager(self, sample_parquet_file):
        """Test using loader as context manager."""
        with ChunkedParquetLoader(sample_parquet_file, chunk_rows=500) as loader:
            chunks = list(loader)
            assert len(chunks) == 2

    def test_single_chunk_for_small_file(self, sample_parquet_file):
        """Test that small files can be loaded in a single chunk."""
        # Request huge chunk size - should still work
        loader = ChunkedParquetLoader(sample_parquet_file, chunk_rows=1_000_000)

        chunks = list(loader)

        # Should be single chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 1000

    def test_progress_tracking(self, sample_parquet_file):
        """Test progress tracking."""
        loader = ChunkedParquetLoader(sample_parquet_file, chunk_rows=500)

        assert loader.get_progress(0) == 0.0
        assert loader.get_progress(500) == 0.5
        assert loader.get_progress(1000) == 1.0

    def test_columns_filtering(self, sample_parquet_file):
        """Test loading specific columns only."""
        loader = ChunkedParquetLoader(
            sample_parquet_file,
            chunk_rows=500,
            columns=['open', 'close', 'volume'],
        )

        chunks = list(loader)

        # Chunks should only have specified columns
        for chunk in chunks:
            assert 'open' in chunk.columns
            assert 'close' in chunk.columns
            assert 'volume' in chunk.columns
            # High and low should be excluded
            assert 'high' not in chunk.columns
            assert 'low' not in chunk.columns


class TestLoadWithMemoryCheck:
    """Tests for the load_with_memory_check convenience function."""

    def test_load_parquet(self, sample_parquet_file):
        """Test loading a parquet file with memory check."""
        df = load_with_memory_check(sample_parquet_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000

    def test_load_csv(self, sample_csv_data):
        """Test loading a CSV file with memory check."""
        df = load_with_memory_check(sample_csv_data, header=None)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000

    def test_raises_on_insufficient_memory(self, sample_parquet_file, monkeypatch):
        """Test that InsufficientMemoryError is raised when memory insufficient."""
        # Mock get_system_memory to return very little available memory
        def mock_get_memory():
            return 0.001, 1000.0  # Only 1KB available

        monkeypatch.setattr(
            'data.memory_utils.get_system_memory',
            mock_get_memory,
        )

        with pytest.raises(InsufficientMemoryError):
            load_with_memory_check(sample_parquet_file)


class TestInsufficientMemoryError:
    """Tests for the InsufficientMemoryError exception."""

    def test_exception_attributes(self):
        """Test exception has correct attributes."""
        error = InsufficientMemoryError(1000, 500)

        assert error.required_mb == 1000
        assert error.available_mb == 500
        assert "1000" in str(error) or "Insufficient" in str(error)

    def test_custom_message(self):
        """Test exception with custom message."""
        error = InsufficientMemoryError(1000, 500, "Custom error message")

        assert error.message == "Custom error message"


class TestFileMemoryEstimate:
    """Tests for FileMemoryEstimate dataclass."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        estimate = FileMemoryEstimate(
            file_path="/path/to/file.parquet",
            file_size_mb=100.5,
            num_rows=1000000,
            num_columns=6,
            estimated_memory_mb=450.25,
            estimation_method='metadata',
        )

        result = estimate.to_dict()

        assert result['file_path'] == "/path/to/file.parquet"
        assert result['file_size_mb'] == 100.5
        assert result['num_rows'] == 1000000
        assert result['num_columns'] == 6
        assert result['estimated_memory_mb'] == 450.25
        assert result['estimation_method'] == 'metadata'


class TestParquetLoaderMemoryIntegration:
    """Tests for memory check integration in ParquetDataLoader."""

    def test_memory_check_enabled_by_default(self, sample_parquet_file):
        """Test that memory checking is enabled by default."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file)

        # Memory check should be enabled
        assert loader._check_memory is True
        assert loader._memory_estimator is not None

    def test_memory_check_can_be_disabled(self, sample_parquet_file):
        """Test that memory checking can be disabled."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file, check_memory=False)

        assert loader._check_memory is False
        assert loader._memory_estimator is None

    def test_load_data_with_memory_check(self, sample_parquet_file):
        """Test loading data with memory check enabled."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file)
        df = loader.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000

    def test_load_data_skip_memory_check(self, sample_parquet_file):
        """Test skipping memory check on load."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file)
        df = loader.load_data(skip_memory_check=True)

        assert len(df) == 1000

    def test_load_chunked(self, sample_parquet_file):
        """Test chunked loading method."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file)
        chunks = list(loader.load_chunked(chunk_rows=500))

        assert len(chunks) == 2
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 1000

    def test_get_memory_estimate(self, sample_parquet_file):
        """Test getting memory estimate."""
        from data.parquet_loader import ParquetDataLoader

        loader = ParquetDataLoader(sample_parquet_file)
        estimate = loader.get_memory_estimate()

        assert 'file_path' in estimate
        assert 'estimated_memory_mb' in estimate
        assert estimate['num_rows'] == 1000


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_parquet_file(self, tmp_path):
        """Test handling of empty parquet file."""
        empty_df = pd.DataFrame({
            'timestamp': pd.DatetimeIndex([], dtype='datetime64[ns, UTC]'),
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
        })

        file_path = tmp_path / "empty.parquet"
        empty_df.to_parquet(file_path, engine='pyarrow')

        estimate = estimate_parquet_memory(file_path)

        assert estimate.num_rows == 0

    def test_very_small_available_memory(self):
        """Test behavior when available memory is very small."""
        # Request more than available
        result = check_memory_available(
            1000000,  # 1TB
            block_threshold=0.9,
        )

        assert result.can_load is False
        assert result.usage_ratio > 1.0

    def test_zero_required_memory(self):
        """Test checking zero required memory."""
        result = check_memory_available(0.0)

        assert result.can_load is True
        assert result.usage_ratio == 0.0
