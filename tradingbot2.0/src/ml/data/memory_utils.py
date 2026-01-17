"""
Memory Estimation Utilities for Large Dataset Loading.

This module prevents OOM (Out Of Memory) errors when loading large datasets
by estimating memory requirements before loading and providing warnings/blocking
when the estimated size exceeds available system memory.

Why This Matters:
- The MES_1s_2years.parquet dataset has 33M rows and can exceed 4GB in memory
- Loading without memory checks can crash the process or cause swapping
- Chunked loading enables processing datasets larger than available RAM

Features:
- Pre-load size estimation for parquet/CSV files using file metadata
- System memory availability check via psutil
- Configurable thresholds for warning vs blocking
- Chunked/streaming loading option for oversized datasets

Usage:
    from ml.data.memory_utils import (
        MemoryEstimator,
        estimate_parquet_memory,
        check_memory_available,
        ChunkedParquetLoader,
        MemoryError,
    )

    # Quick check before loading
    estimator = MemoryEstimator()
    result = estimator.check_can_load("data/MES_1s_2years.parquet")
    if not result.can_load:
        print(f"Cannot load: {result.reason}")

    # Or use chunked loading for large files
    loader = ChunkedParquetLoader("data/MES_1s_2years.parquet", chunk_rows=1_000_000)
    for chunk in loader:
        process(chunk)

Reference: IMPLEMENTATION_PLAN.md Task 10.10
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Memory multiplier to account for pandas overhead (DataFrames use ~2-3x raw data size)
PANDAS_MEMORY_MULTIPLIER = 2.5

# Safety margin - leave this much of available memory free
MEMORY_SAFETY_MARGIN = 0.2  # 20%

# Default thresholds for memory warnings
DEFAULT_WARNING_THRESHOLD = 0.7  # Warn if using >70% of available memory
DEFAULT_BLOCK_THRESHOLD = 0.9  # Block if using >90% of available memory


class MemoryEstimationError(Exception):
    """Raised when memory estimation fails."""
    pass


class InsufficientMemoryError(Exception):
    """Raised when there's not enough memory to load a dataset."""

    def __init__(self, required_mb: float, available_mb: float, message: str = ""):
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.message = message or (
            f"Insufficient memory: requires {required_mb:.1f}MB but only {available_mb:.1f}MB available"
        )
        super().__init__(self.message)


@dataclass
class MemoryCheckResult:
    """Result of a memory availability check."""

    can_load: bool
    estimated_mb: float
    available_mb: float
    total_mb: float
    usage_ratio: float
    reason: str = ""
    warning: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "can_load": self.can_load,
            "estimated_mb": round(self.estimated_mb, 2),
            "available_mb": round(self.available_mb, 2),
            "total_mb": round(self.total_mb, 2),
            "usage_ratio": round(self.usage_ratio, 4),
            "reason": self.reason,
            "warning": self.warning,
        }


@dataclass
class FileMemoryEstimate:
    """Memory estimate for a file."""

    file_path: str
    file_size_mb: float
    num_rows: int
    num_columns: int
    estimated_memory_mb: float
    estimation_method: str  # 'metadata', 'sample', or 'file_size'

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "file_path": self.file_path,
            "file_size_mb": round(self.file_size_mb, 2),
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "estimated_memory_mb": round(self.estimated_memory_mb, 2),
            "estimation_method": self.estimation_method,
        }


def get_system_memory() -> Tuple[float, float]:
    """
    Get system memory information.

    Returns:
        Tuple of (available_mb, total_mb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)
        return available_mb, total_mb
    except ImportError:
        # Fallback for systems without psutil
        logger.warning("psutil not available, using fallback memory estimation")
        return _get_memory_fallback()


def _get_memory_fallback() -> Tuple[float, float]:
    """
    Fallback memory estimation when psutil is not available.

    Uses /proc/meminfo on Linux, sysctl on macOS.
    """
    if sys.platform == 'linux':
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])

                total_kb = meminfo.get('MemTotal', 0)
                available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))

                return available_kb / 1024, total_kb / 1024
        except Exception as e:
            logger.warning(f"Failed to read /proc/meminfo: {e}")

    elif sys.platform == 'darwin':
        try:
            import subprocess
            # Get total memory
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, check=True
            )
            total_bytes = int(result.stdout.strip())
            total_mb = total_bytes / (1024 * 1024)

            # Estimate available (this is approximate on macOS)
            result = subprocess.run(
                ['vm_stat'],
                capture_output=True, text=True, check=True
            )
            # Parse vm_stat output (very rough approximation)
            lines = result.stdout.strip().split('\n')
            page_size = 4096  # Usually 4KB
            free_pages = 0
            for line in lines:
                if 'Pages free' in line:
                    free_pages = int(line.split(':')[1].strip().rstrip('.'))
                    break

            available_mb = (free_pages * page_size) / (1024 * 1024)
            # macOS heavily caches, so add some buffer
            available_mb = max(available_mb, total_mb * 0.3)

            return available_mb, total_mb
        except Exception as e:
            logger.warning(f"Failed to get macOS memory info: {e}")

    # Ultimate fallback: assume 8GB total, 4GB available
    logger.warning("Using default memory assumption: 4GB available of 8GB total")
    return 4096.0, 8192.0


def estimate_parquet_memory(file_path: Union[str, Path]) -> FileMemoryEstimate:
    """
    Estimate memory required to load a parquet file.

    Uses parquet file metadata to estimate size without loading the data.

    Args:
        file_path: Path to the parquet file

    Returns:
        FileMemoryEstimate with memory requirements

    Raises:
        MemoryEstimationError: If estimation fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise MemoryEstimationError(f"File not found: {file_path}")

    if not file_path.suffix == '.parquet':
        raise MemoryEstimationError(f"Not a parquet file: {file_path}")

    try:
        # Get file size on disk
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Read parquet metadata (very fast, doesn't load data)
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata

        num_rows = metadata.num_rows
        num_columns = metadata.num_columns

        # Estimate bytes per row from file size and compression ratio
        # Parquet is typically 4-10x compressed vs in-memory
        # For OHLCV data: 5 float64 (40 bytes) + timestamp (8 bytes) = 48 bytes per row
        # Plus pandas overhead (index, object overhead)
        bytes_per_row = 48  # Base OHLCV data
        bytes_per_row += 8  # Index
        bytes_per_row += 16 * num_columns  # Pandas series overhead per column

        estimated_bytes = num_rows * bytes_per_row * PANDAS_MEMORY_MULTIPLIER
        estimated_mb = estimated_bytes / (1024 * 1024)

        # Cross-check with file size (compressed size * typical ratio)
        # Parquet compression is usually 4-10x for numeric data
        estimated_from_file = file_size_mb * 6 * PANDAS_MEMORY_MULTIPLIER

        # Use the larger estimate for safety
        estimated_mb = max(estimated_mb, estimated_from_file)

        return FileMemoryEstimate(
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            num_rows=num_rows,
            num_columns=num_columns,
            estimated_memory_mb=estimated_mb,
            estimation_method='metadata',
        )

    except Exception as e:
        raise MemoryEstimationError(f"Failed to estimate parquet memory: {e}")


def estimate_csv_memory(
    file_path: Union[str, Path],
    sample_rows: int = 10000
) -> FileMemoryEstimate:
    """
    Estimate memory required to load a CSV file.

    Samples the file to estimate memory requirements.

    Args:
        file_path: Path to the CSV file
        sample_rows: Number of rows to sample for estimation

    Returns:
        FileMemoryEstimate with memory requirements

    Raises:
        MemoryEstimationError: If estimation fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise MemoryEstimationError(f"File not found: {file_path}")

    try:
        # Get file size
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Count total lines (fast)
        with open(file_path, 'r') as f:
            num_rows = sum(1 for _ in f)

        # Sample the file to estimate memory per row
        sample_df = pd.read_csv(
            file_path,
            header=None,
            nrows=min(sample_rows, num_rows)
        )

        num_columns = len(sample_df.columns)

        # Calculate memory usage of sample
        sample_memory = sample_df.memory_usage(deep=True).sum()
        bytes_per_row = sample_memory / len(sample_df)

        # Estimate total memory
        estimated_bytes = num_rows * bytes_per_row * PANDAS_MEMORY_MULTIPLIER
        estimated_mb = estimated_bytes / (1024 * 1024)

        return FileMemoryEstimate(
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            num_rows=num_rows,
            num_columns=num_columns,
            estimated_memory_mb=estimated_mb,
            estimation_method='sample',
        )

    except Exception as e:
        raise MemoryEstimationError(f"Failed to estimate CSV memory: {e}")


def check_memory_available(
    required_mb: float,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> MemoryCheckResult:
    """
    Check if enough memory is available for an operation.

    Args:
        required_mb: Estimated memory requirement in MB
        warning_threshold: Ratio of available memory to warn (default: 0.7)
        block_threshold: Ratio of available memory to block (default: 0.9)

    Returns:
        MemoryCheckResult with availability information
    """
    available_mb, total_mb = get_system_memory()

    # Calculate what ratio of available memory we'd use
    if available_mb > 0:
        usage_ratio = required_mb / available_mb
    else:
        usage_ratio = float('inf')

    # Determine if we can load
    can_load = usage_ratio <= block_threshold

    # Generate reason/warning messages
    reason = ""
    warning = ""

    if not can_load:
        reason = (
            f"Dataset requires {required_mb:.1f}MB but only {available_mb:.1f}MB available "
            f"({usage_ratio:.1%} of available memory exceeds {block_threshold:.0%} threshold)"
        )
    elif usage_ratio > warning_threshold:
        warning = (
            f"Loading will use {usage_ratio:.1%} of available memory "
            f"({required_mb:.1f}MB of {available_mb:.1f}MB). Consider chunked loading."
        )

    return MemoryCheckResult(
        can_load=can_load,
        estimated_mb=required_mb,
        available_mb=available_mb,
        total_mb=total_mb,
        usage_ratio=usage_ratio,
        reason=reason,
        warning=warning,
    )


class MemoryEstimator:
    """
    High-level interface for memory estimation and checking.

    Combines file size estimation with system memory checking to provide
    a simple API for determining if a dataset can be loaded safely.
    """

    def __init__(
        self,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        raise_on_block: bool = False,
    ):
        """
        Initialize the memory estimator.

        Args:
            warning_threshold: Memory usage ratio to emit warning
            block_threshold: Memory usage ratio to block loading
            raise_on_block: If True, raise InsufficientMemoryError when blocked
        """
        self.warning_threshold = warning_threshold
        self.block_threshold = block_threshold
        self.raise_on_block = raise_on_block

    def estimate_file(self, file_path: Union[str, Path]) -> FileMemoryEstimate:
        """
        Estimate memory requirement for a file.

        Automatically detects file type and uses appropriate estimator.

        Args:
            file_path: Path to the data file

        Returns:
            FileMemoryEstimate with memory requirements
        """
        file_path = Path(file_path)

        if file_path.suffix == '.parquet':
            return estimate_parquet_memory(file_path)
        elif file_path.suffix in ['.csv', '.txt']:
            return estimate_csv_memory(file_path)
        else:
            # Fallback: estimate from file size with large multiplier
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return FileMemoryEstimate(
                file_path=str(file_path),
                file_size_mb=file_size_mb,
                num_rows=0,
                num_columns=0,
                estimated_memory_mb=file_size_mb * 10,  # Conservative estimate
                estimation_method='file_size',
            )

    def check_can_load(self, file_path: Union[str, Path]) -> MemoryCheckResult:
        """
        Check if a file can be loaded into available memory.

        Args:
            file_path: Path to the data file

        Returns:
            MemoryCheckResult with availability information

        Raises:
            InsufficientMemoryError: If raise_on_block=True and not enough memory
        """
        estimate = self.estimate_file(file_path)
        result = check_memory_available(
            estimate.estimated_memory_mb,
            self.warning_threshold,
            self.block_threshold,
        )

        if result.warning:
            logger.warning(result.warning)

        if not result.can_load:
            logger.error(result.reason)
            if self.raise_on_block:
                raise InsufficientMemoryError(
                    result.estimated_mb,
                    result.available_mb,
                    result.reason,
                )

        return result

    def get_recommended_chunk_size(
        self,
        file_path: Union[str, Path],
        target_memory_mb: Optional[float] = None,
    ) -> int:
        """
        Get recommended chunk size for loading a file in chunks.

        Args:
            file_path: Path to the data file
            target_memory_mb: Target memory per chunk (default: 25% of available)

        Returns:
            Recommended number of rows per chunk
        """
        estimate = self.estimate_file(file_path)
        available_mb, _ = get_system_memory()

        if target_memory_mb is None:
            # Use 25% of available memory per chunk
            target_memory_mb = available_mb * 0.25

        if estimate.num_rows <= 0:
            # Can't estimate chunk size without row count
            return 100_000  # Default fallback

        # Calculate rows per MB
        mb_per_row = estimate.estimated_memory_mb / estimate.num_rows

        if mb_per_row > 0:
            chunk_rows = int(target_memory_mb / mb_per_row)
        else:
            chunk_rows = 100_000

        # Ensure reasonable bounds
        chunk_rows = max(1000, min(chunk_rows, 10_000_000))

        return chunk_rows


class ChunkedParquetLoader:
    """
    Load parquet files in memory-efficient chunks.

    This class enables processing datasets larger than available RAM
    by loading them in configurable chunks.

    Usage:
        loader = ChunkedParquetLoader("large_file.parquet", chunk_rows=1_000_000)
        for chunk_df in loader:
            process(chunk_df)

        # Or with context manager
        with ChunkedParquetLoader("large_file.parquet") as loader:
            for chunk in loader:
                process(chunk)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        chunk_rows: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize the chunked loader.

        Args:
            file_path: Path to the parquet file
            chunk_rows: Rows per chunk (auto-calculated if None)
            columns: Specific columns to load (None for all)
        """
        self.file_path = Path(file_path)
        self.columns = columns
        self._parquet_file: Optional[pq.ParquetFile] = None
        self._current_row_group = 0

        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        # Get metadata to calculate chunk size
        self._parquet_file = pq.ParquetFile(self.file_path)
        self._metadata = self._parquet_file.metadata
        self._num_row_groups = self._metadata.num_row_groups
        self._total_rows = self._metadata.num_rows

        # Calculate chunk size
        if chunk_rows is None:
            estimator = MemoryEstimator()
            self.chunk_rows = estimator.get_recommended_chunk_size(self.file_path)
        else:
            self.chunk_rows = chunk_rows

        logger.info(
            f"ChunkedParquetLoader initialized: {self._total_rows:,} rows, "
            f"{self._num_row_groups} row groups, {self.chunk_rows:,} rows per chunk"
        )

    def __enter__(self) -> 'ChunkedParquetLoader':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the parquet file."""
        self._parquet_file = None

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        """Iterate over chunks of the parquet file."""
        return self._generate_chunks()

    def _generate_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Generate DataFrame chunks from the parquet file."""
        if self._parquet_file is None:
            self._parquet_file = pq.ParquetFile(self.file_path)

        # Read row groups and yield chunks
        row_offset = 0
        accumulated_df: Optional[pd.DataFrame] = None

        for rg_idx in range(self._num_row_groups):
            # Read row group
            table = self._parquet_file.read_row_group(
                rg_idx,
                columns=self.columns,
            )
            df = table.to_pandas()

            # Accumulate or yield
            if accumulated_df is not None:
                accumulated_df = pd.concat([accumulated_df, df], ignore_index=True)
            else:
                accumulated_df = df

            # Yield chunks when we have enough rows
            while len(accumulated_df) >= self.chunk_rows:
                chunk = accumulated_df.iloc[:self.chunk_rows]
                accumulated_df = accumulated_df.iloc[self.chunk_rows:].reset_index(drop=True)
                yield chunk

        # Yield remaining data
        if accumulated_df is not None and len(accumulated_df) > 0:
            yield accumulated_df

    @property
    def total_rows(self) -> int:
        """Get total number of rows in the file."""
        return self._total_rows

    @property
    def num_chunks(self) -> int:
        """Estimate number of chunks that will be yielded."""
        return max(1, (self._total_rows + self.chunk_rows - 1) // self.chunk_rows)

    def get_progress(self, rows_processed: int) -> float:
        """Get processing progress as a ratio 0-1."""
        if self._total_rows > 0:
            return min(1.0, rows_processed / self._total_rows)
        return 0.0


def load_with_memory_check(
    file_path: Union[str, Path],
    raise_on_warning: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Load a data file with automatic memory checking.

    This is a drop-in replacement for pd.read_parquet/pd.read_csv that
    adds memory safety checks before loading.

    Args:
        file_path: Path to the data file
        raise_on_warning: If True, raise on warning (not just block)
        **read_kwargs: Additional arguments passed to pandas read function

    Returns:
        Loaded DataFrame

    Raises:
        InsufficientMemoryError: If not enough memory available
    """
    file_path = Path(file_path)

    # Check memory
    estimator = MemoryEstimator(
        raise_on_block=True,
        block_threshold=DEFAULT_WARNING_THRESHOLD if raise_on_warning else DEFAULT_BLOCK_THRESHOLD,
    )
    result = estimator.check_can_load(file_path)

    logger.info(
        f"Loading {file_path.name}: estimated {result.estimated_mb:.1f}MB, "
        f"{result.available_mb:.1f}MB available ({result.usage_ratio:.1%})"
    )

    # Load based on file type
    if file_path.suffix == '.parquet':
        return pd.read_parquet(file_path, **read_kwargs)
    elif file_path.suffix in ['.csv', '.txt']:
        return pd.read_csv(file_path, **read_kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


if __name__ == "__main__":
    import sys

    # Demo usage
    if len(sys.argv) < 2:
        print("Usage: python memory_utils.py <parquet_or_csv_file>")
        print("\nDemo with synthetic estimate:")

        # Show system memory
        available, total = get_system_memory()
        print(f"\nSystem Memory:")
        print(f"  Total: {total:.1f} MB ({total/1024:.1f} GB)")
        print(f"  Available: {available:.1f} MB ({available/1024:.1f} GB)")

        # Simulate a large file check
        print(f"\nSimulated check for 4GB dataset:")
        result = check_memory_available(4000)
        print(f"  Can load: {result.can_load}")
        print(f"  Usage ratio: {result.usage_ratio:.1%}")
        if result.warning:
            print(f"  Warning: {result.warning}")
        if result.reason:
            print(f"  Reason: {result.reason}")

        sys.exit(0)

    file_path = sys.argv[1]

    print(f"Analyzing: {file_path}")
    print("=" * 60)

    estimator = MemoryEstimator()

    # Estimate memory
    estimate = estimator.estimate_file(file_path)
    print(f"\nFile Analysis:")
    print(f"  File size: {estimate.file_size_mb:.2f} MB")
    print(f"  Rows: {estimate.num_rows:,}")
    print(f"  Columns: {estimate.num_columns}")
    print(f"  Estimated memory: {estimate.estimated_memory_mb:.2f} MB")
    print(f"  Estimation method: {estimate.estimation_method}")

    # Check if loadable
    result = estimator.check_can_load(file_path)
    print(f"\nMemory Check:")
    print(f"  Available: {result.available_mb:.1f} MB")
    print(f"  Required: {result.estimated_mb:.1f} MB")
    print(f"  Usage ratio: {result.usage_ratio:.1%}")
    print(f"  Can load: {result.can_load}")

    if result.warning:
        print(f"  Warning: {result.warning}")
    if result.reason:
        print(f"  Reason: {result.reason}")

    # Recommend chunk size
    chunk_size = estimator.get_recommended_chunk_size(file_path)
    print(f"\nRecommended chunk size: {chunk_size:,} rows")
    print(f"Estimated chunks: {max(1, estimate.num_rows // chunk_size)}")
