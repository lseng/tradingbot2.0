"""
DataBento Historical Data Client.

Provides a Python client for downloading historical futures data from DataBento
for ML model training and backtesting. The TopstepX API only retains ~7-14 days
of second-level data, so DataBento is used for historical data (3-5 years).

Key Features:
    - Download OHLCV data at multiple timeframes (1s, 1m, 1h, 1d)
    - Continuous contract handling (MES.FUT, ES.FUT, etc.)
    - Parquet storage with year/month partitioning
    - Data quality validation (OHLC relationships, gaps, timestamps)
    - Incremental updates and gap backfill
    - Rate limiting and retry logic

Usage:
    from src.data import DataBentoClient

    # Initialize client (reads DATABENTO_API_KEY from environment)
    client = DataBentoClient()

    # Download 3 years of 1-minute MES data
    result = client.download_ohlcv(
        symbol="MES.FUT",
        schema="ohlcv-1m",
        start="2022-01-01",
        end="2025-01-01",
        output_dir="data/historical/MES/"
    )

    # Validate data quality
    validation = client.validate_data("data/historical/MES/MES_1m.parquet")
    if validation.is_valid:
        print(f"Data valid: {validation.row_count} rows")
    else:
        print(f"Issues: {validation.issues}")

Spec Reference:
    specs/databento-historical-data.md
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class DataBentoError(Exception):
    """Base exception for DataBento client errors."""

    pass


class AuthenticationError(DataBentoError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(DataBentoError):
    """Raised when API rate limit is exceeded."""

    pass


class DataQualityError(DataBentoError):
    """Raised when data quality validation fails."""

    pass


# =============================================================================
# Enums and Data Classes
# =============================================================================


class OHLCVSchema(str, Enum):
    """Available OHLCV data schemas from DataBento."""

    OHLCV_1S = "ohlcv-1s"  # 1-second bars
    OHLCV_1M = "ohlcv-1m"  # 1-minute bars (primary training data)
    OHLCV_5M = "ohlcv-5m"  # 5-minute bars
    OHLCV_15M = "ohlcv-15m"  # 15-minute bars
    OHLCV_1H = "ohlcv-1h"  # 1-hour bars
    OHLCV_1D = "ohlcv-1d"  # Daily bars
    TRADES = "trades"  # Individual trades (tick data)


@dataclass
class DataBentoConfig:
    """
    Configuration for DataBento client.

    Attributes:
        api_key: DataBento API key (or from DATABENTO_API_KEY env var)
        dataset: DataBento dataset ID (default: CME Globex)
        max_retries: Maximum number of retry attempts on failure
        retry_delay: Initial delay between retries (seconds)
        request_timeout: Request timeout in seconds
        output_format: Output file format (parquet recommended)
    """

    api_key: Optional[str] = None
    dataset: str = "GLBX.MDP3"  # CME Globex
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 300  # 5 minutes
    output_format: str = "parquet"

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("DATABENTO_API_KEY")


@dataclass
class GapInfo:
    """Information about a detected data gap."""

    start: datetime
    end: datetime
    duration_seconds: int
    expected_bars: int
    session: str  # "RTH" or "ETH"


@dataclass
class DataValidationResult:
    """Result of data quality validation."""

    is_valid: bool
    row_count: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    issues: List[str] = field(default_factory=list)
    gaps: List[GapInfo] = field(default_factory=list)
    invalid_ohlc_rows: int = 0
    missing_volume_rows: int = 0
    duplicate_rows: int = 0


@dataclass
class DownloadResult:
    """Result of a data download operation."""

    success: bool
    output_path: str
    row_count: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    download_seconds: float
    validation: Optional[DataValidationResult] = None
    error_message: Optional[str] = None


# =============================================================================
# Contract Specifications
# =============================================================================

# DataBento continuous contract symbols
CONTINUOUS_CONTRACTS = {
    "MES": "MES.FUT",  # Micro E-mini S&P 500
    "ES": "ES.FUT",  # E-mini S&P 500
    "MNQ": "MNQ.FUT",  # Micro E-mini Nasdaq
    "NQ": "NQ.FUT",  # E-mini Nasdaq
}

# Trading hours for gap detection (US/Eastern)
# Futures trade nearly 24 hours: Sunday 6 PM - Friday 5 PM ET
# With daily break: 5 PM - 6 PM ET
RTH_START_HOUR = 9  # 9:30 AM ET - Regular Trading Hours start
RTH_START_MINUTE = 30
RTH_END_HOUR = 16  # 4:00 PM ET - Regular Trading Hours end
RTH_END_MINUTE = 0


# =============================================================================
# DataBento Client
# =============================================================================


class DataBentoClient:
    """
    Client for downloading historical futures data from DataBento.

    This client handles authentication, rate limiting, data download,
    and quality validation for OHLCV futures data.

    Example:
        client = DataBentoClient()
        result = client.download_ohlcv(
            symbol="MES.FUT",
            schema="ohlcv-1m",
            start="2022-01-01",
            end="2025-01-01",
        )
    """

    def __init__(self, config: Optional[DataBentoConfig] = None):
        """
        Initialize DataBento client.

        Args:
            config: Client configuration. If None, uses defaults.
        """
        self.config = config or DataBentoConfig()
        self._client = None
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # Minimum seconds between requests

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        return self.config.api_key

    def _get_client(self):
        """
        Get or create DataBento Historical client.

        Returns:
            DataBento Historical client instance

        Raises:
            AuthenticationError: If API key is not configured
        """
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise AuthenticationError(
                "DataBento API key not configured. "
                "Set DATABENTO_API_KEY environment variable or pass api_key to config."
            )

        try:
            import databento as db

            self._client = db.Historical(key=self.api_key)
            logger.info("DataBento client initialized successfully")
            return self._client
        except ImportError:
            raise DataBentoError(
                "databento package not installed. Run: pip install databento"
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to authenticate with DataBento: {e}")

    def _rate_limit(self):
        """Implement simple rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            max_retries: Override default max retries
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            DataBentoError: If all retries fail
        """
        max_attempts = max_retries or self.config.max_retries
        delay = self.config.retry_delay

        for attempt in range(max_attempts):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()

                # Check for rate limiting
                if "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2**attempt)
                        logger.warning(
                            f"Rate limited, waiting {wait_time:.1f}s before retry"
                        )
                        time.sleep(wait_time)
                        continue
                    raise RateLimitError(f"Rate limit exceeded after {max_attempts} attempts")

                # Check for auth errors
                if "auth" in error_msg or "401" in error_msg or "403" in error_msg:
                    raise AuthenticationError(f"Authentication failed: {e}")

                # Other errors - retry with backoff
                if attempt < max_attempts - 1:
                    wait_time = delay * (2**attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise DataBentoError(
                        f"Request failed after {max_attempts} attempts: {e}"
                    )

    def download_ohlcv(
        self,
        symbol: str,
        schema: str = "ohlcv-1m",
        start: str = None,
        end: str = None,
        output_dir: str = "data/historical/",
        validate: bool = True,
    ) -> DownloadResult:
        """
        Download OHLCV data for a symbol.

        Args:
            symbol: Contract symbol (e.g., "MES.FUT" or "MES")
            schema: OHLCV schema (e.g., "ohlcv-1m", "ohlcv-1s")
            start: Start date (YYYY-MM-DD). Default: 3 years ago
            end: End date (YYYY-MM-DD). Default: today
            output_dir: Output directory for parquet files
            validate: Whether to validate data after download

        Returns:
            DownloadResult with success status and metrics

        Example:
            result = client.download_ohlcv(
                symbol="MES.FUT",
                schema="ohlcv-1m",
                start="2022-01-01",
                end="2025-01-01",
            )
        """
        start_download = time.time()

        # Normalize symbol
        if symbol.upper() in CONTINUOUS_CONTRACTS:
            symbol = CONTINUOUS_CONTRACTS[symbol.upper()]

        # Set default date range
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start_dt = datetime.now() - timedelta(days=3 * 365)
            start = start_dt.strftime("%Y-%m-%d")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        symbol_clean = symbol.replace(".", "_")
        schema_clean = schema.replace("-", "_")
        output_file = output_path / f"{symbol_clean}_{schema_clean}.parquet"

        logger.info(
            f"Downloading {symbol} {schema} data from {start} to {end}"
        )

        try:
            # Get DataBento client
            client = self._get_client()

            # Download data with retry
            def fetch_data():
                return client.timeseries.get_range(
                    dataset=self.config.dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=start,
                    end=end,
                )

            data = self._retry_with_backoff(fetch_data)

            # Convert to DataFrame
            df = data.to_df()

            if df.empty:
                return DownloadResult(
                    success=False,
                    output_path=str(output_file),
                    row_count=0,
                    start_time=None,
                    end_time=None,
                    download_seconds=time.time() - start_download,
                    error_message="No data returned from DataBento",
                )

            # Process DataFrame
            df = self._process_ohlcv_dataframe(df)

            # Save to parquet
            df.to_parquet(output_file, index=True, engine="pyarrow")

            download_time = time.time() - start_download
            logger.info(
                f"Downloaded {len(df):,} rows in {download_time:.1f}s, "
                f"saved to {output_file}"
            )

            # Validate if requested
            validation = None
            if validate:
                validation = self.validate_data(str(output_file))
                if not validation.is_valid:
                    logger.warning(f"Data validation issues: {validation.issues}")

            return DownloadResult(
                success=True,
                output_path=str(output_file),
                row_count=len(df),
                start_time=df.index.min() if len(df) > 0 else None,
                end_time=df.index.max() if len(df) > 0 else None,
                download_seconds=download_time,
                validation=validation,
            )

        except (AuthenticationError, RateLimitError):
            raise
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return DownloadResult(
                success=False,
                output_path=str(output_file),
                row_count=0,
                start_time=None,
                end_time=None,
                download_seconds=time.time() - start_download,
                error_message=str(e),
            )

    def _process_ohlcv_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw DataBento DataFrame.

        Args:
            df: Raw DataFrame from DataBento

        Returns:
            Processed DataFrame with standardized columns
        """
        # DataBento uses 'ts_event' for timestamp
        if "ts_event" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True)
            df.set_index("timestamp", inplace=True)
        elif df.index.name != "timestamp":
            # If index is already datetime, ensure UTC
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index.name = "timestamp"

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("o", "open_"):
                rename_map[col] = "open"
            elif col_lower in ("h", "high_"):
                rename_map[col] = "high"
            elif col_lower in ("l", "low_"):
                rename_map[col] = "low"
            elif col_lower in ("c", "close_"):
                rename_map[col] = "close"
            elif col_lower in ("v", "volume_"):
                rename_map[col] = "volume"

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")

        # Sort by timestamp
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        return df

    def validate_data(self, file_path: str) -> DataValidationResult:
        """
        Validate data quality of a parquet file.

        Checks:
            - OHLC relationship: Low <= Open, Close <= High
            - No missing volume data
            - No duplicate timestamps
            - Timestamps in UTC
            - Detects gaps in trading hours

        Args:
            file_path: Path to parquet file

        Returns:
            DataValidationResult with validation status and issues
        """
        issues = []
        gaps = []

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            return DataValidationResult(
                is_valid=False,
                row_count=0,
                start_time=None,
                end_time=None,
                issues=[f"Failed to read file: {e}"],
            )

        row_count = len(df)
        if row_count == 0:
            return DataValidationResult(
                is_valid=False,
                row_count=0,
                start_time=None,
                end_time=None,
                issues=["File contains no data"],
            )

        # Ensure timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)
            else:
                issues.append("No timestamp index or column found")

        # Get time range
        start_time = df.index.min() if len(df) > 0 else None
        end_time = df.index.max() if len(df) > 0 else None

        # Check OHLC relationship: Low <= Open, Close <= High
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # Low should be <= Open and Close
            invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
            # High should be >= Open and Close
            invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
            invalid_ohlc = invalid_low | invalid_high
            invalid_count = invalid_ohlc.sum()

            if invalid_count > 0:
                issues.append(f"Invalid OHLC relationships: {invalid_count} rows")
        else:
            invalid_count = 0
            issues.append("Missing OHLC columns")

        # Check for missing volume
        missing_volume = 0
        if "volume" in df.columns:
            missing_volume = df["volume"].isna().sum()
            if missing_volume > 0:
                issues.append(f"Missing volume: {missing_volume} rows")
        else:
            issues.append("Missing volume column")

        # Check for duplicates
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Duplicate timestamps: {duplicate_count} rows")

        # Check timezone
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                issues.append("Timestamps not timezone-aware (should be UTC)")
            elif str(df.index.tz) != "UTC":
                issues.append(f"Timestamps in {df.index.tz}, should be UTC")

        # Detect gaps (simplified - for 1-minute data)
        gaps = self._detect_gaps(df)
        if gaps:
            issues.append(f"Detected {len(gaps)} data gaps")

        is_valid = len(issues) == 0

        return DataValidationResult(
            is_valid=is_valid,
            row_count=row_count,
            start_time=start_time,
            end_time=end_time,
            issues=issues,
            gaps=gaps,
            invalid_ohlc_rows=invalid_count if "invalid_count" in dir() else 0,
            missing_volume_rows=missing_volume,
            duplicate_rows=duplicate_count,
        )

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        expected_interval_seconds: int = 60,
        max_gap_multiplier: int = 5,
    ) -> List[GapInfo]:
        """
        Detect gaps in time series data.

        Args:
            df: DataFrame with DatetimeIndex
            expected_interval_seconds: Expected bar interval in seconds
            max_gap_multiplier: Gap threshold as multiple of expected interval

        Returns:
            List of detected gaps
        """
        if len(df) < 2:
            return []

        gaps = []
        max_gap = expected_interval_seconds * max_gap_multiplier

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Find gaps exceeding threshold
        gap_mask = time_diffs > pd.Timedelta(seconds=max_gap)
        gap_indices = df.index[gap_mask]

        for idx in gap_indices:
            # Get previous timestamp
            idx_pos = df.index.get_loc(idx)
            if idx_pos > 0:
                prev_idx = df.index[idx_pos - 1]
                gap_duration = (idx - prev_idx).total_seconds()

                # Determine session (simplified)
                hour = idx.hour if hasattr(idx, "hour") else 0
                session = "RTH" if RTH_START_HOUR <= hour < RTH_END_HOUR else "ETH"

                gaps.append(
                    GapInfo(
                        start=prev_idx.to_pydatetime(),
                        end=idx.to_pydatetime(),
                        duration_seconds=int(gap_duration),
                        expected_bars=int(gap_duration / expected_interval_seconds),
                        session=session,
                    )
                )

        return gaps

    def download_incremental(
        self,
        symbol: str,
        existing_file: str,
        output_dir: Optional[str] = None,
        validate: bool = True,
    ) -> DownloadResult:
        """
        Download incremental update to existing data.

        Args:
            symbol: Contract symbol
            existing_file: Path to existing parquet file
            output_dir: Output directory (default: same as existing)
            validate: Whether to validate after download

        Returns:
            DownloadResult with update status
        """
        # Load existing data to find end date
        try:
            existing_df = pd.read_parquet(existing_file)
            if len(existing_df) == 0:
                raise DataBentoError("Existing file is empty")

            last_timestamp = existing_df.index.max()
            start_date = (last_timestamp + timedelta(days=1)).strftime("%Y-%m-%d")

        except Exception as e:
            raise DataBentoError(f"Failed to read existing file: {e}")

        # Determine schema from filename
        file_name = Path(existing_file).stem
        if "_1s" in file_name or "1s" in file_name:
            schema = "ohlcv-1s"
        elif "_1m" in file_name or "1m" in file_name:
            schema = "ohlcv-1m"
        elif "_1h" in file_name or "1h" in file_name:
            schema = "ohlcv-1h"
        elif "_1d" in file_name or "1d" in file_name:
            schema = "ohlcv-1d"
        else:
            schema = "ohlcv-1m"  # Default

        # Download new data
        if output_dir is None:
            output_dir = str(Path(existing_file).parent)

        result = self.download_ohlcv(
            symbol=symbol,
            schema=schema,
            start=start_date,
            output_dir=output_dir,
            validate=False,  # We'll validate the merged file
        )

        if not result.success or result.row_count == 0:
            logger.info(f"No new data available since {start_date}")
            return result

        # Merge with existing data
        try:
            new_df = pd.read_parquet(result.output_path)
            merged_df = pd.concat([existing_df, new_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
            merged_df.sort_index(inplace=True)

            # Save merged data
            merged_df.to_parquet(existing_file, index=True, engine="pyarrow")

            # Clean up temp file if different
            if result.output_path != existing_file:
                os.remove(result.output_path)

            # Validate merged file
            validation = None
            if validate:
                validation = self.validate_data(existing_file)

            return DownloadResult(
                success=True,
                output_path=existing_file,
                row_count=len(merged_df),
                start_time=merged_df.index.min(),
                end_time=merged_df.index.max(),
                download_seconds=result.download_seconds,
                validation=validation,
            )

        except Exception as e:
            raise DataBentoError(f"Failed to merge data: {e}")

    def backfill_gaps(
        self,
        file_path: str,
        symbol: str,
        schema: str = "ohlcv-1m",
    ) -> Tuple[int, int]:
        """
        Detect and backfill gaps in existing data.

        Args:
            file_path: Path to parquet file
            symbol: Contract symbol
            schema: OHLCV schema

        Returns:
            Tuple of (gaps_found, gaps_filled)
        """
        # Validate to find gaps
        validation = self.validate_data(file_path)

        if not validation.gaps:
            logger.info("No gaps detected")
            return (0, 0)

        logger.info(f"Found {len(validation.gaps)} gaps, attempting backfill")

        # Load existing data
        df = pd.read_parquet(file_path)
        gaps_filled = 0

        for gap in validation.gaps:
            try:
                # Download gap data
                gap_start = gap.start.strftime("%Y-%m-%dT%H:%M:%S")
                gap_end = gap.end.strftime("%Y-%m-%dT%H:%M:%S")

                client = self._get_client()
                data = client.timeseries.get_range(
                    dataset=self.config.dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=gap_start,
                    end=gap_end,
                )

                gap_df = data.to_df()
                if len(gap_df) > 0:
                    gap_df = self._process_ohlcv_dataframe(gap_df)
                    df = pd.concat([df, gap_df])
                    gaps_filled += 1
                    logger.info(f"Filled gap: {gap.start} to {gap.end}")

            except Exception as e:
                logger.warning(f"Failed to fill gap {gap.start} to {gap.end}: {e}")

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # Save updated file
        df.to_parquet(file_path, index=True, engine="pyarrow")

        return (len(validation.gaps), gaps_filled)

    def list_available_symbols(self) -> List[str]:
        """
        List available continuous contract symbols.

        Returns:
            List of available symbols
        """
        return list(CONTINUOUS_CONTRACTS.values())

    def get_data_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get summary information about a data file.

        Args:
            file_path: Path to parquet file

        Returns:
            Dictionary with data summary
        """
        try:
            df = pd.read_parquet(file_path)

            return {
                "file_path": file_path,
                "row_count": len(df),
                "columns": list(df.columns),
                "start_time": df.index.min().isoformat() if len(df) > 0 else None,
                "end_time": df.index.max().isoformat() if len(df) > 0 else None,
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            }

        except Exception as e:
            return {"file_path": file_path, "error": str(e)}
