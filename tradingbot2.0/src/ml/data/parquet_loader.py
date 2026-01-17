"""
Parquet Data Loader for 1-Second Futures Data.

This module handles:
- Loading 1-second OHLCV data from parquet files (DataBento format)
- UTC to NY timezone conversion with DST handling
- RTH (Regular Trading Hours) vs ETH (Extended Trading Hours) filtering
- Multi-timeframe aggregation (1s -> 5s, 15s, 1m, 5m, 15m)
- Session boundary detection
- 3-class target variable creation (UP/FLAT/DOWN)
- Memory estimation and checking before loading (Task 10.10)

The primary data asset is MES_1s_2years.parquet (~33M rows, 227MB).

Performance Requirements (from spec):
- Load 33M rows in < 30 seconds
- Filter to RTH only, return ~6-8M rows
- Memory usage < 4GB during loading
- No timezone errors across DST boundaries
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Generator
from datetime import datetime, time
from zoneinfo import ZoneInfo
import logging
from dataclasses import dataclass

from .memory_utils import (
    MemoryEstimator,
    InsufficientMemoryError,
    ChunkedParquetLoader,
    estimate_parquet_memory,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MES contract constants (from spec)
MES_TICK_SIZE = 0.25  # Minimum price movement
MES_TICK_VALUE = 1.25  # Dollar value per tick ($5 per point / 4 ticks)
MES_POINT_VALUE = 5.00  # Dollar value per point

# Trading session times (New York timezone)
RTH_START = time(9, 30)   # Regular Trading Hours start
RTH_END = time(16, 0)     # Regular Trading Hours end
ETH_START = time(18, 0)   # Extended Trading Hours start (previous day)
ETH_END = time(9, 30)     # Extended Trading Hours end (current day)
FLATTEN_TIME = time(16, 30)  # EOD flatten deadline


@dataclass
class SessionInfo:
    """Information about a trading session."""
    date: pd.Timestamp
    rth_start: pd.Timestamp
    rth_end: pd.Timestamp
    bar_count: int
    volume: int
    is_partial: bool = False


class ParquetDataLoader:
    """
    Load and preprocess 1-second futures data from parquet files.

    This loader is optimized for DataBento parquet format with columns:
    - timestamp: datetime64[ns, UTC]
    - open, high, low, close: float64
    - volume: uint64
    - symbol: string (e.g., "MES.c.0")

    Additional DataBento metadata columns (rtype, publisher_id, instrument_id)
    are dropped during loading for memory efficiency.
    """

    # Columns to keep from parquet (drop metadata columns)
    OHLCV_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        data_path: str,
        timezone: str = 'America/New_York',
        check_memory: bool = True,
        memory_block_threshold: float = 0.9,
    ):
        """
        Initialize the parquet loader.

        Args:
            data_path: Path to the parquet file
            timezone: Target timezone for session filtering (default: NY)
            check_memory: If True, check memory before loading (default: True)
            memory_block_threshold: Block loading if estimated usage exceeds this
                fraction of available memory (default: 0.9)
        """
        self.data_path = Path(data_path)
        self.timezone = ZoneInfo(timezone)
        self.raw_data: Optional[pd.DataFrame] = None
        self.session_data: Optional[pd.DataFrame] = None
        self._check_memory = check_memory
        self._memory_block_threshold = memory_block_threshold
        self._memory_estimator = MemoryEstimator(
            block_threshold=memory_block_threshold,
            raise_on_block=True,
        ) if check_memory else None

        if not self.data_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.data_path}")

    def load_data(
        self,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        skip_memory_check: bool = False,
    ) -> pd.DataFrame:
        """
        Load parquet data into memory.

        Args:
            columns: Specific columns to load (default: OHLCV only)
            filters: PyArrow predicate pushdown filters for row filtering
            skip_memory_check: If True, skip memory check even if enabled in __init__

        Returns:
            DataFrame with OHLCV data, timestamp as index

        Raises:
            InsufficientMemoryError: If not enough memory available (when check_memory=True)
        """
        import time as time_module
        start_time = time_module.perf_counter()

        if columns is None:
            # Only load OHLCV columns, skip metadata
            columns = self.OHLCV_COLUMNS

        # Memory check (Task 10.10)
        if self._check_memory and not skip_memory_check and self._memory_estimator:
            result = self._memory_estimator.check_can_load(self.data_path)
            logger.info(
                f"Memory check: estimated {result.estimated_mb:.1f}MB, "
                f"available {result.available_mb:.1f}MB ({result.usage_ratio:.1%} usage)"
            )
            # Warning is logged by the estimator, InsufficientMemoryError raised if blocked

        logger.info(f"Loading parquet from {self.data_path}")

        # Load with pyarrow for best performance
        self.raw_data = pd.read_parquet(
            self.data_path,
            engine='pyarrow',
            columns=columns,
            filters=filters
        )

        load_time = time_module.perf_counter() - start_time
        logger.info(f"Loaded {len(self.raw_data):,} rows in {load_time:.2f}s")

        # Validate and clean data
        self._validate_data()

        # Set timestamp as index and sort
        if 'timestamp' in self.raw_data.columns:
            self.raw_data.set_index('timestamp', inplace=True)
        self.raw_data.sort_index(inplace=True)

        # Log date range
        logger.info(f"Date range: {self.raw_data.index.min()} to {self.raw_data.index.max()}")

        return self.raw_data

    def load_chunked(
        self,
        chunk_rows: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load parquet data in memory-efficient chunks.

        Use this method for datasets that may not fit in memory.
        Each chunk is validated and has timestamp set as index.

        Args:
            chunk_rows: Rows per chunk (auto-calculated if None based on available memory)
            columns: Specific columns to load (default: OHLCV only)

        Yields:
            DataFrame chunks with validated OHLCV data

        Example:
            loader = ParquetDataLoader("large_file.parquet")
            for chunk in loader.load_chunked(chunk_rows=1_000_000):
                process(chunk)
        """
        if columns is None:
            columns = self.OHLCV_COLUMNS

        chunked_loader = ChunkedParquetLoader(
            self.data_path,
            chunk_rows=chunk_rows,
            columns=columns,
        )

        logger.info(
            f"Loading in chunks: {chunked_loader.total_rows:,} total rows, "
            f"~{chunked_loader.num_chunks} chunks of {chunked_loader.chunk_rows:,} rows"
        )

        for i, chunk in enumerate(chunked_loader):
            # Set timestamp as index if present
            if 'timestamp' in chunk.columns:
                chunk = chunk.set_index('timestamp')

            # Sort by index
            chunk = chunk.sort_index()

            logger.debug(f"Yielding chunk {i+1}: {len(chunk):,} rows")
            yield chunk

    def get_memory_estimate(self) -> dict:
        """
        Get memory estimate for loading this parquet file.

        Returns:
            Dictionary with memory estimation details
        """
        estimate = estimate_parquet_memory(self.data_path)
        return estimate.to_dict()

    def _validate_data(self) -> None:
        """Validate data integrity."""
        df = self.raw_data

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Forward fill missing prices, set missing volume to 0
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)

        # Check for negative prices
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        if (df[price_cols] <= 0).any().any():
            logger.warning("Found non-positive prices, filtering...")
            mask = (df[price_cols] > 0).all(axis=1)
            self.raw_data = df[mask]

        # Check for negative volumes
        if 'volume' in df.columns and (df['volume'] < 0).any():
            logger.warning("Found negative volumes, setting to 0...")
            df.loc[df['volume'] < 0, 'volume'] = 0

        # Check OHLC validity
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            if invalid_ohlc.any():
                count = invalid_ohlc.sum()
                logger.warning(f"Found {count} rows with invalid OHLC, filtering...")
                self.raw_data = df[~invalid_ohlc]

    def convert_to_ny_timezone(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Convert UTC timestamps to New York timezone.

        Handles DST transitions correctly using zoneinfo.

        Args:
            df: DataFrame to convert (default: self.raw_data)

        Returns:
            DataFrame with NY timezone index
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = df.copy()

        # Convert index to NY timezone
        if df.index.tz is not None:
            # Already has timezone info, convert to NY
            df.index = df.index.tz_convert(self.timezone)
        else:
            # No timezone, assume UTC and localize first
            df.index = df.index.tz_localize('UTC').tz_convert(self.timezone)

        logger.info(f"Converted to {self.timezone}")

        return df

    def filter_rth(
        self,
        df: Optional[pd.DataFrame] = None,
        start_time: time = RTH_START,
        end_time: time = RTH_END
    ) -> pd.DataFrame:
        """
        Filter to Regular Trading Hours only.

        RTH for CME is 9:30 AM - 4:00 PM New York time.

        Args:
            df: DataFrame to filter (default: self.raw_data)
            start_time: RTH start time (default: 9:30 AM)
            end_time: RTH end time (default: 4:00 PM)

        Returns:
            DataFrame filtered to RTH only
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Ensure data is in NY timezone
        if df.index.tz is None or str(df.index.tz) != str(self.timezone):
            df = self.convert_to_ny_timezone(df)

        # Filter by time of day
        times = df.index.time
        mask = (times >= start_time) & (times < end_time)

        # Also filter out weekends (Saturday=5, Sunday=6)
        weekday_mask = df.index.dayofweek < 5

        filtered = df[mask & weekday_mask]

        logger.info(f"RTH filter: {len(df):,} -> {len(filtered):,} rows "
                   f"({len(filtered)/len(df)*100:.1f}% retained)")

        return filtered

    def filter_eth(
        self,
        df: Optional[pd.DataFrame] = None,
        eth_start: time = ETH_START,
        eth_end: time = ETH_END
    ) -> pd.DataFrame:
        """
        Filter to Extended Trading Hours only.

        ETH for CME is 6:00 PM - 9:30 AM New York time.
        This spans two calendar days (evening to morning).

        Args:
            df: DataFrame to filter (default: self.raw_data)
            eth_start: ETH start time (default: 6:00 PM)
            eth_end: ETH end time (default: 9:30 AM next day)

        Returns:
            DataFrame filtered to ETH only
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Ensure data is in NY timezone
        if df.index.tz is None or str(df.index.tz) != str(self.timezone):
            df = self.convert_to_ny_timezone(df)

        # ETH spans overnight: (6 PM to midnight) OR (midnight to 9:30 AM)
        times = df.index.time
        evening_mask = times >= eth_start  # 6:00 PM to midnight
        morning_mask = times < eth_end     # midnight to 9:30 AM

        # Exclude RTH entirely
        rth_mask = (times >= RTH_START) & (times < RTH_END)

        # Also filter out weekends
        weekday_mask = df.index.dayofweek < 5

        # For Friday evening ETH and Sunday evening start, need special handling
        # Sunday evening (dayofweek=6, time >= 18:00) counts as ETH for Monday
        sunday_evening = (df.index.dayofweek == 6) & evening_mask

        filtered = df[(evening_mask | morning_mask) & ~rth_mask & (weekday_mask | sunday_evening)]

        logger.info(f"ETH filter: {len(df):,} -> {len(filtered):,} rows "
                   f"({len(filtered)/len(df)*100:.1f}% retained)")

        return filtered

    def aggregate_timeframe(
        self,
        df: Optional[pd.DataFrame] = None,
        timeframe: str = '1min'
    ) -> pd.DataFrame:
        """
        Aggregate 1-second data to higher timeframes.

        Args:
            df: DataFrame to aggregate (default: self.raw_data)
            timeframe: Target timeframe ('5s', '15s', '30s', '1min', '5min', '15min')

        Returns:
            Aggregated OHLCV DataFrame
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Validate timeframe
        valid_timeframes = ['1s', '5s', '15s', '30s', '1min', '5min', '15min', '1h', '1d']
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")

        # Map to pandas offset
        offset_map = {
            '1s': '1s', '5s': '5s', '15s': '15s', '30s': '30s',
            '1min': '1min', '5min': '5min', '15min': '15min',
            '1h': '1h', '1d': '1D'
        }
        offset = offset_map[timeframe]

        # Aggregate OHLCV
        aggregated = df.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"Aggregated to {timeframe}: {len(df):,} -> {len(aggregated):,} bars")

        return aggregated

    def get_session_boundaries(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> List[SessionInfo]:
        """
        Detect trading session boundaries.

        A session is defined as continuous RTH period for a single trading day.

        Args:
            df: DataFrame to analyze (default: self.raw_data)

        Returns:
            List of SessionInfo objects
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Ensure data is in NY timezone
        if df.index.tz is None or str(df.index.tz) != str(self.timezone):
            df = self.convert_to_ny_timezone(df)

        # Filter to RTH first
        rth_df = self.filter_rth(df)

        # Group by date
        sessions = []
        for date, group in rth_df.groupby(rth_df.index.date):
            session = SessionInfo(
                date=pd.Timestamp(date),
                rth_start=group.index.min(),
                rth_end=group.index.max(),
                bar_count=len(group),
                volume=int(group['volume'].sum()),
                # Full RTH session is 6.5 hours = 23400 seconds
                is_partial=len(group) < 20000  # Less than ~85% of full session
            )
            sessions.append(session)

        logger.info(f"Found {len(sessions)} trading sessions")
        partial_count = sum(1 for s in sessions if s.is_partial)
        if partial_count > 0:
            logger.warning(f"{partial_count} partial sessions detected (holidays, data gaps)")

        return sessions

    def create_target_variable(
        self,
        df: pd.DataFrame,
        lookahead_seconds: int = 30,
        threshold_ticks: float = 3.0,
        tick_size: float = MES_TICK_SIZE
    ) -> pd.DataFrame:
        """
        Create 3-class target variable for scalping.

        Classes:
        - 0 (DOWN): Price drops more than threshold_ticks
        - 1 (FLAT): Price stays within threshold_ticks
        - 2 (UP): Price rises more than threshold_ticks

        Args:
            df: DataFrame with OHLCV data
            lookahead_seconds: How many seconds ahead to predict (default: 30)
            threshold_ticks: Number of ticks for UP/DOWN classification (default: 3)
            tick_size: Tick size in points (default: 0.25 for MES)

        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()

        # Calculate future price (lookahead seconds ahead)
        # Note: shift(-N) looks N periods into the future
        future_close = df['close'].shift(-lookahead_seconds)

        # Calculate move in ticks
        tick_move = (future_close - df['close']) / tick_size

        # Create 3-class target
        # DOWN = 0, FLAT = 1, UP = 2
        df['target'] = np.where(
            tick_move > threshold_ticks, 2,  # UP
            np.where(tick_move < -threshold_ticks, 0, 1)  # DOWN or FLAT
        )

        # Store the actual tick move for analysis
        df['future_tick_move'] = tick_move

        # Store the future price for debugging
        df['future_close'] = future_close

        # Remove last rows where we don't have future data
        df = df.iloc[:-lookahead_seconds]

        # CRITICAL: Drop future columns to prevent data leakage
        # These columns contain future information that would give the model perfect foresight
        # if accidentally used in feature engineering
        df = df.drop(columns=['future_close', 'future_tick_move'])

        # Calculate class distribution
        class_counts = df['target'].value_counts().sort_index()
        total = len(df)

        logger.info(f"Target distribution (lookahead={lookahead_seconds}s, threshold={threshold_ticks} ticks):")
        logger.info(f"  DOWN (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.1f}%)")
        logger.info(f"  FLAT (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.1f}%)")
        logger.info(f"  UP   (2): {class_counts.get(2, 0):,} ({class_counts.get(2, 0)/total*100:.1f}%)")

        return df

    def get_class_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced classification.

        Uses inverse frequency weighting to handle class imbalance
        (FLAT class is typically dominant at ~60-70%).

        Args:
            df: DataFrame with 'target' column

        Returns:
            Dict mapping class index to weight
        """
        if 'target' not in df.columns:
            raise ValueError("DataFrame must have 'target' column. Call create_target_variable() first.")

        class_counts = df['target'].value_counts()
        total = len(df)
        n_classes = len(class_counts)

        # Inverse frequency weighting: weight = total / (n_classes * count)
        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = total / (n_classes * count)

        logger.info(f"Class weights: {weights}")

        return weights

    def train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally for walk-forward validation.

        No shuffling - maintains time order for proper backtesting.

        Args:
            df: DataFrame to split
            train_ratio: Fraction for training (default: 0.6)
            val_ratio: Fraction for validation (default: 0.2)
            test_ratio: Fraction for testing (default: 0.2)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(f"Split: Train={len(train_df):,} ({train_ratio:.0%}), "
                   f"Val={len(val_df):,} ({val_ratio:.0%}), "
                   f"Test={len(test_df):,} ({test_ratio:.0%})")

        return train_df, val_df, test_df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for scalping.

        Features added:
        - minutes_to_close: Minutes remaining until RTH close (0-390)
        - time_of_day: Cyclical encoding of time (sin/cos)
        - day_of_week: Day of week (0-4 for Mon-Fri)
        - is_first_hour: Boolean for first hour of RTH (9:30-10:30)
        - is_last_hour: Boolean for last hour of RTH (15:00-16:00)
        - is_lunch: Boolean for lunch period (11:30-13:00)

        Args:
            df: DataFrame with datetime index in NY timezone

        Returns:
            DataFrame with time features added
        """
        df = df.copy()

        # Minutes since market open (9:30 AM)
        market_open_minutes = RTH_START.hour * 60 + RTH_START.minute  # 570
        current_minutes = df.index.hour * 60 + df.index.minute
        minutes_since_open = current_minutes - market_open_minutes

        # Minutes to close (RTH ends at 4:00 PM = 960 minutes from midnight)
        market_close_minutes = RTH_END.hour * 60 + RTH_END.minute  # 960
        df['minutes_to_close'] = market_close_minutes - current_minutes
        df['minutes_to_close'] = df['minutes_to_close'].clip(lower=0)  # Clip negative (after close)

        # Normalize to 0-1 range
        df['minutes_to_close_norm'] = df['minutes_to_close'] / 390  # 390 minutes in RTH

        # Cyclical time encoding (captures periodicity)
        seconds_in_day = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        df['time_sin'] = np.sin(2 * np.pi * seconds_in_day / 86400)
        df['time_cos'] = np.cos(2 * np.pi * seconds_in_day / 86400)

        # Day of week
        df['day_of_week'] = df.index.dayofweek

        # Session periods (boolean flags)
        hour = df.index.hour
        minute = df.index.minute

        # First hour: 9:30-10:30
        df['is_first_hour'] = ((hour == 9) & (minute >= 30)) | (hour == 10)

        # Last hour: 15:00-16:00
        df['is_last_hour'] = hour == 15

        # Lunch period: 11:30-13:00 (typically lower volume)
        df['is_lunch'] = ((hour == 11) & (minute >= 30)) | (hour == 12)

        return df


def load_and_prepare_scalping_data(
    data_path: str,
    filter_rth: bool = True,
    lookahead_seconds: int = 30,
    threshold_ticks: float = 3.0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    check_memory: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and prepare data for scalping model.

    This function:
    1. Loads parquet data (with optional memory check)
    2. Converts to NY timezone
    3. Filters to RTH (optional)
    4. Creates 3-class target variable
    5. Adds time features
    6. Splits into train/val/test

    Args:
        data_path: Path to parquet file
        filter_rth: Whether to filter to RTH only (default: True)
        lookahead_seconds: Prediction horizon in seconds (default: 30)
        threshold_ticks: Ticks for UP/DOWN classification (default: 3.0)
        train_ratio: Training set ratio (default: 0.6)
        val_ratio: Validation set ratio (default: 0.2)
        check_memory: If True, check memory before loading (default: True)

    Returns:
        Tuple of (full_df, train_df, val_df, test_df)

    Raises:
        InsufficientMemoryError: If check_memory=True and not enough memory
    """
    loader = ParquetDataLoader(data_path, check_memory=check_memory)

    # Load and convert timezone
    df = loader.load_data()
    df = loader.convert_to_ny_timezone(df)

    # Filter to RTH if requested
    if filter_rth:
        df = loader.filter_rth(df)

    # Create target
    df = loader.create_target_variable(
        df,
        lookahead_seconds=lookahead_seconds,
        threshold_ticks=threshold_ticks
    )

    # Add time features
    df = loader.add_time_features(df)

    # Split data
    train_df, val_df, test_df = loader.train_test_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1 - train_ratio - val_ratio
    )

    return df, train_df, val_df, test_df


if __name__ == "__main__":
    import sys
    import time as time_module

    # Default path to MES 1-second data
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/historical/MES/MES_1s_2years.parquet"

    print("=" * 60)
    print("PARQUET LOADER TEST")
    print("=" * 60)

    start = time_module.perf_counter()

    # Load and prepare data
    full_df, train_df, val_df, test_df = load_and_prepare_scalping_data(
        data_path,
        filter_rth=True,
        lookahead_seconds=30,
        threshold_ticks=3.0
    )

    elapsed = time_module.perf_counter() - start

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total RTH bars: {len(full_df):,}")
    print(f"Training set: {len(train_df):,} bars")
    print(f"Validation set: {len(val_df):,} bars")
    print(f"Test set: {len(test_df):,} bars")
    print(f"Date range: {full_df.index.min()} to {full_df.index.max()}")
    print(f"Total processing time: {elapsed:.2f}s")
    print()
    print("Sample data (last 5 rows):")
    print(full_df[['open', 'high', 'low', 'close', 'volume', 'target', 'minutes_to_close']].tail())
