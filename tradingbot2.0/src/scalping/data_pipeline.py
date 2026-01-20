"""
Data Pipeline for 5-Minute Scalping System

Loads 6.5-year 1-minute MES data, aggregates to 5-minute bars, filters to RTH,
and creates temporal train/validation/test splits.

Data file: data/historical/MES/MES_full_1min_continuous_UNadjusted.txt (122MB, 2.3M rows)
Format: datetime,open,high,low,close,volume (no header)

Why this approach:
- 5-minute bars provide cleaner signals than 1-second data
- RTH-only trading ensures better liquidity and lower slippage
- Temporal splits prevent lookahead bias in time-series data
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Constants
NY_TZ = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data pipeline."""

    data_path: Path
    train_start: str = "2019-05-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31"
    target_timeframe: str = "5min"
    rth_only: bool = True


def load_1min_data(
    file_path: Path,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load 1-minute OHLCV data from CSV/TXT file.

    Args:
        file_path: Path to data file (CSV format, no header)
        validate: Whether to validate data integrity

    Returns:
        DataFrame with datetime index (NY timezone) and OHLCV columns

    Expected file format:
        2019-05-01 00:00:00,2849.00,2849.25,2848.75,2849.00,100
    """
    logger.info(f"Loading 1-minute data from {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load data - no header in file
    df = pd.read_csv(
        file_path,
        names=["datetime", "open", "high", "low", "close", "volume"],
        parse_dates=["datetime"],
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "volume": np.int64,
        },
    )

    logger.info(f"Loaded {len(df):,} rows")

    # Set datetime as index
    df.set_index("datetime", inplace=True)

    # Localize to NY timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)

    # Sort by datetime
    df.sort_index(inplace=True)

    if validate:
        _validate_ohlcv(df)

    logger.info(
        f"Data range: {df.index.min()} to {df.index.max()}"
    )

    return df


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate OHLCV data integrity.

    Checks:
    - No NaN values in OHLCV columns
    - No negative prices
    - OHLC relationships: high >= low, high >= open, high >= close, low <= open, low <= close
    - Volume >= 0
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    # Check for NaN
    nan_counts = df[ohlcv_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
        # Drop rows with NaN
        original_len = len(df)
        df.dropna(subset=ohlcv_cols, inplace=True)
        logger.warning(f"Dropped {original_len - len(df)} rows with NaN values")

    # Check for negative prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if (df[col] <= 0).any():
            bad_count = (df[col] <= 0).sum()
            logger.warning(f"Found {bad_count} non-positive values in {col}")

    # Check OHLC relationships
    invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
    invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
    invalid_hl = df["high"] < df["low"]

    if invalid_high.any():
        logger.warning(f"Found {invalid_high.sum()} rows where high < open or high < close")
    if invalid_low.any():
        logger.warning(f"Found {invalid_low.sum()} rows where low > open or low > close")
    if invalid_hl.any():
        logger.warning(f"Found {invalid_hl.sum()} rows where high < low")

    # Check negative volume
    if (df["volume"] < 0).any():
        bad_count = (df["volume"] < 0).sum()
        logger.warning(f"Found {bad_count} negative volume values")

    logger.info("OHLCV validation complete")


def aggregate_to_5min(
    df: pd.DataFrame,
    label: str = "left",
) -> pd.DataFrame:
    """
    Aggregate 1-minute bars to 5-minute bars.

    Args:
        df: DataFrame with 1-minute OHLCV data (datetime index)
        label: Label for resampled periods ('left' = start of period)

    Returns:
        DataFrame with 5-minute OHLCV data

    Why:
        - 5-minute bars reduce noise vs 1-minute/1-second data
        - More stable patterns for gradient boosted models
        - ~460K total bars, ~127K RTH bars after filtering
    """
    logger.info(f"Aggregating {len(df):,} 1-minute bars to 5-minute bars")

    # Resample to 5-minute bars
    df_5min = df.resample("5min", label=label).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop rows where all OHLC are NaN (market closed periods)
    df_5min.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

    logger.info(f"Aggregated to {len(df_5min):,} 5-minute bars")

    return df_5min


def filter_rth(
    df: pd.DataFrame,
    start_time: time = RTH_START,
    end_time: time = RTH_END,
) -> pd.DataFrame:
    """
    Filter data to Regular Trading Hours only.

    Args:
        df: DataFrame with datetime index
        start_time: RTH start time (default 9:30 AM)
        end_time: RTH end time (default 4:00 PM, exclusive)

    Returns:
        DataFrame filtered to RTH only

    Why:
        - Better liquidity during RTH
        - Lower slippage
        - More predictable patterns
        - Avoids overnight gap risk
    """
    logger.info(f"Filtering to RTH ({start_time} - {end_time})")

    # Filter by time of day
    df_rth = df.between_time(start_time, end_time, inclusive="left")

    # Also filter out weekends (Saturday=5, Sunday=6)
    df_rth = df_rth[df_rth.index.dayofweek < 5]

    logger.info(f"Filtered to {len(df_rth):,} RTH bars")

    return df_rth


def create_temporal_splits(
    df: pd.DataFrame,
    train_start: str = "2019-05-01",
    train_end: str = "2022-12-31",
    val_start: str = "2023-01-01",
    val_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2025-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/validation/test splits.

    Args:
        df: DataFrame with datetime index
        train_start/end: Training period (model fitting)
        val_start/end: Validation period (hyperparameter tuning)
        test_start/end: Test period (final evaluation - NEVER touch during development)

    Returns:
        Tuple of (train_df, val_df, test_df)

    Why:
        - Temporal splits prevent lookahead bias
        - Test set isolation ensures honest evaluation
        - Walk-forward compatible structure
    """
    logger.info("Creating temporal train/val/test splits")

    # Convert string dates to timestamps with timezone
    train_start_dt = pd.Timestamp(train_start, tz=NY_TZ)
    train_end_dt = pd.Timestamp(train_end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    val_start_dt = pd.Timestamp(val_start, tz=NY_TZ)
    val_end_dt = pd.Timestamp(val_end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    test_start_dt = pd.Timestamp(test_start, tz=NY_TZ)
    test_end_dt = pd.Timestamp(test_end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Split data
    train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]
    val_df = df[(df.index >= val_start_dt) & (df.index <= val_end_dt)]
    test_df = df[(df.index >= test_start_dt) & (df.index <= test_end_dt)]

    logger.info(f"Train: {len(train_df):,} bars ({train_start} to {train_end})")
    logger.info(f"Val:   {len(val_df):,} bars ({val_start} to {val_end})")
    logger.info(f"Test:  {len(test_df):,} bars ({test_start} to {test_end})")

    # Verify no overlap
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df.index.max() < val_df.index.min(), "Train/val overlap detected!"
    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df.index.max() < test_df.index.min(), "Val/test overlap detected!"

    return train_df, val_df, test_df


class ScalpingDataPipeline:
    """
    Complete data pipeline for 5-minute scalping system.

    Handles:
    1. Loading 1-minute data from TXT file
    2. Aggregation to 5-minute bars
    3. RTH filtering
    4. Temporal train/val/test splits

    Example usage:
        pipeline = ScalpingDataPipeline(
            data_path=Path("data/historical/MES/MES_full_1min_continuous_UNadjusted.txt")
        )
        train_df, val_df, test_df = pipeline.load_and_split()
    """

    def __init__(self, config: Optional[DataConfig] = None, data_path: Optional[Path] = None):
        """
        Initialize data pipeline.

        Args:
            config: DataConfig with all settings
            data_path: Path to data file (alternative to config)
        """
        if config is not None:
            self.config = config
        elif data_path is not None:
            self.config = DataConfig(data_path=data_path)
        else:
            raise ValueError("Must provide either config or data_path")

        self._raw_1min: Optional[pd.DataFrame] = None
        self._aggregated_5min: Optional[pd.DataFrame] = None
        self._rth_5min: Optional[pd.DataFrame] = None

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw 1-minute data."""
        if self._raw_1min is None:
            self._raw_1min = load_1min_data(self.config.data_path)
        return self._raw_1min

    def aggregate(self) -> pd.DataFrame:
        """Aggregate to 5-minute bars."""
        if self._aggregated_5min is None:
            raw = self.load_raw_data()
            self._aggregated_5min = aggregate_to_5min(raw)
        return self._aggregated_5min

    def filter_rth_data(self) -> pd.DataFrame:
        """Filter to RTH only."""
        if self._rth_5min is None:
            agg = self.aggregate()
            if self.config.rth_only:
                self._rth_5min = filter_rth(agg)
            else:
                self._rth_5min = agg
        return self._rth_5min

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data, aggregate, filter, and split.

        Returns:
            Tuple of (train_df, val_df, test_df) with 5-minute RTH bars
        """
        rth_data = self.filter_rth_data()

        return create_temporal_splits(
            rth_data,
            train_start=self.config.train_start,
            train_end=self.config.train_end,
            val_start=self.config.val_start,
            val_end=self.config.val_end,
            test_start=self.config.test_start,
            test_end=self.config.test_end,
        )

    def get_stats(self) -> dict:
        """Get data statistics."""
        stats = {
            "data_path": str(self.config.data_path),
            "config": {
                "train_period": f"{self.config.train_start} to {self.config.train_end}",
                "val_period": f"{self.config.val_start} to {self.config.val_end}",
                "test_period": f"{self.config.test_start} to {self.config.test_end}",
                "rth_only": self.config.rth_only,
            },
        }

        if self._raw_1min is not None:
            stats["raw_1min_bars"] = len(self._raw_1min)
            stats["raw_date_range"] = f"{self._raw_1min.index.min()} to {self._raw_1min.index.max()}"

        if self._aggregated_5min is not None:
            stats["aggregated_5min_bars"] = len(self._aggregated_5min)

        if self._rth_5min is not None:
            stats["rth_5min_bars"] = len(self._rth_5min)

        return stats
