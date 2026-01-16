"""
Data Loading and Preprocessing Module for Futures Trading ML Model.

This module handles:
- Loading raw OHLCV data from CSV/TXT files
- Data cleaning and validation
- Resampling to daily bars for next-day prediction
- Train/test splitting with temporal awareness
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturesDataLoader:
    """Load and preprocess futures OHLCV data."""

    def __init__(self, data_path: str):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the raw data file (CSV/TXT format)
        """
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.daily_data: Optional[pd.DataFrame] = None

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw 1-minute OHLCV data from file.

        Expected format: datetime,open,high,low,close,volume
        """
        logger.info(f"Loading data from {self.data_path}")

        # Load data - handle both CSV and TXT formats
        self.raw_data = pd.read_csv(
            self.data_path,
            header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['datetime'],
            dtype={
                'open': np.float64,
                'high': np.float64,
                'low': np.float64,
                'close': np.float64,
                'volume': np.int64
            }
        )

        # Set datetime as index
        self.raw_data.set_index('datetime', inplace=True)
        self.raw_data.sort_index(inplace=True)

        # Basic validation
        self._validate_data()

        logger.info(f"Loaded {len(self.raw_data):,} rows")
        logger.info(f"Date range: {self.raw_data.index.min()} to {self.raw_data.index.max()}")

        return self.raw_data

    def _validate_data(self):
        """Validate data integrity."""
        df = self.raw_data

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Forward fill missing prices, set missing volume to 0
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
            df['volume'] = df['volume'].fillna(0)

        # Check for negative prices or volumes
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.warning("Found non-positive prices, filtering...")
            mask = (df[['open', 'high', 'low', 'close']] > 0).all(axis=1)
            self.raw_data = df[mask]

        if (df['volume'] < 0).any():
            logger.warning("Found negative volumes, setting to 0...")
            df.loc[df['volume'] < 0, 'volume'] = 0

        # Check OHLC validity (high >= low, high >= open/close, low <= open/close)
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            logger.warning(f"Found {count} rows with invalid OHLC relationships, filtering...")
            self.raw_data = df[~invalid_ohlc]

    def resample_to_daily(self, session_start: str = "18:00", session_end: str = "17:00") -> pd.DataFrame:
        """
        Resample 1-minute data to daily OHLCV bars.

        CME Futures sessions typically run from 6:00 PM to 5:00 PM next day.

        Args:
            session_start: Session start time (default 18:00 for CME)
            session_end: Session end time (default 17:00 next day)

        Returns:
            DataFrame with daily OHLCV data
        """
        if self.raw_data is None:
            self.load_raw_data()

        logger.info("Resampling to daily bars...")

        # For simplicity, use calendar day resampling
        # In production, you'd want proper session-based resampling
        daily = self.raw_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Filter out weekends and holidays (low volume days)
        min_volume_threshold = daily['volume'].quantile(0.05)
        daily = daily[daily['volume'] >= min_volume_threshold]

        self.daily_data = daily
        logger.info(f"Created {len(daily)} daily bars")

        return daily

    def create_target_variable(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        Create binary target: 1 if next day's close > today's close, 0 otherwise.

        Args:
            df: DataFrame with OHLCV data
            lookahead: Number of days ahead to predict (default 1)

        Returns:
            DataFrame with target column added
        """
        df = df.copy()

        # Binary classification: up (1) or down (0)
        df['target'] = (df['close'].shift(-lookahead) > df['close']).astype(int)

        # Also store the actual return for analysis
        df['next_return'] = df['close'].pct_change(lookahead).shift(-lookahead)

        # Remove last rows where we don't have future data
        df = df.iloc[:-lookahead]

        logger.info(f"Target distribution: Up={df['target'].sum()} ({df['target'].mean():.1%}), "
                   f"Down={len(df) - df['target'].sum()} ({1 - df['target'].mean():.1%})")

        return df

    def train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data temporally (no shuffling to preserve time series).

        Args:
            df: DataFrame to split
            train_ratio: Fraction of data for training
            validation_ratio: Fraction for validation (from remaining after train)

        Returns:
            Tuple of (train_df, test_df, val_df or None)
        """
        n = len(df)
        train_end = int(n * train_ratio)

        train_df = df.iloc[:train_end]
        remaining = df.iloc[train_end:]

        if validation_ratio > 0:
            val_size = int(len(remaining) * validation_ratio / (1 - train_ratio))
            val_df = remaining.iloc[:val_size]
            test_df = remaining.iloc[val_size:]
            logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, test_df, val_df
        else:
            logger.info(f"Split: Train={len(train_df)}, Test={len(remaining)}")
            return train_df, remaining, None


def load_and_prepare_data(
    data_path: str,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and prepare data in one step.

    Args:
        data_path: Path to raw 1-minute data
        train_ratio: Train/test split ratio

    Returns:
        Tuple of (full_df, train_df, test_df)
    """
    loader = FuturesDataLoader(data_path)
    loader.load_raw_data()
    daily = loader.resample_to_daily()
    daily_with_target = loader.create_target_variable(daily)
    train_df, test_df, _ = loader.train_test_split(daily_with_target, train_ratio)

    return daily_with_target, train_df, test_df


if __name__ == "__main__":
    # Test with sample data path
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "/Users/leoneng/Downloads/MES_full_1min_continuous_UNadjusted.txt"

    full_df, train_df, test_df = load_and_prepare_data(data_path)

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total daily bars: {len(full_df)}")
    print(f"Training set: {len(train_df)} ({len(train_df)/len(full_df):.1%})")
    print(f"Test set: {len(test_df)} ({len(test_df)/len(full_df):.1%})")
    print(f"\nSample data:\n{full_df.tail()}")
