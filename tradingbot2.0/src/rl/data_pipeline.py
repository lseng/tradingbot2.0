"""
Data Pipeline for RL Trading System.

Handles:
1. Aggregating 1-second data to 1-minute bars
2. Generating multi-horizon targets (1h, 4h, EOD)
3. Creating features for RL agent observation space
4. Filtering to RTH (Regular Trading Hours) only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHorizonDataPipeline:
    """
    Data pipeline for multi-horizon RL trading.

    Creates features at multiple timeframes:
    - Short-term: 5-min, 15-min momentum
    - Medium-term: 1-hour trend indicators
    - Long-term: 4-hour and daily context
    """

    # RTH trading hours (NY time)
    RTH_START_HOUR = 9
    RTH_START_MIN = 30
    RTH_END_HOUR = 16
    RTH_END_MIN = 0

    # MES tick size
    TICK_SIZE = 0.25

    def __init__(self, data_path: str):
        """
        Initialize pipeline.

        Args:
            data_path: Path to 1-second parquet data
        """
        self.data_path = Path(data_path)

    def load_and_aggregate(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load 1-second data and aggregate to 1-minute bars.

        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            DataFrame with 1-minute OHLCV bars (RTH only)
        """
        logger.info(f"Loading data from {self.data_path}")

        # Read parquet
        df = pd.read_parquet(self.data_path)

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Convert to NY timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/New_York')

        logger.info(f"Loaded {len(df):,} rows, date range: {df.index.min()} to {df.index.max()}")

        # Date filter
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Filter to RTH only
        df = self._filter_rth(df)
        logger.info(f"After RTH filter: {len(df):,} rows")

        # Aggregate to 1-minute bars
        df_1min = self._aggregate_to_1min(df)
        logger.info(f"Aggregated to {len(df_1min):,} 1-minute bars")

        return df_1min

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to Regular Trading Hours only."""
        hour = df.index.hour
        minute = df.index.minute

        # 9:30 AM to 4:00 PM
        rth_mask = (
            ((hour == self.RTH_START_HOUR) & (minute >= self.RTH_START_MIN)) |
            ((hour > self.RTH_START_HOUR) & (hour < self.RTH_END_HOUR)) |
            ((hour == self.RTH_END_HOUR) & (minute == 0))
        )

        return df[rth_mask]

    def _aggregate_to_1min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1-second data to 1-minute OHLCV bars."""
        # Resample to 1-minute
        ohlcv = df.resample('1min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return ohlcv

    def generate_features(
        self,
        df: pd.DataFrame,
        include_multi_horizon: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate features for RL agent.

        Features include:
        - Price action (returns, range, position in range)
        - Momentum indicators at multiple timeframes
        - Volatility measures
        - Volume analysis
        - Time-of-day encoding

        Args:
            df: 1-minute OHLCV DataFrame
            include_multi_horizon: Include 1h, 4h horizon targets

        Returns:
            DataFrame with features, list of feature column names
        """
        df = df.copy()

        feature_cols = []

        # === Price Action ===
        # Returns at multiple windows
        for window in [1, 5, 15, 30, 60]:
            col = f'return_{window}m'
            df[col] = df['close'].pct_change(window)
            feature_cols.append(col)

        # Price position in day's range
        df['daily_high'] = df.groupby(df.index.date)['high'].transform('cummax')
        df['daily_low'] = df.groupby(df.index.date)['low'].transform('cummin')
        df['price_in_range'] = (df['close'] - df['daily_low']) / (df['daily_high'] - df['daily_low'] + 1e-8)
        feature_cols.append('price_in_range')

        # Bar range
        df['bar_range'] = (df['high'] - df['low']) / df['close']
        feature_cols.append('bar_range')

        # === Moving Averages & Momentum ===
        for period in [5, 10, 20, 50]:
            # EMA
            ema_col = f'ema_{period}'
            df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()

            # Price relative to EMA
            rel_col = f'close_vs_ema_{period}'
            df[rel_col] = (df['close'] - df[ema_col]) / df[ema_col]
            feature_cols.append(rel_col)

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (ema_12 - ema_26) / df['close']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        feature_cols.extend(['macd', 'macd_signal', 'macd_hist'])

        # === RSI at multiple timeframes ===
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period) / 100.0 - 0.5  # Center at 0
            feature_cols.append(f'rsi_{period}')

        # === Volatility ===
        for window in [5, 15, 30]:
            col = f'volatility_{window}m'
            df[col] = df['close'].pct_change().rolling(window).std()
            feature_cols.append(col)

        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean() / df['close']
        feature_cols.append('atr_14')

        # === Volume ===
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1)
        feature_cols.append('volume_ratio')

        # Volume trend
        df['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1)
        feature_cols.append('volume_trend')

        # === Time Features ===
        # Encode time of day (cyclical)
        minutes_from_open = (df.index.hour - 9) * 60 + df.index.minute - 30
        total_minutes = 390  # RTH duration

        df['time_sin'] = np.sin(2 * np.pi * minutes_from_open / total_minutes)
        df['time_cos'] = np.cos(2 * np.pi * minutes_from_open / total_minutes)
        feature_cols.extend(['time_sin', 'time_cos'])

        # Day of week
        df['dow'] = df.index.dayofweek / 4.0 - 0.5  # Normalize to [-0.5, 0.5]
        feature_cols.append('dow')

        # === Multi-Horizon Targets (for supervised pre-training) ===
        if include_multi_horizon:
            # 1-hour future return
            df['return_1h'] = df['close'].shift(-60) / df['close'] - 1
            # 4-hour future return
            df['return_4h'] = df['close'].shift(-240) / df['close'] - 1
            # EOD return (from current price to day's close)
            df['eod_close'] = df.groupby(df.index.date)['close'].transform('last')
            df['return_eod'] = df['eod_close'] / df['close'] - 1

        # Clean up
        df = df.dropna(subset=feature_cols)

        logger.info(f"Generated {len(feature_cols)} features, {len(df):,} samples")

        return df, feature_cols

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def prepare_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Prepare data for RL training.

        Args:
            start_date: Start date
            end_date: End date
            train_ratio: Fraction for training (rest for validation)

        Returns:
            (train_df, val_df, feature_columns)
        """
        # Load and aggregate
        df = self.load_and_aggregate(start_date, end_date)

        # Generate features
        df, feature_cols = self.generate_features(df, include_multi_horizon=True)

        # Normalize features
        df, feature_cols = self._normalize_features(df, feature_cols)

        # Time-based split
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        logger.info(f"Train: {len(train_df):,} bars, Val: {len(val_df):,} bars")

        return train_df, val_df, feature_cols

    def _normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Normalize features to have zero mean and unit variance."""
        df = df.copy()

        for col in feature_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 1e-8:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0.0

                # Clip outliers
                df[col] = df[col].clip(-5, 5)

        # Replace any remaining NaN/inf
        df[feature_cols] = df[feature_cols].fillna(0)
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)

        return df, feature_cols


def load_data_for_rl(
    data_path: str = "data/historical/MES/MES_1s_2years.parquet",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Convenience function to load and prepare data for RL training.

    Args:
        data_path: Path to 1-second parquet data
        start_date: Start date filter
        end_date: End date filter
        train_ratio: Training data ratio

    Returns:
        (train_df, val_df, feature_columns)
    """
    pipeline = MultiHorizonDataPipeline(data_path)
    return pipeline.prepare_training_data(start_date, end_date, train_ratio)
