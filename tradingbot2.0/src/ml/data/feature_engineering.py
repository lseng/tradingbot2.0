"""
Feature Engineering Module for Futures Trading ML Model.

Creates technical indicators and derived features for price prediction.

Feature Categories:
1. Price-based: Returns, log returns, price momentum
2. Moving Averages: SMA, EMA at various windows
3. Volatility: ATR, Bollinger Bands, realized volatility
4. Volume: Volume MA, volume ratio, OBV
5. Momentum Indicators: RSI, MACD, Stochastic
6. Pattern Features: Candlestick patterns, support/resistance levels
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical features for ML model."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self.feature_names: List[str] = []

    def add_returns(self, periods: List[int] = [1, 5, 10, 21]) -> 'FeatureEngineer':
        """
        Add return features for different lookback periods.

        Args:
            periods: List of lookback periods (days)
        """
        for period in periods:
            # Simple returns
            self.df[f'return_{period}d'] = self.df['close'].pct_change(period)
            self.feature_names.append(f'return_{period}d')

            # Log returns (more normally distributed)
            self.df[f'log_return_{period}d'] = np.log(self.df['close'] / self.df['close'].shift(period))
            self.feature_names.append(f'log_return_{period}d')

        return self

    def add_moving_averages(
        self,
        sma_periods: List[int] = [5, 10, 20, 50],
        ema_periods: List[int] = [5, 10, 20]
    ) -> 'FeatureEngineer':
        """
        Add Simple and Exponential Moving Averages.

        Creates MA values and relative position of price to MAs.
        """
        # Simple Moving Averages
        for period in sma_periods:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()

            # Price relative to SMA (more useful than raw SMA)
            self.df[f'close_to_sma_{period}'] = (
                self.df['close'] - self.df[f'sma_{period}']
            ) / self.df[f'sma_{period}']
            self.feature_names.append(f'close_to_sma_{period}')

        # Exponential Moving Averages
        for period in ema_periods:
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()

            self.df[f'close_to_ema_{period}'] = (
                self.df['close'] - self.df[f'ema_{period}']
            ) / self.df[f'ema_{period}']
            self.feature_names.append(f'close_to_ema_{period}')

        # MA Crossover features
        if 5 in sma_periods and 20 in sma_periods:
            self.df['sma_5_20_cross'] = (
                self.df['sma_5'] - self.df['sma_20']
            ) / self.df['sma_20']
            self.feature_names.append('sma_5_20_cross')

        if 10 in sma_periods and 50 in sma_periods:
            self.df['sma_10_50_cross'] = (
                self.df['sma_10'] - self.df['sma_50']
            ) / self.df['sma_50']
            self.feature_names.append('sma_10_50_cross')

        return self

    def add_volatility_features(
        self,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vol_windows: List[int] = [5, 10, 21]
    ) -> 'FeatureEngineer':
        """
        Add volatility-based features.

        - ATR (Average True Range)
        - Bollinger Bands position
        - Realized volatility
        """
        # True Range
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift(1))
        low_close = abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR
        self.df['atr'] = true_range.rolling(window=atr_period).mean()
        # Normalized ATR (relative to price)
        self.df['atr_pct'] = self.df['atr'] / self.df['close']
        self.feature_names.append('atr_pct')

        # Bollinger Bands
        sma = self.df['close'].rolling(window=bb_period).mean()
        std = self.df['close'].rolling(window=bb_period).std()
        self.df['bb_upper'] = sma + (bb_std * std)
        self.df['bb_lower'] = sma - (bb_std * std)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / sma
        self.feature_names.append('bb_width')

        # Bollinger Band position (where price is within the bands)
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / (
            self.df['bb_upper'] - self.df['bb_lower']
        )
        self.feature_names.append('bb_position')

        # Realized volatility (annualized standard deviation of returns)
        for window in vol_windows:
            self.df[f'volatility_{window}d'] = (
                self.df['close'].pct_change().rolling(window=window).std() * np.sqrt(252)
            )
            self.feature_names.append(f'volatility_{window}d')

        return self

    def add_momentum_indicators(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_period: int = 14
    ) -> 'FeatureEngineer':
        """
        Add momentum-based technical indicators.

        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator
        """
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        # Normalize RSI to [-1, 1] range for neural network
        self.df['rsi_normalized'] = (self.df['rsi'] - 50) / 50
        self.feature_names.append('rsi_normalized')

        # MACD
        ema_fast = self.df['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=macd_slow, adjust=False).mean()
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=macd_signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        # Normalize MACD features
        self.df['macd_normalized'] = self.df['macd'] / self.df['close']
        self.df['macd_hist_normalized'] = self.df['macd_histogram'] / self.df['close']
        self.feature_names.extend(['macd_normalized', 'macd_hist_normalized'])

        # Stochastic Oscillator
        low_min = self.df['low'].rolling(window=stoch_period).min()
        high_max = self.df['high'].rolling(window=stoch_period).max()
        self.df['stoch_k'] = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=3).mean()
        # Normalize
        self.df['stoch_k_normalized'] = (self.df['stoch_k'] - 50) / 50
        self.df['stoch_d_normalized'] = (self.df['stoch_d'] - 50) / 50
        self.feature_names.extend(['stoch_k_normalized', 'stoch_d_normalized'])

        return self

    def add_volume_features(self, periods: List[int] = [5, 10, 20]) -> 'FeatureEngineer':
        """
        Add volume-based features.

        - Volume moving averages
        - Relative volume
        - On-Balance Volume (OBV) derivative
        """
        for period in periods:
            vol_ma = self.df['volume'].rolling(window=period).mean()
            self.df[f'volume_ratio_{period}d'] = self.df['volume'] / vol_ma
            self.feature_names.append(f'volume_ratio_{period}d')

        # On-Balance Volume (OBV) rate of change
        obv = (np.sign(self.df['close'].diff()) * self.df['volume']).cumsum()
        self.df['obv_roc'] = obv.pct_change(5)  # 5-day OBV change
        self.feature_names.append('obv_roc')

        # Volume-price trend
        self.df['vpt'] = (
            self.df['volume'] * self.df['close'].pct_change()
        ).cumsum()
        self.df['vpt_roc'] = self.df['vpt'].pct_change(5)
        self.feature_names.append('vpt_roc')

        return self

    def add_candlestick_features(self) -> 'FeatureEngineer':
        """
        Add candlestick pattern features.

        - Body size (relative)
        - Wick sizes (upper/lower)
        - Gap direction
        """
        # Body size (positive = bullish, negative = bearish)
        body = self.df['close'] - self.df['open']
        self.df['body_pct'] = body / self.df['open']
        self.feature_names.append('body_pct')

        # Relative body size
        total_range = self.df['high'] - self.df['low']
        self.df['body_range_ratio'] = abs(body) / total_range.replace(0, np.nan)
        self.feature_names.append('body_range_ratio')

        # Upper wick
        upper_wick = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['upper_wick_pct'] = upper_wick / total_range.replace(0, np.nan)
        self.feature_names.append('upper_wick_pct')

        # Lower wick
        lower_wick = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['lower_wick_pct'] = lower_wick / total_range.replace(0, np.nan)
        self.feature_names.append('lower_wick_pct')

        # Gap (open vs previous close)
        self.df['gap_pct'] = (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1)
        self.feature_names.append('gap_pct')

        return self

    def add_time_features(self) -> 'FeatureEngineer':
        """
        Add time-based features (day of week, month, etc.).

        These can capture seasonal patterns.
        """
        if isinstance(self.df.index, pd.DatetimeIndex):
            # Day of week (Monday=0, Friday=4)
            self.df['day_of_week'] = self.df.index.dayofweek
            # Normalize to [-1, 1]
            self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 5)
            self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 5)
            self.feature_names.extend(['day_of_week_sin', 'day_of_week_cos'])

            # Month (for seasonal patterns)
            self.df['month'] = self.df.index.month
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            self.feature_names.extend(['month_sin', 'month_cos'])

        return self

    def generate_all_features(self) -> pd.DataFrame:
        """
        Generate all features in one call.

        Returns:
            DataFrame with all features added
        """
        logger.info("Generating all features...")

        self.add_returns()
        self.add_moving_averages()
        self.add_volatility_features()
        self.add_momentum_indicators()
        self.add_volume_features()
        self.add_candlestick_features()
        self.add_time_features()

        # Remove any rows with NaN values (from lookback windows)
        initial_len = len(self.df)
        self.df = self.df.dropna()
        final_len = len(self.df)

        logger.info(f"Generated {len(self.feature_names)} features")
        logger.info(f"Dropped {initial_len - final_len} rows with NaN values")
        logger.info(f"Final dataset: {final_len} rows")

        return self.df

    def get_feature_names(self) -> List[str]:
        """Return list of generated feature names."""
        return self.feature_names.copy()

    def get_feature_matrix(self) -> np.ndarray:
        """Return feature matrix as numpy array."""
        return self.df[self.feature_names].values


def prepare_features_for_training(
    df: pd.DataFrame,
    normalize: bool = True
) -> tuple:
    """
    Prepare features for neural network training.

    Args:
        df: DataFrame with OHLCV data and target
        normalize: Whether to normalize features

    Returns:
        Tuple of (features_df, feature_names, scaler or None)
    """
    # Generate features
    engineer = FeatureEngineer(df)
    df_features = engineer.generate_all_features()
    feature_names = engineer.get_feature_names()

    scaler = None
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # Fit scaler on training data only (assumed to be passed in)
        df_features[feature_names] = scaler.fit_transform(df_features[feature_names])

    return df_features, feature_names, scaler


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.data_loader import load_and_prepare_data

    # Load sample data
    data_path = "/Users/leoneng/Downloads/MES_full_1min_continuous_UNadjusted.txt"
    full_df, train_df, test_df = load_and_prepare_data(data_path)

    # Generate features on training data
    engineer = FeatureEngineer(train_df)
    featured_df = engineer.generate_all_features()

    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total features: {len(engineer.get_feature_names())}")
    print(f"\nFeature list:")
    for i, name in enumerate(engineer.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
    print(f"\nSample data shape: {featured_df.shape}")
    print(f"\nFeature statistics:\n{featured_df[engineer.get_feature_names()].describe()}")
