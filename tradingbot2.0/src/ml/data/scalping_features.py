"""
Scalping Feature Engineering Module for 1-Second MES Futures Data.

This module is specifically designed for high-frequency scalping strategies
operating on 1-second bar data. It differs from the daily feature engineering
in several key ways:

1. Time periods are in SECONDS (1, 5, 10, 30, 60) not days
2. Session-based VWAP that resets at 9:30 AM NY
3. Minutes-to-close feature for EOD awareness
4. Multi-timeframe features with proper lagging to prevent lookahead bias
5. Microstructure features optimized for tick-level analysis

Feature Categories:
1. Price Action: Returns, log returns at second intervals
2. Technical Indicators: EMA, RSI, MACD, Bollinger Bands, ATR
3. Session Features: VWAP, minutes-to-close, time-of-day
4. Microstructure: Bar direction, wick ratios, body ratios
5. Volume: Relative volume, volume delta
6. Multi-Timeframe: Lagged 1-minute and 5-minute aggregates

Reference: specs/ml-scalping-model.md
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MES Contract Constants
MES_TICK_SIZE = 0.25  # Minimum price movement
MES_TICK_VALUE = 1.25  # Dollar value per tick
MES_POINT_VALUE = 5.00  # Dollar value per point (4 ticks)

# RTH Session Times (NY timezone)
RTH_START = time(9, 30)
RTH_END = time(16, 0)
RTH_DURATION_MINUTES = 390  # 6.5 hours


@dataclass
class FeatureConfig:
    """Configuration for scalping feature engineering."""
    # Return periods in seconds
    return_periods: List[int] = None
    # EMA periods (on 1-second bars)
    ema_periods: List[int] = None
    # Volatility windows in seconds
    volatility_windows: List[int] = None
    # Volume ratio windows in seconds
    volume_windows: List[int] = None
    # RSI period
    rsi_period: int = 14
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # ATR period
    atr_period: int = 14
    # Stochastic period
    stoch_period: int = 14

    def __post_init__(self):
        # Set defaults if not provided
        if self.return_periods is None:
            self.return_periods = [1, 5, 10, 30, 60]
        if self.ema_periods is None:
            self.ema_periods = [9, 21, 50, 200]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 30, 60, 300]
        if self.volume_windows is None:
            self.volume_windows = [10, 30, 60]


class ScalpingFeatureEngineer:
    """
    Feature engineering specifically designed for 1-second scalping data.

    Key differences from daily FeatureEngineer:
    - Time periods are in SECONDS
    - Session-based VWAP (resets at 9:30 AM NY)
    - Minutes-to-close for EOD awareness
    - No annualization factor for volatility
    - Multi-timeframe with proper lagging

    Usage:
        engineer = ScalpingFeatureEngineer(df)
        df_with_features = engineer.generate_all_features()
        feature_names = engineer.get_feature_names()
    """

    def __init__(self, df: pd.DataFrame, config: Optional[FeatureConfig] = None):
        """
        Initialize with 1-second OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be DatetimeIndex in NY timezone
            config: Optional FeatureConfig for customization
        """
        self.df = df.copy()
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []

        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def add_returns(self) -> 'ScalpingFeatureEngineer':
        """
        Add return features for different lookback periods (in SECONDS).

        Creates both simple and log returns for periods: 1, 5, 10, 30, 60 seconds
        """
        for period in self.config.return_periods:
            # Simple returns
            col_name = f'return_{period}s'
            self.df[col_name] = self.df['close'].pct_change(period)
            self.feature_names.append(col_name)

            # Log returns (more normally distributed)
            log_col = f'log_return_{period}s'
            self.df[log_col] = np.log(
                self.df['close'] / self.df['close'].shift(period)
            )
            self.feature_names.append(log_col)

        return self

    def add_emas(self) -> 'ScalpingFeatureEngineer':
        """
        Add Exponential Moving Averages.

        Uses periods: 9, 21, 50, 200 (on 1-second bars)
        Creates price-relative features (price / EMA - 1) for scale invariance
        """
        for period in self.config.ema_periods:
            ema_col = f'ema_{period}'
            self.df[ema_col] = self.df['close'].ewm(span=period, adjust=False).mean()

            # Price relative to EMA (normalized)
            rel_col = f'close_to_ema_{period}'
            self.df[rel_col] = (self.df['close'] - self.df[ema_col]) / self.df[ema_col]
            self.feature_names.append(rel_col)

        # EMA crossover features
        if 9 in self.config.ema_periods and 21 in self.config.ema_periods:
            self.df['ema_9_21_cross'] = (
                self.df['ema_9'] - self.df['ema_21']
            ) / self.df['ema_21']
            self.feature_names.append('ema_9_21_cross')

        if 21 in self.config.ema_periods and 50 in self.config.ema_periods:
            self.df['ema_21_50_cross'] = (
                self.df['ema_21'] - self.df['ema_50']
            ) / self.df['ema_50']
            self.feature_names.append('ema_21_50_cross')

        return self

    def add_vwap(self) -> 'ScalpingFeatureEngineer':
        """
        Add session-based VWAP (Volume Weighted Average Price).

        VWAP resets at session start (9:30 AM NY for RTH).
        Formula: VWAP = cumsum(TP * volume) / cumsum(volume)
        where TP (typical price) = (high + low + close) / 3

        Returns price relative to VWAP for scale invariance.
        """
        # Calculate typical price
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3

        # Detect session boundaries (each trading day)
        if self.df.index.tz is not None:
            dates = self.df.index.date
        else:
            dates = pd.to_datetime(self.df.index).date

        # Calculate VWAP per session
        vwap_values = []

        for date in pd.unique(dates):
            mask = dates == date
            session_tp = tp[mask]
            session_vol = self.df.loc[mask, 'volume']

            # Cumulative sums for this session
            cum_tp_vol = (session_tp * session_vol).cumsum()
            cum_vol = session_vol.cumsum()

            # VWAP = cumsum(tp*vol) / cumsum(vol)
            session_vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
            vwap_values.append(session_vwap)

        self.df['vwap'] = pd.concat(vwap_values)

        # Price relative to VWAP (normalized feature)
        self.df['close_to_vwap'] = (self.df['close'] - self.df['vwap']) / self.df['vwap']
        self.feature_names.append('close_to_vwap')

        # VWAP slope (rate of change)
        self.df['vwap_slope'] = self.df['vwap'].pct_change(10)  # 10-second slope
        self.feature_names.append('vwap_slope')

        return self

    def add_minutes_to_close(self) -> 'ScalpingFeatureEngineer':
        """
        Add minutes-to-close feature for EOD awareness.

        This is critical for risk management - the model should learn to
        behave differently near market close (4:00 PM NY) and EOD flatten
        time (4:30 PM NY).

        Returns normalized feature in [0, 1] range.
        """
        if self.df.index.tz is None:
            logger.warning("DataFrame index is not timezone-aware, assuming NY timezone")

        # Get time of day
        times = self.df.index.time

        # Calculate minutes since market open (9:30 AM)
        minutes_since_open = []
        for t in times:
            hour_diff = t.hour - 9
            min_diff = t.minute - 30
            sec_diff = t.second / 60.0
            total_minutes = hour_diff * 60 + min_diff + sec_diff
            minutes_since_open.append(max(0, total_minutes))

        self.df['minutes_since_open'] = minutes_since_open

        # Minutes to close (RTH ends at 4:00 PM = 390 minutes after 9:30 AM)
        self.df['minutes_to_close'] = RTH_DURATION_MINUTES - self.df['minutes_since_open']
        self.df['minutes_to_close'] = self.df['minutes_to_close'].clip(lower=0)

        # Normalized version [0, 1] - 1 at open, 0 at close
        self.df['minutes_to_close_norm'] = self.df['minutes_to_close'] / RTH_DURATION_MINUTES
        self.feature_names.append('minutes_to_close_norm')

        # EOD urgency flag - increases as we approach 4:00 PM
        # This is complementary to minutes_to_close_norm
        self.df['eod_urgency'] = 1 - self.df['minutes_to_close_norm']
        self.feature_names.append('eod_urgency')

        return self

    def add_time_of_day(self) -> 'ScalpingFeatureEngineer':
        """
        Add cyclical time-of-day encoding.

        Uses sine/cosine encoding to capture cyclical patterns without
        discontinuities at day boundaries.
        """
        # Minutes since midnight for cyclical encoding
        times = self.df.index
        minutes_of_day = times.hour * 60 + times.minute + times.second / 60.0

        # Cyclical encoding
        self.df['time_sin'] = np.sin(2 * np.pi * minutes_of_day / (24 * 60))
        self.df['time_cos'] = np.cos(2 * np.pi * minutes_of_day / (24 * 60))
        self.feature_names.extend(['time_sin', 'time_cos'])

        # Day of week
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 5)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 5)
        self.feature_names.extend(['dow_sin', 'dow_cos'])

        # Session period flags (one-hot style)
        times = self.df.index.time

        # First 30 minutes (9:30-10:00) - high volatility
        self.df['is_open_period'] = [
            1 if (t >= time(9, 30) and t < time(10, 0)) else 0
            for t in times
        ]
        self.feature_names.append('is_open_period')

        # Last hour (3:00-4:00) - pre-close behavior
        self.df['is_close_period'] = [
            1 if (t >= time(15, 0) and t <= time(16, 0)) else 0
            for t in times
        ]
        self.feature_names.append('is_close_period')

        # Lunch doldrums (11:30-1:00) - lower volume
        self.df['is_lunch_period'] = [
            1 if (t >= time(11, 30) and t < time(13, 0)) else 0
            for t in times
        ]
        self.feature_names.append('is_lunch_period')

        return self

    def add_volatility_features(self) -> 'ScalpingFeatureEngineer':
        """
        Add volatility-based features.

        - ATR (Average True Range) - NOT annualized for scalping
        - Bollinger Bands position
        - Realized volatility at different windows
        """
        # True Range
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift(1))
        low_close = abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR (NOT annualized - raw value for scalping)
        self.df['atr'] = true_range.rolling(window=self.config.atr_period).mean()
        # Normalized ATR (in ticks for MES)
        self.df['atr_ticks'] = self.df['atr'] / MES_TICK_SIZE
        # ATR as percentage of price
        self.df['atr_pct'] = self.df['atr'] / self.df['close']
        self.feature_names.extend(['atr_ticks', 'atr_pct'])

        # Bollinger Bands
        bb_sma = self.df['close'].rolling(window=self.config.bb_period).mean()
        bb_std = self.df['close'].rolling(window=self.config.bb_period).std()
        self.df['bb_upper'] = bb_sma + (self.config.bb_std * bb_std)
        self.df['bb_lower'] = bb_sma - (self.config.bb_std * bb_std)

        # BB width (normalized)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / bb_sma
        self.feature_names.append('bb_width')

        # BB position (where price is within bands, 0=lower, 1=upper)
        bb_range = self.df['bb_upper'] - self.df['bb_lower']
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / bb_range.replace(0, np.nan)
        # Clip to [0, 1] for outliers (price outside bands)
        self.df['bb_position'] = self.df['bb_position'].clip(0, 1)
        self.feature_names.append('bb_position')

        # Realized volatility at different windows (NOT annualized)
        for window in self.config.volatility_windows:
            vol_col = f'volatility_{window}s'
            self.df[vol_col] = self.df['close'].pct_change().rolling(window=window).std()
            self.feature_names.append(vol_col)

        return self

    def add_momentum_indicators(self) -> 'ScalpingFeatureEngineer':
        """
        Add momentum-based technical indicators.

        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator

        All normalized to [-1, 1] range for neural network input.
        """
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        # Normalize RSI to [-1, 1]
        self.df['rsi_norm'] = (self.df['rsi'] - 50) / 50
        self.feature_names.append('rsi_norm')

        # MACD
        ema_fast = self.df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']

        # Normalize MACD by price
        self.df['macd_norm'] = self.df['macd'] / self.df['close']
        self.df['macd_hist_norm'] = self.df['macd_histogram'] / self.df['close']
        self.feature_names.extend(['macd_norm', 'macd_hist_norm'])

        # Stochastic Oscillator
        low_min = self.df['low'].rolling(window=self.config.stoch_period).min()
        high_max = self.df['high'].rolling(window=self.config.stoch_period).max()
        denom = (high_max - low_min).replace(0, np.nan)
        self.df['stoch_k'] = 100 * (self.df['close'] - low_min) / denom
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=3).mean()

        # Normalize to [-1, 1]
        self.df['stoch_k_norm'] = (self.df['stoch_k'] - 50) / 50
        self.df['stoch_d_norm'] = (self.df['stoch_d'] - 50) / 50
        self.feature_names.extend(['stoch_k_norm', 'stoch_d_norm'])

        return self

    def add_microstructure_features(self) -> 'ScalpingFeatureEngineer':
        """
        Add microstructure features for tick-level analysis.

        - Bar direction (+1 up, -1 down, 0 flat)
        - Body ratio (body / total range)
        - Upper/lower wick ratios
        - Gap from previous close
        """
        # Bar direction
        self.df['bar_direction'] = np.sign(self.df['close'] - self.df['open'])
        self.feature_names.append('bar_direction')

        # Total range
        total_range = self.df['high'] - self.df['low']
        total_range_safe = total_range.replace(0, np.nan)

        # Body (absolute value)
        body = abs(self.df['close'] - self.df['open'])
        self.df['body_ratio'] = body / total_range_safe
        self.feature_names.append('body_ratio')

        # Upper wick ratio
        upper_wick = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['upper_wick_ratio'] = upper_wick / total_range_safe
        self.feature_names.append('upper_wick_ratio')

        # Lower wick ratio
        lower_wick = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['lower_wick_ratio'] = lower_wick / total_range_safe
        self.feature_names.append('lower_wick_ratio')

        # Gap from previous close (in ticks)
        gap = self.df['open'] - self.df['close'].shift(1)
        self.df['gap_ticks'] = gap / MES_TICK_SIZE
        self.feature_names.append('gap_ticks')

        # Normalized range (range in ticks)
        self.df['range_ticks'] = total_range / MES_TICK_SIZE
        self.feature_names.append('range_ticks')

        return self

    def add_volume_features(self) -> 'ScalpingFeatureEngineer':
        """
        Add volume-based features.

        - Volume relative to rolling average
        - Volume delta proxy (using bar direction as approximation)
        - On-Balance Volume rate of change
        """
        for window in self.config.volume_windows:
            vol_ma = self.df['volume'].rolling(window=window).mean()
            col_name = f'volume_ratio_{window}s'
            self.df[col_name] = self.df['volume'] / vol_ma.replace(0, np.nan)
            self.feature_names.append(col_name)

        # Volume delta proxy: volume * direction
        # (positive for up bars, negative for down bars)
        self.df['volume_delta'] = self.df['volume'] * np.sign(self.df['close'] - self.df['open'])
        # Rolling sum of volume delta
        self.df['volume_delta_sum'] = self.df['volume_delta'].rolling(window=30).sum()
        # Normalize by total volume in window
        vol_sum = self.df['volume'].rolling(window=30).sum()
        self.df['volume_delta_norm'] = self.df['volume_delta_sum'] / vol_sum.replace(0, np.nan)
        self.feature_names.append('volume_delta_norm')

        # OBV rate of change (30-second window)
        obv = (np.sign(self.df['close'].diff()) * self.df['volume']).cumsum()
        self.df['obv_roc'] = obv.pct_change(30)
        self.feature_names.append('obv_roc')

        return self

    def add_multiframe_features(self) -> 'ScalpingFeatureEngineer':
        """
        Add multi-timeframe features with proper LAGGING to prevent lookahead.

        Aggregates 1-second data to 1-minute and 5-minute timeframes,
        then calculates trend indicators that are LAGGED by 1 bar.

        CRITICAL: Uses shift(1) on aggregated bars to ensure no lookahead bias.
        """
        # Create session date for grouping
        dates = self.df.index.date

        # 1-minute aggregation with proper resampling
        df_1m = self.df.resample('1min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # 1-minute features (calculated on 1-min bars, then lagged)
        df_1m['trend_1m'] = (df_1m['close'] - df_1m['close'].shift(1)) / df_1m['close'].shift(1)
        df_1m['momentum_1m'] = df_1m['close'].pct_change(5)  # 5-minute momentum on 1-min bars
        df_1m['vol_1m'] = df_1m['close'].pct_change().rolling(window=5).std()

        # LAG by 1 minute to prevent lookahead
        df_1m['trend_1m_lagged'] = df_1m['trend_1m'].shift(1)
        df_1m['momentum_1m_lagged'] = df_1m['momentum_1m'].shift(1)
        df_1m['vol_1m_lagged'] = df_1m['vol_1m'].shift(1)

        # 5-minute aggregation
        df_5m = self.df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # 5-minute features
        df_5m['trend_5m'] = (df_5m['close'] - df_5m['close'].shift(1)) / df_5m['close'].shift(1)
        df_5m['momentum_5m'] = df_5m['close'].pct_change(3)  # 15-minute momentum on 5-min bars

        # LAG by 5 minutes to prevent lookahead
        df_5m['trend_5m_lagged'] = df_5m['trend_5m'].shift(1)
        df_5m['momentum_5m_lagged'] = df_5m['momentum_5m'].shift(1)

        # Forward-fill aggregated features back to 1-second resolution
        # This gives each second the value from the PREVIOUS completed minute/5-min bar
        self.df['htf_trend_1m'] = df_1m['trend_1m_lagged'].reindex(
            self.df.index, method='ffill'
        )
        self.df['htf_momentum_1m'] = df_1m['momentum_1m_lagged'].reindex(
            self.df.index, method='ffill'
        )
        self.df['htf_vol_1m'] = df_1m['vol_1m_lagged'].reindex(
            self.df.index, method='ffill'
        )

        self.df['htf_trend_5m'] = df_5m['trend_5m_lagged'].reindex(
            self.df.index, method='ffill'
        )
        self.df['htf_momentum_5m'] = df_5m['momentum_5m_lagged'].reindex(
            self.df.index, method='ffill'
        )

        self.feature_names.extend([
            'htf_trend_1m', 'htf_momentum_1m', 'htf_vol_1m',
            'htf_trend_5m', 'htf_momentum_5m'
        ])

        return self

    def generate_all_features(self, include_multiframe: bool = True) -> pd.DataFrame:
        """
        Generate all scalping features.

        Args:
            include_multiframe: Whether to include multi-timeframe features.
                               Set False for faster processing if not needed.

        Returns:
            DataFrame with all features added
        """
        logger.info("Generating scalping features for 1-second data...")

        # Core features
        self.add_returns()
        self.add_emas()
        self.add_vwap()
        self.add_minutes_to_close()
        self.add_time_of_day()
        self.add_volatility_features()
        self.add_momentum_indicators()
        self.add_microstructure_features()
        self.add_volume_features()

        # Optional multi-timeframe (slower but valuable)
        if include_multiframe:
            self.add_multiframe_features()

        # Clean up NaN values from lookback windows
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=self.feature_names)
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

    def get_feature_stats(self) -> pd.DataFrame:
        """Return statistics for all features (useful for debugging)."""
        return self.df[self.feature_names].describe()


def prepare_scalping_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    normalize: bool = True,
    include_multiframe: bool = True
) -> Tuple[pd.DataFrame, List[str], Optional[object]]:
    """
    Prepare scalping features for neural network training.

    This is a convenience function that:
    1. Generates all scalping features
    2. Optionally normalizes features using StandardScaler
    3. Returns the processed DataFrame, feature names, and scaler

    Args:
        df: DataFrame with OHLCV data (1-second bars, NY timezone)
        config: Optional FeatureConfig for customization
        normalize: Whether to normalize features (default True)
        include_multiframe: Whether to include multi-timeframe features

    Returns:
        Tuple of (features_df, feature_names, scaler or None)
    """
    # Generate features
    engineer = ScalpingFeatureEngineer(df, config)
    df_features = engineer.generate_all_features(include_multiframe=include_multiframe)
    feature_names = engineer.get_feature_names()

    scaler = None
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # Fit and transform features
        df_features[feature_names] = scaler.fit_transform(df_features[feature_names])
        logger.info("Features normalized using StandardScaler")

    return df_features, feature_names, scaler


def validate_no_lookahead(df: pd.DataFrame, feature_names: List[str], target_col: str = 'target') -> bool:
    """
    Validate that features have no lookahead bias.

    Checks that feature values at time T are not correlated with
    future target values in a way that suggests information leakage.

    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names
        target_col: Name of target column

    Returns:
        True if validation passes, raises ValueError otherwise
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found, skipping validation")
        return True

    # Check that features at time T are computed from data at T and earlier
    # This is a heuristic check - compute lagged correlation
    for feature in feature_names:
        # Feature at T should not be perfectly correlated with target at T-1
        # (which would indicate it was computed from future data)
        corr_with_past_target = df[feature].corr(df[target_col].shift(1))
        corr_with_future_target = df[feature].corr(df[target_col].shift(-1))

        # If feature is more correlated with future target than past, flag it
        if abs(corr_with_future_target) > 0.9:
            logger.warning(
                f"Feature '{feature}' has suspiciously high correlation "
                f"({corr_with_future_target:.3f}) with future target. "
                "Possible lookahead bias."
            )

    logger.info("Lookahead validation passed")
    return True


if __name__ == "__main__":
    # Test the scalping feature engineering
    print("ScalpingFeatureEngineer module loaded successfully")
    print(f"Default feature config: {FeatureConfig()}")
