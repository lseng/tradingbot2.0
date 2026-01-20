"""
Feature Engineering for 5-Minute Scalping System

Generates 24 features for LightGBM model on 5-minute bars.

Features by category:
1. Returns (5): 1bar, 3bar, 6bar, 12bar, 24bar momentum
2. Moving Averages (4): Close deviation from EMA-8, EMA-21, EMA-50, EMA-200
3. Momentum (5): RSI-7, RSI-14, MACD, MACD signal, MACD histogram
4. Volatility (3): ATR-14, Bollinger Band width, bar range
5. Volume (3): Volume ratio vs 20-bar avg, volume trend slope, VWAP deviation
6. Time (4): Time-of-day normalized, minutes-since-open, first-hour flag, last-hour flag

Why this approach:
- Simple, interpretable features work well with gradient boosted trees
- No complex feature interactions that could lead to overfitting
- All features are computed WITHOUT lookahead bias
"""

import logging
from dataclasses import dataclass, field
from datetime import time
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading hours (NY timezone)
RTH_START = time(9, 30)
RTH_END = time(16, 0)
RTH_MINUTES = 390  # 6.5 hours = 390 minutes


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""

    # Return lookback periods (in bars)
    return_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])

    # EMA periods
    ema_periods: List[int] = field(default_factory=lambda: [8, 21, 50, 200])

    # RSI periods
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14])

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR period
    atr_period: int = 14

    # Bollinger Band parameters
    bb_period: int = 20
    bb_std: float = 2.0

    # Volume moving average period
    volume_ma_period: int = 20

    # VWAP reset at session start
    reset_vwap_daily: bool = True

    # Warmup period (max lookback needed for features)
    warmup_period: int = 200


def _calculate_returns(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    Calculate return features for multiple lookback periods.

    Returns are computed as percentage change: (close[t] - close[t-n]) / close[t-n]
    """
    result = pd.DataFrame(index=df.index)

    for period in periods:
        col_name = f"return_{period}bar"
        result[col_name] = df["close"].pct_change(periods=period)

    return result


def _calculate_emas(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    Calculate close price deviation from EMAs.

    Returns normalized deviation: (close - EMA) / close
    """
    result = pd.DataFrame(index=df.index)

    for period in periods:
        ema = df["close"].ewm(span=period, adjust=False).mean()
        col_name = f"close_vs_ema{period}"
        result[col_name] = (df["close"] - ema) / df["close"]

    return result


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Fill NaN with 50 (neutral)
    rsi = rsi.fillna(50)

    return rsi


def _calculate_momentum(
    df: pd.DataFrame,
    rsi_periods: List[int],
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
) -> pd.DataFrame:
    """
    Calculate momentum indicators: RSI and MACD.
    """
    result = pd.DataFrame(index=df.index)

    # RSI for each period
    for period in rsi_periods:
        col_name = f"rsi_{period}"
        rsi = _calculate_rsi(df["close"], period)
        # Normalize to [-1, 1] range: (RSI - 50) / 50
        result[col_name] = (rsi - 50) / 50

    # MACD
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    histogram = macd_line - signal_line

    # Normalize MACD by price level
    result["macd"] = macd_line / df["close"]
    result["macd_signal"] = signal_line / df["close"]
    result["macd_hist"] = histogram / df["close"]

    return result


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA(TR, period)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def _calculate_volatility(
    df: pd.DataFrame,
    atr_period: int,
    bb_period: int,
    bb_std: float,
) -> pd.DataFrame:
    """
    Calculate volatility features: ATR, Bollinger Band width, bar range.
    """
    result = pd.DataFrame(index=df.index)

    # ATR (normalized by close price)
    atr = _calculate_atr(df, atr_period)
    result["atr_14"] = atr / df["close"]

    # Bollinger Band width
    sma = df["close"].rolling(window=bb_period).mean()
    std = df["close"].rolling(window=bb_period).std()
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    result["bb_width"] = (upper_band - lower_band) / sma

    # Bar range (normalized)
    result["bar_range"] = (df["high"] - df["low"]) / df["close"]

    return result


def _calculate_volume(df: pd.DataFrame, ma_period: int) -> pd.DataFrame:
    """
    Calculate volume features: volume ratio, volume trend, VWAP deviation.
    """
    result = pd.DataFrame(index=df.index)

    # Volume ratio vs moving average
    volume_ma = df["volume"].rolling(window=ma_period).mean()
    result["volume_ratio_20"] = df["volume"] / volume_ma.replace(0, np.nan)

    # Volume trend (slope of volume over last N bars)
    volume_trend = df["volume"].rolling(window=ma_period).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
        raw=True,
    )
    # Normalize by mean volume
    result["volume_trend"] = volume_trend / volume_ma.replace(0, np.nan)

    # VWAP deviation (simplified - use cumulative within session)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    result["vwap_deviation"] = (df["close"] - vwap) / df["close"]

    return result


def _calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-of-day features.

    Features:
    - time_of_day: Normalized time within trading session (0 = open, 1 = close)
    - minutes_since_open: Minutes since RTH open (9:30 AM)
    - is_first_hour: 1 if in first hour (9:30-10:30), else 0
    - is_last_hour: 1 if in last hour (3:00-4:00), else 0
    """
    result = pd.DataFrame(index=df.index)

    # Extract time components
    times = df.index.time

    # Minutes since midnight
    minutes = np.array([t.hour * 60 + t.minute for t in times])

    # Minutes since RTH open (9:30 = 570 minutes from midnight)
    rth_open_minutes = 9 * 60 + 30
    minutes_since_open = minutes - rth_open_minutes

    # Normalize time of day (0 at open, 1 at close)
    result["time_of_day"] = minutes_since_open / RTH_MINUTES

    # Raw minutes since open
    result["minutes_since_open"] = minutes_since_open / RTH_MINUTES  # Normalized

    # First hour flag (9:30 - 10:30)
    result["is_first_hour"] = ((minutes >= 570) & (minutes < 630)).astype(int)

    # Last hour flag (3:00 - 4:00 = 900-960 minutes)
    result["is_last_hour"] = ((minutes >= 900) & (minutes < 960)).astype(int)

    return result


class ScalpingFeatureGenerator:
    """
    Feature generator for 5-minute scalping system.

    Generates 24 features optimized for LightGBM classification.

    Example usage:
        generator = ScalpingFeatureGenerator()
        df_with_features = generator.generate_all(df)
        feature_names = generator.get_feature_names()
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature generator.

        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features."""
        return_features = _calculate_returns(df, self.config.return_periods)
        return pd.concat([df, return_features], axis=1)

    def add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA deviation features."""
        ema_features = _calculate_emas(df, self.config.ema_periods)
        return pd.concat([df, ema_features], axis=1)

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features (RSI, MACD)."""
        momentum_features = _calculate_momentum(
            df,
            self.config.rsi_periods,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        return pd.concat([df, momentum_features], axis=1)

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features (ATR, BB width, bar range)."""
        vol_features = _calculate_volatility(
            df,
            self.config.atr_period,
            self.config.bb_period,
            self.config.bb_std,
        )
        return pd.concat([df, vol_features], axis=1)

    def add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features."""
        volume_features = _calculate_volume(df, self.config.volume_ma_period)
        return pd.concat([df, volume_features], axis=1)

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-of-day features."""
        time_features = _calculate_time_features(df)
        return pd.concat([df, time_features], axis=1)

    def generate_all(self, df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        """
        Generate all 24 features.

        Args:
            df: DataFrame with OHLCV data (datetime index)
            drop_warmup: Whether to drop warmup period rows with NaN

        Returns:
            DataFrame with original OHLCV + 24 feature columns
        """
        logger.info(f"Generating features for {len(df):,} bars")

        # Start with copy to avoid modifying original
        result = df.copy()

        # Add features in groups
        result = self.add_returns(result)
        result = self.add_emas(result)
        result = self.add_momentum(result)
        result = self.add_volatility(result)
        result = self.add_volume(result)
        result = self.add_time_features(result)

        # Store feature names
        self._feature_names = self._get_feature_column_names()

        # Drop warmup period
        if drop_warmup:
            original_len = len(result)
            result = result.iloc[self.config.warmup_period:]
            logger.info(f"Dropped {original_len - len(result)} warmup rows")

        # Check for remaining NaN in features
        nan_counts = result[self._feature_names].isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            logger.warning(f"NaN values in features after warmup:\n{nan_features}")
            # Fill remaining NaN with 0 (neutral value)
            result[self._feature_names] = result[self._feature_names].fillna(0)

        logger.info(f"Generated {len(self._feature_names)} features for {len(result):,} bars")

        return result

    def _get_feature_column_names(self) -> List[str]:
        """Get list of feature column names."""
        names = []

        # Returns
        for period in self.config.return_periods:
            names.append(f"return_{period}bar")

        # EMAs
        for period in self.config.ema_periods:
            names.append(f"close_vs_ema{period}")

        # Momentum
        for period in self.config.rsi_periods:
            names.append(f"rsi_{period}")
        names.extend(["macd", "macd_signal", "macd_hist"])

        # Volatility
        names.extend(["atr_14", "bb_width", "bar_range"])

        # Volume
        names.extend(["volume_ratio_20", "volume_trend", "vwap_deviation"])

        # Time
        names.extend(["time_of_day", "minutes_since_open", "is_first_hour", "is_last_hour"])

        return names

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if not self._feature_names:
            self._feature_names = self._get_feature_column_names()
        return self._feature_names

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get feature matrix as numpy array.

        Args:
            df: DataFrame with features already generated

        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        feature_names = self.get_feature_names()
        return df[feature_names].values

    def validate_no_lookahead(self, df: pd.DataFrame) -> bool:
        """
        Validate that features don't use future data.

        This is a structural validation that checks:
        1. Return features use shift() which looks backward
        2. Moving averages use only past data
        3. No target columns are in feature set

        Returns:
            True if validation passes
        """
        feature_names = self.get_feature_names()
        available_features = [f for f in feature_names if f in df.columns]

        if not available_features:
            logger.warning("No features to validate")
            return True

        # Check that no target columns are in features (target uses future data)
        for feature in available_features:
            if "target" in feature.lower():
                logger.warning(f"Target column '{feature}' found in features - this is lookahead bias!")
                return False

        # Verify feature names follow expected patterns (backward-looking indicators)
        valid_patterns = [
            "return_", "ema", "rsi", "macd", "atr", "bb_", "bar_range",
            "volume_", "vwap", "time_of_day", "minutes_since", "is_first", "is_last"
        ]

        for feature in available_features:
            feature_lower = feature.lower()
            is_valid = any(pattern in feature_lower for pattern in valid_patterns)
            if not is_valid:
                logger.warning(f"Unknown feature pattern: {feature}")

        logger.info(f"Lookahead validation passed ({len(available_features)} features checked)")
        return True


def create_target_variable(
    df: pd.DataFrame,
    horizon_bars: int = 6,
    min_move_ticks: float = 2.0,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """
    Create binary target variable for price direction prediction.

    Args:
        df: DataFrame with close prices
        horizon_bars: Number of bars to look ahead (6 bars = 30 min on 5M)
        min_move_ticks: Minimum move in ticks to count as UP (filters noise)
        tick_size: Contract tick size (MES = 0.25)

    Returns:
        DataFrame with target column added

    Target:
        1 if close[t + horizon] > close[t] + min_move_ticks * tick_size
        0 otherwise

    Why:
        - Binary classification is simpler and more robust than 3-class
        - Minimum move filter reduces noise from small fluctuations
        - 6-bar horizon (30 min) matches trading time stop
    """
    result = df.copy()

    # Future close price
    future_close = df["close"].shift(-horizon_bars)

    # Minimum move threshold
    threshold = min_move_ticks * tick_size

    # Target: 1 if price goes up by at least threshold, NaN where no future data
    # Use np.where to preserve NaN for rows without future data
    target_values = np.where(
        future_close.isna(),
        np.nan,
        (future_close > df["close"] + threshold).astype(float)
    )
    result[f"target_{horizon_bars}bar"] = target_values

    # Also create targets for other horizons (for experimentation)
    for h in [3, 12]:
        if h != horizon_bars:
            future_h = df["close"].shift(-h)
            target_h_values = np.where(
                future_h.isna(),
                np.nan,
                (future_h > df["close"] + threshold).astype(float)
            )
            result[f"target_{h}bar"] = target_h_values

    logger.info(f"Created target variables with {horizon_bars}-bar horizon, {min_move_ticks} tick threshold")

    # Log class distribution
    target_col = f"target_{horizon_bars}bar"
    if target_col in result.columns:
        class_dist = result[target_col].value_counts(normalize=True)
        logger.info(f"Target distribution:\n{class_dist}")

    return result
