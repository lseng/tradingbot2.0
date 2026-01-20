"""
Enhanced Feature Engineering for Trading Bot.

Adds advanced features beyond basic technical indicators:
1. Volume Profile features (VWAP, volume momentum)
2. Advanced momentum indicators (ROC, Williams %R, CCI)
3. Market regime detection (volatility regime, trend strength)
4. Price action patterns (candle patterns, support/resistance)
5. Time-based features (session, day of week effects)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def add_volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-weighted features."""
    df = df.copy()

    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

    # Volume momentum
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['volume_sma_30'] = df['volume'].rolling(30).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_10'].replace(0, 1)
    df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_30'].replace(0, 1) - 1

    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_sma'] = df['obv'].rolling(20).mean()
    df['obv_signal'] = (df['obv'] - df['obv_sma']) / df['obv_sma'].abs().replace(0, 1)

    # Money Flow Index components
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(14).sum()
    negative_mf = negative_flow.rolling(14).sum()
    mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 1)))
    df['mfi'] = mfi / 100 - 0.5  # Normalize to [-0.5, 0.5]

    return df


def add_advanced_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced momentum indicators."""
    df = df.copy()

    # Rate of Change (ROC) at multiple timeframes
    for period in [5, 10, 20, 60]:
        df[f'roc_{period}'] = df['close'].pct_change(period)

    # Williams %R
    for period in [14, 28]:
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        df[f'williams_r_{period}'] = (highest_high - df['close']) / (highest_high - lowest_low + 1e-8)
        df[f'williams_r_{period}'] = df[f'williams_r_{period}'] - 0.5  # Center at 0

    # Commodity Channel Index (CCI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8) / 200  # Normalize

    # Stochastic RSI
    rsi_period = 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-8) - 0.5

    # MACD histogram slope
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    df['macd_hist_slope'] = histogram.diff(3) / 3

    # Momentum divergence (price vs momentum)
    price_slope = df['close'].diff(10) / df['close'].shift(10)
    momentum_slope = df['roc_10'].diff(5)
    df['momentum_divergence'] = np.sign(price_slope) != np.sign(momentum_slope)
    df['momentum_divergence'] = df['momentum_divergence'].astype(float)

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime detection features."""
    df = df.copy()

    # Volatility regime (using ATR percentile)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df['atr_percentile'] = atr.rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    ) - 0.5

    # Trend strength (ADX)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-8))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-8))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(14).mean() / 100  # Normalize to [0, 1]
    df['trend_direction'] = np.sign(plus_di - minus_di)

    # Volatility expansion/contraction
    bb_std = df['close'].rolling(20).std()
    df['bb_width'] = bb_std / df['close'].rolling(20).mean()
    df['bb_width_change'] = df['bb_width'].pct_change(5)

    # Mean reversion vs trending regime
    returns = df['close'].pct_change()
    autocorr = returns.rolling(60).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )
    df['regime_autocorr'] = autocorr.fillna(0)

    # Hurst exponent approximation (simplified)
    def calc_hurst_approx(series):
        if len(series) < 20:
            return 0.5
        lags = range(2, min(20, len(series) // 2))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        if min(tau) <= 0:
            return 0.5
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] / 2

    df['hurst'] = returns.rolling(100).apply(calc_hurst_approx) - 0.5

    return df


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price action pattern features."""
    df = df.copy()

    # Candle body and wick ratios
    body = df['close'] - df['open']
    full_range = df['high'] - df['low'] + 1e-8
    df['body_ratio'] = body / full_range

    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = upper_wick / full_range
    df['lower_wick_ratio'] = lower_wick / full_range

    # Support/Resistance levels (rolling min/max)
    for period in [20, 60]:
        df[f'resistance_{period}'] = df['high'].rolling(period).max()
        df[f'support_{period}'] = df['low'].rolling(period).min()
        df[f'dist_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
        df[f'dist_to_support_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']

    # Breakout signals
    df['breakout_up'] = (df['close'] > df['resistance_20'].shift(1)).astype(float)
    df['breakout_down'] = (df['close'] < df['support_20'].shift(1)).astype(float)

    # Gap detection
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_filled'] = ((df['low'] <= df['close'].shift(1)) & (df['gap'] > 0) |
                        (df['high'] >= df['close'].shift(1)) & (df['gap'] < 0)).astype(float)

    # Inside/Outside bars
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) &
                        (df['low'] > df['low'].shift(1))).astype(float)
    df['outside_bar'] = ((df['high'] > df['high'].shift(1)) &
                         (df['low'] < df['low'].shift(1))).astype(float)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    # Time of day (cyclical encoding)
    minutes_since_open = (df.index.hour - 9) * 60 + df.index.minute - 30
    minutes_since_open = np.clip(minutes_since_open, 0, 390)
    df['time_sin'] = np.sin(2 * np.pi * minutes_since_open / 390)
    df['time_cos'] = np.cos(2 * np.pi * minutes_since_open / 390)

    # Session periods
    df['opening_30min'] = (minutes_since_open <= 30).astype(float)
    df['closing_30min'] = (minutes_since_open >= 360).astype(float)
    df['lunch_hour'] = ((minutes_since_open >= 150) & (minutes_since_open <= 210)).astype(float)

    # Day of week (cyclical encoding)
    day_of_week = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 5)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 5)

    # Monday/Friday effects
    df['is_monday'] = (day_of_week == 0).astype(float)
    df['is_friday'] = (day_of_week == 4).astype(float)

    return df


def generate_enhanced_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate all enhanced features.

    Returns:
        (df_with_features, list_of_feature_columns)
    """
    df = df.copy()

    # Add all feature groups
    df = add_volume_profile_features(df)
    df = add_advanced_momentum_features(df)
    df = add_regime_features(df)
    df = add_price_action_features(df)
    df = add_time_features(df)

    # Define feature columns (excluding intermediate calculations)
    feature_cols = [
        # Volume profile
        'vwap_deviation', 'volume_ratio', 'volume_trend', 'obv_signal', 'mfi',
        # Advanced momentum
        'roc_5', 'roc_10', 'roc_20', 'roc_60',
        'williams_r_14', 'williams_r_28', 'cci', 'stoch_rsi',
        'macd_hist_slope', 'momentum_divergence',
        # Regime
        'atr_percentile', 'adx', 'trend_direction',
        'bb_width', 'bb_width_change', 'regime_autocorr', 'hurst',
        # Price action
        'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        'dist_to_resistance_20', 'dist_to_support_20',
        'dist_to_resistance_60', 'dist_to_support_60',
        'breakout_up', 'breakout_down', 'gap', 'gap_filled',
        'inside_bar', 'outside_bar',
        # Time
        'time_sin', 'time_cos', 'opening_30min', 'closing_30min', 'lunch_hour',
        'dow_sin', 'dow_cos', 'is_monday', 'is_friday',
    ]

    # Filter to only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols


def combine_with_base_features(
    df: pd.DataFrame,
    base_feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combine enhanced features with base features from data pipeline.

    Returns:
        (df_with_all_features, all_feature_columns)
    """
    df, enhanced_cols = generate_enhanced_features(df)

    # Combine feature lists (avoid duplicates)
    all_features = list(base_feature_cols)
    for col in enhanced_cols:
        if col not in all_features:
            all_features.append(col)

    return df, all_features
