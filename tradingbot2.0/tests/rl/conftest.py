"""
Shared fixtures for RL trading system tests.

Provides realistic test data for environment and agent testing.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from zoneinfo import ZoneInfo

# Constants
NY_TZ = ZoneInfo("America/New_York")
MES_TICK_SIZE = 0.25


@pytest.fixture
def ny_tz():
    """New York timezone for datetime operations."""
    return NY_TZ


@pytest.fixture
def sample_rl_data():
    """
    Create sample 1-minute OHLCV data for RL testing.

    Returns 500 bars of realistic MES data during RTH hours (enough for lookback).
    """
    np.random.seed(42)

    # Start at 9:30 AM on a trading day
    start_time = pd.Timestamp("2024-01-02 09:30:00", tz=NY_TZ)

    # Generate 500 1-minute bars
    n_bars = 500
    dates = pd.date_range(start=start_time, periods=n_bars, freq="1min")

    # Generate realistic price movements
    base_price = 4800.0
    returns = np.random.normal(0, 0.0001, n_bars)  # ~0.01% volatility per bar
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high_pct = np.random.uniform(0.0001, 0.0005, n_bars)
    low_pct = np.random.uniform(0.0001, 0.0005, n_bars)

    df = pd.DataFrame({
        "open": np.roll(close_prices, 1),  # Open = previous close
        "high": close_prices * (1 + high_pct),
        "low": close_prices * (1 - low_pct),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=dates)

    # Fix first bar open
    df.iloc[0, df.columns.get_loc("open")] = base_price

    return df


@pytest.fixture
def sample_feature_columns():
    """List of feature column names for RL observation."""
    return [
        "return_1", "return_5", "return_10", "return_30",
        "volatility_10", "volatility_30",
        "rsi_14", "macd", "macd_signal",
        "volume_ratio",
    ]


@pytest.fixture
def sample_rl_data_with_features(sample_rl_data, sample_feature_columns):
    """
    Create sample data with basic features for RL testing.

    Adds simple features to the OHLCV data.
    """
    df = sample_rl_data.copy()

    # Add simple features
    df["return_1"] = df["close"].pct_change(1).fillna(0)
    df["return_5"] = df["close"].pct_change(5).fillna(0)
    df["return_10"] = df["close"].pct_change(10).fillna(0)
    df["return_30"] = df["close"].pct_change(30).fillna(0)

    # Volatility (rolling std of returns)
    df["volatility_10"] = df["return_1"].rolling(10).std().fillna(0)
    df["volatility_30"] = df["return_1"].rolling(30).std().fillna(0)

    # RSI (simplified)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # MACD (simplified)
    exp12 = df["close"].ewm(span=12, adjust=False).mean()
    exp26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Volume ratio
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume_ratio"].fillna(1)

    return df


@pytest.fixture
def multi_day_rl_data():
    """
    Create multi-day 1-minute data spanning RTH hours for multiple days.

    Includes 3 full trading days for testing session boundaries.
    """
    np.random.seed(42)

    # Generate 3 trading days of RTH data
    dates = []
    trading_days = ["2024-01-02", "2024-01-03", "2024-01-04"]

    for day in trading_days:
        # RTH hours: 9:30 AM - 4:00 PM = 390 minutes
        day_start = pd.Timestamp(f"{day} 09:30:00", tz=NY_TZ)
        day_dates = pd.date_range(start=day_start, periods=390, freq="1min")
        dates.extend(day_dates)

    n_bars = len(dates)

    # Generate realistic price movements
    base_price = 4800.0
    returns = np.random.normal(0, 0.0001, n_bars)
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high_pct = np.random.uniform(0.0001, 0.0005, n_bars)
    low_pct = np.random.uniform(0.0001, 0.0005, n_bars)

    df = pd.DataFrame({
        "open": np.roll(close_prices, 1),
        "high": close_prices * (1 + high_pct),
        "low": close_prices * (1 - low_pct),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=pd.DatetimeIndex(dates))

    # Fix first bar open
    df.iloc[0, df.columns.get_loc("open")] = base_price

    return df
