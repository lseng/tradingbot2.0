"""
Shared fixtures for 5-minute scalping system tests.

Provides realistic test data for feature engineering and data pipeline testing.
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
def sample_1min_data():
    """
    Create sample 1-minute OHLCV data for testing.

    Returns 100 bars of realistic MES data during RTH hours.
    """
    np.random.seed(42)

    # Start at 9:30 AM on a trading day
    start_time = pd.Timestamp("2024-01-02 09:30:00", tz=NY_TZ)

    # Generate 100 1-minute bars
    n_bars = 100
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
def sample_5min_data(sample_1min_data):
    """
    Create sample 5-minute OHLCV data by aggregating 1-minute data.
    """
    return sample_1min_data.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })


@pytest.fixture
def multi_day_1min_data():
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

    # Generate OHLC
    high_pct = np.random.uniform(0.0001, 0.0005, n_bars)
    low_pct = np.random.uniform(0.0001, 0.0005, n_bars)

    df = pd.DataFrame({
        "open": np.roll(close_prices, 1),
        "high": close_prices * (1 + high_pct),
        "low": close_prices * (1 - low_pct),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=pd.DatetimeIndex(dates))

    df.iloc[0, df.columns.get_loc("open")] = base_price

    return df


@pytest.fixture
def full_day_5min_data():
    """
    Create full trading day of 5-minute data (78 bars).

    78 bars = 390 minutes / 5 minutes per bar = full RTH session.
    """
    np.random.seed(42)

    start_time = pd.Timestamp("2024-01-02 09:30:00", tz=NY_TZ)
    n_bars = 78  # Full day
    dates = pd.date_range(start=start_time, periods=n_bars, freq="5min")

    base_price = 4800.0
    returns = np.random.normal(0, 0.0003, n_bars)  # Slightly higher vol for 5min
    close_prices = base_price * np.cumprod(1 + returns)

    high_pct = np.random.uniform(0.0002, 0.001, n_bars)
    low_pct = np.random.uniform(0.0002, 0.001, n_bars)

    df = pd.DataFrame({
        "open": np.roll(close_prices, 1),
        "high": close_prices * (1 + high_pct),
        "low": close_prices * (1 - low_pct),
        "close": close_prices,
        "volume": np.random.randint(500, 5000, n_bars),
    }, index=dates)

    df.iloc[0, df.columns.get_loc("open")] = base_price

    return df


@pytest.fixture
def extended_5min_data():
    """
    Create 300 bars of 5-minute data for feature warmup testing.

    300 bars ensures warmup period (200 bars) is covered.
    """
    np.random.seed(42)

    # Start earlier to have enough bars
    start_time = pd.Timestamp("2024-01-02 09:30:00", tz=NY_TZ)
    n_bars = 300
    dates = pd.date_range(start=start_time, periods=n_bars, freq="5min")

    base_price = 4800.0
    returns = np.random.normal(0, 0.0003, n_bars)
    close_prices = base_price * np.cumprod(1 + returns)

    high_pct = np.random.uniform(0.0002, 0.001, n_bars)
    low_pct = np.random.uniform(0.0002, 0.001, n_bars)

    df = pd.DataFrame({
        "open": np.roll(close_prices, 1),
        "high": close_prices * (1 + high_pct),
        "low": close_prices * (1 - low_pct),
        "close": close_prices,
        "volume": np.random.randint(500, 5000, n_bars),
    }, index=dates)

    df.iloc[0, df.columns.get_loc("open")] = base_price

    return df


@pytest.fixture
def sample_csv_file(tmp_path, multi_day_1min_data):
    """
    Create a temporary CSV file with 1-minute data for data pipeline testing.

    Format: datetime,open,high,low,close,volume (no header)
    """
    csv_path = tmp_path / "test_1min_data.txt"

    # Format datetime without timezone info for CSV
    df = multi_day_1min_data.copy()
    df.index = df.index.tz_localize(None)

    # Save without header (matching expected format)
    df.to_csv(csv_path, header=False, index=True)

    return csv_path
