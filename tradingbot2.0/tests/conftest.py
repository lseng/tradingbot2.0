"""
Pytest fixtures for trading bot tests.

This module provides:
- Sample data fixtures for testing
- Mock objects for external dependencies
- Common test utilities
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import tempfile


@pytest.fixture
def sample_ohlcv_data():
    """
    Create sample OHLCV data for testing.

    Returns a DataFrame with 1000 1-second bars of synthetic data.
    """
    np.random.seed(42)

    # Create timestamps for 1000 seconds during RTH
    # Start at 9:30 AM NY on a Monday
    start_time = pd.Timestamp('2024-01-02 09:30:00', tz='America/New_York')
    timestamps = pd.date_range(start=start_time, periods=1000, freq='1s')

    # Create synthetic price data (random walk around 5000)
    base_price = 5000.0
    returns = np.random.randn(1000) * 0.0001  # Small random returns
    close_prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close prices with some noise
    high_prices = close_prices * (1 + np.abs(np.random.randn(1000)) * 0.0002)
    low_prices = close_prices * (1 - np.abs(np.random.randn(1000)) * 0.0002)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Ensure OHLC validity
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Create volume data
    volumes = np.random.randint(10, 1000, 1000)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=timestamps)

    return df


@pytest.fixture
def sample_utc_data():
    """
    Create sample OHLCV data with UTC timestamps for testing timezone conversion.
    """
    np.random.seed(42)

    # Create timestamps in UTC (this is how parquet data comes)
    # 9:30 AM NY = 14:30 UTC during standard time
    start_time = pd.Timestamp('2024-01-02 14:30:00', tz='UTC')
    timestamps = pd.date_range(start=start_time, periods=1000, freq='1s')

    # Create synthetic price data
    base_price = 5000.0
    returns = np.random.randn(1000) * 0.0001
    close_prices = base_price * np.cumprod(1 + returns)

    high_prices = close_prices * (1 + np.abs(np.random.randn(1000)) * 0.0002)
    low_prices = close_prices * (1 - np.abs(np.random.randn(1000)) * 0.0002)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    volumes = np.random.randint(10, 1000, 1000)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=timestamps)

    return df


@pytest.fixture
def sample_parquet_file(sample_utc_data, tmp_path):
    """
    Create a temporary parquet file for testing.
    """
    file_path = tmp_path / "test_data.parquet"

    # Convert to DataFrame with timestamp column (like DataBento format)
    df = sample_utc_data.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    df.to_parquet(file_path, engine='pyarrow', index=False)

    return str(file_path)


@pytest.fixture
def full_day_rth_data():
    """
    Create a full day of RTH data (9:30 AM - 4:00 PM).

    Returns DataFrame with 23400 seconds (6.5 hours) of data.
    """
    np.random.seed(42)

    # 6.5 hours = 23400 seconds
    start_time = pd.Timestamp('2024-01-02 09:30:00', tz='America/New_York')
    timestamps = pd.date_range(start=start_time, periods=23400, freq='1s')

    base_price = 5000.0
    returns = np.random.randn(23400) * 0.0001
    close_prices = base_price * np.cumprod(1 + returns)

    high_prices = close_prices * (1 + np.abs(np.random.randn(23400)) * 0.0002)
    low_prices = close_prices * (1 - np.abs(np.random.randn(23400)) * 0.0002)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    volumes = np.random.randint(10, 1000, 23400)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=timestamps)

    return df


@pytest.fixture
def real_parquet_path():
    """
    Path to the real MES parquet file for integration tests.

    Returns None if file doesn't exist.
    """
    path = Path("data/historical/MES/MES_1s_2years.parquet")
    if path.exists():
        return str(path)
    return None


# Constants for testing
MES_TICK_SIZE = 0.25
MES_TICK_VALUE = 1.25
MES_POINT_VALUE = 5.00
