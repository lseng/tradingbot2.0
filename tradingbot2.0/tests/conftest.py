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


@pytest.fixture
def sample_csv_data(tmp_path):
    """
    Create a temporary CSV file with 1-minute OHLCV data for testing FuturesDataLoader.
    """
    np.random.seed(42)

    # Create 1000 rows of 1-minute data
    start_time = datetime(2024, 1, 2, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(1000)]

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

    # Write to CSV (no header, as expected by FuturesDataLoader)
    file_path = tmp_path / "test_data.csv"
    with open(file_path, 'w') as f:
        for i in range(1000):
            ts = timestamps[i].strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{ts},{open_prices[i]:.2f},{high_prices[i]:.2f},{low_prices[i]:.2f},{close_prices[i]:.2f},{volumes[i]}\n")

    return str(file_path)


@pytest.fixture
def sample_csv_with_issues(tmp_path):
    """
    Create a CSV file with data quality issues for testing validation.
    """
    np.random.seed(42)

    start_time = datetime(2024, 1, 2, 9, 30)

    # Write problematic data: missing values, negative prices, invalid OHLC
    file_path = tmp_path / "test_data_issues.csv"
    with open(file_path, 'w') as f:
        # Good row
        f.write("2024-01-02 09:30:00,5000.00,5001.00,4999.00,5000.50,100\n")
        # Row with negative price (will be filtered)
        f.write("2024-01-02 09:31:00,-5000.00,5001.00,4999.00,5000.50,100\n")
        # Row with invalid OHLC (high < low)
        f.write("2024-01-02 09:32:00,5000.00,4999.00,5001.00,5000.50,100\n")
        # Row with negative volume
        f.write("2024-01-02 09:33:00,5000.00,5001.00,4999.00,5000.50,-100\n")
        # Good row
        f.write("2024-01-02 09:34:00,5000.50,5002.00,4998.00,5001.00,150\n")

    return str(file_path)


@pytest.fixture
def sample_daily_ohlcv():
    """
    Create sample daily OHLCV data for feature engineering testing.
    """
    np.random.seed(42)

    # 200 trading days
    dates = pd.date_range(start='2024-01-02', periods=200, freq='B')

    base_price = 5000.0
    returns = np.random.randn(200) * 0.01  # 1% daily volatility
    close_prices = base_price * np.cumprod(1 + returns)

    high_prices = close_prices * (1 + np.abs(np.random.randn(200)) * 0.005)
    low_prices = close_prices * (1 - np.abs(np.random.randn(200)) * 0.005)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    volumes = np.random.randint(10000, 100000, 200)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def sample_features_and_targets():
    """
    Create sample feature matrix and targets for training testing.
    """
    np.random.seed(42)

    n_samples = 500
    n_features = 40

    # Random features
    X = np.random.randn(n_samples, n_features)

    # 3-class targets (DOWN=0, FLAT=1, UP=2) with realistic distribution
    probs = np.random.rand(n_samples)
    y = np.where(probs < 0.2, 0,  # DOWN ~20%
         np.where(probs < 0.8, 1,  # FLAT ~60%
                  2))  # UP ~20%

    return X, y


@pytest.fixture
def sample_binary_targets():
    """
    Create sample feature matrix and binary targets for training testing.
    """
    np.random.seed(42)

    n_samples = 500
    n_features = 40

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y


@pytest.fixture
def small_neural_net():
    """
    Create a small neural network for testing.
    """
    import torch.nn as nn

    class SmallNet(nn.Module):
        def __init__(self, input_dim=40, num_classes=3):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    return SmallNet()


@pytest.fixture
def mock_topstepx_client():
    """
    Create a mock TopstepX API client for testing live trading.
    """
    from unittest.mock import MagicMock, AsyncMock

    client = MagicMock()
    client.authenticate = AsyncMock(return_value=True)
    client.is_authenticated = True
    client.get_access_token = MagicMock(return_value="test_token")

    return client


@pytest.fixture
def mock_rest_client():
    """
    Create a mock REST client for testing order execution.
    """
    from unittest.mock import MagicMock, AsyncMock

    client = MagicMock()
    client.place_order = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123'
    })
    client.cancel_order = AsyncMock(return_value={'success': True})
    client.get_positions = AsyncMock(return_value=[])
    client.get_account_info = AsyncMock(return_value={
        'balance': 1000.0,
        'available_margin': 900.0
    })

    return client


@pytest.fixture
def mock_ws_client():
    """
    Create a mock WebSocket client for testing.
    """
    from unittest.mock import MagicMock, AsyncMock

    client = MagicMock()
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.subscribe_quotes = AsyncMock()
    client.set_quote_callback = MagicMock()

    return client


@pytest.fixture
def mock_position_manager():
    """
    Create a mock position manager for testing.
    """
    from unittest.mock import MagicMock

    manager = MagicMock()
    manager.has_position = MagicMock(return_value=False)
    manager.get_position = MagicMock(return_value=None)
    manager.update_position = MagicMock()
    manager.close_position = MagicMock()

    return manager
