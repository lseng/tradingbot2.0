"""
DataBento Historical Data Module.

This module provides functionality for downloading, storing, and managing
historical futures data from DataBento for ML model training and backtesting.

Components:
    - DataBentoClient: API client for DataBento data download
    - OHLCVProcessor: Process and validate OHLCV data
    - ParquetStore: Efficient storage with partitioning
    - DataDownloader: High-level download orchestration

Example:
    from src.data import DataBentoClient, download_historical_data

    # Download 3 years of MES 1-minute data
    client = DataBentoClient()
    client.download_ohlcv(
        symbol="MES.FUT",
        schema="ohlcv-1m",
        start="2022-01-01",
        end="2025-01-01",
        output_path="data/historical/MES/"
    )
"""

from .databento_client import (
    DataBentoClient,
    DataBentoConfig,
    DataBentoError,
    AuthenticationError,
    RateLimitError,
    DataQualityError,
    OHLCVSchema,
    DataValidationResult,
    DownloadResult,
    GapInfo,
)

__all__ = [
    # Client
    "DataBentoClient",
    "DataBentoConfig",
    # Errors
    "DataBentoError",
    "AuthenticationError",
    "RateLimitError",
    "DataQualityError",
    # Data types
    "OHLCVSchema",
    "DataValidationResult",
    "DownloadResult",
    "GapInfo",
]
