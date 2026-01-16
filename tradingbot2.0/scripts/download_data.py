#!/usr/bin/env python3
"""
Download Historical Data from DataBento.

This script provides a command-line interface for downloading historical
futures data from DataBento for ML model training and backtesting.

Features:
    - Initial bulk download (3+ years of historical data)
    - Incremental daily updates (append new data)
    - Gap detection and backfill
    - Data quality validation
    - Multiple timeframes (1s, 1m, 1h, 1d)

Usage:
    # Download 3 years of 1-minute MES data
    python scripts/download_data.py --symbol MES --schema 1m --years 3

    # Download 1-second data (high-frequency)
    python scripts/download_data.py --symbol MES --schema 1s --years 2

    # Update existing data file
    python scripts/download_data.py --update data/historical/MES/MES_FUT_ohlcv_1m.parquet

    # Detect and backfill gaps
    python scripts/download_data.py --backfill data/historical/MES/MES_FUT_ohlcv_1m.parquet

    # Validate data quality
    python scripts/download_data.py --validate data/historical/MES/MES_FUT_ohlcv_1m.parquet

Requirements:
    - DATABENTO_API_KEY environment variable set
    - databento package installed (pip install databento)

Spec Reference:
    specs/databento-historical-data.md
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    DataBentoClient,
    DataBentoConfig,
    DataBentoError,
    AuthenticationError,
    RateLimitError,
    DataValidationResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Schema mapping for convenience
SCHEMA_ALIASES = {
    "1s": "ohlcv-1s",
    "1m": "ohlcv-1m",
    "5m": "ohlcv-5m",
    "15m": "ohlcv-15m",
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
    "trades": "trades",
}


def download_data(
    symbol: str,
    schema: str,
    years: int,
    output_dir: str,
    validate: bool = True,
) -> bool:
    """
    Download historical data from DataBento.

    Args:
        symbol: Contract symbol (e.g., "MES", "ES")
        schema: Data schema (e.g., "1m", "ohlcv-1m")
        years: Number of years of historical data
        output_dir: Output directory for parquet files
        validate: Whether to validate data after download

    Returns:
        True if download succeeded
    """
    # Resolve schema alias
    schema_full = SCHEMA_ALIASES.get(schema.lower(), schema)

    # Calculate date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    logger.info(f"Downloading {symbol} {schema_full} data")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")

    try:
        client = DataBentoClient()
        result = client.download_ohlcv(
            symbol=symbol,
            schema=schema_full,
            start=start_date,
            end=end_date,
            output_dir=output_dir,
            validate=validate,
        )

        if result.success:
            logger.info("=" * 60)
            logger.info("DOWNLOAD SUCCESSFUL")
            logger.info("=" * 60)
            logger.info(f"Output file: {result.output_path}")
            logger.info(f"Row count: {result.row_count:,}")
            logger.info(f"Date range: {result.start_time} to {result.end_time}")
            logger.info(f"Download time: {result.download_seconds:.1f}s")

            if result.validation:
                if result.validation.is_valid:
                    logger.info("Data validation: PASSED")
                else:
                    logger.warning("Data validation: ISSUES FOUND")
                    for issue in result.validation.issues:
                        logger.warning(f"  - {issue}")

            return True
        else:
            logger.error(f"Download failed: {result.error_message}")
            return False

    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("Please set DATABENTO_API_KEY environment variable")
        return False
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        logger.error("Please wait and try again later")
        return False
    except DataBentoError as e:
        logger.error(f"DataBento error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def update_data(file_path: str, symbol: str) -> bool:
    """
    Update existing data file with new data.

    Args:
        file_path: Path to existing parquet file
        symbol: Contract symbol

    Returns:
        True if update succeeded
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    logger.info(f"Updating {file_path}")

    try:
        client = DataBentoClient()
        result = client.download_incremental(
            symbol=symbol,
            existing_file=file_path,
            validate=True,
        )

        if result.success:
            logger.info("=" * 60)
            logger.info("UPDATE SUCCESSFUL")
            logger.info("=" * 60)
            logger.info(f"Total rows: {result.row_count:,}")
            logger.info(f"Date range: {result.start_time} to {result.end_time}")
            return True
        else:
            if result.row_count == 0:
                logger.info("No new data available")
                return True
            logger.error(f"Update failed: {result.error_message}")
            return False

    except Exception as e:
        logger.error(f"Update error: {e}")
        raise


def backfill_data(file_path: str, symbol: str, schema: str = "1m") -> bool:
    """
    Detect and backfill gaps in existing data.

    Args:
        file_path: Path to parquet file
        symbol: Contract symbol
        schema: Data schema

    Returns:
        True if backfill completed (even if no gaps found)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Resolve schema alias
    schema_full = SCHEMA_ALIASES.get(schema.lower(), schema)

    logger.info(f"Checking for gaps in {file_path}")

    try:
        client = DataBentoClient()
        gaps_found, gaps_filled = client.backfill_gaps(
            file_path=file_path,
            symbol=symbol,
            schema=schema_full,
        )

        logger.info("=" * 60)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Gaps found: {gaps_found}")
        logger.info(f"Gaps filled: {gaps_filled}")

        return True

    except Exception as e:
        logger.error(f"Backfill error: {e}")
        raise


def validate_data(file_path: str) -> bool:
    """
    Validate data quality of a parquet file.

    Args:
        file_path: Path to parquet file

    Returns:
        True if validation passed
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    logger.info(f"Validating {file_path}")

    try:
        client = DataBentoClient()
        result = client.validate_data(file_path)

        logger.info("=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Row count: {result.row_count:,}")
        logger.info(f"Date range: {result.start_time} to {result.end_time}")
        logger.info(f"Invalid OHLC rows: {result.invalid_ohlc_rows}")
        logger.info(f"Missing volume rows: {result.missing_volume_rows}")
        logger.info(f"Duplicate rows: {result.duplicate_rows}")
        logger.info(f"Gaps detected: {len(result.gaps)}")

        if result.is_valid:
            logger.info("Status: VALID")
        else:
            logger.warning("Status: ISSUES FOUND")
            for issue in result.issues:
                logger.warning(f"  - {issue}")

        if result.gaps:
            logger.info("\nDetected gaps:")
            for gap in result.gaps[:10]:  # Show first 10 gaps
                logger.info(
                    f"  {gap.start} to {gap.end} ({gap.duration_seconds}s, "
                    f"~{gap.expected_bars} bars, {gap.session})"
                )
            if len(result.gaps) > 10:
                logger.info(f"  ... and {len(result.gaps) - 10} more gaps")

        return result.is_valid

    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise


def show_info(file_path: str) -> bool:
    """
    Show information about a data file.

    Args:
        file_path: Path to parquet file

    Returns:
        True if info retrieved successfully
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    try:
        client = DataBentoClient()
        info = client.get_data_info(file_path)

        if "error" in info:
            logger.error(f"Error: {info['error']}")
            return False

        logger.info("=" * 60)
        logger.info("DATA FILE INFO")
        logger.info("=" * 60)
        logger.info(f"File: {info['file_path']}")
        logger.info(f"Rows: {info['row_count']:,}")
        logger.info(f"Columns: {', '.join(info['columns'])}")
        logger.info(f"Start: {info['start_time']}")
        logger.info(f"End: {info['end_time']}")
        logger.info(f"File size: {info['file_size_mb']:.2f} MB")
        logger.info(f"Memory usage: {info['memory_mb']:.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Info error: {e}")
        raise


def list_symbols():
    """List available symbols."""
    client = DataBentoClient()
    symbols = client.list_available_symbols()

    logger.info("=" * 60)
    logger.info("AVAILABLE SYMBOLS")
    logger.info("=" * 60)
    for symbol in symbols:
        logger.info(f"  {symbol}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download historical futures data from DataBento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 3 years of 1-minute MES data
    python scripts/download_data.py --symbol MES --schema 1m --years 3

    # Download 1-second data
    python scripts/download_data.py --symbol MES --schema 1s --years 2

    # Update existing file
    python scripts/download_data.py --update data/historical/MES/MES_FUT_ohlcv_1m.parquet --symbol MES

    # Validate data
    python scripts/download_data.py --validate data/historical/MES/MES_FUT_ohlcv_1m.parquet

    # Backfill gaps
    python scripts/download_data.py --backfill data/historical/MES/MES_FUT_ohlcv_1m.parquet --symbol MES

Environment:
    DATABENTO_API_KEY    DataBento API key (required)
        """,
    )

    # Download options
    parser.add_argument(
        "--symbol",
        "-s",
        type=str,
        default="MES",
        help="Contract symbol (MES, ES, MNQ, NQ)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="1m",
        choices=list(SCHEMA_ALIASES.keys()),
        help="Data schema/timeframe",
    )
    parser.add_argument(
        "--years",
        "-y",
        type=int,
        default=3,
        help="Years of historical data to download",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/historical/MES",
        help="Output directory for data files",
    )

    # Operations
    parser.add_argument(
        "--update",
        type=str,
        metavar="FILE",
        help="Update existing parquet file with new data",
    )
    parser.add_argument(
        "--backfill",
        type=str,
        metavar="FILE",
        help="Detect and backfill gaps in existing file",
    )
    parser.add_argument(
        "--validate",
        type=str,
        metavar="FILE",
        help="Validate data quality of parquet file",
    )
    parser.add_argument(
        "--info",
        type=str,
        metavar="FILE",
        help="Show information about data file",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List available contract symbols",
    )

    # Options
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation after download",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check for API key
    if not os.getenv("DATABENTO_API_KEY"):
        logger.warning("DATABENTO_API_KEY not set - some operations may fail")

    # Execute requested operation
    try:
        if args.list_symbols:
            list_symbols()
            sys.exit(0)

        if args.info:
            success = show_info(args.info)
            sys.exit(0 if success else 1)

        if args.validate:
            success = validate_data(args.validate)
            sys.exit(0 if success else 1)

        if args.update:
            success = update_data(args.update, args.symbol)
            sys.exit(0 if success else 1)

        if args.backfill:
            success = backfill_data(args.backfill, args.symbol, args.schema)
            sys.exit(0 if success else 1)

        # Default: download new data
        success = download_data(
            symbol=args.symbol,
            schema=args.schema,
            years=args.years,
            output_dir=args.output,
            validate=not args.no_validate,
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
