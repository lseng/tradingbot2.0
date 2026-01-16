#!/usr/bin/env python3
"""
Run Live Trading Entry Point for MES Futures Scalping Bot.

This script provides a command-line interface to start live trading
using the TopstepX API with trained ML models.

IMPORTANT REQUIREMENTS:
- Set TOPSTEPX_API_KEY environment variable with your API key
- Optionally set TOPSTEPX_ACCOUNT_ID for specific account
- Ensure model checkpoint exists at specified path
- Review all risk parameters before starting

Usage:
    # Start paper trading (default)
    python scripts/run_live.py

    # Start with specific model
    python scripts/run_live.py --model models/scalper_v1.pt

    # Start live trading (CAUTION!)
    python scripts/run_live.py --live

    # Custom risk parameters
    python scripts/run_live.py --capital 2000 --max-daily-loss 100

Safety Features:
- Paper trading by default (must explicitly enable live)
- All risk limits enforced
- EOD flatten at 4:30 PM NY guaranteed
- CTRL+C for graceful shutdown
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.live_trader import LiveTrader, TradingConfig, run_live_trading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_current_contract_id() -> str:
    """
    Get the current front-month MES contract ID.

    MES contract expiry months:
    - H = March
    - M = June
    - U = September
    - Z = December

    Returns:
        Contract ID string (e.g., "CON.F.US.MES.H26")
    """
    from datetime import datetime

    now = datetime.now()
    year = now.year % 100  # Last 2 digits

    # Determine current expiry month
    # Contracts expire ~2 weeks before end of month
    # Roll to next contract ~1 week before expiry
    month = now.month

    if month <= 2 or (month == 3 and now.day < 10):
        expiry = f"H{year}"  # March
    elif month <= 5 or (month == 6 and now.day < 10):
        expiry = f"M{year}"  # June
    elif month <= 8 or (month == 9 and now.day < 10):
        expiry = f"U{year}"  # September
    elif month <= 11 or (month == 12 and now.day < 10):
        expiry = f"Z{year}"  # December
    else:
        # Roll to next year's March
        expiry = f"H{year + 1}"

    return f"CON.F.US.MES.{expiry}"


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.

    Returns:
        True if valid, False otherwise
    """
    api_key = os.environ.get("TOPSTEPX_API_KEY")

    if not api_key:
        logger.error("TOPSTEPX_API_KEY environment variable is required")
        logger.info("Set it with: export TOPSTEPX_API_KEY='your-api-key'")
        return False

    return True


def validate_model(model_path: str) -> bool:
    """
    Validate that model file exists.

    Args:
        model_path: Path to model checkpoint

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Train a model first with: python src/ml/train_futures_model.py")
        return False

    return True


def confirm_live_trading() -> bool:
    """
    Require explicit user confirmation for live trading.

    Returns:
        True if user confirms, False otherwise
    """
    print()
    print("=" * 60)
    print("WARNING: LIVE TRADING MODE")
    print("=" * 60)
    print()
    print("You are about to start LIVE trading with REAL money.")
    print("This will execute actual trades on your TopstepX account.")
    print()
    print("Ensure you have:")
    print("  - Tested thoroughly in paper trading")
    print("  - Reviewed all risk parameters")
    print("  - Adequate capital in your account")
    print("  - Monitored a full session in paper mode")
    print()

    try:
        response = input("Type 'I ACCEPT THE RISK' to continue: ")
        return response.strip() == "I ACCEPT THE RISK"
    except (EOFError, KeyboardInterrupt):
        return False


async def run_trading(config: TradingConfig) -> None:
    """
    Run the trading session.

    Args:
        config: Trading configuration
    """
    # Get credentials from environment
    api_key = os.environ.get("TOPSTEPX_API_KEY")
    account_id = os.environ.get("TOPSTEPX_ACCOUNT_ID")

    if not api_key:
        raise ValueError("TOPSTEPX_API_KEY environment variable required")

    trader = LiveTrader(
        config=config,
        api_key=api_key,
        account_id=int(account_id) if account_id else None,
    )

    # Handle shutdown signals
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Shutdown signal received - initiating graceful shutdown")
        asyncio.create_task(trader.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run trader
    await trader.start()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MES Futures Live Trading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (default is paper trading)',
    )

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='models/scalper_v1.pt',
        help='Path to trained model checkpoint',
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='models/feature_scaler.pkl',
        help='Path to feature scaler',
    )

    # Contract
    parser.add_argument(
        '--contract',
        type=str,
        default=None,
        help='Contract ID (auto-detects front-month if not specified)',
    )

    # Risk parameters
    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='Starting capital',
    )
    parser.add_argument(
        '--max-daily-loss',
        type=float,
        default=50.0,
        help='Maximum daily loss limit',
    )
    parser.add_argument(
        '--max-trade-risk',
        type=float,
        default=25.0,
        help='Maximum risk per trade',
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.65,
        help='Minimum model confidence for trading',
    )

    # Session times
    parser.add_argument(
        '--session-start',
        type=str,
        default='09:30',
        help='Session start time (HH:MM NY time)',
    )
    parser.add_argument(
        '--session-end',
        type=str,
        default='16:00',
        help='Session end time (HH:MM NY time)',
    )
    parser.add_argument(
        '--flatten-time',
        type=str,
        default='16:25',
        help='EOD flatten start time (HH:MM NY time)',
    )

    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )

    return parser.parse_args()


def parse_time(time_str: str) -> time:
    """Parse time string (HH:MM) to time object."""
    parts = time_str.split(':')
    return time(int(parts[0]), int(parts[1]))


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    logger.info("=" * 60)
    logger.info("MES FUTURES SCALPING BOT")
    logger.info("=" * 60)

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Validate model (warn but don't exit if missing)
    if not validate_model(args.model):
        logger.warning("Model not found - trading will use fallback signals")

    # Confirm live trading if enabled
    paper_trading = not args.live
    if not paper_trading:
        if not confirm_live_trading():
            logger.info("Live trading cancelled by user")
            sys.exit(0)

    # Get contract ID
    contract_id = args.contract or get_current_contract_id()
    logger.info(f"Contract: {contract_id}")

    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build configuration
    config = TradingConfig(
        contract_id=contract_id,
        model_path=args.model,
        scaler_path=args.scaler,
        min_confidence=args.min_confidence,
        starting_capital=args.capital,
        max_daily_loss=args.max_daily_loss,
        max_per_trade_risk=args.max_trade_risk,
        session_start=parse_time(args.session_start),
        session_end=parse_time(args.session_end),
        flatten_time=parse_time(args.flatten_time),
        log_dir=args.log_dir,
        paper_trading=paper_trading,
    )

    # Log configuration
    mode = "PAPER" if paper_trading else "LIVE"
    logger.info(f"Mode: {mode}")
    logger.info(f"Capital: ${config.starting_capital:.2f}")
    logger.info(f"Max daily loss: ${config.max_daily_loss:.2f}")
    logger.info(f"Max trade risk: ${config.max_per_trade_risk:.2f}")
    logger.info(f"Min confidence: {config.min_confidence:.0%}")
    logger.info(f"Session: {config.session_start.strftime('%H:%M')} - {config.session_end.strftime('%H:%M')} NY")
    logger.info(f"Flatten time: {config.flatten_time.strftime('%H:%M')} NY")
    logger.info("=" * 60)

    if paper_trading:
        logger.info("PAPER TRADING MODE - No real trades will be executed")
    else:
        logger.warning("LIVE TRADING MODE - Real trades will be executed!")

    logger.info("Press CTRL+C to stop gracefully")
    logger.info("=" * 60)

    # Run trading
    try:
        asyncio.run(run_trading(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Trading error: {e}")
        raise


if __name__ == '__main__':
    main()
