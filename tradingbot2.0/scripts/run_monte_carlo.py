#!/usr/bin/env python3
"""
Monte Carlo Simulation CLI

Run Monte Carlo simulation to assess strategy robustness by shuffling
trade order and computing confidence intervals for key metrics.

Usage:
    # From a trades CSV file
    python scripts/run_monte_carlo.py --trades trades.csv

    # With custom parameters
    python scripts/run_monte_carlo.py --trades trades.csv --simulations 5000 --capital 2000

    # Save results to JSON
    python scripts/run_monte_carlo.py --trades trades.csv --output results/monte_carlo.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.monte_carlo import (
    MonteCarloSimulator,
    run_monte_carlo_from_csv,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation for strategy robustness assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with trades CSV
    python scripts/run_monte_carlo.py --trades results/trades.csv

    # Run 5000 simulations with $2000 capital
    python scripts/run_monte_carlo.py --trades trades.csv --simulations 5000 --capital 2000

    # Save results to JSON
    python scripts/run_monte_carlo.py --trades trades.csv --output monte_carlo_results.json

    # Set random seed for reproducibility
    python scripts/run_monte_carlo.py --trades trades.csv --seed 42
        """,
    )

    parser.add_argument(
        "--trades",
        type=str,
        required=True,
        help="Path to trades CSV file (must have 'net_pnl' or 'pnl' column)",
    )

    parser.add_argument(
        "--simulations", "-n",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital (default: 1000)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=95.0,
        help="Confidence level for intervals (default: 95)",
    )

    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.5,
        help="Minimum acceptable Sharpe ratio for robustness check (default: 0.5)",
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.20,
        help="Maximum acceptable drawdown for robustness check (default: 0.20 = 20%%)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate trades file exists
    trades_path = Path(args.trades)
    if not trades_path.exists():
        logger.error(f"Trades file not found: {args.trades}")
        sys.exit(1)

    logger.info(f"Loading trades from: {args.trades}")
    logger.info(f"Running {args.simulations} Monte Carlo simulations...")

    try:
        # Run Monte Carlo simulation
        result = run_monte_carlo_from_csv(
            trades_csv=args.trades,
            n_simulations=args.simulations,
            initial_capital=args.capital,
            output_json=args.output,
            random_seed=args.seed,
        )

        # Print summary
        result.print_summary()

        # Check robustness
        is_robust, failures = result.is_robust(
            min_sharpe=args.min_sharpe,
            max_drawdown=args.max_drawdown,
        )

        # Exit code based on robustness
        if is_robust:
            logger.info("Strategy passes robustness checks")
            sys.exit(0)
        else:
            logger.warning("Strategy fails robustness checks")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
