#!/usr/bin/env python3
"""
Validation Set Backtest Runner for 5-Minute Scalping System

This script implements Phase 3.3 of the implementation plan:
1. Load and prepare data (1-min -> 5-min aggregation)
2. Train LightGBM model on training data (2019-2022)
3. Run backtest on validation data (2023)
4. Analyze results and output metrics

The validation set (2023) is used to:
- Tune confidence thresholds
- Identify failure modes
- Validate the model before touching the test set (2024-2025)

Usage:
    python scripts/run_validation_backtest.py [--confidence 0.60] [--output results/validation]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scalping.data_pipeline import ScalpingDataPipeline, DataConfig
from src.scalping.features import ScalpingFeatureGenerator, create_target_variable
from src.scalping.model import ScalpingModel, ModelConfig
from src.scalping.backtest import (
    ScalpingBacktest, BacktestConfig, BacktestResult,
    analyze_results, export_trades_csv, export_summary_json
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Load 1-minute data, aggregate to 5-minute, filter to RTH.

    Returns:
        DataFrame with 5-minute OHLCV data indexed by NY timezone datetime.
    """
    logger.info(f"Loading data from {data_path}")

    # Configure data pipeline
    config = DataConfig(
        data_path=data_path,
        train_start="2019-05-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2025-12-31",
    )

    pipeline = ScalpingDataPipeline(config=config)

    # Load raw data (the methods are self-contained with caching)
    _ = pipeline.load_raw_data()
    raw_bars = len(pipeline._raw_1min) if pipeline._raw_1min is not None else 0
    logger.info(f"Loaded {raw_bars:,} 1-minute bars")

    # Aggregate to 5-minute bars
    df_5min = pipeline.aggregate()
    logger.info(f"Aggregated to {len(df_5min):,} 5-minute bars")

    # Filter to RTH
    df_rth = pipeline.filter_rth_data()
    logger.info(f"Filtered to {len(df_rth):,} RTH 5-minute bars")

    return df_rth, pipeline


def generate_features(df: pd.DataFrame, horizon_bars: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate 24 scalping features and target variable.

    Returns:
        Tuple of (DataFrame with features, list of feature column names)
    """
    logger.info("Generating features...")

    feature_gen = ScalpingFeatureGenerator()
    df_features = feature_gen.generate_all(df.copy())

    # Create target variable (6-bar = 30 min horizon)
    df_features = create_target_variable(df_features, horizon_bars=horizon_bars, min_move_ticks=2)

    feature_names = feature_gen.get_feature_names()
    logger.info(f"Generated {len(feature_names)} features")

    # Rename target column for consistency (target_6bar -> target)
    target_col = f"target_{horizon_bars}bar"
    df_features = df_features.rename(columns={target_col: "target"})

    # Drop warmup rows (first 200 bars have NaN features)
    df_features = df_features.dropna(subset=feature_names + ['target'])
    logger.info(f"After dropping warmup/NaN: {len(df_features):,} bars")

    return df_features, feature_names


def split_data(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_start: str = "2023-01-01",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets by date.

    Returns:
        Tuple of (train_df, val_df)
    """
    train_mask = df.index <= pd.Timestamp(train_end, tz="America/New_York")
    val_mask = (df.index >= pd.Timestamp(val_start, tz="America/New_York")) & \
               (df.index <= pd.Timestamp(val_end, tz="America/New_York"))

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(f"Training data: {len(train_df):,} bars ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Validation data: {len(val_df):,} bars ({val_df.index.min()} to {val_df.index.max()})")

    return train_df, val_df


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: list[str],
    model_config: ModelConfig = None,
) -> ScalpingModel:
    """
    Train LightGBM model on training data with validation early stopping.

    Returns:
        Trained ScalpingModel
    """
    logger.info("Training LightGBM model...")

    X_train = train_df[feature_names].values
    y_train = train_df['target'].values.astype(int)

    X_val = val_df[feature_names].values
    y_val = val_df['target'].values.astype(int)

    # Default config or custom
    if model_config is None:
        model_config = ModelConfig()

    model = ScalpingModel(config=model_config)
    result = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )

    logger.info(f"Training complete:")
    logger.info(f"  Best iteration: {result.best_iteration}")
    logger.info(f"  Validation AUC: {result.val_auc:.4f}")
    logger.info(f"  Validation Accuracy: {result.val_accuracy:.4f}")

    # Log feature importance
    importance = model.feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features by importance:")
    for name, score in top_features:
        logger.info(f"  {name}: {score:.4f}")

    return model


def run_validation_backtest(
    val_df: pd.DataFrame,
    model: ScalpingModel,
    feature_names: list[str],
    config: BacktestConfig,
) -> BacktestResult:
    """
    Run backtest on validation data.

    Returns:
        BacktestResult with trades and metrics
    """
    logger.info(f"Running backtest on validation data ({len(val_df):,} bars)...")

    backtest = ScalpingBacktest(config)
    result = backtest.run(val_df, model, feature_names)

    return result


def print_results(result: BacktestResult) -> None:
    """Print formatted backtest results."""
    metrics = result.metrics
    analysis = analyze_results(result)

    print("\n" + "=" * 60)
    print("VALIDATION BACKTEST RESULTS")
    print("=" * 60)

    # Key metrics
    print(f"\n--- Performance Summary ---")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {metrics.win_rate_pct:.1f}%")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Net P&L: ${metrics.total_return_dollars:.2f}")
    print(f"Max Drawdown: ${metrics.max_drawdown_dollars:.2f}")

    # Trading days
    trading_days = len(result.daily_pnls)
    trades_per_day = result.total_trades / trading_days if trading_days > 0 else 0
    print(f"\nTrading Days: {trading_days}")
    print(f"Trades per Day: {trades_per_day:.1f}")

    # Win/Loss breakdown
    print(f"\n--- Win/Loss Breakdown ---")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Avg Win: ${metrics.avg_win:.2f}")
    print(f"Avg Loss: ${metrics.avg_loss:.2f}")

    # Exit reason breakdown
    print(f"\n--- Exit Reasons ---")
    for reason, count in result.exits_by_reason.items():
        pct = (count / result.total_trades * 100) if result.total_trades > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # By confidence tier
    print(f"\n--- By Confidence Tier ---")
    for tier, data in analysis.get("by_confidence", {}).items():
        print(f"  {tier.upper()}: {data['count']} trades, "
              f"{data['win_rate']:.1f}% win rate, ${data['total_pnl']:.2f} PnL")

    # By hour
    print(f"\n--- By Hour ---")
    for hour, data in sorted(analysis.get("by_hour", {}).items()):
        print(f"  {hour:02d}:00 - {data['count']} trades, "
              f"{data['win_rate']:.1f}% win rate, ${data['total_pnl']:.2f} PnL")

    # Best/Worst days
    print(f"\n--- Best Days ---")
    for day, pnl in analysis.get("best_days", [])[:3]:
        print(f"  {day}: ${pnl:.2f}")

    print(f"\n--- Worst Days ---")
    for day, pnl in analysis.get("worst_days", [])[:3]:
        print(f"  {day}: ${pnl:.2f}")

    # Success criteria check
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK (Validation Phase)")
    print("=" * 60)

    criteria = [
        ("Win Rate > 52%", metrics.win_rate_pct > 52, f"{metrics.win_rate_pct:.1f}%"),
        ("Profit Factor > 1.0", metrics.profit_factor > 1.0, f"{metrics.profit_factor:.2f}"),
        ("Trades per Day >= 2", trades_per_day >= 2, f"{trades_per_day:.1f}"),
    ]

    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value}")

    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Run validation set backtest for 5M scalping system")
    parser.add_argument(
        "--data",
        default="data/historical/MES/MES_full_1min_continuous_UNadjusted.txt",
        help="Path to 1-minute data file"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Minimum confidence threshold for trades (default: 0.60)"
    )
    parser.add_argument(
        "--profit-target",
        type=int,
        default=6,
        help="Profit target in ticks (default: 6)"
    )
    parser.add_argument(
        "--stop-loss",
        type=int,
        default=8,
        help="Stop loss in ticks (default: 8)"
    )
    parser.add_argument(
        "--output",
        default="results/validation",
        help="Output directory for results (default: results/validation)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model to models/ directory"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and prepare data
    df_rth, pipeline = load_and_prepare_data(args.data)

    # Step 2: Generate features
    df_features, feature_names = generate_features(df_rth)

    # Step 3: Split into train/val
    train_df, val_df = split_data(df_features)

    # Step 4: Train model
    model = train_model(train_df, val_df, feature_names)

    # Step 5: Configure backtest
    backtest_config = BacktestConfig(
        min_confidence=args.confidence,
        profit_target_ticks=args.profit_target,
        stop_loss_ticks=args.stop_loss,
        verbose=args.verbose,
    )

    # Step 6: Run backtest
    result = run_validation_backtest(val_df, model, feature_names, backtest_config)

    # Step 7: Print and save results
    print_results(result)

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = output_dir / f"validation_trades_{timestamp}.csv"
    summary_path = output_dir / f"validation_summary_{timestamp}.json"

    export_trades_csv(result, str(trades_path))
    export_summary_json(result, str(summary_path))

    logger.info(f"Results saved to {output_dir}")

    # Save model if requested
    if args.save_model:
        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"5m_scalper_lgbm_{timestamp}"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

    # Return success/failure based on validation criteria
    metrics = result.metrics
    trading_days = len(result.daily_pnls)
    trades_per_day = result.total_trades / trading_days if trading_days > 0 else 0

    passed_criteria = (
        metrics.win_rate_pct > 52 and
        metrics.profit_factor > 1.0 and
        trades_per_day >= 2
    )

    return 0 if passed_criteria else 1


if __name__ == "__main__":
    sys.exit(main())
