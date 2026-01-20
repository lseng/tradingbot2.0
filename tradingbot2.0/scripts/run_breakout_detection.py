#!/usr/bin/env python3
"""
Breakout Detection Strategy Training and Validation Script

This script implements the breakout detection strategy after learning that:
1. Direction prediction failed (AUC 0.51 - no signal)
2. Volatility prediction works (AUC 0.856 - strong signal)
3. Volatility prediction alone doesn't indicate direction

Strategy:
- Use proven volatility prediction to identify when breakout is likely
- Detect consolidation periods using technical features
- Use price position within consolidation range to determine direction
- Only trade when consolidation + high volatility predicted

This is Phase 3.5 of the implementation plan, exploring alternative
strategies after the direct direction prediction approach failed.

Usage:
    python scripts/run_breakout_detection.py [--data path] [--output dir]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scalping.data_pipeline import ScalpingDataPipeline, DataConfig
from src.scalping.features import (
    ScalpingFeatureGenerator,
    create_volatility_target,
)
from src.scalping.model import ScalpingModel, ModelConfig
from src.scalping.breakout import (
    BreakoutFeatureGenerator,
    BreakoutConfig,
    create_breakout_target,
    identify_breakout_setups,
    run_breakout_backtest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load 1-minute data, aggregate to 5-minute, filter to RTH."""
    logger.info(f"Loading data from {data_path}")

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
    _ = pipeline.load_raw_data()
    raw_bars = len(pipeline._raw_1min) if pipeline._raw_1min is not None else 0
    logger.info(f"Loaded {raw_bars:,} 1-minute bars")

    df_5min = pipeline.aggregate()
    logger.info(f"Aggregated to {len(df_5min):,} 5-minute bars")

    df_rth = pipeline.filter_rth_data()
    logger.info(f"Filtered to {len(df_rth):,} RTH 5-minute bars")

    return df_rth


def generate_breakout_features(
    df: pd.DataFrame,
    breakout_config: BreakoutConfig,
) -> Tuple[pd.DataFrame, list]:
    """Generate all features including breakout-specific ones."""
    logger.info("Generating breakout features...")

    feature_gen = BreakoutFeatureGenerator(config=breakout_config)
    df_features = feature_gen.generate_all(df.copy())

    feature_names = feature_gen.get_feature_names()
    logger.info(f"Generated {len(feature_names)} features")

    return df_features, feature_names


def train_volatility_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: list,
) -> Tuple[ScalpingModel, dict]:
    """Train volatility prediction model (known to work well)."""
    logger.info("Training volatility prediction model...")

    # Create volatility target
    train_with_target, train_threshold = create_volatility_target(
        train_df, horizon_bars=6, threshold_percentile=60.0
    )
    val_with_target, _ = create_volatility_target(
        val_df, horizon_bars=6, threshold_percentile=60.0
    )

    # Rename target column
    train_with_target = train_with_target.rename(
        columns={"target_volatility_6bar": "target_vol"}
    )
    val_with_target = val_with_target.rename(
        columns={"target_volatility_6bar": "target_vol"}
    )

    # Only use base features for volatility model
    base_feature_gen = ScalpingFeatureGenerator()
    base_features = base_feature_gen.get_feature_names()
    available_features = [f for f in base_features if f in train_with_target.columns]

    # Drop NaN
    train_clean = train_with_target.dropna(subset=available_features + ["target_vol"])
    val_clean = val_with_target.dropna(subset=available_features + ["target_vol"])

    X_train = train_clean[available_features].values
    y_train = train_clean["target_vol"].values.astype(int)
    X_val = val_clean[available_features].values
    y_val = val_clean["target_vol"].values.astype(int)

    logger.info(f"Training set: {len(X_train):,} samples, {y_train.mean()*100:.1f}% HIGH vol")
    logger.info(f"Validation set: {len(X_val):,} samples, {y_val.mean()*100:.1f}% HIGH vol")

    # Train model
    model_config = ModelConfig(
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_data_in_leaf=100,
    )

    model = ScalpingModel(config=model_config)
    result = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=available_features,
    )

    logger.info(f"Volatility model trained: AUC={result.val_auc:.4f}")

    return model, {
        "auc": result.val_auc,
        "accuracy": result.val_accuracy,
        "threshold": train_threshold,
        "features": available_features,
    }


def get_volatility_predictions(
    model: ScalpingModel,
    df: pd.DataFrame,
    feature_names: list,
) -> np.ndarray:
    """Get volatility predictions for all samples."""
    available_features = [f for f in feature_names if f in df.columns]

    # Handle missing features
    X = df[available_features].fillna(0).values

    predictions = model.predict_proba(X)
    return predictions


def split_data(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_start: str = "2023-01-01",
    val_end: str = "2023-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets."""
    train_mask = df.index <= pd.Timestamp(train_end, tz="America/New_York")
    val_mask = (
        (df.index >= pd.Timestamp(val_start, tz="America/New_York")) &
        (df.index <= pd.Timestamp(val_end, tz="America/New_York"))
    )

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(f"Train: {len(train_df):,} bars ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Val: {len(val_df):,} bars ({val_df.index.min()} to {val_df.index.max()})")

    return train_df, val_df


def analyze_breakout_setups(
    df: pd.DataFrame,
    vol_predictions: np.ndarray,
    vol_threshold: float = 0.60,
    consolidation_threshold: float = 0.60,
) -> dict:
    """Analyze breakout setup quality."""
    results = {}

    # Get breakout targets
    df_with_targets, target_stats = create_breakout_target(
        df,
        horizon_bars=6,
        breakout_threshold_ticks=4.0,
        consolidation_threshold=consolidation_threshold,
    )

    results["target_stats"] = target_stats

    # Filter to valid setups
    is_consolidated = df["consolidation_score"] >= consolidation_threshold
    high_vol_predicted = vol_predictions >= vol_threshold

    # Setup masks
    setup_mask = is_consolidated & high_vol_predicted

    # Count setups
    n_total = len(df)
    n_consolidated = is_consolidated.sum()
    n_high_vol = high_vol_predicted.sum()
    n_setups = setup_mask.sum()

    results["setup_counts"] = {
        "total_bars": n_total,
        "consolidated_bars": int(n_consolidated),
        "high_vol_predicted_bars": int(n_high_vol),
        "valid_setups": int(n_setups),
        "setup_rate": n_setups / n_total * 100 if n_total > 0 else 0,
    }

    # Analyze setups by range position
    if "range_position" in df.columns and n_setups > 0:
        setup_df = df[setup_mask].copy()
        setup_targets = df_with_targets[setup_mask]["target_breakout"].dropna()

        # Near bottom setups (expected up breakout)
        near_bottom = setup_df["range_position"] < 0.35
        near_top = setup_df["range_position"] > 0.65
        middle = ~near_bottom & ~near_top

        if near_bottom.any():
            bottom_targets = df_with_targets[setup_mask & near_bottom]["target_breakout"].dropna()
            results["near_bottom"] = {
                "count": int(near_bottom.sum()),
                "actual_up_breakout": (bottom_targets == 1).mean() * 100 if len(bottom_targets) > 0 else 0,
                "actual_down_breakout": (bottom_targets == 2).mean() * 100 if len(bottom_targets) > 0 else 0,
            }

        if near_top.any():
            top_targets = df_with_targets[setup_mask & near_top]["target_breakout"].dropna()
            results["near_top"] = {
                "count": int(near_top.sum()),
                "actual_up_breakout": (top_targets == 1).mean() * 100 if len(top_targets) > 0 else 0,
                "actual_down_breakout": (top_targets == 2).mean() * 100 if len(top_targets) > 0 else 0,
            }

        if middle.any():
            middle_targets = df_with_targets[setup_mask & middle]["target_breakout"].dropna()
            results["middle"] = {
                "count": int(middle.sum()),
                "actual_up_breakout": (middle_targets == 1).mean() * 100 if len(middle_targets) > 0 else 0,
                "actual_down_breakout": (middle_targets == 2).mean() * 100 if len(middle_targets) > 0 else 0,
            }

    return results


def print_results(results: dict) -> None:
    """Print formatted analysis results."""
    print("\n" + "=" * 70)
    print("BREAKOUT DETECTION STRATEGY ANALYSIS")
    print("=" * 70)

    # Volatility model
    if "volatility_model" in results:
        vm = results["volatility_model"]
        print(f"\n--- Volatility Model (Filter) ---")
        print(f"AUC: {vm['auc']:.4f}")
        print(f"Accuracy: {vm['accuracy']:.4f}")

    # Setup analysis
    if "setup_analysis" in results:
        sa = results["setup_analysis"]

        if "setup_counts" in sa:
            sc = sa["setup_counts"]
            print(f"\n--- Setup Detection ---")
            print(f"Total bars: {sc['total_bars']:,}")
            print(f"Consolidated bars: {sc['consolidated_bars']:,} "
                  f"({sc['consolidated_bars']/sc['total_bars']*100:.1f}%)")
            print(f"High vol predicted: {sc['high_vol_predicted_bars']:,} "
                  f"({sc['high_vol_predicted_bars']/sc['total_bars']*100:.1f}%)")
            print(f"Valid setups: {sc['valid_setups']:,} ({sc['setup_rate']:.1f}%)")

        # Direction accuracy
        print(f"\n--- Direction Prediction Accuracy ---")
        if "near_bottom" in sa:
            nb = sa["near_bottom"]
            print(f"Near range bottom ({nb['count']} setups): "
                  f"UP breakout={nb['actual_up_breakout']:.1f}%, "
                  f"DOWN breakout={nb['actual_down_breakout']:.1f}%")
            expected_up = nb['actual_up_breakout'] > nb['actual_down_breakout']
            print(f"  Strategy predicts UP -> {'CORRECT' if expected_up else 'WRONG'}")

        if "near_top" in sa:
            nt = sa["near_top"]
            print(f"Near range top ({nt['count']} setups): "
                  f"UP breakout={nt['actual_up_breakout']:.1f}%, "
                  f"DOWN breakout={nt['actual_down_breakout']:.1f}%")
            expected_down = nt['actual_down_breakout'] > nt['actual_up_breakout']
            print(f"  Strategy predicts DOWN -> {'CORRECT' if expected_down else 'WRONG'}")

        if "middle" in sa:
            mid = sa["middle"]
            print(f"Middle of range ({mid['count']} setups): "
                  f"UP breakout={mid['actual_up_breakout']:.1f}%, "
                  f"DOWN breakout={mid['actual_down_breakout']:.1f}%")
            print(f"  No trade (skip these)")

    # Backtest results
    if "backtest" in results:
        bt = results["backtest"]
        print(f"\n--- Backtest Results ---")
        print(f"Total trades: {bt['total_trades']}")
        print(f"Total P&L: ${bt['total_pnl']:.2f}")
        print(f"Win rate: {bt['win_rate']:.1f}%")
        print(f"Profit factor: {bt['profit_factor']:.2f}")
        print(f"Avg win: ${bt['avg_win']:.2f}")
        print(f"Avg loss: ${bt['avg_loss']:.2f}")
        print(f"Long trades: {bt['long_trades']}")
        print(f"Short trades: {bt['short_trades']}")

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    success = False

    # Check if direction prediction works
    if "setup_analysis" in results:
        sa = results["setup_analysis"]
        if "near_bottom" in sa and "near_top" in sa:
            nb = sa["near_bottom"]
            nt = sa["near_top"]

            # Check if range position predicts direction correctly
            bottom_edge = nb['actual_up_breakout'] - nb['actual_down_breakout']
            top_edge = nt['actual_down_breakout'] - nt['actual_up_breakout']

            if bottom_edge > 10:
                print(f"[POSITIVE] Near-bottom setups favor UP breakout by {bottom_edge:.1f}pp")
                success = True
            else:
                print(f"[NEGATIVE] Near-bottom setups don't favor UP ({bottom_edge:.1f}pp edge)")

            if top_edge > 10:
                print(f"[POSITIVE] Near-top setups favor DOWN breakout by {top_edge:.1f}pp")
                success = True
            else:
                print(f"[NEGATIVE] Near-top setups don't favor DOWN ({top_edge:.1f}pp edge)")

    # Check backtest results
    if "backtest" in results:
        bt = results["backtest"]
        if bt['total_trades'] > 0:
            if bt['win_rate'] > 55 and bt['profit_factor'] > 1.2:
                print(f"[SUCCESS] Strategy meets minimum criteria (WR>{bt['win_rate']:.0f}%, PF>{bt['profit_factor']:.2f})")
                success = True
            elif bt['win_rate'] > 50 and bt['profit_factor'] > 1.0:
                print(f"[MARGINAL] Strategy shows weak signal (WR={bt['win_rate']:.0f}%, PF={bt['profit_factor']:.2f})")
            else:
                print(f"[FAILED] Strategy not profitable (WR={bt['win_rate']:.0f}%, PF={bt['profit_factor']:.2f})")
        else:
            print(f"[FAILED] No trades generated - check thresholds")

    if not success:
        print("\n[CONCLUSION] Breakout strategy needs refinement or alternative approach")
    else:
        print("\n[CONCLUSION] Breakout strategy shows promise - proceed to Phase 4 analysis")

    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Train and validate breakout detection strategy")
    parser.add_argument(
        "--data",
        default="data/historical/MES/MES_full_1min_continuous_UNadjusted.txt",
        help="Path to 1-minute data file"
    )
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=0.60,
        help="Volatility confidence threshold (default: 0.60)"
    )
    parser.add_argument(
        "--consolidation-threshold",
        type=float,
        default=0.60,
        help="Consolidation score threshold (default: 0.60)"
    )
    parser.add_argument(
        "--output",
        default="results/breakout",
        help="Output directory"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    df_rth = load_and_prepare_data(args.data)

    # Step 2: Generate breakout features
    logger.info("Step 2: Generating features...")
    breakout_config = BreakoutConfig(
        vol_prediction_threshold=args.vol_threshold,
        consolidation_threshold=args.consolidation_threshold,
    )
    df_features, feature_names = generate_breakout_features(df_rth, breakout_config)

    # Step 3: Split data
    logger.info("Step 3: Splitting data...")
    train_df, val_df = split_data(df_features)

    # Step 4: Train volatility model (proven to work)
    logger.info("Step 4: Training volatility model...")
    vol_model, vol_stats = train_volatility_model(train_df, val_df, feature_names)
    all_results["volatility_model"] = vol_stats

    # Step 5: Get volatility predictions for validation set
    logger.info("Step 5: Generating volatility predictions...")
    vol_predictions = get_volatility_predictions(
        vol_model, val_df, vol_stats["features"]
    )

    # Step 6: Analyze breakout setups
    logger.info("Step 6: Analyzing breakout setups...")
    setup_analysis = analyze_breakout_setups(
        val_df,
        vol_predictions,
        vol_threshold=args.vol_threshold,
        consolidation_threshold=args.consolidation_threshold,
    )
    all_results["setup_analysis"] = setup_analysis

    # Step 7: Run backtest
    logger.info("Step 7: Running backtest...")
    trades, backtest_summary = run_breakout_backtest(
        val_df,
        vol_predictions,
        feature_names,
        config=breakout_config,
    )
    all_results["backtest"] = backtest_summary

    # Print results
    print_results(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"breakout_analysis_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save trades
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = output_dir / f"breakout_trades_{timestamp}.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Trades saved to {trades_path}")

    # Return exit code based on results
    if "backtest" in all_results:
        bt = all_results["backtest"]
        if bt["total_trades"] > 0 and bt["profit_factor"] > 1.0:
            logger.info("SUCCESS: Breakout strategy shows positive edge")
            return 0

    logger.warning("Breakout strategy needs refinement")
    return 1


if __name__ == "__main__":
    sys.exit(main())
