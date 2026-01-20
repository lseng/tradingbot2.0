#!/usr/bin/env python3
"""
Mean-Reversion Strategy Runner

This script implements and backtests a mean-reversion strategy that:
1. Uses the proven volatility model (AUC 0.855) to identify LOW volatility periods
2. Trades against RSI extremes during quiet periods (fade oversold/overbought)
3. Uses tighter stops and quicker exits than the breakout strategy

Why this approach:
- Direction prediction FAILED (AUC 0.51, Win rate 38.8%)
- Breakout detection FAILED (Win rate 39.1%, PF 0.50)
- BUT volatility prediction WORKS (AUC 0.855)
- During low-volatility periods, prices tend to mean-revert

Strategy Rules:
- Entry: LOW vol predicted + RSI extreme + price extended from EMA
- Direction: RSI < 30 = buy (oversold), RSI > 70 = sell (overbought)
- Profit target: 4 ticks ($5.00)
- Stop loss: 4 ticks ($5.00)
- Time stop: 3 bars (15 minutes)
- Avoid first/last hour (higher natural volatility)

Usage:
    python scripts/run_mean_reversion.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scalping.data_pipeline import (
    load_1min_data,
    aggregate_to_5min,
    filter_rth,
    create_temporal_splits,
)
from src.scalping.features import (
    ScalpingFeatureGenerator,
    create_volatility_target,
)
from src.scalping.model import ScalpingModel
from src.scalping.mean_reversion import (
    MeanReversionConfig,
    add_mean_reversion_features,
    identify_mean_reversion_setups,
    run_mean_reversion_backtest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def main():
    """Run mean-reversion strategy analysis and backtest."""

    data_path = "data/historical/MES/MES_full_1min_continuous_UNadjusted.txt"

    print("\n" + "=" * 70)
    print("MEAN-REVERSION STRATEGY ANALYSIS")
    print("=" * 70)

    # Step 1: Load and prepare data
    logger.info("Step 1: Loading data...")
    logger.info(f"Loading data from {data_path}")

    df_1min = load_1min_data(data_path)
    logger.info(f"Loaded {len(df_1min):,} 1-minute bars")

    df_5min = aggregate_to_5min(df_1min)
    logger.info(f"Aggregated to {len(df_5min):,} 5-minute bars")

    df_rth = filter_rth(df_5min)
    logger.info(f"Filtered to {len(df_rth):,} RTH 5-minute bars")

    # Step 2: Generate features
    logger.info("Step 2: Generating features...")
    feature_gen = ScalpingFeatureGenerator()
    df_features = feature_gen.generate_all(df_rth, drop_warmup=True)
    feature_names = feature_gen.get_feature_names()
    logger.info(f"Generated {len(feature_names)} features")

    # Add mean-reversion specific features
    config = MeanReversionConfig()
    df_features = add_mean_reversion_features(df_features, config)
    logger.info("Added mean-reversion features")

    # Step 3: Split data
    logger.info("Step 3: Splitting data...")
    train_df, val_df, test_df = create_temporal_splits(df_features)
    logger.info(f"Train: {len(train_df):,} bars ({train_df.index[0]} to {train_df.index[-1]})")
    logger.info(f"Val: {len(val_df):,} bars ({val_df.index[0]} to {val_df.index[-1]})")

    # Step 4: Train volatility model (same as breakout - we reuse this)
    logger.info("Step 4: Training volatility model...")
    logger.info("Training volatility prediction model (same as breakout)...")

    # Create volatility target
    train_with_vol, vol_threshold_train = create_volatility_target(train_df)
    val_with_vol, vol_threshold_val = create_volatility_target(val_df)

    X_train = train_with_vol[feature_names].values
    y_train_vol = train_with_vol["target_volatility_6bar"].values

    X_val = val_with_vol[feature_names].values
    y_val_vol = val_with_vol["target_volatility_6bar"].values

    # Remove NaN
    train_mask = ~np.isnan(y_train_vol)
    val_mask = ~np.isnan(y_val_vol)

    X_train = X_train[train_mask]
    y_train_vol = y_train_vol[train_mask]
    X_val = X_val[val_mask]
    y_val_vol = y_val_vol[val_mask]

    logger.info(f"Training set: {len(X_train):,} samples, {y_train_vol.mean()*100:.1f}% HIGH vol")
    logger.info(f"Validation set: {len(X_val):,} samples, {y_val_vol.mean()*100:.1f}% HIGH vol")

    # Train volatility model
    vol_model = ScalpingModel()
    vol_model.train(X_train, y_train_vol, X_val, y_val_vol)

    vol_auc = vol_model._training_result.val_auc if vol_model._training_result else 0.0
    logger.info(f"Volatility model trained: AUC={vol_auc:.4f}")

    # Step 5: Generate volatility predictions on validation set
    logger.info("Step 5: Generating volatility predictions...")
    val_features = val_df[feature_names].values
    vol_predictions = vol_model.predict_proba(val_features)

    # Step 6: Analyze mean-reversion setups
    logger.info("Step 6: Analyzing mean-reversion setups...")

    # Add RSI raw values if not present
    from src.scalping.features import _calculate_rsi
    rsi_col = f"rsi_{config.rsi_period}_raw"
    if rsi_col not in val_df.columns:
        val_df[rsi_col] = _calculate_rsi(val_df["close"], config.rsi_period)

    # Identify setups
    val_with_setups = identify_mean_reversion_setups(val_df, vol_predictions, config)

    # Analyze setup quality
    setup_analysis = {
        "total_bars": len(val_df),
        "low_vol_bars": (vol_predictions < config.low_vol_threshold).sum(),
        "low_vol_pct": (vol_predictions < config.low_vol_threshold).mean() * 100,
        "oversold_bars": val_with_setups["is_oversold"].sum(),
        "overbought_bars": val_with_setups["is_overbought"].sum(),
        "long_setups": val_with_setups["is_long_setup"].sum(),
        "short_setups": val_with_setups["is_short_setup"].sum(),
    }

    # Step 7: Run backtest
    logger.info("Step 7: Running backtest...")
    trades, summary = run_mean_reversion_backtest(val_df, vol_predictions, config)

    # Save results
    results_dir = Path("results/mean_reversion")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "volatility_model": {
            "auc": vol_auc,
            "accuracy": vol_model.val_accuracy if hasattr(vol_model, 'val_accuracy') else None,
            "threshold": float(vol_threshold_val),
        },
        "config": {
            "low_vol_threshold": config.low_vol_threshold,
            "rsi_oversold": config.rsi_oversold,
            "rsi_overbought": config.rsi_overbought,
            "min_ema_deviation": config.min_ema_deviation,
            "profit_target_ticks": config.profit_target_ticks,
            "stop_loss_ticks": config.stop_loss_ticks,
            "time_stop_bars": config.time_stop_bars,
        },
        "setup_analysis": setup_analysis,
        "backtest": summary,
    }

    results = convert_numpy_types(results)

    results_file = results_dir / f"mean_reversion_analysis_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save trades
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_file = results_dir / f"mean_reversion_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Trades saved to {trades_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("MEAN-REVERSION STRATEGY ANALYSIS")
    print("=" * 70)

    print(f"\n--- Volatility Model (Filter) ---")
    print(f"AUC: {vol_auc:.4f}")
    print(f"Low vol threshold: {config.low_vol_threshold}")

    print(f"\n--- Setup Detection ---")
    print(f"Total bars: {setup_analysis['total_bars']:,}")
    print(f"Low vol periods: {setup_analysis['low_vol_bars']:,} ({setup_analysis['low_vol_pct']:.1f}%)")
    print(f"Oversold readings: {setup_analysis['oversold_bars']:,}")
    print(f"Overbought readings: {setup_analysis['overbought_bars']:,}")
    print(f"Long setups (oversold + low vol): {setup_analysis['long_setups']:,}")
    print(f"Short setups (overbought + low vol): {setup_analysis['short_setups']:,}")

    print(f"\n--- Backtest Results ---")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"Win rate: {summary['win_rate']:.1f}%")
    print(f"Profit factor: {summary['profit_factor']:.2f}")
    print(f"Avg win: ${summary['avg_win']:.2f}")
    print(f"Avg loss: ${summary['avg_loss']:.2f}")
    print(f"Long trades: {summary['long_trades']}")
    print(f"Short trades: {summary['short_trades']}")

    if summary.get("exit_reasons"):
        print(f"\n--- Exit Reasons ---")
        for reason, count in summary["exit_reasons"].items():
            pct = count / summary["total_trades"] * 100 if summary["total_trades"] > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    success_criteria = {
        "win_rate": summary["win_rate"] >= 55,
        "profit_factor": summary["profit_factor"] >= 1.2,
        "trades_per_day": summary["total_trades"] / 252 >= 3,  # Assuming ~252 trading days
        "profitable": summary["total_pnl"] > 0,
    }

    if success_criteria["win_rate"]:
        print(f"[PASS] Win rate >= 55% (actual: {summary['win_rate']:.1f}%)")
    else:
        print(f"[FAIL] Win rate < 55% (actual: {summary['win_rate']:.1f}%)")

    if success_criteria["profit_factor"]:
        print(f"[PASS] Profit factor >= 1.2 (actual: {summary['profit_factor']:.2f})")
    else:
        print(f"[FAIL] Profit factor < 1.2 (actual: {summary['profit_factor']:.2f})")

    trades_per_day = summary["total_trades"] / 252
    if success_criteria["trades_per_day"]:
        print(f"[PASS] Trades per day >= 3 (actual: {trades_per_day:.1f})")
    else:
        print(f"[FAIL] Trades per day < 3 (actual: {trades_per_day:.1f})")

    if success_criteria["profitable"]:
        print(f"[PASS] Strategy is profitable (P&L: ${summary['total_pnl']:.2f})")
    else:
        print(f"[FAIL] Strategy is NOT profitable (P&L: ${summary['total_pnl']:.2f})")

    all_passed = all(success_criteria.values())

    if all_passed:
        print("\n[SUCCESS] Mean-reversion strategy meets all criteria!")
        print("Proceed to Phase 4 analysis and test set evaluation.")
    else:
        print("\n[NEEDS REFINEMENT] Mean-reversion strategy needs parameter tuning or alternative approach")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
