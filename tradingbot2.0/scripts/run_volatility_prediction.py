#!/usr/bin/env python3
"""
Volatility Prediction Training and Validation Script

This script implements an alternative approach after the direction prediction
approach failed validation (AUC 0.51, no predictive signal).

Instead of predicting price direction (UP/DOWN), we predict volatility:
- HIGH volatility periods (good for trading - bigger moves)
- LOW volatility periods (avoid trading - small moves, whipsaws)

Hypothesis: Volatility is more predictable than direction because:
1. Volatility clusters (high vol follows high vol)
2. Time-of-day effects are strong (open/close more volatile)
3. Technical features like ATR and Bollinger Bands directly measure volatility

Strategy:
- Train model to predict which 30-minute windows will have HIGH volatility
- Only trade during predicted HIGH volatility periods
- Can combine with direction prediction or trade breakouts

Usage:
    python scripts/run_volatility_prediction.py [--threshold 60] [--output results/volatility]
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
from src.scalping.features import (
    ScalpingFeatureGenerator,
    create_volatility_target,
    create_target_variable,
)
from src.scalping.model import ScalpingModel, ModelConfig

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


def generate_features_with_volatility_target(
    df: pd.DataFrame,
    horizon_bars: int = 6,
    vol_threshold_percentile: float = 60.0,
) -> tuple[pd.DataFrame, list[str], float]:
    """
    Generate features and volatility target.

    Returns:
        Tuple of (DataFrame with features, feature names, volatility threshold used)
    """
    logger.info("Generating features...")

    feature_gen = ScalpingFeatureGenerator()
    df_features = feature_gen.generate_all(df.copy())

    # Create volatility target
    df_features, vol_threshold = create_volatility_target(
        df_features,
        horizon_bars=horizon_bars,
        threshold_percentile=vol_threshold_percentile,
    )

    # Also create direction target for combined analysis
    df_features = create_target_variable(df_features, horizon_bars=horizon_bars, min_move_ticks=2)

    feature_names = feature_gen.get_feature_names()
    logger.info(f"Generated {len(feature_names)} features")

    # Use volatility target
    target_col = f"target_volatility_{horizon_bars}bar"
    df_features = df_features.rename(columns={target_col: "target_volatility"})

    # Direction target
    direction_target_col = f"target_{horizon_bars}bar"
    df_features = df_features.rename(columns={direction_target_col: "target_direction"})

    # Drop rows with NaN
    drop_cols = feature_names + ['target_volatility', 'target_direction']
    df_features = df_features.dropna(subset=drop_cols)
    logger.info(f"After dropping NaN: {len(df_features):,} bars")

    return df_features, feature_names, vol_threshold


def split_data(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_start: str = "2023-01-01",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets by date."""
    train_mask = df.index <= pd.Timestamp(train_end, tz="America/New_York")
    val_mask = (df.index >= pd.Timestamp(val_start, tz="America/New_York")) & \
               (df.index <= pd.Timestamp(val_end, tz="America/New_York"))

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    logger.info(f"Training data: {len(train_df):,} bars ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Validation data: {len(val_df):,} bars ({val_df.index.min()} to {val_df.index.max()})")

    return train_df, val_df


def train_volatility_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: list[str],
) -> ScalpingModel:
    """Train LightGBM model to predict volatility."""
    logger.info("Training LightGBM volatility model...")

    X_train = train_df[feature_names].values
    y_train = train_df['target_volatility'].values.astype(int)

    X_val = val_df[feature_names].values
    y_val = val_df['target_volatility'].values.astype(int)

    # Log class balance
    train_high_pct = y_train.mean() * 100
    val_high_pct = y_val.mean() * 100
    logger.info(f"Training set: {train_high_pct:.1f}% HIGH volatility")
    logger.info(f"Validation set: {val_high_pct:.1f}% HIGH volatility")

    model_config = ModelConfig(
        # Slightly more regularization to avoid overfitting
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
    )

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
    logger.info("Top 10 features for volatility prediction:")
    for name, score in top_features:
        logger.info(f"  {name}: {score:.4f}")

    return model, result


def analyze_volatility_prediction(
    val_df: pd.DataFrame,
    model: ScalpingModel,
    feature_names: list[str],
    threshold: float = 0.60,
) -> dict:
    """
    Analyze how well volatility predictions work.

    We check:
    1. Does the model predict HIGH volatility accurately?
    2. Does HIGH volatility correlate with direction prediction accuracy?
    3. Can we improve direction trading by filtering on volatility?
    """
    logger.info("Analyzing volatility predictions...")

    X_val = val_df[feature_names].values
    y_vol_true = val_df['target_volatility'].values
    y_dir_true = val_df['target_direction'].values

    # Get volatility predictions
    vol_probs = model.predict_proba(X_val)
    vol_pred = (vol_probs >= threshold).astype(int)

    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    results = {
        "volatility_model": {
            "auc": roc_auc_score(y_vol_true, vol_probs),
            "accuracy": accuracy_score(y_vol_true, vol_pred),
            "precision": precision_score(y_vol_true, vol_pred),
            "recall": recall_score(y_vol_true, vol_pred),
            "f1": f1_score(y_vol_true, vol_pred),
        }
    }

    # Analyze direction accuracy by volatility regime
    high_vol_mask = y_vol_true == 1
    low_vol_mask = y_vol_true == 0

    # Direction accuracy in different regimes
    results["direction_by_regime"] = {
        "high_volatility": {
            "count": int(high_vol_mask.sum()),
            "pct_up": float(y_dir_true[high_vol_mask].mean() * 100) if high_vol_mask.sum() > 0 else 0,
        },
        "low_volatility": {
            "count": int(low_vol_mask.sum()),
            "pct_up": float(y_dir_true[low_vol_mask].mean() * 100) if low_vol_mask.sum() > 0 else 0,
        }
    }

    # Analyze by predicted volatility
    pred_high_mask = vol_pred == 1
    pred_low_mask = vol_pred == 0

    results["by_predicted_volatility"] = {
        "predicted_high": {
            "count": int(pred_high_mask.sum()),
            "actual_high_pct": float(y_vol_true[pred_high_mask].mean() * 100) if pred_high_mask.sum() > 0 else 0,
            "direction_up_pct": float(y_dir_true[pred_high_mask].mean() * 100) if pred_high_mask.sum() > 0 else 0,
        },
        "predicted_low": {
            "count": int(pred_low_mask.sum()),
            "actual_high_pct": float(y_vol_true[pred_low_mask].mean() * 100) if pred_low_mask.sum() > 0 else 0,
            "direction_up_pct": float(y_dir_true[pred_low_mask].mean() * 100) if pred_low_mask.sum() > 0 else 0,
        }
    }

    # Confidence distribution
    results["confidence_distribution"] = {
        "min": float(vol_probs.min()),
        "max": float(vol_probs.max()),
        "mean": float(vol_probs.mean()),
        "std": float(vol_probs.std()),
        "pct_above_60": float((vol_probs >= 0.60).mean() * 100),
        "pct_above_70": float((vol_probs >= 0.70).mean() * 100),
    }

    # Feature correlation with volatility target
    feature_corrs = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X_val[:, i], y_vol_true)[0, 1]
        feature_corrs.append((name, corr))

    feature_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    results["top_feature_correlations"] = {name: round(corr, 4) for name, corr in feature_corrs[:10]}

    return results


def print_results(results: dict, vol_threshold: float) -> None:
    """Print formatted analysis results."""
    print("\n" + "=" * 70)
    print("VOLATILITY PREDICTION ANALYSIS")
    print("=" * 70)

    print(f"\nVolatility threshold percentile: {vol_threshold:.4f}")

    # Model performance
    vm = results["volatility_model"]
    print(f"\n--- Volatility Model Performance ---")
    print(f"AUC: {vm['auc']:.4f}")
    print(f"Accuracy: {vm['accuracy']:.4f}")
    print(f"Precision: {vm['precision']:.4f}")
    print(f"Recall: {vm['recall']:.4f}")
    print(f"F1 Score: {vm['f1']:.4f}")

    # Confidence distribution
    cd = results["confidence_distribution"]
    print(f"\n--- Confidence Distribution ---")
    print(f"Range: {cd['min']:.3f} - {cd['max']:.3f}")
    print(f"Mean: {cd['mean']:.3f} (std: {cd['std']:.3f})")
    print(f"Samples with confidence >= 60%: {cd['pct_above_60']:.1f}%")
    print(f"Samples with confidence >= 70%: {cd['pct_above_70']:.1f}%")

    # Direction by regime
    dbr = results["direction_by_regime"]
    print(f"\n--- Direction by Actual Volatility Regime ---")
    print(f"HIGH volatility: {dbr['high_volatility']['count']:,} bars, "
          f"{dbr['high_volatility']['pct_up']:.1f}% UP")
    print(f"LOW volatility: {dbr['low_volatility']['count']:,} bars, "
          f"{dbr['low_volatility']['pct_up']:.1f}% UP")

    # By predicted volatility
    bpv = results["by_predicted_volatility"]
    print(f"\n--- By Predicted Volatility ---")
    ph = bpv["predicted_high"]
    pl = bpv["predicted_low"]
    print(f"Predicted HIGH: {ph['count']:,} bars, "
          f"actually high: {ph['actual_high_pct']:.1f}%, "
          f"direction UP: {ph['direction_up_pct']:.1f}%")
    print(f"Predicted LOW: {pl['count']:,} bars, "
          f"actually high: {pl['actual_high_pct']:.1f}%, "
          f"direction UP: {pl['direction_up_pct']:.1f}%")

    # Feature correlations
    print(f"\n--- Top Feature Correlations with Volatility ---")
    for name, corr in results["top_feature_correlations"].items():
        print(f"  {name}: {corr:+.4f}")

    # Success assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if vm['auc'] > 0.55:
        print("[SUCCESS] Volatility model has predictive signal (AUC > 0.55)")
    elif vm['auc'] > 0.52:
        print("[MARGINAL] Volatility model has weak signal (AUC 0.52-0.55)")
    else:
        print("[FAILED] Volatility model has no signal (AUC <= 0.52)")

    if cd['pct_above_60'] > 30:
        print("[SUCCESS] Model produces confident predictions (>30% above 60%)")
    else:
        print("[WARNING] Model is uncertain (few predictions above 60%)")

    # Check if filtering on volatility helps direction
    high_actual_high = bpv["predicted_high"]["actual_high_pct"]
    low_actual_high = bpv["predicted_low"]["actual_high_pct"]

    if high_actual_high > low_actual_high + 10:
        print(f"[SUCCESS] Volatility prediction discriminates well "
              f"({high_actual_high:.1f}% vs {low_actual_high:.1f}%)")
    else:
        print(f"[WARNING] Volatility prediction discrimination is weak "
              f"({high_actual_high:.1f}% vs {low_actual_high:.1f}%)")

    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Train and validate volatility prediction model")
    parser.add_argument(
        "--data",
        default="data/historical/MES/MES_full_1min_continuous_UNadjusted.txt",
        help="Path to 1-minute data file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Percentile threshold for HIGH volatility (default: 60)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Confidence threshold for predictions (default: 0.60)"
    )
    parser.add_argument(
        "--output",
        default="results/volatility",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model"
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

    # Step 1: Load data
    df_rth = load_and_prepare_data(args.data)

    # Step 2: Generate features with volatility target
    df_features, feature_names, vol_threshold = generate_features_with_volatility_target(
        df_rth,
        horizon_bars=6,
        vol_threshold_percentile=args.threshold,
    )

    # Step 3: Split data
    train_df, val_df = split_data(df_features)

    # Step 4: Train volatility model
    model, training_result = train_volatility_model(train_df, val_df, feature_names)

    # Step 5: Analyze predictions
    analysis = analyze_volatility_prediction(
        val_df, model, feature_names, threshold=args.confidence
    )

    # Step 6: Print results
    print_results(analysis, vol_threshold)

    # Step 7: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_to_save = {
        "timestamp": timestamp,
        "config": {
            "threshold_percentile": args.threshold,
            "confidence": args.confidence,
            "horizon_bars": 6,
        },
        "volatility_threshold": vol_threshold,
        "training": {
            "best_iteration": training_result.best_iteration,
            "val_auc": training_result.val_auc,
            "val_accuracy": training_result.val_accuracy,
        },
        "analysis": analysis,
    }

    results_path = output_dir / f"volatility_analysis_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save model if requested
    if args.save_model:
        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"volatility_model_{timestamp}"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

    # Return success based on AUC
    auc = analysis["volatility_model"]["auc"]
    if auc > 0.55:
        logger.info(f"SUCCESS: Volatility model has signal (AUC={auc:.4f})")
        return 0
    else:
        logger.warning(f"No signal: Volatility model AUC={auc:.4f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
