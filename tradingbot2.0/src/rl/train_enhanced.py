#!/usr/bin/env python3
"""
Enhanced Training Pipeline with All Improvements.

Combines:
1. Enhanced features (volume profile, regime detection, price action)
2. Regularized model (dropout, layer norm, label smoothing)
3. Walk-forward validation (rolling or anchored windows)
4. Ensemble predictions from multiple models

Usage:
    python src/rl/train_enhanced.py
    python src/rl/train_enhanced.py --walk-forward --epochs 100
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.data_pipeline import MultiHorizonDataPipeline
from src.rl.multi_horizon_model import create_multi_horizon_targets
from src.rl.enhanced_features import generate_enhanced_features, combine_with_base_features
from src.rl.regularized_model import RegularizedMultiHorizonNet, RegularizedTrainer
from src.rl.walk_forward import WalkForwardTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def prepare_enhanced_data(
    data_path: str,
    start_date: str = None,
    end_date: str = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load data and generate enhanced features."""
    logger.info("Loading and preparing enhanced data...")

    # Load base data
    pipeline = MultiHorizonDataPipeline(data_path)
    df = pipeline.load_and_aggregate(start_date, end_date)
    df, base_feature_cols = pipeline.generate_features(df, include_multi_horizon=False)

    # Add multi-horizon targets
    df = create_multi_horizon_targets(df)

    # Generate enhanced features
    df, enhanced_cols = combine_with_base_features(df, base_feature_cols)

    # Drop rows with NaN
    required_cols = enhanced_cols + ['target_1h', 'target_4h', 'target_eod']
    df = df.dropna(subset=required_cols)

    # Handle inf values
    for col in enhanced_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].fillna(0)

    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Total features: {len(enhanced_cols)} ({len(base_feature_cols)} base + {len(enhanced_cols) - len(base_feature_cols)} enhanced)")

    return df, enhanced_cols


def train_single_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 256,
) -> Tuple[RegularizedMultiHorizonNet, StandardScaler, Dict]:
    """Train a single regularized model."""

    # Time-based split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Normalize features
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Clip values
    for split_df in [train_df, val_df, test_df]:
        split_df[feature_cols] = split_df[feature_cols].clip(-5, 5)

    # Create data loaders
    def df_to_loader(df_split, shuffle=True):
        X = torch.FloatTensor(df_split[feature_cols].values)
        y_1h = torch.LongTensor(df_split['target_1h'].values)
        y_4h = torch.LongTensor(df_split['target_4h'].values)
        y_eod = torch.LongTensor(df_split['target_eod'].values)
        dataset = TensorDataset(X, y_1h, y_4h, y_eod)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = df_to_loader(train_df, shuffle=True)
    val_loader = df_to_loader(val_df, shuffle=False)

    # Create model
    model = RegularizedMultiHorizonNet(
        input_dim=len(feature_cols),
        hidden_dims=[512, 256, 128],
        dropout_rate=0.4,
        num_residual_blocks=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Create trainer
    trainer = RegularizedTrainer(
        model,
        learning_rate=0.0005,
        weight_decay=0.05,
        label_smoothing=0.1,
    )
    logger.info(f"Device: {trainer.device}")

    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
    )

    # Evaluate on test set
    model.eval()
    device = trainer.device
    X_test = torch.FloatTensor(test_df[feature_cols].values).to(device)

    with torch.no_grad():
        logits_1h, logits_4h, logits_eod = model(X_test)
        probs_1h = torch.sigmoid(logits_1h).cpu().numpy().flatten()
        probs_4h = torch.sigmoid(logits_4h).cpu().numpy().flatten()
        probs_eod = torch.sigmoid(logits_eod).cpu().numpy().flatten()

    y_1h = test_df['target_1h'].values
    y_4h = test_df['target_4h'].values
    y_eod = test_df['target_eod'].values

    test_results = {
        'acc_1h': ((probs_1h > 0.5).astype(int) == y_1h).mean(),
        'acc_4h': ((probs_4h > 0.5).astype(int) == y_4h).mean(),
        'acc_eod': ((probs_eod > 0.5).astype(int) == y_eod).mean(),
        'n_test': len(test_df),
    }

    # High confidence analysis
    for horizon, probs, y_true in [('1h', probs_1h, y_1h), ('4h', probs_4h, y_4h), ('eod', probs_eod, y_eod)]:
        for threshold in [0.6, 0.7]:
            high_conf_mask = (probs >= threshold) | (probs <= (1 - threshold))
            if high_conf_mask.sum() > 0:
                preds = (probs[high_conf_mask] > 0.5).astype(int)
                acc = (preds == y_true[high_conf_mask]).mean()
                test_results[f'acc_{horizon}_conf{int(threshold*100)}'] = acc
                test_results[f'n_{horizon}_conf{int(threshold*100)}'] = int(high_conf_mask.sum())

    return model, scaler, test_results


def main():
    parser = argparse.ArgumentParser(description="Enhanced training pipeline")
    parser.add_argument("--data", type=str,
                       default="data/historical/MES/MES_1s_2years.parquet",
                       help="Path to data file")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-12-31",
                       help="End date")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--walk-forward", action="store_true",
                       help="Use walk-forward validation")
    parser.add_argument("--train-months", type=int, default=12,
                       help="Training window size (months) for walk-forward")
    parser.add_argument("--test-months", type=int, default=2,
                       help="Test window size (months) for walk-forward")
    parser.add_argument("--output", type=str, default="models/enhanced",
                       help="Output path")

    args = parser.parse_args()

    print("=" * 70)
    print("    ENHANCED TRAINING PIPELINE")
    print("    Regularization + Enhanced Features + Walk-Forward")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:          {args.data}")
    print(f"  Date range:    {args.start_date} to {args.end_date}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Patience:      {args.patience}")
    print(f"  Walk-forward:  {args.walk_forward}")
    print()

    # Load and prepare data
    df, feature_cols = prepare_enhanced_data(
        args.data,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.walk_forward:
        # Walk-forward training
        print("\n" + "-" * 70)
        print("Running Walk-Forward Training...")
        print("-" * 70)

        trainer = WalkForwardTrainer(
            df=df,
            feature_cols=feature_cols,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=2,
            anchored=False,
            model_save_dir=f"{args.output}_walkforward",
        )

        summary = trainer.run(
            epochs=args.epochs,
            patience=args.patience,
        )

        print("\n" + "=" * 70)
        print("    WALK-FORWARD TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nWindows completed: {summary['aggregate']['n_windows']}")
        print(f"Average 1h Accuracy:  {summary['aggregate']['avg_acc_1h']:.3f} ± {summary['aggregate']['std_acc_1h']:.3f}")
        print(f"Average 4h Accuracy:  {summary['aggregate']['avg_acc_4h']:.3f} ± {summary['aggregate']['std_acc_4h']:.3f}")
        print(f"Average EOD Accuracy: {summary['aggregate']['avg_acc_eod']:.3f} ± {summary['aggregate']['std_acc_eod']:.3f}")
        print(f"\nModels saved to: {args.output}_walkforward/")

    else:
        # Single model training
        print("\n" + "-" * 70)
        print("Training Regularized Model...")
        print("-" * 70)

        model, scaler, test_results = train_single_model(
            df,
            feature_cols,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

        print("\n" + "-" * 70)
        print("Test Set Results:")
        print("-" * 70)
        print(f"  1h Accuracy:  {test_results['acc_1h']:.4f} ({test_results['acc_1h']*100:.1f}%)")
        print(f"  4h Accuracy:  {test_results['acc_4h']:.4f} ({test_results['acc_4h']*100:.1f}%)")
        print(f"  EOD Accuracy: {test_results['acc_eod']:.4f} ({test_results['acc_eod']*100:.1f}%)")

        print(f"\nHigh Confidence Predictions:")
        for horizon in ['1h', '4h', 'eod']:
            for conf in [60, 70]:
                key_acc = f'acc_{horizon}_conf{conf}'
                key_n = f'n_{horizon}_conf{conf}'
                if key_acc in test_results:
                    print(f"  {horizon} ({conf}%+ conf): {test_results[key_acc]:.3f} acc, {test_results[key_n]:,} samples")

        # Save model
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': len(feature_cols),
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.4,
                'num_residual_blocks': 2,
            },
            'feature_cols': feature_cols,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'test_results': test_results,
            'args': vars(args),
            'timestamp': datetime.now().isoformat(),
        }

        torch.save(save_dict, f"{output_path}.pt")
        logger.info(f"Model saved to {output_path}.pt")

        # Save info as JSON
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
                return [convert_numpy(v) for v in obj]
            return obj

        info_dict = {
            'model_config': save_dict['model_config'],
            'feature_cols': feature_cols,
            'test_results': convert_numpy(test_results),
            'args': vars(args),
            'timestamp': save_dict['timestamp'],
        }
        with open(f"{output_path}_info.json", 'w') as f:
            json.dump(info_dict, f, indent=2)

        print("\n" + "=" * 70)
        print("    TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nModel saved to: {output_path}.pt")


if __name__ == "__main__":
    main()
