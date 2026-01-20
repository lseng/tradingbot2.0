#!/usr/bin/env python3
"""
Train Multi-Horizon Direction Prediction Model.

This script trains a supervised model to predict price direction at:
- 1 hour ahead
- 4 hours ahead
- End of day (EOD)

The trained model will be used as a "signal generator" for the RL agent.

Usage:
    python src/rl/train_multi_horizon.py
    python src/rl/train_multi_horizon.py --epochs 100 --start-date 2023-01-01
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
from src.rl.multi_horizon_model import (
    MultiHorizonNet,
    MultiHorizonTrainer,
    create_multi_horizon_targets,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def prepare_data(
    data_path: str,
    start_date: str = None,
    end_date: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], StandardScaler]:
    """
    Load and prepare data for multi-horizon training.

    Returns:
        (train_df, val_df, test_df, feature_columns, scaler)
    """
    logger.info("Loading and preparing data...")

    pipeline = MultiHorizonDataPipeline(data_path)

    # Load and aggregate to 1-minute bars
    df = pipeline.load_and_aggregate(start_date, end_date)

    # Generate features
    df, feature_cols = pipeline.generate_features(df, include_multi_horizon=False)

    # Create multi-horizon targets
    df = create_multi_horizon_targets(df)

    # Drop rows with NaN targets (end of day, future lookbacks)
    df = df.dropna(subset=['target_1h', 'target_4h', 'target_eod'] + feature_cols)

    logger.info(f"Total samples after cleaning: {len(df):,}")

    # Print target distributions
    for horizon in ['1h', '4h', 'eod']:
        col = f'target_{horizon}'
        up_pct = df[col].mean() * 100
        logger.info(f"  {horizon}: {up_pct:.1f}% UP, {100-up_pct:.1f}% DOWN")

    # Time-based split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Handle any remaining inf/nan
    for split_df in [train_df, val_df, test_df]:
        split_df[feature_cols] = split_df[feature_cols].replace([np.inf, -np.inf], 0)
        split_df[feature_cols] = split_df[feature_cols].fillna(0)
        split_df[feature_cols] = split_df[feature_cols].clip(-5, 5)

    return train_df, val_df, test_df, feature_cols, scaler


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders."""

    def df_to_tensors(df):
        X = torch.FloatTensor(df[feature_cols].values)
        y_1h = torch.LongTensor(df['target_1h'].values)
        y_4h = torch.LongTensor(df['target_4h'].values)
        y_eod = torch.LongTensor(df['target_eod'].values)
        return TensorDataset(X, y_1h, y_4h, y_eod)

    train_dataset = df_to_tensors(train_df)
    val_dataset = df_to_tensors(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate_model(
    model: MultiHorizonNet,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()

    X_test = torch.FloatTensor(test_df[feature_cols].values).to(device)
    y_1h = test_df['target_1h'].values
    y_4h = test_df['target_4h'].values
    y_eod = test_df['target_eod'].values

    with torch.no_grad():
        logits_1h, logits_4h, logits_eod = model(X_test)
        probs_1h = torch.sigmoid(logits_1h).cpu().numpy().flatten()
        probs_4h = torch.sigmoid(logits_4h).cpu().numpy().flatten()
        probs_eod = torch.sigmoid(logits_eod).cpu().numpy().flatten()

    preds_1h = (probs_1h > 0.5).astype(int)
    preds_4h = (probs_4h > 0.5).astype(int)
    preds_eod = (probs_eod > 0.5).astype(int)

    results = {
        'acc_1h': (preds_1h == y_1h).mean(),
        'acc_4h': (preds_4h == y_4h).mean(),
        'acc_eod': (preds_eod == y_eod).mean(),
    }

    # High confidence analysis
    for horizon, probs, y_true in [('1h', probs_1h, y_1h), ('4h', probs_4h, y_4h), ('eod', probs_eod, y_eod)]:
        for threshold in [0.6, 0.7]:
            high_conf_mask = (probs >= threshold) | (probs <= (1 - threshold))
            if high_conf_mask.sum() > 0:
                preds = (probs[high_conf_mask] > 0.5).astype(int)
                acc = (preds == y_true[high_conf_mask]).mean()
                results[f'acc_{horizon}_conf{int(threshold*100)}'] = acc
                results[f'n_{horizon}_conf{int(threshold*100)}'] = high_conf_mask.sum()

    return results


def main():
    parser = argparse.ArgumentParser(description="Train multi-horizon prediction model")
    parser.add_argument("--data", type=str,
                       default="data/historical/MES/MES_1s_2years.parquet",
                       help="Path to data file")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Start date for training data")
    parser.add_argument("--end-date", type=str, default="2025-06-30",
                       help="End date (leave some for RL testing)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--hidden-dims", type=str, default="256,128,64",
                       help="Hidden layer dimensions")
    parser.add_argument("--output", type=str, default="models/multi_horizon",
                       help="Output path for model")

    args = parser.parse_args()

    print("=" * 70)
    print("    MULTI-HORIZON DIRECTION PREDICTION MODEL")
    print("    Training to predict 1h, 4h, and EOD price direction")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:         {args.data}")
    print(f"  Date range:   {args.start_date} to {args.end_date}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()

    # Prepare data
    train_df, val_df, test_df, feature_cols, scaler = prepare_data(
        args.data,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, feature_cols, batch_size=args.batch_size
    )

    # Create model
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    input_dim = len(feature_cols)

    model = MultiHorizonNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=0.3,
    )

    logger.info(f"Model: {input_dim} inputs -> {hidden_dims} hidden -> 3 outputs")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = MultiHorizonTrainer(
        model,
        learning_rate=args.lr,
        weight_decay=0.01,
    )

    logger.info(f"Device: {trainer.device}")

    # Train
    print("\n" + "-" * 70)
    print("Training...")
    print("-" * 70)

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=10,
    )

    # Evaluate on test set
    print("\n" + "-" * 70)
    print("Evaluating on test set...")
    print("-" * 70)

    test_results = evaluate_model(model, test_df, feature_cols, trainer.device)

    print(f"\nTest Set Results:")
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
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': 0.3,
        },
        'feature_cols': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'training_history': history,
        'test_results': test_results,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(save_dict, f"{output_path}.pt")
    logger.info(f"Model saved to {output_path}.pt")

    # Save info as JSON (convert numpy types to native Python)
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
    print(f"\nThis model will be used by the hybrid RL agent to make")
    print(f"better-informed trading decisions.")


if __name__ == "__main__":
    main()
