#!/usr/bin/env python3
"""
Binary Classification Training Script for Scalping Model.

This script trains a binary classifier (UP vs DOWN only) by:
1. Loading 1-second parquet data with 3-class targets
2. Filtering out FLAT samples to keep only directional moves
3. Remapping labels: DOWN (0) -> 0, UP (2) -> 1
4. Training with BCEWithLogitsLoss for binary classification
5. Saving model checkpoint compatible with backtest system

The key insight: 3-class models predict FLAT ~94% of the time because
that's the dominant class. By training only on directional moves,
the model learns to distinguish UP from DOWN with higher confidence.

Usage:
    python train_binary_scalper.py  # Default settings
    python train_binary_scalper.py --threshold 5.0  # Larger moves only
    python train_binary_scalper.py --model lstm --epochs 100
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from data.parquet_loader import (
    ParquetDataLoader,
    load_and_prepare_scalping_data
)
from data.scalping_features import (
    ScalpingFeatureEngineer,
    prepare_scalping_features
)
from models.neural_networks import create_model
from models.training import ModelTrainer, SequenceDataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train binary (UP vs DOWN) scalping model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/historical/MES/MES_1s_2years.parquet",
        help="Path to parquet file with 1-second OHLCV data"
    )
    parser.add_argument(
        "--include-eth",
        action="store_true",
        help="Include Extended Trading Hours (default: RTH only)"
    )

    # Target variable arguments
    parser.add_argument(
        "--lookahead",
        type=int,
        default=30,
        help="Lookahead window in seconds for target variable (default: 30s)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Tick threshold for UP/DOWN classification (default: 3.0 ticks)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["feedforward", "lstm"],
        default="feedforward",
        help="Model architecture to use"
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,128,64",
        help="Hidden layer dimensions (comma-separated)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for regularization"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="L2 regularization strength"
    )

    # Data split arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation"
    )

    # LSTM-specific arguments
    parser.add_argument(
        "--seq-length",
        type=int,
        default=60,
        help="Sequence length for LSTM"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="scalper_binary",
        help="Name for saved model checkpoint"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Debugging
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for debugging"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only include UP and DOWN samples (no FLAT).

    Args:
        df: DataFrame with 'target' column (0=DOWN, 1=FLAT, 2=UP)

    Returns:
        DataFrame with only DOWN (0) and UP (2) samples,
        target remapped to 0=DOWN, 1=UP
    """
    # Keep only directional moves (no FLAT)
    filtered = df[df['target'] != 1].copy()

    # Remap labels: DOWN stays 0, UP becomes 1
    filtered['target'] = (filtered['target'] == 2).astype(int)

    return filtered


def print_binary_distribution(y: np.ndarray, prefix: str = ""):
    """Print binary class distribution."""
    counts = Counter(y)
    total = len(y)

    print(f"\n{prefix}Binary Class Distribution:")
    down_count = counts.get(0, 0)
    up_count = counts.get(1, 0)
    print(f"  DOWN: {down_count:,} ({down_count/total*100:.1f}%)")
    print(f"  UP:   {up_count:,} ({up_count/total*100:.1f}%)")


def compute_binary_class_weight(y: np.ndarray) -> torch.Tensor:
    """
    Compute class weight for binary imbalanced data.
    Returns pos_weight for BCEWithLogitsLoss.
    """
    counts = Counter(y)
    neg_count = counts.get(0, 1)  # DOWN
    pos_count = counts.get(1, 1)  # UP

    # pos_weight = neg_count / pos_count
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    return torch.FloatTensor([pos_weight])


class BinaryModelTrainer:
    """
    Trainer for binary classification models.
    Uses BCEWithLogitsLoss for numerical stability.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        pos_weight: Optional[torch.Tensor] = None,
        device: str = 'auto'
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # BCEWithLogitsLoss with optional class weighting
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.float().to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_batch)

            # Handle LSTM tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Squeeze to match target shape
            outputs = outputs.squeeze(-1)

            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)

            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.float().to(self.device)

                outputs = self.model(X_batch)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                outputs = outputs.squeeze(-1)

                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * len(y_batch)

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += len(y_batch)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[str] = None
    ) -> Dict:
        """Full training loop with early stopping."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0

                # Save checkpoint
                if checkpoint_dir:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
                    torch.save(self.best_model_state, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()

        return probs


def main():
    """Main training pipeline for binary scalping model."""
    args = parse_args()

    filter_rth = not args.include_eth

    print("\n" + "=" * 70)
    print("    BINARY SCALPING MODEL TRAINING")
    print("    UP vs DOWN Classification (No FLAT)")
    print("=" * 70)
    print(f"\nRun started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:           {args.data}")
    print(f"  Filter RTH:     {filter_rth}")
    print(f"  Lookahead:      {args.lookahead}s")
    print(f"  Threshold:      {args.threshold} ticks")
    print(f"  Model:          {args.model}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load 3-Class Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading 1-Second Parquet Data")
    print("-" * 70)

    try:
        full_df, train_df, val_df, test_df = load_and_prepare_scalping_data(
            data_path=args.data,
            filter_rth=filter_rth,
            lookahead_seconds=args.lookahead,
            threshold_ticks=args.threshold,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            check_memory=False  # Disable memory check - we'll subsample if needed
        )
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        sys.exit(1)

    print(f"\nOriginal 3-class data loaded:")
    print(f"  Total bars:  {len(full_df):,}")
    print(f"  Train:       {len(train_df):,}")
    print(f"  Val:         {len(val_df):,}")
    print(f"  Test:        {len(test_df):,}")

    # Print original 3-class distribution
    class_names = {0: "DOWN", 1: "FLAT", 2: "UP"}
    counts = Counter(full_df['target'])
    print("\n  Original class distribution:")
    for cls in sorted(counts.keys()):
        print(f"    {class_names[cls]}: {counts[cls]:,} ({counts[cls]/len(full_df)*100:.1f}%)")

    # =========================================================================
    # STEP 2: Filter to Binary (UP vs DOWN only)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Filtering to Binary Classification")
    print("-" * 70)

    train_binary = filter_to_binary(train_df)
    val_binary = filter_to_binary(val_df)
    test_binary = filter_to_binary(test_df)

    print(f"\nAfter filtering out FLAT samples:")
    print(f"  Train: {len(train_df):,} -> {len(train_binary):,} ({len(train_binary)/len(train_df)*100:.1f}% kept)")
    print(f"  Val:   {len(val_df):,} -> {len(val_binary):,} ({len(val_binary)/len(val_df)*100:.1f}% kept)")
    print(f"  Test:  {len(test_df):,} -> {len(test_binary):,} ({len(test_binary)/len(test_df)*100:.1f}% kept)")

    print_binary_distribution(train_binary['target'].values, "Training ")

    # Limit samples for debugging
    if args.max_samples:
        print(f"\n  Limiting to {args.max_samples:,} samples for debugging")
        n_train = int(args.max_samples * 0.6)
        n_val = int(args.max_samples * 0.2)
        n_test = int(args.max_samples * 0.2)
        train_binary = train_binary.iloc[:n_train]
        val_binary = val_binary.iloc[:n_val]
        test_binary = test_binary.iloc[:n_test]

    # =========================================================================
    # STEP 3: Feature Engineering
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Generating Scalping Features")
    print("-" * 70)

    print("\nGenerating features for training set...")
    train_features_df, feature_names, scaler = prepare_scalping_features(
        train_binary.copy(),
        normalize=True,
        include_multiframe=True
    )

    print(f"Generating features for validation set...")
    val_engineer = ScalpingFeatureEngineer(val_binary.copy())
    val_features_df = val_engineer.generate_all_features(include_multiframe=True)
    val_features_df[feature_names] = val_features_df[feature_names].replace([np.inf, -np.inf], np.nan)
    val_features_df = val_features_df.dropna(subset=feature_names)
    val_features_df[feature_names] = scaler.transform(val_features_df[feature_names])

    print(f"Generating features for test set...")
    test_engineer = ScalpingFeatureEngineer(test_binary.copy())
    test_features_df = test_engineer.generate_all_features(include_multiframe=True)
    test_features_df[feature_names] = test_features_df[feature_names].replace([np.inf, -np.inf], np.nan)
    test_features_df = test_features_df.dropna(subset=feature_names)
    test_features_df[feature_names] = scaler.transform(test_features_df[feature_names])

    print(f"\nFeatures generated: {len(feature_names)}")
    print(f"  Training samples:   {len(train_features_df):,}")
    print(f"  Validation samples: {len(val_features_df):,}")
    print(f"  Test samples:       {len(test_features_df):,}")

    # =========================================================================
    # STEP 4: Prepare Data for Training
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Preparing Data for Neural Network Training")
    print("-" * 70)

    X_train = train_features_df[feature_names].values
    y_train = train_features_df['target'].values.astype(np.int64)

    X_val = val_features_df[feature_names].values
    y_val = val_features_df['target'].values.astype(np.int64)

    X_test = test_features_df[feature_names].values
    y_test = test_features_df['target'].values.astype(np.int64)

    print(f"\nFeature matrix shape: {X_train.shape}")
    print_binary_distribution(y_train, "Training ")
    print_binary_distribution(y_val, "Validation ")
    print_binary_distribution(y_test, "Test ")

    # Compute class weight for imbalanced binary data
    pos_weight = compute_binary_class_weight(y_train)
    print(f"\nPos weight for BCELoss: {pos_weight.item():.3f}")

    # Create PyTorch tensors
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    input_dim = X_train.shape[1]

    # For binary classification, output dim is 1
    num_classes = 1

    if args.model == 'lstm':
        train_seq = SequenceDataset(X_train, y_train, args.seq_length, num_classes=2)  # Binary
        val_seq = SequenceDataset(X_val, y_val, args.seq_length, num_classes=2)
        test_seq = SequenceDataset(X_test, y_test, args.seq_length, num_classes=2)

        X_train_t, y_train_t = train_seq.get_tensors()
        X_val_t, y_val_t = val_seq.get_tensors()
        X_test_t, y_test_t = test_seq.get_tensors()
    else:
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nDataLoader created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # =========================================================================
    # STEP 5: Create and Train Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Training Binary Classification Model")
    print("-" * 70)

    # Model configuration - output 1 neuron for binary
    model_params = {
        'hidden_dims': hidden_dims,
        'dropout_rate': args.dropout,
        'num_classes': num_classes  # 1 for binary
    } if args.model == 'feedforward' else {
        'hidden_dim': hidden_dims[0],
        'num_layers': 2,
        'dropout_rate': args.dropout,
        'fc_dims': hidden_dims[1:] if len(hidden_dims) > 1 else [64],
        'num_classes': num_classes
    }

    model_config = {
        'type': args.model,
        'params': model_params,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_classes': num_classes,
        'binary': True  # Flag for backtest script
    }

    model = create_model(args.model, input_dim, **model_params)

    print(f"\nModel architecture: {args.model}")
    print(f"  Input dimensions:  {input_dim}")
    print(f"  Output:            1 (binary)")
    print(f"  Hidden layers:     {hidden_dims}")
    print(f"  Dropout rate:      {args.dropout}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:  {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")

    # Create binary trainer
    trainer = BinaryModelTrainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight
    )

    print(f"\nDevice: {trainer.device}")
    print(f"\nStarting training...")

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        early_stopping_patience=15,
        checkpoint_dir=str(output_dir / "checkpoints")
    )

    print(f"\nTraining completed!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Final train acc:  {history['train_acc'][-1]:.4f}")
    print(f"  Final val acc:    {history['val_acc'][-1]:.4f}")

    # =========================================================================
    # STEP 6: Evaluation on Test Set
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Evaluation on Test Set")
    print("-" * 70)

    # Get predictions
    probs = trainer.predict_proba(X_test)
    predictions = (probs > 0.5).astype(int)

    y_test_np = y_test

    # Calculate accuracy
    accuracy = (predictions == y_test_np).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Per-class accuracy
    print("\nPer-class Performance:")
    for cls, name in [(0, "DOWN"), (1, "UP")]:
        mask = y_test_np == cls
        if mask.sum() > 0:
            cls_acc = (predictions[mask] == y_test_np[mask]).mean()
            print(f"  {name}: Accuracy={cls_acc:.3f} ({mask.sum():,} samples)")

    # Confusion matrix
    print("\nConfusion Matrix:")
    confusion = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_test_np, predictions):
        confusion[true, pred] += 1

    print("              Predicted")
    print("              DOWN     UP")
    print(f"  True DOWN  {confusion[0, 0]:6d} {confusion[0, 1]:6d}")
    print(f"  True UP    {confusion[1, 0]:6d} {confusion[1, 1]:6d}")

    # Confidence distribution
    print("\nConfidence Distribution:")
    print(f"  Mean:   {probs.mean():.3f}")
    print(f"  Std:    {probs.std():.3f}")
    print(f"  Min:    {probs.min():.3f}")
    print(f"  Max:    {probs.max():.3f}")

    # High confidence accuracy
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        high_conf_mask = (probs >= threshold) | (probs <= (1 - threshold))
        if high_conf_mask.sum() > 0:
            high_conf_acc = (predictions[high_conf_mask] == y_test_np[high_conf_mask]).mean()
            print(f"  Confidence >= {threshold:.0%}: {high_conf_mask.sum():,} trades, {high_conf_acc:.3f} accuracy")

    # =========================================================================
    # STEP 7: Save Model and Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 7: Saving Model and Results")
    print("-" * 70)

    # Save model checkpoint
    model_path = models_dir / f"{args.model_name}.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'num_classes': num_classes,
        'binary': True,  # Important flag for backtest
        'input_dim': input_dim,
        'training_args': vars(args),
        'training_history': history,
        'test_accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save timestamped version
    timestamped_path = output_dir / f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'num_classes': num_classes,
        'binary': True,
        'input_dim': input_dim,
        'training_args': vars(args),
        'training_history': history,
        'test_accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat()
    }, timestamped_path)
    print(f"Timestamped model saved to: {timestamped_path}")

    # Save results JSON
    results = {
        'run_timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'binary_classification': True,
        'data': {
            'total_directional_samples': len(train_binary) + len(val_binary) + len(test_binary),
            'train_samples': len(train_features_df),
            'val_samples': len(val_features_df),
            'test_samples': len(test_features_df),
            'num_features': len(feature_names)
        },
        'model': {
            'type': args.model,
            'input_dim': input_dim,
            'output_dim': num_classes,
            'hidden_dims': hidden_dims,
            'total_params': total_params
        },
        'training': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'epochs_trained': len(history['train_loss'])
        },
        'evaluation': {
            'test_accuracy': float(accuracy),
            'confusion_matrix': confusion.tolist()
        }
    }

    results_path = output_dir / f"results_binary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("    TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\nKey insight: This binary model only distinguishes UP from DOWN.")
    print(f"During live trading, use a FLAT detector or volatility filter")
    print(f"to avoid trading during sideways markets.")
    print(f"\nTo run backtest:")
    print(f"  python scripts/run_backtest.py --model {model_path}")
    print()

    return accuracy


if __name__ == "__main__":
    main()
