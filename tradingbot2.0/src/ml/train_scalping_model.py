#!/usr/bin/env python3
"""
Training Script for 3-Class Scalping Model on 1-Second Data.

This script integrates the parquet data pipeline with the training system:
1. Load 1-second parquet data using ParquetDataLoader
2. Generate scalping features using ScalpingFeatureEngineer
3. Train 3-class model (DOWN/FLAT/UP) with CrossEntropyLoss
4. Perform walk-forward validation for time-series aware evaluation
5. Save trained model checkpoint

This script connects Phase 1 (data pipeline) to Phase 4 (model training) per
IMPLEMENTATION_PLAN.md. It uses:
- parquet_loader.py: 1-second data loading with 3-class target
- scalping_features.py: Second-based feature engineering
- neural_networks.py: 3-class output with softmax
- training.py: CrossEntropyLoss with class weights

Usage:
    # Train with default settings on parquet data
    python train_scalping_model.py

    # Train with custom parameters
    python train_scalping_model.py --model lstm --epochs 100 --lookahead 30

    # Use custom data path
    python train_scalping_model.py --data /path/to/data.parquet

The trained model will be saved to models/scalper_v1.pt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import data pipeline (Phase 1)
from data.parquet_loader import (
    ParquetDataLoader,
    load_and_prepare_scalping_data
)
from data.scalping_features import (
    ScalpingFeatureEngineer,
    prepare_scalping_features,
    validate_no_lookahead
)

# Import model components (Phase 4)
from models.neural_networks import (
    create_model,
    FeedForwardNet,
    LSTMNet,
    HybridNet,
    ModelPrediction
)
from models.training import (
    ModelTrainer,
    WalkForwardValidator,
    SequenceDataset,
    train_with_walk_forward
)

# Import evaluation utilities
from utils.evaluation import (
    evaluate_model_and_strategy,
    print_evaluation_report,
    plot_results
)

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 3-class scalping model on 1-second parquet data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/historical/MES/MES_1s_2years.parquet",
        help="Path to parquet file with 1-second OHLCV data"
    )
    parser.add_argument(
        "--filter-rth",
        action="store_true",
        default=True,
        help="Filter to Regular Trading Hours only (9:30 AM - 4:00 PM NY)"
    )
    parser.add_argument(
        "--include-eth",
        action="store_true",
        help="Include Extended Trading Hours (disables RTH filter)"
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
        choices=["feedforward", "lstm", "hybrid"],
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
        help="Training batch size (larger for 1-second data)"
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
        help="Fraction of data for training (default: 0.6)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )

    # Walk-forward arguments
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable walk-forward validation"
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=5,
        help="Number of walk-forward validation splits"
    )

    # LSTM-specific arguments
    parser.add_argument(
        "--seq-length",
        type=int,
        default=60,
        help="Sequence length for LSTM (default: 60 = 1 minute of 1-second bars)"
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
        default="scalper_v1",
        help="Name for saved model checkpoint"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting (for headless environments)"
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
        help="Limit number of samples for debugging (default: use all)"
    )
    parser.add_argument(
        "--validate-lookahead",
        action="store_true",
        help="Run lookahead bias validation checks"
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


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.

    For imbalanced 3-class problem (FLAT ~60%, UP/DOWN ~20% each),
    this gives higher weight to minority classes.

    Args:
        y: Target array with class indices (0, 1, 2)
        num_classes: Number of classes

    Returns:
        Tensor of class weights [weight_DOWN, weight_FLAT, weight_UP]
    """
    class_counts = Counter(y)
    total = len(y)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)

    # Normalize weights so they sum to num_classes
    weights_tensor = torch.FloatTensor(weights)
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes

    return weights_tensor


def print_class_distribution(y: np.ndarray, prefix: str = ""):
    """Print class distribution with labels."""
    class_names = {0: "DOWN", 1: "FLAT", 2: "UP"}
    counts = Counter(y)
    total = len(y)

    print(f"\n{prefix}Class Distribution:")
    for cls in sorted(counts.keys()):
        count = counts[cls]
        pct = count / total * 100
        print(f"  {class_names.get(cls, cls)}: {count:,} ({pct:.1f}%)")


def main():
    """Main training pipeline for scalping model."""
    args = parse_args()

    # Handle RTH/ETH filter
    filter_rth = not args.include_eth

    # Print header
    print("\n" + "=" * 70)
    print("    3-CLASS SCALPING MODEL TRAINING")
    print("    Training on 1-Second Data with DOWN/FLAT/UP Classification")
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
    if args.walk_forward:
        print(f"  Walk-forward:   {args.walk_forward_splits} splits")
    print()

    # Set random seed
    set_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load and Prepare 1-Second Data (Phase 1)
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
            val_ratio=args.val_ratio
        )
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find data file: {args.data}")
        print("Make sure the parquet file exists at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        sys.exit(1)

    print(f"\nData loaded successfully!")
    print(f"  Total bars (RTH): {len(full_df):,}")
    print(f"  Date range:       {full_df.index.min()} to {full_df.index.max()}")
    print(f"  Train split:      {len(train_df):,} bars")
    print(f"  Val split:        {len(val_df):,} bars")
    print(f"  Test split:       {len(test_df):,} bars")

    print_class_distribution(full_df['target'].values, "Overall ")

    # Limit samples for debugging if specified
    if args.max_samples and len(train_df) > args.max_samples:
        print(f"\n  Limiting to {args.max_samples:,} samples for debugging")
        train_df = train_df.iloc[:int(args.max_samples * 0.6)]
        val_df = val_df.iloc[:int(args.max_samples * 0.2)]
        test_df = test_df.iloc[:int(args.max_samples * 0.2)]

    # =========================================================================
    # STEP 2: Feature Engineering (Phase 1)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Generating Scalping Features")
    print("-" * 70)

    # Generate features for each split separately to avoid data leakage
    # (Scaler should be fit only on training data)

    print("\nGenerating features for training set...")
    train_features_df, feature_names, scaler = prepare_scalping_features(
        train_df.copy(),
        normalize=True,
        include_multiframe=True
    )

    print(f"\nGenerating features for validation set...")
    val_engineer = ScalpingFeatureEngineer(val_df.copy())
    val_features_df = val_engineer.generate_all_features(include_multiframe=True)
    # Apply same scaler from training
    val_features_df[feature_names] = scaler.transform(val_features_df[feature_names])

    print(f"\nGenerating features for test set...")
    test_engineer = ScalpingFeatureEngineer(test_df.copy())
    test_features_df = test_engineer.generate_all_features(include_multiframe=True)
    # Apply same scaler from training
    test_features_df[feature_names] = scaler.transform(test_features_df[feature_names])

    print(f"\nFeatures generated: {len(feature_names)}")
    print(f"  Training samples:   {len(train_features_df):,}")
    print(f"  Validation samples: {len(val_features_df):,}")
    print(f"  Test samples:       {len(test_features_df):,}")

    # Lookahead bias validation
    if args.validate_lookahead:
        print("\nValidating no lookahead bias in features...")
        is_valid = validate_no_lookahead(train_features_df, feature_names)
        print(f"  Validation passed: {is_valid}")

    # =========================================================================
    # STEP 3: Prepare Data for Training
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Preparing Data for Neural Network Training")
    print("-" * 70)

    # Extract feature matrices and targets
    X_train = train_features_df[feature_names].values
    y_train = train_features_df['target'].values.astype(np.int64)

    X_val = val_features_df[feature_names].values
    y_val = val_features_df['target'].values.astype(np.int64)

    X_test = test_features_df[feature_names].values
    y_test = test_features_df['target'].values.astype(np.int64)

    # Store prices and returns for evaluation
    prices_train = train_features_df['close'].values
    prices_val = val_features_df['close'].values
    prices_test = test_features_df['close'].values

    print(f"\nFeature matrix shape: {X_train.shape}")
    print_class_distribution(y_train, "Training ")
    print_class_distribution(y_val, "Validation ")
    print_class_distribution(y_test, "Test ")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(y_train, num_classes=3)
    print(f"\nClass weights (for CrossEntropyLoss):")
    print(f"  DOWN: {class_weights[0]:.3f}")
    print(f"  FLAT: {class_weights[1]:.3f}")
    print(f"  UP:   {class_weights[2]:.3f}")

    # Create PyTorch tensors
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    input_dim = X_train.shape[1]
    num_classes = 3  # DOWN, FLAT, UP

    if args.model == 'lstm':
        # Create sequences for LSTM
        train_seq = SequenceDataset(X_train, y_train, args.seq_length, num_classes=num_classes)
        val_seq = SequenceDataset(X_val, y_val, args.seq_length, num_classes=num_classes)
        test_seq = SequenceDataset(X_test, y_test, args.seq_length, num_classes=num_classes)

        X_train_t, y_train_t = train_seq.get_tensors()
        X_val_t, y_val_t = val_seq.get_tensors()
        X_test_t, y_test_t = test_seq.get_tensors()

        # Adjust test prices for sequence offset
        prices_test = prices_test[args.seq_length:]
    else:
        # Tabular data for feedforward/hybrid
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)  # CrossEntropyLoss needs LongTensor

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)

    # Create data loaders
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
    # STEP 4: Create and Train Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Training 3-Class Model")
    print("-" * 70)

    # Model configuration
    model_config = {
        'type': args.model,
        'params': {
            'hidden_dims': hidden_dims,
            'dropout_rate': args.dropout,
            'num_classes': num_classes
        } if args.model == 'feedforward' else {
            'hidden_dim': hidden_dims[0],
            'num_layers': 2,
            'dropout_rate': args.dropout,
            'fc_dims': hidden_dims[1:] if len(hidden_dims) > 1 else [64],
            'num_classes': num_classes
        },
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_classes': num_classes
    }

    # Create model (num_classes already in params)
    model = create_model(
        args.model,
        input_dim,
        **model_config['params']
    )

    print(f"\nModel architecture: {args.model}")
    print(f"  Input dimensions:  {input_dim}")
    print(f"  Output classes:    {num_classes}")
    print(f"  Hidden layers:     {hidden_dims}")
    print(f"  Dropout rate:      {args.dropout}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:  {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")

    # Create trainer with class weights
    trainer = ModelTrainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_classes=num_classes,
        class_weights=class_weights
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Class weights:   Applied")

    # Train the model
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
    print(f"  Best val loss:    {min(history['val_loss']):.4f}")

    # =========================================================================
    # STEP 5: Evaluation on Test Set
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Evaluation on Test Set")
    print("-" * 70)

    # Get predictions (move tensor to same device as model)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_t.to(device)
        if args.model == 'lstm':
            logits = model(X_test_device)
        else:
            logits = model(X_test_device)

        probs = torch.softmax(logits, dim=1).cpu()
        predictions = torch.argmax(probs, dim=1).numpy()
        confidence = probs.max(dim=1).values.numpy()

    y_test_np = y_test_t.numpy() if isinstance(y_test_t, torch.Tensor) else y_test

    # Calculate accuracy
    accuracy = (predictions == y_test_np).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Per-class accuracy
    class_names = {0: "DOWN", 1: "FLAT", 2: "UP"}
    print("\nPer-class Performance:")
    for cls in range(3):
        mask = y_test_np == cls
        if mask.sum() > 0:
            cls_acc = (predictions[mask] == y_test_np[mask]).mean()
            cls_precision = (predictions == cls)[y_test_np == cls].sum() / (predictions == cls).sum() if (predictions == cls).sum() > 0 else 0
            print(f"  {class_names[cls]}: Accuracy={cls_acc:.3f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    from collections import Counter
    confusion = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_test_np, predictions):
        confusion[true, pred] += 1

    print("                 Predicted")
    print("              DOWN  FLAT    UP")
    for i, row in enumerate(confusion):
        print(f"  True {class_names[i]:4s}  {row[0]:5d} {row[1]:5d} {row[2]:5d}")

    # Confidence analysis
    print("\nConfidence Distribution:")
    print(f"  Mean confidence: {confidence.mean():.3f}")
    print(f"  Min confidence:  {confidence.min():.3f}")
    print(f"  Max confidence:  {confidence.max():.3f}")

    # High confidence accuracy (trades we would take with 60%+ confidence)
    high_conf_mask = confidence >= 0.60
    if high_conf_mask.sum() > 0:
        high_conf_acc = (predictions[high_conf_mask] == y_test_np[high_conf_mask]).mean()
        print(f"\n  High confidence (>=60%) trades: {high_conf_mask.sum():,}")
        print(f"  High confidence accuracy: {high_conf_acc:.4f}")

    # =========================================================================
    # STEP 6: Walk-Forward Validation (Optional)
    # =========================================================================
    wf_results = None
    if args.walk_forward:
        print("\n" + "-" * 70)
        print("STEP 6: Walk-Forward Validation")
        print("-" * 70)

        # Combine all data for walk-forward
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])

        wf_results = train_with_walk_forward(
            X_all, y_all,
            model_config=model_config,
            n_splits=args.walk_forward_splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_length=args.seq_length if args.model == 'lstm' else None
        )

        print("\nWalk-Forward Results:")
        for fold in wf_results['fold_metrics']:
            print(f"  Fold {fold['fold']}: Test Accuracy = {fold['test_accuracy']:.4f}")
        print(f"\n  Overall Accuracy: {wf_results['overall_accuracy']:.4f}")
        if 'overall_auc' in wf_results:
            print(f"  Overall AUC:      {wf_results['overall_auc']:.4f}")

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
        'class_weights': class_weights.tolist(),
        'num_classes': num_classes,
        'input_dim': input_dim,
        'training_args': vars(args),
        'training_history': history,
        'test_accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Also save timestamped version
    timestamped_path = output_dir / f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'class_weights': class_weights.tolist(),
        'num_classes': num_classes,
        'input_dim': input_dim,
        'training_args': vars(args),
        'training_history': history,
        'test_accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }, timestamped_path)
    print(f"Timestamped model saved to: {timestamped_path}")

    # Save results JSON
    results = {
        'run_timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'data': {
            'total_samples': len(full_df),
            'train_samples': len(train_features_df),
            'val_samples': len(val_features_df),
            'test_samples': len(test_features_df),
            'num_features': len(feature_names),
            'date_range': {
                'start': str(full_df.index.min()),
                'end': str(full_df.index.max())
            }
        },
        'model': {
            'type': args.model,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'hidden_dims': hidden_dims,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'training': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_loss': min(history['val_loss']),
            'epochs_trained': len(history['train_loss'])
        },
        'evaluation': {
            'test_accuracy': float(accuracy),
            'class_distribution': {int(k): int(v) for k, v in Counter(y_test_np).items()},
            'high_confidence_trades': int(high_conf_mask.sum()) if high_conf_mask.sum() > 0 else 0,
            'high_confidence_accuracy': float(high_conf_acc) if high_conf_mask.sum() > 0 else None
        },
        'walk_forward': wf_results if wf_results else None
    }

    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    if high_conf_mask.sum() > 0:
        print(f"High Confidence Accuracy (>=60%): {high_conf_acc:.4f}")
    print(f"\nTo run backtest with this model:")
    print(f"  python scripts/run_backtest.py --model {model_path}")
    print(f"\nTo start paper trading:")
    print(f"  python scripts/run_live.py --model {model_path}")
    print()

    return accuracy


if __name__ == "__main__":
    accuracy = main()
