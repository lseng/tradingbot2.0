#!/usr/bin/env python3
"""
Main Training Script for Futures Price Direction Prediction.

This script orchestrates the full ML pipeline:
1. Load and preprocess data
2. Generate technical features
3. Train neural network models
4. Evaluate with walk-forward validation
5. Generate trading strategy results

Usage:
    python train_futures_model.py --data /path/to/data.txt --model feedforward
    python train_futures_model.py --data /path/to/data.txt --model lstm --epochs 100

╔══════════════════════════════════════════════════════════════════════════════╗
║  IMPORTANT DISCLAIMER - READ BEFORE USING                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This code is for EDUCATIONAL and RESEARCH purposes only.                     ║
║                                                                               ║
║  LIMITATIONS OF ML/AI FOR TRADING:                                            ║
║  1. Markets are highly efficient - edges are hard to find and maintain        ║
║  2. Past performance does NOT predict future results                          ║
║  3. Overfitting is a major risk - models that work on historical data         ║
║     often fail on new data                                                    ║
║  4. Transaction costs, slippage, and market impact erode profits              ║
║  5. Black swan events can cause catastrophic losses                           ║
║  6. Most retail trading strategies underperform buy-and-hold                  ║
║                                                                               ║
║  BEFORE RISKING REAL MONEY:                                                   ║
║  - Paper trade for at least 6-12 months                                       ║
║  - Understand the strategy's limitations                                      ║
║  - Never risk money you can't afford to lose                                  ║
║  - Consider consulting a financial professional                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import FuturesDataLoader, load_and_prepare_data
from data.feature_engineering import FeatureEngineer
from models.neural_networks import create_model, FeedForwardNet, LSTMNet
from models.training import (
    ModelTrainer,
    WalkForwardValidator,
    SequenceDataset,
    train_with_walk_forward
)
from utils.evaluation import (
    evaluate_model_and_strategy,
    print_evaluation_report,
    plot_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural network for futures price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data",
        type=str,
        default="/Users/leoneng/Downloads/MES_full_1min_continuous_UNadjusted.txt",
        help="Path to raw 1-minute OHLCV data file"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["feedforward", "lstm"],
        default="feedforward",
        help="Model architecture to use"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )

    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="128,64,32",
        help="Hidden layer dimensions (comma-separated)"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for regularization"
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

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training"
    )

    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=5,
        help="Number of walk-forward validation splits"
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=20,
        help="Sequence length for LSTM"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting (for headless environments)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training pipeline."""
    args = parse_args()

    # Print header
    print("\n" + "="*70)
    print("    FUTURES PRICE DIRECTION PREDICTION - ML TRAINING PIPELINE")
    print("="*70)
    print(f"\nRun started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Data:           {args.data}")
    print(f"  Model:          {args.model}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Walk-forward:   {args.walk_forward_splits} splits")
    print()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load and Preprocess Data
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading and Preprocessing Data")
    print("-"*70)

    loader = FuturesDataLoader(args.data)
    raw_data = loader.load_raw_data()
    daily_data = loader.resample_to_daily()
    daily_with_target = loader.create_target_variable(daily_data)

    print(f"\nData loaded successfully!")
    print(f"  Raw data points:  {len(raw_data):,}")
    print(f"  Daily bars:       {len(daily_data):,}")
    print(f"  Date range:       {daily_data.index.min().date()} to {daily_data.index.max().date()}")

    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Feature Engineering")
    print("-"*70)

    engineer = FeatureEngineer(daily_with_target)
    featured_data = engineer.generate_all_features()
    feature_names = engineer.get_feature_names()

    print(f"\nFeatures generated: {len(feature_names)}")
    print(f"Final dataset size: {len(featured_data)} samples")

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # =========================================================================
    # STEP 3: Prepare Data for Training
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Preparing Data for Training")
    print("-"*70)

    # Extract features and target
    X = featured_data[feature_names].values
    y = featured_data['target'].values
    prices = featured_data['close'].values
    returns = featured_data['next_return'].values

    # Normalize features
    X_scaled = scaler.fit_transform(X)

    print(f"\nFeature matrix shape: {X_scaled.shape}")
    print(f"Target distribution: Up={y.sum()} ({y.mean():.1%}), Down={len(y) - y.sum()}")

    # =========================================================================
    # STEP 4: Train with Walk-Forward Validation
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Training with Walk-Forward Validation")
    print("-"*70)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    model_config = {
        'type': args.model,
        'params': {
            'hidden_dims': hidden_dims,
            'dropout_rate': args.dropout
        } if args.model == 'feedforward' else {
            'hidden_dim': hidden_dims[0],
            'num_layers': 2,
            'dropout_rate': args.dropout,
            'fc_dims': hidden_dims[1:] if len(hidden_dims) > 1 else [32]
        },
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay
    }

    wf_results = train_with_walk_forward(
        X_scaled, y,
        model_config=model_config,
        n_splits=args.walk_forward_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )

    # =========================================================================
    # STEP 5: Final Model Training
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Training Final Model on All Data")
    print("-"*70)

    # Train final model on 80% of data, evaluate on last 20%
    split_idx = int(len(X_scaled) * args.train_ratio)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prices_test = prices[split_idx:]
    returns_test = returns[split_idx:]

    # Create model
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if args.model == 'lstm':
        train_seq = SequenceDataset(X_train, y_train, args.seq_length)
        test_seq = SequenceDataset(X_test, y_test, args.seq_length)
        X_train_t, y_train_t = train_seq.get_tensors()
        X_test_t, y_test_t = test_seq.get_tensors()
        input_dim = X_scaled.shape[1]
    else:
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
        input_dim = X_scaled.shape[1]

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    final_model = create_model(args.model, input_dim, **model_config['params'])
    trainer = ModelTrainer(
        final_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    history = trainer.train(
        train_loader,
        test_loader,
        epochs=args.epochs,
        early_stopping_patience=15,
        checkpoint_dir=str(output_dir / "checkpoints")
    )

    # =========================================================================
    # STEP 6: Evaluation
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 6: Comprehensive Evaluation")
    print("-"*70)

    # Get predictions
    predictions = trainer.predict(X_test_t.numpy() if args.model != 'lstm' else X_test_t.numpy())
    predictions = predictions.flatten()

    # Adjust arrays for evaluation
    if args.model == 'lstm':
        # LSTM predictions are for samples after seq_length
        y_eval = y_test_t.numpy().flatten()
        prices_eval = prices_test[args.seq_length:]
        returns_eval = returns_test[args.seq_length:]
    else:
        y_eval = y_test
        prices_eval = prices_test
        returns_eval = returns_test

    # Trim to match prediction length
    min_len = min(len(predictions), len(y_eval), len(prices_eval), len(returns_eval))
    predictions = predictions[:min_len]
    y_eval = y_eval[:min_len]
    prices_eval = prices_eval[:min_len]
    returns_eval = returns_eval[:min_len]

    # Full evaluation
    eval_results = evaluate_model_and_strategy(
        y_eval, predictions, prices_eval, returns_eval
    )

    # Print results
    print_evaluation_report(eval_results)

    # Print walk-forward summary
    print("\n📊 WALK-FORWARD VALIDATION SUMMARY")
    print("-"*40)
    for fold in wf_results['fold_metrics']:
        print(f"  Fold {fold['fold']}: Accuracy = {fold['test_accuracy']:.4f}, "
              f"Loss = {fold['test_loss']:.4f}")
    print(f"\n  Overall Accuracy: {wf_results['overall_accuracy']:.4f}")
    print(f"  Overall AUC:      {wf_results['overall_auc']:.4f}")

    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 7: Saving Results")
    print("-"*70)

    # Save model
    import torch
    model_path = output_dir / f"model_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': model_config,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }, model_path)
    print(f"  Model saved to: {model_path}")

    # Save results
    results_to_save = {
        'run_timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'walk_forward_results': {
            'overall_accuracy': wf_results['overall_accuracy'],
            'overall_auc': wf_results['overall_auc'],
            'fold_metrics': wf_results['fold_metrics']
        },
        'final_evaluation': {
            'classification': eval_results['classification'],
            'trading': eval_results['trading'],
            'comparison': eval_results['comparison']
        },
        'training_history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc']
        }
    }

    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # Plot results
    if not args.no_plot:
        try:
            plot_path = output_dir / f"evaluation_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_results(eval_results, save_path=str(plot_path))
        except Exception as e:
            print(f"  Could not generate plot: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nKey Metrics:")
    print(f"  Walk-Forward Accuracy: {wf_results['overall_accuracy']:.4f}")
    print(f"  Test Set Accuracy:     {eval_results['classification']['accuracy']:.4f}")
    print(f"  Test Set AUC:          {eval_results['classification']['auc_roc']:.4f}")
    print(f"  Strategy Return:       {eval_results['trading']['total_return']*100:.2f}%")
    print(f"  Buy & Hold Return:     {eval_results['comparison']['buy_hold_return']*100:.2f}%")
    print(f"  Sharpe Ratio:          {eval_results['trading']['sharpe_ratio']:.3f}")

    print("\n" + "="*70)
    print("⚠️  REMINDER: This is for educational purposes only!")
    print("    Paper trade extensively before risking real capital.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
