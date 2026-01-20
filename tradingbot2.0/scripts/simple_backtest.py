#!/usr/bin/env python3
"""
Simple backtest script without risk manager constraints.
Evaluates binary model prediction quality directly.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.data.parquet_loader import ParquetDataLoader
from src.ml.data.scalping_features import ScalpingFeatureEngineer

def main():
    # Load model
    model_path = "models/scalper_binary.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Get model config
    config = checkpoint.get('model_config', {})
    input_dim = checkpoint.get('input_dim', 56)
    hidden_dims = config.get('params', {}).get('hidden_dims', [256, 128, 64])

    # Recreate model
    from src.ml.models.neural_networks import FeedForwardNet
    model = FeedForwardNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=1  # Binary
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get scaler
    scaler_mean = np.array(checkpoint['scaler_mean'])
    scaler_scale = np.array(checkpoint['scaler_scale'])

    print("Loading data...")
    loader = ParquetDataLoader("data/historical/MES/MES_1s_2years.parquet", check_memory=False)
    df = loader.load_data()
    df = loader.convert_to_ny_timezone(df)
    df = loader.filter_rth(df)

    # Filter to 2025 data only
    df = df[df.index.year == 2025]
    print(f"2025 data: {len(df):,} bars")

    # Limit for memory
    df = df.iloc[:100000]
    print(f"Using first 100K bars: {df.index.min()} to {df.index.max()}")

    # Generate features
    print("Generating features...")
    feature_engineer = ScalpingFeatureEngineer(df)
    df = feature_engineer.generate_all_features()

    # Create target (3-class for evaluation)
    df = loader.create_target_variable(df, lookahead_seconds=30, threshold_ticks=3.0)
    df = df.dropna()
    print(f"After features: {len(df):,} samples")

    # Get features
    feature_names = feature_engineer.feature_names
    X = df[feature_names].values
    X_scaled = (X - scaler_mean) / scaler_scale

    # Run model predictions
    print("Running model predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits.squeeze()).numpy()

    # Binary predictions: prob >= 0.5 = UP (1), prob < 0.5 = DOWN (0)
    binary_preds = (probs >= 0.5).astype(int)

    # Map to 3-class for comparison: 0=DOWN, 2=UP
    pred_class_3 = np.where(binary_preds == 1, 2, 0)  # Map 1->2 (UP), 0->0 (DOWN)

    # Get true labels
    y_true_3class = df['target'].values

    # Filter to only directional samples (no FLAT for fair comparison)
    directional_mask = y_true_3class != 1  # Not FLAT
    X_dir = X_scaled[directional_mask]
    y_dir = y_true_3class[directional_mask]
    probs_dir = probs[directional_mask]
    preds_dir = binary_preds[directional_mask]

    # Convert true 3-class to binary: DOWN(0)->0, UP(2)->1
    y_binary = (y_dir == 2).astype(int)

    print(f"\n=== DIRECTIONAL SAMPLES ONLY ===")
    print(f"Total directional samples: {len(y_binary):,}")
    print(f"  DOWN: {(y_binary == 0).sum():,} ({(y_binary == 0).mean()*100:.1f}%)")
    print(f"  UP:   {(y_binary == 1).sum():,} ({(y_binary == 1).mean()*100:.1f}%)")

    # Overall accuracy
    accuracy = (preds_dir == y_binary).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Per-class accuracy
    down_mask = y_binary == 0
    up_mask = y_binary == 1
    down_acc = (preds_dir[down_mask] == y_binary[down_mask]).mean()
    up_acc = (preds_dir[up_mask] == y_binary[up_mask]).mean()
    print(f"DOWN Accuracy: {down_acc:.4f}")
    print(f"UP Accuracy:   {up_acc:.4f}")

    # Confidence analysis
    print(f"\n=== CONFIDENCE ANALYSIS ===")
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        # High confidence = far from 0.5
        high_conf_mask = (probs_dir >= threshold) | (probs_dir <= (1 - threshold))
        if high_conf_mask.sum() > 0:
            hc_acc = (preds_dir[high_conf_mask] == y_binary[high_conf_mask]).mean()
            print(f"Confidence >= {threshold:.0%}: {high_conf_mask.sum():,} trades, {hc_acc:.3f} accuracy")

    # Simulated trading
    print(f"\n=== SIMULATED TRADING (no risk limits) ===")

    # Parameters
    TICK_SIZE = 0.25
    TICK_VALUE = 1.25  # $1.25 per tick for MES
    COMMISSION = 0.42
    SLIPPAGE_TICKS = 1

    # Only trade high confidence signals
    confidence_threshold = 0.65
    trade_mask = (probs_dir >= confidence_threshold) | (probs_dir <= (1 - confidence_threshold))

    # Get predictions and actuals for trades
    trade_preds = preds_dir[trade_mask]
    trade_actual = y_binary[trade_mask]
    trade_probs = probs_dir[trade_mask]

    num_trades = len(trade_preds)
    correct_trades = (trade_preds == trade_actual).sum()
    win_rate = correct_trades / num_trades if num_trades > 0 else 0

    # Estimate PnL
    # Assume average win = 3 ticks, average loss = 2 ticks (based on stop/target)
    avg_win_ticks = 4  # Target
    avg_loss_ticks = 3  # Stop

    winners = (trade_preds == trade_actual).sum()
    losers = num_trades - winners

    gross_win = winners * avg_win_ticks * TICK_VALUE
    gross_loss = losers * avg_loss_ticks * TICK_VALUE
    total_commission = num_trades * COMMISSION * 2  # Round trip
    total_slippage = num_trades * SLIPPAGE_TICKS * TICK_VALUE

    net_pnl = gross_win - gross_loss - total_commission - total_slippage

    print(f"Confidence threshold: {confidence_threshold:.0%}")
    print(f"Total trades: {num_trades:,}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Winners: {winners:,}, Losers: {losers:,}")
    print(f"Gross Win: ${gross_win:.2f}")
    print(f"Gross Loss: ${gross_loss:.2f}")
    print(f"Commission: ${total_commission:.2f}")
    print(f"Slippage: ${total_slippage:.2f}")
    print(f"Net P&L: ${net_pnl:.2f}")
    print(f"Profit Factor: {gross_win/gross_loss:.2f}" if gross_loss > 0 else "Profit Factor: inf")

    # Expected value per trade
    ev_per_trade = (win_rate * avg_win_ticks - (1 - win_rate) * avg_loss_ticks) * TICK_VALUE
    ev_per_trade -= COMMISSION * 2 + SLIPPAGE_TICKS * TICK_VALUE
    print(f"Expected Value per Trade: ${ev_per_trade:.2f}")


if __name__ == "__main__":
    main()
