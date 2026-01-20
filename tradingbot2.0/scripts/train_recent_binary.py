#!/usr/bin/env python3
"""
Train binary model on recent 2024-2025 data.
Uses time-based train/val/test splits to avoid look-ahead bias.
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.data.parquet_loader import ParquetDataLoader
from src.ml.data.scalping_features import ScalpingFeatureEngineer
from src.ml.models.neural_networks import FeedForwardNet


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BinaryTrainer:
    def __init__(self, model, lr=0.001, weight_decay=0.01, pos_weight=None):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device) if pos_weight else None)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.best_val_loss = float('inf')
        self.best_state = None

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.float().to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X).squeeze(-1)
            loss = self.criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
            correct += ((torch.sigmoid(out) > 0.5).float() == y).sum().item()
            total += len(y)
        return total_loss / total, correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.float().to(self.device)
                out = self.model(X).squeeze(-1)
                loss = self.criterion(out, y)
                total_loss += loss.item() * len(y)
                correct += ((torch.sigmoid(out) > 0.5).float() == y).sum().item()
                total += len(y)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=50, patience=15):
        wait = 0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop at epoch {epoch+1}")
                    break
        if self.best_state:
            self.model.load_state_dict(self.best_state)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            return torch.sigmoid(self.model(X).squeeze(-1)).cpu().numpy()


def main():
    set_seed(42)

    print("="*60)
    print("TRAINING BINARY MODEL ON 2024-2025 DATA")
    print("="*60)

    # Load data - skip memory check
    print("\n[1] Loading data...")
    loader = ParquetDataLoader("data/historical/MES/MES_1s_2years.parquet", check_memory=False)
    df = loader.load_data()
    df = loader.convert_to_ny_timezone(df)
    df = loader.filter_rth(df)

    # Filter to 2024-2025 only
    df = df[(df.index.year >= 2024)]
    print(f"2024-2025 data: {len(df):,} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Sample for memory: take every 3rd bar
    df = df.iloc[::3]
    print(f"After 3x downsampling: {len(df):,} bars")

    # Chronological split: Train on 2024, Test on 2025
    train_df = df[df.index.year == 2024]
    test_df = df[df.index.year == 2025]

    # Further limit for memory
    train_df = train_df.iloc[:500000]
    test_df = test_df.iloc[:200000]

    print(f"\nTrain: {len(train_df):,} (2024)")
    print(f"Test:  {len(test_df):,} (2025)")

    # Generate features
    print("\n[2] Generating features...")

    fe_train = ScalpingFeatureEngineer(train_df)
    train_feat = fe_train.generate_all_features()
    feature_names = fe_train.feature_names

    fe_test = ScalpingFeatureEngineer(test_df)
    test_feat = fe_test.generate_all_features()

    # Create 3-class target
    train_feat = loader.create_target_variable(train_feat, lookahead_seconds=30, threshold_ticks=3.0)
    test_feat = loader.create_target_variable(test_feat, lookahead_seconds=30, threshold_ticks=3.0)

    train_feat = train_feat.dropna(subset=feature_names + ['target'])
    test_feat = test_feat.dropna(subset=feature_names + ['target'])

    print(f"Train samples: {len(train_feat):,}")
    print(f"Test samples:  {len(test_feat):,}")

    # Filter to binary (no FLAT)
    train_binary = train_feat[train_feat['target'] != 1].copy()
    test_binary = test_feat[test_feat['target'] != 1].copy()

    # Remap: DOWN(0)->0, UP(2)->1
    train_binary['target'] = (train_binary['target'] == 2).astype(int)
    test_binary['target'] = (test_binary['target'] == 2).astype(int)

    print(f"\nBinary train: {len(train_binary):,}")
    print(f"Binary test:  {len(test_binary):,}")

    # Split train into train/val (80/20)
    split_idx = int(len(train_binary) * 0.8)
    train_split = train_binary.iloc[:split_idx]
    val_split = train_binary.iloc[split_idx:]

    print(f"Train/Val split: {len(train_split):,} / {len(val_split):,}")

    # Extract features and handle inf values
    X_train = train_split[feature_names].values.astype(np.float32)
    y_train = train_split['target'].values
    X_val = val_split[feature_names].values.astype(np.float32)
    y_val = val_split['target'].values
    X_test = test_binary[feature_names].values.astype(np.float32)
    y_test = test_binary['target'].values

    # Replace inf with nan, then fill with column median
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_val = np.where(np.isinf(X_val), np.nan, X_val)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    # Fill NaN with column median
    for i in range(X_train.shape[1]):
        median = np.nanmedian(X_train[:, i])
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), median, X_train[:, i])
        X_val[:, i] = np.where(np.isnan(X_val[:, i]), median, X_val[:, i])
        X_test[:, i] = np.where(np.isnan(X_test[:, i]), median, X_test[:, i])

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nFeature shape: {X_train.shape}")
    print(f"Train class dist: DOWN={sum(y_train==0):,}, UP={sum(y_train==1):,}")

    # Class weight (use float32 for MPS compatibility)
    pos_weight = torch.tensor([sum(y_train == 0) / sum(y_train == 1)], dtype=torch.float32)
    print(f"Pos weight: {pos_weight.item():.3f}")

    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Create model
    print("\n[3] Training model...")
    model = FeedForwardNet(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3,
        num_classes=1
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = BinaryTrainer(model, lr=0.001, weight_decay=0.01, pos_weight=pos_weight)
    trainer.fit(train_loader, val_loader, epochs=50, patience=15)

    # Evaluate on test set
    print("\n[4] Evaluating on 2025 test set...")
    probs = trainer.predict(X_test)
    preds = (probs >= 0.5).astype(int)

    accuracy = (preds == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    down_mask = y_test == 0
    up_mask = y_test == 1
    print(f"DOWN Accuracy: {(preds[down_mask] == y_test[down_mask]).mean():.4f}")
    print(f"UP Accuracy:   {(preds[up_mask] == y_test[up_mask]).mean():.4f}")

    # Confidence analysis
    print("\nConfidence Analysis:")
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        high_conf = (probs >= thresh) | (probs <= (1 - thresh))
        if high_conf.sum() > 0:
            hc_acc = (preds[high_conf] == y_test[high_conf]).mean()
            print(f"  {thresh:.0%} conf: {high_conf.sum():,} trades, {hc_acc:.3f} acc")

    # Save model
    model_path = Path("models/scalper_binary_2024.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'type': 'feedforward',
            'params': {
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'num_classes': 1
            },
            'binary': True
        },
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'input_dim': X_train.shape[1],
        'binary': True,
        'num_classes': 1,
        'test_accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
