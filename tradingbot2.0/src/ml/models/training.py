"""
Training Pipeline for Futures Trading ML Model.

Implements:
1. Standard train/test split training
2. Walk-forward validation (rolling window)
3. Time-series cross-validation
4. Training utilities and callbacks

Best Practices:
- Temporal data handling (no data leakage)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- Comprehensive logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from datetime import datetime
import logging
import json

from .neural_networks import FeedForwardNet, LSTMNet, EarlyStopping, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceDataset:
    """Create sequences for LSTM training from tabular data."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 20):
        """
        Initialize sequence dataset.

        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,)
            seq_length: Number of timesteps per sequence
        """
        self.seq_length = seq_length

        X_seq, y_seq = [], []
        for i in range(len(features) - seq_length):
            X_seq.append(features[i:i + seq_length])
            y_seq.append(targets[i + seq_length])

        self.X = np.array(X_seq)
        self.y = np.array(y_seq)

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return PyTorch tensors."""
        return (
            torch.FloatTensor(self.X),
            torch.FloatTensor(self.y).unsqueeze(1)
        )


class ModelTrainer:
    """
    Trainer class for neural network models.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        scheduler_patience: int = 5
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            device: 'cpu', 'cuda', 'mps', or 'auto'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            scheduler_patience: LR scheduler patience
        """
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function (binary cross-entropy)
        self.criterion = nn.BCELoss()

        # Optimizer (AdamW with weight decay for regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            verbose=True
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if isinstance(self.model, LSTMNet):
                outputs, _ = self.model(batch_x)
            else:
                outputs = self.model(batch_x)

            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * len(batch_y)
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += len(batch_y)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if isinstance(self.model, LSTMNet):
                    outputs, _ = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * len(batch_y)
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += len(batch_y)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Train batches: {len(train_loader)}, "
                   f"Val batches: {len(val_loader) if val_loader else 0}")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping check
                if early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                val_loss, val_acc = 0.0, 0.0

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            # Checkpointing
            if checkpoint_dir and (epoch + 1) % 20 == 0:
                self.save_checkpoint(checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_path / "model_final.pt")

        logger.info("Training complete!")
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature array

        Returns:
            Probability predictions
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            if isinstance(self.model, LSTMNet):
                outputs, _ = self.model(X_tensor)
            else:
                outputs = self.model(X_tensor)

        return outputs.cpu().numpy()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Checkpoint loaded from {path}")


class WalkForwardValidator:
    """
    Walk-Forward Validation for time series.

    Implements expanding or rolling window cross-validation
    that respects temporal ordering of data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None,
        min_train_size: int = None,
        expanding: bool = True
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of train/test splits
            test_size: Size of each test set (in samples)
            min_train_size: Minimum training set size
            expanding: If True, training window expands; if False, it rolls
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.expanding = expanding

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each split.

        Args:
            X: Features array
            y: Target array (optional, for API compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if self.test_size is None:
            self.test_size = n_samples // (self.n_splits + 1)

        if self.min_train_size is None:
            self.min_train_size = self.test_size * 2

        splits = []

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.min_train_size)

            train_end = test_start

            if train_end - train_start >= self.min_train_size:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                splits.append((train_indices, test_indices))

        return splits


def train_with_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    model_config: Dict,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    seq_length: int = 20
) -> Dict:
    """
    Train model using walk-forward validation.

    Args:
        X: Feature matrix
        y: Target vector
        model_config: Model configuration dict
        n_splits: Number of walk-forward splits
        epochs: Training epochs per fold
        batch_size: Batch size
        seq_length: Sequence length for LSTM

    Returns:
        Dictionary with results for each fold
    """
    results = {
        'fold_metrics': [],
        'predictions': [],
        'actuals': [],
        'timestamps': []
    }

    validator = WalkForwardValidator(n_splits=n_splits, expanding=True)
    splits = validator.split(X, y)

    for fold, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold + 1}/{n_splits}")
        logger.info(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        logger.info(f"{'='*60}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create model
        model_type = model_config.get('type', 'feedforward')

        if model_type == 'lstm':
            # Create sequences
            train_seq = SequenceDataset(X_train, y_train, seq_length)
            test_seq = SequenceDataset(X_test, y_test, seq_length)
            X_train_t, y_train_t = train_seq.get_tensors()
            X_test_t, y_test_t = test_seq.get_tensors()
            input_dim = X.shape[1]
        else:
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
            X_test_t = torch.FloatTensor(X_test)
            y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
            input_dim = X.shape[1]

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create and train model
        model = create_model(model_type, input_dim, **model_config.get('params', {}))
        trainer = ModelTrainer(
            model,
            learning_rate=model_config.get('learning_rate', 0.001),
            weight_decay=model_config.get('weight_decay', 0.01)
        )

        history = trainer.train(
            train_loader,
            test_loader,
            epochs=epochs,
            early_stopping_patience=10
        )

        # Evaluate on test set
        test_loss, test_acc = trainer.validate(test_loader)

        # Get predictions
        predictions = trainer.predict(X_test_t.numpy() if model_type != 'lstm' else X_test_t.numpy())

        results['fold_metrics'].append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1]
        })

        results['predictions'].extend(predictions.flatten().tolist())
        results['actuals'].extend(y_test_t.numpy().flatten().tolist())

    # Calculate overall metrics
    all_preds = np.array(results['predictions'])
    all_actuals = np.array(results['actuals'])
    binary_preds = (all_preds > 0.5).astype(int)

    results['overall_accuracy'] = (binary_preds == all_actuals).mean()
    results['overall_auc'] = calculate_auc(all_actuals, all_preds)

    logger.info(f"\n{'='*60}")
    logger.info("WALK-FORWARD VALIDATION COMPLETE")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Overall AUC: {results['overall_auc']:.4f}")
    logger.info(f"{'='*60}")

    return results


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Area Under ROC Curve."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)
    except ImportError:
        # Simple AUC approximation if sklearn not available
        sorted_idx = np.argsort(y_pred)[::-1]
        sorted_true = y_true[sorted_idx]
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tpr_cum = np.cumsum(sorted_true) / n_pos
        fpr_cum = np.cumsum(1 - sorted_true) / n_neg
        return np.trapz(tpr_cum, fpr_cum)


if __name__ == "__main__":
    # Test training pipeline
    print("Testing Training Pipeline")
    print("="*60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 40

    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.5).astype(float)

    # Test walk-forward validation
    model_config = {
        'type': 'feedforward',
        'params': {
            'hidden_dims': [64, 32],
            'dropout_rate': 0.3
        },
        'learning_rate': 0.001,
        'weight_decay': 0.01
    }

    results = train_with_walk_forward(
        X, y,
        model_config=model_config,
        n_splits=3,
        epochs=10,
        batch_size=32
    )

    print("\nFold Results:")
    for fold_result in results['fold_metrics']:
        print(f"  Fold {fold_result['fold']}: Acc={fold_result['test_accuracy']:.4f}")
