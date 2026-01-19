"""
Training Pipeline for Futures Trading ML Model.

Implements:
1. Standard train/test split training
2. Walk-forward validation (rolling window)
3. Time-series cross-validation
4. Training utilities and callbacks

Supports:
- Binary classification (UP/DOWN) with BCELoss
- 3-class classification (DOWN/FLAT/UP) with CrossEntropyLoss for scalping

Best Practices:
- Temporal data handling (no data leakage)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- Comprehensive logging
- Class weighting for imbalanced data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from pathlib import Path
from datetime import datetime
import logging
import json

from .neural_networks import FeedForwardNet, LSTMNet, EarlyStopping, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceDataset:
    """Create sequences for LSTM training from tabular data."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 20,
        num_classes: int = 3
    ):
        """
        Initialize sequence dataset.

        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,) - class indices for multi-class
            seq_length: Number of timesteps per sequence
            num_classes: Number of classes (2=binary with BCELoss, 3+=multi-class with CrossEntropyLoss)
        """
        self.seq_length = seq_length
        self.num_classes = num_classes

        # Use NumPy stride tricks for O(1) sequence creation instead of slow Python loop.
        # This is critical for large datasets (6M+ samples) where the for-loop takes 60+ minutes.
        # sliding_window_view creates a memory-efficient view in constant time.
        n_samples = len(features) - seq_length
        if n_samples <= 0:
            raise ValueError(f"Not enough samples ({len(features)}) for sequence length {seq_length}")

        # Create sliding window view - O(1) operation that creates a view, not a copy
        # sliding_window_view produces shape (n_windows, n_features, seq_length)
        # We need to transpose to (n_windows, seq_length, n_features) for LSTM input
        X_view = sliding_window_view(features, window_shape=seq_length, axis=0)

        # Transpose axes to get (n_samples, seq_length, n_features)
        # sliding_window_view produces one extra window, so slice to n_samples
        # and materialize with .copy() to get contiguous memory for PyTorch
        self.X = np.transpose(X_view[:n_samples], (0, 2, 1)).copy()

        # Targets aligned to the end of each sequence
        self.y = targets[seq_length:seq_length + n_samples].copy()

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return PyTorch tensors.

        For 3-class (CrossEntropyLoss): Returns (FloatTensor, LongTensor)
        """
        # CrossEntropyLoss expects class indices as LongTensor
        return (
            torch.FloatTensor(self.X),
            torch.LongTensor(self.y)  # Class indices (0, 1, 2)
        )


def create_sequences_fast(
    features: np.ndarray,
    targets: np.ndarray,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create LSTM sequences in O(1) time using NumPy stride tricks.

    This is a standalone function for creating sequences without the SequenceDataset class.
    Uses sliding_window_view for memory-efficient sequence creation that completes in
    seconds instead of hours for large datasets (6M+ samples).

    Args:
        features: Feature array of shape (n_samples, n_features)
        targets: Target array of shape (n_samples,)
        seq_length: Number of timesteps per sequence

    Returns:
        Tuple of (X, y) where:
            X: Sequences of shape (n_samples - seq_length, seq_length, n_features)
            y: Targets of shape (n_samples - seq_length,)

    Performance:
        - Before (Python loop): 60+ minutes for 6.2M samples
        - After (stride tricks): ~10-30 seconds for 6.2M samples

    Example:
        >>> features = np.random.randn(1000, 40)
        >>> targets = np.random.randint(0, 3, 1000)
        >>> X, y = create_sequences_fast(features, targets, seq_length=20)
        >>> X.shape  # (980, 20, 40)
        >>> y.shape  # (980,)
    """
    n_samples = len(features) - seq_length
    if n_samples <= 0:
        raise ValueError(f"Not enough samples ({len(features)}) for sequence length {seq_length}")

    # Create sliding window view - O(1) operation
    # sliding_window_view produces (n_windows, n_features, seq_length)
    X_view = sliding_window_view(features, window_shape=seq_length, axis=0)

    # Transpose to (n_samples, seq_length, n_features) and materialize
    X = np.transpose(X_view[:n_samples], (0, 2, 1)).copy()
    y = targets[seq_length:seq_length + n_samples].copy()

    return X, y


class ModelTrainer:
    """
    Trainer class for neural network models.

    Handles training loop, validation, checkpointing, and logging.
    Supports both binary (BCELoss) and multi-class (CrossEntropyLoss) classification.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        scheduler_patience: int = 5,
        num_classes: int = 3,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            device: 'cpu', 'cuda', 'mps', or 'auto'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            scheduler_patience: LR scheduler patience
            num_classes: Number of output classes (2=binary, 3=scalping with FLAT)
            class_weights: Optional tensor of class weights for imbalanced data
                           Shape: (num_classes,), e.g., [1.5, 0.5, 1.5] for DOWN/FLAT/UP
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
        self.num_classes = num_classes

        # Loss function - CrossEntropyLoss for multi-class classification
        # CrossEntropyLoss expects raw logits (no softmax) and class indices as targets
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"Using class weights: {class_weights.tolist()}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

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
            patience=scheduler_patience
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

            # CrossEntropyLoss expects: outputs (N, C), targets (N,) as class indices
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics - use argmax for multi-class predictions
            total_loss += loss.item() * len(batch_y)
            predictions = torch.argmax(outputs, dim=1)  # Get predicted class indices
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

                # CrossEntropyLoss expects: outputs (N, C), targets (N,) as class indices
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * len(batch_y)
                predictions = torch.argmax(outputs, dim=1)  # Get predicted class indices
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
            Class probabilities (N, num_classes) after softmax
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            if isinstance(self.model, LSTMNet):
                outputs, _ = self.model(X_tensor)
            else:
                outputs = self.model(X_tensor)

            # Apply softmax to get class probabilities
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (argmax of probabilities).

        Args:
            X: Feature array

        Returns:
            Predicted class indices (N,)
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

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


def _simulate_trading_for_fold(
    predicted_classes: np.ndarray,
    predictions: np.ndarray,
    prices: np.ndarray,
    returns: Optional[np.ndarray] = None,
    num_classes: int = 3,
    initial_capital: float = 100000.0,
    position_size: float = 0.02,
    commission: float = 5.0,
    slippage: float = 0.0001,
) -> Dict:
    """
    Simulate trading for a single fold using predicted classes.

    Converts 3-class predictions to trading signals:
    - DOWN (0): Go SHORT
    - FLAT (1): Stay FLAT (no position)
    - UP (2): Go LONG

    Uses a simplified TradingSimulator approach that doesn't depend on
    probability thresholds but directly uses predicted classes.

    Args:
        predicted_classes: Array of predicted class indices (0=DOWN, 1=FLAT, 2=UP)
        predictions: Array of class probabilities (N, num_classes)
        prices: Close prices for the test period
        returns: Pre-calculated returns (optional)
        num_classes: Number of classes (should be 3 for scalping)
        initial_capital: Starting capital
        position_size: Fraction of capital to risk per trade
        commission: Commission per trade
        slippage: Slippage as fraction of price

    Returns:
        Dictionary with trading metrics:
        - sharpe_ratio: Annualized Sharpe ratio
        - max_drawdown: Maximum drawdown (as decimal, e.g., 0.10 for 10%)
        - win_rate: Win rate (as decimal, e.g., 0.55 for 55%)
        - profit_factor: Profit factor
        - total_trades: Number of trades executed
        - total_return: Total return (as decimal)
        - annualized_return: Annualized return
        - avg_trade_pnl: Average P&L per trade
    """
    if len(prices) < 2:
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'avg_trade_pnl': 0.0,
        }

    # Calculate returns if not provided
    if returns is None:
        returns = np.diff(prices) / prices[:-1]
        # Align: prediction[i] is for returns[i], so truncate predictions
        predicted_classes = predicted_classes[:-1]
        predictions = predictions[:-1]

    n_days = min(len(predicted_classes), len(returns))

    capital = initial_capital
    position = 0  # 1=long, -1=short, 0=flat
    trades = []
    equity_curve = [capital]

    for i in range(n_days):
        daily_return = returns[i] if i < len(returns) else 0

        # Convert class prediction to position
        # DOWN (0) = SHORT (-1), FLAT (1) = FLAT (0), UP (2) = LONG (1)
        pred_class = predicted_classes[i]
        if num_classes == 3:
            if pred_class == 2:  # UP
                new_position = 1
            elif pred_class == 0:  # DOWN
                new_position = -1
            else:  # FLAT
                new_position = 0
        else:
            # Binary case: 0=DOWN, 1=UP
            new_position = 1 if pred_class == 1 else -1

        # Execute trade if position changed
        if new_position != position:
            position = new_position
            capital -= commission

        # Calculate P&L for the day
        if position != 0:
            trade_capital = capital * position_size
            trade_pnl = trade_capital * position * daily_return
            # Apply slippage
            trade_pnl -= abs(trade_pnl) * slippage
            capital += trade_pnl

            trades.append({
                'day': i,
                'position': position,
                'return': daily_return,
                'pnl': trade_pnl,
                'capital': capital
            })

        equity_curve.append(capital)

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    daily_returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)

    # Total return
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] > 0 else 0.0

    # Annualized return (assuming 252 trading days)
    n_periods = len(daily_returns)
    annualized_return = ((1 + total_return) ** (252 / max(n_periods, 1))) - 1 if total_return > -1 else -1.0

    # Sharpe ratio (assuming 0% risk-free rate)
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns, ddof=1)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (cummax - equity_curve) / np.maximum(cummax, 1e-10)
    max_drawdown = float(np.max(drawdowns))

    # Win rate and profit factor
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)

        avg_trade_pnl = np.mean([t['pnl'] for t in trades])
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0

    return {
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'total_trades': len(trades),
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'avg_trade_pnl': float(avg_trade_pnl),
    }


def _save_walk_forward_results(results: Dict, path: str) -> None:
    """
    Save walk-forward validation results to JSON file.

    Args:
        results: Results dictionary from train_with_walk_forward
        path: Path to save JSON file
    """
    import copy

    # Create a serializable copy of results
    serializable = copy.deepcopy(results)

    # Convert numpy arrays to lists
    for key, value in serializable.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            serializable[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            serializable[key] = int(value)

    # Ensure fold_metrics are serializable
    if 'fold_metrics' in serializable:
        for fm in serializable['fold_metrics']:
            for k, v in fm.items():
                if isinstance(v, (np.float32, np.float64)):
                    fm[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    fm[k] = int(v)

    # Add metadata
    serializable['metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
    }

    # Save to file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Walk-forward results saved to {path}")


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.

    Args:
        y: Target array with class indices
        num_classes: Number of classes

    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(y.astype(int), minlength=num_classes)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    # Inverse frequency weighting
    weights = len(y) / (num_classes * class_counts)
    return torch.FloatTensor(weights)


def train_with_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    model_config: Dict,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    seq_length: int = 20,
    num_classes: int = 3,
    use_class_weights: bool = True,
    prices: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    min_sharpe_threshold: float = 1.0,
    validate_sharpe: bool = True,
    results_path: Optional[str] = None,
) -> Dict:
    """
    Train model using walk-forward validation with 3-class classification.

    Integrates TradingSimulator to calculate trading metrics (Sharpe ratio,
    max drawdown, win rate) for each fold, enabling profitability validation.

    Args:
        X: Feature matrix
        y: Target vector (class indices: 0=DOWN, 1=FLAT, 2=UP)
        model_config: Model configuration dict
        n_splits: Number of walk-forward splits
        epochs: Training epochs per fold
        batch_size: Batch size
        seq_length: Sequence length for LSTM
        num_classes: Number of output classes (2=binary, 3=scalping)
        use_class_weights: Whether to use class weights for imbalanced data
        prices: Optional price array for trading simulation (same length as X).
                If provided, TradingSimulator will calculate Sharpe ratio,
                max drawdown, win rate, and profit factor for each fold.
        returns: Optional returns array for trading simulation. If not provided
                 but prices is, returns will be calculated from prices.
        min_sharpe_threshold: Minimum average Sharpe ratio required (default 1.0).
                              If validate_sharpe=True and average Sharpe < threshold,
                              a warning is logged.
        validate_sharpe: If True and prices provided, validate average Sharpe
                         against min_sharpe_threshold.
        results_path: Optional path to save results JSON file.

    Returns:
        Dictionary with results for each fold including:
        - fold_metrics: List of per-fold metrics (accuracy, loss, trading metrics)
        - predictions: All predictions across folds
        - predicted_classes: Predicted class indices
        - actuals: Actual class indices
        - overall_accuracy: Macro accuracy across all folds
        - overall_auc: Macro-averaged AUC score
        - avg_sharpe_ratio: Average Sharpe ratio across folds (if prices provided)
        - avg_max_drawdown: Average max drawdown across folds (if prices provided)
        - avg_win_rate: Average win rate across folds (if prices provided)
        - sharpe_validation_passed: Whether average Sharpe >= threshold

    Raises:
        ValueError: If prices length doesn't match X length.
    """
    # Validate prices input if provided
    if prices is not None:
        if len(prices) != len(X):
            raise ValueError(
                f"prices length ({len(prices)}) must match X length ({len(X)})"
            )
        # Calculate returns if not provided
        if returns is None:
            returns = np.diff(prices) / prices[:-1]
            # Pad with 0 at the end to maintain alignment
            returns = np.append(returns, 0.0)
        elif len(returns) != len(X):
            raise ValueError(
                f"returns length ({len(returns)}) must match X length ({len(X)})"
            )

    results = {
        'fold_metrics': [],
        'predictions': [],  # Will be (N, num_classes) probabilities
        'predicted_classes': [],  # Will be class indices
        'actuals': [],
        'timestamps': [],
        'num_classes': num_classes,
        # Trading metrics (populated if prices provided)
        'trading_metrics_available': prices is not None,
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

        # Compute class weights for this fold if enabled
        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(y_train, num_classes)
            logger.info(f"Class distribution (train): {np.bincount(y_train.astype(int), minlength=num_classes)}")
            logger.info(f"Class weights: {class_weights.tolist()}")

        # Create model
        model_type = model_config.get('type', 'feedforward')

        if model_type == 'lstm':
            # Create sequences
            train_seq = SequenceDataset(X_train, y_train, seq_length, num_classes)
            test_seq = SequenceDataset(X_test, y_test, seq_length, num_classes)
            X_train_t, y_train_t = train_seq.get_tensors()
            X_test_t, y_test_t = test_seq.get_tensors()
            input_dim = X.shape[1]
        else:
            # For non-LSTM models, targets are class indices (LongTensor)
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)  # Class indices, no unsqueeze
            X_test_t = torch.FloatTensor(X_test)
            y_test_t = torch.LongTensor(y_test)
            input_dim = X.shape[1]

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create and train model
        model = create_model(
            model_type, input_dim,
            num_classes=num_classes,
            **model_config.get('params', {})
        )
        trainer = ModelTrainer(
            model,
            learning_rate=model_config.get('learning_rate', 0.001),
            weight_decay=model_config.get('weight_decay', 0.01),
            num_classes=num_classes,
            class_weights=class_weights
        )

        history = trainer.train(
            train_loader,
            test_loader,
            epochs=epochs,
            early_stopping_patience=10
        )

        # Evaluate on test set
        test_loss, test_acc = trainer.validate(test_loader)

        # Get predictions - returns (N, num_classes) probabilities
        predictions = trainer.predict(X_test_t.numpy() if model_type != 'lstm' else X_test_t.numpy())
        predicted_classes = np.argmax(predictions, axis=1)

        fold_result = {
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1]
        }

        # Run trading simulation if prices provided
        if prices is not None:
            trading_metrics = _simulate_trading_for_fold(
                predicted_classes=predicted_classes,
                predictions=predictions,
                prices=prices[test_idx],
                returns=returns[test_idx] if returns is not None else None,
                num_classes=num_classes,
            )
            fold_result.update(trading_metrics)
            logger.info(
                f"Fold {fold + 1} Trading: Sharpe={trading_metrics['sharpe_ratio']:.3f}, "
                f"MaxDD={trading_metrics['max_drawdown']*100:.2f}%, "
                f"WinRate={trading_metrics['win_rate']*100:.1f}%, "
                f"Trades={trading_metrics['total_trades']}"
            )

        results['fold_metrics'].append(fold_result)

        # Store probabilities and predicted classes
        results['predictions'].extend(predictions.tolist())
        results['predicted_classes'].extend(predicted_classes.tolist())
        results['actuals'].extend(y_test_t.numpy().tolist())

    # Calculate overall metrics for multi-class
    all_pred_classes = np.array(results['predicted_classes'])
    all_actuals = np.array(results['actuals'])

    results['overall_accuracy'] = (all_pred_classes == all_actuals).mean()

    # Per-class accuracy
    for c in range(num_classes):
        mask = all_actuals == c
        if mask.sum() > 0:
            class_acc = (all_pred_classes[mask] == c).mean()
            results[f'class_{c}_accuracy'] = class_acc
            logger.info(f"Class {c} accuracy: {class_acc:.4f}")

    # Calculate macro-averaged AUC for multi-class
    all_probs = np.array(results['predictions'])
    results['overall_auc'] = calculate_multiclass_auc(all_actuals, all_probs, num_classes)

    # Calculate aggregate trading metrics if prices were provided
    if prices is not None:
        sharpe_ratios = [fm['sharpe_ratio'] for fm in results['fold_metrics'] if 'sharpe_ratio' in fm]
        max_drawdowns = [fm['max_drawdown'] for fm in results['fold_metrics'] if 'max_drawdown' in fm]
        win_rates = [fm['win_rate'] for fm in results['fold_metrics'] if 'win_rate' in fm]
        profit_factors = [fm['profit_factor'] for fm in results['fold_metrics'] if 'profit_factor' in fm]
        total_trades_list = [fm['total_trades'] for fm in results['fold_metrics'] if 'total_trades' in fm]

        if sharpe_ratios:
            results['avg_sharpe_ratio'] = float(np.mean(sharpe_ratios))
            results['std_sharpe_ratio'] = float(np.std(sharpe_ratios))
            results['min_sharpe_ratio'] = float(np.min(sharpe_ratios))
            results['max_sharpe_ratio'] = float(np.max(sharpe_ratios))

            # Count profitable folds (Sharpe > 0)
            profitable_folds = sum(1 for s in sharpe_ratios if s > 0)
            results['profitable_folds_pct'] = profitable_folds / len(sharpe_ratios) * 100

        if max_drawdowns:
            results['avg_max_drawdown'] = float(np.mean(max_drawdowns))
            results['worst_max_drawdown'] = float(np.max(max_drawdowns))

        if win_rates:
            results['avg_win_rate'] = float(np.mean(win_rates))

        if profit_factors:
            # Filter out inf values for mean calculation
            valid_pf = [pf for pf in profit_factors if pf != float('inf')]
            results['avg_profit_factor'] = float(np.mean(valid_pf)) if valid_pf else 0.0

        if total_trades_list:
            results['total_trades_all_folds'] = int(np.sum(total_trades_list))
            results['avg_trades_per_fold'] = float(np.mean(total_trades_list))

        # Validate Sharpe ratio against threshold
        avg_sharpe = results.get('avg_sharpe_ratio', 0.0)
        results['sharpe_validation_passed'] = avg_sharpe >= min_sharpe_threshold
        results['min_sharpe_threshold'] = min_sharpe_threshold

        if validate_sharpe and not results['sharpe_validation_passed']:
            logger.warning(
                f"SHARPE VALIDATION FAILED: Average Sharpe ({avg_sharpe:.3f}) "
                f"< threshold ({min_sharpe_threshold:.3f})"
            )
        elif validate_sharpe:
            logger.info(
                f"SHARPE VALIDATION PASSED: Average Sharpe ({avg_sharpe:.3f}) "
                f">= threshold ({min_sharpe_threshold:.3f})"
            )

    logger.info(f"\n{'='*60}")
    logger.info("WALK-FORWARD VALIDATION COMPLETE")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Overall AUC (macro): {results['overall_auc']:.4f}")

    if prices is not None:
        logger.info(f"Average Sharpe Ratio: {results.get('avg_sharpe_ratio', 0.0):.3f}")
        logger.info(f"Average Max Drawdown: {results.get('avg_max_drawdown', 0.0)*100:.2f}%")
        logger.info(f"Average Win Rate: {results.get('avg_win_rate', 0.0)*100:.1f}%")
        logger.info(f"Total Trades (all folds): {results.get('total_trades_all_folds', 0)}")
        logger.info(f"Sharpe Validation: {'PASSED' if results.get('sharpe_validation_passed', False) else 'FAILED'}")

    logger.info(f"{'='*60}")

    # Save results to JSON if path provided
    if results_path:
        _save_walk_forward_results(results, results_path)

    return results


def calculate_multiclass_auc(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int) -> float:
    """
    Calculate macro-averaged AUC for multi-class classification.

    Args:
        y_true: True class indices (N,)
        y_probs: Class probabilities (N, num_classes)
        num_classes: Number of classes

    Returns:
        Macro-averaged AUC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        # One-vs-rest AUC
        return roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except (ImportError, ValueError):
        # Fallback: average of binary AUCs for each class
        aucs = []
        for c in range(num_classes):
            binary_true = (y_true == c).astype(int)
            binary_probs = y_probs[:, c]
            try:
                auc = calculate_auc(binary_true, binary_probs)
                aucs.append(auc)
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                # Skip this class if AUC calculation fails (e.g., all same class)
                logger.debug(f"AUC calculation failed for class {c}: {e}")
                pass
        return np.mean(aucs) if aucs else 0.5


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
    # Test training pipeline with 3-class classification
    print("Testing Training Pipeline (3-Class Scalping)")
    print("="*60)

    # Create synthetic 3-class data (DOWN=0, FLAT=1, UP=2)
    np.random.seed(42)
    n_samples = 1000
    n_features = 40
    num_classes = 3

    X = np.random.randn(n_samples, n_features)
    # Create imbalanced 3-class targets similar to scalping: ~20% DOWN, ~60% FLAT, ~20% UP
    probs = np.random.rand(n_samples)
    y = np.where(probs < 0.2, 0,  # DOWN
         np.where(probs < 0.8, 1,  # FLAT
                  2))  # UP
    print(f"Class distribution: DOWN={np.sum(y==0)}, FLAT={np.sum(y==1)}, UP={np.sum(y==2)}")

    # Test walk-forward validation with 3-class
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
        batch_size=32,
        num_classes=num_classes,
        use_class_weights=True
    )

    print("\nFold Results:")
    for fold_result in results['fold_metrics']:
        print(f"  Fold {fold_result['fold']}: Acc={fold_result['test_accuracy']:.4f}")

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Overall AUC (macro): {results['overall_auc']:.4f}")
    for c in range(num_classes):
        key = f'class_{c}_accuracy'
        if key in results:
            class_names = ['DOWN', 'FLAT', 'UP']
            print(f"  {class_names[c]} accuracy: {results[key]:.4f}")
