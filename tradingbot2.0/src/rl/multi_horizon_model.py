"""
Multi-Horizon Direction Prediction Model.

This model predicts price direction at multiple time horizons:
- 1 hour ahead
- 4 hours ahead
- End of day (EOD)

Architecture:
- Shared feature extraction layers
- Separate prediction heads for each horizon
- Multi-task learning with weighted loss

The model outputs probabilities for UP direction at each horizon,
which are then used as features for the RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HorizonPrediction:
    """Prediction output for all horizons."""
    prob_up_1h: float
    prob_up_4h: float
    prob_up_eod: float
    confidence_1h: float
    confidence_4h: float
    confidence_eod: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL observation."""
        return np.array([
            self.prob_up_1h,
            self.prob_up_4h,
            self.prob_up_eod,
            self.confidence_1h,
            self.confidence_4h,
            self.confidence_eod,
        ], dtype=np.float32)


class MultiHorizonNet(nn.Module):
    """
    Multi-task neural network for multi-horizon direction prediction.

    Architecture:
    - Shared encoder: Extract common features
    - Horizon-specific heads: Separate predictions for 1h, 4h, EOD

    This allows the model to learn shared representations while
    specializing for each prediction horizon.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Horizon-specific heads
        head_input_dim = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
        head_hidden_dim = hidden_dims[-1]

        # 1-hour prediction head
        self.head_1h = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, 1),  # Binary output (prob of UP)
        )

        # 4-hour prediction head
        self.head_4h = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, 1),
        )

        # EOD prediction head
        self.head_eod = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (logits_1h, logits_4h, logits_eod), each of shape (batch_size, 1)
        """
        # Shared encoding
        features = self.encoder(x)

        # Horizon-specific predictions
        logits_1h = self.head_1h(features)
        logits_4h = self.head_4h(features)
        logits_eod = self.head_eod(features)

        return logits_1h, logits_4h, logits_eod

    def predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get probability predictions."""
        logits_1h, logits_4h, logits_eod = self.forward(x)
        return (
            torch.sigmoid(logits_1h),
            torch.sigmoid(logits_4h),
            torch.sigmoid(logits_eod),
        )

    def predict(self, x: np.ndarray) -> HorizonPrediction:
        """
        Make prediction for a single sample.

        Args:
            x: Feature array of shape (input_dim,)

        Returns:
            HorizonPrediction with probabilities and confidence
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()

            prob_1h, prob_4h, prob_eod = self.predict_proba(x_tensor)

            # Convert to numpy
            p1h = prob_1h.squeeze().cpu().item()
            p4h = prob_4h.squeeze().cpu().item()
            peod = prob_eod.squeeze().cpu().item()

            # Confidence = distance from 0.5 (how certain the model is)
            conf_1h = abs(p1h - 0.5) * 2  # Scale to [0, 1]
            conf_4h = abs(p4h - 0.5) * 2
            conf_eod = abs(peod - 0.5) * 2

            return HorizonPrediction(
                prob_up_1h=p1h,
                prob_up_4h=p4h,
                prob_up_eod=peod,
                confidence_1h=conf_1h,
                confidence_4h=conf_4h,
                confidence_eod=conf_eod,
            )


class MultiHorizonLoss(nn.Module):
    """
    Multi-task loss for multi-horizon prediction.

    Combines BCE loss for each horizon with optional weighting.
    """

    def __init__(
        self,
        weight_1h: float = 1.0,
        weight_4h: float = 1.0,
        weight_eod: float = 1.0,
        pos_weight_1h: Optional[float] = None,
        pos_weight_4h: Optional[float] = None,
        pos_weight_eod: Optional[float] = None,
    ):
        super().__init__()

        self.weight_1h = weight_1h
        self.weight_4h = weight_4h
        self.weight_eod = weight_eod

        # Create loss functions with optional class weighting
        self.loss_1h = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_1h]) if pos_weight_1h else None
        )
        self.loss_4h = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_4h]) if pos_weight_4h else None
        )
        self.loss_eod = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_eod]) if pos_weight_eod else None
        )

    def forward(
        self,
        logits_1h: torch.Tensor,
        logits_4h: torch.Tensor,
        logits_eod: torch.Tensor,
        target_1h: torch.Tensor,
        target_4h: torch.Tensor,
        target_eod: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Returns:
            (total_loss, loss_dict with individual losses)
        """
        loss_1h = self.loss_1h(logits_1h.squeeze(), target_1h.float())
        loss_4h = self.loss_4h(logits_4h.squeeze(), target_4h.float())
        loss_eod = self.loss_eod(logits_eod.squeeze(), target_eod.float())

        total_loss = (
            self.weight_1h * loss_1h +
            self.weight_4h * loss_4h +
            self.weight_eod * loss_eod
        )

        loss_dict = {
            'loss_1h': loss_1h.item(),
            'loss_4h': loss_4h.item(),
            'loss_eod': loss_eod.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict


class MultiHorizonTrainer:
    """
    Trainer for multi-horizon prediction model.
    """

    def __init__(
        self,
        model: MultiHorizonNet,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = MultiHorizonLoss()

        self.best_val_loss = float('inf')
        self.best_state = None

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {'loss_1h': 0, 'loss_4h': 0, 'loss_eod': 0, 'total': 0}
        n_batches = 0

        for batch in train_loader:
            X, y_1h, y_4h, y_eod = batch
            X = X.to(self.device)
            y_1h = y_1h.to(self.device)
            y_4h = y_4h.to(self.device)
            y_eod = y_eod.to(self.device)

            self.optimizer.zero_grad()

            logits_1h, logits_4h, logits_eod = self.model(X)
            loss, loss_dict = self.criterion(
                logits_1h, logits_4h, logits_eod,
                y_1h, y_4h, y_eod
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for key in total_losses:
                total_losses[key] += loss_dict[key]
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    def validate(self, val_loader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_losses = {'loss_1h': 0, 'loss_4h': 0, 'loss_eod': 0, 'total': 0}

        all_preds = {'1h': [], '4h': [], 'eod': []}
        all_targets = {'1h': [], '4h': [], 'eod': []}

        with torch.no_grad():
            for batch in val_loader:
                X, y_1h, y_4h, y_eod = batch
                X = X.to(self.device)
                y_1h = y_1h.to(self.device)
                y_4h = y_4h.to(self.device)
                y_eod = y_eod.to(self.device)

                logits_1h, logits_4h, logits_eod = self.model(X)
                _, loss_dict = self.criterion(
                    logits_1h, logits_4h, logits_eod,
                    y_1h, y_4h, y_eod
                )

                for key in total_losses:
                    total_losses[key] += loss_dict[key]

                # Collect predictions for accuracy
                all_preds['1h'].extend((torch.sigmoid(logits_1h) > 0.5).cpu().numpy().flatten())
                all_preds['4h'].extend((torch.sigmoid(logits_4h) > 0.5).cpu().numpy().flatten())
                all_preds['eod'].extend((torch.sigmoid(logits_eod) > 0.5).cpu().numpy().flatten())
                all_targets['1h'].extend(y_1h.cpu().numpy().flatten())
                all_targets['4h'].extend(y_4h.cpu().numpy().flatten())
                all_targets['eod'].extend(y_eod.cpu().numpy().flatten())

        n_batches = len(val_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Calculate accuracies
        accuracies = {}
        for horizon in ['1h', '4h', 'eod']:
            preds = np.array(all_preds[horizon])
            targets = np.array(all_targets[horizon])
            accuracies[f'acc_{horizon}'] = (preds == targets).mean()

        return avg_losses, accuracies

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        patience: int = 10,
    ) -> Dict:
        """Full training loop."""
        history = {
            'train_loss': [], 'val_loss': [],
            'acc_1h': [], 'acc_4h': [], 'acc_eod': []
        }

        wait = 0
        for epoch in range(epochs):
            train_losses = self.train_epoch(train_loader)
            val_losses, val_accs = self.validate(val_loader)

            self.scheduler.step(val_losses['total'])

            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            for key in val_accs:
                history[key].append(val_accs[key])

            logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Train: {train_losses['total']:.4f} | "
                f"Val: {val_losses['total']:.4f} | "
                f"Acc 1h: {val_accs['acc_1h']:.3f} | "
                f"4h: {val_accs['acc_4h']:.3f} | "
                f"EOD: {val_accs['acc_eod']:.3f}"
            )

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        return history


def create_multi_horizon_targets(
    df: pd.DataFrame,
    horizon_1h_bars: int = 60,
    horizon_4h_bars: int = 240,
) -> pd.DataFrame:
    """
    Create multi-horizon target variables.

    Args:
        df: DataFrame with 1-minute OHLCV data
        horizon_1h_bars: Number of bars for 1-hour horizon (default 60)
        horizon_4h_bars: Number of bars for 4-hour horizon (default 240)

    Returns:
        DataFrame with target columns: target_1h, target_4h, target_eod
    """
    df = df.copy()

    # 1-hour future return (binary: 1 if UP, 0 if DOWN)
    future_1h = df['close'].shift(-horizon_1h_bars)
    df['target_1h'] = (future_1h > df['close']).astype(int)

    # 4-hour future return
    future_4h = df['close'].shift(-horizon_4h_bars)
    df['target_4h'] = (future_4h > df['close']).astype(int)

    # EOD return (from current bar to end of day)
    df['eod_close'] = df.groupby(df.index.date)['close'].transform('last')
    df['target_eod'] = (df['eod_close'] > df['close']).astype(int)

    return df
