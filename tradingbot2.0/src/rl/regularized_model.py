"""
Regularized Multi-Horizon Model with Enhanced Architecture.

Improvements over base model:
1. Higher dropout rates
2. Layer normalization
3. Residual connections
4. Gradient clipping
5. Label smoothing
6. Early stopping with validation monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegularizedPrediction:
    """Prediction output with uncertainty estimates."""
    prob_up_1h: float
    prob_up_4h: float
    prob_up_eod: float
    confidence_1h: float
    confidence_4h: float
    confidence_eod: float
    uncertainty_1h: float  # Model uncertainty
    uncertainty_4h: float
    uncertainty_eod: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL observation."""
        return np.array([
            self.prob_up_1h,
            self.prob_up_4h,
            self.prob_up_eod,
            self.confidence_1h,
            self.confidence_4h,
            self.confidence_eod,
            self.uncertainty_1h,
            self.uncertainty_4h,
            self.uncertainty_eod,
        ], dtype=np.float32)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""

    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.layers(x))


class RegularizedMultiHorizonNet(nn.Module):
    """
    Regularized multi-horizon prediction network.

    Key regularization techniques:
    - Higher dropout (0.4 vs 0.3)
    - Layer normalization (more stable than batch norm)
    - Residual connections (better gradient flow)
    - MC Dropout for uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.4,
        num_residual_blocks: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks for shared representation
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout_rate)
            for _ in range(num_residual_blocks)
        ])

        # Dimension reduction
        self.dim_reduce = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Horizon-specific heads with uncertainty
        self.head_1h = self._make_head(hidden_dims[2], dropout_rate)
        self.head_4h = self._make_head(hidden_dims[2], dropout_rate)
        self.head_eod = self._make_head(hidden_dims[2], dropout_rate)

        # Initialize weights
        self._init_weights()

    def _make_head(self, input_dim: int, dropout_rate: float) -> nn.Module:
        """Create prediction head."""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def _init_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits for each horizon."""
        # Input projection
        h = self.input_proj(x)

        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # Dimension reduction
        h = self.dim_reduce(h)

        # Horizon-specific predictions
        logits_1h = self.head_1h(h)
        logits_4h = self.head_4h(h)
        logits_eod = self.head_eod(h)

        return logits_1h, logits_4h, logits_eod

    def predict_with_uncertainty(
        self,
        x: np.ndarray,
        n_samples: int = 10,
    ) -> RegularizedPrediction:
        """
        Make prediction with MC Dropout uncertainty estimation.

        Uses multiple forward passes with dropout enabled to estimate
        model uncertainty.
        """
        self.train()  # Enable dropout

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()

            # Collect multiple predictions
            probs_1h, probs_4h, probs_eod = [], [], []

            for _ in range(n_samples):
                logits_1h, logits_4h, logits_eod = self.forward(x_tensor)
                probs_1h.append(torch.sigmoid(logits_1h).item())
                probs_4h.append(torch.sigmoid(logits_4h).item())
                probs_eod.append(torch.sigmoid(logits_eod).item())

            # Compute mean and std
            p1h_mean, p1h_std = np.mean(probs_1h), np.std(probs_1h)
            p4h_mean, p4h_std = np.mean(probs_4h), np.std(probs_4h)
            peod_mean, peod_std = np.mean(probs_eod), np.std(probs_eod)

            # Confidence = distance from 0.5
            conf_1h = abs(p1h_mean - 0.5) * 2
            conf_4h = abs(p4h_mean - 0.5) * 2
            conf_eod = abs(peod_mean - 0.5) * 2

            return RegularizedPrediction(
                prob_up_1h=p1h_mean,
                prob_up_4h=p4h_mean,
                prob_up_eod=peod_mean,
                confidence_1h=conf_1h,
                confidence_4h=conf_4h,
                confidence_eod=conf_eod,
                uncertainty_1h=p1h_std,
                uncertainty_4h=p4h_std,
                uncertainty_eod=peod_std,
            )


class LabelSmoothingBCELoss(nn.Module):
    """BCE loss with label smoothing for regularization."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth targets: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


class RegularizedTrainer:
    """
    Trainer with enhanced regularization and early stopping.
    """

    def __init__(
        self,
        model: RegularizedMultiHorizonNet,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.1,
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

        # AdamW with higher weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Label smoothing loss
        self.criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)

        self.best_val_loss = float('inf')
        self.best_state = None
        self.patience_counter = 0

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            X, y_1h, y_4h, y_eod = batch
            X = X.to(self.device)
            y_1h = y_1h.to(self.device).float()
            y_4h = y_4h.to(self.device).float()
            y_eod = y_eod.to(self.device).float()

            self.optimizer.zero_grad()

            logits_1h, logits_4h, logits_eod = self.model(X)

            loss_1h = self.criterion(logits_1h.squeeze(), y_1h)
            loss_4h = self.criterion(logits_4h.squeeze(), y_4h)
            loss_eod = self.criterion(logits_eod.squeeze(), y_eod)

            loss = loss_1h + loss_4h + loss_eod

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()

        return {'total': total_loss / n_batches}

    def validate(self, val_loader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = {'1h': [], '4h': [], 'eod': []}
        all_targets = {'1h': [], '4h': [], 'eod': []}

        with torch.no_grad():
            for batch in val_loader:
                X, y_1h, y_4h, y_eod = batch
                X = X.to(self.device)
                y_1h = y_1h.to(self.device).float()
                y_4h = y_4h.to(self.device).float()
                y_eod = y_eod.to(self.device).float()

                logits_1h, logits_4h, logits_eod = self.model(X)

                loss_1h = self.criterion(logits_1h.squeeze(), y_1h)
                loss_4h = self.criterion(logits_4h.squeeze(), y_4h)
                loss_eod = self.criterion(logits_eod.squeeze(), y_eod)

                total_loss += (loss_1h + loss_4h + loss_eod).item()

                all_preds['1h'].extend((torch.sigmoid(logits_1h) > 0.5).cpu().numpy().flatten())
                all_preds['4h'].extend((torch.sigmoid(logits_4h) > 0.5).cpu().numpy().flatten())
                all_preds['eod'].extend((torch.sigmoid(logits_eod) > 0.5).cpu().numpy().flatten())
                all_targets['1h'].extend(y_1h.cpu().numpy().flatten())
                all_targets['4h'].extend(y_4h.cpu().numpy().flatten())
                all_targets['eod'].extend(y_eod.cpu().numpy().flatten())

        n_batches = len(val_loader)
        avg_loss = total_loss / n_batches

        accuracies = {}
        for horizon in ['1h', '4h', 'eod']:
            preds = np.array(all_preds[horizon])
            targets = np.array(all_targets[horizon])
            accuracies[f'acc_{horizon}'] = (preds == targets).mean()

        return {'total': avg_loss}, accuracies

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        patience: int = 15,
        min_delta: float = 0.001,
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum improvement to reset patience counter
        """
        history = {
            'train_loss': [], 'val_loss': [],
            'acc_1h': [], 'acc_4h': [], 'acc_eod': []
        }

        for epoch in range(epochs):
            train_losses = self.train_epoch(train_loader)
            val_losses, val_accs = self.validate(val_loader)

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

            # Early stopping check
            if val_losses['total'] < self.best_val_loss - min_delta:
                self.best_val_loss = val_losses['total']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                logger.info(f"  -> New best model (val_loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        # Restore best model
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            logger.info(f"Restored best model with val_loss: {self.best_val_loss:.4f}")

        return history
