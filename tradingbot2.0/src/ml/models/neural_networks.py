"""
Neural Network Models for Futures Price Direction Prediction.

This module provides:
1. FeedForwardNet: Simple MLP for tabular features
2. LSTMNet: LSTM for sequential/temporal patterns
3. Combined model with both architectures

Best Practices Implemented:
- Dropout for regularization
- Batch normalization for training stability
- Weight initialization (Xavier/He)
- Early stopping via training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class FeedForwardNet(nn.Module):
    """
    Multi-Layer Perceptron for tabular feature classification.

    Architecture:
    - Input layer -> Hidden layers with BatchNorm, ReLU, Dropout
    - Output: Probability of price going UP (binary classification)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the feed-forward network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (before activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout (after activation)
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (binary classification)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class LSTMNet(nn.Module):
    """
    LSTM Network for sequential pattern recognition.

    Takes a sequence of features over time and predicts next-day direction.
    Good for capturing temporal dependencies in market data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        fc_dims: List[int] = [32]
    ):
        """
        Initialize LSTM network.

        Args:
            input_dim: Number of features per timestep
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            fc_dims: Fully connected layer dimensions after LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers after LSTM
        fc_input_dim = hidden_dim * self.num_directions
        fc_layers = []

        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(fc_input_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            fc_input_dim = fc_dim

        self.fc_layers = nn.Sequential(*fc_layers)

        # Output layer
        self.output_layer = nn.Linear(fc_dims[-1] if fc_dims else hidden_dim * self.num_directions, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output probabilities, final hidden state)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Take only the last timestep output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        fc_out = self.fc_layers(last_output)

        # Output
        output = torch.sigmoid(self.output_layer(fc_out))

        return output, hidden


class HybridNet(nn.Module):
    """
    Hybrid model combining LSTM for temporal patterns and
    MLP for point-in-time features.

    This architecture allows capturing both sequential dependencies
    and static feature interactions.
    """

    def __init__(
        self,
        seq_input_dim: int,
        static_input_dim: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        mlp_hidden: List[int] = [64, 32],
        dropout_rate: float = 0.3
    ):
        """
        Initialize hybrid network.

        Args:
            seq_input_dim: Features per timestep for LSTM
            static_input_dim: Number of point-in-time features for MLP
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            mlp_hidden: MLP hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()

        # LSTM branch for sequential features
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # MLP branch for static features
        mlp_layers = []
        prev_dim = static_input_dim
        for hidden_dim in mlp_hidden:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Combined output layer
        combined_dim = lstm_hidden + mlp_hidden[-1]
        self.output_layers = nn.Sequential(
            nn.Linear(combined_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )

    def forward(
        self,
        seq_x: torch.Tensor,
        static_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq_x: Sequential features (batch_size, seq_length, seq_input_dim)
            static_x: Static features (batch_size, static_input_dim)

        Returns:
            Probability tensor (batch_size, 1)
        """
        # LSTM branch
        lstm_out, _ = self.lstm(seq_x)
        lstm_features = lstm_out[:, -1, :]  # Last timestep

        # MLP branch
        mlp_features = self.mlp(static_x)

        # Combine and predict
        combined = torch.cat([lstm_features, mlp_features], dim=1)
        output = torch.sigmoid(self.output_layers(combined))

        return output


def create_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'feedforward', 'lstm', or 'hybrid'
        input_dim: Input feature dimension
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model
    """
    if model_type.lower() == 'feedforward':
        return FeedForwardNet(input_dim, **kwargs)
    elif model_type.lower() == 'lstm':
        return LSTMNet(input_dim, **kwargs)
    elif model_type.lower() == 'hybrid':
        static_dim = kwargs.pop('static_input_dim', input_dim // 2)
        return HybridNet(input_dim, static_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training when validation loss doesn't improve for 'patience' epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: Model to save/restore weights

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

        return self.should_stop


if __name__ == "__main__":
    # Test models
    print("Testing Neural Network Models")
    print("="*60)

    # Test FeedForwardNet
    batch_size = 32
    input_dim = 40  # Number of features

    model = FeedForwardNet(input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"FeedForwardNet output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test LSTMNet
    seq_length = 20
    model = LSTMNet(input_dim, hidden_dim=64, num_layers=2)
    x = torch.randn(batch_size, seq_length, input_dim)
    output, hidden = model(x)
    print(f"\nLSTMNet output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test HybridNet
    model = HybridNet(seq_input_dim=input_dim, static_input_dim=20)
    seq_x = torch.randn(batch_size, seq_length, input_dim)
    static_x = torch.randn(batch_size, 20)
    output = model(seq_x, static_x)
    print(f"\nHybridNet output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nAll models working correctly!")
