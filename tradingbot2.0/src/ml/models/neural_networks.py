"""
Neural Network Models for Futures Price Direction Prediction.

This module provides:
1. FeedForwardNet: Simple MLP for tabular features
2. LSTMNet: LSTM for sequential/temporal patterns
3. HybridNet: Combined model with both architectures
4. TransformerNet: Transformer encoder for attention-based sequence modeling
5. ModelPrediction: Structured output for inference

Supports both binary classification (UP/DOWN) and 3-class classification
(DOWN/FLAT/UP) for scalping applications.

Best Practices Implemented:
- Dropout for regularization
- Batch normalization for training stability
- Weight initialization (Xavier/He)
- Early stopping via training loop
- 3-class softmax output for scalping (configurable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class ModelPrediction:
    """
    Structured output for model inference.

    Per spec (ml-scalping-model.md), model must return:
    - direction: -1 (short), 0 (flat), 1 (long)
    - confidence: 0-1, max of softmax probabilities
    - predicted_move: expected ticks (derived from class probabilities)
    - volatility: for position sizing (optional, from ATR or auxiliary head)
    - timestamp: when prediction was made
    """
    direction: int  # -1 (short), 0 (flat), 1 (long)
    confidence: float  # 0-1, max of softmax probabilities
    predicted_move: float  # expected price movement in ticks
    volatility: float  # expected volatility for position sizing
    timestamp: datetime
    class_probabilities: Optional[Tuple[float, float, float]] = None  # (DOWN, FLAT, UP)

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        volatility: float = 0.0,
        timestamp: Optional[datetime] = None,
        tick_expectations: Tuple[float, float, float] = (-4.0, 0.0, 4.0)
    ) -> 'ModelPrediction':
        """
        Create ModelPrediction from raw model logits.

        Args:
            logits: Raw model output tensor of shape (num_classes,) or (1, num_classes)
            volatility: Expected volatility for position sizing
            timestamp: When prediction was made (defaults to now)
            tick_expectations: Expected tick move for each class (DOWN, FLAT, UP)

        Returns:
            ModelPrediction instance
        """
        if logits.dim() > 1:
            logits = logits.squeeze(0)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=0)
        probs_list = probs.tolist()

        # Determine direction from predicted class
        predicted_class = torch.argmax(probs).item()
        # Map: 0 (DOWN) -> -1, 1 (FLAT) -> 0, 2 (UP) -> 1
        direction_map = {0: -1, 1: 0, 2: 1}
        direction = direction_map.get(predicted_class, 0)

        # Confidence is the max probability
        confidence = probs.max().item()

        # Expected move is probability-weighted average of tick expectations
        predicted_move = sum(p * t for p, t in zip(probs_list, tick_expectations))

        return cls(
            direction=direction,
            confidence=confidence,
            predicted_move=predicted_move,
            volatility=volatility,
            timestamp=timestamp or datetime.now(),
            class_probabilities=tuple(probs_list) if len(probs_list) == 3 else None
        )


class FeedForwardNet(nn.Module):
    """
    Multi-Layer Perceptron for tabular feature classification.

    Architecture:
    - Input layer -> Hidden layers with BatchNorm, ReLU, Dropout
    - Output: Raw logits for num_classes (2 for binary, 3 for scalping)

    For training: Returns raw logits (CrossEntropyLoss applies log_softmax internally)
    For inference: Use get_probabilities() or ModelPrediction.from_logits()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        num_classes: int = 3
    ):
        """
        Initialize the feed-forward network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
            num_classes: Number of output classes (2=binary, 3=scalping with FLAT)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.num_classes = num_classes

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

        # Output layer (num_classes outputs for classification)
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)

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
        Forward pass - returns raw logits.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            Note: For training with CrossEntropyLoss, use raw logits directly.
                  For inference, use get_probabilities() or softmax manually.
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x  # Raw logits - no activation

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (for inference).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor, volatility: float = 0.0) -> List[ModelPrediction]:
        """
        Make structured predictions for inference.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            volatility: ATR or other volatility measure for position sizing

        Returns:
            List of ModelPrediction instances, one per sample
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = []
            for i in range(logits.shape[0]):
                pred = ModelPrediction.from_logits(logits[i], volatility=volatility)
                predictions.append(pred)
            return predictions


class LSTMNet(nn.Module):
    """
    LSTM Network for sequential pattern recognition.

    Takes a sequence of features over time and predicts price direction.
    Good for capturing temporal dependencies in market data.

    For training: Returns raw logits (CrossEntropyLoss applies log_softmax internally)
    For inference: Use get_probabilities() or ModelPrediction.from_logits()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        fc_dims: List[int] = [32],
        num_classes: int = 3
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
            num_classes: Number of output classes (2=binary, 3=scalping with FLAT)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_classes = num_classes

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

        # Output layer (num_classes outputs)
        self.output_layer = nn.Linear(fc_dims[-1] if fc_dims else hidden_dim * self.num_directions, num_classes)

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
        Forward pass - returns raw logits.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (logits, final hidden state)
            logits shape: (batch_size, num_classes)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Take only the last timestep output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        fc_out = self.fc_layers(last_output)

        # Output - raw logits, no activation
        output = self.output_layer(fc_out)

        return output, hidden

    def get_probabilities(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get class probabilities (for inference).

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (probabilities, final hidden state)
        """
        logits, hidden = self.forward(x, hidden)
        return F.softmax(logits, dim=1), hidden

    def predict(
        self,
        x: torch.Tensor,
        volatility: float = 0.0,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[List[ModelPrediction], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make structured predictions for inference.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            volatility: ATR or other volatility measure for position sizing
            hidden: Optional initial hidden state

        Returns:
            Tuple of (list of ModelPrediction instances, final hidden state)
        """
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(x, hidden)
            predictions = []
            for i in range(logits.shape[0]):
                pred = ModelPrediction.from_logits(logits[i], volatility=volatility)
                predictions.append(pred)
            return predictions, hidden


class HybridNet(nn.Module):
    """
    Hybrid model combining LSTM for temporal patterns and
    MLP for point-in-time features.

    This architecture allows capturing both sequential dependencies
    and static feature interactions.

    For training: Returns raw logits (CrossEntropyLoss applies log_softmax internally)
    For inference: Use get_probabilities() or ModelPrediction.from_logits()
    """

    def __init__(
        self,
        seq_input_dim: int,
        static_input_dim: int,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        mlp_hidden: List[int] = [64, 32],
        dropout_rate: float = 0.3,
        num_classes: int = 3
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
            num_classes: Number of output classes (2=binary, 3=scalping with FLAT)
        """
        super().__init__()

        self.num_classes = num_classes

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

        # Combined output layer (num_classes outputs)
        combined_dim = lstm_hidden + mlp_hidden[-1]
        self.output_layers = nn.Sequential(
            nn.Linear(combined_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # num_classes outputs
        )

    def forward(
        self,
        seq_x: torch.Tensor,
        static_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass - returns raw logits.

        Args:
            seq_x: Sequential features (batch_size, seq_length, seq_input_dim)
            static_x: Static features (batch_size, static_input_dim)

        Returns:
            Logits tensor (batch_size, num_classes)
        """
        # LSTM branch
        lstm_out, _ = self.lstm(seq_x)
        lstm_features = lstm_out[:, -1, :]  # Last timestep

        # MLP branch
        mlp_features = self.mlp(static_x)

        # Combine and predict - raw logits, no activation
        combined = torch.cat([lstm_features, mlp_features], dim=1)
        output = self.output_layers(combined)

        return output

    def get_probabilities(
        self,
        seq_x: torch.Tensor,
        static_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get class probabilities (for inference).

        Args:
            seq_x: Sequential features (batch_size, seq_length, seq_input_dim)
            static_x: Static features (batch_size, static_input_dim)

        Returns:
            Probability tensor (batch_size, num_classes)
        """
        logits = self.forward(seq_x, static_x)
        return F.softmax(logits, dim=1)

    def predict(
        self,
        seq_x: torch.Tensor,
        static_x: torch.Tensor,
        volatility: float = 0.0
    ) -> List[ModelPrediction]:
        """
        Make structured predictions for inference.

        Args:
            seq_x: Sequential features (batch_size, seq_length, seq_input_dim)
            static_x: Static features (batch_size, static_input_dim)
            volatility: ATR or other volatility measure for position sizing

        Returns:
            List of ModelPrediction instances, one per sample
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(seq_x, static_x)
            predictions = []
            for i in range(logits.shape[0]):
                pred = ModelPrediction.from_logits(logits[i], volatility=volatility)
                predictions.append(pred)
            return predictions


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies, as described in "Attention is All You Need".

    This allows the Transformer to understand sequence order since attention
    is permutation-invariant without positional information.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding/model dimension
            max_len: Maximum sequence length to support
            dropout: Dropout probability applied after adding positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """
    Transformer Encoder for sequential pattern recognition.

    Uses self-attention mechanism to capture global dependencies in the
    input sequence. Particularly effective for:
    - Capturing long-range dependencies in time series
    - Parallel processing (unlike RNNs)
    - Learning complex feature interactions

    Architecture (per ml-scalping-model.md spec):
        Input (sequence) → Positional Encoding
                        → Multi-Head Attention × N layers
                        → Dense(num_classes) → Softmax

    For training: Returns raw logits (CrossEntropyLoss applies log_softmax internally)
    For inference: Use get_probabilities() or ModelPrediction.from_logits()
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
        max_seq_len: int = 500,
        num_classes: int = 3,
        pooling: str = 'last'
    ):
        """
        Initialize Transformer encoder.

        Args:
            input_dim: Number of features per timestep
            d_model: Transformer model dimension (embedding size).
                     Must be divisible by nhead.
            nhead: Number of attention heads. More heads allow the model
                   to attend to different parts of the sequence simultaneously.
            num_layers: Number of transformer encoder layers. More layers
                        allow for deeper feature extraction.
            dim_feedforward: Dimension of the feedforward network in each
                            transformer layer (typically 2-4x d_model).
            dropout_rate: Dropout probability for regularization.
            max_seq_len: Maximum sequence length for positional encoding.
            num_classes: Number of output classes (3 for UP/FLAT/DOWN).
            pooling: How to aggregate sequence output ('last', 'mean', 'cls').
                    - 'last': Use last timestep output
                    - 'mean': Average all timestep outputs
                    - 'cls': Use [CLS] token (prepended to sequence)
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pooling = pooling
        self.use_cls_token = (pooling == 'cls')

        # Input projection: map features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Optional CLS token for classification (like BERT)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout_rate)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,  # Input shape: (batch, seq, feature)
            activation='gelu'  # GELU activation (better for transformers)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Final layer normalization
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - returns raw logits.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            src_key_padding_mask: Optional mask for padded sequences.
                                  Shape: (batch_size, seq_length)
                                  True values indicate positions to ignore.

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)

        # Add CLS token if using 'cls' pooling
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)

            # Extend padding mask for CLS token
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pooling to get single representation per sequence
        if self.pooling == 'last':
            # Use last timestep (or last non-padded if using mask)
            x = x[:, -1, :]
        elif self.pooling == 'mean':
            # Average over sequence dimension
            if src_key_padding_mask is not None:
                # Mask out padded positions before averaging
                mask = ~src_key_padding_mask.unsqueeze(-1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'cls':
            # Use CLS token output (first position)
            x = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Output layer - raw logits, no activation
        x = self.fc(x)

        return x

    def get_probabilities(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get class probabilities (for inference).

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            src_key_padding_mask: Optional mask for padded sequences

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x, src_key_padding_mask)
        return F.softmax(logits, dim=1)

    def predict(
        self,
        x: torch.Tensor,
        volatility: float = 0.0,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> List[ModelPrediction]:
        """
        Make structured predictions for inference.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            volatility: ATR or other volatility measure for position sizing
            src_key_padding_mask: Optional mask for padded sequences

        Returns:
            List of ModelPrediction instances, one per sample
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, src_key_padding_mask)
            predictions = []
            for i in range(logits.shape[0]):
                pred = ModelPrediction.from_logits(logits[i], volatility=volatility)
                predictions.append(pred)
            return predictions


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'feedforward', 'lstm', 'hybrid', or 'transformer'
        input_dim: Input feature dimension
        num_classes: Number of output classes (2=binary, 3=scalping with FLAT)
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model

    Example:
        # FeedForward for tabular features
        model = create_model('feedforward', 40, num_classes=3)

        # LSTM for sequences
        model = create_model('lstm', 40, hidden_dim=64, num_layers=2)

        # Transformer for attention-based sequence modeling
        model = create_model('transformer', 40, d_model=64, nhead=4, num_layers=2)
    """
    if model_type.lower() == 'feedforward':
        return FeedForwardNet(input_dim, num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'lstm':
        return LSTMNet(input_dim, num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'hybrid':
        static_dim = kwargs.pop('static_input_dim', input_dim // 2)
        return HybridNet(input_dim, static_dim, num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'transformer':
        return TransformerNet(input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported: 'feedforward', 'lstm', 'hybrid', 'transformer'")


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
    # Test models with 3-class output for scalping
    print("Testing Neural Network Models (3-Class Scalping)")
    print("="*60)

    # Test FeedForwardNet
    batch_size = 32
    input_dim = 40  # Number of features
    num_classes = 3  # DOWN, FLAT, UP

    model = FeedForwardNet(input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"FeedForwardNet output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {num_classes})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test probabilities
    probs = model.get_probabilities(x)
    print(f"  Probabilities sum: {probs.sum(dim=1).mean():.4f} (should be ~1.0)")

    # Test ModelPrediction
    predictions = model.predict(x[:2])
    for i, pred in enumerate(predictions):
        print(f"  Prediction {i}: direction={pred.direction}, confidence={pred.confidence:.3f}")

    # Test LSTMNet
    seq_length = 20
    model = LSTMNet(input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes)
    x = torch.randn(batch_size, seq_length, input_dim)
    output, hidden = model(x)
    print(f"\nLSTMNet output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {num_classes})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test HybridNet
    model = HybridNet(seq_input_dim=input_dim, static_input_dim=20, num_classes=num_classes)
    seq_x = torch.randn(batch_size, seq_length, input_dim)
    static_x = torch.randn(batch_size, 20)
    output = model(seq_x, static_x)
    print(f"\nHybridNet output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {num_classes})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test TransformerNet
    model = TransformerNet(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=num_classes
    )
    x = torch.randn(batch_size, seq_length, input_dim)
    output = model(x)
    print(f"\nTransformerNet output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {num_classes})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test TransformerNet with different pooling methods
    for pooling in ['last', 'mean', 'cls']:
        model = TransformerNet(input_dim=input_dim, d_model=64, nhead=4, pooling=pooling)
        output = model(x)
        print(f"  Pooling '{pooling}': output shape {output.shape}")

    # Test probabilities
    probs = model.get_probabilities(x)
    print(f"  Probabilities sum: {probs.sum(dim=1).mean():.4f} (should be ~1.0)")

    # Test create_model factory
    print("\n" + "="*60)
    print("Testing create_model factory:")
    for model_type in ['feedforward', 'lstm', 'hybrid', 'transformer']:
        model = create_model(model_type, input_dim, num_classes=3)
        print(f"  {model_type}: num_classes={model.num_classes}")

    # Test ModelPrediction from_logits
    print("\n" + "="*60)
    print("Testing ModelPrediction.from_logits:")
    logits = torch.tensor([[-2.0, 0.5, 1.0]])  # Should predict UP
    pred = ModelPrediction.from_logits(logits)
    print(f"  Logits: {logits.tolist()}")
    print(f"  Direction: {pred.direction} (1=UP)")
    print(f"  Confidence: {pred.confidence:.3f}")
    print(f"  Predicted move: {pred.predicted_move:.2f} ticks")
    print(f"  Class probs: {pred.class_probabilities}")

    print("\nAll models working correctly with 3-class output!")
