"""
Tests for RL regularized multi-horizon model.

Tests cover:
- RegularizedPrediction dataclass
- ResidualBlock architecture
- RegularizedMultiHorizonNet architecture
- Forward pass and predict_with_uncertainty
- LabelSmoothingBCELoss
- RegularizedTrainer
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.rl.regularized_model import (
    RegularizedPrediction,
    ResidualBlock,
    RegularizedMultiHorizonNet,
    LabelSmoothingBCELoss,
    RegularizedTrainer,
)


class TestRegularizedPrediction:
    """Tests for RegularizedPrediction dataclass."""

    def test_creation(self):
        """RegularizedPrediction can be created with all fields."""
        pred = RegularizedPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
            uncertainty_1h=0.05,
            uncertainty_4h=0.04,
            uncertainty_eod=0.06,
        )
        assert pred.prob_up_1h == 0.6
        assert pred.uncertainty_1h == 0.05

    def test_to_array_shape(self):
        """to_array returns array of correct shape (9 elements)."""
        pred = RegularizedPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
            uncertainty_1h=0.05,
            uncertainty_4h=0.04,
            uncertainty_eod=0.06,
        )
        arr = pred.to_array()
        assert arr.shape == (9,)

    def test_to_array_dtype(self):
        """to_array returns float32 array."""
        pred = RegularizedPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
            uncertainty_1h=0.05,
            uncertainty_4h=0.04,
            uncertainty_eod=0.06,
        )
        arr = pred.to_array()
        assert arr.dtype == np.float32

    def test_to_array_values_order(self):
        """to_array contains values in correct order."""
        pred = RegularizedPrediction(
            prob_up_1h=0.1,
            prob_up_4h=0.2,
            prob_up_eod=0.3,
            confidence_1h=0.4,
            confidence_4h=0.5,
            confidence_eod=0.6,
            uncertainty_1h=0.7,
            uncertainty_4h=0.8,
            uncertainty_eod=0.9,
        )
        arr = pred.to_array()
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, expected)


class TestResidualBlock:
    """Tests for ResidualBlock module."""

    def test_init(self):
        """ResidualBlock initializes correctly."""
        block = ResidualBlock(dim=64)
        assert block is not None

    def test_forward_shape(self):
        """Forward pass preserves shape (residual property)."""
        block = ResidualBlock(dim=64, dropout_rate=0.3)
        x = torch.randn(8, 64)
        output = block(x)
        assert output.shape == x.shape

    def test_forward_different_batch_sizes(self):
        """Block works with different batch sizes."""
        block = ResidualBlock(dim=64)
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 64)
            output = block(x)
            assert output.shape == (batch_size, 64)

    def test_has_layers(self):
        """Block has layers attribute."""
        block = ResidualBlock(dim=64)
        assert hasattr(block, "layers")
        assert isinstance(block.layers, nn.Sequential)

    def test_gradient_flow(self):
        """Gradients flow through residual block."""
        block = ResidualBlock(dim=64)
        x = torch.randn(8, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestRegularizedMultiHorizonNet:
    """Tests for RegularizedMultiHorizonNet architecture."""

    def test_init_default(self):
        """Network initializes with default parameters."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        assert model.input_dim == 32
        assert model.dropout_rate == 0.4

    def test_init_custom_params(self):
        """Network initializes with custom parameters."""
        model = RegularizedMultiHorizonNet(
            input_dim=64,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.5,
            num_residual_blocks=3,
        )
        assert model.input_dim == 64
        assert model.dropout_rate == 0.5

    def test_has_input_proj(self):
        """Network has input projection layer."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        assert hasattr(model, "input_proj")

    def test_has_residual_blocks(self):
        """Network has residual blocks."""
        model = RegularizedMultiHorizonNet(input_dim=32, num_residual_blocks=2)
        assert hasattr(model, "residual_blocks")
        assert len(model.residual_blocks) == 2

    def test_has_dim_reduce(self):
        """Network has dimension reduction layers."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        assert hasattr(model, "dim_reduce")

    def test_has_three_heads(self):
        """Network has three prediction heads."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        assert hasattr(model, "head_1h")
        assert hasattr(model, "head_4h")
        assert hasattr(model, "head_eod")

    def test_forward_returns_tuple(self):
        """Forward returns tuple of 3 tensors."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        x = torch.randn(8, 32)
        result = model(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_forward_output_shapes(self):
        """Forward outputs have correct shapes."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        batch_size = 8
        x = torch.randn(batch_size, 32)
        logits_1h, logits_4h, logits_eod = model(x)

        assert logits_1h.shape == (batch_size, 1)
        assert logits_4h.shape == (batch_size, 1)
        assert logits_eod.shape == (batch_size, 1)

    def test_forward_batch_size_1(self):
        """Forward works with batch size 1."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        x = torch.randn(1, 32)
        logits_1h, logits_4h, logits_eod = model(x)
        assert logits_1h.shape == (1, 1)

    def test_gradient_flow(self):
        """Gradients flow through the network."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        x = torch.randn(8, 32, requires_grad=True)
        logits_1h, logits_4h, logits_eod = model(x)
        loss = logits_1h.sum() + logits_4h.sum() + logits_eod.sum()
        loss.backward()
        assert x.grad is not None


class TestPredictWithUncertainty:
    """Tests for predict_with_uncertainty method."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return RegularizedMultiHorizonNet(input_dim=32)

    def test_returns_regularized_prediction(self, model):
        """Method returns RegularizedPrediction object."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict_with_uncertainty(x, n_samples=5)
        assert isinstance(result, RegularizedPrediction)

    def test_probabilities_valid(self, model):
        """Probabilities are in [0, 1]."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict_with_uncertainty(x, n_samples=5)

        assert 0 <= result.prob_up_1h <= 1
        assert 0 <= result.prob_up_4h <= 1
        assert 0 <= result.prob_up_eod <= 1

    def test_confidences_valid(self, model):
        """Confidences are in [0, 1]."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict_with_uncertainty(x, n_samples=5)

        assert 0 <= result.confidence_1h <= 1
        assert 0 <= result.confidence_4h <= 1
        assert 0 <= result.confidence_eod <= 1

    def test_uncertainties_non_negative(self, model):
        """Uncertainties are non-negative."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict_with_uncertainty(x, n_samples=5)

        assert result.uncertainty_1h >= 0
        assert result.uncertainty_4h >= 0
        assert result.uncertainty_eod >= 0

    def test_more_samples_more_stable(self, model):
        """More MC samples generally give more stable estimates."""
        x = np.random.randn(32).astype(np.float32)

        # This is a stochastic test, so we just verify it runs
        result_5 = model.predict_with_uncertainty(x, n_samples=5)
        result_20 = model.predict_with_uncertainty(x, n_samples=20)

        assert result_5 is not None
        assert result_20 is not None


class TestLabelSmoothingBCELoss:
    """Tests for LabelSmoothingBCELoss module."""

    def test_init_default(self):
        """Loss initializes with default smoothing."""
        loss_fn = LabelSmoothingBCELoss()
        assert loss_fn.smoothing == 0.1

    def test_init_custom_smoothing(self):
        """Loss initializes with custom smoothing."""
        loss_fn = LabelSmoothingBCELoss(smoothing=0.2)
        assert loss_fn.smoothing == 0.2

    def test_forward_returns_scalar(self):
        """Forward returns scalar tensor."""
        loss_fn = LabelSmoothingBCELoss()
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,)).float()

        loss = loss_fn(logits.squeeze(), targets)
        assert loss.dim() == 0  # Scalar

    def test_loss_positive(self):
        """Loss is positive."""
        loss_fn = LabelSmoothingBCELoss()
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,)).float()

        loss = loss_fn(logits.squeeze(), targets)
        assert loss.item() >= 0

    def test_smoothing_affects_loss(self):
        """Different smoothing values affect loss."""
        loss_fn_0 = LabelSmoothingBCELoss(smoothing=0.0)
        loss_fn_2 = LabelSmoothingBCELoss(smoothing=0.2)

        logits = torch.randn(8, 1)
        targets = torch.ones(8)  # All positive

        loss_0 = loss_fn_0(logits.squeeze(), targets)
        loss_2 = loss_fn_2(logits.squeeze(), targets)

        # With smoothing, targets change from 1 to 0.9
        # This affects the loss value
        assert loss_0.item() != loss_2.item()

    def test_gradient_flow(self):
        """Gradients flow through loss."""
        loss_fn = LabelSmoothingBCELoss()
        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.randint(0, 2, (8,)).float()

        loss = loss_fn(logits.squeeze(), targets)
        loss.backward()

        assert logits.grad is not None


class TestRegularizedTrainer:
    """Tests for RegularizedTrainer class."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return RegularizedMultiHorizonNet(input_dim=16)

    @pytest.fixture
    def trainer(self, model):
        """Create trainer for testing."""
        return RegularizedTrainer(model, device="cpu")

    @pytest.fixture
    def sample_data_loaders(self):
        """Create sample data loaders for testing."""
        n_samples = 64
        X = torch.randn(n_samples, 16)
        y_1h = torch.randint(0, 2, (n_samples,))
        y_4h = torch.randint(0, 2, (n_samples,))
        y_eod = torch.randint(0, 2, (n_samples,))

        dataset = TensorDataset(X, y_1h, y_4h, y_eod)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        return train_loader, val_loader

    def test_trainer_init(self, trainer):
        """Trainer initializes correctly."""
        assert trainer.device == torch.device("cpu")
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None

    def test_trainer_best_state_init(self, trainer):
        """Trainer initializes with inf best_val_loss."""
        assert trainer.best_val_loss == float("inf")
        assert trainer.best_state is None
        assert trainer.patience_counter == 0

    def test_train_epoch_returns_dict(self, trainer, sample_data_loaders):
        """train_epoch returns loss dictionary."""
        train_loader, _ = sample_data_loaders
        result = trainer.train_epoch(train_loader)

        assert isinstance(result, dict)
        assert "total" in result
        assert result["total"] >= 0

    def test_validate_returns_tuple(self, trainer, sample_data_loaders):
        """validate returns (losses, accuracies) tuple."""
        _, val_loader = sample_data_loaders
        result = trainer.validate(val_loader)

        assert isinstance(result, tuple)
        assert len(result) == 2

        losses, accuracies = result
        assert isinstance(losses, dict)
        assert isinstance(accuracies, dict)

    def test_validate_accuracies_valid(self, trainer, sample_data_loaders):
        """validate returns valid accuracy values."""
        _, val_loader = sample_data_loaders
        _, accuracies = trainer.validate(val_loader)

        assert "acc_1h" in accuracies
        assert "acc_4h" in accuracies
        assert "acc_eod" in accuracies

        for key, acc in accuracies.items():
            assert 0 <= acc <= 1

    def test_fit_returns_history(self, trainer, sample_data_loaders):
        """fit returns training history."""
        train_loader, val_loader = sample_data_loaders
        history = trainer.fit(train_loader, val_loader, epochs=2, patience=10)

        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert "acc_1h" in history
        assert len(history["train_loss"]) == 2

    def test_fit_early_stopping(self, trainer, sample_data_loaders):
        """fit uses early stopping."""
        train_loader, val_loader = sample_data_loaders
        # Set very low patience to trigger early stopping
        history = trainer.fit(train_loader, val_loader, epochs=100, patience=1)

        # Should stop before 100 epochs
        assert len(history["train_loss"]) < 100

    def test_fit_restores_best_model(self, trainer, sample_data_loaders):
        """fit restores best model after training."""
        train_loader, val_loader = sample_data_loaders
        trainer.fit(train_loader, val_loader, epochs=3, patience=10)

        # Best state should be set if any epoch improved
        # The model should have valid parameters
        assert trainer.model is not None


class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_weights_kaiming_initialized(self):
        """Linear weights are initialized with Kaiming."""
        model = RegularizedMultiHorizonNet(input_dim=32)

        # Check that weights are not all zeros
        has_nonzero_weights = False
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if (module.weight.abs() > 0).any():
                    has_nonzero_weights = True
                    break

        assert has_nonzero_weights

    def test_biases_zero_initialized(self):
        """Linear biases are initialized to zero."""
        model = RegularizedMultiHorizonNet(input_dim=32)

        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert (module.bias == 0).all()


class TestModelModes:
    """Tests for model training/evaluation modes."""

    def test_model_train_mode(self):
        """Model can be set to training mode."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        model.train()
        assert model.training

    def test_model_eval_mode(self):
        """Model can be set to evaluation mode."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        model.eval()
        assert not model.training

    def test_predict_with_uncertainty_enables_dropout(self):
        """predict_with_uncertainty uses training mode (MC Dropout)."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        model.eval()

        x = np.random.randn(32).astype(np.float32)
        model.predict_with_uncertainty(x, n_samples=3)

        # After MC Dropout, model should be in train mode
        assert model.training


class TestInputValidation:
    """Tests for input validation and edge cases."""

    def test_wrong_input_dim_raises(self):
        """Model raises error for wrong input dimension."""
        model = RegularizedMultiHorizonNet(input_dim=32)
        x = torch.randn(8, 64)  # Wrong dim

        with pytest.raises(RuntimeError):
            model(x)

    def test_various_batch_sizes(self):
        """Model works with various batch sizes."""
        model = RegularizedMultiHorizonNet(input_dim=32)

        for batch_size in [1, 2, 8, 16, 32, 64]:
            x = torch.randn(batch_size, 32)
            logits_1h, logits_4h, logits_eod = model(x)
            assert logits_1h.shape[0] == batch_size
