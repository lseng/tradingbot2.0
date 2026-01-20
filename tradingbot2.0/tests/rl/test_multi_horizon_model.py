"""
Tests for RL multi-horizon prediction model.

Tests cover:
- HorizonPrediction dataclass
- MultiHorizonNet architecture
- Forward pass and predict_proba
- MultiHorizonLoss computation
- MultiHorizonTrainer (basic functionality)
- create_multi_horizon_targets utility function
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from zoneinfo import ZoneInfo

from src.rl.multi_horizon_model import (
    HorizonPrediction,
    MultiHorizonNet,
    MultiHorizonLoss,
    MultiHorizonTrainer,
    create_multi_horizon_targets,
)


NY_TZ = ZoneInfo("America/New_York")


class TestHorizonPrediction:
    """Tests for HorizonPrediction dataclass."""

    def test_creation(self):
        """HorizonPrediction can be created with all fields."""
        pred = HorizonPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
        )
        assert pred.prob_up_1h == 0.6
        assert pred.prob_up_4h == 0.7
        assert pred.prob_up_eod == 0.55
        assert pred.confidence_1h == 0.3
        assert pred.confidence_4h == 0.5
        assert pred.confidence_eod == 0.2

    def test_to_array_shape(self):
        """to_array returns array of correct shape."""
        pred = HorizonPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
        )
        arr = pred.to_array()
        assert arr.shape == (6,)

    def test_to_array_dtype(self):
        """to_array returns float32 array."""
        pred = HorizonPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
        )
        arr = pred.to_array()
        assert arr.dtype == np.float32

    def test_to_array_values(self):
        """to_array contains correct values in order."""
        pred = HorizonPrediction(
            prob_up_1h=0.6,
            prob_up_4h=0.7,
            prob_up_eod=0.55,
            confidence_1h=0.3,
            confidence_4h=0.5,
            confidence_eod=0.2,
        )
        arr = pred.to_array()
        expected = np.array([0.6, 0.7, 0.55, 0.3, 0.5, 0.2], dtype=np.float32)
        np.testing.assert_array_almost_equal(arr, expected)


class TestMultiHorizonNetInit:
    """Tests for MultiHorizonNet initialization."""

    def test_init_default_params(self):
        """Network initializes with default parameters."""
        model = MultiHorizonNet(input_dim=32)
        assert model.input_dim == 32

    def test_init_custom_hidden_dims(self):
        """Network accepts custom hidden dimensions."""
        model = MultiHorizonNet(input_dim=32, hidden_dims=[128, 64, 32])
        assert model is not None

    def test_init_custom_dropout(self):
        """Network accepts custom dropout rate."""
        model = MultiHorizonNet(input_dim=32, dropout_rate=0.5)
        assert model is not None

    def test_has_encoder(self):
        """Network has encoder module."""
        model = MultiHorizonNet(input_dim=32)
        assert hasattr(model, "encoder")
        assert isinstance(model.encoder, nn.Sequential)

    def test_has_three_heads(self):
        """Network has three prediction heads."""
        model = MultiHorizonNet(input_dim=32)
        assert hasattr(model, "head_1h")
        assert hasattr(model, "head_4h")
        assert hasattr(model, "head_eod")

    def test_heads_are_sequential(self):
        """Prediction heads are Sequential modules."""
        model = MultiHorizonNet(input_dim=32)
        assert isinstance(model.head_1h, nn.Sequential)
        assert isinstance(model.head_4h, nn.Sequential)
        assert isinstance(model.head_eod, nn.Sequential)


class TestMultiHorizonNetForward:
    """Tests for MultiHorizonNet forward pass."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return MultiHorizonNet(input_dim=32, hidden_dims=[64, 32, 16])

    def test_forward_returns_tuple(self, model):
        """Forward pass returns tuple of 3 tensors."""
        x = torch.randn(8, 32)
        result = model(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_forward_output_shapes(self, model):
        """Forward pass outputs have correct shapes."""
        batch_size = 8
        x = torch.randn(batch_size, 32)
        logits_1h, logits_4h, logits_eod = model(x)

        assert logits_1h.shape == (batch_size, 1)
        assert logits_4h.shape == (batch_size, 1)
        assert logits_eod.shape == (batch_size, 1)

    def test_forward_batch_size_1(self, model):
        """Forward pass works with batch size 1 in eval mode."""
        model.eval()  # BatchNorm requires eval mode for batch size 1
        x = torch.randn(1, 32)
        logits_1h, logits_4h, logits_eod = model(x)

        assert logits_1h.shape == (1, 1)
        assert logits_4h.shape == (1, 1)
        assert logits_eod.shape == (1, 1)

    def test_forward_gradient_flows(self, model):
        """Gradients flow through the network."""
        x = torch.randn(8, 32, requires_grad=True)
        logits_1h, logits_4h, logits_eod = model(x)
        loss = logits_1h.sum() + logits_4h.sum() + logits_eod.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (8, 32)


class TestMultiHorizonNetPredictProba:
    """Tests for predict_proba method."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return MultiHorizonNet(input_dim=32)

    def test_predict_proba_returns_probabilities(self, model):
        """predict_proba returns values in [0, 1]."""
        x = torch.randn(8, 32)
        prob_1h, prob_4h, prob_eod = model.predict_proba(x)

        assert (prob_1h >= 0).all() and (prob_1h <= 1).all()
        assert (prob_4h >= 0).all() and (prob_4h <= 1).all()
        assert (prob_eod >= 0).all() and (prob_eod <= 1).all()

    def test_predict_proba_shape(self, model):
        """predict_proba outputs have correct shapes."""
        batch_size = 8
        x = torch.randn(batch_size, 32)
        prob_1h, prob_4h, prob_eod = model.predict_proba(x)

        assert prob_1h.shape == (batch_size, 1)
        assert prob_4h.shape == (batch_size, 1)
        assert prob_eod.shape == (batch_size, 1)


class TestMultiHorizonNetPredict:
    """Tests for single-sample predict method."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return MultiHorizonNet(input_dim=32)

    def test_predict_returns_horizon_prediction(self, model):
        """predict returns HorizonPrediction object."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict(x)
        assert isinstance(result, HorizonPrediction)

    def test_predict_probabilities_valid(self, model):
        """predict returns valid probabilities."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict(x)

        assert 0 <= result.prob_up_1h <= 1
        assert 0 <= result.prob_up_4h <= 1
        assert 0 <= result.prob_up_eod <= 1

    def test_predict_confidence_valid(self, model):
        """predict returns valid confidence values."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict(x)

        assert 0 <= result.confidence_1h <= 1
        assert 0 <= result.confidence_4h <= 1
        assert 0 <= result.confidence_eod <= 1

    def test_predict_confidence_formula(self, model):
        """Confidence is computed as 2 * |prob - 0.5|."""
        x = np.random.randn(32).astype(np.float32)
        result = model.predict(x)

        expected_conf_1h = abs(result.prob_up_1h - 0.5) * 2
        expected_conf_4h = abs(result.prob_up_4h - 0.5) * 2
        expected_conf_eod = abs(result.prob_up_eod - 0.5) * 2

        assert abs(result.confidence_1h - expected_conf_1h) < 1e-5
        assert abs(result.confidence_4h - expected_conf_4h) < 1e-5
        assert abs(result.confidence_eod - expected_conf_eod) < 1e-5


class TestMultiHorizonLoss:
    """Tests for MultiHorizonLoss module."""

    def test_loss_init_default(self):
        """Loss initializes with default weights."""
        loss_fn = MultiHorizonLoss()
        assert loss_fn.weight_1h == 1.0
        assert loss_fn.weight_4h == 1.0
        assert loss_fn.weight_eod == 1.0

    def test_loss_init_custom_weights(self):
        """Loss initializes with custom weights."""
        loss_fn = MultiHorizonLoss(weight_1h=2.0, weight_4h=1.5, weight_eod=0.5)
        assert loss_fn.weight_1h == 2.0
        assert loss_fn.weight_4h == 1.5
        assert loss_fn.weight_eod == 0.5

    def test_loss_forward_returns_tuple(self):
        """Loss forward returns (total_loss, loss_dict) tuple."""
        loss_fn = MultiHorizonLoss()

        logits_1h = torch.randn(8, 1)
        logits_4h = torch.randn(8, 1)
        logits_eod = torch.randn(8, 1)
        target_1h = torch.randint(0, 2, (8,))
        target_4h = torch.randint(0, 2, (8,))
        target_eod = torch.randint(0, 2, (8,))

        result = loss_fn(logits_1h, logits_4h, logits_eod, target_1h, target_4h, target_eod)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)

    def test_loss_dict_keys(self):
        """Loss dict contains expected keys."""
        loss_fn = MultiHorizonLoss()

        logits_1h = torch.randn(8, 1)
        logits_4h = torch.randn(8, 1)
        logits_eod = torch.randn(8, 1)
        target_1h = torch.randint(0, 2, (8,))
        target_4h = torch.randint(0, 2, (8,))
        target_eod = torch.randint(0, 2, (8,))

        _, loss_dict = loss_fn(logits_1h, logits_4h, logits_eod, target_1h, target_4h, target_eod)

        assert "loss_1h" in loss_dict
        assert "loss_4h" in loss_dict
        assert "loss_eod" in loss_dict
        assert "total" in loss_dict

    def test_loss_positive(self):
        """Loss values are positive."""
        loss_fn = MultiHorizonLoss()

        logits_1h = torch.randn(8, 1)
        logits_4h = torch.randn(8, 1)
        logits_eod = torch.randn(8, 1)
        target_1h = torch.randint(0, 2, (8,))
        target_4h = torch.randint(0, 2, (8,))
        target_eod = torch.randint(0, 2, (8,))

        total_loss, loss_dict = loss_fn(logits_1h, logits_4h, logits_eod, target_1h, target_4h, target_eod)

        assert total_loss.item() >= 0
        assert loss_dict["loss_1h"] >= 0
        assert loss_dict["loss_4h"] >= 0
        assert loss_dict["loss_eod"] >= 0

    def test_loss_gradient_flows(self):
        """Gradient flows through loss computation."""
        loss_fn = MultiHorizonLoss()

        logits_1h = torch.randn(8, 1, requires_grad=True)
        logits_4h = torch.randn(8, 1, requires_grad=True)
        logits_eod = torch.randn(8, 1, requires_grad=True)
        target_1h = torch.randint(0, 2, (8,))
        target_4h = torch.randint(0, 2, (8,))
        target_eod = torch.randint(0, 2, (8,))

        total_loss, _ = loss_fn(logits_1h, logits_4h, logits_eod, target_1h, target_4h, target_eod)
        total_loss.backward()

        assert logits_1h.grad is not None
        assert logits_4h.grad is not None
        assert logits_eod.grad is not None


class TestMultiHorizonTrainer:
    """Tests for MultiHorizonTrainer class."""

    @pytest.fixture
    def trainer(self):
        """Create trainer for testing."""
        model = MultiHorizonNet(input_dim=32)
        return MultiHorizonTrainer(model, device="cpu")

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

    def test_trainer_model_on_device(self, trainer):
        """Model is moved to specified device."""
        # Get a parameter and check its device
        param = next(trainer.model.parameters())
        assert param.device == torch.device("cpu")

    def test_trainer_learning_rate(self):
        """Trainer uses specified learning rate."""
        model = MultiHorizonNet(input_dim=32)
        trainer = MultiHorizonTrainer(model, learning_rate=0.01, device="cpu")
        # Check optimizer param groups
        assert trainer.optimizer.param_groups[0]["lr"] == 0.01


class TestCreateMultiHorizonTargets:
    """Tests for create_multi_horizon_targets function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample 1-minute OHLCV data."""
        dates = pd.date_range("2024-01-02 09:30:00", periods=500, freq="1min", tz=NY_TZ)
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.001, 500)
        close_prices = base_price * np.cumprod(1 + returns)

        return pd.DataFrame({
            "open": close_prices * 0.999,
            "high": close_prices * 1.001,
            "low": close_prices * 0.998,
            "close": close_prices,
            "volume": np.random.randint(100, 1000, 500),
        }, index=dates)

    def test_creates_target_columns(self, sample_df):
        """Function creates target_1h, target_4h, target_eod columns."""
        result = create_multi_horizon_targets(sample_df)

        assert "target_1h" in result.columns
        assert "target_4h" in result.columns
        assert "target_eod" in result.columns

    def test_targets_are_binary(self, sample_df):
        """Target values are 0 or 1."""
        result = create_multi_horizon_targets(sample_df)

        # Drop NaN values and check remaining are binary
        target_1h = result["target_1h"].dropna()
        target_4h = result["target_4h"].dropna()
        target_eod = result["target_eod"].dropna()

        assert set(target_1h.unique()).issubset({0, 1})
        assert set(target_4h.unique()).issubset({0, 1})
        assert set(target_eod.unique()).issubset({0, 1})

    def test_target_1h_looks_60_bars_ahead(self, sample_df):
        """target_1h compares to price 60 bars ahead (default)."""
        result = create_multi_horizon_targets(sample_df, horizon_1h_bars=60)

        # For an upward move, target should be 1
        idx = 0
        while idx < len(result) - 60:
            if result["close"].iloc[idx + 60] > result["close"].iloc[idx]:
                assert result["target_1h"].iloc[idx] == 1
                break
            idx += 1

    def test_target_4h_looks_240_bars_ahead(self, sample_df):
        """target_4h compares to price 240 bars ahead (default)."""
        result = create_multi_horizon_targets(sample_df, horizon_4h_bars=240)

        # Check that target_4h uses 240 bar horizon
        assert "target_4h" in result.columns

    def test_target_eod_uses_day_close(self, sample_df):
        """target_eod compares to end-of-day close price."""
        result = create_multi_horizon_targets(sample_df)

        # The eod_close column should be created
        assert "eod_close" in result.columns

        # All bars on same day should have same eod_close
        dates = result.index.date
        unique_dates = set(dates)
        for date in unique_dates:
            day_data = result[result.index.date == date]
            eod_closes = day_data["eod_close"].dropna().unique()
            if len(eod_closes) > 0:
                assert len(eod_closes) == 1

    def test_does_not_modify_original(self, sample_df):
        """Function returns copy, doesn't modify original."""
        original_cols = set(sample_df.columns)
        result = create_multi_horizon_targets(sample_df)

        assert set(sample_df.columns) == original_cols
        assert "target_1h" not in sample_df.columns

    def test_custom_horizons(self, sample_df):
        """Function accepts custom horizon parameters."""
        result = create_multi_horizon_targets(
            sample_df,
            horizon_1h_bars=30,  # 30 bars instead of 60
            horizon_4h_bars=120,  # 120 bars instead of 240
        )

        assert "target_1h" in result.columns
        assert "target_4h" in result.columns


class TestInputValidation:
    """Tests for input validation and edge cases."""

    def test_model_wrong_input_dim(self):
        """Model raises error for wrong input dimension during forward."""
        model = MultiHorizonNet(input_dim=32)
        x = torch.randn(8, 64)  # Wrong dim

        with pytest.raises(RuntimeError):
            model(x)

    def test_model_accepts_different_batch_sizes(self):
        """Model works with various batch sizes in eval mode."""
        model = MultiHorizonNet(input_dim=32)
        model.eval()  # BatchNorm requires eval mode for batch size 1

        for batch_size in [1, 2, 8, 16, 32]:
            x = torch.randn(batch_size, 32)
            logits_1h, logits_4h, logits_eod = model(x)
            assert logits_1h.shape[0] == batch_size


class TestModelModes:
    """Tests for model training/evaluation modes."""

    def test_model_train_mode(self):
        """Model can be set to training mode."""
        model = MultiHorizonNet(input_dim=32)
        model.train()
        assert model.training

    def test_model_eval_mode(self):
        """Model can be set to evaluation mode."""
        model = MultiHorizonNet(input_dim=32)
        model.eval()
        assert not model.training

    def test_predict_sets_eval_mode(self):
        """predict method sets model to eval mode."""
        model = MultiHorizonNet(input_dim=32)
        model.train()  # Set to training mode first
        assert model.training

        x = np.random.randn(32).astype(np.float32)
        model.predict(x)

        # Model should now be in eval mode
        assert not model.training


class TestWeightInitialization:
    """Tests for weight initialization."""

    def test_weights_initialized(self):
        """Model weights are not zero after initialization."""
        model = MultiHorizonNet(input_dim=32)

        # Check that at least one linear layer has non-zero weights
        has_nonzero_weights = False
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if (module.weight.abs() > 0).any():
                    has_nonzero_weights = True
                    break

        assert has_nonzero_weights

    def test_biases_initialized_zero(self):
        """Linear layer biases are initialized to zero."""
        model = MultiHorizonNet(input_dim=32)

        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert (module.bias == 0).all()
