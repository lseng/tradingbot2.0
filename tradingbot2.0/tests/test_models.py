"""
Comprehensive Unit Tests for Neural Network Models (3-Class Classification).

Tests cover:
1. Model architecture correctness (output shapes, num_classes)
2. Forward pass validation (raw logits, probabilities, predictions)
3. ModelPrediction dataclass functionality
4. Training integration with CrossEntropyLoss
5. Backward compatibility and edge cases

These tests ensure Phase 4.1 (3-Class Classification) is correctly implemented.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import sys
import os

# Add src/ml to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'ml'))

from models.neural_networks import (
    FeedForwardNet,
    LSTMNet,
    HybridNet,
    ModelPrediction,
    create_model,
    EarlyStopping
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def input_dim():
    """Standard input dimension."""
    return 40


@pytest.fixture
def seq_length():
    """Standard sequence length for LSTM tests."""
    return 20


@pytest.fixture
def num_classes():
    """Standard 3-class output for scalping."""
    return 3


@pytest.fixture
def feedforward_model(input_dim, num_classes):
    """Create a FeedForwardNet for testing."""
    return FeedForwardNet(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        dropout_rate=0.3,
        num_classes=num_classes
    )


@pytest.fixture
def lstm_model(input_dim, num_classes):
    """Create an LSTMNet for testing."""
    return LSTMNet(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout_rate=0.3,
        num_classes=num_classes
    )


@pytest.fixture
def hybrid_model(input_dim, num_classes):
    """Create a HybridNet for testing."""
    return HybridNet(
        seq_input_dim=input_dim,
        static_input_dim=20,
        lstm_hidden=32,
        lstm_layers=1,
        num_classes=num_classes
    )


# ============================================================================
# FeedForwardNet Tests
# ============================================================================

class TestFeedForwardNet:
    """Tests for FeedForwardNet with 3-class output."""

    def test_output_shape(self, feedforward_model, batch_size, input_dim, num_classes):
        """Test output shape is (batch_size, num_classes)."""
        x = torch.randn(batch_size, input_dim)
        output = feedforward_model(x)
        assert output.shape == (batch_size, num_classes)

    def test_output_is_raw_logits(self, feedforward_model, batch_size, input_dim):
        """Test that output is raw logits (not probabilities)."""
        x = torch.randn(batch_size, input_dim)
        output = feedforward_model(x)
        # Raw logits can be negative and don't sum to 1
        assert output.min() < 0 or output.max() > 1

    def test_get_probabilities_sums_to_one(self, feedforward_model, batch_size, input_dim):
        """Test that get_probabilities returns valid probabilities."""
        x = torch.randn(batch_size, input_dim)
        probs = feedforward_model.get_probabilities(x)
        # Each row should sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)

    def test_get_probabilities_range(self, feedforward_model, batch_size, input_dim):
        """Test that probabilities are in [0, 1] range."""
        x = torch.randn(batch_size, input_dim)
        probs = feedforward_model.get_probabilities(x)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_returns_model_predictions(self, feedforward_model, batch_size, input_dim):
        """Test predict() returns list of ModelPrediction instances."""
        x = torch.randn(batch_size, input_dim)
        predictions = feedforward_model.predict(x)
        assert len(predictions) == batch_size
        assert all(isinstance(p, ModelPrediction) for p in predictions)

    def test_predict_direction_values(self, feedforward_model, batch_size, input_dim):
        """Test that direction is in {-1, 0, 1}."""
        x = torch.randn(batch_size, input_dim)
        predictions = feedforward_model.predict(x)
        for p in predictions:
            assert p.direction in [-1, 0, 1]

    def test_predict_confidence_range(self, feedforward_model, batch_size, input_dim):
        """Test that confidence is in [0, 1]."""
        x = torch.randn(batch_size, input_dim)
        predictions = feedforward_model.predict(x)
        for p in predictions:
            assert 0 <= p.confidence <= 1

    def test_num_classes_attribute(self, feedforward_model, num_classes):
        """Test num_classes attribute is set correctly."""
        assert feedforward_model.num_classes == num_classes

    def test_binary_classification(self, input_dim):
        """Test backward compatibility with 2-class."""
        model = FeedForwardNet(input_dim=input_dim, num_classes=2)
        x = torch.randn(4, input_dim)
        output = model(x)
        assert output.shape == (4, 2)
        assert model.num_classes == 2

    def test_single_sample(self, feedforward_model, input_dim):
        """Test with single sample (batch_size=1)."""
        feedforward_model.eval()  # BatchNorm requires batch_size > 1 in train mode
        x = torch.randn(1, input_dim)
        output = feedforward_model(x)
        assert output.shape == (1, 3)

    def test_weight_initialization(self, feedforward_model):
        """Test that weights are initialized (not all zeros)."""
        for name, param in feedforward_model.named_parameters():
            if 'weight' in name:
                assert not torch.all(param == 0)


# ============================================================================
# LSTMNet Tests
# ============================================================================

class TestLSTMNet:
    """Tests for LSTMNet with 3-class output."""

    def test_output_shape(self, lstm_model, batch_size, seq_length, input_dim, num_classes):
        """Test output shape is (batch_size, num_classes)."""
        x = torch.randn(batch_size, seq_length, input_dim)
        output, hidden = lstm_model(x)
        assert output.shape == (batch_size, num_classes)

    def test_hidden_state_shape(self, lstm_model, batch_size, seq_length, input_dim):
        """Test hidden state dimensions."""
        x = torch.randn(batch_size, seq_length, input_dim)
        output, (h, c) = lstm_model(x)
        # h: (num_layers * num_directions, batch, hidden_dim)
        assert h.shape[1] == batch_size
        assert c.shape[1] == batch_size

    def test_get_probabilities(self, lstm_model, batch_size, seq_length, input_dim):
        """Test get_probabilities returns valid probabilities."""
        x = torch.randn(batch_size, seq_length, input_dim)
        probs, hidden = lstm_model.get_probabilities(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)

    def test_predict_returns_model_predictions(self, lstm_model, batch_size, seq_length, input_dim):
        """Test predict() returns list of ModelPrediction instances."""
        x = torch.randn(batch_size, seq_length, input_dim)
        predictions, hidden = lstm_model.predict(x)
        assert len(predictions) == batch_size
        assert all(isinstance(p, ModelPrediction) for p in predictions)

    def test_num_classes_attribute(self, lstm_model, num_classes):
        """Test num_classes attribute is set correctly."""
        assert lstm_model.num_classes == num_classes

    def test_bidirectional(self, input_dim):
        """Test bidirectional LSTM."""
        model = LSTMNet(input_dim=input_dim, bidirectional=True, num_classes=3)
        x = torch.randn(4, 10, input_dim)
        output, hidden = model(x)
        assert output.shape == (4, 3)

    def test_with_initial_hidden(self, lstm_model, batch_size, seq_length, input_dim):
        """Test passing initial hidden state."""
        x = torch.randn(batch_size, seq_length, input_dim)
        # Get initial hidden state
        output1, hidden1 = lstm_model(x)
        # Use it for next forward pass
        output2, hidden2 = lstm_model(x, hidden1)
        assert output2.shape == (batch_size, 3)


# ============================================================================
# HybridNet Tests
# ============================================================================

class TestHybridNet:
    """Tests for HybridNet with 3-class output."""

    def test_output_shape(self, hybrid_model, batch_size, seq_length, input_dim, num_classes):
        """Test output shape is (batch_size, num_classes)."""
        seq_x = torch.randn(batch_size, seq_length, input_dim)
        static_x = torch.randn(batch_size, 20)
        output = hybrid_model(seq_x, static_x)
        assert output.shape == (batch_size, num_classes)

    def test_get_probabilities(self, hybrid_model, batch_size, seq_length, input_dim):
        """Test get_probabilities returns valid probabilities."""
        seq_x = torch.randn(batch_size, seq_length, input_dim)
        static_x = torch.randn(batch_size, 20)
        probs = hybrid_model.get_probabilities(seq_x, static_x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)

    def test_predict_returns_model_predictions(self, hybrid_model, batch_size, seq_length, input_dim):
        """Test predict() returns list of ModelPrediction instances."""
        seq_x = torch.randn(batch_size, seq_length, input_dim)
        static_x = torch.randn(batch_size, 20)
        predictions = hybrid_model.predict(seq_x, static_x)
        assert len(predictions) == batch_size
        assert all(isinstance(p, ModelPrediction) for p in predictions)

    def test_num_classes_attribute(self, hybrid_model, num_classes):
        """Test num_classes attribute is set correctly."""
        assert hybrid_model.num_classes == num_classes


# ============================================================================
# ModelPrediction Tests
# ============================================================================

class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""

    def test_from_logits_down_prediction(self):
        """Test from_logits correctly predicts DOWN (class 0 -> direction -1)."""
        logits = torch.tensor([2.0, -1.0, -1.0])  # Strong DOWN signal
        pred = ModelPrediction.from_logits(logits)
        assert pred.direction == -1  # DOWN
        assert pred.confidence > 0.5  # Should be confident

    def test_from_logits_flat_prediction(self):
        """Test from_logits correctly predicts FLAT (class 1 -> direction 0)."""
        logits = torch.tensor([-1.0, 2.0, -1.0])  # Strong FLAT signal
        pred = ModelPrediction.from_logits(logits)
        assert pred.direction == 0  # FLAT
        assert pred.confidence > 0.5

    def test_from_logits_up_prediction(self):
        """Test from_logits correctly predicts UP (class 2 -> direction 1)."""
        logits = torch.tensor([-1.0, -1.0, 2.0])  # Strong UP signal
        pred = ModelPrediction.from_logits(logits)
        assert pred.direction == 1  # UP
        assert pred.confidence > 0.5

    def test_from_logits_confidence(self):
        """Test confidence is max probability."""
        logits = torch.tensor([0.0, 0.0, 5.0])  # Very confident UP
        pred = ModelPrediction.from_logits(logits)
        # Softmax should give very high prob to class 2
        assert pred.confidence > 0.9

    def test_from_logits_class_probabilities(self):
        """Test class_probabilities are correct."""
        logits = torch.tensor([1.0, 1.0, 1.0])  # Equal logits
        pred = ModelPrediction.from_logits(logits)
        assert pred.class_probabilities is not None
        assert len(pred.class_probabilities) == 3
        # All should be ~1/3
        for prob in pred.class_probabilities:
            assert abs(prob - 1/3) < 0.01

    def test_from_logits_predicted_move(self):
        """Test predicted_move is weighted average of tick expectations."""
        # Default tick_expectations = (-4.0, 0.0, 4.0)
        logits = torch.tensor([0.0, 0.0, 5.0])  # Confident UP
        pred = ModelPrediction.from_logits(logits)
        # Should be positive (UP expected)
        assert pred.predicted_move > 0

    def test_from_logits_custom_tick_expectations(self):
        """Test custom tick expectations."""
        logits = torch.tensor([5.0, 0.0, 0.0])  # Confident DOWN
        pred = ModelPrediction.from_logits(
            logits,
            tick_expectations=(-10.0, 0.0, 10.0)
        )
        # Should be negative (DOWN expected)
        assert pred.predicted_move < 0

    def test_from_logits_with_batch_dim(self):
        """Test from_logits handles batched logits (squeezed)."""
        logits = torch.tensor([[1.0, 0.0, 0.0]])  # Shape (1, 3)
        pred = ModelPrediction.from_logits(logits)
        assert pred.direction == -1  # DOWN

    def test_from_logits_timestamp(self):
        """Test timestamp is set."""
        logits = torch.tensor([0.0, 1.0, 0.0])
        timestamp = datetime(2025, 1, 1, 10, 30, 0)
        pred = ModelPrediction.from_logits(logits, timestamp=timestamp)
        assert pred.timestamp == timestamp

    def test_from_logits_volatility(self):
        """Test volatility is passed through."""
        logits = torch.tensor([0.0, 1.0, 0.0])
        pred = ModelPrediction.from_logits(logits, volatility=2.5)
        assert pred.volatility == 2.5


# ============================================================================
# create_model Factory Tests
# ============================================================================

class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_feedforward(self, input_dim):
        """Test creating feedforward model."""
        model = create_model('feedforward', input_dim, num_classes=3)
        assert isinstance(model, FeedForwardNet)
        assert model.num_classes == 3

    def test_create_lstm(self, input_dim):
        """Test creating LSTM model."""
        model = create_model('lstm', input_dim, num_classes=3)
        assert isinstance(model, LSTMNet)
        assert model.num_classes == 3

    def test_create_hybrid(self, input_dim):
        """Test creating hybrid model."""
        model = create_model('hybrid', input_dim, num_classes=3, static_input_dim=20)
        assert isinstance(model, HybridNet)
        assert model.num_classes == 3

    def test_default_num_classes(self, input_dim):
        """Test default num_classes is 3."""
        model = create_model('feedforward', input_dim)
        assert model.num_classes == 3

    def test_case_insensitive(self, input_dim):
        """Test model_type is case insensitive."""
        model1 = create_model('FEEDFORWARD', input_dim)
        model2 = create_model('FeedForward', input_dim)
        assert isinstance(model1, FeedForwardNet)
        assert isinstance(model2, FeedForwardNet)

    def test_unknown_type_raises(self, input_dim):
        """Test unknown model type raises ValueError."""
        with pytest.raises(ValueError):
            create_model('unknown', input_dim)

    def test_kwargs_passed_through(self, input_dim):
        """Test additional kwargs are passed to model."""
        model = create_model('feedforward', input_dim, hidden_dims=[128, 64])
        assert model.hidden_dims == [128, 64]


# ============================================================================
# EarlyStopping Tests
# ============================================================================

class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_no_stop_on_improvement(self):
        """Test no early stopping when loss improves."""
        early_stop = EarlyStopping(patience=3)
        model = FeedForwardNet(10, num_classes=3)

        losses = [1.0, 0.9, 0.8, 0.7]
        for loss in losses:
            should_stop = early_stop(loss, model)
            assert not should_stop

    def test_stop_after_patience(self):
        """Test stops after patience epochs without improvement."""
        early_stop = EarlyStopping(patience=3)
        model = FeedForwardNet(10, num_classes=3)

        # Improve, then plateau
        early_stop(1.0, model)
        early_stop(0.9, model)  # Best loss
        early_stop(0.95, model)  # No improvement
        early_stop(0.95, model)  # No improvement
        should_stop = early_stop(0.95, model)  # patience=3 reached

        assert should_stop

    def test_resets_counter_on_improvement(self):
        """Test counter resets when loss improves."""
        early_stop = EarlyStopping(patience=3)
        model = FeedForwardNet(10, num_classes=3)

        early_stop(1.0, model)
        early_stop(1.1, model)  # +1 counter
        early_stop(0.9, model)  # Reset counter (improvement)
        early_stop(1.0, model)  # +1 counter
        early_stop(1.0, model)  # +2 counter
        should_stop = early_stop(1.0, model)  # +3 counter = stop

        assert should_stop

    def test_min_delta(self):
        """Test min_delta for improvement threshold."""
        early_stop = EarlyStopping(patience=2, min_delta=0.1)
        model = FeedForwardNet(10, num_classes=3)

        early_stop(1.0, model)
        early_stop(0.95, model)  # Improvement < min_delta, doesn't count
        should_stop = early_stop(0.92, model)  # patience reached

        assert should_stop


# ============================================================================
# Training Integration Tests
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for 3-class training."""

    def test_crossentropy_loss_compatibility(self, feedforward_model, batch_size, input_dim):
        """Test model outputs work with CrossEntropyLoss."""
        x = torch.randn(batch_size, input_dim)
        targets = torch.randint(0, 3, (batch_size,))  # Class indices

        criterion = nn.CrossEntropyLoss()
        output = feedforward_model(x)

        # Should not raise an error
        loss = criterion(output, targets)
        assert loss.item() >= 0

    def test_crossentropy_with_class_weights(self, feedforward_model, batch_size, input_dim):
        """Test CrossEntropyLoss with class weights."""
        x = torch.randn(batch_size, input_dim)
        targets = torch.randint(0, 3, (batch_size,))

        # Typical scalping class weights (FLAT dominant)
        weights = torch.tensor([1.5, 0.5, 1.5])
        criterion = nn.CrossEntropyLoss(weight=weights)

        output = feedforward_model(x)
        loss = criterion(output, targets)

        assert loss.item() >= 0

    def test_backward_pass(self, feedforward_model, batch_size, input_dim):
        """Test backward pass and gradient computation."""
        x = torch.randn(batch_size, input_dim)
        targets = torch.randint(0, 3, (batch_size,))

        criterion = nn.CrossEntropyLoss()
        output = feedforward_model(x)
        loss = criterion(output, targets)

        loss.backward()

        # Check gradients exist
        for param in feedforward_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_optimization_step(self, feedforward_model, batch_size, input_dim):
        """Test complete optimization step."""
        optimizer = torch.optim.Adam(feedforward_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(batch_size, input_dim)
        targets = torch.randint(0, 3, (batch_size,))

        # Initial prediction
        initial_output = feedforward_model(x).detach().clone()

        # Optimization step
        optimizer.zero_grad()
        output = feedforward_model(x)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        new_output = feedforward_model(x).detach()
        assert not torch.allclose(initial_output, new_output)

    def test_lstm_training(self, lstm_model, batch_size, seq_length, input_dim):
        """Test LSTM training with CrossEntropyLoss."""
        x = torch.randn(batch_size, seq_length, input_dim)
        targets = torch.randint(0, 3, (batch_size,))

        criterion = nn.CrossEntropyLoss()
        output, _ = lstm_model(x)
        loss = criterion(output, targets)

        loss.backward()

        # Check LSTM gradients exist
        for name, param in lstm_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_batch(self, feedforward_model, input_dim):
        """Test handling of empty batch (should fail gracefully)."""
        x = torch.randn(0, input_dim)
        output = feedforward_model(x)
        assert output.shape == (0, 3)

    def test_large_batch(self, feedforward_model, input_dim):
        """Test with large batch size."""
        x = torch.randn(1024, input_dim)
        output = feedforward_model(x)
        assert output.shape == (1024, 3)

    def test_train_vs_eval_mode(self, feedforward_model, batch_size, input_dim):
        """Test dropout affects train vs eval mode."""
        x = torch.randn(batch_size, input_dim)

        feedforward_model.train()
        train_outputs = [feedforward_model(x).detach() for _ in range(5)]

        feedforward_model.eval()
        eval_outputs = [feedforward_model(x).detach() for _ in range(5)]

        # Eval outputs should be identical
        for i in range(1, len(eval_outputs)):
            assert torch.allclose(eval_outputs[0], eval_outputs[i])

    def test_extreme_logits(self):
        """Test ModelPrediction handles extreme logits."""
        # Very large logits
        logits = torch.tensor([100.0, -100.0, -100.0])
        pred = ModelPrediction.from_logits(logits)
        assert pred.direction == -1  # DOWN
        assert pred.confidence > 0.99

    def test_model_save_load(self, feedforward_model, input_dim, tmp_path):
        """Test model can be saved and loaded."""
        x = torch.randn(4, input_dim)

        # Save
        torch.save(feedforward_model.state_dict(), tmp_path / "model.pt")

        # Create new model with SAME architecture and load
        # Must match: hidden_dims=[64, 32] from fixture
        new_model = FeedForwardNet(input_dim, hidden_dims=[64, 32], num_classes=3)
        new_model.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))

        # Should produce same output
        feedforward_model.eval()
        new_model.eval()

        original_out = feedforward_model(x)
        loaded_out = new_model(x)

        assert torch.allclose(original_out, loaded_out)


# ============================================================================
# Inference Latency Test
# ============================================================================

class TestInferenceLatency:
    """Test inference latency requirements (< 10ms)."""

    def test_feedforward_inference_latency(self, feedforward_model, input_dim):
        """Test FeedForwardNet inference is fast enough."""
        import time

        feedforward_model.eval()
        x = torch.randn(1, input_dim)

        # Warm up
        for _ in range(10):
            feedforward_model(x)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                feedforward_model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_time = np.mean(times)
        max_time = np.max(times)

        # Should be well under 10ms on any modern CPU
        assert avg_time < 10, f"Average inference time {avg_time:.2f}ms exceeds 10ms"
        # 99th percentile should also be under 10ms
        p99_time = np.percentile(times, 99)
        assert p99_time < 10, f"P99 inference time {p99_time:.2f}ms exceeds 10ms"

    def test_lstm_inference_latency(self, lstm_model, seq_length, input_dim):
        """Test LSTMNet inference is fast enough."""
        import time

        lstm_model.eval()
        x = torch.randn(1, seq_length, input_dim)

        # Warm up
        for _ in range(10):
            lstm_model(x)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                lstm_model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_time = np.mean(times)

        # LSTM is slower, but should still be under 10ms
        assert avg_time < 10, f"Average inference time {avg_time:.2f}ms exceeds 10ms"
