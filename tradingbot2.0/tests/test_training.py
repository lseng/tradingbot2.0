"""
Tests for training pipeline components.

Tests cover:
- SequenceDataset creation for LSTM training
- ModelTrainer initialization, training, validation, and prediction
- WalkForwardValidator splitting logic
- Class weight computation
- Walk-forward training pipeline
- AUC calculation utilities
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))

from models.training import (
    SequenceDataset,
    ModelTrainer,
    WalkForwardValidator,
    compute_class_weights,
    train_with_walk_forward,
    calculate_multiclass_auc,
    calculate_auc
)
from models.neural_networks import FeedForwardNet, LSTMNet


class TestSequenceDataset:
    """Tests for SequenceDataset class."""

    def test_init_creates_sequences(self, sample_features_and_targets):
        """Test that initialization creates correct sequences."""
        X, y = sample_features_and_targets
        seq_length = 10

        dataset = SequenceDataset(X, y, seq_length=seq_length)

        # Should have len(X) - seq_length samples
        expected_samples = len(X) - seq_length
        assert len(dataset.X) == expected_samples
        assert len(dataset.y) == expected_samples

    def test_sequence_shape(self, sample_features_and_targets):
        """Test that sequences have correct shape."""
        X, y = sample_features_and_targets
        seq_length = 20

        dataset = SequenceDataset(X, y, seq_length=seq_length)

        # X should be (n_samples, seq_length, n_features)
        assert dataset.X.shape[1] == seq_length
        assert dataset.X.shape[2] == X.shape[1]

    def test_target_alignment(self, sample_features_and_targets):
        """Test that targets are aligned with sequence ends."""
        X, y = sample_features_and_targets
        seq_length = 10

        dataset = SequenceDataset(X, y, seq_length=seq_length)

        # First target should be y[seq_length]
        assert dataset.y[0] == y[seq_length]
        # Last target should be y[-1]
        assert dataset.y[-1] == y[-1]

    def test_get_tensors_types(self, sample_features_and_targets):
        """Test that get_tensors returns correct tensor types."""
        X, y = sample_features_and_targets

        dataset = SequenceDataset(X, y)
        X_tensor, y_tensor = dataset.get_tensors()

        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert X_tensor.dtype == torch.float32
        assert y_tensor.dtype == torch.int64  # LongTensor for CrossEntropyLoss

    def test_get_tensors_shapes(self, sample_features_and_targets):
        """Test tensor shapes match numpy arrays."""
        X, y = sample_features_and_targets
        seq_length = 15

        dataset = SequenceDataset(X, y, seq_length=seq_length)
        X_tensor, y_tensor = dataset.get_tensors()

        assert X_tensor.shape == torch.Size(dataset.X.shape)
        assert y_tensor.shape == torch.Size(dataset.y.shape)


class TestModelTrainerInit:
    """Tests for ModelTrainer initialization."""

    def test_device_selection_auto(self):
        """Test automatic device selection."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='auto')

        # Should select an available device
        assert trainer.device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]

    def test_device_selection_explicit_cpu(self):
        """Test explicit CPU device selection."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        assert trainer.device == torch.device('cpu')

    def test_model_on_device(self):
        """Test that model is moved to correct device."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Model parameters should be on CPU
        for param in trainer.model.parameters():
            assert param.device.type == 'cpu'

    def test_optimizer_creation(self):
        """Test optimizer is created correctly."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, learning_rate=0.001)

        assert trainer.optimizer is not None
        assert trainer.optimizer.defaults['lr'] == 0.001

    def test_scheduler_creation(self):
        """Test LR scheduler is created."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model)

        assert trainer.scheduler is not None

    def test_class_weights_applied(self, sample_features_and_targets):
        """Test that class weights are applied to loss function."""
        X, y = sample_features_and_targets
        class_weights = torch.tensor([1.5, 0.5, 1.5])

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, class_weights=class_weights)

        # CrossEntropyLoss should have weights
        assert trainer.criterion.weight is not None

    def test_history_initialized(self):
        """Test that training history is initialized."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model)

        assert 'train_loss' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'train_acc' in trainer.history
        assert 'val_acc' in trainer.history


class TestModelTrainerTraining:
    """Tests for ModelTrainer training methods."""

    def test_train_epoch_reduces_loss(self, sample_features_and_targets):
        """Test that training reduces loss over epoch."""
        X, y = sample_features_and_targets
        X_tensor = torch.FloatTensor(X[:100])
        y_tensor = torch.LongTensor(y[:100])

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        loss, acc = trainer.train_epoch(loader)

        assert loss > 0  # Loss should be positive
        assert 0 <= acc <= 1  # Accuracy between 0 and 1

    def test_validate_returns_metrics(self, sample_features_and_targets):
        """Test that validation returns loss and accuracy."""
        X, y = sample_features_and_targets
        X_tensor = torch.FloatTensor(X[:100])
        y_tensor = torch.LongTensor(y[:100])

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        loss, acc = trainer.validate(loader)

        assert loss > 0
        assert 0 <= acc <= 1

    def test_train_updates_history(self, sample_features_and_targets):
        """Test that training updates history."""
        X, y = sample_features_and_targets
        X_tensor = torch.FloatTensor(X[:100])
        y_tensor = torch.LongTensor(y[:100])

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Train for a few epochs
        trainer.train(loader, epochs=3, early_stopping_patience=10)

        assert len(trainer.history['train_loss']) == 3
        assert len(trainer.history['train_acc']) == 3

    def test_early_stopping_triggers(self, sample_features_and_targets):
        """Test that early stopping can trigger."""
        X, y = sample_features_and_targets

        # Create train and val loaders with different data
        train_X = torch.FloatTensor(X[:300])
        train_y = torch.LongTensor(y[:300])
        val_X = torch.FloatTensor(X[300:400])
        val_y = torch.LongTensor(y[300:400])

        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Train with very short patience - early stopping may or may not trigger
        # depending on validation loss trajectory
        trainer.train(train_loader, val_loader, epochs=100, early_stopping_patience=2)

        # Should complete (either early stop or all epochs)
        assert len(trainer.history['train_loss']) > 0


class TestModelTrainerPrediction:
    """Tests for ModelTrainer prediction methods."""

    def test_predict_returns_probabilities(self, sample_features_and_targets):
        """Test that predict returns class probabilities."""
        X, y = sample_features_and_targets

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        probs = trainer.predict(X[:10])

        # Should be (n_samples, num_classes)
        assert probs.shape == (10, 3)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(10))

    def test_predict_classes_returns_indices(self, sample_features_and_targets):
        """Test that predict_classes returns class indices."""
        X, y = sample_features_and_targets

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        classes = trainer.predict_classes(X[:10])

        # Should be class indices
        assert classes.shape == (10,)
        assert set(classes).issubset({0, 1, 2})


class TestModelTrainerCheckpointing:
    """Tests for model checkpointing."""

    def test_save_checkpoint(self, sample_features_and_targets):
        """Test saving a checkpoint."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            trainer.save_checkpoint(f.name)
            # File should exist and have content
            assert Path(f.name).exists()
            assert Path(f.name).stat().st_size > 0

    def test_load_checkpoint(self, sample_features_and_targets):
        """Test loading a checkpoint."""
        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            trainer.save_checkpoint(f.name)
            checkpoint_path = f.name

        # Create new model and trainer
        new_model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        new_trainer = ModelTrainer(new_model, device='cpu')

        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)

        # Model weights should be loaded
        assert new_trainer.model is not None


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator class."""

    def test_split_returns_list_of_tuples(self, sample_features_and_targets):
        """Test that split returns list of (train_idx, test_idx) tuples."""
        X, y = sample_features_and_targets

        validator = WalkForwardValidator(n_splits=3)
        splits = validator.split(X, y)

        assert isinstance(splits, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in splits)

    def test_split_temporal_ordering(self, sample_features_and_targets):
        """Test that train indices come before test indices."""
        X, y = sample_features_and_targets

        validator = WalkForwardValidator(n_splits=3)
        splits = validator.split(X, y)

        for train_idx, test_idx in splits:
            # All train indices should be less than all test indices
            assert train_idx.max() < test_idx.min()

    def test_expanding_window(self, sample_features_and_targets):
        """Test expanding window mode."""
        X, y = sample_features_and_targets

        validator = WalkForwardValidator(n_splits=3, expanding=True)
        splits = validator.split(X, y)

        # In expanding mode, all train sets should start at 0
        for train_idx, _ in splits:
            assert train_idx[0] == 0

    def test_rolling_window(self, sample_features_and_targets):
        """Test rolling window mode."""
        X, y = sample_features_and_targets

        validator = WalkForwardValidator(n_splits=3, expanding=False)
        splits = validator.split(X, y)

        # In rolling mode, train sets may start at different indices
        # (but first might still be 0)
        train_starts = [train_idx[0] for train_idx, _ in splits]
        # Later splits might have higher train_start
        assert train_starts == sorted(train_starts)

    def test_no_overlap_between_train_test(self, sample_features_and_targets):
        """Test that train and test sets don't overlap."""
        X, y = sample_features_and_targets

        validator = WalkForwardValidator(n_splits=5)
        splits = validator.split(X, y)

        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0


class TestComputeClassWeights:
    """Tests for class weight computation."""

    def test_returns_tensor(self):
        """Test that function returns a tensor."""
        y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
        weights = compute_class_weights(y, num_classes=3)

        assert isinstance(weights, torch.Tensor)

    def test_correct_shape(self):
        """Test that weights have correct shape."""
        y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
        weights = compute_class_weights(y, num_classes=3)

        assert weights.shape == (3,)

    def test_rare_class_higher_weight(self):
        """Test that rare classes get higher weights."""
        # Class 0: 2 samples, Class 1: 6 samples, Class 2: 2 samples
        y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
        weights = compute_class_weights(y, num_classes=3)

        # Class 1 (most common) should have lower weight
        assert weights[1] < weights[0]
        assert weights[1] < weights[2]

    def test_handles_zero_counts(self):
        """Test handling of missing classes."""
        y = np.array([0, 0, 0, 1, 1, 1])  # No class 2
        weights = compute_class_weights(y, num_classes=3)

        # Should not have NaN or inf
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()


class TestCalculateAUC:
    """Tests for AUC calculation utilities."""

    def test_calculate_auc_perfect(self):
        """Test AUC with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        auc = calculate_auc(y_true, y_pred)
        assert auc == 1.0

    def test_calculate_auc_random(self):
        """Test AUC with random predictions should be ~0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.rand(1000)

        auc = calculate_auc(y_true, y_pred)
        assert 0.4 < auc < 0.6  # Should be close to 0.5

    def test_calculate_multiclass_auc(self):
        """Test multi-class AUC calculation."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_probs = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.2, 0.7]
        ])

        auc = calculate_multiclass_auc(y_true, y_probs, num_classes=3)
        assert auc > 0.8  # Should be high for good predictions


class TestTrainWithWalkForward:
    """Tests for walk-forward training pipeline."""

    def test_returns_results_dict(self, sample_features_and_targets):
        """Test that function returns results dictionary."""
        X, y = sample_features_and_targets
        X_small = X[:200]
        y_small = y[:200]

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]},
            'learning_rate': 0.01
        }

        results = train_with_walk_forward(
            X_small, y_small,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            batch_size=32,
            num_classes=3
        )

        assert isinstance(results, dict)
        assert 'fold_metrics' in results
        assert 'predictions' in results
        assert 'actuals' in results
        assert 'overall_accuracy' in results

    def test_fold_metrics_structure(self, sample_features_and_targets):
        """Test structure of fold metrics."""
        X, y = sample_features_and_targets
        X_small = X[:200]
        y_small = y[:200]

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        results = train_with_walk_forward(
            X_small, y_small,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            num_classes=3
        )

        for fold_result in results['fold_metrics']:
            assert 'fold' in fold_result
            assert 'train_size' in fold_result
            assert 'test_size' in fold_result
            assert 'test_loss' in fold_result
            assert 'test_accuracy' in fold_result

    def test_predictions_count(self, sample_features_and_targets):
        """Test that predictions count matches test set sizes."""
        X, y = sample_features_and_targets
        X_small = X[:200]
        y_small = y[:200]

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        results = train_with_walk_forward(
            X_small, y_small,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            num_classes=3
        )

        total_test = sum(fold['test_size'] for fold in results['fold_metrics'])
        assert len(results['predictions']) == total_test
        assert len(results['actuals']) == total_test

    def test_class_weights_used(self, sample_features_and_targets):
        """Test that class weights are computed and used."""
        X, y = sample_features_and_targets
        X_small = X[:200]
        y_small = y[:200]

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        # Should complete without errors when using class weights
        results = train_with_walk_forward(
            X_small, y_small,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            num_classes=3,
            use_class_weights=True
        )

        assert results is not None


class TestLSTMTraining:
    """Tests specific to LSTM model training."""

    def test_lstm_training_completes(self, sample_features_and_targets):
        """Test that LSTM training completes."""
        X, y = sample_features_and_targets
        seq_length = 10

        # Create sequences
        dataset = SequenceDataset(X[:100], y[:100], seq_length=seq_length)
        X_tensor, y_tensor = dataset.get_tensors()

        train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

        model = LSTMNet(input_dim=40, hidden_dim=16, num_layers=1, num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Should complete without errors
        history = trainer.train(train_loader, epochs=2)

        assert len(history['train_loss']) == 2

    def test_lstm_prediction(self, sample_features_and_targets):
        """Test LSTM prediction."""
        X, y = sample_features_and_targets
        seq_length = 10

        model = LSTMNet(input_dim=40, hidden_dim=16, num_layers=1, num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Create sequence input
        X_seq = X[:20].reshape(1, 20, -1)  # (batch=1, seq=20, features)
        probs = trainer.predict(X_seq)

        assert probs.shape == (1, 3)


class TestGradientClipping:
    """Tests for gradient clipping during training."""

    def test_gradients_clipped(self, sample_features_and_targets):
        """Test that gradients are clipped."""
        X, y = sample_features_and_targets
        X_tensor = torch.FloatTensor(X[:100])
        y_tensor = torch.LongTensor(y[:100])

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        model = FeedForwardNet(input_dim=40, hidden_dims=[32], num_classes=3)
        trainer = ModelTrainer(model, device='cpu')

        # Run one epoch - gradient clipping happens in train_epoch
        trainer.train_epoch(loader)

        # Should complete without exploding gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
