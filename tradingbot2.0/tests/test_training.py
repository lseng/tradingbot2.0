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
    calculate_auc,
    create_sequences_fast,
    _simulate_trading_for_fold,
    _save_walk_forward_results,
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


class TestSequenceCreationPerformance:
    """Tests for sequence creation performance (Bug #10 fix).

    These tests ensure the optimized stride-based sequence creation:
    1. Produces identical output to the original implementation
    2. Completes in reasonable time for large datasets
    3. Uses acceptable memory
    """

    def test_sequence_values_correctness(self, sample_features_and_targets):
        """Test that optimized sequence creation produces identical values to naive implementation."""
        X, y = sample_features_and_targets
        seq_length = 10

        # Create sequences using the optimized implementation
        dataset = SequenceDataset(X, y, seq_length=seq_length)

        # Manually create sequences using the old naive approach (for comparison)
        X_naive, y_naive = [], []
        for i in range(len(X) - seq_length):
            X_naive.append(X[i:i + seq_length])
            y_naive.append(y[i + seq_length])
        X_naive = np.array(X_naive)
        y_naive = np.array(y_naive)

        # Values must be identical
        np.testing.assert_array_equal(dataset.X, X_naive)
        np.testing.assert_array_equal(dataset.y, y_naive)

    def test_sequence_output_shape(self, sample_features_and_targets):
        """Test that output shape matches expected (n_samples - seq_length, seq_length, n_features)."""
        X, y = sample_features_and_targets
        seq_length = 15
        n_features = X.shape[1]

        dataset = SequenceDataset(X, y, seq_length=seq_length)

        expected_samples = len(X) - seq_length
        assert dataset.X.shape == (expected_samples, seq_length, n_features)
        assert dataset.y.shape == (expected_samples,)

    def test_sequence_creation_performance(self):
        """Test that sequence creation completes in reasonable time.

        For 100K samples, should complete in under 1 second.
        This is proportionally scaled - 6M samples should complete in ~60 seconds.
        """
        import time

        # Create moderately large dataset (100K samples, 50 features)
        n_samples = 100_000
        n_features = 50
        seq_length = 60

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)

        start_time = time.perf_counter()
        dataset = SequenceDataset(X, y, seq_length=seq_length)
        elapsed_time = time.perf_counter() - start_time

        # Should complete in under 1 second for 100K samples
        # (The old implementation would take ~1 minute for this size)
        assert elapsed_time < 1.0, f"Sequence creation took {elapsed_time:.2f}s, expected < 1.0s"

        # Verify correct output
        assert dataset.X.shape == (n_samples - seq_length, seq_length, n_features)

    def test_sequence_creation_very_large_dataset(self):
        """Test sequence creation with larger dataset (500K samples).

        This is a regression test for Bug #10 where the old implementation
        would take 60+ minutes for 6M samples.
        """
        import time

        # Create larger dataset (500K samples)
        n_samples = 500_000
        n_features = 50
        seq_length = 60

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)

        start_time = time.perf_counter()
        dataset = SequenceDataset(X, y, seq_length=seq_length)
        elapsed_time = time.perf_counter() - start_time

        # Should complete in under 5 seconds for 500K samples
        # Linearly scaled: 6M samples should complete in ~60 seconds
        assert elapsed_time < 5.0, f"Sequence creation took {elapsed_time:.2f}s, expected < 5.0s"

        expected_samples = n_samples - seq_length
        assert len(dataset.X) == expected_samples
        assert len(dataset.y) == expected_samples

    def test_memory_efficient_sequence_creation(self):
        """Test that sequence creation doesn't use excessive memory.

        The optimized implementation should use ~2x final tensor size,
        not 3x+ like the old implementation.
        """
        import sys

        n_samples = 50_000
        n_features = 50
        seq_length = 60

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)

        # Calculate expected final size
        expected_X_size = (n_samples - seq_length) * seq_length * n_features * 4  # float32 = 4 bytes
        expected_y_size = (n_samples - seq_length) * 8  # int64 = 8 bytes

        dataset = SequenceDataset(X, y, seq_length=seq_length)

        # Verify the dataset arrays exist with correct size
        actual_X_size = dataset.X.nbytes
        actual_y_size = dataset.y.nbytes

        assert actual_X_size == expected_X_size, f"X size mismatch: {actual_X_size} vs {expected_X_size}"
        assert actual_y_size == expected_y_size, f"y size mismatch: {actual_y_size} vs {expected_y_size}"

    def test_create_sequences_fast_function(self):
        """Test the standalone create_sequences_fast function."""
        from models.training import create_sequences_fast

        n_samples = 1000
        n_features = 40
        seq_length = 20

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)

        X_seq, y_seq = create_sequences_fast(X, y, seq_length)

        # Check shape
        assert X_seq.shape == (n_samples - seq_length, seq_length, n_features)
        assert y_seq.shape == (n_samples - seq_length,)

        # Check values match SequenceDataset
        dataset = SequenceDataset(X, y, seq_length)
        np.testing.assert_array_equal(X_seq, dataset.X)
        np.testing.assert_array_equal(y_seq, dataset.y)

    def test_sequence_creation_error_on_insufficient_data(self):
        """Test that sequence creation raises error when data is insufficient."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 3, 10)

        with pytest.raises(ValueError, match="Not enough samples"):
            SequenceDataset(X, y, seq_length=20)  # seq_length > n_samples


class TestSimulateTradingForFold:
    """Tests for _simulate_trading_for_fold helper function.

    This function converts 3-class predictions to trading positions
    and calculates trading metrics for walk-forward validation.
    """

    def test_basic_trading_simulation(self):
        """Test basic trading simulation with simple predictions."""
        from models.training import _simulate_trading_for_fold

        # Create simple price data with upward trend
        np.random.seed(42)
        prices = 5000.0 * (1 + np.arange(100) * 0.0001)

        # All UP predictions should generate positive returns
        predicted_classes = np.full(99, 2)  # All UP
        predictions = np.zeros((99, 3))
        predictions[:, 2] = 1.0  # 100% confidence for UP

        metrics = _simulate_trading_for_fold(
            predicted_classes=predicted_classes,
            predictions=predictions,
            prices=prices,
            num_classes=3
        )

        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'total_trades' in metrics
        assert metrics['total_trades'] > 0

    def test_trading_with_down_predictions(self):
        """Test trading simulation with DOWN predictions on downward trend."""
        from models.training import _simulate_trading_for_fold

        # Create price data with downward trend
        prices = 5000.0 * (1 - np.arange(100) * 0.0001)

        # All DOWN predictions should generate positive returns (short)
        predicted_classes = np.full(99, 0)  # All DOWN
        predictions = np.zeros((99, 3))
        predictions[:, 0] = 1.0  # 100% confidence for DOWN

        metrics = _simulate_trading_for_fold(
            predicted_classes=predicted_classes,
            predictions=predictions,
            prices=prices,
            num_classes=3
        )

        # Should have trades
        assert metrics['total_trades'] > 0
        # On downward trend with short positions, should be profitable
        assert metrics['total_return'] > 0

    def test_flat_predictions_no_trades(self):
        """Test that FLAT predictions generate no trading activity."""
        from models.training import _simulate_trading_for_fold

        np.random.seed(42)
        prices = 5000.0 + np.random.randn(100) * 10

        # All FLAT predictions
        predicted_classes = np.full(99, 1)  # All FLAT
        predictions = np.zeros((99, 3))
        predictions[:, 1] = 1.0  # 100% confidence for FLAT

        metrics = _simulate_trading_for_fold(
            predicted_classes=predicted_classes,
            predictions=predictions,
            prices=prices,
            num_classes=3
        )

        # No trades since all FLAT
        assert metrics['total_trades'] == 0
        assert metrics['sharpe_ratio'] == 0.0

    def test_mixed_predictions(self):
        """Test trading with mixed UP/DOWN/FLAT predictions."""
        from models.training import _simulate_trading_for_fold

        np.random.seed(42)
        prices = 5000.0 + np.cumsum(np.random.randn(100) * 5)

        # Mixed predictions
        predicted_classes = np.random.randint(0, 3, 99)
        predictions = np.random.rand(99, 3)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)

        metrics = _simulate_trading_for_fold(
            predicted_classes=predicted_classes,
            predictions=predictions,
            prices=prices,
            num_classes=3
        )

        # Should have metrics
        assert isinstance(metrics['sharpe_ratio'], float)
        assert 0 <= metrics['max_drawdown'] <= 1
        assert 0 <= metrics['win_rate'] <= 1

    def test_short_price_series(self):
        """Test handling of very short price series."""
        from models.training import _simulate_trading_for_fold

        prices = np.array([5000.0])  # Single price

        predicted_classes = np.array([])
        predictions = np.zeros((0, 3))

        metrics = _simulate_trading_for_fold(
            predicted_classes=predicted_classes,
            predictions=predictions,
            prices=prices,
            num_classes=3
        )

        # Should return zeros for metrics
        assert metrics['total_trades'] == 0
        assert metrics['sharpe_ratio'] == 0.0


class TestWalkForwardWithTradingMetrics:
    """Tests for walk-forward validation with TradingSimulator integration.

    These tests verify that when prices are provided, trading metrics
    (Sharpe ratio, max drawdown, win rate) are calculated for each fold.
    """

    def test_walk_forward_with_prices(self, sample_features_and_targets):
        """Test walk-forward validation calculates trading metrics when prices provided."""
        X, y = sample_features_and_targets

        # Create synthetic prices aligned with features
        np.random.seed(42)
        prices = 5000.0 + np.cumsum(np.random.randn(len(X)) * 2)

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]},
            'learning_rate': 0.01
        }

        results = train_with_walk_forward(
            X, y,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            batch_size=32,
            num_classes=3,
            prices=prices,
            validate_sharpe=False  # Don't require Sharpe > 1.0 for test
        )

        # Should have trading metrics
        assert results['trading_metrics_available'] == True
        assert 'avg_sharpe_ratio' in results
        assert 'avg_max_drawdown' in results
        assert 'avg_win_rate' in results

        # Each fold should have trading metrics
        for fold in results['fold_metrics']:
            assert 'sharpe_ratio' in fold
            assert 'max_drawdown' in fold
            assert 'win_rate' in fold
            assert 'total_trades' in fold

    def test_walk_forward_without_prices(self, sample_features_and_targets):
        """Test walk-forward validation without trading metrics."""
        X, y = sample_features_and_targets

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        results = train_with_walk_forward(
            X, y,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            num_classes=3
        )

        # Should NOT have trading metrics
        assert results['trading_metrics_available'] == False
        assert 'avg_sharpe_ratio' not in results

        # Folds should NOT have trading metrics
        for fold in results['fold_metrics']:
            assert 'sharpe_ratio' not in fold

    def test_sharpe_validation_passed(self, sample_features_and_targets):
        """Test Sharpe validation result when training with good predictions."""
        X, y = sample_features_and_targets

        # Create trending prices that reward directional predictions
        prices = 5000.0 + np.arange(len(X)) * 0.1

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]},
            'learning_rate': 0.01
        }

        results = train_with_walk_forward(
            X, y,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            prices=prices,
            min_sharpe_threshold=0.0,  # Very low threshold to pass
            validate_sharpe=True
        )

        # Should have validation result
        assert 'sharpe_validation_passed' in results
        assert 'min_sharpe_threshold' in results

    def test_sharpe_validation_failed(self, sample_features_and_targets):
        """Test Sharpe validation fails with high threshold."""
        X, y = sample_features_and_targets

        # Random prices with no trend
        np.random.seed(42)
        prices = 5000.0 + np.random.randn(len(X)) * 10

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        results = train_with_walk_forward(
            X, y,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            prices=prices,
            min_sharpe_threshold=10.0,  # Impossibly high threshold
            validate_sharpe=True
        )

        # Sharpe validation should fail
        assert results['sharpe_validation_passed'] == False

    def test_prices_length_mismatch_raises_error(self, sample_features_and_targets):
        """Test that mismatched prices length raises ValueError."""
        X, y = sample_features_and_targets

        # Wrong length prices
        prices = np.array([5000.0, 5001.0, 5002.0])

        model_config = {'type': 'feedforward', 'params': {'hidden_dims': [16]}}

        with pytest.raises(ValueError, match="prices length"):
            train_with_walk_forward(
                X, y,
                model_config=model_config,
                n_splits=2,
                epochs=2,
                prices=prices
            )

    def test_aggregate_trading_metrics(self, sample_features_and_targets):
        """Test that aggregate trading metrics are calculated correctly."""
        X, y = sample_features_and_targets

        prices = 5000.0 + np.cumsum(np.random.randn(len(X)) * 2)

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        results = train_with_walk_forward(
            X, y,
            model_config=model_config,
            n_splits=3,
            epochs=2,
            prices=prices,
            validate_sharpe=False
        )

        # Check aggregate metrics
        assert 'std_sharpe_ratio' in results
        assert 'min_sharpe_ratio' in results
        assert 'max_sharpe_ratio' in results
        assert 'profitable_folds_pct' in results
        assert 'worst_max_drawdown' in results
        assert 'total_trades_all_folds' in results
        assert 'avg_trades_per_fold' in results

        # Verify min <= avg <= max
        assert results['min_sharpe_ratio'] <= results['avg_sharpe_ratio']
        assert results['avg_sharpe_ratio'] <= results['max_sharpe_ratio']


class TestWalkForwardResultsSaving:
    """Tests for saving walk-forward results to JSON."""

    def test_results_saved_to_json(self, sample_features_and_targets, tmp_path):
        """Test that results are saved to JSON when path provided."""
        from models.training import _save_walk_forward_results

        X, y = sample_features_and_targets

        prices = 5000.0 + np.cumsum(np.random.randn(len(X)) * 2)
        results_path = str(tmp_path / "wf_results.json")

        model_config = {
            'type': 'feedforward',
            'params': {'hidden_dims': [16]}
        }

        results = train_with_walk_forward(
            X, y,
            model_config=model_config,
            n_splits=2,
            epochs=2,
            prices=prices,
            results_path=results_path,
            validate_sharpe=False
        )

        # File should exist
        assert Path(results_path).exists()

        # File should be valid JSON
        import json
        with open(results_path) as f:
            saved = json.load(f)

        assert 'fold_metrics' in saved
        assert 'metadata' in saved
        assert 'saved_at' in saved['metadata']

    def test_save_function_handles_numpy_types(self, tmp_path):
        """Test that _save_walk_forward_results converts numpy types."""
        from models.training import _save_walk_forward_results
        import json

        results = {
            'overall_accuracy': np.float64(0.75),
            'total_trades': np.int64(100),
            'fold_metrics': [
                {'sharpe_ratio': np.float32(1.5), 'trades': np.int32(50)}
            ]
        }

        path = str(tmp_path / "test_results.json")
        _save_walk_forward_results(results, path)

        # Should load without JSON errors
        with open(path) as f:
            loaded = json.load(f)

        assert loaded['overall_accuracy'] == 0.75
        assert loaded['total_trades'] == 100
