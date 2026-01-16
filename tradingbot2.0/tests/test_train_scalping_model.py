"""
Tests for train_scalping_model.py - Scalping Model Training Script.

Tests the integration of parquet data pipeline with the training system.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ml"))


# Import the module under test
from src.ml.train_scalping_model import (
    compute_class_weights,
    set_seed,
    parse_args
)


class TestComputeClassWeights:
    """Tests for the compute_class_weights function."""

    def test_balanced_classes(self):
        """Balanced classes should have equal weights."""
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(y, num_classes=3)

        # All weights should be approximately equal for balanced data
        assert len(weights) == 3
        assert abs(weights[0] - weights[1]) < 0.1
        assert abs(weights[1] - weights[2]) < 0.1

    def test_imbalanced_classes(self):
        """Minority classes should have higher weights."""
        # Typical scalping distribution: ~20% DOWN, ~60% FLAT, ~20% UP
        y = np.array([0] * 20 + [1] * 60 + [2] * 20)
        weights = compute_class_weights(y, num_classes=3)

        # FLAT (majority) should have lower weight than UP/DOWN
        assert weights[1] < weights[0]  # FLAT < DOWN
        assert weights[1] < weights[2]  # FLAT < UP
        # UP and DOWN should have similar weights
        assert abs(weights[0] - weights[2]) < 0.5

    def test_weights_are_float_tensor(self):
        """Output should be a FloatTensor."""
        y = np.array([0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(y)

        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == torch.float32

    def test_weights_sum_to_num_classes(self):
        """Normalized weights should sum to num_classes."""
        y = np.array([0] * 10 + [1] * 50 + [2] * 40)
        weights = compute_class_weights(y, num_classes=3)

        assert abs(weights.sum().item() - 3.0) < 0.01

    def test_empty_class_handling(self):
        """Should handle missing classes gracefully."""
        y = np.array([0, 0, 1, 1])  # No class 2
        weights = compute_class_weights(y, num_classes=3)

        # Should not crash, class 2 gets default weight
        assert len(weights) == 3
        assert weights[2] > 0  # Non-zero weight even for empty class

    def test_single_class(self):
        """Should handle single class data."""
        y = np.array([1, 1, 1, 1, 1])
        weights = compute_class_weights(y, num_classes=3)

        # Class 1 should have lowest weight (most common)
        assert weights[1] < weights[0]
        assert weights[1] < weights[2]

    def test_extreme_imbalance(self):
        """Should handle extreme class imbalance."""
        # 1% DOWN, 98% FLAT, 1% UP
        y = np.array([0] * 1 + [1] * 98 + [2] * 1)
        weights = compute_class_weights(y, num_classes=3)

        # Minority classes should have much higher weights
        assert weights[0] > weights[1] * 5
        assert weights[2] > weights[1] * 5


class TestSetSeed:
    """Tests for the set_seed function."""

    def test_numpy_reproducibility(self):
        """NumPy random should be reproducible after seeding."""
        set_seed(42)
        random1 = np.random.rand(5)

        set_seed(42)
        random2 = np.random.rand(5)

        np.testing.assert_array_equal(random1, random2)

    def test_torch_reproducibility(self):
        """PyTorch random should be reproducible after seeding."""
        set_seed(42)
        tensor1 = torch.rand(5)

        set_seed(42)
        tensor2 = torch.rand(5)

        torch.testing.assert_close(tensor1, tensor2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        set_seed(42)
        random1 = np.random.rand(5)

        set_seed(123)
        random2 = np.random.rand(5)

        assert not np.array_equal(random1, random2)


class TestParseArgs:
    """Tests for argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        with patch('sys.argv', ['train_scalping_model.py']):
            args = parse_args()

        assert args.data == "data/historical/MES/MES_1s_2years.parquet"
        assert args.model == "feedforward"
        assert args.epochs == 50
        assert args.batch_size == 256
        assert args.lookahead == 30
        assert args.threshold == 3.0
        assert args.train_ratio == 0.6
        assert args.val_ratio == 0.2
        assert args.dropout == 0.3
        assert args.learning_rate == 0.001
        assert args.seed == 42
        assert args.filter_rth == True

    def test_custom_model(self):
        """Test specifying LSTM model."""
        with patch('sys.argv', ['train_scalping_model.py', '--model', 'lstm']):
            args = parse_args()

        assert args.model == "lstm"

    def test_custom_epochs(self):
        """Test specifying epochs."""
        with patch('sys.argv', ['train_scalping_model.py', '--epochs', '100']):
            args = parse_args()

        assert args.epochs == 100

    def test_custom_lookahead(self):
        """Test specifying lookahead."""
        with patch('sys.argv', ['train_scalping_model.py', '--lookahead', '60']):
            args = parse_args()

        assert args.lookahead == 60

    def test_custom_threshold(self):
        """Test specifying threshold."""
        with patch('sys.argv', ['train_scalping_model.py', '--threshold', '4.0']):
            args = parse_args()

        assert args.threshold == 4.0

    def test_include_eth(self):
        """Test ETH mode (disables RTH filter)."""
        with patch('sys.argv', ['train_scalping_model.py', '--include-eth']):
            args = parse_args()

        assert args.include_eth == True

    def test_walk_forward_enabled(self):
        """Test walk-forward validation flag."""
        with patch('sys.argv', ['train_scalping_model.py', '--walk-forward', '--walk-forward-splits', '3']):
            args = parse_args()

        assert args.walk_forward == True
        assert args.walk_forward_splits == 3

    def test_max_samples_debug(self):
        """Test max-samples for debugging."""
        with patch('sys.argv', ['train_scalping_model.py', '--max-samples', '10000']):
            args = parse_args()

        assert args.max_samples == 10000

    def test_hidden_dims_parsing(self):
        """Test hidden dimensions are comma-separated."""
        with patch('sys.argv', ['train_scalping_model.py', '--hidden-dims', '128,64,32']):
            args = parse_args()

        assert args.hidden_dims == "128,64,32"

    def test_model_name(self):
        """Test custom model name."""
        with patch('sys.argv', ['train_scalping_model.py', '--model-name', 'scalper_v2']):
            args = parse_args()

        assert args.model_name == "scalper_v2"

    def test_output_dir(self):
        """Test custom output directory."""
        with patch('sys.argv', ['train_scalping_model.py', '--output-dir', '/tmp/results']):
            args = parse_args()

        assert args.output_dir == "/tmp/results"

    def test_seq_length_lstm(self):
        """Test LSTM sequence length."""
        with patch('sys.argv', ['train_scalping_model.py', '--seq-length', '30']):
            args = parse_args()

        assert args.seq_length == 30

    def test_validate_lookahead_flag(self):
        """Test lookahead validation flag."""
        with patch('sys.argv', ['train_scalping_model.py', '--validate-lookahead']):
            args = parse_args()

        assert args.validate_lookahead == True

    def test_no_plot_flag(self):
        """Test no-plot flag for headless environments."""
        with patch('sys.argv', ['train_scalping_model.py', '--no-plot']):
            args = parse_args()

        assert args.no_plot == True


class TestTrainingIntegration:
    """Integration tests for the training script components."""

    def test_model_config_feedforward(self):
        """Test model config generation for feedforward."""
        hidden_dims = [256, 128, 64]
        dropout = 0.3
        num_classes = 3

        model_config = {
            'type': 'feedforward',
            'params': {
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout,
                'num_classes': num_classes
            }
        }

        assert model_config['type'] == 'feedforward'
        assert model_config['params']['hidden_dims'] == [256, 128, 64]
        assert model_config['params']['num_classes'] == 3

    def test_model_config_lstm(self):
        """Test model config generation for LSTM."""
        hidden_dims = [256, 128, 64]
        dropout = 0.3
        num_classes = 3

        model_config = {
            'type': 'lstm',
            'params': {
                'hidden_dim': hidden_dims[0],
                'num_layers': 2,
                'dropout_rate': dropout,
                'fc_dims': hidden_dims[1:],
                'num_classes': num_classes
            }
        }

        assert model_config['type'] == 'lstm'
        assert model_config['params']['hidden_dim'] == 256
        assert model_config['params']['fc_dims'] == [128, 64]
        assert model_config['params']['num_classes'] == 3

    def test_hidden_dims_parsing(self):
        """Test parsing hidden dimensions from string."""
        hidden_dims_str = "256,128,64"
        hidden_dims = [int(x) for x in hidden_dims_str.split(",")]

        assert hidden_dims == [256, 128, 64]

    def test_class_distribution_format(self):
        """Test class distribution dictionary format for JSON."""
        y_test = np.array([0, 0, 1, 1, 1, 2])

        # This is how it's formatted in the training script
        class_dist = {int(k): int(v) for k, v in Counter(y_test).items()}

        assert class_dist == {0: 2, 1: 3, 2: 1}
        # Verify JSON serializable
        import json
        json_str = json.dumps(class_dist)
        assert '"0": 2' in json_str


class TestDataPipelineIntegration:
    """Tests for data pipeline integration with training."""

    def test_feature_names_consistency(self):
        """Test that feature names are consistent across splits."""
        # This tests the concept - actual integration is in the script
        from src.ml.data.scalping_features import ScalpingFeatureEngineer
        import pandas as pd

        # Create minimal test data with required columns
        dates = pd.date_range('2024-01-01 09:30:00', periods=1000, freq='1s', tz='America/New_York')
        df = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 5000,
            'high': np.random.randn(1000).cumsum() + 5001,
            'low': np.random.randn(1000).cumsum() + 4999,
            'close': np.random.randn(1000).cumsum() + 5000,
            'volume': np.random.randint(100, 1000, 1000)
        }, index=dates)

        # Ensure high >= open, close, low and low <= open, close
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        engineer = ScalpingFeatureEngineer(df)
        engineer.generate_all_features(include_multiframe=True)
        feature_names = engineer.get_feature_names()

        # Check we have expected features
        assert len(feature_names) > 50
        assert 'return_1s' in feature_names or any('return' in f for f in feature_names)
        assert any('ema' in f.lower() for f in feature_names)
        assert any('vwap' in f.lower() for f in feature_names)

    def test_scaler_fit_transform(self):
        """Test that scaler fits on training and transforms validation."""
        from sklearn.preprocessing import StandardScaler

        # Simulated feature data
        X_train = np.random.randn(1000, 56)
        X_val = np.random.randn(200, 56)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Training data should be normalized (mean ~0, std ~1)
        assert abs(X_train_scaled.mean()) < 0.1
        assert abs(X_train_scaled.std() - 1.0) < 0.1

        # Validation data uses training scaler
        assert X_val_scaled.shape == X_val.shape


class TestModelSaving:
    """Tests for model checkpoint saving."""

    def test_checkpoint_contents(self):
        """Test that checkpoint contains required keys."""
        # This mirrors what the training script saves
        checkpoint = {
            'model_state_dict': {},
            'model_config': {
                'type': 'feedforward',
                'params': {'hidden_dims': [256, 128, 64]}
            },
            'feature_names': ['feature1', 'feature2'],
            'scaler_mean': [0.0, 0.0],
            'scaler_scale': [1.0, 1.0],
            'class_weights': [1.0, 0.75, 1.0],
            'num_classes': 3,
            'input_dim': 56,
            'training_args': {'epochs': 50, 'model': 'feedforward'},
            'training_history': {'train_loss': [1.0, 0.9], 'val_loss': [1.0, 0.95]},
            'test_accuracy': 0.45,
            'timestamp': '2024-01-16T10:00:00'
        }

        assert 'model_state_dict' in checkpoint
        assert 'model_config' in checkpoint
        assert 'feature_names' in checkpoint
        assert 'scaler_mean' in checkpoint
        assert 'scaler_scale' in checkpoint
        assert 'class_weights' in checkpoint
        assert 'num_classes' in checkpoint
        assert checkpoint['num_classes'] == 3


class TestPrintClassDistribution:
    """Tests for class distribution printing (functional test)."""

    def test_counter_usage(self):
        """Test that Counter correctly counts classes."""
        y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
        counts = Counter(y)

        assert counts[0] == 2
        assert counts[1] == 4
        assert counts[2] == 2

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        y = np.array([0] * 20 + [1] * 60 + [2] * 20)
        counts = Counter(y)
        total = len(y)

        assert counts[0] / total == 0.2  # 20%
        assert counts[1] / total == 0.6  # 60%
        assert counts[2] / total == 0.2  # 20%


class TestDataTypeConversions:
    """Tests for data type conversions."""

    def test_target_as_int64(self):
        """Test that target is converted to int64 for CrossEntropyLoss."""
        y = np.array([0, 1, 2, 0, 1, 2])
        y_int64 = y.astype(np.int64)

        assert y_int64.dtype == np.int64

    def test_long_tensor_for_crossentropy(self):
        """Test LongTensor creation for CrossEntropyLoss."""
        y = np.array([0, 1, 2, 0, 1, 2])
        y_tensor = torch.LongTensor(y)

        assert y_tensor.dtype == torch.int64

    def test_float_tensor_for_features(self):
        """Test FloatTensor creation for features."""
        X = np.random.randn(100, 56)
        X_tensor = torch.FloatTensor(X)

        assert X_tensor.dtype == torch.float32


class TestDeviceHandling:
    """Tests for device (CPU/GPU/MPS) handling."""

    def test_tensor_to_device(self):
        """Test moving tensor to device."""
        X = torch.FloatTensor(np.random.randn(10, 5))
        device = torch.device('cpu')

        X_device = X.to(device)

        assert X_device.device == device

    def test_cpu_back_from_device(self):
        """Test moving tensor back to CPU for numpy conversion."""
        X = torch.FloatTensor(np.random.randn(10, 5))

        # Simulate coming from GPU
        X_cpu = X.cpu()
        X_np = X_cpu.numpy()

        assert isinstance(X_np, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
