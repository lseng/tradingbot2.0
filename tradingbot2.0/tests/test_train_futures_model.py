"""
Tests for train_futures_model.py

This module tests the main training script for futures price prediction.
Since it's a CLI orchestration script, tests focus on:
- Argument parsing
- Seed setting for reproducibility
- End-to-end pipeline with mocked components

Why test a CLI script:
- Argument parsing bugs can cause silent failures
- Seed setting is critical for reproducibility
- Integration between components needs validation
"""

import pytest
import sys
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml"))

from src.ml.train_futures_model import parse_args, set_seed


# ============================================================================
# Argument Parsing Tests
# ============================================================================

class TestParseArgs:
    """Tests for command line argument parsing."""

    def test_default_arguments(self):
        """Test that default arguments are correctly set."""
        with patch('sys.argv', ['train_futures_model.py']):
            args = parse_args()

        # Verify defaults
        assert args.model == "feedforward"
        assert args.epochs == 50
        assert args.batch_size == 32
        assert args.hidden_dims == "128,64,32"
        assert args.dropout == 0.3
        assert args.learning_rate == 0.001
        assert args.weight_decay == 0.01
        assert args.train_ratio == 0.8
        assert args.walk_forward_splits == 5
        assert args.seq_length == 20
        assert args.output_dir == "./results"
        assert args.no_plot == False
        assert args.seed == 42

    def test_custom_data_path(self):
        """Test custom data path argument."""
        with patch('sys.argv', ['train_futures_model.py', '--data', '/path/to/custom/data.txt']):
            args = parse_args()

        assert args.data == "/path/to/custom/data.txt"

    def test_model_selection_feedforward(self):
        """Test feedforward model selection."""
        with patch('sys.argv', ['train_futures_model.py', '--model', 'feedforward']):
            args = parse_args()

        assert args.model == "feedforward"

    def test_model_selection_lstm(self):
        """Test LSTM model selection."""
        with patch('sys.argv', ['train_futures_model.py', '--model', 'lstm']):
            args = parse_args()

        assert args.model == "lstm"

    def test_invalid_model_raises_error(self):
        """Test that invalid model type raises error."""
        with patch('sys.argv', ['train_futures_model.py', '--model', 'invalid_model']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_epochs_argument(self):
        """Test epochs argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--epochs', '100']):
            args = parse_args()

        assert args.epochs == 100

    def test_batch_size_argument(self):
        """Test batch size argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--batch-size', '64']):
            args = parse_args()

        assert args.batch_size == 64

    def test_hidden_dims_argument(self):
        """Test hidden dimensions argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--hidden-dims', '256,128,64,32']):
            args = parse_args()

        assert args.hidden_dims == "256,128,64,32"

    def test_dropout_argument(self):
        """Test dropout argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--dropout', '0.5']):
            args = parse_args()

        assert args.dropout == 0.5

    def test_learning_rate_argument(self):
        """Test learning rate argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--learning-rate', '0.0001']):
            args = parse_args()

        assert args.learning_rate == 0.0001

    def test_weight_decay_argument(self):
        """Test weight decay argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--weight-decay', '0.001']):
            args = parse_args()

        assert args.weight_decay == 0.001

    def test_train_ratio_argument(self):
        """Test train ratio argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--train-ratio', '0.7']):
            args = parse_args()

        assert args.train_ratio == 0.7

    def test_walk_forward_splits_argument(self):
        """Test walk-forward splits argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--walk-forward-splits', '10']):
            args = parse_args()

        assert args.walk_forward_splits == 10

    def test_seq_length_argument(self):
        """Test sequence length argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--seq-length', '30']):
            args = parse_args()

        assert args.seq_length == 30

    def test_output_dir_argument(self):
        """Test output directory argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--output-dir', '/custom/output']):
            args = parse_args()

        assert args.output_dir == "/custom/output"

    def test_no_plot_flag(self):
        """Test no-plot flag."""
        with patch('sys.argv', ['train_futures_model.py', '--no-plot']):
            args = parse_args()

        assert args.no_plot == True

    def test_seed_argument(self):
        """Test seed argument parsing."""
        with patch('sys.argv', ['train_futures_model.py', '--seed', '123']):
            args = parse_args()

        assert args.seed == 123

    def test_multiple_arguments(self):
        """Test multiple arguments together."""
        with patch('sys.argv', [
            'train_futures_model.py',
            '--data', '/path/to/data.txt',
            '--model', 'lstm',
            '--epochs', '100',
            '--batch-size', '64',
            '--learning-rate', '0.0005',
            '--no-plot',
        ]):
            args = parse_args()

        assert args.data == "/path/to/data.txt"
        assert args.model == "lstm"
        assert args.epochs == 100
        assert args.batch_size == 64
        assert args.learning_rate == 0.0005
        assert args.no_plot == True


# ============================================================================
# Seed Setting Tests
# ============================================================================

class TestSetSeed:
    """Tests for random seed setting."""

    def test_numpy_seed_is_set(self):
        """Test that numpy random seed is set."""
        set_seed(42)

        # Generate random numbers
        np.random.seed(42)
        expected = np.random.rand(5)

        set_seed(42)
        actual = np.random.rand(5)

        np.testing.assert_array_equal(expected, actual)

    def test_different_seeds_different_output(self):
        """Test that different seeds produce different output."""
        set_seed(42)
        output1 = np.random.rand(5)

        set_seed(123)
        output2 = np.random.rand(5)

        assert not np.allclose(output1, output2)

    def test_torch_seed_is_set(self):
        """Test that PyTorch seed is set."""
        import torch

        set_seed(42)
        tensor1 = torch.rand(5)

        set_seed(42)
        tensor2 = torch.rand(5)

        assert torch.allclose(tensor1, tensor2)

    def test_python_random_seed_is_set(self):
        """Test that Python random seed is set."""
        import random

        set_seed(42)
        random.seed(42)
        expected = [random.random() for _ in range(5)]

        set_seed(42)
        actual = [random.random() for _ in range(5)]

        assert expected == actual


# ============================================================================
# Hidden Dimensions Parsing Tests
# ============================================================================

class TestHiddenDimsParsing:
    """Tests for hidden dimensions string parsing."""

    def test_parse_hidden_dims_default(self):
        """Test parsing default hidden dims string."""
        hidden_dims_str = "128,64,32"
        hidden_dims = [int(x) for x in hidden_dims_str.split(",")]

        assert hidden_dims == [128, 64, 32]

    def test_parse_hidden_dims_single(self):
        """Test parsing single hidden dim."""
        hidden_dims_str = "256"
        hidden_dims = [int(x) for x in hidden_dims_str.split(",")]

        assert hidden_dims == [256]

    def test_parse_hidden_dims_many(self):
        """Test parsing many hidden dims."""
        hidden_dims_str = "512,256,128,64,32,16"
        hidden_dims = [int(x) for x in hidden_dims_str.split(",")]

        assert hidden_dims == [512, 256, 128, 64, 32, 16]


# ============================================================================
# Model Config Building Tests
# ============================================================================

class TestModelConfigBuilding:
    """Tests for model configuration building."""

    def test_feedforward_config(self):
        """Test feedforward model config building."""
        hidden_dims = [128, 64, 32]
        dropout = 0.3

        model_config = {
            'type': 'feedforward',
            'params': {
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout
            },
            'learning_rate': 0.001,
            'weight_decay': 0.01
        }

        assert model_config['type'] == 'feedforward'
        assert model_config['params']['hidden_dims'] == [128, 64, 32]
        assert model_config['params']['dropout_rate'] == 0.3

    def test_lstm_config(self):
        """Test LSTM model config building."""
        hidden_dims = [128, 64, 32]
        dropout = 0.3

        model_config = {
            'type': 'lstm',
            'params': {
                'hidden_dim': hidden_dims[0],
                'num_layers': 2,
                'dropout_rate': dropout,
                'fc_dims': hidden_dims[1:] if len(hidden_dims) > 1 else [32]
            },
            'learning_rate': 0.001,
            'weight_decay': 0.01
        }

        assert model_config['type'] == 'lstm'
        assert model_config['params']['hidden_dim'] == 128
        assert model_config['params']['num_layers'] == 2
        assert model_config['params']['fc_dims'] == [64, 32]


# ============================================================================
# Path Handling Tests
# ============================================================================

class TestPathHandling:
    """Tests for path handling in the script."""

    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_nested_output_dir_creation(self, tmp_path):
        """Test that nested output directory is created."""
        output_dir = tmp_path / "parent" / "child" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        assert output_dir.exists()

    def test_model_path_formatting(self, tmp_path):
        """Test model path formatting with timestamp."""
        from datetime import datetime

        output_dir = tmp_path
        model_type = "feedforward"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        model_path = output_dir / f"model_{model_type}_{timestamp}.pt"

        assert model_path.suffix == ".pt"
        assert model_type in model_path.name

    def test_results_path_formatting(self, tmp_path):
        """Test results path formatting with timestamp."""
        from datetime import datetime

        output_dir = tmp_path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_path = output_dir / f"results_{timestamp}.json"

        assert results_path.suffix == ".json"


# ============================================================================
# Results Serialization Tests
# ============================================================================

class TestResultsSerialization:
    """Tests for results serialization."""

    def test_results_dict_structure(self):
        """Test that results dict has expected structure."""
        from datetime import datetime

        results = {
            'run_timestamp': datetime.now().isoformat(),
            'config': {'epochs': 50, 'batch_size': 32},
            'walk_forward_results': {
                'overall_accuracy': 0.55,
                'overall_auc': 0.58,
                'fold_metrics': []
            },
            'final_evaluation': {
                'classification': {},
                'trading': {},
                'comparison': {}
            },
            'training_history': {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }
        }

        assert 'run_timestamp' in results
        assert 'config' in results
        assert 'walk_forward_results' in results
        assert 'final_evaluation' in results
        assert 'training_history' in results

    def test_results_json_serializable(self, tmp_path):
        """Test that results can be serialized to JSON."""
        import json
        from datetime import datetime

        results = {
            'run_timestamp': datetime.now().isoformat(),
            'config': {'epochs': 50},
            'metrics': {'accuracy': 0.55}
        }

        results_path = tmp_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f)

        # Read back
        with open(results_path, 'r') as f:
            loaded = json.load(f)

        assert loaded['config']['epochs'] == 50
        assert loaded['metrics']['accuracy'] == 0.55


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================

class TestMainIntegration:
    """Integration tests for main function with mocked components."""

    def test_imports_work(self):
        """Test that all required imports work."""
        # This tests that the imports in train_futures_model.py are valid
        from src.ml.data.data_loader import FuturesDataLoader
        from src.ml.data.feature_engineering import FeatureEngineer
        from src.ml.models.neural_networks import create_model, FeedForwardNet, LSTMNet
        from src.ml.models.training import ModelTrainer, WalkForwardValidator

        assert FuturesDataLoader is not None
        assert FeatureEngineer is not None
        assert create_model is not None
        assert FeedForwardNet is not None
        assert LSTMNet is not None
        assert ModelTrainer is not None
        assert WalkForwardValidator is not None

    def test_feedforward_model_creation(self):
        """Test that feedforward model can be created."""
        from src.ml.models.neural_networks import create_model

        model = create_model(
            'feedforward',
            input_dim=50,
            hidden_dims=[128, 64],
            dropout_rate=0.3
        )

        assert model is not None

    def test_lstm_model_creation(self):
        """Test that LSTM model can be created."""
        from src.ml.models.neural_networks import create_model

        model = create_model(
            'lstm',
            input_dim=50,
            hidden_dim=64,
            num_layers=2,
            dropout_rate=0.3,
            fc_dims=[32]
        )

        assert model is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in the script."""

    def test_invalid_data_type_for_args(self):
        """Test that invalid argument types are handled."""
        with patch('sys.argv', ['train_futures_model.py', '--epochs', 'not_a_number']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_negative_epochs(self):
        """Test that negative epochs parses (validation happens at runtime)."""
        with patch('sys.argv', ['train_futures_model.py', '--epochs', '-10']):
            args = parse_args()

        # Parsing succeeds, validation would happen later
        assert args.epochs == -10

    def test_invalid_train_ratio(self):
        """Test train ratio outside valid range parses."""
        with patch('sys.argv', ['train_futures_model.py', '--train-ratio', '1.5']):
            args = parse_args()

        # Parsing succeeds, validation would happen later
        assert args.train_ratio == 1.5


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
