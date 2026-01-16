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
import pandas as pd

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
# Main Function Tests with Full Mocking
# ============================================================================

class TestMainFunction:
    """Tests for the main() function with mocked components.

    Why test main():
    - It's the core orchestration logic (lines 191-475)
    - Without tests, 77% of train_futures_model.py is untested
    - Tests verify the pipeline steps execute in correct order
    - Catches integration issues between components early
    """

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        n_samples = 200
        n_features = 10

        # Create mock DataFrame with required columns
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        df = pd.DataFrame({
            'open': np.random.randn(n_samples) * 10 + 100,
            'high': np.random.randn(n_samples) * 10 + 105,
            'low': np.random.randn(n_samples) * 10 + 95,
            'close': np.random.randn(n_samples) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'target': np.random.randint(0, 2, n_samples),
            'next_return': np.random.randn(n_samples) * 0.01
        }, index=dates)

        # Add feature columns
        for i in range(n_features):
            df[f'feature_{i}'] = np.random.randn(n_samples)

        return df, [f'feature_{i}' for i in range(n_features)]

    @pytest.fixture
    def mock_components(self, mock_data, tmp_path):
        """Create mocked components for main() testing."""
        import torch
        import pandas as pd

        df, feature_names = mock_data

        # Mock FuturesDataLoader
        mock_loader = MagicMock()
        mock_loader.load_raw_data.return_value = df
        mock_loader.resample_to_daily.return_value = df
        mock_loader.create_target_variable.return_value = df

        # Mock FeatureEngineer
        mock_engineer = MagicMock()
        mock_engineer.generate_all_features.return_value = df
        mock_engineer.get_feature_names.return_value = feature_names

        # Mock walk-forward results
        mock_wf_results = {
            'overall_accuracy': 0.55,
            'overall_auc': 0.58,
            'fold_metrics': [
                {'fold': 1, 'test_accuracy': 0.54, 'test_loss': 0.69},
                {'fold': 2, 'test_accuracy': 0.56, 'test_loss': 0.68},
            ]
        }

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            'train_loss': [0.7, 0.65, 0.6],
            'val_loss': [0.72, 0.68, 0.65],
            'train_acc': [0.50, 0.55, 0.58],
            'val_acc': [0.48, 0.52, 0.55]
        }
        mock_trainer.predict.return_value = np.random.rand(40)

        # Mock evaluation results
        mock_eval_results = {
            'classification': {
                'accuracy': 0.55,
                'auc_roc': 0.58,
                'precision': 0.54,
                'recall': 0.56,
                'f1': 0.55
            },
            'trading': {
                'total_return': 0.10,
                'sharpe_ratio': 0.8,
                'max_drawdown': 0.05
            },
            'comparison': {
                'buy_hold_return': 0.08,
                'excess_return': 0.02
            }
        }

        return {
            'loader': mock_loader,
            'engineer': mock_engineer,
            'wf_results': mock_wf_results,
            'trainer': mock_trainer,
            'eval_results': mock_eval_results,
            'tmp_path': tmp_path
        }

    @patch('src.ml.train_futures_model.FuturesDataLoader')
    @patch('src.ml.train_futures_model.FeatureEngineer')
    @patch('src.ml.train_futures_model.train_with_walk_forward')
    @patch('src.ml.train_futures_model.create_model')
    @patch('src.ml.train_futures_model.ModelTrainer')
    @patch('src.ml.train_futures_model.evaluate_model_and_strategy')
    @patch('src.ml.train_futures_model.print_evaluation_report')
    @patch('src.ml.train_futures_model.plot_results')
    def test_main_feedforward_full_pipeline(
        self, mock_plot, mock_print_report, mock_evaluate,
        mock_trainer_class, mock_create_model, mock_wf_train,
        mock_engineer_class, mock_loader_class,
        mock_components, tmp_path
    ):
        """Test main() executes all steps for feedforward model.

        Why: Ensures all 7 pipeline steps execute in order without errors.
        """
        from src.ml.train_futures_model import main
        import pandas as pd

        # Setup mocks
        n_samples = 200
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame({
            'open': np.random.randn(n_samples) * 10 + 100,
            'high': np.random.randn(n_samples) * 10 + 105,
            'low': np.random.randn(n_samples) * 10 + 95,
            'close': np.random.randn(n_samples) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'target': np.random.randint(0, 2, n_samples),
            'next_return': np.random.randn(n_samples) * 0.01
        }, index=dates)
        for name in feature_names:
            df[name] = np.random.randn(n_samples)

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_raw_data.return_value = df
        mock_loader_instance.resample_to_daily.return_value = df
        mock_loader_instance.create_target_variable.return_value = df
        mock_loader_class.return_value = mock_loader_instance

        mock_engineer_instance = MagicMock()
        mock_engineer_instance.generate_all_features.return_value = df
        mock_engineer_instance.get_feature_names.return_value = feature_names
        mock_engineer_class.return_value = mock_engineer_instance

        mock_wf_train.return_value = mock_components['wf_results']

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_create_model.return_value = mock_model

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = mock_components['trainer'].train.return_value
        mock_trainer_instance.predict.return_value = np.random.rand(40)
        mock_trainer_class.return_value = mock_trainer_instance

        mock_evaluate.return_value = mock_components['eval_results']

        # Run main with mocked args
        with patch('sys.argv', [
            'train_futures_model.py',
            '--data', str(tmp_path / 'test_data.txt'),
            '--model', 'feedforward',
            '--epochs', '1',
            '--output-dir', str(tmp_path),
            '--no-plot'
        ]):
            main()

        # Verify all steps were called
        mock_loader_class.assert_called_once()
        mock_loader_instance.load_raw_data.assert_called_once()
        mock_engineer_class.assert_called_once()
        mock_engineer_instance.generate_all_features.assert_called_once()
        mock_wf_train.assert_called_once()
        mock_create_model.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_print_report.assert_called_once()

    @patch('src.ml.train_futures_model.FuturesDataLoader')
    @patch('src.ml.train_futures_model.FeatureEngineer')
    @patch('src.ml.train_futures_model.train_with_walk_forward')
    @patch('src.ml.train_futures_model.create_model')
    @patch('src.ml.train_futures_model.ModelTrainer')
    @patch('src.ml.train_futures_model.evaluate_model_and_strategy')
    @patch('src.ml.train_futures_model.print_evaluation_report')
    def test_main_lstm_pipeline(
        self, mock_print_report, mock_evaluate,
        mock_trainer_class, mock_create_model, mock_wf_train,
        mock_engineer_class, mock_loader_class,
        mock_components, tmp_path
    ):
        """Test main() with LSTM model uses SequenceDataset.

        Why: LSTM requires different data preparation (sequences) than feedforward.
        """
        from src.ml.train_futures_model import main
        import pandas as pd

        # Setup mocks
        n_samples = 200
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame({
            'open': np.random.randn(n_samples) * 10 + 100,
            'high': np.random.randn(n_samples) * 10 + 105,
            'low': np.random.randn(n_samples) * 10 + 95,
            'close': np.random.randn(n_samples) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'target': np.random.randint(0, 2, n_samples),
            'next_return': np.random.randn(n_samples) * 0.01
        }, index=dates)
        for name in feature_names:
            df[name] = np.random.randn(n_samples)

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_raw_data.return_value = df
        mock_loader_instance.resample_to_daily.return_value = df
        mock_loader_instance.create_target_variable.return_value = df
        mock_loader_class.return_value = mock_loader_instance

        mock_engineer_instance = MagicMock()
        mock_engineer_instance.generate_all_features.return_value = df
        mock_engineer_instance.get_feature_names.return_value = feature_names
        mock_engineer_class.return_value = mock_engineer_instance

        mock_wf_train.return_value = mock_components['wf_results']

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_create_model.return_value = mock_model

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = mock_components['trainer'].train.return_value
        mock_trainer_instance.predict.return_value = np.random.rand(40)
        mock_trainer_class.return_value = mock_trainer_instance

        mock_evaluate.return_value = mock_components['eval_results']

        # Run main with LSTM model
        with patch('sys.argv', [
            'train_futures_model.py',
            '--data', str(tmp_path / 'test_data.txt'),
            '--model', 'lstm',
            '--epochs', '1',
            '--seq-length', '10',
            '--output-dir', str(tmp_path),
            '--no-plot'
        ]):
            main()

        # Verify LSTM model was created with correct config
        mock_create_model.assert_called_once()
        call_args = mock_create_model.call_args
        assert call_args[0][0] == 'lstm'  # First positional arg is model type

    @patch('src.ml.train_futures_model.FuturesDataLoader')
    @patch('src.ml.train_futures_model.FeatureEngineer')
    @patch('src.ml.train_futures_model.train_with_walk_forward')
    @patch('src.ml.train_futures_model.create_model')
    @patch('src.ml.train_futures_model.ModelTrainer')
    @patch('src.ml.train_futures_model.evaluate_model_and_strategy')
    @patch('src.ml.train_futures_model.print_evaluation_report')
    def test_main_creates_output_files(
        self, mock_print_report, mock_evaluate,
        mock_trainer_class, mock_create_model, mock_wf_train,
        mock_engineer_class, mock_loader_class,
        mock_components, tmp_path
    ):
        """Test main() creates model and results files.

        Why: Verifies artifacts are saved for reproducibility and deployment.
        """
        from src.ml.train_futures_model import main
        import pandas as pd

        # Setup mocks
        n_samples = 200
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame({
            'open': np.random.randn(n_samples) * 10 + 100,
            'high': np.random.randn(n_samples) * 10 + 105,
            'low': np.random.randn(n_samples) * 10 + 95,
            'close': np.random.randn(n_samples) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'target': np.random.randint(0, 2, n_samples),
            'next_return': np.random.randn(n_samples) * 0.01
        }, index=dates)
        for name in feature_names:
            df[name] = np.random.randn(n_samples)

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_raw_data.return_value = df
        mock_loader_instance.resample_to_daily.return_value = df
        mock_loader_instance.create_target_variable.return_value = df
        mock_loader_class.return_value = mock_loader_instance

        mock_engineer_instance = MagicMock()
        mock_engineer_instance.generate_all_features.return_value = df
        mock_engineer_instance.get_feature_names.return_value = feature_names
        mock_engineer_class.return_value = mock_engineer_instance

        mock_wf_train.return_value = mock_components['wf_results']

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_create_model.return_value = mock_model

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = mock_components['trainer'].train.return_value
        mock_trainer_instance.predict.return_value = np.random.rand(40)
        mock_trainer_class.return_value = mock_trainer_instance

        mock_evaluate.return_value = mock_components['eval_results']

        output_dir = tmp_path / "output"

        # Run main
        with patch('sys.argv', [
            'train_futures_model.py',
            '--data', str(tmp_path / 'test_data.txt'),
            '--output-dir', str(output_dir),
            '--epochs', '1',
            '--no-plot'
        ]):
            main()

        # Verify output directory was created
        assert output_dir.exists()

        # Verify model file was created
        model_files = list(output_dir.glob('model_*.pt'))
        assert len(model_files) >= 1, "Model file should be created"

        # Verify results file was created
        results_files = list(output_dir.glob('results_*.json'))
        assert len(results_files) >= 1, "Results file should be created"

    @patch('src.ml.train_futures_model.FuturesDataLoader')
    @patch('src.ml.train_futures_model.FeatureEngineer')
    @patch('src.ml.train_futures_model.train_with_walk_forward')
    @patch('src.ml.train_futures_model.create_model')
    @patch('src.ml.train_futures_model.ModelTrainer')
    @patch('src.ml.train_futures_model.evaluate_model_and_strategy')
    @patch('src.ml.train_futures_model.print_evaluation_report')
    @patch('src.ml.train_futures_model.plot_results')
    def test_main_handles_plot_exception(
        self, mock_plot, mock_print_report, mock_evaluate,
        mock_trainer_class, mock_create_model, mock_wf_train,
        mock_engineer_class, mock_loader_class,
        mock_components, tmp_path, capsys
    ):
        """Test main() handles plot exceptions gracefully.

        Why: Plotting may fail in headless environments - shouldn't crash pipeline.
        """
        from src.ml.train_futures_model import main
        import pandas as pd

        # Setup mocks
        n_samples = 200
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame({
            'open': np.random.randn(n_samples) * 10 + 100,
            'high': np.random.randn(n_samples) * 10 + 105,
            'low': np.random.randn(n_samples) * 10 + 95,
            'close': np.random.randn(n_samples) * 10 + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'target': np.random.randint(0, 2, n_samples),
            'next_return': np.random.randn(n_samples) * 0.01
        }, index=dates)
        for name in feature_names:
            df[name] = np.random.randn(n_samples)

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_raw_data.return_value = df
        mock_loader_instance.resample_to_daily.return_value = df
        mock_loader_instance.create_target_variable.return_value = df
        mock_loader_class.return_value = mock_loader_instance

        mock_engineer_instance = MagicMock()
        mock_engineer_instance.generate_all_features.return_value = df
        mock_engineer_instance.get_feature_names.return_value = feature_names
        mock_engineer_class.return_value = mock_engineer_instance

        mock_wf_train.return_value = mock_components['wf_results']

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_create_model.return_value = mock_model

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = mock_components['trainer'].train.return_value
        mock_trainer_instance.predict.return_value = np.random.rand(40)
        mock_trainer_class.return_value = mock_trainer_instance

        mock_evaluate.return_value = mock_components['eval_results']

        # Make plot_results raise an exception
        mock_plot.side_effect = Exception("Display not available")

        # Run main without --no-plot (should attempt to plot)
        with patch('sys.argv', [
            'train_futures_model.py',
            '--data', str(tmp_path / 'test_data.txt'),
            '--output-dir', str(tmp_path),
            '--epochs', '1'
        ]):
            # Should not raise
            main()

        # Verify error message was printed
        captured = capsys.readouterr()
        assert "Could not generate plot" in captured.out


# ============================================================================
# CUDA Seed Setting Tests
# ============================================================================

class TestCudaSeedSetting:
    """Tests for CUDA seed setting in set_seed()."""

    def test_cuda_seed_when_available(self):
        """Test CUDA seed is set when CUDA is available."""
        import torch

        if torch.cuda.is_available():
            with patch.object(torch.cuda, 'manual_seed_all') as mock_cuda_seed:
                set_seed(42)
                mock_cuda_seed.assert_called_once_with(42)

    def test_set_seed_works_regardless_of_cuda(self):
        """Test set_seed works whether CUDA is available or not.

        Why: set_seed() should work correctly in any environment - with or without GPU.
        The function conditionally sets CUDA seed based on cuda.is_available().
        """
        import torch

        # set_seed should work without errors
        set_seed(42)

        # Verify reproducibility is achieved
        torch.manual_seed(42)
        expected = torch.rand(5)

        set_seed(42)
        actual = torch.rand(5)

        assert torch.allclose(expected, actual), "Seeds should produce reproducible results"


# ============================================================================
# Data Pipeline Step Tests
# ============================================================================

class TestDataPipelineSteps:
    """Tests for individual data pipeline steps."""

    def test_feature_scaler_normalization(self):
        """Test StandardScaler normalizes features correctly.

        Why: Normalization is critical for neural network convergence.
        """
        from sklearn.preprocessing import StandardScaler

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check mean is ~0 and std is ~1
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0, 0], decimal=5)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), [1, 1], decimal=5)

    def test_train_test_split_preserves_order(self):
        """Test train/test split maintains temporal order.

        Why: Time series data must not be shuffled to avoid lookahead bias.
        """
        X = np.arange(100)
        train_ratio = 0.8
        split_idx = int(len(X) * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]

        # Verify order preserved
        assert X_train[-1] < X_test[0], "Train should come before test temporally"
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_train_test_split_with_custom_ratio(self):
        """Test train/test split with different ratios."""
        X = np.arange(100)

        for ratio in [0.5, 0.7, 0.9]:
            split_idx = int(len(X) * ratio)
            X_train, X_test = X[:split_idx], X[split_idx:]

            expected_train = int(100 * ratio)
            assert len(X_train) == expected_train
            assert len(X_test) == 100 - expected_train


# ============================================================================
# Model Config Edge Cases
# ============================================================================

class TestModelConfigEdgeCases:
    """Tests for edge cases in model config building."""

    def test_lstm_config_single_hidden_dim(self):
        """Test LSTM config with single hidden dim uses default fc_dims."""
        hidden_dims = [128]

        model_config = {
            'type': 'lstm',
            'params': {
                'hidden_dim': hidden_dims[0],
                'num_layers': 2,
                'dropout_rate': 0.3,
                'fc_dims': hidden_dims[1:] if len(hidden_dims) > 1 else [32]
            }
        }

        assert model_config['params']['hidden_dim'] == 128
        assert model_config['params']['fc_dims'] == [32]  # Default fallback

    def test_hidden_dims_parsing_large_values(self):
        """Test parsing large hidden dim values."""
        hidden_dims_str = "1024,512,256,128"
        hidden_dims = [int(x) for x in hidden_dims_str.split(",")]

        assert hidden_dims == [1024, 512, 256, 128]

    def test_hidden_dims_parsing_with_spaces(self):
        """Test parsing hidden dims with potential whitespace."""
        hidden_dims_str = "128, 64, 32"  # Spaces after commas
        hidden_dims = [int(x.strip()) for x in hidden_dims_str.split(",")]

        assert hidden_dims == [128, 64, 32]


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
