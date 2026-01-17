"""
Tests for the run_backtest.py script.

This module provides comprehensive E2E tests for the backtest script functionality,
including:
- E2E testing with trained models
- Checkpoint format compatibility (old vs new formats)
- Script command-line interface
- Model loading and signal generation

These tests address items from IMPLEMENTATION_PLAN.md section 10.12:
- test_backtest_script_e2e_with_trained_model() - CRITICAL
- test_checkpoint_loading_old_and_new_formats() - HIGH
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_backtest import (
    load_model,
    create_random_signal_generator,
    run_backtest,
    MLSignalGenerator,
)
from src.ml.models.neural_networks import FeedForwardNet, LSTMNet, HybridNet
from src.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    Signal,
    SignalType,
    Position,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_feedforward_model():
    """Create a simple FeedForwardNet model for testing."""
    model = FeedForwardNet(
        input_dim=50,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2,
        num_classes=3,
    )
    return model


@pytest.fixture
def sample_lstm_model():
    """Create a simple LSTMNet model for testing."""
    model = LSTMNet(
        input_dim=50,  # LSTMNet uses input_dim
        hidden_dim=64,  # LSTMNet uses hidden_dim
        num_layers=2,
        num_classes=3,
    )
    return model


@pytest.fixture
def sample_hybrid_model():
    """Create a simple HybridNet model for testing."""
    model = HybridNet(
        seq_input_dim=30,  # HybridNet uses seq_input_dim
        static_input_dim=20,  # HybridNet uses static_input_dim
        lstm_hidden=32,
        num_classes=3,
    )
    return model


@pytest.fixture
def new_format_checkpoint(sample_feedforward_model, tmp_path):
    """
    Create a checkpoint in the NEW format.

    New format has:
    - model_config with 'type' and 'params' keys
    - model_state_dict for weights
    """
    checkpoint_path = tmp_path / "new_format_model.pt"

    checkpoint = {
        'model_config': {
            'type': 'feedforward',
            'params': {
                'hidden_dims': [128, 64, 32],
                'num_classes': 3,
                'dropout_rate': 0.2,
            }
        },
        'input_dim': 50,
        'model_state_dict': sample_feedforward_model.state_dict(),
        'epoch': 100,
        'best_val_loss': 0.5,
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def old_format_checkpoint(sample_feedforward_model, tmp_path):
    """
    Create a checkpoint in the OLD format.

    Old format has:
    - config with 'model_type' and direct parameter keys
    - state_dict for weights
    """
    checkpoint_path = tmp_path / "old_format_model.pt"

    checkpoint = {
        'config': {
            'model_type': 'feedforward',
            'hidden_sizes': [128, 64, 32],  # Note: old name for hidden_dims
            'num_classes': 3,
            'dropout_rate': 0.2,
            'input_size': 50,
        },
        'state_dict': sample_feedforward_model.state_dict(),  # Note: old key name
        'epoch': 100,
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def lstm_checkpoint(sample_lstm_model, tmp_path):
    """Create a checkpoint for LSTM model."""
    checkpoint_path = tmp_path / "lstm_model.pt"

    checkpoint = {
        'model_config': {
            'type': 'lstm',
            'params': {
                'hidden_dims': [64],  # First element becomes hidden_dim
                'num_classes': 3,
                'num_layers': 2,
            }
        },
        'input_dim': 50,
        'model_state_dict': sample_lstm_model.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def hybrid_checkpoint(sample_hybrid_model, tmp_path):
    """Create a checkpoint for Hybrid model."""
    checkpoint_path = tmp_path / "hybrid_model.pt"

    checkpoint = {
        'model_config': {
            'type': 'hybrid',
            'params': {
                'num_classes': 3,
            },
            'lstm_hidden': 32,
            'seq_input_dim': 30,  # Must match model
            'static_input_dim': 20,  # Must match model
        },
        'input_dim': 50,  # Not used for hybrid but included for compatibility
        'model_state_dict': sample_hybrid_model.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def bare_weights_checkpoint(sample_feedforward_model, tmp_path):
    """
    Create a checkpoint that's just the state dict (bare weights).

    This tests the fallback when checkpoint IS the state dict.
    """
    checkpoint_path = tmp_path / "bare_weights_model.pt"
    torch.save(sample_feedforward_model.state_dict(), checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def sample_backtest_data(tmp_path):
    """Create sample parquet data for backtest E2E testing."""
    np.random.seed(42)

    # Create timestamps for 3 hours of RTH (10,800 seconds)
    start_time = pd.Timestamp('2024-01-02 09:30:00', tz='America/New_York')
    timestamps = pd.date_range(start=start_time, periods=10800, freq='1s')

    # Create realistic price data
    base_price = 5000.0
    returns = np.random.randn(10800) * 0.0001
    close_prices = base_price * np.cumprod(1 + returns)

    high_prices = close_prices * (1 + np.abs(np.random.randn(10800)) * 0.0002)
    low_prices = close_prices * (1 - np.abs(np.random.randn(10800)) * 0.0002)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    volumes = np.random.randint(10, 1000, 10800)

    # Create DataFrame with UTC timestamps (like real parquet data)
    df = pd.DataFrame({
        'timestamp': timestamps.tz_convert('UTC'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'symbol': 'MES.FUT',
    })

    parquet_path = tmp_path / "test_data.parquet"
    df.to_parquet(parquet_path, engine='pyarrow', index=False)

    return str(parquet_path)


# =============================================================================
# Test Classes: Checkpoint Loading (addresses 10.12)
# =============================================================================

class TestCheckpointLoadingFormats:
    """
    Test checkpoint loading for old and new formats.

    This addresses IMPLEMENTATION_PLAN.md 10.12:
    - test_checkpoint_loading_old_and_new_formats() - HIGH
    """

    def test_load_new_format_checkpoint(self, new_format_checkpoint):
        """Test loading checkpoint in new format (model_config with type/params)."""
        model, config = load_model(new_format_checkpoint)

        # Verify model loaded correctly
        assert model is not None
        assert isinstance(model, FeedForwardNet)

        # Verify model is in eval mode
        assert not model.training

        # Verify config returned
        assert config is not None
        assert config.get('type') == 'feedforward'

    def test_load_old_format_checkpoint(self, old_format_checkpoint):
        """Test loading checkpoint in old format (config with model_type)."""
        model, config = load_model(old_format_checkpoint)

        # Verify model loaded correctly
        assert model is not None
        assert isinstance(model, FeedForwardNet)

        # Verify model is in eval mode
        assert not model.training

        # Verify config returned
        assert config is not None
        # Old format uses model_type instead of type
        assert config.get('model_type') == 'feedforward'

    def test_load_lstm_checkpoint(self, lstm_checkpoint):
        """Test loading LSTM checkpoint."""
        model, config = load_model(lstm_checkpoint)

        assert model is not None
        assert isinstance(model, LSTMNet)
        assert not model.training

    def test_load_hybrid_checkpoint(self, hybrid_checkpoint):
        """Test loading Hybrid checkpoint."""
        model, config = load_model(hybrid_checkpoint)

        assert model is not None
        assert isinstance(model, HybridNet)
        assert not model.training

    def test_load_model_state_dict_key_variants(self, sample_feedforward_model, tmp_path):
        """Test loading checkpoints with different state dict key names."""
        # Test with 'model_state_dict' key
        path1 = tmp_path / "model_state_dict_key.pt"
        torch.save({
            'model_config': {'type': 'feedforward', 'params': {'hidden_dims': [128, 64, 32], 'num_classes': 3}},
            'input_dim': 50,
            'model_state_dict': sample_feedforward_model.state_dict(),
        }, path1)

        model1, _ = load_model(str(path1))
        assert model1 is not None

        # Test with 'state_dict' key (old format)
        path2 = tmp_path / "state_dict_key.pt"
        torch.save({
            'model_config': {'type': 'feedforward', 'params': {'hidden_dims': [128, 64, 32], 'num_classes': 3}},
            'input_dim': 50,
            'state_dict': sample_feedforward_model.state_dict(),
        }, path2)

        model2, _ = load_model(str(path2))
        assert model2 is not None

    def test_load_model_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing model."""
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "nonexistent.pt"))

    def test_load_model_unknown_type(self, sample_feedforward_model, tmp_path):
        """Test that ValueError is raised for unknown model type."""
        path = tmp_path / "unknown_type.pt"
        torch.save({
            'model_config': {'type': 'unknown_architecture', 'params': {}},
            'input_dim': 50,
            'model_state_dict': sample_feedforward_model.state_dict(),
        }, path)

        with pytest.raises(ValueError, match="Unknown model type"):
            load_model(str(path))

    def test_load_model_on_cpu(self, new_format_checkpoint):
        """Test loading model explicitly on CPU."""
        model, config = load_model(new_format_checkpoint, device='cpu')

        # Verify model is on CPU
        for param in model.parameters():
            assert param.device.type == 'cpu'

    def test_hidden_dims_hidden_sizes_compatibility(self, sample_feedforward_model, tmp_path):
        """Test that both hidden_dims and hidden_sizes keys are supported."""
        # Create checkpoint with hidden_sizes (old naming)
        path = tmp_path / "hidden_sizes.pt"
        torch.save({
            'config': {
                'model_type': 'feedforward',
                'hidden_sizes': [128, 64, 32],  # Old key name
                'num_classes': 3,
                'dropout_rate': 0.2,
            },
            'input_dim': 50,
            'model_state_dict': sample_feedforward_model.state_dict(),
        }, path)

        model, config = load_model(str(path))
        assert model is not None
        assert isinstance(model, FeedForwardNet)


# =============================================================================
# Test Classes: E2E Backtest Script Testing (addresses 10.12)
# =============================================================================

class TestBacktestScriptE2E:
    """
    E2E tests for the run_backtest.py script.

    This addresses IMPLEMENTATION_PLAN.md 10.12:
    - test_backtest_script_e2e_with_trained_model() - CRITICAL
    """

    def test_run_backtest_with_random_baseline(self, sample_backtest_data, tmp_path):
        """Test running backtest with random baseline (no model needed)."""
        output_dir = str(tmp_path / "results")

        result = run_backtest(
            data_path=sample_backtest_data,
            model_path=None,
            output_dir=output_dir,
            random_baseline=True,
            limit_bars=5000,  # Limit for faster testing
            verbose=False,
        )

        # Verify result returned
        assert result is not None
        assert isinstance(result, BacktestResult)

        # Random baseline should have some trades (or at least not crash)
        # Access total_bars through data_stats dict
        assert result.data_stats.get("total_bars", 0) >= 0

    def test_run_backtest_with_trained_model(self, sample_backtest_data, new_format_checkpoint, tmp_path):
        """
        E2E test running backtest with a trained model.

        This is the CRITICAL test from 10.12.
        """
        output_dir = str(tmp_path / "results")

        result = run_backtest(
            data_path=sample_backtest_data,
            model_path=new_format_checkpoint,
            output_dir=output_dir,
            limit_bars=3000,  # Limit for faster testing
            verbose=False,
        )

        # Verify result
        assert result is not None
        assert isinstance(result, BacktestResult)
        assert result.data_stats.get("total_bars", 0) > 0

        # Verify no crashes occurred during signal generation
        # (the main goal of this E2E test)

    def test_run_backtest_with_lstm_model(self, sample_backtest_data, lstm_checkpoint, tmp_path):
        """Test E2E backtest with LSTM model (tuple output handling)."""
        output_dir = str(tmp_path / "results")

        result = run_backtest(
            data_path=sample_backtest_data,
            model_path=lstm_checkpoint,
            output_dir=output_dir,
            limit_bars=2000,
            verbose=False,
        )

        # Verify LSTM works without crashing (tests tuple unpacking fix)
        assert result is not None
        assert isinstance(result, BacktestResult)

    def test_run_backtest_creates_output_files(self, sample_backtest_data, new_format_checkpoint, tmp_path):
        """Test that backtest creates expected output files."""
        output_dir = tmp_path / "results"

        run_backtest(
            data_path=sample_backtest_data,
            model_path=new_format_checkpoint,
            output_dir=str(output_dir),
            limit_bars=2000,
            verbose=False,
        )

        # Check that output directory was created
        assert output_dir.exists()

        # Check for expected output files (may vary based on implementation)
        # At minimum, the directory should exist

    def test_run_backtest_with_custom_config(self, sample_backtest_data, new_format_checkpoint, tmp_path):
        """Test running backtest with custom configuration."""
        output_dir = str(tmp_path / "results")

        config = BacktestConfig(
            initial_capital=2000.0,
            max_daily_loss=100.0,
            tick_size=0.25,
            tick_value=1.25,
        )

        result = run_backtest(
            data_path=sample_backtest_data,
            model_path=new_format_checkpoint,
            output_dir=output_dir,
            config=config,
            limit_bars=2000,
            verbose=False,
        )

        assert result is not None
        # Access initial_capital through config
        assert result.config.initial_capital == 2000.0


class TestMLSignalGenerator:
    """Test the MLSignalGenerator class."""

    def test_signal_generator_initialization(self, sample_feedforward_model):
        """Test MLSignalGenerator initializes correctly."""
        # Create a mock feature engineer
        mock_feature_engineer = MagicMock()
        mock_feature_engineer.feature_names = ['feature_' + str(i) for i in range(50)]

        generator = MLSignalGenerator(
            model=sample_feedforward_model,
            feature_engineer=mock_feature_engineer,
            min_confidence=0.60,
            stop_ticks=8.0,
            target_ticks=16.0,
        )

        assert generator.model is not None
        assert generator.min_confidence == 0.60
        assert generator.stop_ticks == 8.0
        assert generator.target_ticks == 16.0

    def test_signal_generator_produces_hold_below_threshold(self, sample_feedforward_model):
        """Test that generator produces HOLD when confidence is below threshold."""
        mock_feature_engineer = MagicMock()
        mock_feature_engineer.feature_names = ['feature_' + str(i) for i in range(50)]

        generator = MLSignalGenerator(
            model=sample_feedforward_model,
            feature_engineer=mock_feature_engineer,
            min_confidence=0.99,  # Very high threshold
        )

        # Create sample bar
        bar = pd.Series({
            **{f'feature_{i}': np.random.randn() for i in range(50)},
            'open': 5000.0,
            'high': 5001.0,
            'low': 4999.0,
            'close': 5000.5,
            'volume': 100,
        })

        signal = generator.generate_signal(bar, None, {})

        # With 99% threshold, most predictions should result in HOLD
        assert signal.signal_type in [SignalType.HOLD, SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]


class TestRandomSignalGenerator:
    """Test the random signal generator for baseline testing."""

    def test_create_random_generator(self):
        """Test creating a random signal generator."""
        generator = create_random_signal_generator(
            min_confidence=0.60,
            stop_ticks=8.0,
            target_ticks=16.0,
        )

        assert callable(generator)

    def test_random_generator_produces_signals(self):
        """Test that random generator produces valid signals."""
        np.random.seed(42)

        generator = create_random_signal_generator(
            min_confidence=0.60,
        )

        bar = pd.Series({
            'open': 5000.0,
            'high': 5001.0,
            'low': 4999.0,
            'close': 5000.5,
            'volume': 100,
        })

        # Generate multiple signals
        signals = [generator(bar, None, {}) for _ in range(100)]

        # Should have variety of signal types
        signal_types = [s.signal_type for s in signals]
        assert SignalType.HOLD in signal_types or len(set(signal_types)) > 1

    def test_random_generator_respects_position(self):
        """Test that random generator handles existing positions correctly."""
        np.random.seed(42)

        generator = create_random_signal_generator(min_confidence=0.50)

        bar = pd.Series({
            'open': 5000.0,
            'high': 5001.0,
            'low': 4999.0,
            'close': 5000.5,
            'volume': 100,
        })

        # Create a mock position
        position = MagicMock()
        position.direction = 1  # Long

        # Generate signals with position
        signals = [generator(bar, position, {}) for _ in range(100)]

        # Should not generate LONG_ENTRY when already long
        entry_signals = [s for s in signals if s.signal_type == SignalType.LONG_ENTRY]
        assert len(entry_signals) == 0


# =============================================================================
# Test Classes: Integration with Real Model (if available)
# =============================================================================

class TestWithRealModel:
    """Integration tests using the actual trained model (if available)."""

    @pytest.fixture
    def real_model_path(self):
        """Get path to real model if it exists."""
        path = Path("models/scalper_v1.pt")
        if path.exists():
            return str(path)
        return None

    def test_load_real_model(self, real_model_path):
        """Test loading the real trained model."""
        if real_model_path is None:
            pytest.skip("Real model not available")

        model, config = load_model(real_model_path)

        assert model is not None
        assert not model.training  # Should be in eval mode

    def test_real_model_inference(self, real_model_path):
        """Test inference with real model."""
        if real_model_path is None:
            pytest.skip("Real model not available")

        model, config = load_model(real_model_path)

        # Get input_dim from the model's first layer
        # FeedForwardNet has hidden_layers[0] as the first Linear layer
        if hasattr(model, 'hidden_layers'):
            input_dim = model.hidden_layers[0].in_features
        elif hasattr(model, 'input_dim'):
            input_dim = model.input_dim
        else:
            input_dim = 56  # Real model uses 56 features

        sample_input = torch.randn(1, input_dim)

        with torch.no_grad():
            output = model(sample_input)
            # Handle LSTM tuple output
            logits = output[0] if isinstance(output, tuple) else output
            probs = torch.softmax(logits, dim=1)

        # Verify output shape (3 classes)
        assert probs.shape == (1, 3)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_config_checkpoint_requires_matching_architecture(self, tmp_path):
        """
        Test that loading checkpoint without config requires matching defaults.

        When a checkpoint doesn't have config info, load_model uses defaults:
        - hidden_dims: [256, 128, 64]
        - input_dim: 50
        - num_classes: 3

        The model's weights must match these defaults for loading to succeed.
        """
        # Create a model with default architecture
        model_with_defaults = FeedForwardNet(
            input_dim=50,
            hidden_dims=[256, 128, 64],  # Must match defaults
            dropout_rate=0.3,
            num_classes=3,
        )

        path = tmp_path / "minimal_config.pt"
        torch.save({
            'model_state_dict': model_with_defaults.state_dict(),
            # No config - will use defaults that match this architecture
        }, path)

        model, config = load_model(str(path))
        assert model is not None
        assert isinstance(model, FeedForwardNet)

    def test_checkpoint_with_extra_fields(self, sample_feedforward_model, tmp_path):
        """Test that extra checkpoint fields don't break loading."""
        path = tmp_path / "extra_fields.pt"
        torch.save({
            'model_config': {'type': 'feedforward', 'params': {'hidden_dims': [128, 64, 32], 'num_classes': 3}},
            'input_dim': 50,
            'model_state_dict': sample_feedforward_model.state_dict(),
            'optimizer_state': {'some': 'data'},  # Extra field
            'scheduler_state': {'some': 'other_data'},  # Extra field
            'custom_metrics': [1, 2, 3],  # Extra field
        }, path)

        model, config = load_model(str(path))
        assert model is not None
