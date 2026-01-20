"""
Tests for RL hybrid trading environment.

Tests cover:
- HybridTradingEnvironment class
- ML prediction integration
- Observation space composition
- Signal-aware reward calculation
- load_ml_model function
- create_hybrid_env factory function
"""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch
from gymnasium import spaces
from zoneinfo import ZoneInfo

from src.rl.trading_env import TradingEnvironment, Action
from src.rl.hybrid_env import (
    HybridTradingEnvironment,
    load_ml_model,
    create_hybrid_env,
)
from src.rl.multi_horizon_model import MultiHorizonNet, HorizonPrediction


NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:30:00", periods=500, freq="1min", tz=NY_TZ)

    base_price = 4800.0
    returns = np.random.normal(0, 0.0002, 500)
    close_prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": np.roll(close_prices, 1),
        "high": close_prices * (1 + np.random.uniform(0.0001, 0.0005, 500)),
        "low": close_prices * (1 - np.random.uniform(0.0001, 0.0005, 500)),
        "close": close_prices,
        "volume": np.random.randint(100, 1000, 500),
    }, index=dates)
    df.iloc[0, df.columns.get_loc("open")] = base_price

    # Add some features
    df["return_1"] = df["close"].pct_change(1).fillna(0)
    df["return_5"] = df["close"].pct_change(5).fillna(0)
    df["volatility"] = df["close"].pct_change().rolling(10).std().fillna(0)
    df["rsi"] = 50.0  # Simplified

    return df


@pytest.fixture
def feature_columns():
    """Feature columns for testing."""
    return ["return_1", "return_5", "volatility", "rsi"]


@pytest.fixture
def mock_ml_model():
    """Create a mock ML model for testing."""
    model = MultiHorizonNet(input_dim=4)
    model.eval()
    return model


class TestHybridTradingEnvironmentInit:
    """Tests for HybridTradingEnvironment initialization."""

    def test_init_without_ml_model(self, sample_ohlcv_data, feature_columns):
        """Environment initializes without ML model."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=None,
        )
        assert env.ml_model is None

    def test_init_with_ml_model(self, sample_ohlcv_data, feature_columns, mock_ml_model):
        """Environment initializes with ML model."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_ml_model,
        )
        assert env.ml_model is not None

    def test_observation_space_includes_ml(self, sample_ohlcv_data, feature_columns):
        """Observation space is larger to accommodate ML predictions."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=None,
            lookback_window=60,
        )

        # Base: 60 * 4 features + 4 position info = 244
        # ML: 6 predictions (3 probs + 3 confidences)
        # Total: 250
        expected_dim = 60 * len(feature_columns) + 4 + 6
        assert env.observation_space.shape[0] == expected_dim

    def test_ml_obs_dim_attribute(self, sample_ohlcv_data, feature_columns):
        """Environment has ml_obs_dim attribute."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
        )
        assert env.ml_obs_dim == 6

    def test_signal_reward_weight(self, sample_ohlcv_data, feature_columns):
        """Environment stores signal reward weight."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            signal_reward_weight=0.2,
        )
        assert env.signal_reward_weight == 0.2

    def test_device_attribute(self, sample_ohlcv_data, feature_columns):
        """Environment stores device attribute."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            device="cpu",
        )
        assert env.device == torch.device("cpu")

    def test_inherits_from_base(self, sample_ohlcv_data, feature_columns):
        """HybridTradingEnvironment inherits from TradingEnvironment."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
        )
        assert isinstance(env, TradingEnvironment)


class TestMLPredictionRetrieval:
    """Tests for _get_ml_prediction method."""

    def test_returns_neutral_without_model(self, sample_ohlcv_data, feature_columns):
        """Returns neutral predictions when no ML model."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=None,
        )
        env.reset()
        preds = env._get_ml_prediction()

        assert preds.shape == (6,)
        # Neutral predictions: 0.5 probs, 0.0 confidences
        np.testing.assert_array_almost_equal(
            preds, [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        )

    def test_returns_array_with_model(self, sample_ohlcv_data, feature_columns, mock_ml_model):
        """Returns predictions array with ML model."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_ml_model,
            ml_feature_cols=feature_columns,
        )
        env.reset()
        preds = env._get_ml_prediction()

        assert preds.shape == (6,)
        assert preds.dtype == np.float32

    def test_predictions_cached(self, sample_ohlcv_data, feature_columns, mock_ml_model):
        """ML predictions are cached in _last_ml_prediction."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_ml_model,
            ml_feature_cols=feature_columns,
        )
        env.reset()
        env._get_ml_prediction()

        assert env._last_ml_prediction is not None
        assert isinstance(env._last_ml_prediction, HorizonPrediction)


class TestObservationComposition:
    """Tests for observation composition including ML predictions."""

    def test_observation_shape(self, sample_ohlcv_data, feature_columns):
        """Observation has correct shape."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            lookback_window=60,
        )
        obs, _ = env.reset()

        expected_dim = 60 * len(feature_columns) + 4 + 6
        assert obs.shape == (expected_dim,)

    def test_observation_dtype(self, sample_ohlcv_data, feature_columns):
        """Observation is float32."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
        )
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_observation_contains_ml_preds(self, sample_ohlcv_data, feature_columns):
        """Observation contains ML predictions at the end."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=None,
            lookback_window=60,
        )
        obs, _ = env.reset()

        # Last 6 elements should be ML predictions (neutral without model)
        ml_preds = obs[-6:]
        expected = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(ml_preds, expected)


class TestSignalRewardCalculation:
    """Tests for _calculate_signal_reward method."""

    @pytest.fixture
    def env(self, sample_ohlcv_data, feature_columns):
        """Create environment for testing."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            signal_reward_weight=0.1,
        )
        env.reset()
        return env

    def test_zero_reward_no_prediction(self, env):
        """Returns 0 when no ML prediction."""
        reward = env._calculate_signal_reward(Action.LONG, None)
        assert reward == 0.0

    def test_zero_reward_zero_weight(self, sample_ohlcv_data, feature_columns):
        """Returns 0 when signal_reward_weight is 0."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            signal_reward_weight=0.0,
        )
        env.reset()

        pred = HorizonPrediction(0.8, 0.7, 0.6, 0.6, 0.5, 0.4)
        reward = env._calculate_signal_reward(Action.LONG, pred)
        assert reward == 0.0

    def test_zero_reward_low_confidence(self, env):
        """Returns 0 for low confidence predictions."""
        pred = HorizonPrediction(0.8, 0.7, 0.6, 0.2, 0.2, 0.2)  # Low confidence
        reward = env._calculate_signal_reward(Action.LONG, pred)
        assert reward == 0.0

    def test_positive_reward_following_high_conf_signal(self, env):
        """Positive reward for following high confidence signal."""
        # High prob_up_1h (>0.6) with high confidence (>0.5)
        pred = HorizonPrediction(0.8, 0.7, 0.6, 0.7, 0.5, 0.4)
        reward = env._calculate_signal_reward(Action.LONG, pred)
        assert reward > 0

    def test_negative_reward_contradicting_strong_signal(self, env):
        """Negative reward for contradicting strong signal."""
        # High prob_up_1h (>0.6) with very high confidence (>0.7)
        pred = HorizonPrediction(0.8, 0.7, 0.6, 0.9, 0.8, 0.7)
        reward = env._calculate_signal_reward(Action.SHORT, pred)  # Contradicting
        assert reward < 0

    def test_no_penalty_for_flat(self, env):
        """No penalty for going FLAT even with strong signal."""
        pred = HorizonPrediction(0.8, 0.7, 0.6, 0.9, 0.8, 0.7)
        reward = env._calculate_signal_reward(Action.FLAT, pred)
        # FLAT should not be penalized
        assert reward >= 0


class TestStepIntegration:
    """Tests for step method integration."""

    def test_step_returns_correct_format(self, sample_ohlcv_data, feature_columns):
        """Step returns (obs, reward, terminated, truncated, info) tuple."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
        )
        env.reset()
        result = env.step(Action.LONG)

        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_includes_ml_preds_in_info(self, sample_ohlcv_data, feature_columns, mock_ml_model):
        """Step includes ML predictions in info dict."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_ml_model,
            ml_feature_cols=feature_columns,
        )
        env.reset()
        _, _, _, _, info = env.step(Action.LONG)

        assert "ml_prob_up_1h" in info
        assert "ml_prob_up_4h" in info
        assert "ml_prob_up_eod" in info


class TestCreateHybridEnv:
    """Tests for create_hybrid_env factory function."""

    def test_returns_hybrid_env(self, sample_ohlcv_data, feature_columns):
        """Factory returns HybridTradingEnvironment."""
        env = create_hybrid_env(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
        )
        assert isinstance(env, HybridTradingEnvironment)

    def test_passes_kwargs(self, sample_ohlcv_data, feature_columns):
        """Factory passes additional kwargs to environment."""
        env = create_hybrid_env(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            initial_balance=5000.0,
            max_daily_loss=200.0,
        )
        assert env.initial_balance == 5000.0
        assert env.max_daily_loss == 200.0

    def test_none_model_path(self, sample_ohlcv_data, feature_columns):
        """Factory handles None model path."""
        env = create_hybrid_env(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model_path=None,
        )
        assert env.ml_model is None

    def test_nonexistent_model_path(self, sample_ohlcv_data, feature_columns):
        """Factory handles non-existent model path."""
        env = create_hybrid_env(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model_path="/nonexistent/path/model.pt",
        )
        assert env.ml_model is None


class TestLoadMLModel:
    """Tests for load_ml_model function."""

    def test_load_ml_model_exists(self):
        """load_ml_model function is importable."""
        from src.rl.hybrid_env import load_ml_model
        assert callable(load_ml_model)

    def test_load_ml_model_returns_tuple(self, tmp_path):
        """load_ml_model returns 4-element tuple when successful."""
        # Create a mock checkpoint file
        # Use explicit hidden_dims to match what will be in the config
        hidden_dims = [256, 128, 64]  # Default for MultiHorizonNet
        model = MultiHorizonNet(input_dim=4, hidden_dims=hidden_dims)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": 4,
                "hidden_dims": hidden_dims,  # Must match model architecture
                "dropout_rate": 0.3,
            },
            "scaler_mean": [0.0, 0.0, 0.0, 0.0],
            "scaler_scale": [1.0, 1.0, 1.0, 1.0],
            "feature_cols": ["f1", "f2", "f3", "f4"],
        }

        model_path = tmp_path / "test_model.pt"
        torch.save(checkpoint, model_path)

        result = load_ml_model(str(model_path), device="cpu")

        assert isinstance(result, tuple)
        assert len(result) == 4

        loaded_model, scaler_mean, scaler_scale, feature_cols = result
        assert isinstance(loaded_model, MultiHorizonNet)
        assert isinstance(scaler_mean, np.ndarray)
        assert isinstance(scaler_scale, np.ndarray)
        assert isinstance(feature_cols, list)


class TestResetBehavior:
    """Tests for reset behavior with ML predictions."""

    def test_reset_clears_ml_prediction_cache(self, sample_ohlcv_data, feature_columns, mock_ml_model):
        """Reset doesn't crash with ML model."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_ml_model,
            ml_feature_cols=feature_columns,
        )

        # First reset
        obs1, _ = env.reset()
        assert obs1 is not None

        # Take some steps
        env.step(Action.LONG)
        env.step(Action.FLAT)

        # Second reset
        obs2, _ = env.reset()
        assert obs2 is not None

    def test_reset_with_seed(self, sample_ohlcv_data, feature_columns):
        """Reset with seed is reproducible."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
        )

        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ml_prediction_error_handling(self, sample_ohlcv_data, feature_columns):
        """Environment handles ML prediction errors gracefully."""
        # Create a mock model that raises an error
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Test error")

        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_model,
        )
        env.reset()

        # Should return neutral predictions instead of crashing
        preds = env._get_ml_prediction()
        expected = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, expected)

    def test_missing_feature_columns(self, sample_ohlcv_data, feature_columns, mock_ml_model):
        """Environment handles missing feature columns."""
        env = HybridTradingEnvironment(
            df=sample_ohlcv_data,
            feature_columns=feature_columns,
            ml_model=mock_ml_model,
            ml_feature_cols=["nonexistent_col1", "nonexistent_col2"],
        )
        env.reset()

        # Should not crash, returns predictions with zeros for missing
        preds = env._get_ml_prediction()
        assert preds.shape == (6,)
