"""
Hybrid Trading Environment for RL Agent.

This environment combines:
1. Market features (price action, indicators)
2. ML model predictions (1h, 4h, EOD direction probabilities)
3. Position and risk state

The RL agent receives both raw market data AND expert ML predictions,
allowing it to learn when to trust the ML model and when to override.

Observation Space:
    [market_features] + [ml_predictions] + [position_state]

    market_features: 60-bar lookback of normalized features
    ml_predictions: [prob_1h, prob_4h, prob_eod, conf_1h, conf_4h, conf_eod]
    position_state: [position, unrealized_pnl, daily_pnl, time_to_close]

Actions:
    0: FLAT (close any position)
    1: LONG
    2: SHORT
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import logging

from .trading_env import TradingEnvironment, TradeRecord, Action
from .multi_horizon_model import MultiHorizonNet, HorizonPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridTradingEnvironment(TradingEnvironment):
    """
    Hybrid trading environment with ML signal integration.

    Extends the base TradingEnvironment to include:
    - ML model predictions as part of observation
    - Signal-aware reward shaping (bonus for following high-confidence signals)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        ml_model: Optional[MultiHorizonNet] = None,
        ml_scaler_mean: Optional[np.ndarray] = None,
        ml_scaler_scale: Optional[np.ndarray] = None,
        ml_feature_cols: Optional[List[str]] = None,
        initial_balance: float = 1000.0,
        max_daily_loss: float = 100.0,
        max_position: int = 1,
        lookback_window: int = 60,
        signal_reward_weight: float = 0.1,
        render_mode: Optional[str] = None,
        device: str = 'cpu',
    ):
        """
        Initialize hybrid trading environment.

        Args:
            df: DataFrame with OHLCV data and features
            feature_columns: List of feature column names for RL observation
            ml_model: Trained multi-horizon prediction model
            ml_scaler_mean: Scaler mean for ML model features
            ml_scaler_scale: Scaler scale for ML model features
            ml_feature_cols: Feature columns for ML model
            initial_balance: Starting account balance
            max_daily_loss: Maximum daily loss before stopping
            max_position: Maximum position size
            lookback_window: Number of bars for observation window
            signal_reward_weight: Weight for signal-following reward component
            render_mode: Rendering mode
            device: Device for ML inference
        """
        # Initialize base environment
        super().__init__(
            df=df,
            feature_columns=feature_columns,
            initial_balance=initial_balance,
            max_daily_loss=max_daily_loss,
            max_position=max_position,
            lookback_window=lookback_window,
            render_mode=render_mode,
        )

        # ML model components
        self.ml_model = ml_model
        self.ml_scaler_mean = ml_scaler_mean
        self.ml_scaler_scale = ml_scaler_scale
        self.ml_feature_cols = ml_feature_cols or feature_columns
        self.device = torch.device(device)
        self.signal_reward_weight = signal_reward_weight

        if self.ml_model is not None:
            self.ml_model.to(self.device)
            self.ml_model.eval()

        # Update observation space to include ML predictions
        # ML predictions: [prob_1h, prob_4h, prob_eod, conf_1h, conf_4h, conf_eod]
        n_features = len(feature_columns)
        base_obs_dim = lookback_window * n_features + 4  # base features + position info
        ml_obs_dim = 6  # 3 probs + 3 confidences

        self.ml_obs_dim = ml_obs_dim
        total_obs_dim = base_obs_dim + ml_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        # Cache for ML predictions
        self._last_ml_prediction: Optional[HorizonPrediction] = None

    def _get_ml_prediction(self) -> np.ndarray:
        """Get ML model predictions for current state."""
        if self.ml_model is None:
            # Return neutral predictions if no model
            return np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

        try:
            # Get current bar features for ML model
            current_bar = self.df.iloc[min(self.current_idx, len(self.df) - 1)]

            # Extract features for ML model
            ml_features = []
            for col in self.ml_feature_cols:
                if col in current_bar.index:
                    ml_features.append(current_bar[col])
                else:
                    ml_features.append(0.0)

            ml_features = np.array(ml_features, dtype=np.float32)

            # Apply ML scaler
            if self.ml_scaler_mean is not None and self.ml_scaler_scale is not None:
                ml_features = (ml_features - self.ml_scaler_mean) / (self.ml_scaler_scale + 1e-8)

            # Handle inf/nan
            ml_features = np.nan_to_num(ml_features, nan=0.0, posinf=0.0, neginf=0.0)
            ml_features = np.clip(ml_features, -5, 5)

            # Get prediction
            self._last_ml_prediction = self.ml_model.predict(ml_features)
            return self._last_ml_prediction.to_array()

        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """Get observation including ML predictions."""
        # Get base observation (market features + position state)
        base_obs = super()._get_observation()

        # Get ML predictions
        ml_obs = self._get_ml_prediction()

        # Concatenate
        obs = np.concatenate([base_obs, ml_obs])

        return obs.astype(np.float32)

    def _calculate_signal_reward(
        self,
        action: int,
        ml_prediction: Optional[HorizonPrediction],
    ) -> float:
        """
        Calculate bonus/penalty for following/ignoring ML signals.

        High confidence signals that are followed get a small bonus.
        This encourages the RL agent to learn when to trust the ML model.
        """
        if ml_prediction is None or self.signal_reward_weight == 0:
            return 0.0

        # Determine ML suggested action based on 1h prediction
        prob_up = ml_prediction.prob_up_1h
        confidence = ml_prediction.confidence_1h

        if confidence < 0.3:  # Low confidence - no signal
            return 0.0

        # ML suggests LONG if prob > 0.5, SHORT if prob < 0.5
        if prob_up > 0.6:
            ml_action = Action.LONG
        elif prob_up < 0.4:
            ml_action = Action.SHORT
        else:
            ml_action = Action.FLAT

        # Calculate alignment reward
        if action == ml_action and confidence > 0.5:
            # Reward for following high-confidence signals
            return self.signal_reward_weight * confidence
        elif action != ml_action and action != Action.FLAT and confidence > 0.7:
            # Small penalty for strongly contradicting ML
            return -self.signal_reward_weight * 0.5

        return 0.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with hybrid reward.
        """
        # Store ML prediction before step
        ml_pred = self._last_ml_prediction

        # Execute base step
        obs, reward, terminated, truncated, info = super().step(action)

        # Add signal-following reward component
        signal_reward = self._calculate_signal_reward(action, ml_pred)
        reward += signal_reward

        # Add ML predictions to info
        if ml_pred is not None:
            info['ml_prob_up_1h'] = ml_pred.prob_up_1h
            info['ml_prob_up_4h'] = ml_pred.prob_up_4h
            info['ml_prob_up_eod'] = ml_pred.prob_up_eod

        return obs, reward, terminated, truncated, info


def load_ml_model(model_path: str, device: str = 'cpu') -> Tuple[MultiHorizonNet, np.ndarray, np.ndarray, List[str]]:
    """
    Load trained multi-horizon model.

    Args:
        model_path: Path to model checkpoint

    Returns:
        (model, scaler_mean, scaler_scale, feature_cols)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint['model_config']
    model = MultiHorizonNet(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config.get('dropout_rate', 0.3),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler_mean = np.array(checkpoint['scaler_mean'])
    scaler_scale = np.array(checkpoint['scaler_scale'])
    feature_cols = checkpoint['feature_cols']

    logger.info(f"Loaded ML model from {model_path}")
    logger.info(f"  Input dim: {config['input_dim']}")
    logger.info(f"  Features: {len(feature_cols)}")

    return model, scaler_mean, scaler_scale, feature_cols


def create_hybrid_env(
    df: pd.DataFrame,
    feature_columns: list,
    ml_model_path: Optional[str] = None,
    device: str = 'cpu',
    **kwargs,
) -> HybridTradingEnvironment:
    """
    Factory function to create hybrid trading environment.

    Args:
        df: DataFrame with features
        feature_columns: RL feature columns
        ml_model_path: Path to trained ML model (optional)
        device: Device for ML inference
        **kwargs: Additional environment parameters

    Returns:
        HybridTradingEnvironment instance
    """
    ml_model = None
    ml_scaler_mean = None
    ml_scaler_scale = None
    ml_feature_cols = None

    if ml_model_path and Path(ml_model_path).exists():
        ml_model, ml_scaler_mean, ml_scaler_scale, ml_feature_cols = load_ml_model(
            ml_model_path, device
        )

    return HybridTradingEnvironment(
        df=df,
        feature_columns=feature_columns,
        ml_model=ml_model,
        ml_scaler_mean=ml_scaler_mean,
        ml_scaler_scale=ml_scaler_scale,
        ml_feature_cols=ml_feature_cols,
        device=device,
        **kwargs,
    )
