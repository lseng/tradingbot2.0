#!/usr/bin/env python3
"""
Train Final Enhanced Hybrid RL Agent.

Uses:
1. Enhanced features (66 total)
2. Walk-forward ensemble predictions
3. Regularized PPO training
4. Proper train/test split

Usage:
    python src/rl/train_final_agent.py --timesteps 2000000
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.preprocessing import StandardScaler

from src.rl.data_pipeline import MultiHorizonDataPipeline
from src.rl.multi_horizon_model import create_multi_horizon_targets
from src.rl.enhanced_features import combine_with_base_features
from src.rl.trading_env import TradingEnvironment
from src.rl.regularized_model import RegularizedMultiHorizonNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class EnhancedHybridEnvironment(TradingEnvironment):
    """
    Enhanced hybrid trading environment with:
    - Enhanced features (volume profile, regime, etc.)
    - Ensemble ML predictions (from walk-forward models)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        ensemble_models: Optional[List[Tuple[RegularizedMultiHorizonNet, StandardScaler]]] = None,
        initial_balance: float = 1000.0,
        max_daily_loss: float = 100.0,
        max_position: int = 1,
        lookback_window: int = 60,
        signal_reward_weight: float = 0.05,
        render_mode: Optional[str] = None,
    ):
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

        self.ensemble_models = ensemble_models or []
        self.signal_reward_weight = signal_reward_weight

        # Update observation space to include ensemble predictions
        # Ensemble output: [mean_1h, std_1h, mean_4h, std_4h, mean_eod, std_eod]
        n_features = len(feature_columns)
        base_obs_dim = lookback_window * n_features + 4
        ensemble_obs_dim = 6 if self.ensemble_models else 0

        self.ensemble_obs_dim = ensemble_obs_dim
        total_obs_dim = base_obs_dim + ensemble_obs_dim

        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        self._last_ensemble_pred = None

    def _get_ensemble_prediction(self) -> np.ndarray:
        """Get ensemble prediction from all models."""
        if not self.ensemble_models:
            return np.array([], dtype=np.float32)

        try:
            current_bar = self.df.iloc[min(self.current_idx, len(self.df) - 1)]
            features = current_bar[self.feature_columns].values.astype(np.float32)

            all_probs = {'1h': [], '4h': [], 'eod': []}

            for model, scaler in self.ensemble_models:
                # Scale features
                scaled = scaler.transform(features.reshape(1, -1))
                scaled = np.clip(scaled, -5, 5)
                scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

                x = torch.FloatTensor(scaled)

                with torch.no_grad():
                    model.eval()
                    logits_1h, logits_4h, logits_eod = model(x)
                    all_probs['1h'].append(torch.sigmoid(logits_1h).item())
                    all_probs['4h'].append(torch.sigmoid(logits_4h).item())
                    all_probs['eod'].append(torch.sigmoid(logits_eod).item())

            self._last_ensemble_pred = {
                'mean_1h': np.mean(all_probs['1h']),
                'std_1h': np.std(all_probs['1h']),
                'mean_4h': np.mean(all_probs['4h']),
                'std_4h': np.std(all_probs['4h']),
                'mean_eod': np.mean(all_probs['eod']),
                'std_eod': np.std(all_probs['eod']),
            }

            return np.array([
                self._last_ensemble_pred['mean_1h'],
                self._last_ensemble_pred['std_1h'],
                self._last_ensemble_pred['mean_4h'],
                self._last_ensemble_pred['std_4h'],
                self._last_ensemble_pred['mean_eod'],
                self._last_ensemble_pred['std_eod'],
            ], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Ensemble prediction error: {e}")
            return np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """Get observation including ensemble predictions."""
        base_obs = super()._get_observation()

        if self.ensemble_models:
            ensemble_obs = self._get_ensemble_prediction()
            obs = np.concatenate([base_obs, ensemble_obs])
        else:
            obs = base_obs

        return obs.astype(np.float32)


def load_ensemble_models(
    model_dir: str = "models/enhanced_walkforward",
) -> List[Tuple[RegularizedMultiHorizonNet, StandardScaler]]:
    """Load all walk-forward models as ensemble."""
    model_dir = Path(model_dir)
    model_files = sorted(model_dir.glob("window_*.pt"))

    models = []
    for model_file in model_files:
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

        feature_cols = checkpoint['feature_cols']

        model = RegularizedMultiHorizonNet(
            input_dim=len(feature_cols),
            hidden_dims=[512, 256, 128],
            dropout_rate=0.4,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'])
        scaler.scale_ = np.array(checkpoint['scaler_scale'])

        models.append((model, scaler))

    logger.info(f"Loaded {len(models)} ensemble models")
    return models


def prepare_data(
    data_path: str,
    start_date: str = None,
    end_date: str = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare data with enhanced features."""
    logger.info("Loading data with enhanced features...")

    pipeline = MultiHorizonDataPipeline(data_path)
    df = pipeline.load_and_aggregate(start_date, end_date)
    df, base_feature_cols = pipeline.generate_features(df, include_multi_horizon=False)

    # Add enhanced features
    df, enhanced_cols = combine_with_base_features(df, base_feature_cols)

    # Add targets
    df = create_multi_horizon_targets(df)

    # Clean data
    df = df.dropna(subset=enhanced_cols)
    for col in enhanced_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].fillna(0)

    logger.info(f"Prepared {len(df):,} samples with {len(enhanced_cols)} features")

    return df, enhanced_cols


class TradingMetricsCallback(BaseCallback):
    """Callback to log trading metrics during training."""

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if self.verbose > 0:
                logger.info(f"Step {self.n_calls}: Mean reward (last 100): {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

        return True


def main():
    parser = argparse.ArgumentParser(description="Train final enhanced hybrid RL agent")
    parser.add_argument("--data", type=str,
                       default="data/historical/MES/MES_1s_2years.parquet",
                       help="Path to data file")
    parser.add_argument("--ensemble-dir", type=str,
                       default="models/enhanced_walkforward",
                       help="Directory with ensemble models")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-09-30",
                       help="End date (leave some for testing)")
    parser.add_argument("--timesteps", type=int, default=2000000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--output", type=str, default="models/final_hybrid_agent",
                       help="Output path")

    args = parser.parse_args()

    print("=" * 70)
    print("    FINAL ENHANCED HYBRID RL AGENT")
    print("    Enhanced Features + Walk-Forward Ensemble + PPO")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:          {args.data}")
    print(f"  Date range:    {args.start_date} to {args.end_date}")
    print(f"  Timesteps:     {args.timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print()

    # Load data
    df, feature_cols = prepare_data(
        args.data,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Load ensemble models
    print("\n" + "-" * 70)
    print("Loading ensemble models...")
    print("-" * 70)

    ensemble_models = load_ensemble_models(args.ensemble_dir)

    # Split data
    n_samples = len(df)
    train_end = int(n_samples * 0.85)

    train_df = df.iloc[:train_end].copy()
    eval_df = df.iloc[train_end:].copy()

    logger.info(f"Train: {len(train_df):,} samples, Eval: {len(eval_df):,} samples")

    # Create environments
    print("\n" + "-" * 70)
    print("Creating training environments...")
    print("-" * 70)

    def make_train_env():
        env = EnhancedHybridEnvironment(
            df=train_df,
            feature_columns=feature_cols,
            ensemble_models=ensemble_models,
            initial_balance=1000.0,
            max_daily_loss=100.0,
            signal_reward_weight=0.05,
        )
        return Monitor(env)

    train_envs = DummyVecEnv([make_train_env for _ in range(args.n_envs)])

    # Create PPO agent
    print("\n" + "-" * 70)
    print("Initializing PPO agent...")
    print("-" * 70)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 256, 128],
            vf=[512, 256, 128],
        )
    )

    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
    )

    logger.info(f"Observation space: {train_envs.observation_space.shape}")
    logger.info(f"Action space: {train_envs.action_space}")

    # Train
    print("\n" + "-" * 70)
    print(f"Training for {args.timesteps:,} timesteps...")
    print("-" * 70)

    callback = TradingMetricsCallback(log_freq=20000)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")

    # Evaluate
    print("\n" + "-" * 70)
    print("Evaluating on held-out data...")
    print("-" * 70)

    eval_env = EnhancedHybridEnvironment(
        df=eval_df,
        feature_columns=feature_cols,
        ensemble_models=ensemble_models,
        initial_balance=1000.0,
        max_daily_loss=100.0,
    )

    rewards, pnls, trades, wins = [], [], [], []
    for ep in range(30):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        summary = eval_env.get_trade_summary()
        pnls.append(summary.get('total_pnl', 0))
        trades.append(summary.get('num_trades', 0))
        wins.append(summary.get('win_rate', 0))

    print(f"\nEvaluation Results (30 episodes):")
    print(f"  Mean Reward:   {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean PnL:      ${np.mean(pnls):.2f} ± ${np.std(pnls):.2f}")
    print(f"  Mean Trades:   {np.mean(trades):.1f}")
    print(f"  Mean Win Rate: {np.mean(wins)*100:.1f}%")
    print(f"  Profitable:    {sum(1 for p in pnls if p > 0)}/30 episodes")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(output_path))
    logger.info(f"Model saved to {output_path}.zip")

    # Save info
    info_dict = {
        'args': vars(args),
        'eval_results': {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_pnl': float(np.mean(pnls)),
            'std_pnl': float(np.std(pnls)),
            'mean_trades': float(np.mean(trades)),
            'mean_win_rate': float(np.mean(wins)),
            'profitable_episodes': sum(1 for p in pnls if p > 0),
        },
        'feature_cols': feature_cols,
        'n_ensemble_models': len(ensemble_models),
        'timestamp': datetime.now().isoformat(),
    }

    with open(f"{output_path}_info.json", 'w') as f:
        json.dump(info_dict, f, indent=2)

    print("\n" + "=" * 70)
    print("    TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}.zip")

    train_envs.close()


if __name__ == "__main__":
    main()
