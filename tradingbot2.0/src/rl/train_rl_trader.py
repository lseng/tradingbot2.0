#!/usr/bin/env python3
"""
Train RL Trading Agent using PPO.

This script:
1. Loads and prepares multi-horizon data
2. Creates the trading environment
3. Trains a PPO agent to maximize PnL
4. Evaluates on validation data
5. Saves the trained policy

Usage:
    python src/rl/train_rl_trader.py
    python src/rl/train_rl_trader.py --timesteps 500000 --start-date 2024-01-01
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.trading_env import TradingEnvironment, create_env_from_data
from src.rl.data_pipeline import load_data_for_rl, MultiHorizonDataPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """Custom callback to log trading metrics during training."""

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        # Get info from environment
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'num_trades' in info:
                    self.episode_trades.append(info['num_trades'])

        # Log periodically
        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                logger.info(f"Step {self.n_calls}: Avg Reward = {np.mean(recent_rewards):.2f}, "
                           f"Avg Trades = {np.mean(self.episode_trades[-100:]) if self.episode_trades else 0:.1f}")

        return True


def create_train_val_envs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    n_envs: int = 4,
    **env_kwargs,
) -> tuple:
    """Create vectorized training and validation environments."""

    def make_train_env():
        env = TradingEnvironment(train_df, feature_cols, **env_kwargs)
        return Monitor(env)

    def make_val_env():
        env = TradingEnvironment(val_df, feature_cols, **env_kwargs)
        return Monitor(env)

    # Use DummyVecEnv for simplicity (SubprocVecEnv for parallel if needed)
    train_envs = DummyVecEnv([make_train_env for _ in range(n_envs)])
    val_env = DummyVecEnv([make_val_env])

    return train_envs, val_env


def train_ppo_agent(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_envs: int = 4,
    save_path: Optional[str] = None,
    **env_kwargs,
) -> tuple:
    """
    Train PPO agent on trading environment.

    Args:
        train_df: Training data with features
        val_df: Validation data
        feature_cols: Feature column names
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        n_envs: Number of parallel environments
        save_path: Path to save model
        **env_kwargs: Additional environment parameters

    Returns:
        (trained_model, training_info)
    """
    logger.info("Creating environments...")

    train_envs, val_env = create_train_val_envs(
        train_df, val_df, feature_cols, n_envs=n_envs, **env_kwargs
    )

    # Calculate observation and action dimensions
    obs_dim = train_envs.observation_space.shape[0]
    logger.info(f"Observation dim: {obs_dim}, Action space: {train_envs.action_space}")

    # Create PPO model
    logger.info("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=learning_rate,
        n_steps=2048,  # Steps per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.2,  # PPO clip range
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard (install tensorboard to enable)
        device='auto',
    )

    # Custom policy network (optional - could customize later)
    logger.info(f"Model policy: {model.policy}")

    # Callbacks
    callbacks = [
        TradingCallback(log_freq=10000, verbose=1),
    ]

    # Add eval callback if we have validation data
    if len(val_df) > 1000:
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=str(save_path) if save_path else "./models/",
            log_path="./logs/",
            eval_freq=max(10000, total_timesteps // 10),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_callback)

    # Train
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,  # Disable progress bar
    )

    # Save model
    if save_path:
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    # Get training info
    training_info = {
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'n_envs': n_envs,
        'obs_dim': obs_dim,
        'feature_cols': feature_cols,
    }

    # Cleanup
    train_envs.close()
    val_env.close()

    return model, training_info


def evaluate_agent(
    model: PPO,
    df: pd.DataFrame,
    feature_cols: list,
    n_episodes: int = 10,
    **env_kwargs,
) -> Dict[str, Any]:
    """
    Evaluate trained agent on data.

    Args:
        model: Trained PPO model
        df: Evaluation data
        feature_cols: Feature columns
        n_episodes: Number of evaluation episodes
        **env_kwargs: Environment parameters

    Returns:
        Evaluation metrics dict
    """
    env = TradingEnvironment(df, feature_cols, **env_kwargs)

    all_rewards = []
    all_trades = []
    all_pnls = []

    for ep in range(n_episodes):
        obs, info = env.reset(options={'start_day_idx': ep % len(env.trading_days)})
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        all_rewards.append(episode_reward)

        summary = env.get_trade_summary()
        all_trades.append(summary.get('num_trades', 0))
        all_pnls.append(summary.get('total_pnl', 0))

    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_trades': np.mean(all_trades),
        'mean_pnl': np.mean(all_pnls),
        'total_pnl': sum(all_pnls),
        'win_rate': sum(1 for p in all_pnls if p > 0) / len(all_pnls) if all_pnls else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--data", type=str,
                       default="data/historical/MES/MES_1s_2years.parquet",
                       help="Path to data file")
    parser.add_argument("--start-date", type=str, default="2024-01-01",
                       help="Start date for training data")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date for training data")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--initial-balance", type=float, default=1000.0,
                       help="Initial account balance")
    parser.add_argument("--max-daily-loss", type=float, default=100.0,
                       help="Maximum daily loss")
    parser.add_argument("--output", type=str, default="models/rl_trader",
                       help="Output path for model")

    args = parser.parse_args()

    print("=" * 60)
    print("    RL TRADING AGENT TRAINING")
    print("    Optimizing for PnL with Multi-Horizon Predictions")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:           {args.data}")
    print(f"  Date range:     {args.start_date} to {args.end_date or 'latest'}")
    print(f"  Timesteps:      {args.timesteps:,}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Parallel envs:  {args.n_envs}")
    print()

    # Load and prepare data
    logger.info("Loading and preparing data...")
    pipeline = MultiHorizonDataPipeline(args.data)

    train_df, val_df, feature_cols = pipeline.prepare_training_data(
        start_date=args.start_date,
        end_date=args.end_date,
        train_ratio=0.8,
    )

    logger.info(f"Train data: {len(train_df):,} bars ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Val data: {len(val_df):,} bars ({val_df.index.min()} to {val_df.index.max()})")
    logger.info(f"Features: {len(feature_cols)}")

    # Environment parameters
    env_kwargs = {
        'initial_balance': args.initial_balance,
        'max_daily_loss': args.max_daily_loss,
        'lookback_window': 60,  # 1 hour of 1-min bars
    }

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Train
    model, training_info = train_ppo_agent(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        n_envs=args.n_envs,
        save_path=str(output_path),
        **env_kwargs,
    )

    # Evaluate on validation data
    logger.info("\nEvaluating on validation data...")
    val_metrics = evaluate_agent(model, val_df, feature_cols, n_episodes=20, **env_kwargs)

    print("\n" + "=" * 60)
    print("    TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nValidation Results:")
    print(f"  Mean Reward:  {val_metrics['mean_reward']:.2f} (+/- {val_metrics['std_reward']:.2f})")
    print(f"  Mean Trades:  {val_metrics['mean_trades']:.1f} per episode")
    print(f"  Mean PnL:     ${val_metrics['mean_pnl']:.2f}")
    print(f"  Total PnL:    ${val_metrics['total_pnl']:.2f}")
    print(f"  Win Rate:     {val_metrics['win_rate']:.1%}")

    # Save training info
    info_path = output_path.parent / f"{output_path.name}_info.json"
    with open(info_path, 'w') as f:
        json.dump({
            'training_info': training_info,
            'validation_metrics': val_metrics,
            'args': vars(args),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nModel saved to: {output_path}")
    print(f"Info saved to: {info_path}")


if __name__ == "__main__":
    main()
