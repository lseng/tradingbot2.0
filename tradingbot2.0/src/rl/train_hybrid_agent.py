#!/usr/bin/env python3
"""
Train Hybrid RL Agent with ML Signal Integration.

This script trains a PPO agent that receives both:
1. Raw market features (price action, indicators)
2. ML model predictions (1h, 4h, EOD direction probabilities)

The RL agent learns when to trust the ML signals and when to override them,
optimizing directly for trading PnL.

Architecture:
    MARKET DATA → SUPERVISED ML MODEL → RL AGENT → TRADING ACTIONS
                  (1h, 4h, EOD probs)

Usage:
    python src/rl/train_hybrid_agent.py
    python src/rl/train_hybrid_agent.py --timesteps 500000 --ml-model models/multi_horizon.pt
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.rl.data_pipeline import MultiHorizonDataPipeline
from src.rl.hybrid_env import HybridTradingEnvironment, load_ml_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """Callback to log trading-specific metrics during training."""

    def __init__(self, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_pnls = []

    def _on_step(self) -> bool:
        # Collect episode info
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

            if 'num_trades' in info:
                pass  # Could track per-episode trade stats

        if self.n_calls % self.eval_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if self.verbose > 0:
                logger.info(f"Step {self.n_calls}: Mean reward (last 100): {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

        return True


def prepare_data(
    data_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare data for hybrid training."""
    logger.info("Loading data for hybrid training...")

    pipeline = MultiHorizonDataPipeline(data_path)
    df = pipeline.load_and_aggregate(start_date, end_date)
    df, feature_cols = pipeline.generate_features(df, include_multi_horizon=False)

    # Drop rows with NaN features
    df = df.dropna(subset=feature_cols)

    # Handle inf values
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    logger.info(f"Prepared {len(df):,} samples with {len(feature_cols)} features")

    return df, feature_cols


def make_hybrid_env(
    df: pd.DataFrame,
    feature_cols: List[str],
    ml_model_path: Optional[str] = None,
    device: str = 'cpu',
    initial_balance: float = 1000.0,
    max_daily_loss: float = 100.0,
    signal_reward_weight: float = 0.1,
) -> HybridTradingEnvironment:
    """Create a hybrid trading environment."""
    from src.rl.hybrid_env import load_ml_model

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
        feature_columns=feature_cols,
        ml_model=ml_model,
        ml_scaler_mean=ml_scaler_mean,
        ml_scaler_scale=ml_scaler_scale,
        ml_feature_cols=ml_feature_cols,
        initial_balance=initial_balance,
        max_daily_loss=max_daily_loss,
        signal_reward_weight=signal_reward_weight,
        device=device,
    )


def create_vec_envs(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_envs: int = 4,
    ml_model_path: Optional[str] = None,
    device: str = 'cpu',
    **env_kwargs,
) -> DummyVecEnv:
    """Create vectorized environments for parallel training."""

    def make_env():
        env = make_hybrid_env(
            df, feature_cols, ml_model_path, device, **env_kwargs
        )
        return Monitor(env)

    # Use DummyVecEnv to avoid serialization issues with ML model
    envs = DummyVecEnv([make_env for _ in range(n_envs)])

    return envs


def evaluate_agent(
    model: PPO,
    eval_env: HybridTradingEnvironment,
    n_episodes: int = 10,
) -> Dict:
    """Evaluate trained agent on evaluation environment."""
    logger.info(f"Evaluating agent over {n_episodes} episodes...")

    all_rewards = []
    all_pnls = []
    all_trades = []
    all_wins = []

    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        all_rewards.append(episode_reward)

        # Get trade summary
        summary = eval_env.get_trade_summary()
        all_pnls.append(summary.get('total_pnl', 0))
        all_trades.append(summary.get('num_trades', 0))
        all_wins.append(summary.get('win_rate', 0))

    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_pnl': np.mean(all_pnls),
        'std_pnl': np.std(all_pnls),
        'mean_trades': np.mean(all_trades),
        'mean_win_rate': np.mean(all_wins),
        'total_episodes': n_episodes,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train hybrid RL agent with ML signals")
    parser.add_argument("--data", type=str,
                       default="data/historical/MES/MES_1s_2years.parquet",
                       help="Path to data file")
    parser.add_argument("--ml-model", type=str,
                       default="models/multi_horizon.pt",
                       help="Path to pre-trained ML model")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Start date for training data")
    parser.add_argument("--end-date", type=str, default="2025-06-30",
                       help="End date for training data")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--initial-balance", type=float, default=1000.0,
                       help="Initial account balance")
    parser.add_argument("--max-daily-loss", type=float, default=100.0,
                       help="Maximum daily loss limit")
    parser.add_argument("--signal-weight", type=float, default=0.1,
                       help="Weight for ML signal-following reward")
    parser.add_argument("--output", type=str, default="models/hybrid_rl_agent",
                       help="Output path for model")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for ML inference (cpu/cuda/mps)")

    args = parser.parse_args()

    print("=" * 70)
    print("    HYBRID RL TRADING AGENT")
    print("    Combining ML Predictions with Reinforcement Learning")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:           {args.data}")
    print(f"  ML Model:       {args.ml_model}")
    print(f"  Date range:     {args.start_date} to {args.end_date}")
    print(f"  Timesteps:      {args.timesteps:,}")
    print(f"  Parallel envs:  {args.n_envs}")
    print(f"  Signal weight:  {args.signal_weight}")
    print()

    # Prepare data
    df, feature_cols = prepare_data(
        args.data,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Split data for train/eval
    n_samples = len(df)
    train_end = int(n_samples * 0.85)

    train_df = df.iloc[:train_end].copy()
    eval_df = df.iloc[train_end:].copy()

    logger.info(f"Train: {len(train_df):,} samples, Eval: {len(eval_df):,} samples")

    # Create environments
    print("\n" + "-" * 70)
    print("Creating training environments...")
    print("-" * 70)

    train_envs = create_vec_envs(
        train_df,
        feature_cols,
        n_envs=args.n_envs,
        ml_model_path=args.ml_model,
        device=args.device,
        initial_balance=args.initial_balance,
        max_daily_loss=args.max_daily_loss,
        signal_reward_weight=args.signal_weight,
    )

    # Create eval environment
    eval_env = make_hybrid_env(
        eval_df,
        feature_cols,
        ml_model_path=args.ml_model,
        device=args.device,
        initial_balance=args.initial_balance,
        max_daily_loss=args.max_daily_loss,
        signal_reward_weight=args.signal_weight,
    )

    # Create PPO agent with custom policy network
    print("\n" + "-" * 70)
    print("Initializing PPO agent...")
    print("-" * 70)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64],  # Policy network
            vf=[256, 128, 64],  # Value network
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
        ent_coef=0.01,  # Entropy bonus for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard for now
    )

    logger.info(f"PPO agent created")
    logger.info(f"  Observation space: {train_envs.observation_space.shape}")
    logger.info(f"  Action space: {train_envs.action_space}")

    # Setup callbacks
    metrics_callback = TradingMetricsCallback(eval_freq=10000)

    # Train
    print("\n" + "-" * 70)
    print("Training hybrid RL agent...")
    print("-" * 70)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=metrics_callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Evaluate
    print("\n" + "-" * 70)
    print("Evaluating trained agent...")
    print("-" * 70)

    eval_results = evaluate_agent(model, eval_env, n_episodes=20)

    print(f"\nEvaluation Results (20 episodes):")
    print(f"  Mean Reward:    {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean PnL:       ${eval_results['mean_pnl']:.2f} ± ${eval_results['std_pnl']:.2f}")
    print(f"  Mean Trades:    {eval_results['mean_trades']:.1f}")
    print(f"  Mean Win Rate:  {eval_results['mean_win_rate']*100:.1f}%")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(f"{output_path}")
    logger.info(f"Model saved to {output_path}.zip")

    # Save training info
    info_dict = {
        'args': vars(args),
        'eval_results': eval_results,
        'feature_cols': feature_cols,
        'train_samples': len(train_df),
        'eval_samples': len(eval_df),
        'observation_dim': train_envs.observation_space.shape[0],
        'timestamp': datetime.now().isoformat(),
    }

    with open(f"{output_path}_info.json", 'w') as f:
        json.dump(info_dict, f, indent=2)

    print("\n" + "=" * 70)
    print("    TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}.zip")
    print(f"\nThe hybrid agent combines ML predictions with RL decision-making")
    print(f"for optimal trade entry/exit timing.")

    # Close environments
    train_envs.close()


if __name__ == "__main__":
    main()
