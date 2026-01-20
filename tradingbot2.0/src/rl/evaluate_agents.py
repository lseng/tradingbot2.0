#!/usr/bin/env python3
"""
Evaluate and Compare RL Trading Agents.

Compares pure RL baseline vs hybrid RL agent with ML signals
on the same evaluation dataset.

Usage:
    python src/rl/evaluate_agents.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from src.rl.data_pipeline import MultiHorizonDataPipeline
from src.rl.trading_env import TradingEnvironment
from src.rl.hybrid_env import HybridTradingEnvironment, load_ml_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def evaluate_pure_rl(
    model_path: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_episodes: int = 50,
) -> Dict:
    """Evaluate pure RL agent (no ML signals)."""
    logger.info(f"Evaluating pure RL agent from {model_path}...")

    model = PPO.load(model_path)

    env = TradingEnvironment(
        df=df,
        feature_columns=feature_cols,
        initial_balance=1000.0,
        max_daily_loss=100.0,
    )

    all_rewards = []
    all_pnls = []
    all_trades = []
    all_win_rates = []
    all_profit_factors = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        all_rewards.append(episode_reward)

        summary = env.get_trade_summary()
        all_pnls.append(summary.get('total_pnl', 0))
        all_trades.append(summary.get('num_trades', 0))
        all_win_rates.append(summary.get('win_rate', 0))
        pf = summary.get('profit_factor', 0)
        if pf != float('inf'):
            all_profit_factors.append(pf)

    return {
        'model': 'Pure RL',
        'episodes': n_episodes,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_pnl': np.mean(all_pnls),
        'std_pnl': np.std(all_pnls),
        'total_pnl': np.sum(all_pnls),
        'mean_trades': np.mean(all_trades),
        'total_trades': np.sum(all_trades),
        'mean_win_rate': np.mean(all_win_rates),
        'mean_profit_factor': np.mean(all_profit_factors) if all_profit_factors else 0,
        'profitable_episodes': sum(1 for p in all_pnls if p > 0),
    }


def evaluate_hybrid(
    model_path: str,
    ml_model_path: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_episodes: int = 50,
    device: str = 'cpu',
) -> Dict:
    """Evaluate hybrid RL agent (with ML signals)."""
    logger.info(f"Evaluating hybrid agent from {model_path}...")

    model = PPO.load(model_path)

    ml_model = None
    ml_scaler_mean = None
    ml_scaler_scale = None
    ml_feature_cols = None

    if ml_model_path and Path(ml_model_path).exists():
        ml_model, ml_scaler_mean, ml_scaler_scale, ml_feature_cols = load_ml_model(
            ml_model_path, device
        )

    env = HybridTradingEnvironment(
        df=df,
        feature_columns=feature_cols,
        ml_model=ml_model,
        ml_scaler_mean=ml_scaler_mean,
        ml_scaler_scale=ml_scaler_scale,
        ml_feature_cols=ml_feature_cols,
        initial_balance=1000.0,
        max_daily_loss=100.0,
        signal_reward_weight=0.1,
        device=device,
    )

    all_rewards = []
    all_pnls = []
    all_trades = []
    all_win_rates = []
    all_profit_factors = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        all_rewards.append(episode_reward)

        summary = env.get_trade_summary()
        all_pnls.append(summary.get('total_pnl', 0))
        all_trades.append(summary.get('num_trades', 0))
        all_win_rates.append(summary.get('win_rate', 0))
        pf = summary.get('profit_factor', 0)
        if pf != float('inf'):
            all_profit_factors.append(pf)

    return {
        'model': 'Hybrid RL + ML',
        'episodes': n_episodes,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_pnl': np.mean(all_pnls),
        'std_pnl': np.std(all_pnls),
        'total_pnl': np.sum(all_pnls),
        'mean_trades': np.mean(all_trades),
        'total_trades': np.sum(all_trades),
        'mean_win_rate': np.mean(all_win_rates),
        'mean_profit_factor': np.mean(all_profit_factors) if all_profit_factors else 0,
        'profitable_episodes': sum(1 for p in all_pnls if p > 0),
    }


def print_comparison(pure_results: Dict, hybrid_results: Dict):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 80)
    print("                    AGENT COMPARISON RESULTS")
    print("=" * 80)

    metrics = [
        ('Mean Reward', 'mean_reward', '.2f', ''),
        ('Std Reward', 'std_reward', '.2f', ''),
        ('Mean PnL', 'mean_pnl', '.2f', '$'),
        ('Std PnL', 'std_pnl', '.2f', '$'),
        ('Total PnL', 'total_pnl', '.2f', '$'),
        ('Mean Trades/Ep', 'mean_trades', '.1f', ''),
        ('Total Trades', 'total_trades', 'd', ''),
        ('Mean Win Rate', 'mean_win_rate', '.1%', ''),
        ('Mean Profit Factor', 'mean_profit_factor', '.2f', ''),
        ('Profitable Episodes', 'profitable_episodes', 'd', ''),
    ]

    print(f"\n{'Metric':<25} {'Pure RL':>15} {'Hybrid RL':>15} {'Improvement':>15}")
    print("-" * 80)

    for name, key, fmt, prefix in metrics:
        pure_val = pure_results.get(key, 0)
        hybrid_val = hybrid_results.get(key, 0)

        if key in ['mean_pnl', 'total_pnl', 'mean_reward']:
            improvement = hybrid_val - pure_val
            imp_str = f"{'+' if improvement > 0 else ''}{improvement:{fmt}}"
        elif key == 'mean_win_rate':
            improvement = (hybrid_val - pure_val) * 100
            imp_str = f"{'+' if improvement > 0 else ''}{improvement:.1f}pp"
        else:
            if pure_val != 0:
                improvement = (hybrid_val - pure_val) / abs(pure_val) * 100
                imp_str = f"{'+' if improvement > 0 else ''}{improvement:.1f}%"
            else:
                imp_str = "N/A"

        pure_str = f"{prefix}{pure_val:{fmt}}"
        hybrid_str = f"{prefix}{hybrid_val:{fmt}}"

        print(f"{name:<25} {pure_str:>15} {hybrid_str:>15} {imp_str:>15}")

    print("-" * 80)

    # Summary
    pnl_improvement = hybrid_results['mean_pnl'] - pure_results['mean_pnl']
    win_rate_improvement = (hybrid_results['mean_win_rate'] - pure_results['mean_win_rate']) * 100

    print(f"\nSUMMARY:")
    print(f"  The hybrid agent {'outperforms' if pnl_improvement > 0 else 'underperforms'} pure RL by ${abs(pnl_improvement):.2f} per episode")
    print(f"  Win rate improvement: {'+' if win_rate_improvement > 0 else ''}{win_rate_improvement:.1f} percentage points")

    if hybrid_results['mean_pnl'] > 0:
        print(f"\n  *** HYBRID AGENT IS PROFITABLE! ***")
    elif hybrid_results['mean_pnl'] > pure_results['mean_pnl']:
        print(f"\n  Hybrid agent shows improvement, but more training may be needed.")


def main():
    parser = argparse.ArgumentParser(description="Compare RL trading agents")
    parser.add_argument("--data", type=str,
                       default="data/historical/MES/MES_1s_2years.parquet",
                       help="Path to data file")
    parser.add_argument("--pure-rl-model", type=str,
                       default="models/rl_trader_2024.zip",
                       help="Path to pure RL model")
    parser.add_argument("--hybrid-model", type=str,
                       default="models/hybrid_rl_agent.zip",
                       help="Path to hybrid RL model")
    parser.add_argument("--ml-model", type=str,
                       default="models/multi_horizon.pt",
                       help="Path to ML model for hybrid agent")
    parser.add_argument("--eval-start", type=str, default="2025-07-01",
                       help="Start date for evaluation")
    parser.add_argument("--eval-end", type=str, default="2025-12-31",
                       help="End date for evaluation")
    parser.add_argument("--n-episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for ML inference")

    args = parser.parse_args()

    print("=" * 80)
    print("           RL TRADING AGENT EVALUATION & COMPARISON")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data:          {args.data}")
    print(f"  Eval period:   {args.eval_start} to {args.eval_end}")
    print(f"  Episodes:      {args.n_episodes}")
    print(f"  Pure RL:       {args.pure_rl_model}")
    print(f"  Hybrid:        {args.hybrid_model}")
    print()

    # Load evaluation data
    logger.info("Loading evaluation data...")
    pipeline = MultiHorizonDataPipeline(args.data)
    df = pipeline.load_and_aggregate(args.eval_start, args.eval_end)
    df, feature_cols = pipeline.generate_features(df, include_multi_horizon=False)
    df = df.dropna(subset=feature_cols)

    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    logger.info(f"Loaded {len(df):,} evaluation samples")

    # Evaluate pure RL
    print("\n" + "-" * 80)
    print("Evaluating Pure RL Agent...")
    print("-" * 80)

    pure_results = evaluate_pure_rl(
        args.pure_rl_model,
        df,
        feature_cols,
        n_episodes=args.n_episodes,
    )

    # Evaluate hybrid
    print("\n" + "-" * 80)
    print("Evaluating Hybrid RL + ML Agent...")
    print("-" * 80)

    hybrid_results = evaluate_hybrid(
        args.hybrid_model,
        args.ml_model,
        df,
        feature_cols,
        n_episodes=args.n_episodes,
        device=args.device,
    )

    # Print comparison
    print_comparison(pure_results, hybrid_results)

    # Save results (convert numpy types to native Python)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = {
        'pure_rl': convert_numpy(pure_results),
        'hybrid': convert_numpy(hybrid_results),
        'eval_period': f"{args.eval_start} to {args.eval_end}",
        'n_samples': len(df),
        'timestamp': datetime.now().isoformat(),
    }

    with open('models/agent_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to models/agent_comparison.json")


if __name__ == "__main__":
    main()
