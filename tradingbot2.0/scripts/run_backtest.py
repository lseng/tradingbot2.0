#!/usr/bin/env python3
"""
Run Backtest Entry Point for MES Futures Scalping Bot.

This script provides a command-line interface to run backtests using the
backtesting engine with trained ML models on historical 1-second data.

Features:
- Load trained PyTorch models for signal generation
- Run single backtests or walk-forward validation
- Generate comprehensive performance reports
- Export trade logs and equity curves

Usage:
    # Run backtest with defaults
    python scripts/run_backtest.py

    # Run with specific model and data
    python scripts/run_backtest.py --model models/scalper_v1.pt --data data/historical/MES/MES_1s_2years.parquet

    # Run walk-forward validation
    python scripts/run_backtest.py --walk-forward --verbose

    # Custom parameters
    python scripts/run_backtest.py --stop-ticks 10 --target-ticks 20 --min-confidence 0.65
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Signal,
    SignalType,
    WalkForwardValidator,
    Position,
    OrderFillMode,
)
from src.ml.data.parquet_loader import ParquetDataLoader, load_and_prepare_scalping_data
from src.ml.data.scalping_features import ScalpingFeatureEngineer
from src.ml.models.neural_networks import FeedForwardNet, LSTMNet, HybridNet, ModelPrediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """
    Signal generator using trained ML model for predictions.

    This class wraps a trained PyTorch model and generates trading signals
    based on model predictions and confidence thresholds.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_engineer: ScalpingFeatureEngineer,
        min_confidence: float = 0.60,
        stop_ticks: float = 8.0,
        target_ticks: float = 16.0,
        device: str = 'cpu',
    ):
        """
        Initialize the ML signal generator.

        Args:
            model: Trained PyTorch model
            feature_engineer: Feature engineering instance
            min_confidence: Minimum confidence for trading
            stop_ticks: Default stop loss in ticks
            target_ticks: Default take profit in ticks
            device: Device to run inference on
        """
        self.model = model
        self.model.eval()
        self.feature_engineer = feature_engineer
        self.min_confidence = min_confidence
        self.stop_ticks = stop_ticks
        self.target_ticks = target_ticks
        self.device = device

        # Cache for features
        self._feature_cache: Dict[str, np.ndarray] = {}

    def generate_signal(
        self,
        bar: pd.Series,
        position: Optional[Position],
        context: Dict[str, Any],
    ) -> Signal:
        """
        Generate a trading signal for the current bar.

        Args:
            bar: Current OHLCV bar with features
            position: Current position (None if flat)
            context: Additional context (features DataFrame, etc.)

        Returns:
            Signal with direction and confidence
        """
        try:
            # Get features from the bar
            features = self._get_features(bar, context)

            if features is None:
                return Signal(SignalType.HOLD, reason="No features available")

            # Run model inference
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                logits = self.model(features_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            # Get prediction and confidence
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            # Check confidence threshold
            if confidence < self.min_confidence:
                return Signal(SignalType.HOLD, confidence=confidence, predicted_class=predicted_class)

            # If we have a position, check for exits
            if position is not None:
                return self._generate_exit_signal(position, predicted_class, confidence)

            # Generate entry signal
            return self._generate_entry_signal(predicted_class, confidence)

        except Exception as e:
            logger.warning(f"Error generating signal: {e}")
            return Signal(SignalType.HOLD, reason=f"Error: {str(e)}")

    def _get_features(
        self,
        bar: pd.Series,
        context: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Extract feature values from bar data."""
        # Try to get features from bar if they exist
        feature_names = self.feature_engineer.feature_names if hasattr(self.feature_engineer, 'feature_names') else []

        if not feature_names:
            # If no feature names, try to find numeric columns that look like features
            feature_cols = [col for col in bar.index if col.startswith(('return_', 'ema_', 'rsi_', 'vol_', 'vwap_'))]
            if feature_cols:
                return bar[feature_cols].values
            return None

        # Extract features by name
        features = []
        for name in feature_names:
            if name in bar.index:
                features.append(bar[name])
            else:
                # Missing feature - use 0 as default
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _generate_entry_signal(
        self,
        predicted_class: int,
        confidence: float,
    ) -> Signal:
        """Generate entry signal based on prediction."""
        if predicted_class == 2:  # UP
            return Signal(
                SignalType.LONG_ENTRY,
                confidence=confidence,
                predicted_class=predicted_class,
                stop_ticks=self.stop_ticks,
                target_ticks=self.target_ticks,
                reason="Model predicts UP",
            )
        elif predicted_class == 0:  # DOWN
            return Signal(
                SignalType.SHORT_ENTRY,
                confidence=confidence,
                predicted_class=predicted_class,
                stop_ticks=self.stop_ticks,
                target_ticks=self.target_ticks,
                reason="Model predicts DOWN",
            )
        else:  # FLAT
            return Signal(SignalType.HOLD, confidence=confidence, predicted_class=predicted_class)

    def _generate_exit_signal(
        self,
        position: Position,
        predicted_class: int,
        confidence: float,
    ) -> Signal:
        """Generate exit signal if prediction reverses."""
        # Exit long if prediction is DOWN
        if position.direction == 1 and predicted_class == 0:
            return Signal(
                SignalType.EXIT_LONG,
                confidence=confidence,
                predicted_class=predicted_class,
                reason="Model predicts DOWN",
            )
        # Exit short if prediction is UP
        elif position.direction == -1 and predicted_class == 2:
            return Signal(
                SignalType.EXIT_SHORT,
                confidence=confidence,
                predicted_class=predicted_class,
                reason="Model predicts UP",
            )

        return Signal(SignalType.HOLD, confidence=confidence, predicted_class=predicted_class)


def load_model(model_path: str, device: str = 'cpu') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Device to load model to

    Returns:
        Tuple of (model, config)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get model configuration
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'feedforward')
    input_size = config.get('input_size', 50)
    hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
    num_classes = config.get('num_classes', 3)

    # Create model based on type
    if model_type == 'feedforward':
        model = FeedForwardNet(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
        )
    elif model_type == 'lstm':
        model = LSTMNet(
            input_size=input_size,
            hidden_size=hidden_sizes[0] if hidden_sizes else 128,
            num_layers=config.get('num_layers', 2),
            num_classes=num_classes,
        )
    elif model_type == 'hybrid':
        model = HybridNet(
            input_size=input_size,
            lstm_hidden=config.get('lstm_hidden', 64),
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Loaded {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, config


def create_random_signal_generator(
    min_confidence: float = 0.60,
    stop_ticks: float = 8.0,
    target_ticks: float = 16.0,
):
    """
    Create a random signal generator for baseline testing.

    This generator produces random signals to establish a baseline
    (should produce ~0 expectancy after costs).
    """
    def random_signal_generator(
        bar: pd.Series,
        position: Optional[Position],
        context: Dict[str, Any],
    ) -> Signal:
        # Random prediction: 0=DOWN, 1=FLAT, 2=UP
        # Bias toward FLAT (60% FLAT, 20% DOWN, 20% UP)
        rand = np.random.random()
        if rand < 0.2:
            predicted_class = 0  # DOWN
        elif rand < 0.4:
            predicted_class = 2  # UP
        else:
            predicted_class = 1  # FLAT

        confidence = np.random.uniform(0.5, 0.9)

        # Only trade with sufficient confidence
        if confidence < min_confidence:
            return Signal(SignalType.HOLD, confidence=confidence)

        # Generate signals based on position and prediction
        if position is not None:
            if position.direction == 1 and predicted_class == 0:
                return Signal(SignalType.EXIT_LONG, confidence=confidence, predicted_class=predicted_class)
            elif position.direction == -1 and predicted_class == 2:
                return Signal(SignalType.EXIT_SHORT, confidence=confidence, predicted_class=predicted_class)
            return Signal(SignalType.HOLD, confidence=confidence)

        if predicted_class == 2:
            return Signal(
                SignalType.LONG_ENTRY,
                confidence=confidence,
                predicted_class=predicted_class,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
            )
        elif predicted_class == 0:
            return Signal(
                SignalType.SHORT_ENTRY,
                confidence=confidence,
                predicted_class=predicted_class,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
            )

        return Signal(SignalType.HOLD, confidence=confidence)

    return random_signal_generator


def run_backtest(
    data_path: str,
    model_path: Optional[str] = None,
    output_dir: str = './results',
    config: Optional[BacktestConfig] = None,
    walk_forward: bool = False,
    verbose: bool = False,
    random_baseline: bool = False,
    limit_bars: Optional[int] = None,
) -> BacktestResult:
    """
    Run a complete backtest.

    Args:
        data_path: Path to parquet data file
        model_path: Path to trained model (None for random baseline)
        output_dir: Directory to save results
        config: Backtest configuration
        walk_forward: Whether to run walk-forward validation
        verbose: Print verbose output
        random_baseline: Use random signals for baseline test
        limit_bars: Limit number of bars for quick testing

    Returns:
        BacktestResult with performance metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use default config if not provided
    if config is None:
        config = BacktestConfig()

    # Load data
    logger.info(f"Loading data from {data_path}")
    loader = ParquetDataLoader(data_path)
    df = loader.load_data()
    df = loader.convert_to_ny_timezone(df)
    df = loader.filter_rth(df)

    # Limit bars for quick testing
    if limit_bars is not None and len(df) > limit_bars:
        logger.info(f"Limiting to {limit_bars:,} bars for quick test")
        df = df.iloc[:limit_bars]

    # Add features
    logger.info("Computing features...")
    feature_engineer = ScalpingFeatureEngineer()
    df = feature_engineer.add_all_features(df)

    # Create target variable
    df = loader.create_target_variable(df, lookahead_seconds=30, threshold_ticks=3.0)

    # Remove rows with NaN features
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_rows - len(df):,} rows with NaN values")

    # Create signal generator
    if random_baseline:
        logger.info("Using RANDOM signal generator (baseline test)")
        signal_generator = create_random_signal_generator(
            min_confidence=config.min_confidence,
            stop_ticks=config.default_stop_ticks,
            target_ticks=config.default_target_ticks,
        )
    elif model_path is not None and os.path.exists(model_path):
        logger.info(f"Using ML model from {model_path}")
        model, model_config = load_model(model_path)
        ml_generator = MLSignalGenerator(
            model=model,
            feature_engineer=feature_engineer,
            min_confidence=config.min_confidence,
            stop_ticks=config.default_stop_ticks,
            target_ticks=config.default_target_ticks,
        )
        signal_generator = ml_generator.generate_signal
    else:
        logger.info("No model provided, using random baseline")
        signal_generator = create_random_signal_generator(
            min_confidence=config.min_confidence,
            stop_ticks=config.default_stop_ticks,
            target_ticks=config.default_target_ticks,
        )

    # Create backtest engine
    engine = BacktestEngine(config=config)

    # Run backtest
    if walk_forward:
        logger.info("Running walk-forward validation...")
        validator = WalkForwardValidator(
            training_months=6,
            validation_months=1,
            test_months=1,
            step_months=1,
        )
        results = validator.run_walk_forward(df, engine, signal_generator, verbose=verbose)

        # Combine results
        if results:
            result = results[-1]  # Use last fold for main result

            # Save all fold results
            for i, fold_result in enumerate(results):
                fold_path = output_path / f"fold_{i+1}"
                fold_path.mkdir(exist_ok=True)
                fold_result.report.export_all(str(fold_path))

            logger.info(f"Walk-forward complete: {len(results)} folds")
        else:
            raise ValueError("Walk-forward validation produced no results")
    else:
        logger.info("Running single backtest...")
        result = engine.run(df, signal_generator, verbose=verbose)

    # Log results
    metrics = result.report.metrics
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trades: {metrics.total_trades}")
    logger.info(f"Win rate: {metrics.win_rate:.1%}")
    logger.info(f"Profit factor: {metrics.profit_factor:.2f}")
    logger.info(f"Total return: ${metrics.total_net_pnl:.2f} ({metrics.total_return_pct:.2%})")
    logger.info(f"Max drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2%})")
    logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    logger.info(f"Sortino ratio: {metrics.sortino_ratio:.2f}")
    logger.info(f"Calmar ratio: {metrics.calmar_ratio:.2f}")
    logger.info(f"Execution time: {result.execution_time_seconds:.2f}s")
    logger.info("=" * 60)

    # Export results
    logger.info(f"Exporting results to {output_path}")
    result.report.export_all(str(output_path))

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MES Futures Scalping Backtest',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and model paths
    parser.add_argument(
        '--data',
        type=str,
        default='data/historical/MES/MES_1s_2years.parquet',
        help='Path to parquet data file',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model checkpoint (.pt)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results',
    )

    # Backtest configuration
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=1000.0,
        help='Initial account capital',
    )
    parser.add_argument(
        '--stop-ticks',
        type=float,
        default=8.0,
        help='Default stop loss in ticks',
    )
    parser.add_argument(
        '--target-ticks',
        type=float,
        default=16.0,
        help='Default take profit in ticks',
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.60,
        help='Minimum confidence threshold',
    )
    parser.add_argument(
        '--max-daily-loss',
        type=float,
        default=50.0,
        help='Maximum daily loss limit',
    )

    # Run modes
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Run walk-forward validation instead of single backtest',
    )
    parser.add_argument(
        '--random-baseline',
        action='store_true',
        help='Run with random signals (baseline test)',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output',
    )
    parser.add_argument(
        '--limit-bars',
        type=int,
        default=None,
        help='Limit number of bars for quick testing',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        min_confidence=args.min_confidence,
        default_stop_ticks=args.stop_ticks,
        default_target_ticks=args.target_ticks,
        max_daily_loss=args.max_daily_loss,
    )

    # Run backtest
    try:
        result = run_backtest(
            data_path=args.data,
            model_path=args.model,
            output_dir=args.output,
            config=config,
            walk_forward=args.walk_forward,
            verbose=args.verbose,
            random_baseline=args.random_baseline,
            limit_bars=args.limit_bars,
        )

        # Return exit code based on result
        if result.report.metrics.total_trades == 0:
            logger.warning("No trades generated - check model and data")
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == '__main__':
    main()
