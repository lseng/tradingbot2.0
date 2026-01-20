"""
Walk-Forward Training System for Trading Models.

Walk-forward analysis is a method for validating trading strategies that:
1. Trains on a rolling window of historical data
2. Validates on the next out-of-sample period
3. Rolls forward and repeats

This prevents look-ahead bias and tests strategy robustness across
different market conditions.

Example:
    Window 1: Train on Jan-Jun, Test on Jul-Aug
    Window 2: Train on Mar-Aug, Test on Sep-Oct
    Window 3: Train on May-Oct, Test on Nov-Dec
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .regularized_model import RegularizedMultiHorizonNet, RegularizedTrainer
from .multi_horizon_model import create_multi_horizon_targets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """
    Walk-forward training system for trading models.

    Implements anchored or rolling walk-forward validation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_months: int = 12,
        test_months: int = 2,
        step_months: int = 2,
        anchored: bool = False,
        model_save_dir: str = "models/walk_forward",
    ):
        """
        Initialize walk-forward trainer.

        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            train_months: Size of training window in months
            test_months: Size of test window in months
            step_months: Step size between windows
            anchored: If True, always start from the beginning (expanding window)
            model_save_dir: Directory to save models
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.anchored = anchored
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Get date range
        self.start_date = self.df.index.min()
        self.end_date = self.df.index.max()

        logger.info(f"Data range: {self.start_date} to {self.end_date}")

    def generate_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate train/test windows for walk-forward validation.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []

        # Start first window
        if self.anchored:
            train_start = self.start_date
        else:
            train_start = self.start_date

        while True:
            # Calculate window boundaries
            if self.anchored:
                # Anchored: always start from beginning, expand training
                train_end = train_start + pd.DateOffset(months=self.train_months + len(windows) * self.step_months)
            else:
                # Rolling: fixed training window size
                train_end = train_start + pd.DateOffset(months=self.train_months)

            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Check if test period extends beyond data
            if test_end > self.end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Move to next window
            if self.anchored:
                # Anchored doesn't move train_start
                pass
            else:
                train_start = train_start + pd.DateOffset(months=self.step_months)

            # Safety check for anchored mode
            if self.anchored and len(windows) > 20:
                break

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def prepare_window_data(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> Tuple[DataLoader, DataLoader, pd.DataFrame, StandardScaler]:
        """Prepare data loaders for a specific window."""

        # Split data
        train_mask = (self.df.index >= train_start) & (self.df.index < train_end)
        test_mask = (self.df.index >= test_start) & (self.df.index < test_end)

        train_df = self.df[train_mask].copy()
        test_df = self.df[test_mask].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Empty split: train={len(train_df)}, test={len(test_df)}")

        # Fit scaler on training data only
        scaler = StandardScaler()
        train_df[self.feature_cols] = scaler.fit_transform(train_df[self.feature_cols])
        test_df[self.feature_cols] = scaler.transform(test_df[self.feature_cols])

        # Handle inf/nan
        for split_df in [train_df, test_df]:
            split_df[self.feature_cols] = split_df[self.feature_cols].replace([np.inf, -np.inf], 0)
            split_df[self.feature_cols] = split_df[self.feature_cols].fillna(0)
            split_df[self.feature_cols] = split_df[self.feature_cols].clip(-5, 5)

        # Create data loaders
        def df_to_loader(df, batch_size=256, shuffle=True):
            X = torch.FloatTensor(df[self.feature_cols].values)
            y_1h = torch.LongTensor(df['target_1h'].values)
            y_4h = torch.LongTensor(df['target_4h'].values)
            y_eod = torch.LongTensor(df['target_eod'].values)
            dataset = TensorDataset(X, y_1h, y_4h, y_eod)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        train_loader = df_to_loader(train_df, shuffle=True)
        test_loader = df_to_loader(test_df, shuffle=False)

        return train_loader, test_loader, test_df, scaler

    def train_window(
        self,
        window_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
    ) -> Tuple[RegularizedMultiHorizonNet, Dict]:
        """Train model for a single window."""

        input_dim = len(self.feature_cols)

        model = RegularizedMultiHorizonNet(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.4,
            num_residual_blocks=2,
        )

        trainer = RegularizedTrainer(
            model,
            learning_rate=0.0005,
            weight_decay=0.05,
            label_smoothing=0.1,
        )

        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            patience=patience,
        )

        return model, history

    def evaluate_window(
        self,
        model: RegularizedMultiHorizonNet,
        test_df: pd.DataFrame,
        scaler: StandardScaler,
    ) -> Dict[str, float]:
        """Evaluate model on test window."""
        model.eval()

        device = next(model.parameters()).device
        X_test = torch.FloatTensor(test_df[self.feature_cols].values).to(device)

        with torch.no_grad():
            logits_1h, logits_4h, logits_eod = model(X_test)
            probs_1h = torch.sigmoid(logits_1h).cpu().numpy().flatten()
            probs_4h = torch.sigmoid(logits_4h).cpu().numpy().flatten()
            probs_eod = torch.sigmoid(logits_eod).cpu().numpy().flatten()

        y_1h = test_df['target_1h'].values
        y_4h = test_df['target_4h'].values
        y_eod = test_df['target_eod'].values

        results = {
            'acc_1h': ((probs_1h > 0.5).astype(int) == y_1h).mean(),
            'acc_4h': ((probs_4h > 0.5).astype(int) == y_4h).mean(),
            'acc_eod': ((probs_eod > 0.5).astype(int) == y_eod).mean(),
            'n_samples': len(test_df),
        }

        # High confidence accuracy
        for horizon, probs, y_true in [('1h', probs_1h, y_1h), ('4h', probs_4h, y_4h), ('eod', probs_eod, y_eod)]:
            high_conf_mask = (probs >= 0.6) | (probs <= 0.4)
            if high_conf_mask.sum() > 0:
                preds = (probs[high_conf_mask] > 0.5).astype(int)
                results[f'acc_{horizon}_conf60'] = (preds == y_true[high_conf_mask]).mean()
                results[f'n_{horizon}_conf60'] = high_conf_mask.sum()

        return results

    def run(
        self,
        epochs: int = 50,
        patience: int = 10,
    ) -> Dict:
        """
        Run full walk-forward training and evaluation.

        Returns:
            Dictionary with all window results and aggregate metrics
        """
        windows = self.generate_windows()

        all_results = []
        all_models = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {i+1}/{len(windows)}")
            logger.info(f"Train: {train_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {test_start.date()} to {test_end.date()}")
            logger.info(f"{'='*60}")

            try:
                # Prepare data
                train_loader, test_loader, test_df, scaler = self.prepare_window_data(
                    train_start, train_end, test_start, test_end
                )

                logger.info(f"Train samples: {len(train_loader.dataset):,}")
                logger.info(f"Test samples: {len(test_df):,}")

                # Train
                model, history = self.train_window(
                    i, train_loader, test_loader, epochs, patience
                )

                # Evaluate
                results = self.evaluate_window(model, test_df, scaler)
                results['window_idx'] = i
                results['train_start'] = str(train_start.date())
                results['train_end'] = str(train_end.date())
                results['test_start'] = str(test_start.date())
                results['test_end'] = str(test_end.date())

                logger.info(f"\nWindow {i+1} Results:")
                logger.info(f"  1h Acc:  {results['acc_1h']:.3f}")
                logger.info(f"  4h Acc:  {results['acc_4h']:.3f}")
                logger.info(f"  EOD Acc: {results['acc_eod']:.3f}")

                all_results.append(results)
                all_models.append((model, scaler))

                # Save window model
                model_path = self.model_save_dir / f"window_{i:02d}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler_mean': scaler.mean_.tolist(),
                    'scaler_scale': scaler.scale_.tolist(),
                    'feature_cols': self.feature_cols,
                    'results': results,
                }, model_path)

            except Exception as e:
                logger.error(f"Window {i+1} failed: {e}")
                continue

        # Aggregate results
        aggregate = {
            'n_windows': len(all_results),
            'avg_acc_1h': np.mean([r['acc_1h'] for r in all_results]),
            'avg_acc_4h': np.mean([r['acc_4h'] for r in all_results]),
            'avg_acc_eod': np.mean([r['acc_eod'] for r in all_results]),
            'std_acc_1h': np.std([r['acc_1h'] for r in all_results]),
            'std_acc_4h': np.std([r['acc_4h'] for r in all_results]),
            'std_acc_eod': np.std([r['acc_eod'] for r in all_results]),
        }

        # Save summary
        summary = {
            'aggregate': aggregate,
            'windows': all_results,
            'config': {
                'train_months': self.train_months,
                'test_months': self.test_months,
                'step_months': self.step_months,
                'anchored': self.anchored,
                'n_features': len(self.feature_cols),
            },
            'timestamp': datetime.now().isoformat(),
        }

        with open(self.model_save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Windows completed: {aggregate['n_windows']}")
        logger.info(f"Average 1h Accuracy:  {aggregate['avg_acc_1h']:.3f} ± {aggregate['std_acc_1h']:.3f}")
        logger.info(f"Average 4h Accuracy:  {aggregate['avg_acc_4h']:.3f} ± {aggregate['std_acc_4h']:.3f}")
        logger.info(f"Average EOD Accuracy: {aggregate['avg_acc_eod']:.3f} ± {aggregate['std_acc_eod']:.3f}")

        return summary


class EnsemblePredictor:
    """
    Ensemble predictor using multiple walk-forward models.

    Combines predictions from models trained on different time periods
    for more robust predictions.
    """

    def __init__(self, model_dir: str = "models/walk_forward"):
        self.model_dir = Path(model_dir)
        self.models = []
        self.scalers = []
        self.feature_cols = None

        self._load_models()

    def _load_models(self):
        """Load all window models."""
        model_files = sorted(self.model_dir.glob("window_*.pt"))

        for model_file in model_files:
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

            if self.feature_cols is None:
                self.feature_cols = checkpoint['feature_cols']

            # Recreate model
            model = RegularizedMultiHorizonNet(
                input_dim=len(self.feature_cols),
                hidden_dims=[512, 256, 128],
                dropout_rate=0.4,
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Recreate scaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(checkpoint['scaler_mean'])
            scaler.scale_ = np.array(checkpoint['scaler_scale'])

            self.models.append(model)
            self.scalers.append(scaler)

        logger.info(f"Loaded {len(self.models)} ensemble models")

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Make ensemble prediction.

        Returns mean and std of predictions across all models.
        """
        all_probs = {'1h': [], '4h': [], 'eod': []}

        for model, scaler in zip(self.models, self.scalers):
            # Scale features
            scaled = scaler.transform(features.reshape(1, -1))
            scaled = np.clip(scaled, -5, 5)

            x = torch.FloatTensor(scaled)

            with torch.no_grad():
                logits_1h, logits_4h, logits_eod = model(x)
                all_probs['1h'].append(torch.sigmoid(logits_1h).item())
                all_probs['4h'].append(torch.sigmoid(logits_4h).item())
                all_probs['eod'].append(torch.sigmoid(logits_eod).item())

        return {
            'prob_1h_mean': np.mean(all_probs['1h']),
            'prob_1h_std': np.std(all_probs['1h']),
            'prob_4h_mean': np.mean(all_probs['4h']),
            'prob_4h_std': np.std(all_probs['4h']),
            'prob_eod_mean': np.mean(all_probs['eod']),
            'prob_eod_std': np.std(all_probs['eod']),
        }
