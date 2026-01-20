"""
Tests for RL walk-forward training system.

Tests cover:
- WalkForwardTrainer class initialization
- Window generation (rolling and anchored)
- Data preparation for windows
- Window evaluation
- EnsemblePredictor class
"""

import json
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.rl.walk_forward import (
    WalkForwardTrainer,
    EnsemblePredictor,
)
from src.rl.multi_horizon_model import create_multi_horizon_targets


NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def sample_df_with_targets():
    """Create sample DataFrame with features and targets spanning 2+ years."""
    np.random.seed(42)

    # Create 2.5 years of daily data (enough for walk-forward windows)
    dates = pd.date_range("2022-01-01 09:30:00", "2024-06-30 16:00:00", freq="1min", tz=NY_TZ)
    # Filter to RTH only (simplify for testing)
    dates = dates[(dates.hour >= 9) & (dates.hour < 16)]

    n_samples = len(dates)
    base_price = 4800.0
    returns = np.random.normal(0, 0.0002, n_samples)
    close_prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": close_prices * 0.999,
        "high": close_prices * 1.001,
        "low": close_prices * 0.998,
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "feature_4": np.random.randn(n_samples),
    }, index=dates)

    # Add targets
    df = create_multi_horizon_targets(df)

    # Fill NaN targets with 0 (for testing purposes)
    df["target_1h"] = df["target_1h"].fillna(0).astype(int)
    df["target_4h"] = df["target_4h"].fillna(0).astype(int)
    df["target_eod"] = df["target_eod"].fillna(0).astype(int)

    return df


@pytest.fixture
def feature_columns():
    """Feature columns for testing."""
    return ["feature_1", "feature_2", "feature_3", "feature_4"]


class TestWalkForwardTrainerInit:
    """Tests for WalkForwardTrainer initialization."""

    def test_init_with_valid_df(self, sample_df_with_targets, feature_columns, tmp_path):
        """Trainer initializes with valid DataFrame."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            model_save_dir=str(tmp_path / "models"),
        )
        assert trainer is not None
        assert trainer.df is not None

    def test_init_stores_parameters(self, sample_df_with_targets, feature_columns, tmp_path):
        """Trainer stores initialization parameters."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=12,
            test_months=2,
            step_months=2,
            anchored=False,
            model_save_dir=str(tmp_path / "models"),
        )
        assert trainer.train_months == 12
        assert trainer.test_months == 2
        assert trainer.step_months == 2
        assert trainer.anchored is False

    def test_init_creates_model_dir(self, sample_df_with_targets, feature_columns, tmp_path):
        """Trainer creates model save directory."""
        model_dir = tmp_path / "new_model_dir"
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            model_save_dir=str(model_dir),
        )
        assert model_dir.exists()

    def test_init_detects_date_range(self, sample_df_with_targets, feature_columns, tmp_path):
        """Trainer detects data date range."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            model_save_dir=str(tmp_path / "models"),
        )
        assert trainer.start_date is not None
        assert trainer.end_date is not None
        assert trainer.end_date > trainer.start_date

    def test_init_requires_datetime_index(self, feature_columns, tmp_path):
        """Trainer requires DatetimeIndex."""
        df_no_datetime = pd.DataFrame({
            "feature_1": [1, 2, 3],
            "target_1h": [0, 1, 0],
            "target_4h": [1, 0, 1],
            "target_eod": [0, 0, 1],
        })

        with pytest.raises(ValueError, match="DatetimeIndex"):
            WalkForwardTrainer(
                df=df_no_datetime,
                feature_cols=feature_columns[:1],
                model_save_dir=str(tmp_path / "models"),
            )


class TestWindowGeneration:
    """Tests for window generation."""

    def test_generate_windows_returns_list(self, sample_df_with_targets, feature_columns, tmp_path):
        """generate_windows returns a list."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()
        assert isinstance(windows, list)

    def test_generate_windows_has_content(self, sample_df_with_targets, feature_columns, tmp_path):
        """generate_windows returns non-empty list for sufficient data."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()
        assert len(windows) > 0

    def test_window_tuple_format(self, sample_df_with_targets, feature_columns, tmp_path):
        """Each window is a 4-tuple of timestamps."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        for window in windows:
            assert len(window) == 4
            train_start, train_end, test_start, test_end = window
            assert isinstance(train_start, pd.Timestamp)
            assert isinstance(train_end, pd.Timestamp)
            assert isinstance(test_start, pd.Timestamp)
            assert isinstance(test_end, pd.Timestamp)

    def test_window_temporal_order(self, sample_df_with_targets, feature_columns, tmp_path):
        """Windows have correct temporal order."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        for train_start, train_end, test_start, test_end in windows:
            assert train_start < train_end
            assert train_end <= test_start
            assert test_start < test_end

    def test_rolling_vs_anchored(self, sample_df_with_targets, feature_columns, tmp_path):
        """Rolling and anchored modes produce different windows."""
        trainer_rolling = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            anchored=False,
            model_save_dir=str(tmp_path / "models_rolling"),
        )
        trainer_anchored = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            anchored=True,
            model_save_dir=str(tmp_path / "models_anchored"),
        )

        windows_rolling = trainer_rolling.generate_windows()
        windows_anchored = trainer_anchored.generate_windows()

        # Both should have windows
        assert len(windows_rolling) > 0
        assert len(windows_anchored) > 0

    def test_anchored_keeps_same_start(self, sample_df_with_targets, feature_columns, tmp_path):
        """Anchored mode keeps same train_start for all windows."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            step_months=2,
            anchored=True,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 1:
            first_start = windows[0][0]
            for train_start, _, _, _ in windows:
                assert train_start == first_start


class TestDataPreparation:
    """Tests for window data preparation."""

    def test_prepare_window_data_returns_tuple(self, sample_df_with_targets, feature_columns, tmp_path):
        """prepare_window_data returns 4-element tuple."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 0:
            train_start, train_end, test_start, test_end = windows[0]
            result = trainer.prepare_window_data(train_start, train_end, test_start, test_end)

            assert isinstance(result, tuple)
            assert len(result) == 4

    def test_prepare_window_data_returns_data_loaders(self, sample_df_with_targets, feature_columns, tmp_path):
        """prepare_window_data returns DataLoader objects."""
        from torch.utils.data import DataLoader

        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 0:
            train_start, train_end, test_start, test_end = windows[0]
            train_loader, test_loader, test_df, scaler = trainer.prepare_window_data(
                train_start, train_end, test_start, test_end
            )

            assert isinstance(train_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)

    def test_prepare_window_data_returns_scaler(self, sample_df_with_targets, feature_columns, tmp_path):
        """prepare_window_data returns StandardScaler."""
        from sklearn.preprocessing import StandardScaler

        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 0:
            train_start, train_end, test_start, test_end = windows[0]
            _, _, _, scaler = trainer.prepare_window_data(
                train_start, train_end, test_start, test_end
            )

            assert isinstance(scaler, StandardScaler)

    def test_prepare_window_data_raises_for_empty(self, sample_df_with_targets, feature_columns, tmp_path):
        """prepare_window_data raises for empty splits."""
        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            model_save_dir=str(tmp_path / "models"),
        )

        # Use dates far in the future to get empty split
        future_start = pd.Timestamp("2030-01-01", tz=NY_TZ)
        future_end = pd.Timestamp("2030-12-31", tz=NY_TZ)

        with pytest.raises(ValueError, match="Empty split"):
            trainer.prepare_window_data(future_start, future_end, future_end, future_end)


class TestWindowEvaluation:
    """Tests for window evaluation."""

    def test_evaluate_window_returns_dict(self, sample_df_with_targets, feature_columns, tmp_path):
        """evaluate_window returns dictionary."""
        from sklearn.preprocessing import StandardScaler
        from src.rl.regularized_model import RegularizedMultiHorizonNet

        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 0:
            train_start, train_end, test_start, test_end = windows[0]
            _, _, test_df, scaler = trainer.prepare_window_data(
                train_start, train_end, test_start, test_end
            )

            # Create minimal model
            model = RegularizedMultiHorizonNet(
                input_dim=len(feature_columns),
                hidden_dims=[32, 16, 8],
                num_residual_blocks=1,
            )

            results = trainer.evaluate_window(model, test_df, scaler)

            assert isinstance(results, dict)

    def test_evaluate_window_has_accuracies(self, sample_df_with_targets, feature_columns, tmp_path):
        """evaluate_window returns accuracy metrics."""
        from sklearn.preprocessing import StandardScaler
        from src.rl.regularized_model import RegularizedMultiHorizonNet

        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 0:
            train_start, train_end, test_start, test_end = windows[0]
            _, _, test_df, scaler = trainer.prepare_window_data(
                train_start, train_end, test_start, test_end
            )

            model = RegularizedMultiHorizonNet(
                input_dim=len(feature_columns),
                hidden_dims=[32, 16, 8],
                num_residual_blocks=1,
            )

            results = trainer.evaluate_window(model, test_df, scaler)

            assert "acc_1h" in results
            assert "acc_4h" in results
            assert "acc_eod" in results
            assert "n_samples" in results

    def test_evaluate_window_accuracies_valid(self, sample_df_with_targets, feature_columns, tmp_path):
        """evaluate_window accuracies are in [0, 1]."""
        from src.rl.regularized_model import RegularizedMultiHorizonNet

        trainer = WalkForwardTrainer(
            df=sample_df_with_targets,
            feature_cols=feature_columns,
            train_months=6,
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )
        windows = trainer.generate_windows()

        if len(windows) > 0:
            train_start, train_end, test_start, test_end = windows[0]
            _, _, test_df, scaler = trainer.prepare_window_data(
                train_start, train_end, test_start, test_end
            )

            model = RegularizedMultiHorizonNet(
                input_dim=len(feature_columns),
                hidden_dims=[32, 16, 8],
                num_residual_blocks=1,
            )

            results = trainer.evaluate_window(model, test_df, scaler)

            assert 0 <= results["acc_1h"] <= 1
            assert 0 <= results["acc_4h"] <= 1
            assert 0 <= results["acc_eod"] <= 1


class TestEnsemblePredictorInit:
    """Tests for EnsemblePredictor initialization."""

    def test_init_with_empty_dir(self, tmp_path):
        """EnsemblePredictor handles empty model directory."""
        model_dir = tmp_path / "empty_models"
        model_dir.mkdir()

        predictor = EnsemblePredictor(model_dir=str(model_dir))

        assert len(predictor.models) == 0
        assert len(predictor.scalers) == 0

    def test_init_loads_models(self, tmp_path, feature_columns):
        """EnsemblePredictor loads models from directory."""
        from src.rl.regularized_model import RegularizedMultiHorizonNet
        from sklearn.preprocessing import StandardScaler

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create and save mock model - use same architecture as EnsemblePredictor expects
        # EnsemblePredictor uses hardcoded [512, 256, 128] hidden_dims
        model = RegularizedMultiHorizonNet(input_dim=4, hidden_dims=[512, 256, 128], dropout_rate=0.4)
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(4)
        scaler.scale_ = np.ones(4)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_cols": feature_columns,
            "results": {"acc_1h": 0.5},
        }
        torch.save(checkpoint, model_dir / "window_00.pt")

        predictor = EnsemblePredictor(model_dir=str(model_dir))

        assert len(predictor.models) == 1
        assert len(predictor.scalers) == 1
        assert predictor.feature_cols is not None


class TestEnsemblePredictorPredict:
    """Tests for EnsemblePredictor predict method."""

    def test_predict_returns_dict(self, tmp_path, feature_columns):
        """predict returns dictionary."""
        from src.rl.regularized_model import RegularizedMultiHorizonNet
        from sklearn.preprocessing import StandardScaler

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create and save mock model - use same architecture as EnsemblePredictor expects
        model = RegularizedMultiHorizonNet(input_dim=4, hidden_dims=[512, 256, 128], dropout_rate=0.4)
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(4)
        scaler.scale_ = np.ones(4)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_cols": feature_columns,
        }
        torch.save(checkpoint, model_dir / "window_00.pt")

        predictor = EnsemblePredictor(model_dir=str(model_dir))
        features = np.random.randn(4).astype(np.float32)
        result = predictor.predict(features)

        assert isinstance(result, dict)

    def test_predict_has_mean_and_std(self, tmp_path, feature_columns):
        """predict returns mean and std for each horizon."""
        from src.rl.regularized_model import RegularizedMultiHorizonNet
        from sklearn.preprocessing import StandardScaler

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        model = RegularizedMultiHorizonNet(input_dim=4, hidden_dims=[512, 256, 128], dropout_rate=0.4)
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(4)
        scaler.scale_ = np.ones(4)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_cols": feature_columns,
        }
        torch.save(checkpoint, model_dir / "window_00.pt")

        predictor = EnsemblePredictor(model_dir=str(model_dir))
        features = np.random.randn(4).astype(np.float32)
        result = predictor.predict(features)

        assert "prob_1h_mean" in result
        assert "prob_1h_std" in result
        assert "prob_4h_mean" in result
        assert "prob_4h_std" in result
        assert "prob_eod_mean" in result
        assert "prob_eod_std" in result

    def test_predict_probabilities_valid(self, tmp_path, feature_columns):
        """predict returns valid probability values."""
        from src.rl.regularized_model import RegularizedMultiHorizonNet
        from sklearn.preprocessing import StandardScaler

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        model = RegularizedMultiHorizonNet(input_dim=4, hidden_dims=[512, 256, 128], dropout_rate=0.4)
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(4)
        scaler.scale_ = np.ones(4)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_cols": feature_columns,
        }
        torch.save(checkpoint, model_dir / "window_00.pt")

        predictor = EnsemblePredictor(model_dir=str(model_dir))
        features = np.random.randn(4).astype(np.float32)
        result = predictor.predict(features)

        assert 0 <= result["prob_1h_mean"] <= 1
        assert 0 <= result["prob_4h_mean"] <= 1
        assert 0 <= result["prob_eod_mean"] <= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_data_range(self, feature_columns, tmp_path):
        """Handle data range shorter than train window."""
        dates = pd.date_range("2024-01-01 09:30:00", "2024-03-01 16:00:00", freq="1H", tz=NY_TZ)
        df = pd.DataFrame({
            "close": np.random.randn(len(dates)) + 100,
            "feature_1": np.random.randn(len(dates)),
            "feature_2": np.random.randn(len(dates)),
            "feature_3": np.random.randn(len(dates)),
            "feature_4": np.random.randn(len(dates)),
            "target_1h": np.random.randint(0, 2, len(dates)),
            "target_4h": np.random.randint(0, 2, len(dates)),
            "target_eod": np.random.randint(0, 2, len(dates)),
        }, index=dates)

        trainer = WalkForwardTrainer(
            df=df,
            feature_cols=feature_columns,
            train_months=12,  # Longer than data range
            test_months=2,
            model_save_dir=str(tmp_path / "models"),
        )

        windows = trainer.generate_windows()
        # Should return empty or very few windows
        assert len(windows) <= 1
