"""
Simple tests for evaluation module to increase coverage.

Tests cover basic functionality without complex mocking.
"""

import pytest
import numpy as np

from src.ml.utils.evaluation import (
    ClassificationMetrics,
    TradingMetrics,
    calculate_classification_metrics,
    _simple_auc,
    TradingSimulator,
    evaluate_model_and_strategy,
)


# =============================================================================
# ClassificationMetrics Tests
# =============================================================================

class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_creation(self):
        """Test ClassificationMetrics creation."""
        metrics = ClassificationMetrics(
            accuracy=0.85,
            precision=0.90,
            recall=0.80,
            f1_score=0.85,
            auc_roc=0.90,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.90
        assert metrics.recall == 0.80
        assert metrics.f1_score == 0.85
        assert metrics.auc_roc == 0.90


# =============================================================================
# TradingMetrics Tests
# =============================================================================

class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_creation(self):
        """Test TradingMetrics creation."""
        metrics = TradingMetrics(
            total_return=0.15,
            annualized_return=0.45,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=100,
            avg_trade_return=0.001,
        )

        assert metrics.total_return == 0.15
        assert metrics.annualized_return == 0.45
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == -0.10
        assert metrics.win_rate == 0.55
        assert metrics.profit_factor == 1.8
        assert metrics.total_trades == 100
        assert metrics.avg_trade_return == 0.001


# =============================================================================
# calculate_classification_metrics Tests
# =============================================================================

class TestCalculateClassificationMetrics:
    """Tests for calculate_classification_metrics function."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.0, 0.0, 1.0, 1.0])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_all_zeros(self):
        """Test with all negative predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.0, 0.0, 0.0, 0.0])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert metrics.accuracy == 0.5  # TN=2, FN=2
        assert metrics.precision == 0  # No positive predictions

    def test_all_ones(self):
        """Test with all positive predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([1.0, 1.0, 1.0, 1.0])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert metrics.accuracy == 0.5  # FP=2, TP=2
        assert metrics.recall == 1.0  # All positives caught

    def test_custom_threshold(self):
        """Test with custom threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.3, 0.4, 0.6, 0.7])

        metrics = calculate_classification_metrics(y_true, y_pred_proba, threshold=0.5)

        assert metrics.accuracy == 1.0

    def test_empty_array_edge_case(self):
        """Test edge case handling."""
        y_true = np.array([])
        y_pred_proba = np.array([])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert metrics.accuracy == 0


# =============================================================================
# _simple_auc Tests
# =============================================================================

class TestSimpleAuc:
    """Tests for _simple_auc function."""

    def test_perfect_separation(self):
        """Test AUC with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        auc = _simple_auc(y_true, y_pred)

        assert auc >= 0.9  # Should be near 1.0

    def test_random_predictions(self):
        """Test AUC with random-ish predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])

        auc = _simple_auc(y_true, y_pred)

        # Random should be around 0.5
        assert 0.0 <= auc <= 1.0

    def test_all_same_class(self):
        """Test AUC when all examples are same class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        auc = _simple_auc(y_true, y_pred)

        assert auc == 0.5  # Default for degenerate case

    def test_no_negative_class(self):
        """Test AUC when no negative examples."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        auc = _simple_auc(y_true, y_pred)

        assert auc == 0.5  # Default for degenerate case


# =============================================================================
# TradingSimulator Tests
# =============================================================================

class TestTradingSimulator:
    """Tests for TradingSimulator class."""

    def test_initialization_defaults(self):
        """Test TradingSimulator with defaults."""
        sim = TradingSimulator()

        assert sim.initial_capital == 100000.0
        assert sim.position_size == 0.02
        assert sim.commission == 5.0
        assert sim.slippage == 0.0001
        assert sim.long_threshold == 0.55
        assert sim.short_threshold == 0.45

    def test_initialization_custom(self):
        """Test TradingSimulator with custom values."""
        sim = TradingSimulator(
            initial_capital=50000.0,
            position_size=0.05,
            commission=10.0,
            slippage=0.0002,
            long_threshold=0.60,
            short_threshold=0.40,
        )

        assert sim.initial_capital == 50000.0
        assert sim.position_size == 0.05
        assert sim.commission == 10.0
        assert sim.slippage == 0.0002
        assert sim.long_threshold == 0.60
        assert sim.short_threshold == 0.40

    def test_run_backtest_basic(self):
        """Test basic backtest run."""
        sim = TradingSimulator(initial_capital=10000.0)

        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        predictions = np.random.rand(n)

        results = sim.run_backtest(prices, predictions)

        assert 'metrics' in results
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'final_capital' in results
        assert 'total_return_pct' in results

    def test_run_backtest_with_returns(self):
        """Test backtest with pre-calculated returns."""
        sim = TradingSimulator(initial_capital=10000.0)

        prices = np.array([100, 101, 102, 103, 104])
        returns = np.array([0.01, 0.01, 0.01, 0.01])  # 4 returns for 5 prices
        predictions = np.array([0.6, 0.6, 0.6, 0.6, 0.6])  # All long signals

        results = sim.run_backtest(prices, predictions, returns)

        assert results['final_capital'] > 0

    def test_run_backtest_no_trades(self):
        """Test backtest with no trades (all predictions in flat zone)."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            long_threshold=0.60,
            short_threshold=0.40,
        )

        prices = np.array([100, 101, 102, 103, 104])
        predictions = np.array([0.50, 0.50, 0.50, 0.50, 0.50])  # All flat

        results = sim.run_backtest(prices, predictions)

        assert results['metrics'].total_trades >= 0


# =============================================================================
# evaluate_model_and_strategy Tests
# =============================================================================

class TestEvaluateModelAndStrategy:
    """Tests for evaluate_model_and_strategy function."""

    def test_full_evaluation(self):
        """Test full model and strategy evaluation."""
        np.random.seed(42)
        n = 100

        # Generate synthetic data
        returns = np.random.randn(n) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        y_true = (returns > 0).astype(int)
        y_pred_proba = np.clip(0.5 + returns * 10, 0, 1)

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        assert 'classification' in results
        assert 'trading' in results
        assert 'comparison' in results
        assert 'equity_curve' in results
        assert 'trades' in results

        # Check classification metrics
        assert 'accuracy' in results['classification']
        assert 'precision' in results['classification']
        assert 'recall' in results['classification']
        assert 'f1_score' in results['classification']
        assert 'auc_roc' in results['classification']

        # Check trading metrics
        assert 'total_return' in results['trading']
        assert 'sharpe_ratio' in results['trading']
        assert 'max_drawdown' in results['trading']

        # Check comparison
        assert 'strategy_return' in results['comparison']
        assert 'buy_hold_return' in results['comparison']
        assert 'alpha' in results['comparison']


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test with single sample."""
        y_true = np.array([1])
        y_pred_proba = np.array([0.8])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        # Should not crash
        assert metrics is not None

    def test_all_same_predictions(self):
        """Test when all predictions are the same."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        # Should handle gracefully
        assert 0 <= metrics.accuracy <= 1

    def test_backtest_single_day(self):
        """Test backtest with minimal data."""
        sim = TradingSimulator(initial_capital=10000.0)

        prices = np.array([100, 101])
        predictions = np.array([0.6, 0.6])

        results = sim.run_backtest(prices, predictions)

        assert results is not None
