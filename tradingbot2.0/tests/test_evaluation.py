"""
Unit tests for ML evaluation module.

Tests cover:
- ClassificationMetrics and TradingMetrics dataclasses
- calculate_classification_metrics function
- _simple_auc function
- TradingSimulator class
- evaluate_model_and_strategy function
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.utils.evaluation import (
    ClassificationMetrics,
    TradingMetrics,
    calculate_classification_metrics,
    _simple_auc,
    TradingSimulator,
    evaluate_model_and_strategy,
)


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic ClassificationMetrics creation."""
        cm = np.array([[50, 10], [5, 35]])
        metrics = ClassificationMetrics(
            accuracy=0.85,
            precision=0.78,
            recall=0.88,
            f1_score=0.82,
            auc_roc=0.90,
            confusion_matrix=cm,
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.78
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.82
        assert metrics.auc_roc == 0.90
        assert np.array_equal(metrics.confusion_matrix, cm)

    def test_perfect_metrics(self):
        """Test perfect classification metrics."""
        cm = np.array([[50, 0], [0, 50]])
        metrics = ClassificationMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            auc_roc=1.0,
            confusion_matrix=cm,
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_roc == 1.0


class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic TradingMetrics creation."""
        metrics = TradingMetrics(
            total_return=0.15,
            annualized_return=0.30,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=100,
            avg_trade_return=0.0015,
        )

        assert metrics.total_return == 0.15
        assert metrics.annualized_return == 0.30
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == -0.10
        assert metrics.win_rate == 0.55
        assert metrics.profit_factor == 1.8
        assert metrics.total_trades == 100
        assert metrics.avg_trade_return == 0.0015

    def test_losing_strategy_metrics(self):
        """Test metrics for a losing strategy."""
        metrics = TradingMetrics(
            total_return=-0.20,
            annualized_return=-0.35,
            sharpe_ratio=-0.5,
            max_drawdown=-0.30,
            win_rate=0.40,
            profit_factor=0.5,
            total_trades=50,
            avg_trade_return=-0.002,
        )

        assert metrics.total_return < 0
        assert metrics.sharpe_ratio < 0
        assert metrics.win_rate < 0.5
        assert metrics.profit_factor < 1.0


class TestCalculateClassificationMetrics:
    """Tests for calculate_classification_metrics function."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_all_same_predictions(self):
        """Test with all same predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        # All predicted as 1 (above threshold 0.5)
        assert metrics.accuracy == 0.5  # Half are correct

    def test_custom_threshold(self):
        """Test with custom threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.3, 0.6, 0.7, 0.9])

        # With threshold 0.5: [0, 1, 1, 1] -> accuracy = 0.75
        metrics_05 = calculate_classification_metrics(y_true, y_pred_proba, threshold=0.5)

        # With threshold 0.65: [0, 0, 1, 1] -> accuracy = 1.0
        metrics_065 = calculate_classification_metrics(y_true, y_pred_proba, threshold=0.65)

        assert metrics_065.accuracy > metrics_05.accuracy

    def test_confusion_matrix_structure(self):
        """Test confusion matrix structure."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.3, 0.7, 0.4, 0.8])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        # Confusion matrix should be 2x2
        assert metrics.confusion_matrix.shape == (2, 2)

    def test_empty_positive_class(self):
        """Test with no positive samples."""
        y_true = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        # Should handle gracefully
        assert metrics.accuracy == 1.0  # All predicted as 0, all true are 0
        assert metrics.recall == 0  # No positive samples

    def test_empty_negative_class(self):
        """Test with no negative samples."""
        y_true = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.6, 0.7, 0.8, 0.9])

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        # Should handle gracefully
        assert metrics.accuracy == 1.0
        assert metrics.recall == 1.0


class TestSimpleAuc:
    """Tests for _simple_auc function."""

    def test_all_positive_class(self):
        """Test AUC with all positive samples."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])

        auc = _simple_auc(y_true, y_pred)

        # Should return 0.5 (undefined case)
        assert auc == 0.5

    def test_all_negative_class(self):
        """Test AUC with all negative samples."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        auc = _simple_auc(y_true, y_pred)

        # Should return 0.5 (undefined case)
        assert auc == 0.5


class TestTradingSimulator:
    """Tests for TradingSimulator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        sim = TradingSimulator()

        assert sim.initial_capital == 100000.0
        assert sim.position_size == 0.02
        assert sim.commission == 5.0
        assert sim.slippage == 0.0001
        assert sim.long_threshold == 0.55
        assert sim.short_threshold == 0.45

    def test_init_custom(self):
        """Test custom initialization."""
        sim = TradingSimulator(
            initial_capital=50000.0,
            position_size=0.05,
            commission=2.0,
            slippage=0.0002,
            long_threshold=0.60,
            short_threshold=0.40,
        )

        assert sim.initial_capital == 50000.0
        assert sim.position_size == 0.05
        assert sim.commission == 2.0
        assert sim.slippage == 0.0002
        assert sim.long_threshold == 0.60
        assert sim.short_threshold == 0.40

    def test_run_backtest_basic(self):
        """Test basic backtest run."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
        )

        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        predictions = np.array([0.6, 0.6, 0.6, 0.6])  # Always long

        results = sim.run_backtest(prices, predictions)

        assert 'metrics' in results
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'final_capital' in results
        assert 'total_return_pct' in results

    def test_run_backtest_with_returns(self):
        """Test backtest with pre-calculated returns."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
        )

        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        predictions = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

        results = sim.run_backtest(prices, predictions, returns)

        assert results['final_capital'] > 10000.0  # Should profit from long positions

    def test_run_backtest_no_trades(self):
        """Test backtest with no trades (all predictions neutral)."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            long_threshold=0.55,
            short_threshold=0.45,
        )

        prices = np.array([100.0, 101.0, 102.0, 103.0])
        predictions = np.array([0.50, 0.50, 0.50])  # All neutral

        results = sim.run_backtest(prices, predictions)

        # No trades should be taken
        assert results['metrics'].total_trades == 0

    def test_run_backtest_long_positions(self):
        """Test backtest with all long positions."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
            long_threshold=0.55,
            short_threshold=0.45,
        )

        prices = np.array([100.0, 102.0, 104.0, 106.0])  # Uptrend
        predictions = np.array([0.7, 0.7, 0.7])  # All long signals

        results = sim.run_backtest(prices, predictions)

        assert results['final_capital'] > 10000.0  # Should profit

    def test_run_backtest_short_positions(self):
        """Test backtest with all short positions."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
            long_threshold=0.55,
            short_threshold=0.45,
        )

        prices = np.array([100.0, 98.0, 96.0, 94.0])  # Downtrend
        predictions = np.array([0.3, 0.3, 0.3])  # All short signals

        results = sim.run_backtest(prices, predictions)

        assert results['final_capital'] > 10000.0  # Should profit from shorts

    def test_run_backtest_commission_impact(self):
        """Test that commissions reduce returns."""
        sim_no_commission = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
        )

        sim_with_commission = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=10.0,
            slippage=0.0,
        )

        prices = np.array([100.0, 101.0, 102.0, 103.0])
        predictions = np.array([0.7, 0.3, 0.7])  # Switch positions

        results_no_comm = sim_no_commission.run_backtest(prices, predictions)
        results_with_comm = sim_with_commission.run_backtest(prices, predictions)

        assert results_with_comm['final_capital'] < results_no_comm['final_capital']


class TestTradingMetricsCalculation:
    """Tests for trading metrics calculation."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio calculation for profitable strategy."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
        )

        # Consistent positive returns
        prices = np.linspace(100, 110, 20)
        predictions = np.ones(19) * 0.7  # All long

        results = sim.run_backtest(prices, predictions)

        # Positive Sharpe for consistent gains
        assert results['metrics'].sharpe_ratio > 0

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
        )

        # Price goes up then down
        prices = np.array([100.0, 110.0, 105.0, 100.0, 95.0])
        predictions = np.array([0.7, 0.7, 0.7, 0.7])  # All long

        results = sim.run_backtest(prices, predictions)

        # Max drawdown should be negative
        assert results['metrics'].max_drawdown <= 0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            position_size=0.1,
            commission=0.0,
            slippage=0.0,
        )

        # Mix of up and down days
        prices = np.array([100.0, 101.0, 100.0, 101.0, 100.0])
        predictions = np.array([0.7, 0.7, 0.7, 0.7])  # All long

        results = sim.run_backtest(prices, predictions)

        # Win rate should be between 0 and 1
        assert 0 <= results['metrics'].win_rate <= 1


class TestEvaluateModelAndStrategy:
    """Tests for evaluate_model_and_strategy function."""

    def test_comprehensive_evaluation(self):
        """Test comprehensive model evaluation."""
        # Create sample data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.3, 0.4, 0.7, 0.8, 0.2, 0.9, 0.1, 0.6])
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107])
        returns = np.diff(prices) / prices[:-1]

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        # Check structure
        assert 'classification' in results
        assert 'trading' in results
        assert 'comparison' in results
        assert 'equity_curve' in results
        assert 'trades' in results

    def test_classification_metrics_structure(self):
        """Test classification metrics in results."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.2, 0.4, 0.6, 0.8])
        prices = np.array([100, 101, 102, 103])
        returns = np.array([0.01, 0.01, 0.01])

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        cls = results['classification']
        assert 'accuracy' in cls
        assert 'precision' in cls
        assert 'recall' in cls
        assert 'f1_score' in cls
        assert 'auc_roc' in cls
        assert 'confusion_matrix' in cls

    def test_trading_metrics_structure(self):
        """Test trading metrics in results."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.2, 0.4, 0.6, 0.8])
        prices = np.array([100, 101, 102, 103])
        returns = np.array([0.01, 0.01, 0.01])

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        trading = results['trading']
        assert 'total_return' in trading
        assert 'annualized_return' in trading
        assert 'sharpe_ratio' in trading
        assert 'max_drawdown' in trading
        assert 'win_rate' in trading
        assert 'profit_factor' in trading
        assert 'total_trades' in trading
        assert 'avg_trade_return' in trading

    def test_comparison_metrics_structure(self):
        """Test comparison metrics in results."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.2, 0.4, 0.6, 0.8])
        prices = np.array([100, 101, 102, 103])
        returns = np.array([0.01, 0.01, 0.01])

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        comp = results['comparison']
        assert 'strategy_return' in comp
        assert 'buy_hold_return' in comp
        assert 'alpha' in comp

    def test_alpha_calculation(self):
        """Test alpha (excess return) calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.2, 0.3, 0.7, 0.8, 0.1, 0.9])
        prices = np.array([100, 101, 102, 103, 104, 105])
        returns = np.diff(prices) / prices[:-1]

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        # Alpha = strategy_return - buy_hold_return
        expected_alpha = results['comparison']['strategy_return'] - results['comparison']['buy_hold_return']
        assert abs(results['comparison']['alpha'] - expected_alpha) < 1e-10


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_day_backtest(self):
        """Test backtest with single day of data."""
        sim = TradingSimulator()

        prices = np.array([100.0, 101.0])
        predictions = np.array([0.7])

        results = sim.run_backtest(prices, predictions)

        assert 'metrics' in results

    def test_identical_prices(self):
        """Test backtest with no price movement."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            commission=0.0,
            slippage=0.0,
        )

        prices = np.array([100.0, 100.0, 100.0, 100.0])
        predictions = np.array([0.7, 0.7, 0.7])

        results = sim.run_backtest(prices, predictions)

        # No price movement means no profit/loss from positions
        # Only change would be from commissions
        assert results['final_capital'] == 10000.0

    def test_extreme_predictions(self):
        """Test with extreme prediction values."""
        sim = TradingSimulator()

        prices = np.array([100.0, 101.0, 102.0, 103.0])
        predictions = np.array([0.0, 1.0, 0.0])  # Extreme values

        results = sim.run_backtest(prices, predictions)

        # Should handle extreme predictions gracefully
        assert 'metrics' in results

    def test_large_dataset(self):
        """Test with larger dataset."""
        sim = TradingSimulator(
            initial_capital=10000.0,
            commission=0.0,
            slippage=0.0,
        )

        np.random.seed(42)
        n_days = 1000
        # Random walk prices
        prices = 100.0 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
        predictions = np.random.random(n_days - 1)

        results = sim.run_backtest(prices, predictions)

        # Equity curve length depends on internal processing
        assert len(results['equity_curve']) > 0
        assert results['metrics'].total_trades >= 0


class TestMetricsBounds:
    """Tests for metric value bounds."""

    def test_accuracy_bounds(self):
        """Test that accuracy is between 0 and 1."""
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.random(100)

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert 0 <= metrics.accuracy <= 1

    def test_precision_bounds(self):
        """Test that precision is between 0 and 1."""
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.random(100)

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert 0 <= metrics.precision <= 1

    def test_recall_bounds(self):
        """Test that recall is between 0 and 1."""
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.random(100)

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert 0 <= metrics.recall <= 1

    def test_f1_bounds(self):
        """Test that F1 score is between 0 and 1."""
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.random(100)

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert 0 <= metrics.f1_score <= 1

    def test_auc_bounds(self):
        """Test that AUC is between 0 and 1."""
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.random(100)

        metrics = calculate_classification_metrics(y_true, y_pred_proba)

        assert 0 <= metrics.auc_roc <= 1

    def test_win_rate_bounds(self):
        """Test that win rate is between 0 and 1."""
        sim = TradingSimulator()

        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        predictions = np.random.random(99)

        results = sim.run_backtest(prices, predictions)

        assert 0 <= results['metrics'].win_rate <= 1
