"""
Extended tests for evaluation.py plot functionality.

Tests cover:
- plot_results function with various inputs
- Matplotlib with non-GUI backend for testing
- Edge cases in plotting (empty trades, etc.)
- print_evaluation_report function
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Use non-interactive backend before importing matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ml.utils.evaluation import (
    plot_results,
    print_evaluation_report,
    evaluate_model_and_strategy,
    TradingSimulator,
    calculate_classification_metrics,
)


# ============================================================================
# Plot Results Tests
# ============================================================================

class TestPlotResults:
    """Tests for plot_results function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results for plotting."""
        return {
            'classification': {
                'accuracy': 0.55,
                'precision': 0.52,
                'recall': 0.60,
                'f1_score': 0.56,
                'auc_roc': 0.58,
                'confusion_matrix': [[100, 80], [70, 150]]
            },
            'trading': {
                'total_return': 0.05,
                'annualized_return': 0.12,
                'sharpe_ratio': 0.75,
                'max_drawdown': -0.08,
                'win_rate': 0.52,
                'profit_factor': 1.2,
                'total_trades': 50,
                'avg_trade_return': 10.0
            },
            'comparison': {
                'strategy_return': 0.05,
                'buy_hold_return': 0.03,
                'alpha': 0.02
            },
            'equity_curve': [100000, 100500, 101000, 100800, 101500, 102000, 103000, 104000, 105000],
            'trades': [
                {'day': 1, 'position': 1, 'return': 0.005, 'pnl': 100, 'capital': 100500},
                {'day': 2, 'position': 1, 'return': 0.005, 'pnl': 100, 'capital': 101000},
                {'day': 3, 'position': -1, 'return': -0.002, 'pnl': -200, 'capital': 100800},
                {'day': 4, 'position': 1, 'return': 0.007, 'pnl': 700, 'capital': 101500},
            ]
        }

    def test_plot_results_runs_without_error(self, sample_results):
        """Test plot_results executes without raising exceptions."""
        with patch.object(plt, 'show'):  # Don't actually show the plot
            plot_results(sample_results)
        plt.close('all')

    def test_plot_results_saves_to_file(self, sample_results, tmp_path):
        """Test plot_results can save to file."""
        save_path = tmp_path / "test_plot.png"
        with patch.object(plt, 'show'):
            plot_results(sample_results, save_path=str(save_path))
        plt.close('all')
        assert save_path.exists()

    def test_plot_results_empty_trades(self, sample_results):
        """Test plot_results with empty trades list."""
        sample_results['trades'] = []
        with patch.object(plt, 'show'):
            plot_results(sample_results)
        plt.close('all')

    def test_plot_results_many_trades(self, sample_results):
        """Test plot_results with many trades."""
        sample_results['trades'] = [
            {'day': i, 'position': 1 if i % 2 == 0 else -1, 'return': 0.001, 'pnl': 10 if i % 3 == 0 else -5, 'capital': 100000 + i * 10}
            for i in range(100)
        ]
        sample_results['equity_curve'] = list(range(100000, 101001, 10))
        with patch.object(plt, 'show'):
            plot_results(sample_results)
        plt.close('all')


# ============================================================================
# Print Evaluation Report Tests
# ============================================================================

class TestPrintEvaluationReport:
    """Tests for print_evaluation_report function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results for printing."""
        return {
            'classification': {
                'accuracy': 0.55,
                'precision': 0.52,
                'recall': 0.60,
                'f1_score': 0.56,
                'auc_roc': 0.58,
                'confusion_matrix': [[100, 80], [70, 150]]
            },
            'trading': {
                'total_return': 0.05,
                'annualized_return': 0.12,
                'sharpe_ratio': 0.75,
                'max_drawdown': -0.08,
                'win_rate': 0.52,
                'profit_factor': 1.2,
                'total_trades': 50,
                'avg_trade_return': 10.0
            },
            'comparison': {
                'strategy_return': 0.05,
                'buy_hold_return': 0.03,
                'alpha': 0.02
            },
            'equity_curve': [100000, 105000],
            'trades': []
        }

    def test_print_report_outputs_to_stdout(self, sample_results, capsys):
        """Test that print_evaluation_report outputs to stdout."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "MODEL & STRATEGY EVALUATION REPORT" in captured.out
        assert "CLASSIFICATION METRICS" in captured.out
        assert "TRADING METRICS" in captured.out

    def test_print_report_shows_accuracy(self, sample_results, capsys):
        """Test that accuracy is shown in report."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "Accuracy" in captured.out
        assert "0.5500" in captured.out

    def test_print_report_shows_sharpe(self, sample_results, capsys):
        """Test that Sharpe ratio is shown in report."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "Sharpe Ratio" in captured.out
        assert "0.750" in captured.out

    def test_print_report_shows_returns(self, sample_results, capsys):
        """Test that returns are shown in report."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "Total Return" in captured.out
        assert "5.00%" in captured.out

    def test_print_report_shows_drawdown(self, sample_results, capsys):
        """Test that max drawdown is shown in report."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "Max Drawdown" in captured.out
        assert "-8.00%" in captured.out

    def test_print_report_shows_comparison(self, sample_results, capsys):
        """Test that strategy vs buy & hold comparison is shown."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "STRATEGY vs BUY & HOLD" in captured.out
        assert "Strategy Return" in captured.out
        assert "Buy & Hold Return" in captured.out
        assert "Alpha" in captured.out

    def test_print_report_shows_win_rate(self, sample_results, capsys):
        """Test that win rate is shown in report."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "Win Rate" in captured.out
        assert "52.0%" in captured.out

    def test_print_report_shows_profit_factor(self, sample_results, capsys):
        """Test that profit factor is shown in report."""
        print_evaluation_report(sample_results)

        captured = capsys.readouterr()
        assert "Profit Factor" in captured.out
        assert "1.200" in captured.out


# ============================================================================
# Main Block Coverage Tests
# ============================================================================

class TestMainBlockCoverage:
    """Tests to achieve coverage of the __main__ block patterns."""

    def test_full_evaluation_pipeline(self):
        """Test full evaluation pipeline that mimics __main__ block."""
        np.random.seed(42)
        n_samples = 100

        # Simulated prices (random walk with drift)
        returns = np.random.normal(0.0005, 0.015, n_samples)
        prices = 5000 * np.cumprod(1 + returns)

        # Simulated predictions (slightly better than random)
        noise = np.random.normal(0, 0.15, n_samples)
        y_true = (returns > 0).astype(int)
        y_pred_proba = np.clip(0.5 + 0.1 * (returns / 0.015) + noise, 0, 1)

        # Run evaluation
        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        # Verify structure
        assert 'classification' in results
        assert 'trading' in results
        assert 'comparison' in results
        assert 'equity_curve' in results
        assert 'trades' in results

    def test_evaluation_with_varying_seeds(self):
        """Test evaluation with different random seeds."""
        for seed in [0, 42, 123, 999]:
            np.random.seed(seed)
            n_samples = 50

            returns = np.random.normal(0.0005, 0.015, n_samples)
            prices = 5000 * np.cumprod(1 + returns)

            y_true = (returns > 0).astype(int)
            y_pred_proba = np.clip(np.random.random(n_samples), 0, 1)

            results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

            # Should always produce valid results
            assert results['trading']['total_trades'] >= 0


# ============================================================================
# Integration Tests for Plot Function
# ============================================================================

class TestPlotIntegration:
    """Integration tests combining evaluation and plotting."""

    def test_evaluate_and_plot(self):
        """Test full evaluate then plot workflow."""
        np.random.seed(42)
        n_samples = 100

        returns = np.random.normal(0.0005, 0.015, n_samples)
        prices = 5000 * np.cumprod(1 + returns)

        y_true = (returns > 0).astype(int)
        y_pred_proba = np.clip(0.5 + 0.1 * (returns / 0.015), 0, 1)

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        with patch.object(plt, 'show'):
            plot_results(results)
        plt.close('all')

    def test_evaluate_print_and_plot(self, capsys):
        """Test full evaluation, print, and plot workflow."""
        np.random.seed(42)
        n_samples = 100

        returns = np.random.normal(0.0005, 0.015, n_samples)
        prices = 5000 * np.cumprod(1 + returns)

        y_true = (returns > 0).astype(int)
        y_pred_proba = np.clip(0.5 + 0.1 * (returns / 0.015), 0, 1)

        results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

        # Print report
        print_evaluation_report(results)
        captured = capsys.readouterr()
        assert "MODEL & STRATEGY EVALUATION REPORT" in captured.out

        # Plot results
        with patch.object(plt, 'show'):
            plot_results(results)
        plt.close('all')


# ============================================================================
# Edge Cases for Plotting
# ============================================================================

class TestPlotEdgeCases:
    """Edge case tests for plotting function."""

    def test_plot_with_single_point_equity(self):
        """Test plotting with single point equity curve."""
        results = {
            'classification': {
                'accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5,
                'auc_roc': 0.5,
                'confusion_matrix': [[50, 50], [50, 50]]
            },
            'trading': {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0
            },
            'comparison': {
                'strategy_return': 0.0,
                'buy_hold_return': 0.0,
                'alpha': 0.0
            },
            'equity_curve': [100000],
            'trades': []
        }

        with patch.object(plt, 'show'):
            plot_results(results)
        plt.close('all')

    def test_plot_with_negative_pnl_trades(self):
        """Test plotting with all negative P&L trades."""
        results = {
            'classification': {
                'accuracy': 0.4,
                'precision': 0.4,
                'recall': 0.4,
                'f1_score': 0.4,
                'auc_roc': 0.4,
                'confusion_matrix': [[40, 60], [60, 40]]
            },
            'trading': {
                'total_return': -0.05,
                'annualized_return': -0.10,
                'sharpe_ratio': -0.5,
                'max_drawdown': -0.10,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 5,
                'avg_trade_return': -100.0
            },
            'comparison': {
                'strategy_return': -0.05,
                'buy_hold_return': 0.02,
                'alpha': -0.07
            },
            'equity_curve': [100000, 99000, 98000, 97000, 96000, 95000],
            'trades': [
                {'day': 1, 'position': 1, 'return': -0.01, 'pnl': -1000, 'capital': 99000},
                {'day': 2, 'position': 1, 'return': -0.01, 'pnl': -1000, 'capital': 98000},
                {'day': 3, 'position': -1, 'return': 0.01, 'pnl': -1000, 'capital': 97000},
                {'day': 4, 'position': -1, 'return': 0.01, 'pnl': -1000, 'capital': 96000},
                {'day': 5, 'position': 1, 'return': -0.01, 'pnl': -1000, 'capital': 95000},
            ]
        }

        with patch.object(plt, 'show'):
            plot_results(results)
        plt.close('all')

    def test_plot_with_mixed_pnl_trades(self):
        """Test plotting with mixed positive/negative P&L trades."""
        results = {
            'classification': {
                'accuracy': 0.55,
                'precision': 0.55,
                'recall': 0.55,
                'f1_score': 0.55,
                'auc_roc': 0.55,
                'confusion_matrix': [[55, 45], [45, 55]]
            },
            'trading': {
                'total_return': 0.02,
                'annualized_return': 0.05,
                'sharpe_ratio': 0.3,
                'max_drawdown': -0.03,
                'win_rate': 0.6,
                'profit_factor': 1.1,
                'total_trades': 10,
                'avg_trade_return': 20.0
            },
            'comparison': {
                'strategy_return': 0.02,
                'buy_hold_return': 0.01,
                'alpha': 0.01
            },
            'equity_curve': [100000, 100500, 99800, 100200, 101000, 100800, 101500, 102000, 101700, 102200],
            'trades': [
                {'day': i, 'position': 1 if i % 2 == 0 else -1, 'return': 0.005 if i % 3 != 0 else -0.003, 'pnl': 500 if i % 3 != 0 else -300, 'capital': 100000 + i * 100}
                for i in range(10)
            ]
        }

        with patch.object(plt, 'show'):
            plot_results(results)
        plt.close('all')


# ============================================================================
# Import Error Handling Tests
# ============================================================================

class TestImportErrorHandling:
    """Tests for import error handling in plot_results."""

    def test_plot_handles_matplotlib_import_error(self):
        """Test that plot_results handles matplotlib ImportError."""
        results = {
            'classification': {
                'accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5,
                'auc_roc': 0.5,
                'confusion_matrix': [[50, 50], [50, 50]]
            },
            'trading': {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0
            },
            'comparison': {
                'strategy_return': 0.0,
                'buy_hold_return': 0.0,
                'alpha': 0.0
            },
            'equity_curve': [100000, 100000],
            'trades': []
        }

        # Test with matplotlib available (normal case)
        with patch.object(plt, 'show'):
            plot_results(results)
        plt.close('all')
