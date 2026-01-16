"""
Evaluation and Backtesting Module for Trading ML Model.

Provides:
1. Classification metrics (accuracy, precision, recall, F1, AUC)
2. Trading-specific metrics (Sharpe ratio, max drawdown, win rate)
3. Strategy simulation and backtesting
4. Visualization of results

IMPORTANT RISK DISCLAIMER:
- Past performance does not guarantee future results
- Simulated trading does not account for real-world factors like slippage, liquidity
- Always paper trade before risking real capital
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray


@dataclass
class TradingMetrics:
    """Container for trading strategy metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> ClassificationMetrics:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        ClassificationMetrics object
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    except (ImportError, ValueError):
        # Approximate AUC
        auc_roc = _simple_auc(y_true, y_pred_proba)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_roc=auc_roc,
        confusion_matrix=confusion_matrix
    )


def _simple_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple AUC calculation without sklearn."""
    sorted_idx = np.argsort(y_pred)[::-1]
    sorted_true = y_true[sorted_idx]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_cum = np.cumsum(sorted_true) / n_pos
    fpr_cum = np.cumsum(1 - sorted_true) / n_neg
    return np.trapz(tpr_cum, fpr_cum)


class TradingSimulator:
    """
    Simulate a trading strategy based on model predictions.

    Strategy:
    - If P(up) > threshold: Go LONG
    - If P(up) < (1 - threshold): Go SHORT
    - Otherwise: Stay FLAT

    This is a simplified simulation - real trading has many more factors.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size: float = 0.02,  # 2% of capital per trade
        commission: float = 5.0,  # Fixed commission per trade
        slippage: float = 0.0001,  # 1 bps slippage
        long_threshold: float = 0.55,
        short_threshold: float = 0.45
    ):
        """
        Initialize trading simulator.

        Args:
            initial_capital: Starting capital
            position_size: Fraction of capital to risk per trade
            commission: Commission per trade
            slippage: Slippage as fraction of price
            long_threshold: Probability threshold to go long
            short_threshold: Probability threshold to go short
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def run_backtest(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run backtest simulation.

        Args:
            prices: Close prices for each day
            predictions: Model prediction probabilities
            returns: Pre-calculated returns (optional)

        Returns:
            Dictionary with backtest results
        """
        if returns is None:
            returns = np.diff(prices) / prices[:-1]
            # Align with predictions (predictions are for next day's return)
            returns = returns[:-1]  # Remove last return
            prices = prices[:-1]
            predictions = predictions[:-1]

        n_days = len(predictions)
        capital = self.initial_capital
        position = 0  # 1=long, -1=short, 0=flat
        trades = []
        equity_curve = [capital]

        for i in range(n_days):
            daily_return = returns[i] if i < len(returns) else 0

            # Determine signal
            if predictions[i] >= self.long_threshold:
                new_position = 1
            elif predictions[i] <= self.short_threshold:
                new_position = -1
            else:
                new_position = 0

            # Execute trade if position changed
            if new_position != position:
                # Close existing position
                if position != 0:
                    # Closing trade
                    pass  # Already accounted for in daily P&L

                # Open new position
                position = new_position
                capital -= self.commission

            # Calculate P&L for the day
            if position != 0:
                trade_capital = capital * self.position_size
                trade_pnl = trade_capital * position * daily_return
                # Apply slippage
                trade_pnl -= abs(trade_pnl) * self.slippage
                capital += trade_pnl

                trades.append({
                    'day': i,
                    'position': position,
                    'return': daily_return,
                    'pnl': trade_pnl,
                    'capital': capital
                })

            equity_curve.append(capital)

        # Calculate metrics
        equity_curve = np.array(equity_curve)
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]

        metrics = self._calculate_trading_metrics(
            equity_curve,
            daily_returns,
            trades
        )

        return {
            'metrics': metrics,
            'equity_curve': equity_curve.tolist(),
            'trades': trades,
            'final_capital': capital,
            'total_return_pct': (capital - self.initial_capital) / self.initial_capital * 100
        }

    def _calculate_trading_metrics(
        self,
        equity_curve: np.ndarray,
        daily_returns: np.ndarray,
        trades: List[Dict]
    ) -> TradingMetrics:
        """Calculate trading performance metrics."""

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Annualized return (assuming 252 trading days)
        n_days = len(daily_returns)
        annualized_return = ((1 + total_return) ** (252 / max(n_days, 1))) - 1

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdowns)

        # Win rate and profit factor
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]

            win_rate = len(winning_trades) / len(trades)

            total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1

            profit_factor = total_profit / total_loss if total_loss > 0 else total_profit

            avg_trade_return = np.mean([t['pnl'] for t in trades])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0

        return TradingMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return
        )


def evaluate_model_and_strategy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray
) -> Dict:
    """
    Comprehensive evaluation of model and trading strategy.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        prices: Close prices
        returns: Actual next-day returns

    Returns:
        Dictionary with all evaluation results
    """
    # Classification metrics
    class_metrics = calculate_classification_metrics(y_true, y_pred_proba)

    # Trading simulation
    simulator = TradingSimulator()
    backtest_results = simulator.run_backtest(prices, y_pred_proba, returns)

    # Buy and hold comparison
    buy_hold_return = (prices[-1] - prices[0]) / prices[0]

    return {
        'classification': {
            'accuracy': class_metrics.accuracy,
            'precision': class_metrics.precision,
            'recall': class_metrics.recall,
            'f1_score': class_metrics.f1_score,
            'auc_roc': class_metrics.auc_roc,
            'confusion_matrix': class_metrics.confusion_matrix.tolist()
        },
        'trading': {
            'total_return': backtest_results['metrics'].total_return,
            'annualized_return': backtest_results['metrics'].annualized_return,
            'sharpe_ratio': backtest_results['metrics'].sharpe_ratio,
            'max_drawdown': backtest_results['metrics'].max_drawdown,
            'win_rate': backtest_results['metrics'].win_rate,
            'profit_factor': backtest_results['metrics'].profit_factor,
            'total_trades': backtest_results['metrics'].total_trades,
            'avg_trade_return': backtest_results['metrics'].avg_trade_return
        },
        'comparison': {
            'strategy_return': backtest_results['metrics'].total_return,
            'buy_hold_return': buy_hold_return,
            'alpha': backtest_results['metrics'].total_return - buy_hold_return
        },
        'equity_curve': backtest_results['equity_curve'],
        'trades': backtest_results['trades']
    }


def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""

    print("\n" + "="*70)
    print("MODEL & STRATEGY EVALUATION REPORT")
    print("="*70)

    print("\n📊 CLASSIFICATION METRICS")
    print("-"*40)
    cls = results['classification']
    print(f"  Accuracy:   {cls['accuracy']:.4f}")
    print(f"  Precision:  {cls['precision']:.4f}")
    print(f"  Recall:     {cls['recall']:.4f}")
    print(f"  F1 Score:   {cls['f1_score']:.4f}")
    print(f"  AUC-ROC:    {cls['auc_roc']:.4f}")

    print("\n📈 TRADING METRICS")
    print("-"*40)
    trading = results['trading']
    print(f"  Total Return:      {trading['total_return']*100:.2f}%")
    print(f"  Annualized Return: {trading['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio:      {trading['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:      {trading['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:          {trading['win_rate']*100:.1f}%")
    print(f"  Profit Factor:     {trading['profit_factor']:.3f}")
    print(f"  Total Trades:      {trading['total_trades']}")

    print("\n🔄 STRATEGY vs BUY & HOLD")
    print("-"*40)
    comp = results['comparison']
    print(f"  Strategy Return:   {comp['strategy_return']*100:.2f}%")
    print(f"  Buy & Hold Return: {comp['buy_hold_return']*100:.2f}%")
    print(f"  Alpha (excess):    {comp['alpha']*100:.2f}%")

    print("\n" + "="*70)


def plot_results(results: Dict, save_path: Optional[str] = None):
    """
    Plot evaluation results.

    Args:
        results: Evaluation results dictionary
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Equity Curve
        ax1 = axes[0, 0]
        equity = results['equity_curve']
        ax1.plot(equity, 'b-', linewidth=1.5, label='Strategy Equity')
        ax1.axhline(y=equity[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = axes[0, 1]
        equity_arr = np.array(equity)
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax * 100
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown, 'r-', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Trading Day')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Trade Distribution
        ax3 = axes[1, 0]
        if results['trades']:
            pnls = [t['pnl'] for t in results['trades']]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.6)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_title('Trade P&L Distribution')
            ax3.set_xlabel('Trade #')
            ax3.set_ylabel('P&L ($)')
            ax3.grid(True, alpha=0.3)

        # 4. Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics_text = f"""
        CLASSIFICATION METRICS
        ─────────────────────
        Accuracy:    {results['classification']['accuracy']:.4f}
        Precision:   {results['classification']['precision']:.4f}
        Recall:      {results['classification']['recall']:.4f}
        F1 Score:    {results['classification']['f1_score']:.4f}
        AUC-ROC:     {results['classification']['auc_roc']:.4f}

        TRADING METRICS
        ─────────────────────
        Total Return:   {results['trading']['total_return']*100:.2f}%
        Sharpe Ratio:   {results['trading']['sharpe_ratio']:.3f}
        Max Drawdown:   {results['trading']['max_drawdown']*100:.2f}%
        Win Rate:       {results['trading']['win_rate']*100:.1f}%
        Total Trades:   {results['trading']['total_trades']}
        """
        ax4.text(0.1, 0.5, metrics_text, family='monospace', fontsize=10,
                verticalalignment='center', transform=ax4.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib not available. Skipping plots.")


if __name__ == "__main__":
    # Test evaluation module
    print("Testing Evaluation Module")
    print("="*60)

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 500

    # Simulated prices (random walk with drift)
    returns = np.random.normal(0.0005, 0.015, n_samples)
    prices = 5000 * np.cumprod(1 + returns)

    # Simulated predictions (slightly better than random)
    noise = np.random.normal(0, 0.15, n_samples)
    y_true = (returns > 0).astype(int)
    y_pred_proba = np.clip(0.5 + 0.1 * (returns / 0.015) + noise, 0, 1)

    # Run evaluation
    results = evaluate_model_and_strategy(y_true, y_pred_proba, prices, returns)

    # Print report
    print_evaluation_report(results)

    # Try to plot
    try:
        plot_results(results)
    except Exception as e:
        print(f"Could not plot results: {e}")
