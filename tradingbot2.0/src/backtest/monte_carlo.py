"""
Monte Carlo Simulation for Backtest Robustness Assessment

This module provides Monte Carlo simulation capabilities for assessing
strategy robustness. By shuffling the order of trades and computing
equity curves, we can estimate confidence intervals for key metrics.

Why Monte Carlo Matters:
- A profitable backtest may be due to lucky trade sequencing
- Monte Carlo reveals the range of possible outcomes
- Confidence intervals help set realistic expectations
- Worst-case scenarios inform risk management

Key Features:
- Trade order randomization within constraints
- Bootstrap confidence intervals for all key metrics
- Parallel simulation for performance
- Integration with existing backtest infrastructure

Usage:
    from src.backtest.monte_carlo import MonteCarloSimulator

    # From a completed backtest
    trades = backtest_result.report.trade_log.get_trades()
    simulator = MonteCarloSimulator(trades, n_simulations=1000)
    result = simulator.run()

    print(f"Final Equity 95% CI: {result.final_equity_ci}")
    print(f"Max Drawdown 95% CI: {result.max_drawdown_ci}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import logging

from .trade_logger import TradeRecord
from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_win_rate,
    calculate_expectancy,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """
    Confidence interval for a metric.

    Attributes:
        lower: Lower bound of the interval
        upper: Upper bound of the interval
        median: Median value
        mean: Mean value
        std: Standard deviation
        percentile: The confidence level (e.g., 95 for 95% CI)
    """
    lower: float
    upper: float
    median: float
    mean: float
    std: float
    percentile: float = 95.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "median": round(self.median, 4),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "percentile": self.percentile,
        }

    def __str__(self) -> str:
        return f"[{self.lower:.2f}, {self.upper:.2f}] (median={self.median:.2f})"


@dataclass
class SimulationRun:
    """
    Results from a single Monte Carlo simulation run.

    Attributes:
        run_id: Unique identifier for this run
        final_equity: Final equity value
        max_drawdown_pct: Maximum drawdown percentage
        max_drawdown_dollars: Maximum drawdown in dollars
        sharpe_ratio: Sharpe ratio for this permutation
        sortino_ratio: Sortino ratio for this permutation
        profit_factor: Profit factor
        win_rate: Win rate percentage
        total_trades: Number of trades
        equity_curve: Full equity curve (optional, memory intensive)
    """
    run_id: int
    final_equity: float
    max_drawdown_pct: float
    max_drawdown_dollars: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    equity_curve: Optional[List[float]] = None


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation.

    Attributes:
        n_simulations: Number of simulations to run (default 1000)
        confidence_level: Confidence level for intervals (default 95)
        initial_capital: Starting capital (default 1000)
        random_seed: Random seed for reproducibility (optional)
        store_equity_curves: Whether to store all equity curves (memory intensive)
        n_workers: Number of parallel workers (default 1, no parallelism)
    """
    n_simulations: int = 1000
    confidence_level: float = 95.0
    initial_capital: float = 1000.0
    random_seed: Optional[int] = None
    store_equity_curves: bool = False
    n_workers: int = 1


@dataclass
class MonteCarloResult:
    """
    Complete results from Monte Carlo simulation.

    Contains confidence intervals for all key metrics and
    detailed statistics about the simulations.

    Attributes:
        config: Configuration used for simulation
        n_simulations_completed: Number of successful simulations
        original_final_equity: Final equity from original trade sequence
        original_max_drawdown: Max drawdown from original sequence
        original_sharpe: Sharpe ratio from original sequence

        final_equity_ci: Confidence interval for final equity
        max_drawdown_ci: Confidence interval for max drawdown (%)
        sharpe_ratio_ci: Confidence interval for Sharpe ratio
        sortino_ratio_ci: Confidence interval for Sortino ratio
        profit_factor_ci: Confidence interval for profit factor
        win_rate_ci: Confidence interval for win rate

        percentile_rankings: Where original results rank in simulations
        all_runs: List of all simulation runs (if stored)
        summary_stats: Summary statistics dictionary
    """
    config: MonteCarloConfig
    n_simulations_completed: int

    # Original sequence metrics
    original_final_equity: float
    original_max_drawdown: float
    original_sharpe: float
    original_sortino: float
    original_profit_factor: float
    original_win_rate: float

    # Confidence intervals
    final_equity_ci: ConfidenceInterval
    max_drawdown_ci: ConfidenceInterval
    sharpe_ratio_ci: ConfidenceInterval
    sortino_ratio_ci: ConfidenceInterval
    profit_factor_ci: ConfidenceInterval
    win_rate_ci: ConfidenceInterval

    # Percentile rankings (where does original fall in distribution?)
    percentile_rankings: Dict[str, float] = field(default_factory=dict)

    # Detailed runs (optional)
    all_runs: List[SimulationRun] = field(default_factory=list)

    # Summary statistics
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "n_simulations": self.config.n_simulations,
                "confidence_level": self.config.confidence_level,
                "initial_capital": self.config.initial_capital,
                "random_seed": self.config.random_seed,
            },
            "n_simulations_completed": self.n_simulations_completed,
            "original_metrics": {
                "final_equity": round(self.original_final_equity, 2),
                "max_drawdown_pct": round(self.original_max_drawdown, 4),
                "sharpe_ratio": round(self.original_sharpe, 3),
                "sortino_ratio": round(self.original_sortino, 3),
                "profit_factor": round(self.original_profit_factor, 3),
                "win_rate_pct": round(self.original_win_rate, 2),
            },
            "confidence_intervals": {
                "final_equity": self.final_equity_ci.to_dict(),
                "max_drawdown_pct": self.max_drawdown_ci.to_dict(),
                "sharpe_ratio": self.sharpe_ratio_ci.to_dict(),
                "sortino_ratio": self.sortino_ratio_ci.to_dict(),
                "profit_factor": self.profit_factor_ci.to_dict(),
                "win_rate_pct": self.win_rate_ci.to_dict(),
            },
            "percentile_rankings": {
                k: round(v, 2) for k, v in self.percentile_rankings.items()
            },
            "summary_stats": self.summary_stats,
        }

    def export_json(self, filepath: str) -> None:
        """Export results to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def is_robust(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.20,
        min_profit_factor: float = 1.2,
    ) -> Tuple[bool, List[str]]:
        """
        Check if the strategy is robust based on Monte Carlo results.

        A strategy is considered robust if:
        1. 95% CI lower bound for Sharpe > min_sharpe
        2. 95% CI upper bound for max drawdown < max_drawdown
        3. 95% CI lower bound for profit factor > min_profit_factor

        Args:
            min_sharpe: Minimum acceptable Sharpe ratio (lower CI bound)
            max_drawdown: Maximum acceptable drawdown (upper CI bound)
            min_profit_factor: Minimum acceptable profit factor (lower CI bound)

        Returns:
            Tuple of (is_robust, list of failure reasons)
        """
        failures = []

        if self.sharpe_ratio_ci.lower < min_sharpe:
            failures.append(
                f"Sharpe CI lower bound ({self.sharpe_ratio_ci.lower:.3f}) "
                f"< minimum ({min_sharpe})"
            )

        if self.max_drawdown_ci.upper > max_drawdown:
            failures.append(
                f"Max drawdown CI upper bound ({self.max_drawdown_ci.upper:.2%}) "
                f"> maximum ({max_drawdown:.2%})"
            )

        if self.profit_factor_ci.lower < min_profit_factor:
            failures.append(
                f"Profit factor CI lower bound ({self.profit_factor_ci.lower:.3f}) "
                f"< minimum ({min_profit_factor})"
            )

        return len(failures) == 0, failures

    def print_summary(self) -> None:
        """Print a formatted summary of results."""
        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 60)
        print(f"\nSimulations: {self.n_simulations_completed}")
        print(f"Confidence Level: {self.config.confidence_level}%")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")

        print("\n" + "-" * 60)
        print("ORIGINAL vs SIMULATED RESULTS")
        print("-" * 60)

        print(f"\n{'Metric':<20} {'Original':<15} {'95% CI':<25} {'Percentile':<10}")
        print("-" * 70)

        print(
            f"{'Final Equity':<20} "
            f"${self.original_final_equity:<14,.2f} "
            f"[${self.final_equity_ci.lower:,.2f}, ${self.final_equity_ci.upper:,.2f}] "
            f"{self.percentile_rankings.get('final_equity', 0):.1f}%"
        )

        print(
            f"{'Max Drawdown':<20} "
            f"{self.original_max_drawdown:<14.2%} "
            f"[{self.max_drawdown_ci.lower:.2%}, {self.max_drawdown_ci.upper:.2%}] "
            f"{self.percentile_rankings.get('max_drawdown', 0):.1f}%"
        )

        print(
            f"{'Sharpe Ratio':<20} "
            f"{self.original_sharpe:<14.3f} "
            f"[{self.sharpe_ratio_ci.lower:.3f}, {self.sharpe_ratio_ci.upper:.3f}] "
            f"{self.percentile_rankings.get('sharpe_ratio', 0):.1f}%"
        )

        print(
            f"{'Sortino Ratio':<20} "
            f"{self.original_sortino:<14.3f} "
            f"[{self.sortino_ratio_ci.lower:.3f}, {self.sortino_ratio_ci.upper:.3f}] "
            f"{self.percentile_rankings.get('sortino_ratio', 0):.1f}%"
        )

        print(
            f"{'Profit Factor':<20} "
            f"{self.original_profit_factor:<14.3f} "
            f"[{self.profit_factor_ci.lower:.3f}, {self.profit_factor_ci.upper:.3f}] "
            f"{self.percentile_rankings.get('profit_factor', 0):.1f}%"
        )

        print(
            f"{'Win Rate':<20} "
            f"{self.original_win_rate:<14.2f}% "
            f"[{self.win_rate_ci.lower:.2f}%, {self.win_rate_ci.upper:.2f}%] "
            f"{self.percentile_rankings.get('win_rate', 0):.1f}%"
        )

        print("\n" + "-" * 60)
        print("ROBUSTNESS CHECK")
        print("-" * 60)

        is_robust, failures = self.is_robust()
        if is_robust:
            print("\n✓ Strategy PASSES robustness checks")
        else:
            print("\n✗ Strategy FAILS robustness checks:")
            for failure in failures:
                print(f"  - {failure}")

        print("\n" + "=" * 60)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trade sequence analysis.

    This simulator shuffles the order of trades to assess how much
    of the backtest's performance is due to the specific sequence
    of trades versus the underlying edge.

    Why Trade Shuffling Works:
    - Trade P&Ls are assumed to be independent (no serial correlation)
    - Shuffling preserves the distribution of returns
    - Different orderings reveal the range of possible equity paths
    - Helps identify luck vs skill in backtest results

    Limitations:
    - Assumes trades are independent (may not hold for mean-reversion)
    - Does not account for changing market conditions
    - Commission/slippage already baked into trade P&Ls

    Usage:
        trades = backtest_result.report.trade_log.get_trades()
        simulator = MonteCarloSimulator(trades)
        result = simulator.run()
    """

    def __init__(
        self,
        trades: List[TradeRecord],
        n_simulations: int = 1000,
        initial_capital: float = 1000.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            trades: List of TradeRecord from a completed backtest
            n_simulations: Number of permutations to run (default 1000)
            initial_capital: Starting capital (default 1000)
            random_seed: Random seed for reproducibility (optional)
        """
        self.trades = trades
        self.trade_pnls = [t.net_pnl for t in trades]
        self.config = MonteCarloConfig(
            n_simulations=n_simulations,
            initial_capital=initial_capital,
            random_seed=random_seed,
        )

        if random_seed is not None:
            np.random.seed(random_seed)

        # Precompute original metrics
        self._original_metrics = self._compute_metrics_from_pnls(self.trade_pnls)

    def _compute_equity_curve(self, pnls: List[float]) -> np.ndarray:
        """
        Compute equity curve from sequence of P&Ls.

        Args:
            pnls: List of trade P&Ls in order

        Returns:
            Numpy array of equity values
        """
        equity = np.zeros(len(pnls) + 1)
        equity[0] = self.config.initial_capital
        for i, pnl in enumerate(pnls):
            equity[i + 1] = equity[i] + pnl
        return equity

    def _compute_daily_returns(self, equity: np.ndarray) -> np.ndarray:
        """
        Compute daily returns from equity curve.

        For trade-level equity (not time-based), we treat each trade
        as a "period" for return calculation.

        Args:
            equity: Equity curve array

        Returns:
            Array of period returns
        """
        if len(equity) < 2:
            return np.array([])

        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        return returns

    def _compute_metrics_from_pnls(
        self,
        pnls: List[float],
    ) -> Dict[str, float]:
        """
        Compute all metrics from a sequence of P&Ls.

        Args:
            pnls: List of trade P&Ls

        Returns:
            Dictionary of computed metrics
        """
        if not pnls:
            return {
                "final_equity": self.config.initial_capital,
                "max_drawdown_pct": 0.0,
                "max_drawdown_dollars": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
            }

        equity = self._compute_equity_curve(pnls)
        returns = self._compute_daily_returns(equity)

        # Calculate max drawdown
        max_dd_pct, max_dd_dollars, _, _, _ = calculate_max_drawdown(equity)

        # Calculate risk-adjusted returns
        sharpe = calculate_sharpe_ratio(returns) if len(returns) > 0 else 0.0
        sortino = calculate_sortino_ratio(returns) if len(returns) > 0 else 0.0

        # Calculate trade metrics
        pnl_array = np.array(pnls)
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0

        profit_factor = calculate_profit_factor(gross_profit, gross_loss)
        win_rate = calculate_win_rate(len(wins), len(pnls))

        return {
            "final_equity": float(equity[-1]),
            "max_drawdown_pct": float(max_dd_pct),
            "max_drawdown_dollars": float(max_dd_dollars),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
        }

    def _run_single_simulation(self, run_id: int) -> SimulationRun:
        """
        Run a single Monte Carlo simulation.

        Args:
            run_id: Unique identifier for this run

        Returns:
            SimulationRun with computed metrics
        """
        # Shuffle trade P&Ls
        shuffled_pnls = list(self.trade_pnls)
        np.random.shuffle(shuffled_pnls)

        # Compute metrics
        metrics = self._compute_metrics_from_pnls(shuffled_pnls)

        # Optionally store equity curve
        equity_curve = None
        if self.config.store_equity_curves:
            equity_curve = list(self._compute_equity_curve(shuffled_pnls))

        return SimulationRun(
            run_id=run_id,
            final_equity=metrics["final_equity"],
            max_drawdown_pct=metrics["max_drawdown_pct"],
            max_drawdown_dollars=metrics["max_drawdown_dollars"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            profit_factor=metrics["profit_factor"],
            win_rate=metrics["win_rate"],
            total_trades=len(self.trade_pnls),
            equity_curve=equity_curve,
        )

    def _compute_confidence_interval(
        self,
        values: np.ndarray,
        confidence_level: float = 95.0,
    ) -> ConfidenceInterval:
        """
        Compute confidence interval from simulation results.

        Args:
            values: Array of metric values from all simulations
            confidence_level: Confidence level (default 95)

        Returns:
            ConfidenceInterval object
        """
        if len(values) == 0:
            return ConfidenceInterval(
                lower=0.0,
                upper=0.0,
                median=0.0,
                mean=0.0,
                std=0.0,
                percentile=confidence_level,
            )

        alpha = (100 - confidence_level) / 2
        lower = float(np.percentile(values, alpha))
        upper = float(np.percentile(values, 100 - alpha))

        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            median=float(np.median(values)),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            percentile=confidence_level,
        )

    def _compute_percentile_ranking(
        self,
        original_value: float,
        simulated_values: np.ndarray,
        higher_is_better: bool = True,
    ) -> float:
        """
        Compute where the original value ranks in the simulated distribution.

        Args:
            original_value: The original metric value
            simulated_values: Array of simulated values
            higher_is_better: If True, higher percentile is better

        Returns:
            Percentile ranking (0-100)
        """
        if len(simulated_values) == 0:
            return 50.0

        if higher_is_better:
            # How many simulations did worse?
            pct = (simulated_values < original_value).mean() * 100
        else:
            # How many simulations did better? (for drawdown, lower is better)
            pct = (simulated_values > original_value).mean() * 100

        return float(pct)

    def run(
        self,
        confidence_level: Optional[float] = None,
        store_equity_curves: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Shuffles trade order n_simulations times and computes
        confidence intervals for all key metrics.

        Args:
            confidence_level: Override default confidence level
            store_equity_curves: Store all equity curves (memory intensive)
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            MonteCarloResult with all computed statistics
        """
        if confidence_level is not None:
            self.config.confidence_level = confidence_level
        self.config.store_equity_curves = store_equity_curves

        if len(self.trades) == 0:
            logger.warning("No trades to simulate")
            empty_ci = ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0.0)
            return MonteCarloResult(
                config=self.config,
                n_simulations_completed=0,
                original_final_equity=self.config.initial_capital,
                original_max_drawdown=0.0,
                original_sharpe=0.0,
                original_sortino=0.0,
                original_profit_factor=0.0,
                original_win_rate=0.0,
                final_equity_ci=empty_ci,
                max_drawdown_ci=empty_ci,
                sharpe_ratio_ci=empty_ci,
                sortino_ratio_ci=empty_ci,
                profit_factor_ci=empty_ci,
                win_rate_ci=empty_ci,
            )

        logger.info(
            f"Starting Monte Carlo simulation: "
            f"{self.config.n_simulations} simulations, "
            f"{len(self.trades)} trades"
        )

        # Run simulations
        runs: List[SimulationRun] = []
        for i in range(self.config.n_simulations):
            run = self._run_single_simulation(i)
            runs.append(run)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, self.config.n_simulations)

        # Extract arrays for analysis
        final_equities = np.array([r.final_equity for r in runs])
        max_drawdowns = np.array([r.max_drawdown_pct for r in runs])
        sharpe_ratios = np.array([r.sharpe_ratio for r in runs])
        sortino_ratios = np.array([r.sortino_ratio for r in runs])
        profit_factors = np.array([r.profit_factor for r in runs])
        win_rates = np.array([r.win_rate for r in runs])

        # Compute confidence intervals
        conf_level = self.config.confidence_level
        final_equity_ci = self._compute_confidence_interval(final_equities, conf_level)
        max_drawdown_ci = self._compute_confidence_interval(max_drawdowns, conf_level)
        sharpe_ratio_ci = self._compute_confidence_interval(sharpe_ratios, conf_level)
        sortino_ratio_ci = self._compute_confidence_interval(sortino_ratios, conf_level)
        profit_factor_ci = self._compute_confidence_interval(profit_factors, conf_level)
        win_rate_ci = self._compute_confidence_interval(win_rates, conf_level)

        # Compute percentile rankings
        orig = self._original_metrics
        percentile_rankings = {
            "final_equity": self._compute_percentile_ranking(
                orig["final_equity"], final_equities, higher_is_better=True
            ),
            "max_drawdown": self._compute_percentile_ranking(
                orig["max_drawdown_pct"], max_drawdowns, higher_is_better=False
            ),
            "sharpe_ratio": self._compute_percentile_ranking(
                orig["sharpe_ratio"], sharpe_ratios, higher_is_better=True
            ),
            "sortino_ratio": self._compute_percentile_ranking(
                orig["sortino_ratio"], sortino_ratios, higher_is_better=True
            ),
            "profit_factor": self._compute_percentile_ranking(
                orig["profit_factor"], profit_factors, higher_is_better=True
            ),
            "win_rate": self._compute_percentile_ranking(
                orig["win_rate"], win_rates, higher_is_better=True
            ),
        }

        # Summary statistics
        summary_stats = {
            "simulations_completed": len(runs),
            "trades_per_simulation": len(self.trades),
            "worst_case_equity": float(np.min(final_equities)),
            "best_case_equity": float(np.max(final_equities)),
            "probability_of_profit": float((final_equities > self.config.initial_capital).mean()),
            "probability_of_positive_sharpe": float((sharpe_ratios > 0).mean()),
            "worst_case_drawdown": float(np.max(max_drawdowns)),
            "best_case_drawdown": float(np.min(max_drawdowns)),
        }

        logger.info(
            f"Monte Carlo complete: "
            f"Equity CI [{final_equity_ci.lower:.2f}, {final_equity_ci.upper:.2f}], "
            f"Sharpe CI [{sharpe_ratio_ci.lower:.3f}, {sharpe_ratio_ci.upper:.3f}]"
        )

        return MonteCarloResult(
            config=self.config,
            n_simulations_completed=len(runs),
            original_final_equity=orig["final_equity"],
            original_max_drawdown=orig["max_drawdown_pct"],
            original_sharpe=orig["sharpe_ratio"],
            original_sortino=orig["sortino_ratio"],
            original_profit_factor=orig["profit_factor"],
            original_win_rate=orig["win_rate"],
            final_equity_ci=final_equity_ci,
            max_drawdown_ci=max_drawdown_ci,
            sharpe_ratio_ci=sharpe_ratio_ci,
            sortino_ratio_ci=sortino_ratio_ci,
            profit_factor_ci=profit_factor_ci,
            win_rate_ci=win_rate_ci,
            percentile_rankings=percentile_rankings,
            all_runs=runs if store_equity_curves else [],
            summary_stats=summary_stats,
        )

    @classmethod
    def from_trade_pnls(
        cls,
        pnls: List[float],
        n_simulations: int = 1000,
        initial_capital: float = 1000.0,
        random_seed: Optional[int] = None,
    ) -> 'MonteCarloSimulator':
        """
        Create simulator from a list of P&Ls (without full TradeRecord).

        Useful when you only have P&L values, not full trade records.

        Args:
            pnls: List of trade P&Ls
            n_simulations: Number of simulations
            initial_capital: Starting capital
            random_seed: Random seed

        Returns:
            MonteCarloSimulator instance
        """
        from datetime import datetime
        from .trade_logger import ExitReason

        # Create minimal TradeRecord objects
        trades = []
        now = datetime.now()
        for i, pnl in enumerate(pnls):
            trade = TradeRecord(
                trade_id=i + 1,
                entry_time=now,
                exit_time=now,
                direction=1 if pnl > 0 else -1,
                entry_price=100.0,
                exit_price=100.0 + (pnl / 5.0),  # Approximate
                contracts=1,
                gross_pnl=pnl,
                commission=0.0,
                slippage=0.0,
                net_pnl=pnl,
                exit_reason=ExitReason.SIGNAL,
            )
            trades.append(trade)

        return cls(
            trades=trades,
            n_simulations=n_simulations,
            initial_capital=initial_capital,
            random_seed=random_seed,
        )


def run_monte_carlo_from_csv(
    trades_csv: str,
    n_simulations: int = 1000,
    initial_capital: float = 1000.0,
    output_json: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation from a trades CSV file.

    CSV must have at minimum a 'net_pnl' column.

    Args:
        trades_csv: Path to trades CSV file
        n_simulations: Number of simulations
        initial_capital: Starting capital
        output_json: Optional path to save results
        random_seed: Random seed for reproducibility

    Returns:
        MonteCarloResult
    """
    import csv

    pnls = []
    with open(trades_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pnl = float(row.get('net_pnl', row.get('pnl', 0)))
            pnls.append(pnl)

    if not pnls:
        raise ValueError(f"No trades found in {trades_csv}")

    simulator = MonteCarloSimulator.from_trade_pnls(
        pnls=pnls,
        n_simulations=n_simulations,
        initial_capital=initial_capital,
        random_seed=random_seed,
    )

    result = simulator.run()

    if output_json:
        result.export_json(output_json)
        logger.info(f"Results saved to {output_json}")

    return result
