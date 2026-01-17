"""
Performance Metrics for Backtesting

This module calculates comprehensive performance metrics for evaluating
trading strategy performance. These metrics help assess both profitability
and risk characteristics.

Key Metrics:
- Return metrics: Total return, CAGR, daily/monthly returns
- Risk metrics: Sharpe, Sortino, Calmar ratios
- Drawdown analysis: Max drawdown, duration, recovery
- Trade metrics: Win rate, profit factor, expectancy
- Consistency: Win days %, streaks, best/worst days

Why these metrics matter:
- Sharpe > 1.0: Good risk-adjusted returns
- Sortino > 1.5: Good downside risk management
- Calmar > 0.5: Acceptable drawdown vs returns tradeoff
- Win rate > 52%: Beating random on 3-class prediction
- Profit factor > 1.5: Wins significantly exceed losses
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest.

    All metrics are calculated from trade logs and equity curves.
    This dataclass provides a complete picture of strategy performance.

    Why each metric category matters:
    - Return metrics: Raw profitability
    - Risk metrics: Quality of returns relative to volatility
    - Drawdown metrics: Worst-case scenarios
    - Trade metrics: Execution quality and edge
    - Consistency metrics: Reliability over time
    - Cost metrics: Transaction efficiency
    """

    # Period info
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    # Return metrics
    total_return_pct: float = 0.0
    total_return_dollars: float = 0.0
    cagr_pct: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0
    monthly_return_mean: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    daily_var_95: float = 0.0
    daily_var_99: float = 0.0

    # Drawdown metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0

    # Consistency metrics
    win_days_pct: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    best_day_pct: float = 0.0
    worst_day_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trades_per_day: float = 0.0

    # Cost metrics
    total_commission: float = 0.0
    total_slippage: float = 0.0
    cost_per_trade: float = 0.0
    cost_pct_of_gross: float = 0.0

    # Additional context
    initial_capital: float = 0.0
    final_capital: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0

    # Returns source indicator for Sharpe/Sortino
    # "equity": Calculated from equity curve (includes mark-to-market)
    # "closed_trades": Calculated from closed trade returns only
    returns_source: str = "equity"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        # Helper to convert numpy types to native Python types
        def _to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        # Handle datetime serialization
        start_str = None
        end_str = None
        if self.start_date is not None:
            try:
                start_str = self.start_date.isoformat()
            except AttributeError:
                start_str = str(self.start_date)
        if self.end_date is not None:
            try:
                end_str = self.end_date.isoformat()
            except AttributeError:
                end_str = str(self.end_date)

        return {
            "period": {
                "start_date": start_str,
                "end_date": end_str,
                "trading_days": int(_to_native(self.trading_days)),
            },
            "returns": {
                "total_return_pct": float(round(_to_native(self.total_return_pct), 4)),
                "total_return_dollars": float(round(_to_native(self.total_return_dollars), 2)),
                "cagr_pct": float(round(_to_native(self.cagr_pct), 4)),
                "daily_return_mean": float(round(_to_native(self.daily_return_mean), 6)),
                "daily_return_std": float(round(_to_native(self.daily_return_std), 6)),
                "monthly_return_mean": float(round(_to_native(self.monthly_return_mean), 4)),
            },
            "risk": {
                "sharpe_ratio": float(round(_to_native(self.sharpe_ratio), 3)),
                "sortino_ratio": float(round(_to_native(self.sortino_ratio), 3)),
                "calmar_ratio": float(round(_to_native(self.calmar_ratio), 3)),
                "daily_var_95": float(round(_to_native(self.daily_var_95), 4)),
                "daily_var_99": float(round(_to_native(self.daily_var_99), 4)),
                "returns_source": self.returns_source,  # "equity" or "closed_trades"
            },
            "drawdown": {
                "max_drawdown_pct": float(round(_to_native(self.max_drawdown_pct), 4)),
                "max_drawdown_dollars": float(round(_to_native(self.max_drawdown_dollars), 2)),
                "max_drawdown_duration_days": int(_to_native(self.max_drawdown_duration_days)),
                "avg_drawdown_pct": float(round(_to_native(self.avg_drawdown_pct), 4)),
                "recovery_factor": float(round(_to_native(self.recovery_factor), 3)),
            },
            "trades": {
                "total_trades": int(_to_native(self.total_trades)),
                "winning_trades": int(_to_native(self.winning_trades)),
                "losing_trades": int(_to_native(self.losing_trades)),
                "win_rate_pct": float(round(_to_native(self.win_rate_pct), 2)),
                "profit_factor": float(round(_to_native(self.profit_factor), 3)),
                "avg_trade_pnl": float(round(_to_native(self.avg_trade_pnl), 2)),
                "avg_win": float(round(_to_native(self.avg_win), 2)),
                "avg_loss": float(round(_to_native(self.avg_loss), 2)),
                "largest_win": float(round(_to_native(self.largest_win), 2)),
                "largest_loss": float(round(_to_native(self.largest_loss), 2)),
                "expectancy": float(round(_to_native(self.expectancy), 2)),
                "expectancy_ratio": float(round(_to_native(self.expectancy_ratio), 3)),
            },
            "consistency": {
                "win_days_pct": float(round(_to_native(self.win_days_pct), 2)),
                "best_day_pnl": float(round(_to_native(self.best_day_pnl), 2)),
                "worst_day_pnl": float(round(_to_native(self.worst_day_pnl), 2)),
                "best_day_pct": float(round(_to_native(self.best_day_pct), 4)),
                "worst_day_pct": float(round(_to_native(self.worst_day_pct), 4)),
                "max_consecutive_wins": int(_to_native(self.max_consecutive_wins)),
                "max_consecutive_losses": int(_to_native(self.max_consecutive_losses)),
                "avg_trades_per_day": float(round(_to_native(self.avg_trades_per_day), 2)),
            },
            "costs": {
                "total_commission": float(round(_to_native(self.total_commission), 2)),
                "total_slippage": float(round(_to_native(self.total_slippage), 2)),
                "cost_per_trade": float(round(_to_native(self.cost_per_trade), 2)),
                "cost_pct_of_gross": float(round(_to_native(self.cost_pct_of_gross), 4)),
            },
            "capital": {
                "initial": float(round(_to_native(self.initial_capital), 2)),
                "final": float(round(_to_native(self.final_capital), 2)),
                "gross_profit": float(round(_to_native(self.gross_profit), 2)),
                "gross_loss": float(round(_to_native(self.gross_loss), 2)),
                "net_profit": float(round(_to_native(self.net_profit), 2)),
            },
        }


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(252),
) -> float:
    """
    Calculate Sharpe ratio from daily returns.

    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns * sqrt(252)

    Interpretation:
    - < 0: Losing money
    - 0-1: Suboptimal risk-adjusted returns
    - 1-2: Good
    - 2-3: Very good
    - > 3: Excellent (or suspicious)

    Args:
        returns: Array of period returns (typically daily)
        risk_free_rate: Risk-free rate per period (default 0)
        annualization_factor: Factor to annualize (sqrt(252) for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    returns = np.array(returns)
    excess_returns = returns - risk_free_rate

    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    return (mean_return / std_return) * annualization_factor


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(252),
) -> float:
    """
    Calculate Sortino ratio from daily returns.

    Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation * sqrt(252)

    Unlike Sharpe, Sortino only penalizes downside volatility.
    This is often preferred for trading strategies where large gains are good.

    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate per period
        annualization_factor: Factor to annualize

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)

    # Downside returns only (negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No downside = infinite Sortino, cap at high value
        return 10.0 if mean_return > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return (mean_return / downside_std) * annualization_factor


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: float = 1.0,
) -> float:
    """
    Calculate Calmar ratio.

    Calmar Ratio = CAGR / Max Drawdown

    Measures return per unit of maximum drawdown.
    Good for assessing risk of catastrophic loss.

    Interpretation:
    - < 0.5: Poor
    - 0.5-1.0: Acceptable
    - 1.0-2.0: Good
    - > 2.0: Excellent

    Args:
        total_return: Total return (as decimal, e.g., 0.25 for 25%)
        max_drawdown: Maximum drawdown (as decimal, e.g., 0.10 for 10%)
        years: Number of years in period

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0 or years <= 0:
        return 0.0

    # Calculate CAGR
    if total_return <= -1:
        return 0.0  # Lost everything

    cagr = (1 + total_return) ** (1 / years) - 1

    return cagr / abs(max_drawdown)


def calculate_max_drawdown(
    equity_curve: np.ndarray,
) -> Tuple[float, float, int, int, int]:
    """
    Calculate maximum drawdown and related metrics.

    Drawdown = (Peak - Trough) / Peak

    Args:
        equity_curve: Array of equity values over time

    Returns:
        Tuple of (max_dd_pct, max_dd_dollars, peak_idx, trough_idx, duration_bars)
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0, 0, 0, 0

    equity = np.array(equity_curve)

    # Running maximum (high water mark)
    running_max = np.maximum.accumulate(equity)

    # Drawdown at each point
    drawdowns = (running_max - equity) / running_max
    drawdown_dollars = running_max - equity

    # Find maximum drawdown
    max_dd_idx = np.argmax(drawdowns)
    max_dd_pct = drawdowns[max_dd_idx]
    max_dd_dollars_val = drawdown_dollars[max_dd_idx]

    # Find peak before max drawdown
    peak_idx = np.argmax(equity[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Calculate duration (bars from peak to trough)
    duration = max_dd_idx - peak_idx

    return max_dd_pct, max_dd_dollars_val, peak_idx, max_dd_idx, duration


def calculate_drawdown_series(
    equity_curve: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate drawdown series for visualization.

    Args:
        equity_curve: Array of equity values

    Returns:
        Tuple of (drawdown_pct_series, drawdown_dollar_series)
    """
    if len(equity_curve) == 0:
        return np.array([]), np.array([])

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)

    drawdown_pct = (running_max - equity) / running_max
    drawdown_dollars = running_max - equity

    return drawdown_pct, drawdown_dollars


def calculate_win_rate(wins: int, total: int) -> float:
    """Calculate win rate as percentage."""
    if total == 0:
        return 0.0
    return (wins / total) * 100


def calculate_profit_factor(gross_profit: float, gross_loss: float) -> float:
    """
    Calculate profit factor.

    Profit Factor = Gross Profit / Gross Loss

    Args:
        gross_profit: Sum of all winning trades (positive)
        gross_loss: Sum of all losing trades (positive, absolute value)

    Returns:
        Profit factor (> 1 is profitable)
    """
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def calculate_expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate expectancy per trade.

    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)

    This tells you the expected value of each trade.

    Args:
        win_rate: Win rate as decimal (0-1)
        avg_win: Average winning trade (positive)
        avg_loss: Average losing trade (positive, will be negated)

    Returns:
        Expected value per trade in dollars
    """
    loss_rate = 1 - win_rate
    return (win_rate * avg_win) - (loss_rate * abs(avg_loss))


def calculate_consecutive_streaks(
    trade_results: List[float],
) -> Tuple[int, int]:
    """
    Calculate maximum consecutive wins and losses.

    Args:
        trade_results: List of trade P&Ls

    Returns:
        Tuple of (max_consecutive_wins, max_consecutive_losses)
    """
    if not trade_results:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in trade_results:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        # pnl == 0: breakeven, reset both
        else:
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


def calculate_daily_stats(
    daily_pnls: List[float],
) -> Tuple[float, float, float, int, int]:
    """
    Calculate daily P&L statistics.

    Args:
        daily_pnls: List of daily P&L values

    Returns:
        Tuple of (best_day, worst_day, win_days_pct, win_days, total_days)
    """
    if not daily_pnls:
        return 0.0, 0.0, 0.0, 0, 0

    pnls = np.array(daily_pnls)
    best_day = float(np.max(pnls))
    worst_day = float(np.min(pnls))
    win_days = int(np.sum(pnls > 0))
    total_days = len(pnls)
    win_days_pct = (win_days / total_days) * 100 if total_days > 0 else 0.0

    return best_day, worst_day, win_days_pct, win_days, total_days


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Value at Risk (VaR).

    VaR represents the maximum expected loss at a given confidence level.
    For 95% VaR, there's a 5% chance of losing more than this amount.

    Args:
        returns: Array of returns
        confidence: Confidence level (0.95 for 95%)

    Returns:
        VaR as a positive number (loss)
    """
    if len(returns) == 0:
        return 0.0

    returns = np.array(returns)
    percentile = (1 - confidence) * 100  # e.g., 5 for 95% confidence
    var = -np.percentile(returns, percentile)

    return max(0, var)  # Return positive loss


def calculate_metrics(
    trade_pnls: List[float],
    equity_curve: List[float],
    initial_capital: float,
    trading_days: int,
    total_commission: float = 0.0,
    total_slippage: float = 0.0,
    daily_pnls: Optional[List[float]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    use_closed_trade_returns: bool = False,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from trade results.

    This is the main function to call after a backtest completes.
    It computes all metrics needed for strategy evaluation.

    Args:
        trade_pnls: List of net P&L for each trade
        equity_curve: List of equity values over time
        initial_capital: Starting capital
        trading_days: Number of trading days in backtest
        total_commission: Total commission paid
        total_slippage: Total slippage cost
        daily_pnls: Optional list of daily P&L (for consistency metrics)
        start_date: Start date of backtest
        end_date: End date of backtest
        use_closed_trade_returns: If True and daily_pnls is provided, calculate
            Sharpe/Sortino from closed trade daily returns instead of equity curve.
            This avoids exaggerated volatility from mark-to-market swings during
            short-duration scalping trades. Requires daily_pnls to be provided.

    Returns:
        PerformanceMetrics with all calculated values

    Note:
        The returns_source field in the returned metrics indicates which method
        was used: "equity" for equity curve or "closed_trades" for daily P&L.
    """
    metrics = PerformanceMetrics()

    # Period info
    metrics.start_date = start_date
    metrics.end_date = end_date
    metrics.trading_days = trading_days
    metrics.initial_capital = initial_capital

    if not trade_pnls or not equity_curve:
        metrics.final_capital = initial_capital
        return metrics

    # Convert to numpy arrays
    pnls = np.array(trade_pnls)
    equity = np.array(equity_curve)

    # Basic stats
    metrics.total_trades = len(pnls)
    metrics.winning_trades = int(np.sum(pnls > 0))
    metrics.losing_trades = int(np.sum(pnls < 0))
    metrics.final_capital = float(equity[-1]) if len(equity) > 0 else initial_capital

    # Return metrics
    metrics.total_return_dollars = metrics.final_capital - initial_capital
    metrics.total_return_pct = (
        metrics.total_return_dollars / initial_capital
        if initial_capital > 0 else 0.0
    )

    # CAGR
    years = trading_days / 252 if trading_days > 0 else 1.0
    if metrics.total_return_pct > -1:
        metrics.cagr_pct = (1 + metrics.total_return_pct) ** (1 / years) - 1
    else:
        metrics.cagr_pct = -1.0

    # Daily returns calculation - two options:
    # 1. From equity curve (includes mark-to-market unrealized P&L)
    # 2. From closed trade daily P&L (no intraday volatility)
    #
    # For scalping strategies with short-duration trades, option 2 may be preferred
    # as it avoids exaggerated volatility from mark-to-market swings.
    daily_returns: Optional[np.ndarray] = None

    # Option 1: Calculate from closed trade daily returns (recommended for scalping)
    if use_closed_trade_returns and daily_pnls is not None and len(daily_pnls) > 0:
        # Build daily equity curve from daily P&L
        daily_equity = [initial_capital]
        for pnl in daily_pnls:
            daily_equity.append(daily_equity[-1] + pnl)
        daily_equity = np.array(daily_equity)

        # Calculate returns from daily equity (excludes intraday volatility)
        daily_returns = np.diff(daily_equity) / daily_equity[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        daily_returns = daily_returns[~np.isinf(daily_returns)]
        metrics.returns_source = "closed_trades"

    # Option 2: Calculate from full equity curve (default - includes mark-to-market)
    elif len(equity) > 1:
        daily_returns = np.diff(equity) / equity[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        daily_returns = daily_returns[~np.isinf(daily_returns)]
        metrics.returns_source = "equity"

    if daily_returns is not None and len(daily_returns) > 0:
        metrics.daily_return_mean = float(np.mean(daily_returns))
        metrics.daily_return_std = float(np.std(daily_returns, ddof=1))
        metrics.monthly_return_mean = metrics.daily_return_mean * 21  # ~21 trading days/month

        # Risk metrics (Sharpe/Sortino calculated from chosen returns source)
        metrics.sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        metrics.sortino_ratio = calculate_sortino_ratio(daily_returns)

        # VaR
        metrics.daily_var_95 = calculate_var(daily_returns, 0.95)
        metrics.daily_var_99 = calculate_var(daily_returns, 0.99)

    # Drawdown metrics
    max_dd_pct, max_dd_dollars, peak_idx, trough_idx, duration = calculate_max_drawdown(equity)
    metrics.max_drawdown_pct = float(max_dd_pct)
    metrics.max_drawdown_dollars = float(max_dd_dollars)

    # Estimate duration in days (assuming ~6.5 hours of bars per day for 1-second data)
    # Or use actual days if daily equity is provided
    if daily_pnls is not None and len(daily_pnls) > 0:
        # Use daily data for duration
        daily_equity = [initial_capital]
        for pnl in daily_pnls:
            daily_equity.append(daily_equity[-1] + pnl)
        _, _, _, _, day_duration = calculate_max_drawdown(np.array(daily_equity))
        metrics.max_drawdown_duration_days = day_duration
    else:
        # Rough estimate: divide bar duration by bars per day
        bars_per_day = len(equity) / trading_days if trading_days > 0 else 1
        metrics.max_drawdown_duration_days = int(duration / bars_per_day) if bars_per_day > 0 else 0

    # Average drawdown
    dd_pct_series, _ = calculate_drawdown_series(equity)
    metrics.avg_drawdown_pct = float(np.mean(dd_pct_series)) if len(dd_pct_series) > 0 else 0.0

    # Calmar ratio
    metrics.calmar_ratio = calculate_calmar_ratio(
        metrics.total_return_pct,
        metrics.max_drawdown_pct,
        years,
    )

    # Recovery factor
    if metrics.max_drawdown_dollars > 0:
        metrics.recovery_factor = metrics.total_return_dollars / metrics.max_drawdown_dollars
    else:
        metrics.recovery_factor = float('inf') if metrics.total_return_dollars > 0 else 0.0

    # Trade metrics
    metrics.win_rate_pct = calculate_win_rate(metrics.winning_trades, metrics.total_trades)

    # Gross profit/loss
    winning_pnls = pnls[pnls > 0]
    losing_pnls = pnls[pnls < 0]

    metrics.gross_profit = float(np.sum(winning_pnls)) if len(winning_pnls) > 0 else 0.0
    metrics.gross_loss = float(np.abs(np.sum(losing_pnls))) if len(losing_pnls) > 0 else 0.0
    metrics.net_profit = metrics.gross_profit - metrics.gross_loss

    metrics.profit_factor = calculate_profit_factor(metrics.gross_profit, metrics.gross_loss)

    # Average trade stats
    metrics.avg_trade_pnl = float(np.mean(pnls))
    metrics.avg_win = float(np.mean(winning_pnls)) if len(winning_pnls) > 0 else 0.0
    metrics.avg_loss = float(np.abs(np.mean(losing_pnls))) if len(losing_pnls) > 0 else 0.0
    metrics.largest_win = float(np.max(winning_pnls)) if len(winning_pnls) > 0 else 0.0
    metrics.largest_loss = float(np.abs(np.min(losing_pnls))) if len(losing_pnls) > 0 else 0.0

    # Expectancy
    win_rate_decimal = metrics.win_rate_pct / 100
    metrics.expectancy = calculate_expectancy(win_rate_decimal, metrics.avg_win, metrics.avg_loss)

    # Expectancy ratio (relative to avg loss)
    if metrics.avg_loss > 0:
        metrics.expectancy_ratio = metrics.expectancy / metrics.avg_loss
    else:
        metrics.expectancy_ratio = 0.0

    # Consistency metrics
    max_wins, max_losses = calculate_consecutive_streaks(list(pnls))
    metrics.max_consecutive_wins = max_wins
    metrics.max_consecutive_losses = max_losses

    if daily_pnls is not None and len(daily_pnls) > 0:
        best, worst, win_pct, _, _ = calculate_daily_stats(daily_pnls)
        metrics.best_day_pnl = best
        metrics.worst_day_pnl = worst
        metrics.win_days_pct = win_pct
        metrics.best_day_pct = best / initial_capital if initial_capital > 0 else 0.0
        metrics.worst_day_pct = worst / initial_capital if initial_capital > 0 else 0.0

    metrics.avg_trades_per_day = metrics.total_trades / trading_days if trading_days > 0 else 0.0

    # Cost metrics
    metrics.total_commission = total_commission
    metrics.total_slippage = total_slippage
    metrics.cost_per_trade = (
        (total_commission + total_slippage) / metrics.total_trades
        if metrics.total_trades > 0 else 0.0
    )

    # Cost as % of gross profit
    if metrics.gross_profit > 0:
        metrics.cost_pct_of_gross = (total_commission + total_slippage) / metrics.gross_profit
    else:
        metrics.cost_pct_of_gross = 0.0

    return metrics


def compare_metrics(
    baseline: PerformanceMetrics,
    comparison: PerformanceMetrics,
) -> Dict[str, float]:
    """
    Compare two sets of metrics and return deltas.

    Useful for comparing walk-forward folds or strategy variants.

    Args:
        baseline: Baseline metrics
        comparison: Metrics to compare

    Returns:
        Dictionary of metric deltas (comparison - baseline)
    """
    return {
        "sharpe_delta": comparison.sharpe_ratio - baseline.sharpe_ratio,
        "sortino_delta": comparison.sortino_ratio - baseline.sortino_ratio,
        "calmar_delta": comparison.calmar_ratio - baseline.calmar_ratio,
        "return_delta_pct": comparison.total_return_pct - baseline.total_return_pct,
        "win_rate_delta": comparison.win_rate_pct - baseline.win_rate_pct,
        "profit_factor_delta": comparison.profit_factor - baseline.profit_factor,
        "max_dd_delta": comparison.max_drawdown_pct - baseline.max_drawdown_pct,
        "expectancy_delta": comparison.expectancy - baseline.expectancy,
    }
