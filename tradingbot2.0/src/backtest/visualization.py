"""
Interactive Visualization for Backtesting Results

This module provides Plotly-based interactive visualizations for analyzing
backtest results. Interactive charts are essential for:
1. Exploring equity curves with zoom/pan
2. Identifying drawdown periods with hover details
3. Analyzing trade distributions
4. Comparing walk-forward fold performance
5. Detecting overfitting patterns

Why Plotly?
- Interactive zoom/pan for detailed analysis
- Hover tooltips with trade details
- Export to HTML for sharing
- Subplots for comprehensive dashboards
- No server required (offline rendering)

Output Formats:
- HTML files (interactive, shareable)
- PNG images (static, for reports)
- JSON data (for programmatic access)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore

from src.backtest.trade_logger import TradeRecord, EquityPoint, TradeLog, EquityCurve, BacktestReport
from src.backtest.metrics import PerformanceMetrics, calculate_drawdown_series


@dataclass
class DrawdownPeriod:
    """
    Represents a single drawdown period with timing details.

    Attributes:
        start_time: When drawdown began (peak)
        trough_time: When maximum drawdown occurred
        end_time: When equity recovered to peak (or None if still in drawdown)
        peak_equity: Equity at start of drawdown
        trough_equity: Minimum equity during drawdown
        max_drawdown_pct: Maximum drawdown as percentage
        max_drawdown_dollars: Maximum drawdown in dollars
        duration_bars: Number of bars from start to trough
        recovery_bars: Number of bars from trough to recovery (or None)
    """
    start_time: datetime
    trough_time: datetime
    end_time: Optional[datetime]
    peak_equity: float
    trough_equity: float
    max_drawdown_pct: float
    max_drawdown_dollars: float
    duration_bars: int
    recovery_bars: Optional[int]

    @property
    def is_recovered(self) -> bool:
        """Has equity recovered from this drawdown?"""
        return self.end_time is not None

    @property
    def total_bars(self) -> Optional[int]:
        """Total bars from start to recovery."""
        if self.recovery_bars is None:
            return None
        return self.duration_bars + self.recovery_bars


def identify_drawdown_periods(
    equity_points: List[EquityPoint],
    min_drawdown_pct: float = 0.01,
) -> List[DrawdownPeriod]:
    """
    Identify distinct drawdown periods from equity curve.

    A drawdown period starts when equity drops from a peak and ends
    when equity recovers to a new high.

    Args:
        equity_points: List of equity points over time
        min_drawdown_pct: Minimum drawdown % to consider (filters noise)

    Returns:
        List of DrawdownPeriod objects, sorted by start time
    """
    if not equity_points:
        return []

    periods = []
    peak_equity = equity_points[0].equity
    peak_time = equity_points[0].timestamp
    peak_idx = 0

    in_drawdown = False
    current_trough_equity = peak_equity
    current_trough_time = peak_time
    current_trough_idx = 0
    drawdown_start_idx = 0

    for i, point in enumerate(equity_points):
        if point.equity > peak_equity:
            # New high - if we were in a drawdown, close it
            if in_drawdown:
                max_dd_pct = (peak_equity - current_trough_equity) / peak_equity
                if max_dd_pct >= min_drawdown_pct:
                    periods.append(DrawdownPeriod(
                        start_time=peak_time,
                        trough_time=current_trough_time,
                        end_time=point.timestamp,
                        peak_equity=peak_equity,
                        trough_equity=current_trough_equity,
                        max_drawdown_pct=max_dd_pct,
                        max_drawdown_dollars=peak_equity - current_trough_equity,
                        duration_bars=current_trough_idx - drawdown_start_idx,
                        recovery_bars=i - current_trough_idx,
                    ))
                in_drawdown = False

            # Update peak
            peak_equity = point.equity
            peak_time = point.timestamp
            peak_idx = i
            current_trough_equity = point.equity
            current_trough_time = point.timestamp
            current_trough_idx = i

        elif point.equity < peak_equity:
            # In drawdown
            if not in_drawdown:
                in_drawdown = True
                drawdown_start_idx = peak_idx

            # Track trough
            if point.equity < current_trough_equity:
                current_trough_equity = point.equity
                current_trough_time = point.timestamp
                current_trough_idx = i

    # Handle ongoing drawdown at end
    if in_drawdown:
        max_dd_pct = (peak_equity - current_trough_equity) / peak_equity
        if max_dd_pct >= min_drawdown_pct:
            periods.append(DrawdownPeriod(
                start_time=peak_time,
                trough_time=current_trough_time,
                end_time=None,  # Still in drawdown
                peak_equity=peak_equity,
                trough_equity=current_trough_equity,
                max_drawdown_pct=max_dd_pct,
                max_drawdown_dollars=peak_equity - current_trough_equity,
                duration_bars=current_trough_idx - drawdown_start_idx,
                recovery_bars=None,
            ))

    return periods


def _require_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install it with: pip install plotly>=5.0.0"
        )


class BacktestVisualizer:
    """
    Interactive visualizations for backtest results.

    This class provides methods to create Plotly charts for analyzing
    trading strategy performance.

    Usage:
        report = engine.run_backtest(data, model)
        viz = BacktestVisualizer(report)

        # Interactive equity curve
        fig = viz.plot_equity_curve()
        fig.write_html("equity.html")

        # Full dashboard
        fig = viz.create_dashboard()
        fig.show()
    """

    def __init__(
        self,
        report: Optional[BacktestReport] = None,
        trade_log: Optional[TradeLog] = None,
        equity_curve: Optional[EquityCurve] = None,
        metrics: Optional[PerformanceMetrics] = None,
    ):
        """
        Initialize visualizer with backtest results.

        Args:
            report: Complete backtest report (preferred)
            trade_log: Trade log if no report
            equity_curve: Equity curve if no report
            metrics: Performance metrics if no report
        """
        _require_plotly()

        if report is not None:
            self.trade_log = report.trade_log
            self.equity_curve = report.equity_curve
            self.metrics = report.metrics
        else:
            self.trade_log = trade_log
            self.equity_curve = equity_curve
            self.metrics = metrics

        # Default colors
        self.colors = {
            'equity': '#2196F3',  # Blue
            'drawdown': '#F44336',  # Red
            'win': '#4CAF50',  # Green
            'loss': '#F44336',  # Red
            'long': '#2196F3',  # Blue
            'short': '#FF9800',  # Orange
            'flat': '#9E9E9E',  # Gray
            'target': '#4CAF50',  # Green
            'stop': '#F44336',  # Red
            'eod': '#9C27B0',  # Purple
        }

    def plot_equity_curve(
        self,
        title: str = "Equity Curve",
        show_trades: bool = True,
        show_drawdown: bool = True,
        height: int = 600,
    ) -> 'go.Figure':
        """
        Create interactive equity curve plot.

        Features:
        - Equity line with high-water mark
        - Drawdown shaded area
        - Trade entry/exit markers
        - Hover details

        Args:
            title: Chart title
            show_trades: Show trade markers
            show_drawdown: Show drawdown overlay
            height: Chart height in pixels

        Returns:
            Plotly Figure object
        """
        _require_plotly()

        if self.equity_curve is None:
            raise ValueError("No equity curve data available")

        points = self.equity_curve.get_points()
        if not points:
            raise ValueError("Equity curve is empty")

        timestamps = [p.timestamp for p in points]
        equity_values = [p.equity for p in points]
        drawdown_values = [p.drawdown_pct * 100 for p in points]  # Convert to %

        # Calculate high-water mark
        hwm = np.maximum.accumulate(equity_values)

        if show_drawdown:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("Equity", "Drawdown (%)"),
            )

            # Equity trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=equity_values,
                    name="Equity",
                    line=dict(color=self.colors['equity'], width=2),
                    hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

            # High-water mark
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=hwm.tolist(),
                    name="High Water Mark",
                    line=dict(color=self.colors['equity'], width=1, dash='dot'),
                    opacity=0.5,
                    hovertemplate="<b>%{x}</b><br>HWM: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

            # Drawdown trace (inverted so deeper drawdown goes down)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[-d for d in drawdown_values],
                    name="Drawdown",
                    fill='tozeroy',
                    line=dict(color=self.colors['drawdown'], width=1),
                    fillcolor='rgba(244, 67, 54, 0.3)',
                    hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
                ),
                row=2, col=1,
            )

            fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=equity_values,
                    name="Equity",
                    line=dict(color=self.colors['equity'], width=2),
                    hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>",
                )
            )

            fig.update_yaxes(title_text="Equity ($)")

        # Add trade markers if available
        if show_trades and self.trade_log is not None:
            trades = self.trade_log.get_trades()
            if trades:
                # Entry markers
                entry_times = [t.entry_time for t in trades]
                entry_prices = []

                # Find equity at entry times
                for t in trades:
                    # Find closest equity point
                    closest_idx = min(
                        range(len(timestamps)),
                        key=lambda i: abs((timestamps[i] - t.entry_time).total_seconds())
                    )
                    entry_prices.append(equity_values[closest_idx])

                entry_colors = [
                    self.colors['long'] if t.direction == 1 else self.colors['short']
                    for t in trades
                ]

                fig.add_trace(
                    go.Scatter(
                        x=entry_times,
                        y=entry_prices,
                        mode='markers',
                        name='Entries',
                        marker=dict(
                            size=8,
                            color=entry_colors,
                            symbol='triangle-up',
                        ),
                        hovertemplate="<b>Entry</b><br>%{x}<br>$%{y:,.2f}<extra></extra>",
                    ),
                    row=1 if show_drawdown else None,
                    col=1 if show_drawdown else None,
                )

        fig.update_layout(
            title=title,
            height=height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
        )

        return fig

    def plot_trade_distribution(
        self,
        title: str = "Trade P&L Distribution",
        bins: int = 30,
        height: int = 500,
    ) -> 'go.Figure':
        """
        Create histogram of trade P&L distribution.

        Shows distribution of winning and losing trades with
        statistics overlay.

        Args:
            title: Chart title
            bins: Number of histogram bins
            height: Chart height in pixels

        Returns:
            Plotly Figure object
        """
        _require_plotly()

        if self.trade_log is None:
            raise ValueError("No trade log data available")

        trades = self.trade_log.get_trades()
        if not trades:
            raise ValueError("Trade log is empty")

        pnls = [t.net_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        fig = go.Figure()

        # Winning trades
        if wins:
            fig.add_trace(
                go.Histogram(
                    x=wins,
                    name=f"Winners ({len(wins)})",
                    marker_color=self.colors['win'],
                    opacity=0.7,
                    nbinsx=bins,
                    hovertemplate="P&L: $%{x:.2f}<br>Count: %{y}<extra></extra>",
                )
            )

        # Losing trades
        if losses:
            fig.add_trace(
                go.Histogram(
                    x=losses,
                    name=f"Losers ({len(losses)})",
                    marker_color=self.colors['loss'],
                    opacity=0.7,
                    nbinsx=bins,
                    hovertemplate="P&L: $%{x:.2f}<br>Count: %{y}<extra></extra>",
                )
            )

        # Add statistics annotations
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = (len(wins) / len(pnls)) * 100 if pnls else 0

        stats_text = (
            f"Win Rate: {win_rate:.1f}%<br>"
            f"Avg Win: ${avg_win:.2f}<br>"
            f"Avg Loss: ${avg_loss:.2f}<br>"
            f"Total Trades: {len(pnls)}"
        )

        fig.add_annotation(
            x=0.98,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=title,
            xaxis_title="Trade P&L ($)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=height,
            showlegend=True,
        )

        return fig

    def plot_drawdown_analysis(
        self,
        title: str = "Drawdown Analysis",
        top_n: int = 5,
        height: int = 600,
    ) -> 'go.Figure':
        """
        Create drawdown visualization with duration markers.

        Shows:
        - Drawdown over time
        - Top N drawdown periods highlighted
        - Duration and recovery annotations

        Args:
            title: Chart title
            top_n: Number of largest drawdowns to highlight
            height: Chart height in pixels

        Returns:
            Plotly Figure object
        """
        _require_plotly()

        if self.equity_curve is None:
            raise ValueError("No equity curve data available")

        points = self.equity_curve.get_points()
        if not points:
            raise ValueError("Equity curve is empty")

        timestamps = [p.timestamp for p in points]
        drawdown_pct = [p.drawdown_pct * 100 for p in points]

        # Identify drawdown periods
        periods = identify_drawdown_periods(points, min_drawdown_pct=0.01)

        # Sort by max drawdown and take top N
        top_periods = sorted(
            periods,
            key=lambda p: p.max_drawdown_pct,
            reverse=True
        )[:top_n]

        fig = go.Figure()

        # Main drawdown trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[-d for d in drawdown_pct],  # Invert for visual
                name="Drawdown",
                fill='tozeroy',
                line=dict(color=self.colors['drawdown'], width=1),
                fillcolor='rgba(244, 67, 54, 0.3)',
                hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
            )
        )

        # Highlight top drawdown periods
        colors = ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3']

        for i, period in enumerate(top_periods):
            color = colors[i % len(colors)]

            # Add vertical span for drawdown period
            fig.add_vrect(
                x0=period.start_time,
                x1=period.trough_time,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
            )

            # Add annotation for this drawdown
            duration_text = f"#{i+1}: {period.max_drawdown_pct*100:.1f}%"
            if period.duration_bars > 0:
                duration_text += f" ({period.duration_bars} bars)"

            fig.add_annotation(
                x=period.trough_time,
                y=-period.max_drawdown_pct * 100,
                text=duration_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                font=dict(size=10, color=color),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
            )

        # Create summary table
        if top_periods:
            summary_text = "<b>Top Drawdowns</b><br>"
            for i, p in enumerate(top_periods):
                recovery = f"{p.recovery_bars}b" if p.recovery_bars else "ongoing"
                summary_text += f"#{i+1}: {p.max_drawdown_pct*100:.1f}% | {p.duration_bars}b to trough | {recovery}<br>"

            fig.add_annotation(
                x=0.02,
                y=0.02,
                xref="paper",
                yref="paper",
                text=summary_text,
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#cccccc",
                borderwidth=1,
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=height,
            showlegend=True,
        )

        return fig

    def plot_time_of_day_analysis(
        self,
        title: str = "Performance by Hour",
        height: int = 400,
    ) -> 'go.Figure':
        """
        Create bar chart of P&L by hour of day.

        Useful for identifying best/worst trading hours.

        Args:
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly Figure object
        """
        _require_plotly()

        if self.trade_log is None:
            raise ValueError("No trade log data available")

        trades = self.trade_log.get_trades()
        if not trades:
            raise ValueError("Trade log is empty")

        # Group by hour
        hourly_pnl: Dict[int, List[float]] = {h: [] for h in range(24)}

        for trade in trades:
            hour = trade.entry_time.hour
            hourly_pnl[hour].append(trade.net_pnl)

        # Calculate stats per hour
        hours = list(range(24))
        avg_pnl = []
        trade_counts = []

        for h in hours:
            pnls = hourly_pnl[h]
            avg_pnl.append(np.mean(pnls) if pnls else 0)
            trade_counts.append(len(pnls))

        # Colors based on positive/negative
        bar_colors = [
            self.colors['win'] if p > 0 else self.colors['loss']
            for p in avg_pnl
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=hours,
                y=avg_pnl,
                marker_color=bar_colors,
                text=[f"n={c}" for c in trade_counts],
                textposition='outside',
                hovertemplate=(
                    "<b>Hour %{x}:00</b><br>"
                    "Avg P&L: $%{y:.2f}<br>"
                    "Trades: %{text}<extra></extra>"
                ),
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Average P&L ($)",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            height=height,
        )

        return fig

    def create_metrics_table(
        self,
        title: str = "Performance Metrics",
    ) -> 'go.Figure':
        """
        Create formatted table of key metrics.

        Args:
            title: Table title

        Returns:
            Plotly Figure with table
        """
        _require_plotly()

        if self.metrics is None:
            raise ValueError("No metrics data available")

        m = self.metrics

        # Key metrics to display
        headers = ["Category", "Metric", "Value"]

        cells = [
            # Returns
            ["Returns", "Total Return", f"{m.total_return_pct*100:.2f}%"],
            ["Returns", "Total P&L", f"${m.total_return_dollars:,.2f}"],
            ["Returns", "CAGR", f"{m.cagr_pct*100:.2f}%"],

            # Risk
            ["Risk", "Sharpe Ratio", f"{m.sharpe_ratio:.2f}"],
            ["Risk", "Sortino Ratio", f"{m.sortino_ratio:.2f}"],
            ["Risk", "Calmar Ratio", f"{m.calmar_ratio:.2f}"],

            # Drawdown
            ["Drawdown", "Max Drawdown", f"{m.max_drawdown_pct*100:.2f}%"],
            ["Drawdown", "Max DD ($)", f"${m.max_drawdown_dollars:,.2f}"],
            ["Drawdown", "Max DD Duration", f"{m.max_drawdown_duration_days} days"],

            # Trades
            ["Trades", "Total Trades", f"{m.total_trades}"],
            ["Trades", "Win Rate", f"{m.win_rate_pct:.1f}%"],
            ["Trades", "Profit Factor", f"{m.profit_factor:.2f}"],
            ["Trades", "Avg Trade", f"${m.avg_trade_pnl:.2f}"],
            ["Trades", "Expectancy", f"${m.expectancy:.2f}"],

            # Best/Worst
            ["Extremes", "Largest Win", f"${m.largest_win:.2f}"],
            ["Extremes", "Largest Loss", f"${m.largest_loss:.2f}"],
            ["Extremes", "Max Consec Wins", f"{m.max_consecutive_wins}"],
            ["Extremes", "Max Consec Losses", f"{m.max_consecutive_losses}"],

            # Costs
            ["Costs", "Total Commission", f"${m.total_commission:.2f}"],
            ["Costs", "Total Slippage", f"${m.total_slippage:.2f}"],
            ["Costs", "Cost/Trade", f"${m.cost_per_trade:.2f}"],
        ]

        # Transpose for table format
        categories = [c[0] for c in cells]
        metrics_names = [c[1] for c in cells]
        values = [c[2] for c in cells]

        # Color code by category
        category_colors = {
            "Returns": "rgb(227, 242, 253)",
            "Risk": "rgb(232, 245, 233)",
            "Drawdown": "rgb(255, 243, 224)",
            "Trades": "rgb(243, 229, 245)",
            "Extremes": "rgb(255, 235, 238)",
            "Costs": "rgb(236, 239, 241)",
        }

        cell_colors = [
            [category_colors.get(cat, "white") for cat in categories],
            ["white"] * len(categories),
            ["white"] * len(categories),
        ]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='rgb(66, 66, 66)',
                font=dict(color='white', size=12),
                align='left',
            ),
            cells=dict(
                values=[categories, metrics_names, values],
                fill_color=cell_colors,
                align=['left', 'left', 'right'],
                font=dict(size=11),
                height=25,
            ),
        )])

        fig.update_layout(
            title=title,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        return fig

    def create_dashboard(
        self,
        title: str = "Backtest Dashboard",
        height: int = 1200,
    ) -> 'go.Figure':
        """
        Create comprehensive dashboard with all visualizations.

        Layout:
        - Row 1: Equity curve with drawdown
        - Row 2: Trade distribution | Time of day analysis
        - Row 3: Metrics summary

        Args:
            title: Dashboard title
            height: Total dashboard height

        Returns:
            Plotly Figure with all subplots
        """
        _require_plotly()

        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.45, 0.30, 0.25],
            column_widths=[0.5, 0.5],
            specs=[
                [{"colspan": 2, "secondary_y": True}, None],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "table", "colspan": 2}, None],
            ],
            subplot_titles=(
                "Equity Curve",
                "Trade P&L Distribution", "P&L by Hour",
                "Performance Metrics",
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        # Row 1: Equity Curve
        if self.equity_curve is not None:
            points = self.equity_curve.get_points()
            if points:
                timestamps = [p.timestamp for p in points]
                equity_values = [p.equity for p in points]
                drawdown_values = [p.drawdown_pct * 100 for p in points]
                hwm = np.maximum.accumulate(equity_values)

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=equity_values,
                        name="Equity",
                        line=dict(color=self.colors['equity'], width=2),
                    ),
                    row=1, col=1, secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=hwm.tolist(),
                        name="High Water Mark",
                        line=dict(color=self.colors['equity'], width=1, dash='dot'),
                        opacity=0.5,
                    ),
                    row=1, col=1, secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=[-d for d in drawdown_values],
                        name="Drawdown %",
                        fill='tozeroy',
                        line=dict(color=self.colors['drawdown'], width=1),
                        fillcolor='rgba(244, 67, 54, 0.3)',
                    ),
                    row=1, col=1, secondary_y=True,
                )

        # Row 2 Left: Trade Distribution
        if self.trade_log is not None:
            trades = self.trade_log.get_trades()
            if trades:
                wins = [t.net_pnl for t in trades if t.net_pnl > 0]
                losses = [t.net_pnl for t in trades if t.net_pnl <= 0]

                if wins:
                    fig.add_trace(
                        go.Histogram(
                            x=wins,
                            name=f"Winners ({len(wins)})",
                            marker_color=self.colors['win'],
                            opacity=0.7,
                        ),
                        row=2, col=1,
                    )

                if losses:
                    fig.add_trace(
                        go.Histogram(
                            x=losses,
                            name=f"Losers ({len(losses)})",
                            marker_color=self.colors['loss'],
                            opacity=0.7,
                        ),
                        row=2, col=1,
                    )

        # Row 2 Right: Time of Day
        if self.trade_log is not None:
            trades = self.trade_log.get_trades()
            if trades:
                hourly_pnl: Dict[int, List[float]] = {h: [] for h in range(24)}
                for trade in trades:
                    hour = trade.entry_time.hour
                    hourly_pnl[hour].append(trade.net_pnl)

                hours = list(range(24))
                avg_pnl = [
                    np.mean(hourly_pnl[h]) if hourly_pnl[h] else 0
                    for h in hours
                ]

                bar_colors = [
                    self.colors['win'] if p > 0 else self.colors['loss']
                    for p in avg_pnl
                ]

                fig.add_trace(
                    go.Bar(
                        x=hours,
                        y=avg_pnl,
                        name="Avg P&L by Hour",
                        marker_color=bar_colors,
                    ),
                    row=2, col=2,
                )

        # Row 3: Metrics Table
        if self.metrics is not None:
            m = self.metrics

            # Key metrics for summary
            metrics_data = [
                ["Return", f"{m.total_return_pct*100:.1f}%", "Sharpe", f"{m.sharpe_ratio:.2f}"],
                ["Win Rate", f"{m.win_rate_pct:.1f}%", "Sortino", f"{m.sortino_ratio:.2f}"],
                ["Profit Factor", f"{m.profit_factor:.2f}", "Calmar", f"{m.calmar_ratio:.2f}"],
                ["Avg Trade", f"${m.avg_trade_pnl:.2f}", "Max DD", f"{m.max_drawdown_pct*100:.1f}%"],
                ["Total Trades", f"{m.total_trades}", "Expectancy", f"${m.expectancy:.2f}"],
            ]

            col1 = [row[0] for row in metrics_data]
            col2 = [row[1] for row in metrics_data]
            col3 = [row[2] for row in metrics_data]
            col4 = [row[3] for row in metrics_data]

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Metric", "Value", "Metric", "Value"],
                        fill_color='rgb(66, 66, 66)',
                        font=dict(color='white', size=11),
                        align='center',
                    ),
                    cells=dict(
                        values=[col1, col2, col3, col4],
                        fill_color=[
                            ['rgb(227, 242, 253)'] * 5,
                            ['white'] * 5,
                            ['rgb(232, 245, 233)'] * 5,
                            ['white'] * 5,
                        ],
                        align=['left', 'right', 'left', 'right'],
                        font=dict(size=11),
                        height=24,
                    ),
                ),
                row=3, col=1,
            )

        # Update axes labels
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1, secondary_y=True)
        fig.update_xaxes(title_text="P&L ($)", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=2, col=2)

        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            height=height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='overlay',
        )

        return fig


class WalkForwardVisualizer:
    """
    Visualizations for walk-forward validation results.

    Specialized charts for analyzing multiple fold results
    and detecting overfitting.
    """

    def __init__(
        self,
        fold_reports: List[BacktestReport],
        in_sample_metrics: Optional[List[PerformanceMetrics]] = None,
    ):
        """
        Initialize with fold results.

        Args:
            fold_reports: List of BacktestReport for each OOS fold
            in_sample_metrics: Optional list of IS metrics for comparison
        """
        _require_plotly()

        self.fold_reports = fold_reports
        self.in_sample_metrics = in_sample_metrics

        self.colors = {
            'oos': '#2196F3',  # Blue
            'is': '#FF9800',  # Orange
            'equity': '#4CAF50',  # Green
        }

    def plot_fold_comparison(
        self,
        title: str = "Walk-Forward Fold Comparison",
        height: int = 500,
    ) -> 'go.Figure':
        """
        Create bar chart comparing metrics across folds.

        Shows Sharpe ratio, return, and win rate for each fold
        to identify consistency and potential overfitting.

        Args:
            title: Chart title
            height: Chart height

        Returns:
            Plotly Figure
        """
        _require_plotly()

        if not self.fold_reports:
            raise ValueError("No fold reports available")

        fold_nums = list(range(1, len(self.fold_reports) + 1))

        # Extract metrics
        sharpe_oos = []
        returns_oos = []
        win_rates_oos = []

        for report in self.fold_reports:
            if report.metrics:
                sharpe_oos.append(report.metrics.sharpe_ratio)
                returns_oos.append(report.metrics.total_return_pct * 100)
                win_rates_oos.append(report.metrics.win_rate_pct)
            else:
                sharpe_oos.append(0)
                returns_oos.append(0)
                win_rates_oos.append(0)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Sharpe Ratio", "Return (%)", "Win Rate (%)"),
        )

        # Sharpe Ratio
        fig.add_trace(
            go.Bar(
                x=fold_nums,
                y=sharpe_oos,
                name="OOS Sharpe",
                marker_color=self.colors['oos'],
            ),
            row=1, col=1,
        )

        # Add IS comparison if available
        if self.in_sample_metrics:
            sharpe_is = [m.sharpe_ratio if m else 0 for m in self.in_sample_metrics]
            fig.add_trace(
                go.Bar(
                    x=fold_nums,
                    y=sharpe_is,
                    name="IS Sharpe",
                    marker_color=self.colors['is'],
                    opacity=0.5,
                ),
                row=1, col=1,
            )

        # Returns
        fig.add_trace(
            go.Bar(
                x=fold_nums,
                y=returns_oos,
                name="OOS Return",
                marker_color=self.colors['oos'],
                showlegend=False,
            ),
            row=1, col=2,
        )

        if self.in_sample_metrics:
            returns_is = [
                m.total_return_pct * 100 if m else 0
                for m in self.in_sample_metrics
            ]
            fig.add_trace(
                go.Bar(
                    x=fold_nums,
                    y=returns_is,
                    name="IS Return",
                    marker_color=self.colors['is'],
                    opacity=0.5,
                    showlegend=False,
                ),
                row=1, col=2,
            )

        # Win Rate
        fig.add_trace(
            go.Bar(
                x=fold_nums,
                y=win_rates_oos,
                name="OOS Win Rate",
                marker_color=self.colors['oos'],
                showlegend=False,
            ),
            row=1, col=3,
        )

        if self.in_sample_metrics:
            win_rates_is = [
                m.win_rate_pct if m else 0
                for m in self.in_sample_metrics
            ]
            fig.add_trace(
                go.Bar(
                    x=fold_nums,
                    y=win_rates_is,
                    name="IS Win Rate",
                    marker_color=self.colors['is'],
                    opacity=0.5,
                    showlegend=False,
                ),
                row=1, col=3,
            )

        # Add reference lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=3)

        fig.update_xaxes(title_text="Fold", tickmode='linear')
        fig.update_layout(
            title=title,
            height=height,
            showlegend=True,
            barmode='group',
        )

        return fig

    def plot_combined_equity(
        self,
        title: str = "Walk-Forward Combined Equity",
        height: int = 500,
    ) -> 'go.Figure':
        """
        Create stitched equity curve from all OOS folds.

        This shows what the strategy would have returned
        trading only on out-of-sample data.

        Args:
            title: Chart title
            height: Chart height

        Returns:
            Plotly Figure
        """
        _require_plotly()

        if not self.fold_reports:
            raise ValueError("No fold reports available")

        # Stitch equity curves
        combined_timestamps = []
        combined_equity = []
        fold_boundaries = []

        current_equity = None

        for i, report in enumerate(self.fold_reports):
            if report.equity_curve is None:
                continue

            points = report.equity_curve.get_points()
            if not points:
                continue

            if current_equity is None:
                # First fold - use actual values
                current_equity = points[0].equity

            # Calculate equity multiplier to chain from previous fold
            start_equity = points[0].equity
            multiplier = current_equity / start_equity if start_equity > 0 else 1

            # Record fold boundary
            fold_boundaries.append((points[0].timestamp, i + 1))

            for point in points:
                combined_timestamps.append(point.timestamp)
                adjusted_equity = point.equity * multiplier
                combined_equity.append(adjusted_equity)

            # Update current equity for next fold
            current_equity = combined_equity[-1] if combined_equity else current_equity

        if not combined_timestamps:
            raise ValueError("No equity data in fold reports")

        # Calculate combined metrics
        equity_array = np.array(combined_equity)
        hwm = np.maximum.accumulate(equity_array)
        drawdown = (hwm - equity_array) / hwm * 100

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Combined OOS Equity", "Drawdown (%)"),
        )

        # Equity
        fig.add_trace(
            go.Scatter(
                x=combined_timestamps,
                y=combined_equity,
                name="Combined OOS Equity",
                line=dict(color=self.colors['equity'], width=2),
            ),
            row=1, col=1,
        )

        # High water mark
        fig.add_trace(
            go.Scatter(
                x=combined_timestamps,
                y=hwm.tolist(),
                name="High Water Mark",
                line=dict(color=self.colors['equity'], width=1, dash='dot'),
                opacity=0.5,
            ),
            row=1, col=1,
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=combined_timestamps,
                y=(-drawdown).tolist(),
                name="Drawdown",
                fill='tozeroy',
                line=dict(color='red', width=1),
                fillcolor='rgba(244, 67, 54, 0.3)',
            ),
            row=2, col=1,
        )

        # Add fold boundary markers
        for ts, fold_num in fold_boundaries:
            # Use shape instead of vline to avoid datetime annotation issues
            fig.add_shape(
                type="line",
                x0=ts,
                x1=ts,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=1, dash="dash"),
                opacity=0.5,
            )
            # Add annotation separately
            fig.add_annotation(
                x=ts,
                y=1.05,
                yref="paper",
                text=f"Fold {fold_num}",
                showarrow=False,
                font=dict(size=10, color="gray"),
            )

        # Summary stats
        total_return = (combined_equity[-1] / combined_equity[0] - 1) * 100
        max_dd = float(np.max(drawdown))

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=(
                f"<b>Combined OOS Results</b><br>"
                f"Return: {total_return:.1f}%<br>"
                f"Max DD: {max_dd:.1f}%<br>"
                f"Folds: {len(self.fold_reports)}"
            ),
            showarrow=False,
            font=dict(size=11),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.update_layout(
            title=title,
            height=height,
            showlegend=True,
        )

        return fig

    def plot_overfitting_analysis(
        self,
        title: str = "Overfitting Analysis",
        height: int = 400,
    ) -> 'go.Figure':
        """
        Create scatter plot comparing IS vs OOS metrics.

        Points on the diagonal indicate no overfitting.
        Points below the diagonal indicate overfitting.

        Args:
            title: Chart title
            height: Chart height

        Returns:
            Plotly Figure
        """
        _require_plotly()

        if not self.in_sample_metrics:
            raise ValueError("In-sample metrics required for overfitting analysis")

        if len(self.in_sample_metrics) != len(self.fold_reports):
            raise ValueError("IS and OOS metric counts must match")

        sharpe_is = []
        sharpe_oos = []

        for is_m, report in zip(self.in_sample_metrics, self.fold_reports):
            if is_m and report.metrics:
                sharpe_is.append(is_m.sharpe_ratio)
                sharpe_oos.append(report.metrics.sharpe_ratio)

        if not sharpe_is:
            raise ValueError("No valid metric pairs found")

        fig = go.Figure()

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=sharpe_is,
                y=sharpe_oos,
                mode='markers+text',
                name='Folds',
                marker=dict(size=12, color=self.colors['oos']),
                text=[f"F{i+1}" for i in range(len(sharpe_is))],
                textposition='top center',
                hovertemplate=(
                    "<b>Fold %{text}</b><br>"
                    "IS Sharpe: %{x:.2f}<br>"
                    "OOS Sharpe: %{y:.2f}<extra></extra>"
                ),
            )
        )

        # Diagonal line (no overfitting)
        max_val = max(max(sharpe_is), max(sharpe_oos)) * 1.2
        min_val = min(min(sharpe_is), min(sharpe_oos)) * 0.8

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='No Overfit Line',
                line=dict(color='gray', dash='dash'),
            )
        )

        # Reference lines
        fig.add_hline(y=0, line_color="lightgray", opacity=0.5)
        fig.add_vline(x=0, line_color="lightgray", opacity=0.5)

        # Calculate average degradation
        degradation = [oos - is_ for is_, oos in zip(sharpe_is, sharpe_oos)]
        avg_degradation = np.mean(degradation)

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=(
                f"<b>Overfitting Analysis</b><br>"
                f"Avg Sharpe Degradation: {avg_degradation:.2f}<br>"
                f"{'⚠️ Potential overfitting' if avg_degradation < -0.5 else '✓ Acceptable'}"
            ),
            showarrow=False,
            font=dict(size=11),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
        )

        fig.update_layout(
            title=title,
            xaxis_title="In-Sample Sharpe Ratio",
            yaxis_title="Out-of-Sample Sharpe Ratio",
            height=height,
            showlegend=True,
        )

        return fig


def export_visualization(
    fig: 'go.Figure',
    filepath: str,
    format: str = 'html',
    **kwargs,
) -> str:
    """
    Export Plotly figure to file.

    Args:
        fig: Plotly Figure object
        filepath: Output path (extension will be added if missing)
        format: 'html', 'png', 'json'
        **kwargs: Additional args for write methods

    Returns:
        Path to created file
    """
    _require_plotly()

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'html':
        if not path.suffix:
            path = path.with_suffix('.html')
        fig.write_html(str(path), **kwargs)
    elif format == 'png':
        if not path.suffix:
            path = path.with_suffix('.png')
        fig.write_image(str(path), **kwargs)
    elif format == 'json':
        if not path.suffix:
            path = path.with_suffix('.json')
        fig.write_json(str(path), **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")

    return str(path)
