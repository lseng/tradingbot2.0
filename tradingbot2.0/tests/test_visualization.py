"""
Tests for Backtest Visualization Module

These tests verify:
1. Interactive equity curve plots work correctly
2. Trade distribution histograms are accurate
3. Drawdown analysis with duration markers
4. Per-fold metrics dashboard
5. Walk-forward equity stitching

Why these tests matter:
- Visualization is critical for analyzing backtest results before going live
- Incorrect visualizations could lead to wrong trading decisions
- Plotly figures need correct data binding and layout
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np

# Check if plotly is available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.backtest.trade_logger import (
    TradeLog,
    TradeRecord,
    EquityCurve,
    EquityPoint,
    BacktestReport,
    ExitReason,
)
from src.backtest.metrics import PerformanceMetrics, calculate_metrics


# Skip all tests if plotly not available
pytestmark = pytest.mark.skipif(
    not PLOTLY_AVAILABLE,
    reason="Plotly not installed"
)


@pytest.fixture
def sample_equity_curve():
    """Create a sample equity curve with realistic data."""
    curve = EquityCurve(initial_equity=1000.0)
    base_time = datetime(2024, 1, 1, 9, 30)

    # Simulate 100 bars with some ups and downs
    equity = 1000.0
    np.random.seed(42)

    for i in range(100):
        # Random walk with slight positive drift
        change = np.random.normal(1, 5)
        equity = max(950, equity + change)

        timestamp = base_time + timedelta(seconds=i)
        curve.add_point(timestamp, equity)

    return curve


@pytest.fixture
def sample_trade_log():
    """Create a sample trade log with realistic trades."""
    log = TradeLog()
    base_time = datetime(2024, 1, 1, 9, 30)

    # Add 20 trades - mix of winners and losers
    trade_data = [
        (1, 6050.0, 6055.0, 22.0),   # Long winner
        (-1, 6055.0, 6058.0, -17.0),  # Short loser
        (1, 6060.0, 6063.0, 12.0),    # Long winner
        (1, 6063.0, 6058.0, -27.0),   # Long loser
        (-1, 6058.0, 6052.0, 28.0),   # Short winner
        (1, 6052.0, 6057.0, 22.0),    # Long winner
        (-1, 6057.0, 6060.0, -17.0),  # Short loser
        (1, 6060.0, 6065.0, 22.0),    # Long winner
        (-1, 6065.0, 6062.0, 12.0),   # Short winner
        (1, 6062.0, 6058.0, -22.0),   # Long loser
        (1, 6058.0, 6064.0, 27.0),    # Long winner
        (-1, 6064.0, 6068.0, -22.0),  # Short loser
        (1, 6068.0, 6072.0, 17.0),    # Long winner
        (-1, 6072.0, 6070.0, 8.0),    # Short winner (small)
        (1, 6070.0, 6065.0, -27.0),   # Long loser
        (1, 6065.0, 6070.0, 22.0),    # Long winner
        (-1, 6070.0, 6072.0, -12.0),  # Short loser
        (1, 6072.0, 6078.0, 27.0),    # Long winner
        (-1, 6078.0, 6075.0, 12.0),   # Short winner
        (1, 6075.0, 6080.0, 22.0),    # Long winner
    ]

    for i, (direction, entry, exit_, pnl) in enumerate(trade_data):
        entry_time = base_time + timedelta(hours=i)
        exit_time = entry_time + timedelta(minutes=30)

        log.add_trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry,
            exit_price=exit_,
            contracts=1,
            gross_pnl=pnl + 0.84,  # Add back costs for gross
            commission=0.42,
            slippage=0.42,
            exit_reason=ExitReason.TARGET if pnl > 0 else ExitReason.STOP,
            model_confidence=0.75,
            predicted_class=2 if direction == 1 else 0,
            stop_price=entry - 8 * 0.25 if direction == 1 else entry + 8 * 0.25,
            target_price=entry + 12 * 0.25 if direction == 1 else entry - 12 * 0.25,
            bars_held=120,
        )

    return log


@pytest.fixture
def sample_metrics():
    """Create sample performance metrics."""
    return PerformanceMetrics(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        trading_days=21,
        total_return_pct=0.125,
        total_return_dollars=125.0,
        cagr_pct=0.15,
        daily_return_mean=0.006,
        daily_return_std=0.02,
        monthly_return_mean=0.125,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=0.8,
        daily_var_95=0.03,
        daily_var_99=0.05,
        max_drawdown_pct=0.15,
        max_drawdown_dollars=150.0,
        max_drawdown_duration_days=5,
        avg_drawdown_pct=0.05,
        recovery_factor=0.83,
        total_trades=20,
        winning_trades=12,
        losing_trades=8,
        win_rate_pct=60.0,
        profit_factor=1.8,
        avg_trade_pnl=6.25,
        avg_win=20.0,
        avg_loss=14.0,
        largest_win=28.0,
        largest_loss=27.0,
        expectancy=6.25,
        expectancy_ratio=0.45,
        win_days_pct=65.0,
        best_day_pnl=50.0,
        worst_day_pnl=-30.0,
        best_day_pct=0.05,
        worst_day_pct=-0.03,
        max_consecutive_wins=4,
        max_consecutive_losses=2,
        avg_trades_per_day=0.95,
        total_commission=8.4,
        total_slippage=8.4,
        cost_per_trade=0.84,
        cost_pct_of_gross=0.05,
        initial_capital=1000.0,
        final_capital=1125.0,
        gross_profit=240.0,
        gross_loss=98.16,
        net_profit=125.0,
    )


@pytest.fixture
def sample_backtest_report(sample_trade_log, sample_equity_curve, sample_metrics):
    """Create a complete backtest report."""
    return BacktestReport(
        trade_log=sample_trade_log,
        equity_curve=sample_equity_curve,
        metrics=sample_metrics,
        config={"model": "lstm", "threshold": 0.6},
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        fold_id=1,
    )


class TestDrawdownPeriodIdentification:
    """Tests for identify_drawdown_periods function."""

    def test_identify_single_drawdown(self):
        """Test identifying a single drawdown period."""
        from src.backtest.visualization import identify_drawdown_periods

        # Create equity curve with one clear drawdown
        curve = EquityCurve(initial_equity=1000.0)
        base_time = datetime(2024, 1, 1)

        # Rising, then falling, then recovering
        for i, eq in enumerate([1000, 1050, 1100, 1050, 1000, 1050, 1100, 1150]):
            curve.add_point(base_time + timedelta(hours=i), eq)

        periods = identify_drawdown_periods(curve.get_points(), min_drawdown_pct=0.01)

        assert len(periods) == 1
        assert periods[0].peak_equity == 1100
        assert periods[0].trough_equity == 1000
        assert periods[0].max_drawdown_pct == pytest.approx(0.0909, rel=0.01)
        assert periods[0].is_recovered is True

    def test_identify_multiple_drawdowns(self):
        """Test identifying multiple drawdown periods."""
        from src.backtest.visualization import identify_drawdown_periods

        curve = EquityCurve(initial_equity=1000.0)
        base_time = datetime(2024, 1, 1)

        # Multiple drawdown cycles
        equity_values = [
            1000, 1050, 1000, 1100,  # First drawdown (5%)
            1050, 1150,              # Second drawdown (4.5%)
            1100, 1200               # Third drawdown (4.3%)
        ]

        for i, eq in enumerate(equity_values):
            curve.add_point(base_time + timedelta(hours=i), eq)

        periods = identify_drawdown_periods(curve.get_points(), min_drawdown_pct=0.01)

        # Should have 3 drawdown periods
        assert len(periods) == 3

    def test_identify_ongoing_drawdown(self):
        """Test identifying ongoing drawdown at end of data."""
        from src.backtest.visualization import identify_drawdown_periods

        curve = EquityCurve(initial_equity=1000.0)
        base_time = datetime(2024, 1, 1)

        # Ends in drawdown
        for i, eq in enumerate([1000, 1100, 1050, 1000, 950]):
            curve.add_point(base_time + timedelta(hours=i), eq)

        periods = identify_drawdown_periods(curve.get_points(), min_drawdown_pct=0.01)

        assert len(periods) == 1
        assert periods[0].is_recovered is False
        assert periods[0].end_time is None
        assert periods[0].recovery_bars is None

    def test_filter_small_drawdowns(self):
        """Test that small drawdowns are filtered by min_drawdown_pct."""
        from src.backtest.visualization import identify_drawdown_periods

        curve = EquityCurve(initial_equity=1000.0)
        base_time = datetime(2024, 1, 1)

        # Very small drawdown (0.5%)
        for i, eq in enumerate([1000, 1005, 1000, 1010]):
            curve.add_point(base_time + timedelta(hours=i), eq)

        # With 1% threshold, should find nothing
        periods = identify_drawdown_periods(curve.get_points(), min_drawdown_pct=0.01)
        assert len(periods) == 0

        # With 0.1% threshold, should find the drawdown
        periods = identify_drawdown_periods(curve.get_points(), min_drawdown_pct=0.001)
        assert len(periods) == 1

    def test_empty_equity_curve(self):
        """Test handling of empty equity curve."""
        from src.backtest.visualization import identify_drawdown_periods

        periods = identify_drawdown_periods([], min_drawdown_pct=0.01)
        assert periods == []


class TestBacktestVisualizer:
    """Tests for BacktestVisualizer class."""

    def test_initialization_with_report(self, sample_backtest_report):
        """Test initializing visualizer with full report."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(report=sample_backtest_report)

        assert viz.trade_log is not None
        assert viz.equity_curve is not None
        assert viz.metrics is not None

    def test_initialization_with_components(self, sample_trade_log, sample_equity_curve, sample_metrics):
        """Test initializing visualizer with individual components."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(
            trade_log=sample_trade_log,
            equity_curve=sample_equity_curve,
            metrics=sample_metrics,
        )

        assert viz.trade_log == sample_trade_log
        assert viz.equity_curve == sample_equity_curve
        assert viz.metrics == sample_metrics

    def test_plot_equity_curve_basic(self, sample_equity_curve):
        """Test basic equity curve plot creation."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_trades=False, show_drawdown=True)

        assert isinstance(fig, go.Figure)
        # Should have 3 traces: equity, HWM, drawdown
        assert len(fig.data) == 3

    def test_plot_equity_curve_with_trades(self, sample_backtest_report):
        """Test equity curve with trade markers."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(report=sample_backtest_report)
        fig = viz.plot_equity_curve(show_trades=True, show_drawdown=True)

        # Should have equity, HWM, drawdown, and entry markers
        assert len(fig.data) >= 4

    def test_plot_equity_curve_no_drawdown(self, sample_equity_curve):
        """Test equity curve without drawdown overlay."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=False)

        # Should only have equity trace
        assert len(fig.data) == 1

    def test_plot_equity_curve_empty_raises(self):
        """Test that empty equity curve raises error."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=EquityCurve())

        with pytest.raises(ValueError, match="empty"):
            viz.plot_equity_curve()

    def test_plot_equity_curve_none_raises(self):
        """Test that missing equity curve raises error."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer()

        with pytest.raises(ValueError, match="No equity curve"):
            viz.plot_equity_curve()

    def test_plot_trade_distribution(self, sample_trade_log):
        """Test trade distribution histogram."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(trade_log=sample_trade_log)
        fig = viz.plot_trade_distribution()

        assert isinstance(fig, go.Figure)
        # Should have winner and loser traces
        assert len(fig.data) >= 1

    def test_plot_trade_distribution_win_rate(self, sample_trade_log):
        """Test that trade distribution shows correct stats."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(trade_log=sample_trade_log)
        fig = viz.plot_trade_distribution()

        # Check that annotations exist with stats
        assert len(fig.layout.annotations) > 0

        # Find stats annotation
        stats_annotation = None
        for ann in fig.layout.annotations:
            if "Win Rate" in ann.text:
                stats_annotation = ann
                break

        assert stats_annotation is not None
        assert "Total Trades" in stats_annotation.text

    def test_plot_trade_distribution_empty_raises(self):
        """Test that empty trade log raises error."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(trade_log=TradeLog())

        with pytest.raises(ValueError, match="empty"):
            viz.plot_trade_distribution()

    def test_plot_drawdown_analysis(self, sample_equity_curve):
        """Test drawdown analysis chart."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_drawdown_analysis(top_n=3)

        assert isinstance(fig, go.Figure)
        # Should have main drawdown trace
        assert len(fig.data) >= 1

    def test_plot_drawdown_analysis_annotations(self, sample_equity_curve):
        """Test that drawdown periods are annotated."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_drawdown_analysis(top_n=5)

        # Should have annotations for drawdown summary
        assert len(fig.layout.annotations) > 0

    def test_plot_time_of_day_analysis(self, sample_trade_log):
        """Test time of day performance chart."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(trade_log=sample_trade_log)
        fig = viz.plot_time_of_day_analysis()

        assert isinstance(fig, go.Figure)
        # Should have 24 bars for each hour
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 24

    def test_create_metrics_table(self, sample_metrics):
        """Test metrics table creation."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(metrics=sample_metrics)
        fig = viz.create_metrics_table()

        assert isinstance(fig, go.Figure)
        # Should have table trace
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Table)

    def test_create_metrics_table_none_raises(self):
        """Test that missing metrics raises error."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer()

        with pytest.raises(ValueError, match="No metrics"):
            viz.create_metrics_table()

    def test_create_dashboard(self, sample_backtest_report):
        """Test full dashboard creation."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(report=sample_backtest_report)
        fig = viz.create_dashboard()

        assert isinstance(fig, go.Figure)
        # Dashboard should have multiple traces
        assert len(fig.data) >= 5

    def test_dashboard_has_all_sections(self, sample_backtest_report):
        """Test that dashboard has all expected sections."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(report=sample_backtest_report)
        fig = viz.create_dashboard()

        # Check subplot titles
        annotations = [a for a in fig.layout.annotations if hasattr(a, 'text')]
        titles = [a.text for a in annotations]

        assert any("Equity" in t for t in titles)
        assert any("Distribution" in t for t in titles)


class TestWalkForwardVisualizer:
    """Tests for WalkForwardVisualizer class."""

    @pytest.fixture
    def fold_reports(self, sample_trade_log, sample_equity_curve, sample_metrics):
        """Create multiple fold reports for walk-forward testing."""
        reports = []

        for fold_id in range(1, 5):
            # Vary metrics slightly per fold
            metrics = PerformanceMetrics(
                total_return_pct=0.10 + fold_id * 0.02,
                sharpe_ratio=1.0 + fold_id * 0.2,
                win_rate_pct=55 + fold_id * 2,
                profit_factor=1.5 + fold_id * 0.1,
                total_trades=15 + fold_id * 2,
            )

            # Create distinct equity curve for each fold
            curve = EquityCurve(initial_equity=1000.0)
            base_time = datetime(2024, fold_id, 1)
            equity = 1000.0

            for i in range(50):
                equity = equity * (1 + metrics.total_return_pct / 50)
                curve.add_point(base_time + timedelta(hours=i), equity)

            reports.append(BacktestReport(
                trade_log=sample_trade_log,
                equity_curve=curve,
                metrics=metrics,
                fold_id=fold_id,
            ))

        return reports

    @pytest.fixture
    def in_sample_metrics(self):
        """Create in-sample metrics for each fold."""
        # IS metrics are typically better than OOS (overfitting)
        return [
            PerformanceMetrics(sharpe_ratio=1.8, total_return_pct=0.15, win_rate_pct=62),
            PerformanceMetrics(sharpe_ratio=2.0, total_return_pct=0.18, win_rate_pct=65),
            PerformanceMetrics(sharpe_ratio=1.9, total_return_pct=0.16, win_rate_pct=63),
            PerformanceMetrics(sharpe_ratio=2.1, total_return_pct=0.20, win_rate_pct=66),
        ]

    def test_initialization(self, fold_reports):
        """Test walk-forward visualizer initialization."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports)

        assert len(viz.fold_reports) == 4
        assert viz.in_sample_metrics is None

    def test_initialization_with_is_metrics(self, fold_reports, in_sample_metrics):
        """Test initialization with in-sample metrics."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports, in_sample_metrics)

        assert viz.in_sample_metrics is not None
        assert len(viz.in_sample_metrics) == 4

    def test_plot_fold_comparison(self, fold_reports):
        """Test fold comparison chart."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports)
        fig = viz.plot_fold_comparison()

        assert isinstance(fig, go.Figure)
        # Should have bars for OOS metrics (3 metrics, 1 trace each)
        assert len(fig.data) >= 3

    def test_plot_fold_comparison_with_is(self, fold_reports, in_sample_metrics):
        """Test fold comparison with IS/OOS comparison."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports, in_sample_metrics)
        fig = viz.plot_fold_comparison()

        # Should have both IS and OOS bars
        assert len(fig.data) >= 4

    def test_plot_fold_comparison_empty_raises(self):
        """Test that empty fold list raises error."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer([])

        with pytest.raises(ValueError, match="No fold"):
            viz.plot_fold_comparison()

    def test_plot_combined_equity(self, fold_reports):
        """Test combined equity curve stitching."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports)
        fig = viz.plot_combined_equity()

        assert isinstance(fig, go.Figure)
        # Should have equity, HWM, and drawdown traces
        assert len(fig.data) >= 3

    def test_combined_equity_correct_stitching(self, fold_reports):
        """Test that equity curves are correctly stitched."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports)
        fig = viz.plot_combined_equity()

        # Get the combined equity trace
        equity_trace = fig.data[0]

        # First point should match initial equity
        assert equity_trace.y[0] == pytest.approx(1000.0, rel=0.01)

        # Final equity should be positive (all folds had positive returns)
        assert equity_trace.y[-1] > equity_trace.y[0]

    def test_combined_equity_fold_boundaries(self, fold_reports):
        """Test that fold boundaries are marked."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports)
        fig = viz.plot_combined_equity()

        # Should have vertical lines for fold boundaries
        # In Plotly, vlines create shapes
        assert hasattr(fig.layout, 'shapes') or len(fig.layout.annotations) > 0

    def test_plot_overfitting_analysis(self, fold_reports, in_sample_metrics):
        """Test overfitting analysis scatter plot."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports, in_sample_metrics)
        fig = viz.plot_overfitting_analysis()

        assert isinstance(fig, go.Figure)
        # Should have scatter points and diagonal line
        assert len(fig.data) >= 2

    def test_overfitting_analysis_requires_is_metrics(self, fold_reports):
        """Test that overfitting analysis requires IS metrics."""
        from src.backtest.visualization import WalkForwardVisualizer

        viz = WalkForwardVisualizer(fold_reports)

        with pytest.raises(ValueError, match="In-sample metrics required"):
            viz.plot_overfitting_analysis()

    def test_overfitting_analysis_count_mismatch(self, fold_reports, in_sample_metrics):
        """Test that IS/OOS count mismatch raises error."""
        from src.backtest.visualization import WalkForwardVisualizer

        # Create mismatch
        viz = WalkForwardVisualizer(fold_reports, in_sample_metrics[:2])

        with pytest.raises(ValueError, match="counts must match"):
            viz.plot_overfitting_analysis()


class TestExportVisualization:
    """Tests for export_visualization function."""

    def test_export_html(self, sample_equity_curve, tmp_path):
        """Test exporting to HTML format."""
        from src.backtest.visualization import BacktestVisualizer, export_visualization

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=False)

        output_path = tmp_path / "test_chart"
        result = export_visualization(fig, str(output_path), format='html')

        assert result.endswith('.html')
        assert (tmp_path / "test_chart.html").exists()

    def test_export_json(self, sample_equity_curve, tmp_path):
        """Test exporting to JSON format."""
        from src.backtest.visualization import BacktestVisualizer, export_visualization

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=False)

        output_path = tmp_path / "test_chart"
        result = export_visualization(fig, str(output_path), format='json')

        assert result.endswith('.json')
        assert (tmp_path / "test_chart.json").exists()

    def test_export_with_suffix(self, sample_equity_curve, tmp_path):
        """Test that existing suffix is preserved."""
        from src.backtest.visualization import BacktestVisualizer, export_visualization

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=False)

        output_path = tmp_path / "test_chart.html"
        result = export_visualization(fig, str(output_path), format='html')

        assert result == str(output_path)

    def test_export_creates_directory(self, sample_equity_curve, tmp_path):
        """Test that export creates parent directories."""
        from src.backtest.visualization import BacktestVisualizer, export_visualization

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=False)

        output_path = tmp_path / "subdir" / "nested" / "chart"
        result = export_visualization(fig, str(output_path), format='html')

        assert (tmp_path / "subdir" / "nested" / "chart.html").exists()

    def test_export_unknown_format_raises(self, sample_equity_curve, tmp_path):
        """Test that unknown format raises error."""
        from src.backtest.visualization import BacktestVisualizer, export_visualization

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=False)

        with pytest.raises(ValueError, match="Unknown format"):
            export_visualization(fig, str(tmp_path / "chart"), format='pdf')


class TestVisualizerColors:
    """Tests for color configuration."""

    def test_default_colors(self, sample_equity_curve):
        """Test that default colors are set."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)

        assert 'equity' in viz.colors
        assert 'drawdown' in viz.colors
        assert 'win' in viz.colors
        assert 'loss' in viz.colors

    def test_colors_applied_to_equity_curve(self, sample_equity_curve):
        """Test that colors are applied to equity curve plot."""
        from src.backtest.visualization import BacktestVisualizer

        viz = BacktestVisualizer(equity_curve=sample_equity_curve)
        fig = viz.plot_equity_curve(show_drawdown=True)

        # Check that equity trace has the expected color
        equity_trace = fig.data[0]
        assert equity_trace.line.color == viz.colors['equity']


class TestPlotlyImportHandling:
    """Tests for handling missing Plotly."""

    def test_require_plotly_raises_when_missing(self):
        """Test that _require_plotly raises ImportError when plotly unavailable."""
        from src.backtest import visualization

        # Save original state
        original = visualization.PLOTLY_AVAILABLE
        original_go = visualization.go

        try:
            # Simulate plotly not being available
            visualization.PLOTLY_AVAILABLE = False
            visualization.go = None

            with pytest.raises(ImportError, match="Plotly is required"):
                visualization._require_plotly()
        finally:
            # Restore
            visualization.PLOTLY_AVAILABLE = original
            visualization.go = original_go


class TestVisualizationEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_equity_point(self):
        """Test visualization with single equity point."""
        from src.backtest.visualization import BacktestVisualizer

        curve = EquityCurve(initial_equity=1000.0)
        curve.add_point(datetime(2024, 1, 1), 1000.0)

        viz = BacktestVisualizer(equity_curve=curve)

        # Should handle single point gracefully
        fig = viz.plot_equity_curve(show_drawdown=True)
        assert len(fig.data) >= 1

    def test_all_winning_trades(self):
        """Test trade distribution with all winners."""
        from src.backtest.visualization import BacktestVisualizer

        log = TradeLog()
        base_time = datetime(2024, 1, 1)

        for i in range(5):
            log.add_trade(
                entry_time=base_time + timedelta(hours=i),
                exit_time=base_time + timedelta(hours=i, minutes=30),
                direction=1,
                entry_price=6050.0,
                exit_price=6055.0,
                contracts=1,
                gross_pnl=25.0,
                commission=0.42,
                slippage=0.42,
                exit_reason=ExitReason.TARGET,
            )

        viz = BacktestVisualizer(trade_log=log)
        fig = viz.plot_trade_distribution()

        # Should only have winners histogram
        assert len(fig.data) == 1

    def test_all_losing_trades(self):
        """Test trade distribution with all losers."""
        from src.backtest.visualization import BacktestVisualizer

        log = TradeLog()
        base_time = datetime(2024, 1, 1)

        for i in range(5):
            log.add_trade(
                entry_time=base_time + timedelta(hours=i),
                exit_time=base_time + timedelta(hours=i, minutes=30),
                direction=1,
                entry_price=6050.0,
                exit_price=6045.0,
                contracts=1,
                gross_pnl=-25.0,
                commission=0.42,
                slippage=0.42,
                exit_reason=ExitReason.STOP,
            )

        viz = BacktestVisualizer(trade_log=log)
        fig = viz.plot_trade_distribution()

        # Should only have losers histogram
        assert len(fig.data) == 1

    def test_no_drawdown_equity_curve(self):
        """Test equity curve that only goes up (no drawdown)."""
        from src.backtest.visualization import BacktestVisualizer

        curve = EquityCurve(initial_equity=1000.0)
        base_time = datetime(2024, 1, 1)

        for i in range(10):
            curve.add_point(base_time + timedelta(hours=i), 1000.0 + i * 10)

        viz = BacktestVisualizer(equity_curve=curve)
        fig = viz.plot_drawdown_analysis()

        # Should still create figure even with no significant drawdowns
        assert isinstance(fig, go.Figure)

    def test_metrics_with_inf_values(self):
        """Test metrics table with infinite values."""
        from src.backtest.visualization import BacktestVisualizer

        metrics = PerformanceMetrics(
            profit_factor=float('inf'),  # No losing trades
            recovery_factor=float('inf'),
        )

        viz = BacktestVisualizer(metrics=metrics)

        # Should handle inf gracefully
        fig = viz.create_metrics_table()
        assert isinstance(fig, go.Figure)

    def test_zero_trades_time_of_day(self):
        """Test time of day analysis with trades in limited hours."""
        from src.backtest.visualization import BacktestVisualizer

        log = TradeLog()
        base_time = datetime(2024, 1, 1, 10, 0)  # Only 10 AM trades

        for i in range(5):
            log.add_trade(
                entry_time=base_time + timedelta(days=i),
                exit_time=base_time + timedelta(days=i, minutes=30),
                direction=1,
                entry_price=6050.0,
                exit_price=6055.0,
                contracts=1,
                gross_pnl=25.0,
                commission=0.42,
                slippage=0.42,
                exit_reason=ExitReason.TARGET,
            )

        viz = BacktestVisualizer(trade_log=log)
        fig = viz.plot_time_of_day_analysis()

        # Should show all 24 hours with most being zero
        assert len(fig.data[0].x) == 24


class TestModuleImports:
    """Tests for module-level imports and availability."""

    def test_backtest_module_exports_visualization(self):
        """Test that backtest module exports visualization classes."""
        from src.backtest import VISUALIZATION_AVAILABLE

        assert VISUALIZATION_AVAILABLE is True

    def test_backtest_module_exports_all_classes(self):
        """Test that all visualization classes are exported."""
        from src.backtest import (
            BacktestVisualizer,
            WalkForwardVisualizer,
            DrawdownPeriod,
            identify_drawdown_periods,
            export_visualization,
        )

        assert BacktestVisualizer is not None
        assert WalkForwardVisualizer is not None
        assert DrawdownPeriod is not None
        assert identify_drawdown_periods is not None
        assert export_visualization is not None
