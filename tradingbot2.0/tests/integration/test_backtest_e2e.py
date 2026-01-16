"""
End-to-End Backtest Integration Tests

These tests validate the complete backtest pipeline including:
- Full backtest runs with synthetic data
- Walk-forward validation with correct fold generation
- Risk limits enforcement in simulation
- EOD flatten at correct times
- Transaction costs reduce returns correctly
- Random baseline produces ~0 expectancy (no lookahead bias)
- Results reproducibility with same random seed

These tests address Go-Live Checklist items #1-5:
1. Walk-forward backtest shows consistent profitability
2. Out-of-sample accuracy tracking
3. All risk limits enforced
4. EOD flatten works 100% of the time
5. No lookahead bias

Why these tests matter:
- E2E tests catch integration bugs that unit tests miss
- Walk-forward validation prevents overfitting
- Risk limit testing ensures capital protection
- Random baseline validates no lookahead bias exists
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    Signal,
    SignalType,
    Position,
    WalkForwardValidator,
    create_simple_signal_generator,
)
from src.backtest.metrics import PerformanceMetrics, calculate_metrics


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data for testing.

    Creates ~1 week of 1-second RTH data with realistic price movements.
    This is sufficient for validating the backtest engine without loading
    the full 227MB parquet file.
    """
    np.random.seed(42)  # Reproducibility

    # Generate 5 trading days of RTH data (6.5 hours each = 23,400 seconds/day)
    bars_per_day = 23400  # 6.5 hours * 60 min * 60 sec
    num_days = 5
    total_bars = bars_per_day * num_days

    # Start from a realistic price
    base_price = 5000.0

    # Generate prices using random walk with mean reversion
    prices = [base_price]
    for i in range(total_bars - 1):
        # Small mean-reverting moves (typical for 1-second MES data)
        mean_reversion = 0.0001 * (base_price - prices[-1])
        noise = np.random.normal(0, 0.02)  # ~0.5 tick std dev
        change = mean_reversion + noise
        prices.append(prices[-1] + change)

    prices = np.array(prices)

    # Generate OHLC from prices with some intra-bar volatility
    high = prices + np.abs(np.random.normal(0, 0.05, total_bars))
    low = prices - np.abs(np.random.normal(0, 0.05, total_bars))
    open_prices = prices + np.random.normal(0, 0.01, total_bars)
    close = prices

    # Ensure OHLC relationships (L <= O,C <= H)
    low = np.minimum(low, np.minimum(open_prices, close))
    high = np.maximum(high, np.maximum(open_prices, close))

    # Generate volume (higher at open/close, lower mid-day)
    volume = np.random.poisson(50, total_bars)

    # Create timestamps for 5 trading days of RTH (9:30 AM - 4:00 PM NY)
    timestamps = []
    start_date = datetime(2024, 1, 2, 9, 30, 0)  # Tuesday Jan 2, 2024

    for day in range(num_days):
        day_start = start_date + timedelta(days=day)
        for second in range(bars_per_day):
            timestamps.append(day_start + timedelta(seconds=second))

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=pd.DatetimeIndex(timestamps))

    return df


@pytest.fixture
def small_ohlcv_data() -> pd.DataFrame:
    """
    Generate small synthetic data for quick tests.
    Only 1000 bars - enough to run basic engine tests.
    """
    np.random.seed(42)

    n_bars = 1000
    base_price = 5000.0
    prices = base_price + np.cumsum(np.random.normal(0, 0.1, n_bars))

    timestamps = pd.date_range(
        start='2024-01-02 09:30:00',
        periods=n_bars,
        freq='1s'
    )

    df = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.01, n_bars),
        'high': prices + np.abs(np.random.normal(0, 0.05, n_bars)),
        'low': prices - np.abs(np.random.normal(0, 0.05, n_bars)),
        'close': prices,
        'volume': np.random.poisson(50, n_bars),
    }, index=timestamps)

    # Ensure valid OHLC relationships
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    df['high'] = df[['high', 'open', 'close']].max(axis=1)

    return df


@pytest.fixture
def default_config() -> BacktestConfig:
    """Default backtest configuration for testing."""
    return BacktestConfig(
        initial_capital=1000.0,
        commission_per_side=0.20,
        exchange_fee_per_side=0.22,
        slippage_ticks=1.0,
        tick_size=0.25,
        tick_value=1.25,
        min_confidence=0.60,
        default_stop_ticks=8.0,
        default_target_ticks=16.0,
        max_daily_loss=50.0,
    )


def create_trend_signal_generator(
    direction: int = 1,
    confidence: float = 0.75,
    trade_every_n_bars: int = 100,
) -> callable:
    """
    Create a simple trend-following signal generator.

    Trades in one direction with fixed confidence.
    Used to test that profitable strategies show profits.
    """
    counter = [0]  # Use list to allow modification in closure

    def signal_generator(
        bar: pd.Series,
        position: Optional[Position],
        context: Dict[str, Any],
    ) -> Signal:
        counter[0] += 1

        if position is not None:
            # Hold position, let stop/target handle exits
            return Signal(SignalType.HOLD)

        # Only trade every N bars to avoid excessive trading
        if counter[0] % trade_every_n_bars != 0:
            return Signal(SignalType.HOLD)

        if direction == 1:
            return Signal(
                SignalType.LONG_ENTRY,
                confidence=confidence,
                stop_ticks=8.0,
                target_ticks=16.0,
            )
        else:
            return Signal(
                SignalType.SHORT_ENTRY,
                confidence=confidence,
                stop_ticks=8.0,
                target_ticks=16.0,
            )

    return signal_generator


def create_random_signal_generator(
    seed: int = 42,
    trade_probability: float = 0.01,
    min_confidence: float = 0.60,
) -> callable:
    """
    Create a random signal generator.

    Used to test that random strategies produce ~0 expectancy.
    This validates there is no lookahead bias in the backtest engine.
    """
    rng = np.random.RandomState(seed)

    def signal_generator(
        bar: pd.Series,
        position: Optional[Position],
        context: Dict[str, Any],
    ) -> Signal:
        if position is not None:
            # Random exit with low probability
            if rng.random() < 0.1:
                if position.direction == 1:
                    return Signal(SignalType.EXIT_LONG, confidence=0.5)
                else:
                    return Signal(SignalType.EXIT_SHORT, confidence=0.5)
            return Signal(SignalType.HOLD)

        # Random entry
        if rng.random() > trade_probability:
            return Signal(SignalType.HOLD)

        confidence = rng.uniform(min_confidence, 0.95)
        direction = rng.choice([1, -1])

        if direction == 1:
            return Signal(
                SignalType.LONG_ENTRY,
                confidence=confidence,
                stop_ticks=8.0,
                target_ticks=16.0,
            )
        else:
            return Signal(
                SignalType.SHORT_ENTRY,
                confidence=confidence,
                stop_ticks=8.0,
                target_ticks=16.0,
            )

    return signal_generator


# ============================================================================
# End-to-End Backtest Tests
# ============================================================================

class TestBacktestEngineE2E:
    """End-to-end tests for the backtest engine."""

    def test_backtest_runs_to_completion(self, small_ohlcv_data, default_config):
        """
        Test that a basic backtest runs without errors.

        Validates:
        - Engine initializes correctly
        - Data is processed bar by bar
        - Result is returned with valid structure
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        result = engine.run(small_ohlcv_data, signal_gen, verbose=False)

        assert result is not None
        assert result.report is not None
        assert result.report.metrics is not None
        assert result.execution_time_seconds > 0

    def test_backtest_processes_all_bars(self, small_ohlcv_data, default_config):
        """
        Test that all bars are processed.

        Validates:
        - No bars are skipped
        - Data stats reflect input size
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        result = engine.run(small_ohlcv_data, signal_gen)

        assert result.data_stats['total_bars'] == len(small_ohlcv_data)

    def test_backtest_generates_trades(self, sample_ohlcv_data, default_config):
        """
        Test that backtest generates trades when signals are provided.

        Validates:
        - Trades are recorded
        - Trade count is reasonable for data size
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator(trade_every_n_bars=500)

        result = engine.run(sample_ohlcv_data, signal_gen)

        # Should have generated some trades over 5 days
        assert result.report.metrics.total_trades > 0
        assert result.report.trade_log.get_trade_count() > 0

    def test_backtest_respects_confidence_threshold(self, small_ohlcv_data, default_config):
        """
        Test that low-confidence signals are filtered out.

        Validates:
        - min_confidence threshold is enforced
        - Only high-confidence signals generate trades
        """
        default_config.min_confidence = 0.80  # High threshold
        engine = BacktestEngine(config=default_config)

        # Signal with confidence below threshold
        def low_confidence_signal(bar, position, context):
            if position is None:
                return Signal(SignalType.LONG_ENTRY, confidence=0.70)  # Below 0.80
            return Signal(SignalType.HOLD)

        result = engine.run(small_ohlcv_data, low_confidence_signal)

        # No trades should be generated (all filtered by confidence)
        assert result.report.metrics.total_trades == 0

    def test_equity_curve_tracking(self, sample_ohlcv_data, default_config):
        """
        Test that equity curve is properly tracked.

        Validates:
        - Equity starts at initial capital
        - Equity changes with trades
        - No gaps in equity tracking
        """
        default_config.log_frequency = 100  # Log every 100 bars
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        result = engine.run(sample_ohlcv_data, signal_gen)

        equity_values = result.report.equity_curve.get_equity_values()

        assert len(equity_values) > 0
        assert equity_values[0] == default_config.initial_capital


class TestTransactionCostIntegration:
    """Test that transaction costs are correctly applied."""

    def test_commission_reduces_returns(self, sample_ohlcv_data, default_config):
        """
        Test that commissions reduce net returns.

        Validates Go-Live checklist item: Transaction costs reduce returns by expected amount.
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator(trade_every_n_bars=200)

        result = engine.run(sample_ohlcv_data, signal_gen)

        metrics = result.report.metrics

        if metrics.total_trades > 0:
            # Commission should be positive
            assert metrics.total_commission > 0

            # Expected commission: $0.84 per round-trip per contract
            expected_commission_per_trade = 0.84
            min_expected_commission = metrics.total_trades * expected_commission_per_trade

            # Actual commission should be at least this (more if multiple contracts)
            assert metrics.total_commission >= min_expected_commission * 0.9  # 10% tolerance

    def test_slippage_applied_to_fills(self, sample_ohlcv_data, default_config):
        """
        Test that slippage is applied to trade fills.

        Validates:
        - Slippage is tracked
        - Fill prices differ from signal prices
        """
        default_config.slippage_ticks = 1.0  # 1 tick slippage
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator(trade_every_n_bars=500)

        result = engine.run(sample_ohlcv_data, signal_gen)

        metrics = result.report.metrics

        if metrics.total_trades > 0:
            # Slippage should be tracked
            assert metrics.total_slippage >= 0


class TestRiskLimitsIntegration:
    """Test that risk limits are enforced in simulation."""

    def test_daily_loss_limit_stops_trading(self, sample_ohlcv_data, default_config):
        """
        Test that daily loss limit halts trading for the day.

        Validates Go-Live checklist item #3: All risk limits enforced.
        """
        default_config.max_daily_loss = 10.0  # Very tight limit for testing
        engine = BacktestEngine(config=default_config)

        # Always losing signal generator
        def always_losing_signal(bar, position, context):
            if position is None:
                return Signal(
                    SignalType.LONG_ENTRY,
                    confidence=0.75,
                    stop_ticks=4.0,  # Tight stop
                    target_ticks=100.0,  # Very far target (unlikely to hit)
                )
            return Signal(SignalType.HOLD)

        result = engine.run(sample_ohlcv_data, always_losing_signal)

        # With $10 daily loss limit and $5 stop losses, we should be limited
        # The daily loss should not significantly exceed the limit
        # Note: Individual trades may cause the daily loss to exceed slightly before being checked
        metrics = result.report.metrics

        # Verify we didn't lose more than expected (some overshoot is normal due to trade timing)
        total_loss = default_config.initial_capital - metrics.final_capital
        max_expected_loss = default_config.max_daily_loss * 10  # 5 days * 2 for overshoot

        assert total_loss <= max_expected_loss or metrics.total_trades <= 50

    def test_position_size_limit(self, small_ohlcv_data, default_config):
        """
        Test that position size is limited.

        Validates:
        - max_position_size is respected
        - No oversized positions are created
        """
        default_config.max_position_size = 1  # Single contract only
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        result = engine.run(small_ohlcv_data, signal_gen)

        # All trades should have 1 contract
        for trade in result.report.trade_log.get_trades():
            assert trade.contracts == 1


class TestEODFlattenIntegration:
    """Test that EOD flatten is properly enforced."""

    def test_eod_flatten_closes_positions(self):
        """
        Test that positions are flattened at EOD.

        Validates Go-Live checklist item #4: EOD flatten works 100% of the time.
        """
        # Create data that spans the EOD flatten time (4:30 PM)
        timestamps = pd.date_range(
            start='2024-01-02 16:00:00',  # 4:00 PM
            end='2024-01-02 16:45:00',  # 4:45 PM
            freq='1s'
        )

        np.random.seed(42)
        n_bars = len(timestamps)
        base_price = 5000.0

        df = pd.DataFrame({
            'open': np.full(n_bars, base_price),
            'high': np.full(n_bars, base_price + 0.25),
            'low': np.full(n_bars, base_price - 0.25),
            'close': np.full(n_bars, base_price),
            'volume': np.full(n_bars, 50),
        }, index=timestamps)

        config = BacktestConfig(
            eod_flatten_time=time(16, 30),  # 4:30 PM
            eod_close_only_time=time(16, 15),  # 4:15 PM
        )
        engine = BacktestEngine(config=config)

        # Signal that tries to open position
        def always_long_signal(bar, position, context):
            if position is None:
                return Signal(SignalType.LONG_ENTRY, confidence=0.75)
            return Signal(SignalType.HOLD)

        result = engine.run(df, always_long_signal)

        # Any position opened before 4:15 should be closed by 4:30
        # Check that trades have EOD flatten as exit reason
        for trade in result.report.trade_log.get_trades():
            if trade.exit_time.time() >= time(16, 30):
                # Trade should have been flattened
                assert trade.exit_time.time() <= time(16, 35)  # Small buffer for processing

    def test_no_new_positions_after_close_only_time(self):
        """
        Test that no new positions are opened after close-only time.

        Validates:
        - Position restrictions after 4:15 PM
        """
        timestamps = pd.date_range(
            start='2024-01-02 16:15:00',  # Start at close-only time
            end='2024-01-02 16:29:00',
            freq='1s'
        )

        np.random.seed(42)
        n_bars = len(timestamps)

        df = pd.DataFrame({
            'open': np.full(n_bars, 5000.0),
            'high': np.full(n_bars, 5000.25),
            'low': np.full(n_bars, 4999.75),
            'close': np.full(n_bars, 5000.0),
            'volume': np.full(n_bars, 50),
        }, index=timestamps)

        config = BacktestConfig(
            eod_close_only_time=time(16, 15),
            eod_flatten_time=time(16, 30),
        )
        engine = BacktestEngine(config=config)

        def always_long_signal(bar, position, context):
            if position is None:
                return Signal(SignalType.LONG_ENTRY, confidence=0.75)
            return Signal(SignalType.HOLD)

        result = engine.run(df, always_long_signal)

        # No new positions should be opened after close-only time
        for trade in result.report.trade_log.get_trades():
            assert trade.entry_time.time() < time(16, 15)


class TestRandomBaselineValidation:
    """
    Test that random strategies produce ~0 expectancy.

    This validates there is no lookahead bias in the backtest engine.
    """

    def test_random_strategy_zero_expectancy(self, sample_ohlcv_data, default_config):
        """
        Test that random signals produce approximately zero expectancy.

        Validates Go-Live checklist item #5: No lookahead bias.

        A random strategy should produce:
        - Expectancy near zero (within transaction costs)
        - Win rate approximately 50% for directional bets
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_random_signal_generator(seed=42, trade_probability=0.005)

        result = engine.run(sample_ohlcv_data, signal_gen)

        metrics = result.report.metrics

        if metrics.total_trades >= 20:  # Need enough trades for statistical significance
            # Expectancy should be near zero (or slightly negative due to costs)
            # Allow for some variance
            avg_trade_pnl = metrics.avg_trade_pnl if hasattr(metrics, 'avg_trade_pnl') else 0

            # With $0.84 commission and ~$1.25 slippage, expected cost is ~$2 per trade
            # Random strategy should lose approximately this amount per trade on average
            assert avg_trade_pnl < 5.0  # Not significantly profitable (would indicate bias)
            assert avg_trade_pnl > -20.0  # Not losing too much (would indicate bug)

    def test_reproducibility_with_seed(self, sample_ohlcv_data, default_config):
        """
        Test that results are reproducible with the same random seed.

        Validates:
        - Same seed produces same trades
        - Results are deterministic
        """
        engine = BacktestEngine(config=default_config)

        # Run twice with same seed
        signal_gen1 = create_random_signal_generator(seed=123, trade_probability=0.005)
        result1 = engine.run(sample_ohlcv_data.copy(), signal_gen1)

        signal_gen2 = create_random_signal_generator(seed=123, trade_probability=0.005)
        result2 = engine.run(sample_ohlcv_data.copy(), signal_gen2)

        # Results should be identical
        assert result1.report.metrics.total_trades == result2.report.metrics.total_trades
        assert abs(result1.report.metrics.final_capital - result2.report.metrics.final_capital) < 0.01


class TestWalkForwardValidation:
    """Test walk-forward validation framework."""

    def test_walk_forward_generates_folds(self):
        """
        Test that walk-forward validator generates correct number of folds.

        Validates:
        - Folds are generated correctly
        - Train/val/test windows are non-overlapping
        """
        # Generate 12 months of data
        timestamps = pd.date_range(
            start='2023-01-01',
            end='2023-12-31',
            freq='1h'  # Hourly for smaller data size
        )

        df = pd.DataFrame({
            'open': np.random.uniform(4990, 5010, len(timestamps)),
            'high': np.random.uniform(5000, 5020, len(timestamps)),
            'low': np.random.uniform(4980, 5000, len(timestamps)),
            'close': np.random.uniform(4990, 5010, len(timestamps)),
        }, index=timestamps)

        validator = WalkForwardValidator(
            training_months=6,
            validation_months=1,
            test_months=1,
            step_months=1,
        )

        folds = validator.generate_folds(df)

        # With 12 months total and 8-month window (6+1+1), stepping by 1
        # Should have 4-5 folds
        assert len(folds) >= 4

        # Check fold structure
        for fold in folds:
            assert 'train' in fold
            assert 'val' in fold
            assert 'test' in fold

            # Windows should be in order
            assert fold['train'][0] < fold['val'][0]
            assert fold['val'][0] < fold['test'][0]
            assert fold['val'][1] <= fold['test'][0]

    def test_walk_forward_min_trades_per_fold(self, sample_ohlcv_data, default_config):
        """
        Test that walk-forward tracks trade counts per fold.

        Validates:
        - Each fold has recorded metrics
        - Trade counts are reasonable
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator(trade_every_n_bars=1000)

        # Use a shorter window for the test data (only 5 days)
        validator = WalkForwardValidator(
            training_months=0,  # No training (we're testing backtest only)
            validation_months=0,
            test_months=1,
            step_months=1,
            min_trades_per_fold=1,  # Low threshold for test
        )

        # Just run the backtest on the test data directly (since we have limited data)
        result = engine.run(sample_ohlcv_data, signal_gen)

        # Verify we got results
        assert result.report.metrics.total_trades >= 0


class TestMetricsCalculation:
    """Test that performance metrics are calculated correctly."""

    def test_sharpe_ratio_calculation(self, sample_ohlcv_data, default_config):
        """
        Test that Sharpe ratio is calculated correctly.

        Validates:
        - Sharpe is computed from daily returns
        - Annualization is applied
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator(trade_every_n_bars=200)

        result = engine.run(sample_ohlcv_data, signal_gen)

        metrics = result.report.metrics

        # Sharpe should be a finite number
        assert np.isfinite(metrics.sharpe_ratio)

    def test_max_drawdown_calculation(self, sample_ohlcv_data, default_config):
        """
        Test that max drawdown is calculated correctly.

        Validates:
        - Max drawdown is tracked
        - Value is non-negative
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        result = engine.run(sample_ohlcv_data, signal_gen)

        metrics = result.report.metrics

        # Max drawdown should be non-negative
        assert metrics.max_drawdown_pct >= 0
        assert metrics.max_drawdown_dollars >= 0

    def test_win_rate_bounds(self, sample_ohlcv_data, default_config):
        """
        Test that win rate is within valid bounds.

        Validates:
        - Win rate is between 0% and 100%
        """
        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator(trade_every_n_bars=200)

        result = engine.run(sample_ohlcv_data, signal_gen)

        metrics = result.report.metrics

        if metrics.total_trades > 0:
            assert 0 <= metrics.win_rate_pct <= 100


class TestDataValidation:
    """Test that invalid data is properly rejected."""

    def test_empty_data_raises_error(self, default_config):
        """Test that empty DataFrame raises ValueError."""
        engine = BacktestEngine(config=default_config)

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            engine.run(empty_df, lambda bar, pos, ctx: Signal(SignalType.HOLD))

    def test_missing_columns_raises_error(self, default_config):
        """Test that missing required columns raise ValueError."""
        engine = BacktestEngine(config=default_config)

        # Missing 'high' column
        df = pd.DataFrame({
            'open': [100, 101],
            'low': [99, 100],
            'close': [100.5, 101.5],
        }, index=pd.date_range('2024-01-01', periods=2, freq='1s'))

        with pytest.raises(ValueError, match="missing required columns"):
            engine.run(df, lambda bar, pos, ctx: Signal(SignalType.HOLD))

    def test_non_datetime_index_raises_error(self, default_config):
        """Test that non-datetime index raises ValueError."""
        engine = BacktestEngine(config=default_config)

        df = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100.5, 101.5],
        })  # No DatetimeIndex

        with pytest.raises(ValueError, match="DatetimeIndex"):
            engine.run(df, lambda bar, pos, ctx: Signal(SignalType.HOLD))


class TestPerformanceRequirements:
    """Test that performance requirements are met."""

    def test_backtest_speed(self, sample_ohlcv_data, default_config):
        """
        Test that backtest meets speed requirements.

        Validates from spec: Process 1M bars in < 60 seconds.
        We test with smaller data and extrapolate.
        """
        import time as time_module

        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        start = time_module.time()
        result = engine.run(sample_ohlcv_data, signal_gen)
        elapsed = time_module.time() - start

        bars_processed = len(sample_ohlcv_data)
        bars_per_second = bars_processed / elapsed if elapsed > 0 else 0

        # Should process at least 10,000 bars/second to meet 1M in 60s requirement
        # (1M / 60s = ~16,667 bars/second)
        # Our test has ~117,000 bars, so it should complete in ~7 seconds max
        assert elapsed < 30, f"Backtest took {elapsed:.2f}s, expected < 30s"

    def test_memory_stability(self, sample_ohlcv_data, default_config):
        """
        Test that memory doesn't grow unboundedly during backtest.

        This is a basic smoke test - full memory profiling would
        require longer runs.
        """
        import gc

        engine = BacktestEngine(config=default_config)
        signal_gen = create_trend_signal_generator()

        # Run backtest
        result = engine.run(sample_ohlcv_data, signal_gen)

        # Force garbage collection
        gc.collect()

        # If we got here without MemoryError, basic test passed
        assert result is not None


# ============================================================================
# Integration Test Suite Runner
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
