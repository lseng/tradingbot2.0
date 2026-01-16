"""
Tests for Performance Monitoring Module.

Tests the performance monitoring infrastructure that tracks:
- WebSocket quote reception latency (<100ms)
- Order execution timing (signal to fill <1000ms)
- Order placement round-trip (<500ms)
- Memory stability over 8-hour sessions

Reference: specs/live-trading-execution.md
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.lib.performance_monitor import (
    PerformanceMonitor,
    MetricType,
    LatencyStats,
    LatencySample,
    MemorySnapshot,
    Timer,
    AsyncTimer,
    measure_time,
    get_global_monitor,
    set_global_monitor,
    PERFORMANCE_THRESHOLDS,
)
from src.trading.order_executor import ExecutionTiming, EntryResult, ExecutionStatus
from src.api.topstepx_ws import Quote


class TestPerformanceThresholds:
    """Test performance threshold constants."""

    def test_quote_latency_threshold(self):
        """WebSocket quote reception latency threshold is 100ms."""
        assert PERFORMANCE_THRESHOLDS[MetricType.QUOTE_LATENCY] == 100.0

    def test_order_execution_threshold(self):
        """Order execution (signal to fill) threshold is 1000ms."""
        assert PERFORMANCE_THRESHOLDS[MetricType.ORDER_EXECUTION] == 1000.0

    def test_order_round_trip_threshold(self):
        """Order placement round-trip threshold is 500ms."""
        assert PERFORMANCE_THRESHOLDS[MetricType.ORDER_ROUND_TRIP] == 500.0

    def test_feature_calculation_threshold(self):
        """Feature calculation threshold is 5ms."""
        assert PERFORMANCE_THRESHOLDS[MetricType.FEATURE_CALCULATION] == 5.0

    def test_inference_threshold(self):
        """Model inference threshold is 10ms."""
        assert PERFORMANCE_THRESHOLDS[MetricType.INFERENCE] == 10.0


class TestLatencySample:
    """Test LatencySample dataclass."""

    def test_create_sample(self):
        """Test creating a latency sample."""
        sample = LatencySample(
            metric_type=MetricType.QUOTE_LATENCY,
            latency_ms=50.5,
        )
        assert sample.metric_type == MetricType.QUOTE_LATENCY
        assert sample.latency_ms == 50.5
        assert isinstance(sample.timestamp, datetime)
        assert sample.metadata == {}

    def test_sample_with_metadata(self):
        """Test sample with metadata."""
        sample = LatencySample(
            metric_type=MetricType.ORDER_EXECUTION,
            latency_ms=100.0,
            metadata={"order_id": "123", "contract": "MES"},
        )
        assert sample.metadata["order_id"] == "123"
        assert sample.metadata["contract"] == "MES"


class TestLatencyStats:
    """Test LatencyStats dataclass."""

    def test_empty_stats(self):
        """Test empty stats."""
        stats = LatencyStats(metric_type=MetricType.QUOTE_LATENCY)
        assert stats.count == 0
        assert stats.min_ms == float('inf')
        assert stats.max_ms == 0.0
        assert stats.violations == 0

    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = LatencyStats(
            metric_type=MetricType.ORDER_EXECUTION,
            count=100,
            min_ms=50.0,
            max_ms=500.0,
            mean_ms=150.0,
            p50_ms=120.0,
            p95_ms=300.0,
            p99_ms=450.0,
            violations=5,
            violation_threshold_ms=1000.0,
        )
        d = stats.to_dict()

        assert d["metric_type"] == "order_execution"
        assert d["count"] == 100
        assert d["min_ms"] == 50.0
        assert d["max_ms"] == 500.0
        assert d["violations"] == 5
        assert d["violation_rate_pct"] == 5.0
        assert d["threshold_ms"] == 1000.0


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=100.5,
            peak_mb=150.0,
            allocated_blocks=1000,
        )
        assert snapshot.current_mb == 100.5
        assert snapshot.peak_mb == 150.0
        assert snapshot.allocated_blocks == 1000

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        ts = datetime(2026, 1, 16, 12, 0, 0)
        snapshot = MemorySnapshot(
            timestamp=ts,
            current_mb=100.5,
            peak_mb=150.0,
        )
        d = snapshot.to_dict()

        assert d["timestamp"] == "2026-01-16T12:00:00"
        assert d["current_mb"] == 100.5
        assert d["peak_mb"] == 150.0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_init(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert not monitor._running
        assert monitor._start_time is None

    def test_start_stop(self):
        """Test starting and stopping monitor."""
        monitor = PerformanceMonitor()
        monitor.start()

        assert monitor._running
        assert monitor._start_time is not None

        monitor.stop()
        assert not monitor._running

    def test_record_latency(self):
        """Test recording latency."""
        monitor = PerformanceMonitor()

        monitor.record_latency(MetricType.QUOTE_LATENCY, 50.0)
        monitor.record_latency(MetricType.QUOTE_LATENCY, 75.0)
        monitor.record_latency(MetricType.QUOTE_LATENCY, 60.0)

        stats = monitor.get_stats(MetricType.QUOTE_LATENCY)
        assert stats.count == 3
        assert stats.min_ms == 50.0
        assert stats.max_ms == 75.0

    def test_record_latency_with_violation(self):
        """Test recording latency that exceeds threshold."""
        violations_recorded = []

        def on_violation(metric_type, latency_ms):
            violations_recorded.append((metric_type, latency_ms))

        monitor = PerformanceMonitor(on_violation=on_violation)

        # Quote latency threshold is 100ms
        monitor.record_latency(MetricType.QUOTE_LATENCY, 50.0)  # OK
        monitor.record_latency(MetricType.QUOTE_LATENCY, 150.0)  # Violation

        assert len(violations_recorded) == 1
        assert violations_recorded[0] == (MetricType.QUOTE_LATENCY, 150.0)

        stats = monitor.get_stats(MetricType.QUOTE_LATENCY)
        assert stats.violations == 1

    def test_has_violations(self):
        """Test checking for violations."""
        monitor = PerformanceMonitor()

        assert not monitor.has_violations()

        monitor.record_latency(MetricType.QUOTE_LATENCY, 150.0)  # Violation

        assert monitor.has_violations()
        assert monitor.has_violations(MetricType.QUOTE_LATENCY)
        assert not monitor.has_violations(MetricType.INFERENCE)

    def test_get_violation_count(self):
        """Test getting violation count."""
        monitor = PerformanceMonitor()

        monitor.record_latency(MetricType.QUOTE_LATENCY, 150.0)  # Violation
        monitor.record_latency(MetricType.QUOTE_LATENCY, 200.0)  # Violation
        monitor.record_latency(MetricType.ORDER_EXECUTION, 1500.0)  # Violation

        assert monitor.get_violation_count() == 3
        assert monitor.get_violation_count(MetricType.QUOTE_LATENCY) == 2
        assert monitor.get_violation_count(MetricType.ORDER_EXECUTION) == 1

    def test_get_stats_empty(self):
        """Test getting stats with no samples."""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats(MetricType.INFERENCE)

        assert stats.count == 0
        assert stats.violations == 0

    def test_get_stats_with_samples(self):
        """Test getting stats with multiple samples."""
        monitor = PerformanceMonitor()

        # Add 100 samples
        for i in range(100):
            monitor.record_latency(MetricType.INFERENCE, float(i + 1))

        stats = monitor.get_stats(MetricType.INFERENCE)

        assert stats.count == 100
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        assert 45 < stats.mean_ms < 55  # Should be around 50.5

    def test_get_all_stats(self):
        """Test getting all stats."""
        monitor = PerformanceMonitor()

        monitor.record_latency(MetricType.QUOTE_LATENCY, 50.0)
        monitor.record_latency(MetricType.INFERENCE, 5.0)

        all_stats = monitor.get_all_stats()

        assert MetricType.QUOTE_LATENCY.value in all_stats
        assert MetricType.INFERENCE.value in all_stats
        assert all_stats[MetricType.QUOTE_LATENCY.value].count == 1
        assert all_stats[MetricType.INFERENCE.value].count == 1

    def test_get_uptime(self):
        """Test getting uptime."""
        monitor = PerformanceMonitor()

        # Not started
        assert monitor.get_uptime() == timedelta()

        monitor.start()
        time.sleep(0.1)

        uptime = monitor.get_uptime()
        assert uptime.total_seconds() > 0

        monitor.stop()

    def test_circular_buffer(self):
        """Test that samples are limited by max_samples."""
        monitor = PerformanceMonitor(max_samples=10)

        # Add 20 samples
        for i in range(20):
            monitor.record_latency(MetricType.QUOTE_LATENCY, float(i))

        stats = monitor.get_stats(MetricType.QUOTE_LATENCY)
        assert stats.count == 10  # Only last 10 samples

    def test_to_dict(self):
        """Test converting monitor state to dictionary."""
        monitor = PerformanceMonitor()
        monitor.start()

        monitor.record_latency(MetricType.QUOTE_LATENCY, 50.0)

        d = monitor.to_dict()

        assert "uptime_seconds" in d
        assert "start_time" in d
        assert "metrics" in d
        assert "memory_trend" in d
        assert "total_violations" in d

        monitor.stop()

    def test_generate_report(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor()
        monitor.start()

        monitor.record_latency(MetricType.QUOTE_LATENCY, 50.0)
        monitor.record_latency(MetricType.INFERENCE, 5.0)

        report = monitor.generate_report()

        assert "PERFORMANCE MONITORING REPORT" in report
        assert "quote_latency" in report
        assert "inference" in report
        assert "MEMORY" in report

        monitor.stop()


class TestTimer:
    """Test Timer context manager."""

    def test_timer_basic(self):
        """Test basic timer usage."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms

        assert t.elapsed_ms > 5  # Should be at least 5ms
        assert t.elapsed_ms < 100  # But less than 100ms

    def test_timer_with_monitor(self):
        """Test timer with monitor recording."""
        monitor = PerformanceMonitor()

        with Timer(monitor, MetricType.INFERENCE):
            time.sleep(0.01)

        stats = monitor.get_stats(MetricType.INFERENCE)
        assert stats.count == 1
        assert stats.min_ms > 5

    def test_timer_elapsed_seconds(self):
        """Test elapsed time in seconds."""
        with Timer() as t:
            time.sleep(0.05)  # 50ms

        assert 0.04 < t.elapsed_s < 0.1

    def test_timer_no_exception_on_early_access(self):
        """Test accessing elapsed before exit."""
        t = Timer()
        assert t.elapsed_ms == 0.0


class TestAsyncTimer:
    """Test AsyncTimer context manager."""

    @pytest.mark.asyncio
    async def test_async_timer_basic(self):
        """Test basic async timer usage."""
        async with AsyncTimer() as t:
            await asyncio.sleep(0.01)

        assert t.elapsed_ms > 5

    @pytest.mark.asyncio
    async def test_async_timer_with_monitor(self):
        """Test async timer with monitor recording."""
        monitor = PerformanceMonitor()

        async with AsyncTimer(monitor, MetricType.ORDER_EXECUTION):
            await asyncio.sleep(0.01)

        stats = monitor.get_stats(MetricType.ORDER_EXECUTION)
        assert stats.count == 1


class TestMeasureTimeDecorator:
    """Test measure_time decorator."""

    def test_sync_function(self):
        """Test decorating sync function."""
        monitor = PerformanceMonitor()

        @measure_time(monitor, MetricType.FEATURE_CALCULATION)
        def compute():
            time.sleep(0.01)
            return 42

        result = compute()

        assert result == 42
        stats = monitor.get_stats(MetricType.FEATURE_CALCULATION)
        assert stats.count == 1
        assert stats.min_ms > 5

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test decorating async function."""
        monitor = PerformanceMonitor()

        @measure_time(monitor, MetricType.ORDER_ROUND_TRIP)
        async def async_compute():
            await asyncio.sleep(0.01)
            return 42

        result = await async_compute()

        assert result == 42
        stats = monitor.get_stats(MetricType.ORDER_ROUND_TRIP)
        assert stats.count == 1


class TestGlobalMonitor:
    """Test global monitor functions."""

    def test_get_global_monitor(self):
        """Test getting global monitor."""
        monitor = get_global_monitor()
        assert isinstance(monitor, PerformanceMonitor)

    def test_set_global_monitor(self):
        """Test setting global monitor."""
        custom_monitor = PerformanceMonitor(max_samples=100)
        set_global_monitor(custom_monitor)

        assert get_global_monitor() is custom_monitor


class TestExecutionTiming:
    """Test ExecutionTiming dataclass from order_executor."""

    def test_empty_timing(self):
        """Test timing with no values."""
        timing = ExecutionTiming()
        assert timing.placement_latency_ms is None
        assert timing.fill_latency_ms is None
        assert timing.total_latency_ms is None

    def test_placement_latency(self):
        """Test placement latency calculation."""
        timing = ExecutionTiming(
            signal_time=0.0,
            order_placed_time=0.1,  # 100ms later
        )
        assert timing.placement_latency_ms == pytest.approx(100.0)

    def test_fill_latency(self):
        """Test fill latency calculation."""
        timing = ExecutionTiming(
            order_placed_time=0.0,
            fill_received_time=0.2,  # 200ms later
        )
        assert timing.fill_latency_ms == pytest.approx(200.0)

    def test_total_latency(self):
        """Test total latency calculation."""
        timing = ExecutionTiming(
            signal_time=0.0,
            order_placed_time=0.1,
            fill_received_time=0.5,  # 500ms total
        )
        assert timing.total_latency_ms == pytest.approx(500.0)

    def test_to_dict(self):
        """Test converting timing to dictionary."""
        timing = ExecutionTiming(
            signal_time=0.0,
            order_placed_time=0.1,
            fill_received_time=0.5,
        )
        d = timing.to_dict()

        assert d["placement_latency_ms"] == pytest.approx(100.0)
        assert d["fill_latency_ms"] == pytest.approx(400.0)
        assert d["total_latency_ms"] == pytest.approx(500.0)


class TestEntryResultTiming:
    """Test EntryResult with timing information."""

    def test_entry_result_with_timing(self):
        """Test entry result includes timing."""
        timing = ExecutionTiming(
            signal_time=0.0,
            order_placed_time=0.1,
            fill_received_time=0.5,
        )
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.00,
            entry_fill_size=1,
            timing=timing,
        )

        assert result.success
        assert result.timing is not None
        assert result.execution_latency_ms == pytest.approx(500.0)

    def test_entry_result_no_timing(self):
        """Test entry result without timing."""
        result = EntryResult(
            status=ExecutionStatus.FILLED,
            entry_fill_price=5000.00,
        )

        assert result.execution_latency_ms is None


class TestQuoteLatencyTracking:
    """Test Quote dataclass latency tracking."""

    def test_quote_with_server_timestamp_seconds(self):
        """Test quote parsing with Unix timestamp in seconds."""
        now = datetime.utcnow()
        server_ts = now.timestamp()

        quote = Quote.from_signalr({
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
            "timestamp": server_ts,
        })

        assert quote.server_timestamp is not None
        assert quote.reception_latency_ms is not None
        # Latency should be small (local processing time)
        assert quote.reception_latency_ms < 100

    def test_quote_with_server_timestamp_milliseconds(self):
        """Test quote parsing with Unix timestamp in milliseconds."""
        now = datetime.utcnow()
        server_ts_ms = now.timestamp() * 1000

        quote = Quote.from_signalr({
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
            "timestamp": server_ts_ms,
        })

        assert quote.server_timestamp is not None
        assert quote.reception_latency_ms is not None
        assert quote.reception_latency_ms < 100

    def test_quote_with_iso_timestamp(self):
        """Test quote parsing with ISO format timestamp."""
        now = datetime.utcnow()
        iso_ts = now.isoformat()

        quote = Quote.from_signalr({
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
            "timestamp": iso_ts,
        })

        assert quote.server_timestamp is not None
        assert quote.reception_latency_ms is not None

    def test_quote_without_server_timestamp(self):
        """Test quote parsing without server timestamp."""
        quote = Quote.from_signalr({
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
        })

        assert quote.server_timestamp is None
        assert quote.reception_latency_ms is None
        assert quote.timestamp is not None  # Local timestamp still set

    def test_quote_negative_latency_clamped(self):
        """Test that negative latency (clock skew) is clamped to 0."""
        # Server timestamp in the future (clock skew)
        future_ts = (datetime.utcnow() + timedelta(seconds=10)).timestamp()

        quote = Quote.from_signalr({
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
            "timestamp": future_ts,
        })

        # Latency should be clamped to 0
        assert quote.reception_latency_ms == 0.0


class TestMemoryMonitoring:
    """Test memory monitoring functionality."""

    def test_record_memory(self):
        """Test recording memory snapshot."""
        monitor = PerformanceMonitor()
        monitor.start()

        snapshot = monitor.record_memory()

        assert snapshot.current_mb >= 0
        assert snapshot.peak_mb >= snapshot.current_mb

        monitor.stop()

    def test_memory_trend_insufficient_data(self):
        """Test memory trend with insufficient data."""
        monitor = PerformanceMonitor()

        trend = monitor.get_memory_trend()

        assert trend["status"] == "insufficient_data"

    def test_memory_trend_stable(self):
        """Test memory trend detection for stable memory."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Simulate stable memory (same value for all snapshots)
        for _ in range(100):
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                current_mb=100.0,
                peak_mb=100.0,
            )
            monitor._memory_samples.append(snapshot)

        trend = monitor.get_memory_trend()

        assert trend["status"] == "stable"
        assert trend["growth_pct"] < 20

        monitor.stop()

    def test_memory_trend_growing(self):
        """Test memory trend detection for growing memory."""
        monitor = PerformanceMonitor()

        # Simulate growing memory (leak pattern)
        for i in range(100):
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                current_mb=100.0 + i * 1.0,  # Growing by 1MB per sample
                peak_mb=100.0 + i * 1.0,
            )
            monitor._memory_samples.append(snapshot)

        trend = monitor.get_memory_trend()

        # Growth should be detected
        assert trend["growth_mb"] > 0
        # With 100 samples, early mean ~ 104.5, late mean ~ 194.5
        # Growth is ~90MB which is ~86% - should be flagged as growing
        assert trend["status"] == "growing"


class TestPerformanceIntegration:
    """Integration tests for performance monitoring."""

    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        monitor = PerformanceMonitor()
        monitor.start()

        # Simulate quote latencies
        for _ in range(50):
            monitor.record_latency(MetricType.QUOTE_LATENCY, 50.0 + (time.time() % 30))

        # Simulate order execution
        for _ in range(10):
            monitor.record_latency(MetricType.ORDER_EXECUTION, 300.0 + (time.time() % 100))

        # Simulate some violations
        monitor.record_latency(MetricType.QUOTE_LATENCY, 150.0)  # >100ms
        monitor.record_latency(MetricType.ORDER_EXECUTION, 1200.0)  # >1000ms

        # Get report
        report = monitor.generate_report()
        assert "VIOLATION" in report or "violations" in report.lower()

        # Check stats
        quote_stats = monitor.get_stats(MetricType.QUOTE_LATENCY)
        assert quote_stats.count == 51
        assert quote_stats.violations >= 1

        order_stats = monitor.get_stats(MetricType.ORDER_EXECUTION)
        assert order_stats.count == 11
        assert order_stats.violations >= 1

        monitor.stop()

    def test_threshold_boundary_values(self):
        """Test latencies at exact threshold boundaries."""
        monitor = PerformanceMonitor()

        # Exactly at threshold (should not be violation)
        monitor.record_latency(MetricType.QUOTE_LATENCY, 100.0)
        assert monitor.get_stats(MetricType.QUOTE_LATENCY).violations == 0

        # Just above threshold (should be violation)
        monitor.record_latency(MetricType.QUOTE_LATENCY, 100.1)
        assert monitor.get_stats(MetricType.QUOTE_LATENCY).violations == 1

    def test_percentile_calculations(self):
        """Test percentile calculations with known data."""
        monitor = PerformanceMonitor()

        # Add values 1-100
        for i in range(1, 101):
            monitor.record_latency(MetricType.INFERENCE, float(i))

        stats = monitor.get_stats(MetricType.INFERENCE)

        # P50 should be around 50
        assert 45 <= stats.p50_ms <= 55

        # P95 should be around 95
        assert 90 <= stats.p95_ms <= 100

        # P99 should be around 99
        assert 95 <= stats.p99_ms <= 100
