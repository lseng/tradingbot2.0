"""
Performance Monitor for Live Trading.

Tracks and validates live trading performance requirements:
- WebSocket quote reception latency < 100ms
- Market orders execute within 1 second of signal
- Order placement round-trip < 500ms
- Memory stability over 8-hour sessions

Reference: specs/live-trading-execution.md
"""

import asyncio
import logging
import time
import statistics
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics tracked."""
    QUOTE_LATENCY = "quote_latency"
    ORDER_EXECUTION = "order_execution"
    ORDER_ROUND_TRIP = "order_round_trip"
    FEATURE_CALCULATION = "feature_calculation"
    INFERENCE = "inference"
    MEMORY = "memory"


@dataclass
class LatencySample:
    """A single latency measurement."""
    metric_type: MetricType
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Statistics for a latency metric."""
    metric_type: MetricType
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    violations: int = 0
    violation_threshold_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "violations": self.violations,
            "violation_rate_pct": round(self.violations / self.count * 100, 2) if self.count > 0 else 0,
            "threshold_ms": self.violation_threshold_ms,
        }


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    current_mb: float
    peak_mb: float
    allocated_blocks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "allocated_blocks": self.allocated_blocks,
        }


# Performance thresholds from spec
PERFORMANCE_THRESHOLDS = {
    MetricType.QUOTE_LATENCY: 100.0,  # ms - WebSocket quote reception
    MetricType.ORDER_EXECUTION: 1000.0,  # ms - Signal to fill
    MetricType.ORDER_ROUND_TRIP: 500.0,  # ms - Order placement round-trip
    MetricType.FEATURE_CALCULATION: 5.0,  # ms - Feature calc
    MetricType.INFERENCE: 10.0,  # ms - Model inference
}


class PerformanceMonitor:
    """
    Centralized performance monitoring for live trading.

    Tracks latencies, validates against thresholds, and monitors memory.

    Usage:
        monitor = PerformanceMonitor()
        monitor.start()

        # Record latencies
        monitor.record_latency(MetricType.QUOTE_LATENCY, latency_ms=50.5)

        # Get stats
        stats = monitor.get_stats(MetricType.QUOTE_LATENCY)
        print(f"P99 quote latency: {stats.p99_ms}ms")

        # Check for violations
        if monitor.has_violations():
            print("Performance thresholds exceeded!")

        monitor.stop()
    """

    def __init__(
        self,
        max_samples: int = 10000,
        memory_check_interval: float = 60.0,
        on_violation: Optional[Callable[[MetricType, float], None]] = None,
    ):
        """
        Initialize performance monitor.

        Args:
            max_samples: Max samples to keep per metric (circular buffer)
            memory_check_interval: Seconds between memory checks
            on_violation: Callback when threshold exceeded
        """
        self._max_samples = max_samples
        self._memory_check_interval = memory_check_interval
        self._on_violation = on_violation

        # Latency samples (circular buffers)
        self._samples: Dict[MetricType, deque] = {
            metric: deque(maxlen=max_samples)
            for metric in MetricType
            if metric != MetricType.MEMORY
        }

        # Memory samples
        self._memory_samples: deque = deque(maxlen=max_samples)

        # Track violations
        self._violations: Dict[MetricType, int] = {
            metric: 0 for metric in MetricType
        }

        # State
        self._running = False
        self._memory_task: Optional[asyncio.Task] = None
        self._start_time: Optional[datetime] = None

        # Tracemalloc for memory tracking
        self._tracemalloc_started = False

    def start(self) -> None:
        """Start performance monitoring."""
        self._running = True
        self._start_time = datetime.now()

        # Start tracemalloc for memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True

        # Start memory monitoring task (only if event loop is running)
        try:
            loop = asyncio.get_running_loop()
            self._memory_task = loop.create_task(self._memory_monitor_loop())
        except RuntimeError:
            # No event loop running - skip async memory monitoring
            # Memory can still be recorded manually via record_memory()
            self._memory_task = None

        logger.info("Performance monitor started")

    def stop(self) -> None:
        """Stop performance monitoring."""
        self._running = False

        if self._memory_task:
            self._memory_task.cancel()
            self._memory_task = None

        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False

        logger.info("Performance monitor stopped")

    def record_latency(
        self,
        metric_type: MetricType,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a latency measurement.

        Args:
            metric_type: Type of metric
            latency_ms: Latency in milliseconds
            metadata: Optional metadata about the measurement
        """
        if metric_type == MetricType.MEMORY:
            return  # Memory tracked separately

        sample = LatencySample(
            metric_type=metric_type,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        self._samples[metric_type].append(sample)

        # Check threshold
        threshold = PERFORMANCE_THRESHOLDS.get(metric_type)
        if threshold and latency_ms > threshold:
            self._violations[metric_type] += 1

            if self._on_violation:
                self._on_violation(metric_type, latency_ms)

            logger.warning(
                f"Performance violation: {metric_type.value} = {latency_ms:.2f}ms "
                f"(threshold: {threshold}ms)"
            )

    def record_memory(self) -> MemorySnapshot:
        """Record current memory usage."""
        current, peak = tracemalloc.get_traced_memory()

        # Convert bytes to MB
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)

        # Get allocation count
        try:
            stats = tracemalloc.get_traceback_filters()
            allocated_blocks = len(tracemalloc.take_snapshot().statistics('filename'))
        except Exception:
            allocated_blocks = 0

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=current_mb,
            peak_mb=peak_mb,
            allocated_blocks=allocated_blocks,
        )

        self._memory_samples.append(snapshot)

        return snapshot

    def get_stats(self, metric_type: MetricType) -> LatencyStats:
        """
        Get statistics for a metric.

        Args:
            metric_type: Type of metric

        Returns:
            LatencyStats with aggregated statistics
        """
        if metric_type == MetricType.MEMORY:
            # Return memory stats differently
            return self._get_memory_stats()

        samples = self._samples.get(metric_type, deque())

        if not samples:
            return LatencyStats(
                metric_type=metric_type,
                violation_threshold_ms=PERFORMANCE_THRESHOLDS.get(metric_type, 0),
            )

        latencies = [s.latency_ms for s in samples]
        sorted_latencies = sorted(latencies)

        count = len(latencies)
        threshold = PERFORMANCE_THRESHOLDS.get(metric_type, float('inf'))

        return LatencyStats(
            metric_type=metric_type,
            count=count,
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            p50_ms=sorted_latencies[int(count * 0.50)] if count > 0 else 0,
            p95_ms=sorted_latencies[int(count * 0.95)] if count >= 20 else max(latencies),
            p99_ms=sorted_latencies[int(count * 0.99)] if count >= 100 else max(latencies),
            violations=self._violations.get(metric_type, 0),
            violation_threshold_ms=threshold,
        )

    def _get_memory_stats(self) -> LatencyStats:
        """Get memory statistics as LatencyStats-like object."""
        if not self._memory_samples:
            return LatencyStats(metric_type=MetricType.MEMORY)

        memory_values = [s.current_mb for s in self._memory_samples]

        return LatencyStats(
            metric_type=MetricType.MEMORY,
            count=len(memory_values),
            min_ms=min(memory_values),  # Using ms field for MB
            max_ms=max(memory_values),
            mean_ms=statistics.mean(memory_values),
            p50_ms=statistics.median(memory_values),
            p95_ms=sorted(memory_values)[int(len(memory_values) * 0.95)] if len(memory_values) >= 20 else max(memory_values),
            p99_ms=sorted(memory_values)[int(len(memory_values) * 0.99)] if len(memory_values) >= 100 else max(memory_values),
            violations=0,
            violation_threshold_ms=0,
        )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all metrics."""
        return {
            metric.value: self.get_stats(metric)
            for metric in MetricType
        }

    def has_violations(self, metric_type: Optional[MetricType] = None) -> bool:
        """
        Check if there are performance violations.

        Args:
            metric_type: Specific metric to check, or None for any

        Returns:
            True if there are violations
        """
        if metric_type:
            return self._violations.get(metric_type, 0) > 0

        return any(v > 0 for v in self._violations.values())

    def get_violation_count(self, metric_type: Optional[MetricType] = None) -> int:
        """Get total violation count."""
        if metric_type:
            return self._violations.get(metric_type, 0)

        return sum(self._violations.values())

    def get_uptime(self) -> timedelta:
        """Get monitoring uptime."""
        if self._start_time:
            return datetime.now() - self._start_time
        return timedelta()

    def get_memory_trend(self) -> Dict[str, Any]:
        """
        Analyze memory trend to detect leaks.

        Returns:
            Dict with memory trend analysis
        """
        if len(self._memory_samples) < 10:
            return {"status": "insufficient_data", "samples": len(self._memory_samples)}

        samples = list(self._memory_samples)

        # Compare first 10% to last 10%
        early_samples = samples[:len(samples) // 10]
        late_samples = samples[-len(samples) // 10:]

        early_mean = statistics.mean(s.current_mb for s in early_samples)
        late_mean = statistics.mean(s.current_mb for s in late_samples)

        growth_mb = late_mean - early_mean
        growth_pct = (growth_mb / early_mean * 100) if early_mean > 0 else 0

        # Check for concerning growth (>20% over session)
        is_stable = growth_pct < 20

        return {
            "status": "stable" if is_stable else "growing",
            "early_mean_mb": round(early_mean, 2),
            "late_mean_mb": round(late_mean, 2),
            "growth_mb": round(growth_mb, 2),
            "growth_pct": round(growth_pct, 2),
            "samples": len(samples),
        }

    def generate_report(self) -> str:
        """Generate performance report."""
        uptime = self.get_uptime()

        lines = [
            "=" * 60,
            "PERFORMANCE MONITORING REPORT",
            "=" * 60,
            f"Uptime: {uptime}",
            "",
            "LATENCY METRICS:",
            "-" * 40,
        ]

        for metric in MetricType:
            if metric == MetricType.MEMORY:
                continue

            stats = self.get_stats(metric)
            threshold = PERFORMANCE_THRESHOLDS.get(metric, 0)

            status = "OK" if stats.p99_ms <= threshold else "VIOLATION"

            lines.append(
                f"  {metric.value}:"
            )
            lines.append(
                f"    Count: {stats.count}, P99: {stats.p99_ms:.2f}ms, "
                f"Threshold: {threshold}ms [{status}]"
            )

            if stats.violations > 0:
                lines.append(
                    f"    Violations: {stats.violations} ({stats.violations / stats.count * 100:.1f}%)"
                    if stats.count > 0 else f"    Violations: {stats.violations}"
                )

        lines.append("")
        lines.append("MEMORY:")
        lines.append("-" * 40)

        trend = self.get_memory_trend()
        if trend["status"] == "insufficient_data":
            lines.append(f"  Insufficient data ({trend['samples']} samples)")
        else:
            lines.append(f"  Status: {trend['status'].upper()}")
            lines.append(f"  Early mean: {trend['early_mean_mb']} MB")
            lines.append(f"  Late mean: {trend['late_mean_mb']} MB")
            lines.append(f"  Growth: {trend['growth_mb']} MB ({trend['growth_pct']}%)")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    async def _memory_monitor_loop(self) -> None:
        """Background task to monitor memory."""
        while self._running:
            try:
                self.record_memory()
                await asyncio.sleep(self._memory_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "uptime_seconds": self.get_uptime().total_seconds(),
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "metrics": {
                metric.value: self.get_stats(metric).to_dict()
                for metric in MetricType
            },
            "memory_trend": self.get_memory_trend(),
            "total_violations": self.get_violation_count(),
        }


class Timer:
    """
    Context manager for timing operations.

    Usage:
        with Timer(monitor, MetricType.INFERENCE) as t:
            result = model(input)
        print(f"Inference took {t.elapsed_ms}ms")
    """

    def __init__(
        self,
        monitor: Optional[PerformanceMonitor] = None,
        metric_type: Optional[MetricType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize timer.

        Args:
            monitor: Performance monitor to record to (optional)
            metric_type: Type of metric to record (required if monitor provided)
            metadata: Optional metadata
        """
        self._monitor = monitor
        self._metric_type = metric_type
        self._metadata = metadata or {}

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record."""
        self._end_time = time.perf_counter()

        if self._monitor and self._metric_type:
            self._monitor.record_latency(
                self._metric_type,
                self.elapsed_ms,
                self._metadata,
            )

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start_time is None:
            return 0.0

        end = self._end_time or time.perf_counter()
        return (end - self._start_time) * 1000

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000


class AsyncTimer:
    """
    Async context manager for timing operations.

    Usage:
        async with AsyncTimer(monitor, MetricType.ORDER_EXECUTION) as t:
            await place_order()
        print(f"Order execution took {t.elapsed_ms}ms")
    """

    def __init__(
        self,
        monitor: Optional[PerformanceMonitor] = None,
        metric_type: Optional[MetricType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize async timer."""
        self._timer = Timer(monitor, metric_type, metadata)

    async def __aenter__(self) -> "AsyncTimer":
        """Start timing."""
        self._timer.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record."""
        self._timer.__exit__(exc_type, exc_val, exc_tb)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self._timer.elapsed_ms

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self._timer.elapsed_s


def measure_time(
    monitor: Optional[PerformanceMonitor] = None,
    metric_type: Optional[MetricType] = None,
):
    """
    Decorator to measure function execution time.

    Usage:
        @measure_time(monitor, MetricType.FEATURE_CALCULATION)
        def calculate_features(data):
            ...
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with Timer(monitor, metric_type, {"function": func.__name__}):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with AsyncTimer(monitor, metric_type, {"function": func.__name__}):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Global monitor instance (optional)
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def set_global_monitor(monitor: PerformanceMonitor) -> None:
    """Set global performance monitor."""
    global _global_monitor
    _global_monitor = monitor
