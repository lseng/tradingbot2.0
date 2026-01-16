"""
Inference Optimization and Benchmarking for MES Scalping Models.

This module provides:
1. Benchmarking utilities to verify <10ms inference latency
2. Batch inference for efficient backtesting
3. Latency profiling for feature calculation + model inference
4. Hardware-specific benchmarks (CPU, GPU if available)
5. Optional ONNX export for production deployment

Performance Requirements (from specs/live-trading-execution.md):
- Inference Latency: < 10ms per prediction
- Feature Calculation: < 5ms per bar
- Total End-to-End: < 15ms (features + inference + signal generation)

Reference: specs/ml-scalping-model.md, specs/live-trading-execution.md
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any
from datetime import datetime
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from .neural_networks import (
    FeedForwardNet,
    LSTMNet,
    HybridNet,
    ModelPrediction,
    create_model,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    model_type: str
    input_dim: int
    num_classes: int

    # Latency statistics (in milliseconds)
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Sample stats
    num_samples: int
    warmup_iterations: int

    # Hardware info
    device: str
    hardware_info: str = ""

    # Pass/fail against requirements
    meets_requirement: bool = False
    requirement_ms: float = 10.0

    def __str__(self) -> str:
        status = "✓ PASS" if self.meets_requirement else "✗ FAIL"
        return (
            f"{self.model_name} ({self.model_type}) on {self.device}: "
            f"mean={self.mean_latency_ms:.2f}ms, p95={self.p95_latency_ms:.2f}ms, "
            f"p99={self.p99_latency_ms:.2f}ms [{status}]"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "num_samples": self.num_samples,
            "warmup_iterations": self.warmup_iterations,
            "device": self.device,
            "hardware_info": self.hardware_info,
            "meets_requirement": self.meets_requirement,
            "requirement_ms": self.requirement_ms,
        }


@dataclass
class EndToEndBenchmarkResult:
    """Results from end-to-end benchmark (features + inference)."""
    model_benchmark: BenchmarkResult
    feature_mean_ms: float
    feature_p95_ms: float
    total_mean_ms: float
    total_p95_ms: float
    meets_total_requirement: bool  # < 15ms total

    def __str__(self) -> str:
        status = "✓ PASS" if self.meets_total_requirement else "✗ FAIL"
        return (
            f"End-to-End: features={self.feature_mean_ms:.2f}ms, "
            f"inference={self.model_benchmark.mean_latency_ms:.2f}ms, "
            f"total={self.total_mean_ms:.2f}ms (p95={self.total_p95_ms:.2f}ms) [{status}]"
        )


class InferenceBenchmark:
    """
    Benchmarking utilities for model inference latency.

    Usage:
        benchmark = InferenceBenchmark()

        # Benchmark a single model
        result = benchmark.benchmark_model(model, input_dim=50)
        print(result)

        # Benchmark all model types
        results = benchmark.benchmark_all_models(input_dim=50)
        for r in results:
            print(r)
    """

    def __init__(
        self,
        requirement_ms: float = 10.0,
        warmup_iterations: int = 100,
        num_iterations: int = 1000,
        device: Optional[str] = None,
    ):
        """
        Initialize benchmark.

        Args:
            requirement_ms: Maximum allowed latency in milliseconds
            warmup_iterations: Number of warmup iterations (not measured)
            num_iterations: Number of iterations to measure
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.requirement_ms = requirement_ms
        self.warmup_iterations = warmup_iterations
        self.num_iterations = num_iterations

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.hardware_info = self._get_hardware_info()
        logger.info(f"InferenceBenchmark initialized on {self.device}: {self.hardware_info}")

    def _get_hardware_info(self) -> str:
        """Get hardware information string."""
        import platform

        info_parts = [platform.processor() or platform.machine()]

        if self.device.type == 'cuda' and torch.cuda.is_available():
            info_parts.append(torch.cuda.get_device_name(0))
        elif self.device.type == 'mps':
            info_parts.append("Apple Silicon")

        return ", ".join(info_parts)

    def benchmark_model(
        self,
        model: nn.Module,
        input_dim: int,
        seq_length: int = 20,
        batch_size: int = 1,
        model_name: str = "model",
        model_type: str = "unknown",
    ) -> BenchmarkResult:
        """
        Benchmark a single model.

        Args:
            model: PyTorch model to benchmark
            input_dim: Input feature dimension
            seq_length: Sequence length for LSTM models
            batch_size: Batch size (1 for live trading, larger for backtest)
            model_name: Name for reporting
            model_type: Model type for reporting

        Returns:
            BenchmarkResult with latency statistics
        """
        model = model.to(self.device)
        model.eval()

        # Determine input shape based on model type
        is_lstm = isinstance(model, LSTMNet)
        is_hybrid = isinstance(model, HybridNet)

        if is_hybrid:
            # HybridNet needs sequential and static inputs
            seq_input = torch.randn(batch_size, seq_length, model.lstm.input_size, device=self.device)
            static_input = torch.randn(batch_size, model.mlp[0].in_features, device=self.device)
            inputs = (seq_input, static_input)
        elif is_lstm:
            inputs = (torch.randn(batch_size, seq_length, input_dim, device=self.device),)
        else:
            inputs = (torch.randn(batch_size, input_dim, device=self.device),)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                if is_lstm:
                    _ = model(*inputs)[0]  # LSTM returns (output, hidden)
                else:
                    _ = model(*inputs)

        # Synchronize if using GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        latencies_ms = []
        with torch.no_grad():
            for _ in range(self.num_iterations):
                start = time.perf_counter()

                if is_lstm:
                    _ = model(*inputs)[0]
                else:
                    _ = model(*inputs)

                # Synchronize if using GPU
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies_ms.append((end - start) * 1000)

        # Calculate statistics
        latencies_sorted = sorted(latencies_ms)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p99_idx = int(len(latencies_sorted) * 0.99)

        num_classes = getattr(model, 'num_classes', 3)

        result = BenchmarkResult(
            model_name=model_name,
            model_type=model_type,
            input_dim=input_dim,
            num_classes=num_classes,
            mean_latency_ms=statistics.mean(latencies_ms),
            median_latency_ms=statistics.median(latencies_ms),
            std_latency_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            min_latency_ms=min(latencies_ms),
            max_latency_ms=max(latencies_ms),
            p95_latency_ms=latencies_sorted[p95_idx],
            p99_latency_ms=latencies_sorted[p99_idx],
            num_samples=self.num_iterations,
            warmup_iterations=self.warmup_iterations,
            device=str(self.device),
            hardware_info=self.hardware_info,
            requirement_ms=self.requirement_ms,
        )

        # Check if meets requirement (p99 should be under requirement)
        result.meets_requirement = result.p99_latency_ms < self.requirement_ms

        return result

    def benchmark_all_models(
        self,
        input_dim: int = 50,
        num_classes: int = 3,
        seq_length: int = 20,
    ) -> List[BenchmarkResult]:
        """
        Benchmark all model types.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            seq_length: Sequence length for LSTM models

        Returns:
            List of BenchmarkResult for each model type
        """
        results = []

        # FeedForward configurations
        ff_configs = [
            {"hidden_dims": [128, 64, 32], "name": "FeedForward-128-64-32"},
            {"hidden_dims": [64, 32], "name": "FeedForward-64-32"},
            {"hidden_dims": [256, 128, 64, 32], "name": "FeedForward-256-128-64-32"},
        ]

        for config in ff_configs:
            model = FeedForwardNet(
                input_dim=input_dim,
                hidden_dims=config["hidden_dims"],
                num_classes=num_classes,
            )
            result = self.benchmark_model(
                model,
                input_dim=input_dim,
                model_name=config["name"],
                model_type="feedforward",
            )
            results.append(result)
            logger.info(str(result))

        # LSTM configurations
        lstm_configs = [
            {"hidden_dim": 64, "num_layers": 2, "name": "LSTM-64-2layer"},
            {"hidden_dim": 32, "num_layers": 1, "name": "LSTM-32-1layer"},
        ]

        for config in lstm_configs:
            model = LSTMNet(
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                num_classes=num_classes,
            )
            result = self.benchmark_model(
                model,
                input_dim=input_dim,
                seq_length=seq_length,
                model_name=config["name"],
                model_type="lstm",
            )
            results.append(result)
            logger.info(str(result))

        # Hybrid configuration
        model = HybridNet(
            seq_input_dim=input_dim,
            static_input_dim=input_dim // 2,
            lstm_hidden=32,
            lstm_layers=1,
            mlp_hidden=[64, 32],
            num_classes=num_classes,
        )
        result = self.benchmark_model(
            model,
            input_dim=input_dim,
            seq_length=seq_length,
            model_name="Hybrid-32lstm-64-32mlp",
            model_type="hybrid",
        )
        results.append(result)
        logger.info(str(result))

        return results


class BatchInference:
    """
    Batch inference for efficient backtesting.

    Processes multiple samples at once for significantly faster backtests.

    Usage:
        batch_inference = BatchInference(model, batch_size=256)

        # Add samples to queue
        for features in all_features:
            batch_inference.add(features)

        # Get all predictions
        predictions = batch_inference.flush()
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        """
        Initialize batch inference.

        Args:
            model: PyTorch model
            batch_size: Number of samples to process at once
            device: Device to run on
        """
        self.model = model
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Sample queue
        self._queue: List[np.ndarray] = []
        self._predictions: List[ModelPrediction] = []

        # Track if model is LSTM-based
        self._is_lstm = isinstance(model, LSTMNet)
        self._is_hybrid = isinstance(model, HybridNet)

        logger.info(f"BatchInference initialized with batch_size={batch_size} on {self.device}")

    def add(
        self,
        features: Union[np.ndarray, torch.Tensor],
        volatility: float = 0.0,
    ) -> Optional[List[ModelPrediction]]:
        """
        Add a sample to the queue.

        Args:
            features: Feature vector or array
            volatility: ATR for position sizing

        Returns:
            List of predictions if batch was processed, None otherwise
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        self._queue.append((features, volatility))

        if len(self._queue) >= self.batch_size:
            return self._process_batch()

        return None

    def _process_batch(self) -> List[ModelPrediction]:
        """Process current batch."""
        if not self._queue:
            return []

        # Stack features
        features_list = [item[0] for item in self._queue]
        volatilities = [item[1] for item in self._queue]

        features_batch = np.stack(features_list)
        features_tensor = torch.tensor(features_batch, dtype=torch.float32, device=self.device)

        # Run inference
        with torch.no_grad():
            if self._is_lstm:
                # Add sequence dimension if needed
                if features_tensor.dim() == 2:
                    features_tensor = features_tensor.unsqueeze(1)  # (batch, 1, features)
                logits, _ = self.model(features_tensor)
            elif self._is_hybrid:
                # Split features for hybrid model
                seq_features = features_tensor.unsqueeze(1)
                static_features = features_tensor[:, :features_tensor.shape[1] // 2]
                logits = self.model(seq_features, static_features)
            else:
                logits = self.model(features_tensor)

        # Convert to predictions
        predictions = []
        for i in range(logits.shape[0]):
            pred = ModelPrediction.from_logits(
                logits[i],
                volatility=volatilities[i],
            )
            predictions.append(pred)

        # Clear queue
        self._queue.clear()

        return predictions

    def flush(self) -> List[ModelPrediction]:
        """
        Flush any remaining samples in queue.

        Returns:
            List of all predictions from remaining samples
        """
        if not self._queue:
            return []
        return self._process_batch()

    def process_all(
        self,
        features_array: np.ndarray,
        volatilities: Optional[np.ndarray] = None,
    ) -> List[ModelPrediction]:
        """
        Process all samples at once with automatic batching.

        Args:
            features_array: Array of shape (num_samples, num_features)
            volatilities: Optional array of volatilities per sample

        Returns:
            List of ModelPrediction for all samples
        """
        num_samples = features_array.shape[0]

        if volatilities is None:
            volatilities = np.zeros(num_samples)

        all_predictions = []

        # Process in batches
        for i in range(0, num_samples, self.batch_size):
            batch_end = min(i + self.batch_size, num_samples)
            batch_features = features_array[i:batch_end]
            batch_volatilities = volatilities[i:batch_end]

            # Add all samples in this batch
            for j, (feat, vol) in enumerate(zip(batch_features, batch_volatilities)):
                self._queue.append((feat, vol))

            # Process batch
            predictions = self._process_batch()
            all_predictions.extend(predictions)

        return all_predictions


def benchmark_feature_calculation(
    num_iterations: int = 1000,
    warmup_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark feature calculation latency.

    Returns:
        Dictionary with latency statistics in milliseconds
    """
    from src.trading.rt_features import RealTimeFeatureEngine, OHLCV, RTFeaturesConfig

    # Initialize engine
    config = RTFeaturesConfig()
    engine = RealTimeFeatureEngine(config)

    # Create sample bars for warmup (need enough for EMA initialization)
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    for i in range(210):  # Need 200+ bars for max EMA period
        bar = OHLCV(
            timestamp=base_time.replace(second=i % 60, minute=10 + i // 60),
            open=5000.0 + np.random.randn() * 2,
            high=5002.0 + np.random.randn() * 2,
            low=4998.0 + np.random.randn() * 2,
            close=5001.0 + np.random.randn() * 2,
            volume=int(100 + np.random.randn() * 20),
        )
        engine.update(bar)

    # Warmup
    for i in range(warmup_iterations):
        bar = OHLCV(
            timestamp=base_time.replace(second=i % 60, minute=15 + i // 60),
            open=5000.0 + np.random.randn() * 2,
            high=5002.0 + np.random.randn() * 2,
            low=4998.0 + np.random.randn() * 2,
            close=5001.0 + np.random.randn() * 2,
            volume=int(100 + np.random.randn() * 20),
        )
        _ = engine.update(bar)

    # Benchmark
    latencies_ms = []
    for i in range(num_iterations):
        bar = OHLCV(
            timestamp=base_time.replace(second=i % 60, minute=20 + i // 60),
            open=5000.0 + np.random.randn() * 2,
            high=5002.0 + np.random.randn() * 2,
            low=4998.0 + np.random.randn() * 2,
            close=5001.0 + np.random.randn() * 2,
            volume=int(100 + np.random.randn() * 20),
        )

        start = time.perf_counter()
        _ = engine.update(bar)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    latencies_sorted = sorted(latencies_ms)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p99_idx = int(len(latencies_sorted) * 0.99)

    return {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p95_ms": latencies_sorted[p95_idx],
        "p99_ms": latencies_sorted[p99_idx],
        "meets_requirement": latencies_sorted[p99_idx] < 5.0,  # < 5ms requirement
    }


def run_full_benchmark(
    input_dim: int = 50,
    num_classes: int = 3,
) -> Dict[str, Any]:
    """
    Run full benchmark suite (features + all model types).

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes

    Returns:
        Dictionary with all benchmark results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "models": [],
        "features": None,
        "summary": {},
    }

    # Benchmark features
    logger.info("Benchmarking feature calculation...")
    try:
        feature_results = benchmark_feature_calculation()
        results["features"] = feature_results
        logger.info(f"Feature calculation: mean={feature_results['mean_ms']:.2f}ms, p99={feature_results['p99_ms']:.2f}ms")
    except Exception as e:
        logger.warning(f"Feature benchmark failed: {e}")
        results["features"] = {"error": str(e)}

    # Benchmark models
    logger.info("Benchmarking models...")
    benchmark = InferenceBenchmark()
    model_results = benchmark.benchmark_all_models(input_dim=input_dim, num_classes=num_classes)

    for r in model_results:
        results["models"].append(r.to_dict())

    # Summary
    all_pass = all(r.meets_requirement for r in model_results)
    fastest = min(model_results, key=lambda r: r.mean_latency_ms)

    results["summary"] = {
        "all_models_pass": all_pass,
        "fastest_model": fastest.model_name,
        "fastest_mean_ms": fastest.mean_latency_ms,
        "fastest_p99_ms": fastest.p99_latency_ms,
        "device": str(benchmark.device),
        "hardware_info": benchmark.hardware_info,
    }

    logger.info(f"Benchmark complete. All models pass: {all_pass}")
    logger.info(f"Fastest model: {fastest.model_name} ({fastest.mean_latency_ms:.2f}ms mean)")

    return results


def verify_latency_requirements() -> bool:
    """
    Verify that the system meets latency requirements.

    Requirements:
    - Inference: < 10ms (p99)
    - Features: < 5ms (p99)
    - Total: < 15ms

    Returns:
        True if all requirements are met
    """
    logger.info("Verifying latency requirements...")

    results = run_full_benchmark()

    # Check inference
    models_pass = results["summary"]["all_models_pass"]

    # Check features
    features_pass = True
    if results["features"] and "meets_requirement" in results["features"]:
        features_pass = results["features"]["meets_requirement"]

    all_pass = models_pass and features_pass

    if all_pass:
        logger.info("✓ All latency requirements met!")
    else:
        if not models_pass:
            logger.error("✗ Model inference latency exceeds 10ms requirement")
        if not features_pass:
            logger.error("✗ Feature calculation latency exceeds 5ms requirement")

    return all_pass


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("MES Scalping Model Inference Benchmark")
    print("=" * 70)
    print()

    # Run full benchmark
    results = run_full_benchmark()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Device: {results['summary']['device']}")
    print(f"Hardware: {results['summary']['hardware_info']}")
    print()

    if results["features"] and "mean_ms" in results["features"]:
        print("Feature Calculation:")
        f = results["features"]
        status = "✓ PASS" if f.get("meets_requirement", False) else "✗ FAIL"
        print(f"  Mean: {f['mean_ms']:.2f}ms, P99: {f['p99_ms']:.2f}ms [{status}]")
        print()

    print("Model Inference:")
    for m in results["models"]:
        status = "✓ PASS" if m["meets_requirement"] else "✗ FAIL"
        print(f"  {m['model_name']}: mean={m['mean_latency_ms']:.2f}ms, p99={m['p99_latency_ms']:.2f}ms [{status}]")

    print()
    print(f"All Models Pass: {results['summary']['all_models_pass']}")
    print(f"Fastest Model: {results['summary']['fastest_model']} ({results['summary']['fastest_mean_ms']:.2f}ms)")
    print()

    # Final verification
    print("=" * 70)
    if verify_latency_requirements():
        print("RESULT: ✓ System meets all latency requirements for live trading")
    else:
        print("RESULT: ✗ System does NOT meet latency requirements")
    print("=" * 70)
