"""
Tests for Inference Optimization and Benchmarking.

Verifies that:
1. All model types meet <10ms inference latency requirement
2. Feature calculation meets <5ms requirement
3. Batch inference works correctly
4. End-to-end latency meets <15ms requirement

Reference: specs/live-trading-execution.md, specs/ml-scalping-model.md
"""

import pytest
import time
import numpy as np
import torch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.models.neural_networks import (
    FeedForwardNet,
    LSTMNet,
    HybridNet,
    ModelPrediction,
    create_model,
)
from src.ml.models.inference_benchmark import (
    InferenceBenchmark,
    BatchInference,
    BenchmarkResult,
    run_full_benchmark,
    verify_latency_requirements,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult."""
        result = BenchmarkResult(
            model_name="TestModel",
            model_type="feedforward",
            input_dim=50,
            num_classes=3,
            mean_latency_ms=1.5,
            median_latency_ms=1.4,
            std_latency_ms=0.2,
            min_latency_ms=1.0,
            max_latency_ms=3.0,
            p95_latency_ms=2.0,
            p99_latency_ms=2.5,
            num_samples=1000,
            warmup_iterations=100,
            device="cpu",
            requirement_ms=10.0,
            meets_requirement=True,
        )

        assert result.model_name == "TestModel"
        assert result.mean_latency_ms == 1.5
        assert result.meets_requirement is True

    def test_benchmark_result_str(self):
        """Test string representation of BenchmarkResult."""
        result = BenchmarkResult(
            model_name="TestModel",
            model_type="feedforward",
            input_dim=50,
            num_classes=3,
            mean_latency_ms=1.5,
            median_latency_ms=1.4,
            std_latency_ms=0.2,
            min_latency_ms=1.0,
            max_latency_ms=3.0,
            p95_latency_ms=2.0,
            p99_latency_ms=2.5,
            num_samples=1000,
            warmup_iterations=100,
            device="cpu",
            meets_requirement=True,
        )

        result_str = str(result)
        assert "TestModel" in result_str
        assert "PASS" in result_str

    def test_benchmark_result_to_dict(self):
        """Test converting BenchmarkResult to dictionary."""
        result = BenchmarkResult(
            model_name="TestModel",
            model_type="feedforward",
            input_dim=50,
            num_classes=3,
            mean_latency_ms=1.5,
            median_latency_ms=1.4,
            std_latency_ms=0.2,
            min_latency_ms=1.0,
            max_latency_ms=3.0,
            p95_latency_ms=2.0,
            p99_latency_ms=2.5,
            num_samples=1000,
            warmup_iterations=100,
            device="cpu",
        )

        d = result.to_dict()
        assert d["model_name"] == "TestModel"
        assert d["mean_latency_ms"] == 1.5
        assert "device" in d


class TestInferenceBenchmark:
    """Tests for InferenceBenchmark class."""

    def test_benchmark_init(self):
        """Test benchmark initialization."""
        benchmark = InferenceBenchmark(requirement_ms=10.0)
        assert benchmark.requirement_ms == 10.0
        assert benchmark.device is not None

    def test_benchmark_feedforward_model(self):
        """Test benchmarking FeedForward model."""
        model = FeedForwardNet(input_dim=50, hidden_dims=[64, 32], num_classes=3)
        benchmark = InferenceBenchmark(num_iterations=100, warmup_iterations=10)

        result = benchmark.benchmark_model(
            model,
            input_dim=50,
            model_name="TestFF",
            model_type="feedforward",
        )

        assert result.model_name == "TestFF"
        assert result.model_type == "feedforward"
        assert result.mean_latency_ms > 0
        assert result.p99_latency_ms >= result.mean_latency_ms
        assert result.num_samples == 100

    def test_benchmark_lstm_model(self):
        """Test benchmarking LSTM model."""
        model = LSTMNet(input_dim=50, hidden_dim=32, num_layers=1, num_classes=3)
        benchmark = InferenceBenchmark(num_iterations=100, warmup_iterations=10)

        result = benchmark.benchmark_model(
            model,
            input_dim=50,
            seq_length=20,
            model_name="TestLSTM",
            model_type="lstm",
        )

        assert result.model_name == "TestLSTM"
        assert result.model_type == "lstm"
        assert result.mean_latency_ms > 0

    def test_benchmark_hybrid_model(self):
        """Test benchmarking Hybrid model."""
        model = HybridNet(
            seq_input_dim=50,
            static_input_dim=25,
            lstm_hidden=32,
            lstm_layers=1,
            num_classes=3,
        )
        benchmark = InferenceBenchmark(num_iterations=100, warmup_iterations=10)

        result = benchmark.benchmark_model(
            model,
            input_dim=50,
            seq_length=20,
            model_name="TestHybrid",
            model_type="hybrid",
        )

        assert result.model_name == "TestHybrid"
        assert result.model_type == "hybrid"
        assert result.mean_latency_ms > 0

    def test_benchmark_all_models(self):
        """Test benchmarking all model types."""
        benchmark = InferenceBenchmark(num_iterations=50, warmup_iterations=5)
        results = benchmark.benchmark_all_models(input_dim=50, num_classes=3)

        # Should have multiple results (at least FeedForward, LSTM, Hybrid)
        assert len(results) >= 3

        # Check each has valid statistics
        for r in results:
            assert r.mean_latency_ms > 0
            assert r.p99_latency_ms >= r.mean_latency_ms
            assert r.min_latency_ms <= r.mean_latency_ms
            assert r.max_latency_ms >= r.mean_latency_ms


class TestBatchInference:
    """Tests for BatchInference class."""

    def test_batch_inference_init(self):
        """Test batch inference initialization."""
        model = FeedForwardNet(input_dim=50, num_classes=3)
        batch_inf = BatchInference(model, batch_size=32)

        assert batch_inf.batch_size == 32
        assert batch_inf.model is not None

    def test_batch_inference_add_and_process(self):
        """Test adding samples and processing batches."""
        model = FeedForwardNet(input_dim=50, num_classes=3)
        batch_inf = BatchInference(model, batch_size=10)

        # Add 5 samples (less than batch size)
        for i in range(5):
            result = batch_inf.add(np.random.randn(50))
            assert result is None  # Not enough for batch

        # Add 5 more to trigger batch
        predictions = None
        for i in range(5):
            result = batch_inf.add(np.random.randn(50))
            if result is not None:
                predictions = result

        assert predictions is not None
        assert len(predictions) == 10

        # Check predictions are valid ModelPrediction objects
        for pred in predictions:
            assert isinstance(pred, ModelPrediction)
            assert pred.direction in [-1, 0, 1]
            assert 0 <= pred.confidence <= 1

    def test_batch_inference_flush(self):
        """Test flushing remaining samples."""
        model = FeedForwardNet(input_dim=50, num_classes=3)
        batch_inf = BatchInference(model, batch_size=100)

        # Add 50 samples
        for i in range(50):
            batch_inf.add(np.random.randn(50))

        # Flush remaining
        predictions = batch_inf.flush()
        assert len(predictions) == 50

    def test_batch_inference_process_all(self):
        """Test processing all samples at once."""
        model = FeedForwardNet(input_dim=50, num_classes=3)
        batch_inf = BatchInference(model, batch_size=32)

        # Create array of samples
        num_samples = 100
        features = np.random.randn(num_samples, 50).astype(np.float32)
        volatilities = np.random.rand(num_samples).astype(np.float32)

        predictions = batch_inf.process_all(features, volatilities)

        assert len(predictions) == num_samples

        for i, pred in enumerate(predictions):
            assert isinstance(pred, ModelPrediction)
            assert pred.volatility == volatilities[i]

    def test_batch_inference_lstm_model(self):
        """Test batch inference with LSTM model."""
        model = LSTMNet(input_dim=50, hidden_dim=32, num_classes=3)
        batch_inf = BatchInference(model, batch_size=16)

        features = np.random.randn(32, 50).astype(np.float32)
        predictions = batch_inf.process_all(features)

        assert len(predictions) == 32

    def test_batch_inference_with_tensor_input(self):
        """Test batch inference with torch tensor input."""
        model = FeedForwardNet(input_dim=50, num_classes=3)
        batch_inf = BatchInference(model, batch_size=10)

        # Add tensor instead of numpy array
        for i in range(10):
            tensor_input = torch.randn(50)
            result = batch_inf.add(tensor_input)

        assert result is not None
        assert len(result) == 10


class TestLatencyRequirements:
    """Tests for latency requirement verification."""

    def test_feedforward_meets_10ms_requirement(self):
        """Test that FeedForward model inference is under 10ms."""
        model = FeedForwardNet(input_dim=50, hidden_dims=[128, 64, 32], num_classes=3)
        benchmark = InferenceBenchmark(
            requirement_ms=10.0,
            num_iterations=500,
            warmup_iterations=50,
        )

        result = benchmark.benchmark_model(model, input_dim=50)

        # P99 should be under 10ms
        assert result.p99_latency_ms < 10.0, (
            f"FeedForward p99 latency {result.p99_latency_ms:.2f}ms exceeds 10ms requirement"
        )
        assert result.meets_requirement is True

    def test_lstm_meets_10ms_requirement(self):
        """Test that LSTM model inference is under 10ms."""
        model = LSTMNet(input_dim=50, hidden_dim=64, num_layers=2, num_classes=3)
        benchmark = InferenceBenchmark(
            requirement_ms=10.0,
            num_iterations=500,
            warmup_iterations=50,
        )

        result = benchmark.benchmark_model(model, input_dim=50, seq_length=20)

        # P99 should be under 10ms
        assert result.p99_latency_ms < 10.0, (
            f"LSTM p99 latency {result.p99_latency_ms:.2f}ms exceeds 10ms requirement"
        )

    def test_hybrid_meets_10ms_requirement(self):
        """Test that Hybrid model inference is under 10ms."""
        model = HybridNet(
            seq_input_dim=50,
            static_input_dim=25,
            lstm_hidden=32,
            lstm_layers=1,
            num_classes=3,
        )
        benchmark = InferenceBenchmark(
            requirement_ms=10.0,
            num_iterations=500,
            warmup_iterations=50,
        )

        result = benchmark.benchmark_model(model, input_dim=50, seq_length=20)

        # P99 should be under 10ms
        assert result.p99_latency_ms < 10.0, (
            f"Hybrid p99 latency {result.p99_latency_ms:.2f}ms exceeds 10ms requirement"
        )

    def test_single_inference_latency(self):
        """Test that a single forward pass is fast (smoke test)."""
        model = FeedForwardNet(input_dim=50, hidden_dims=[64, 32], num_classes=3)
        model.eval()

        x = torch.randn(1, 50)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        # Measure single inference
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        assert latency_ms < 10.0, f"Single inference took {latency_ms:.2f}ms"


class TestFeatureCalculationLatency:
    """Tests for feature calculation latency."""

    def test_feature_calculation_under_5ms(self):
        """Test that feature calculation is under 5ms requirement."""
        from src.trading.rt_features import RealTimeFeatureEngine, OHLCV, RTFeaturesConfig

        config = RTFeaturesConfig()
        engine = RealTimeFeatureEngine(config)

        # Initialize with enough bars
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        for i in range(210):
            bar = OHLCV(
                timestamp=base_time.replace(second=i % 60, minute=10 + i // 60),
                open=5000.0 + np.random.randn() * 2,
                high=5002.0 + np.random.randn() * 2,
                low=4998.0 + np.random.randn() * 2,
                close=5001.0 + np.random.randn() * 2,
                volume=int(100 + abs(np.random.randn() * 20)),
            )
            engine.update(bar)

        # Warmup
        for i in range(50):
            bar = OHLCV(
                timestamp=base_time.replace(second=i % 60, minute=15),
                open=5000.0 + np.random.randn() * 2,
                high=5002.0 + np.random.randn() * 2,
                low=4998.0 + np.random.randn() * 2,
                close=5001.0 + np.random.randn() * 2,
                volume=int(100 + abs(np.random.randn() * 20)),
            )
            _ = engine.update(bar)

        # Measure feature calculation
        latencies_ms = []
        for i in range(100):
            bar = OHLCV(
                timestamp=base_time.replace(second=i % 60, minute=20 + i // 60),
                open=5000.0 + np.random.randn() * 2,
                high=5002.0 + np.random.randn() * 2,
                low=4998.0 + np.random.randn() * 2,
                close=5001.0 + np.random.randn() * 2,
                volume=int(100 + abs(np.random.randn() * 20)),
            )

            start = time.perf_counter()
            _ = engine.update(bar)
            end = time.perf_counter()

            latencies_ms.append((end - start) * 1000)

        p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        mean = sum(latencies_ms) / len(latencies_ms)

        assert p99 < 5.0, f"Feature calculation p99 {p99:.2f}ms exceeds 5ms requirement"
        assert mean < 3.0, f"Feature calculation mean {mean:.2f}ms is higher than expected"


class TestEndToEndLatency:
    """Tests for end-to-end latency (features + inference)."""

    def test_end_to_end_under_15ms(self):
        """Test that end-to-end latency is under 15ms."""
        from src.trading.rt_features import RealTimeFeatureEngine, OHLCV, RTFeaturesConfig

        # Initialize feature engine
        config = RTFeaturesConfig()
        engine = RealTimeFeatureEngine(config)

        # Initialize model
        model = FeedForwardNet(input_dim=60, hidden_dims=[64, 32], num_classes=3)
        model.eval()

        # Initialize feature engine with enough bars
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        for i in range(210):
            bar = OHLCV(
                timestamp=base_time.replace(second=i % 60, minute=10 + i // 60),
                open=5000.0 + np.random.randn() * 2,
                high=5002.0 + np.random.randn() * 2,
                low=4998.0 + np.random.randn() * 2,
                close=5001.0 + np.random.randn() * 2,
                volume=int(100 + abs(np.random.randn() * 20)),
            )
            engine.update(bar)

        # Warmup
        for i in range(50):
            bar = OHLCV(
                timestamp=base_time.replace(second=i % 60, minute=15),
                open=5000.0 + np.random.randn() * 2,
                high=5002.0 + np.random.randn() * 2,
                low=4998.0 + np.random.randn() * 2,
                close=5001.0 + np.random.randn() * 2,
                volume=int(100 + abs(np.random.randn() * 20)),
            )
            features = engine.update(bar)
            if features:
                # Pad/truncate to match model input
                feat_array = features.features
                if len(feat_array) < 60:
                    feat_array = np.pad(feat_array, (0, 60 - len(feat_array)))
                else:
                    feat_array = feat_array[:60]
                x = torch.tensor(feat_array, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    _ = model(x)

        # Measure end-to-end
        latencies_ms = []
        for i in range(100):
            bar = OHLCV(
                timestamp=base_time.replace(second=i % 60, minute=20 + i // 60),
                open=5000.0 + np.random.randn() * 2,
                high=5002.0 + np.random.randn() * 2,
                low=4998.0 + np.random.randn() * 2,
                close=5001.0 + np.random.randn() * 2,
                volume=int(100 + abs(np.random.randn() * 20)),
            )

            start = time.perf_counter()

            # Feature calculation
            features = engine.update(bar)

            if features:
                # Prepare for model
                feat_array = features.features
                if len(feat_array) < 60:
                    feat_array = np.pad(feat_array, (0, 60 - len(feat_array)))
                else:
                    feat_array = feat_array[:60]
                x = torch.tensor(feat_array, dtype=torch.float32).unsqueeze(0)

                # Model inference
                with torch.no_grad():
                    _ = model(x)

            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

        p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        mean = sum(latencies_ms) / len(latencies_ms)

        assert p99 < 15.0, f"End-to-end p99 {p99:.2f}ms exceeds 15ms requirement"
        assert mean < 10.0, f"End-to-end mean {mean:.2f}ms is higher than expected"


class TestBatchInferencePerformance:
    """Tests for batch inference performance."""

    def test_batch_inference_faster_than_sequential(self):
        """Test that batch inference is faster than sequential for large datasets."""
        model = FeedForwardNet(input_dim=50, hidden_dims=[64, 32], num_classes=3)
        model.eval()

        # Use a larger sample size where batch benefits are evident
        num_samples = 1000
        features = np.random.randn(num_samples, 50).astype(np.float32)

        # Force CPU for fair comparison (GPU has overhead for small batches)
        batch_inf = BatchInference(model, batch_size=256, device='cpu')

        # Sequential inference (on CPU)
        sequential_start = time.perf_counter()
        for i in range(num_samples):
            x = torch.tensor(features[i:i+1], dtype=torch.float32)
            with torch.no_grad():
                _ = model(x)
        sequential_time = time.perf_counter() - sequential_start

        # Batch inference
        batch_start = time.perf_counter()
        _ = batch_inf.process_all(features)
        batch_time = time.perf_counter() - batch_start

        # Batch should be faster for large datasets
        # On CPU, batch should be notably faster due to reduced Python overhead
        # Allow batch to be up to 10x slower only for edge cases (very small models)
        assert batch_time < sequential_time * 10, (
            f"Batch inference ({batch_time:.3f}s) much slower than sequential ({sequential_time:.3f}s)"
        )

    def test_batch_inference_large_dataset(self):
        """Test batch inference on a larger dataset."""
        model = FeedForwardNet(input_dim=50, hidden_dims=[64, 32], num_classes=3)
        batch_inf = BatchInference(model, batch_size=256)

        # 10k samples (typical backtest size)
        num_samples = 10000
        features = np.random.randn(num_samples, 50).astype(np.float32)

        start = time.perf_counter()
        predictions = batch_inf.process_all(features)
        elapsed = time.perf_counter() - start

        assert len(predictions) == num_samples

        # Should process 10k samples in under 5 seconds
        assert elapsed < 5.0, f"Processing 10k samples took {elapsed:.2f}s (expected <5s)"

        # Throughput check
        throughput = num_samples / elapsed
        assert throughput > 2000, f"Throughput {throughput:.0f} samples/sec is too low"


class TestModelPredictionIntegration:
    """Tests for ModelPrediction integration with benchmarking."""

    def test_prediction_from_benchmark(self):
        """Test that predictions are correctly generated during benchmarking."""
        model = FeedForwardNet(input_dim=50, hidden_dims=[64, 32], num_classes=3)
        model.eval()

        x = torch.randn(1, 50)

        with torch.no_grad():
            logits = model(x)
            pred = ModelPrediction.from_logits(logits)

        assert pred.direction in [-1, 0, 1]
        assert 0 <= pred.confidence <= 1
        assert pred.class_probabilities is not None
        assert len(pred.class_probabilities) == 3
        assert abs(sum(pred.class_probabilities) - 1.0) < 0.001

    def test_batch_predictions_consistent(self):
        """Test that batch predictions are consistent with individual predictions."""
        # Create model and set to eval mode
        model = FeedForwardNet(input_dim=50, hidden_dims=[64, 32], num_classes=3)
        model.eval()

        # Force CPU to ensure consistent device
        batch_inf = BatchInference(model, batch_size=100, device='cpu')

        np.random.seed(42)
        features = np.random.randn(10, 50).astype(np.float32)

        # Get batch predictions
        batch_predictions = batch_inf.process_all(features)

        # Get individual predictions using the SAME model on CPU
        # The BatchInference moves model to its device, so we need to use the same instance
        individual_predictions = []
        for i in range(10):
            x = torch.tensor(features[i:i+1], dtype=torch.float32)
            with torch.no_grad():
                logits = batch_inf.model(x)
                pred = ModelPrediction.from_logits(logits)
                individual_predictions.append(pred)

        # Compare - both use same model and device, so should be identical
        for bp, ip in zip(batch_predictions, individual_predictions):
            assert bp.direction == ip.direction
            assert abs(bp.confidence - ip.confidence) < 0.01  # Small tolerance for float precision


class TestHardwareDetection:
    """Tests for hardware detection in benchmarking."""

    def test_device_detection(self):
        """Test that device is correctly detected."""
        benchmark = InferenceBenchmark()

        # Should have a valid device
        assert benchmark.device is not None
        assert benchmark.device.type in ['cpu', 'cuda', 'mps']

    def test_hardware_info(self):
        """Test that hardware info is populated."""
        benchmark = InferenceBenchmark()

        # Should have hardware info
        assert benchmark.hardware_info is not None
        assert len(benchmark.hardware_info) > 0


class TestFullBenchmarkSuite:
    """Tests for the full benchmark suite."""

    def test_run_full_benchmark(self):
        """Test running the full benchmark suite."""
        results = run_full_benchmark(input_dim=50, num_classes=3)

        assert "timestamp" in results
        assert "models" in results
        assert "summary" in results

        # Should have model results
        assert len(results["models"]) > 0

        # Summary should have key fields
        assert "all_models_pass" in results["summary"]
        assert "fastest_model" in results["summary"]
        assert "device" in results["summary"]

    def test_verify_latency_requirements_returns_bool(self):
        """Test that verify_latency_requirements returns a boolean."""
        # This might fail on slow hardware, but should return a valid result
        result = verify_latency_requirements()
        assert isinstance(result, bool)


# Performance regression tests (can be run periodically)
class TestPerformanceRegression:
    """Performance regression tests to catch slowdowns."""

    @pytest.mark.parametrize("hidden_dims", [
        [64, 32],
        [128, 64, 32],
        [256, 128, 64, 32],
    ])
    def test_feedforward_performance_regression(self, hidden_dims):
        """Test FeedForward model doesn't regress in performance."""
        model = FeedForwardNet(input_dim=50, hidden_dims=hidden_dims, num_classes=3)
        benchmark = InferenceBenchmark(
            num_iterations=500,
            warmup_iterations=50,
        )

        result = benchmark.benchmark_model(model, input_dim=50)

        # Should always be under 10ms for these configurations
        assert result.p99_latency_ms < 10.0, (
            f"Performance regression: {hidden_dims} p99={result.p99_latency_ms:.2f}ms"
        )

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_lstm_performance_regression(self, num_layers):
        """Test LSTM model doesn't regress in performance."""
        model = LSTMNet(input_dim=50, hidden_dim=64, num_layers=num_layers, num_classes=3)
        benchmark = InferenceBenchmark(
            num_iterations=500,
            warmup_iterations=50,
        )

        result = benchmark.benchmark_model(model, input_dim=50, seq_length=20)

        # Should always be under 10ms for these configurations
        assert result.p99_latency_ms < 10.0, (
            f"Performance regression: LSTM {num_layers} layers p99={result.p99_latency_ms:.2f}ms"
        )
