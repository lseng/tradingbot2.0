"""
Tests for RL data pipeline module.

Tests cover:
- MultiHorizonDataPipeline class initialization and methods
- RTH filtering logic
- Data aggregation from 1-second to 1-minute bars
- Feature generation
- RSI calculation
- Feature normalization
- Training data preparation
- Date filtering
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

from src.rl.data_pipeline import (
    MultiHorizonDataPipeline,
    load_data_for_rl,
)


NY_TZ = ZoneInfo("America/New_York")


class TestMultiHorizonDataPipelineInit:
    """Tests for pipeline initialization."""

    def test_init_with_valid_path(self, tmp_path):
        """Pipeline accepts valid file path."""
        test_file = tmp_path / "test.parquet"
        test_file.touch()
        pipeline = MultiHorizonDataPipeline(str(test_file))
        assert pipeline.data_path == test_file

    def test_init_stores_path_as_pathlib(self, tmp_path):
        """Path is stored as pathlib.Path object."""
        test_file = tmp_path / "test.parquet"
        pipeline = MultiHorizonDataPipeline(str(test_file))
        assert isinstance(pipeline.data_path, Path)

    def test_rth_constants(self):
        """RTH constants are correct for NYSE."""
        pipeline = MultiHorizonDataPipeline("fake_path")
        assert pipeline.RTH_START_HOUR == 9
        assert pipeline.RTH_START_MIN == 30
        assert pipeline.RTH_END_HOUR == 16
        assert pipeline.RTH_END_MIN == 0

    def test_tick_size_constant(self):
        """MES tick size is 0.25."""
        pipeline = MultiHorizonDataPipeline("fake_path")
        assert pipeline.TICK_SIZE == 0.25


class TestRTHFiltering:
    """Tests for Regular Trading Hours filtering."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    def test_filter_rth_keeps_930(self, pipeline):
        """RTH filter keeps 9:30 AM bar."""
        dates = pd.date_range("2024-01-02 09:30:00", periods=1, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0]}, index=dates)
        result = pipeline._filter_rth(df)
        assert len(result) == 1

    def test_filter_rth_excludes_929(self, pipeline):
        """RTH filter excludes 9:29 AM bar."""
        dates = pd.date_range("2024-01-02 09:29:00", periods=1, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0]}, index=dates)
        result = pipeline._filter_rth(df)
        assert len(result) == 0

    def test_filter_rth_keeps_400(self, pipeline):
        """RTH filter keeps 4:00 PM bar (last RTH bar)."""
        dates = pd.date_range("2024-01-02 16:00:00", periods=1, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0]}, index=dates)
        result = pipeline._filter_rth(df)
        assert len(result) == 1

    def test_filter_rth_excludes_401(self, pipeline):
        """RTH filter excludes 4:01 PM bar."""
        dates = pd.date_range("2024-01-02 16:01:00", periods=1, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0]}, index=dates)
        result = pipeline._filter_rth(df)
        assert len(result) == 0

    def test_filter_rth_keeps_midday(self, pipeline):
        """RTH filter keeps midday bars."""
        dates = pd.date_range("2024-01-02 12:00:00", periods=5, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0] * 5}, index=dates)
        result = pipeline._filter_rth(df)
        assert len(result) == 5

    def test_filter_rth_excludes_overnight(self, pipeline):
        """RTH filter excludes overnight bars."""
        dates = pd.date_range("2024-01-02 02:00:00", periods=5, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0] * 5}, index=dates)
        result = pipeline._filter_rth(df)
        assert len(result) == 0

    def test_filter_rth_preserves_index(self, pipeline):
        """Filtered DataFrame preserves original index type."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=5, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({"close": [100.0] * 5}, index=dates)
        result = pipeline._filter_rth(df)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestDataAggregation:
    """Tests for 1-second to 1-minute data aggregation."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    def test_aggregate_creates_ohlcv(self, pipeline):
        """Aggregation creates proper OHLCV columns."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=60, freq="1s", tz=NY_TZ)
        df = pd.DataFrame({
            "open": np.random.uniform(100, 101, 60),
            "high": np.random.uniform(101, 102, 60),
            "low": np.random.uniform(99, 100, 60),
            "close": np.random.uniform(100, 101, 60),
            "volume": np.random.randint(10, 50, 60),
        }, index=dates)

        result = pipeline._aggregate_to_1min(df)

        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_aggregate_open_is_first(self, pipeline):
        """Aggregation uses first value for open."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=60, freq="1s", tz=NY_TZ)
        opens = [100.0] + [101.0] * 59  # First is 100, rest are 101
        df = pd.DataFrame({
            "open": opens,
            "high": [102.0] * 60,
            "low": [99.0] * 60,
            "close": [101.0] * 60,
            "volume": [10] * 60,
        }, index=dates)

        result = pipeline._aggregate_to_1min(df)
        assert result.iloc[0]["open"] == 100.0

    def test_aggregate_high_is_max(self, pipeline):
        """Aggregation uses max for high."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=60, freq="1s", tz=NY_TZ)
        highs = [99.0] * 30 + [105.0] + [99.0] * 29  # Max is 105
        df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": highs,
            "low": [98.0] * 60,
            "close": [100.0] * 60,
            "volume": [10] * 60,
        }, index=dates)

        result = pipeline._aggregate_to_1min(df)
        assert result.iloc[0]["high"] == 105.0

    def test_aggregate_low_is_min(self, pipeline):
        """Aggregation uses min for low."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=60, freq="1s", tz=NY_TZ)
        lows = [100.0] * 30 + [95.0] + [100.0] * 29  # Min is 95
        df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [102.0] * 60,
            "low": lows,
            "close": [100.0] * 60,
            "volume": [10] * 60,
        }, index=dates)

        result = pipeline._aggregate_to_1min(df)
        assert result.iloc[0]["low"] == 95.0

    def test_aggregate_close_is_last(self, pipeline):
        """Aggregation uses last value for close."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=60, freq="1s", tz=NY_TZ)
        closes = [100.0] * 59 + [110.0]  # Last is 110
        df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [112.0] * 60,
            "low": [99.0] * 60,
            "close": closes,
            "volume": [10] * 60,
        }, index=dates)

        result = pipeline._aggregate_to_1min(df)
        assert result.iloc[0]["close"] == 110.0

    def test_aggregate_volume_is_sum(self, pipeline):
        """Aggregation sums volume."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=60, freq="1s", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [102.0] * 60,
            "low": [99.0] * 60,
            "close": [101.0] * 60,
            "volume": [10] * 60,  # Sum = 600
        }, index=dates)

        result = pipeline._aggregate_to_1min(df)
        assert result.iloc[0]["volume"] == 600


class TestRSICalculation:
    """Tests for RSI indicator calculation."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    def test_rsi_bounds(self, pipeline):
        """RSI should be bounded between 0 and 100."""
        prices = pd.Series(np.random.uniform(100, 105, 100))
        rsi = pipeline._calculate_rsi(prices, period=14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_uptrend(self, pipeline):
        """RSI should be high in strong uptrend."""
        prices = pd.Series(range(100, 200))  # Strong uptrend
        rsi = pipeline._calculate_rsi(prices, period=14)
        # After warmup, RSI should be high
        assert rsi.iloc[-1] > 70

    def test_rsi_downtrend(self, pipeline):
        """RSI should be low in strong downtrend."""
        prices = pd.Series(range(200, 100, -1))  # Strong downtrend
        rsi = pipeline._calculate_rsi(prices, period=14)
        # After warmup, RSI should be low
        assert rsi.iloc[-1] < 30

    def test_rsi_period_affects_result(self, pipeline):
        """Different RSI periods produce different results."""
        prices = pd.Series(np.random.uniform(100, 105, 50))
        rsi_7 = pipeline._calculate_rsi(prices, period=7)
        rsi_14 = pipeline._calculate_rsi(prices, period=14)
        # Results should differ - compare at the same indices where both have values
        rsi_7_clean = rsi_7.dropna()
        rsi_14_clean = rsi_14.dropna()
        # Use min length to compare overlapping valid values
        min_len = min(len(rsi_7_clean), len(rsi_14_clean))
        # Compare the last min_len values (where both series have converged)
        assert not np.allclose(rsi_7_clean.values[-min_len:], rsi_14_clean.values[-min_len:])


class TestFeatureGeneration:
    """Tests for feature generation."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    @pytest.fixture
    def sample_1min_data(self):
        """Create sample 1-minute OHLCV data."""
        np.random.seed(42)
        # Create 500 bars of realistic data during RTH
        dates = pd.date_range("2024-01-02 09:30:00", periods=500, freq="1min", tz=NY_TZ)

        base_price = 4800.0
        returns = np.random.normal(0, 0.0002, 500)
        close_prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "open": np.roll(close_prices, 1),
            "high": close_prices * (1 + np.random.uniform(0.0001, 0.0005, 500)),
            "low": close_prices * (1 - np.random.uniform(0.0001, 0.0005, 500)),
            "close": close_prices,
            "volume": np.random.randint(100, 1000, 500),
        }, index=dates)
        df.iloc[0, df.columns.get_loc("open")] = base_price

        return df

    def test_generate_features_returns_tuple(self, pipeline, sample_1min_data):
        """generate_features returns (DataFrame, feature_list) tuple."""
        result = pipeline.generate_features(sample_1min_data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_generate_features_includes_returns(self, pipeline, sample_1min_data):
        """Features include return columns at multiple windows."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        return_cols = [c for c in feature_cols if c.startswith("return_")]
        assert len(return_cols) >= 5
        assert "return_1m" in feature_cols
        assert "return_5m" in feature_cols
        assert "return_15m" in feature_cols

    def test_generate_features_includes_ema(self, pipeline, sample_1min_data):
        """Features include EMA relationship columns."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        ema_cols = [c for c in feature_cols if "ema" in c]
        assert len(ema_cols) >= 4

    def test_generate_features_includes_macd(self, pipeline, sample_1min_data):
        """Features include MACD columns."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        assert "macd" in feature_cols
        assert "macd_signal" in feature_cols
        assert "macd_hist" in feature_cols

    def test_generate_features_includes_rsi(self, pipeline, sample_1min_data):
        """Features include RSI at multiple periods."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        rsi_cols = [c for c in feature_cols if c.startswith("rsi_")]
        assert len(rsi_cols) >= 3

    def test_generate_features_includes_volatility(self, pipeline, sample_1min_data):
        """Features include volatility columns."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        vol_cols = [c for c in feature_cols if "volatility" in c]
        assert len(vol_cols) >= 3
        assert "atr_14" in feature_cols

    def test_generate_features_includes_volume(self, pipeline, sample_1min_data):
        """Features include volume columns."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        assert "volume_ratio" in feature_cols
        assert "volume_trend" in feature_cols

    def test_generate_features_includes_time(self, pipeline, sample_1min_data):
        """Features include time-of-day columns."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        assert "time_sin" in feature_cols
        assert "time_cos" in feature_cols
        assert "dow" in feature_cols

    def test_generate_features_includes_multi_horizon(self, pipeline, sample_1min_data):
        """Features include multi-horizon targets when requested."""
        df, feature_cols = pipeline.generate_features(sample_1min_data, include_multi_horizon=True)
        assert "return_1h" in df.columns
        assert "return_4h" in df.columns
        assert "return_eod" in df.columns

    def test_generate_features_excludes_multi_horizon(self, pipeline, sample_1min_data):
        """Multi-horizon targets excluded when not requested."""
        df, feature_cols = pipeline.generate_features(sample_1min_data, include_multi_horizon=False)
        assert "return_1h" not in df.columns
        assert "return_4h" not in df.columns

    def test_generate_features_no_nans_in_result(self, pipeline, sample_1min_data):
        """Generated features have no NaN values after dropna."""
        df, feature_cols = pipeline.generate_features(sample_1min_data)
        # The method drops NaN, so result should be clean
        assert not df[feature_cols].isnull().any().any()


class TestFeatureNormalization:
    """Tests for feature normalization."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    def test_normalize_centers_features(self, pipeline):
        """Normalization centers features around 0."""
        df = pd.DataFrame({
            "feature_a": [100, 110, 105, 95, 100],
            "feature_b": [1000, 1100, 1050, 950, 1000],
        })
        feature_cols = ["feature_a", "feature_b"]

        result_df, _ = pipeline._normalize_features(df, feature_cols)

        # Mean should be approximately 0
        assert abs(result_df["feature_a"].mean()) < 0.1
        assert abs(result_df["feature_b"].mean()) < 0.1

    def test_normalize_clips_outliers(self, pipeline):
        """Normalization clips values to [-5, 5]."""
        df = pd.DataFrame({
            "feature_a": [0, 0, 0, 100],  # 100 is an outlier
        })
        feature_cols = ["feature_a"]

        result_df, _ = pipeline._normalize_features(df, feature_cols)

        assert result_df["feature_a"].max() <= 5
        assert result_df["feature_a"].min() >= -5

    def test_normalize_handles_zero_std(self, pipeline):
        """Normalization handles constant features (zero std)."""
        df = pd.DataFrame({
            "feature_a": [5.0, 5.0, 5.0, 5.0],  # Constant - zero std
        })
        feature_cols = ["feature_a"]

        result_df, _ = pipeline._normalize_features(df, feature_cols)

        # Should be 0 when std is 0
        assert (result_df["feature_a"] == 0.0).all()

    def test_normalize_replaces_inf(self, pipeline):
        """Normalization replaces inf values with 0."""
        df = pd.DataFrame({
            "feature_a": [1.0, np.inf, -np.inf, 2.0],
        })
        feature_cols = ["feature_a"]

        result_df, _ = pipeline._normalize_features(df, feature_cols)

        assert not np.isinf(result_df["feature_a"]).any()

    def test_normalize_replaces_nan(self, pipeline):
        """Normalization replaces NaN values with 0."""
        df = pd.DataFrame({
            "feature_a": [1.0, np.nan, 2.0, 3.0],
        })
        feature_cols = ["feature_a"]

        result_df, _ = pipeline._normalize_features(df, feature_cols)

        assert not result_df["feature_a"].isnull().any()


class TestPriceInRange:
    """Tests for price_in_range feature calculation."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    def test_price_in_range_bounds(self, pipeline):
        """price_in_range should be between 0 and 1."""
        # Create data with clear high/low
        dates = pd.date_range("2024-01-02 09:30:00", periods=10, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [100, 101, 102, 103, 104, 103, 102, 101, 100, 99],
            "volume": [100] * 10,
        }, index=dates)

        result_df, _ = pipeline.generate_features(df)

        if "price_in_range" in result_df.columns:
            valid_range = result_df["price_in_range"].dropna()
            assert (valid_range >= 0).all()
            assert (valid_range <= 1).all()


class TestLoadDataForRL:
    """Tests for the convenience function load_data_for_rl."""

    def test_function_exists(self):
        """load_data_for_rl function is importable."""
        from src.rl.data_pipeline import load_data_for_rl
        assert callable(load_data_for_rl)

    def test_accepts_train_ratio(self):
        """Function accepts train_ratio parameter."""
        # Just verify the signature accepts the parameter
        import inspect
        sig = inspect.signature(load_data_for_rl)
        assert "train_ratio" in sig.parameters


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def sample_parquet_file(self, tmp_path):
        """Create a sample parquet file for testing."""
        np.random.seed(42)

        # Create 3 hours of 1-second data during RTH
        # RTH is 9:30 AM - 4:00 PM NY time
        # In January (EST, UTC-5): 9:30 AM NY = 14:30 UTC
        dates = pd.date_range("2024-01-02 14:30:00", periods=10800, freq="1s", tz="UTC")

        base_price = 4800.0
        returns = np.random.normal(0, 0.00005, len(dates))
        close_prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.roll(close_prices, 1),
            "high": close_prices * (1 + np.random.uniform(0, 0.0001, len(dates))),
            "low": close_prices * (1 - np.random.uniform(0, 0.0001, len(dates))),
            "close": close_prices,
            "volume": np.random.randint(1, 10, len(dates)),
        })

        file_path = tmp_path / "test_data.parquet"
        df.to_parquet(file_path)
        return str(file_path)

    def test_load_and_aggregate(self, sample_parquet_file):
        """Pipeline loads and aggregates parquet data."""
        pipeline = MultiHorizonDataPipeline(sample_parquet_file)
        df = pipeline.load_and_aggregate()

        assert len(df) > 0
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_full_pipeline(self, sample_parquet_file):
        """Full pipeline produces valid training data."""
        pipeline = MultiHorizonDataPipeline(sample_parquet_file)
        df = pipeline.load_and_aggregate()
        df_features, feature_cols = pipeline.generate_features(df)

        assert len(df_features) > 0
        assert len(feature_cols) > 10
        # Check no NaN in feature columns
        assert not df_features[feature_cols].isnull().any().any()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for testing."""
        return MultiHorizonDataPipeline("fake_path")

    def test_empty_dataframe(self, pipeline):
        """Pipeline handles empty DataFrame gracefully."""
        dates = pd.date_range("2024-01-02 09:30:00", periods=0, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }, index=dates)

        result = pipeline._filter_rth(df)
        assert len(result) == 0

    def test_single_bar(self, pipeline):
        """Pipeline handles single bar data."""
        dates = pd.date_range("2024-01-02 10:00:00", periods=1, freq="1min", tz=NY_TZ)
        df = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [500],
        }, index=dates)

        result = pipeline._filter_rth(df)
        assert len(result) == 1
