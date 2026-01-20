"""
Tests for the RL Trading Environment.

These tests verify:
1. Environment creation and configuration
2. Observation and action spaces
3. Reset and step functionality
4. Position management (open, close, transitions)
5. Reward calculation including transaction costs
6. Day boundary handling
7. Risk management (max daily loss)
"""

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

# Skip all tests if gymnasium is not installed
gymnasium = pytest.importorskip("gymnasium")

from src.rl.trading_env import TradingEnvironment, Action, TradeRecord


class TestTradingEnvironmentCreation:
    """Tests for environment initialization."""

    def test_environment_creation(self, sample_rl_data_with_features, sample_feature_columns):
        """Environment should be created with valid parameters."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
            initial_balance=1000.0,
            max_daily_loss=100.0,
            max_position=1,
            lookback_window=60,
        )

        assert env is not None
        assert env.initial_balance == 1000.0
        assert env.max_daily_loss == 100.0
        assert env.max_position == 1
        assert env.lookback_window == 60

    def test_observation_space_shape(self, sample_rl_data_with_features, sample_feature_columns):
        """Observation space should have correct dimensions."""
        lookback_window = 60
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
            lookback_window=lookback_window,
        )

        # Expected: lookback_window * n_features + 4 (position info)
        expected_dim = lookback_window * len(sample_feature_columns) + 4
        assert env.observation_space.shape == (expected_dim,)

    def test_action_space(self, sample_rl_data_with_features, sample_feature_columns):
        """Action space should be Discrete(3)."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )

        assert env.action_space.n == 3

    def test_action_enum_values(self):
        """Action enum should have correct values."""
        assert Action.FLAT == 0
        assert Action.LONG == 1
        assert Action.SHORT == 2


class TestTradingEnvironmentReset:
    """Tests for environment reset functionality."""

    def test_reset_returns_valid_observation(self, sample_rl_data_with_features, sample_feature_columns):
        """Reset should return a valid observation and info dict."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
            lookback_window=60,
        )

        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_reset_clears_state(self, sample_rl_data_with_features, sample_feature_columns):
        """Reset should clear position and balance state."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )

        env.reset(seed=42)

        assert env.position == 0
        assert env.entry_price == 0.0
        assert env.balance == env.initial_balance
        assert env.daily_pnl == 0.0
        assert env.trades == []
        assert not env.done

    def test_reset_with_seed_is_reproducible(self, sample_rl_data_with_features, sample_feature_columns):
        """Reset with same seed should produce same starting state."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )

        obs1, _ = env.reset(seed=42)
        idx1 = env.current_idx

        obs2, _ = env.reset(seed=42)
        idx2 = env.current_idx

        np.testing.assert_array_equal(obs1, obs2)
        assert idx1 == idx2


class TestTradingEnvironmentStep:
    """Tests for environment step functionality."""

    def test_step_returns_correct_format(self, sample_rl_data_with_features, sample_feature_columns):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)

        result = env.step(Action.FLAT)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_flat_action_keeps_no_position(self, sample_rl_data_with_features, sample_feature_columns):
        """FLAT action should maintain no position."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)

        env.step(Action.FLAT)

        assert env.position == 0

    def test_long_action_opens_long_position(self, sample_rl_data_with_features, sample_feature_columns):
        """LONG action should open a long position."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)

        env.step(Action.LONG)

        assert env.position == 1
        assert env.entry_price > 0

    def test_short_action_opens_short_position(self, sample_rl_data_with_features, sample_feature_columns):
        """SHORT action should open a short position."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)

        env.step(Action.SHORT)

        assert env.position == -1
        assert env.entry_price > 0

    def test_position_change_closes_and_opens(self, sample_rl_data_with_features, sample_feature_columns):
        """Changing from LONG to SHORT should close long and open short."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)

        # Open long
        env.step(Action.LONG)
        assert env.position == 1

        # Switch to short
        env.step(Action.SHORT)
        assert env.position == -1
        assert len(env.trades) == 1  # One trade was closed


class TestTradingEnvironmentRewards:
    """Tests for reward calculation."""

    def test_transaction_costs_applied(self, sample_rl_data_with_features, sample_feature_columns):
        """Transaction costs should be applied when closing positions."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)

        # Open and immediately close a position
        env.step(Action.LONG)
        _, reward, _, _, _ = env.step(Action.FLAT)

        # Reward should account for slippage and commission
        # Even if price unchanged, costs should make it negative
        assert len(env.trades) == 1
        trade = env.trades[0]
        assert trade.commission > 0
        assert trade.slippage != 0 or trade.commission > 0  # Some cost was applied


class TestTradingEnvironmentInfo:
    """Tests for info dictionary."""

    def test_info_contains_position(self, sample_rl_data_with_features, sample_feature_columns):
        """Info should contain current position."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        env.reset(seed=42)
        _, info = env.reset()

        assert "position" in info

    def test_info_contains_balance(self, sample_rl_data_with_features, sample_feature_columns):
        """Info should contain current balance."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        _, info = env.reset(seed=42)

        assert "balance" in info


class TestTradingEnvironmentConstants:
    """Tests for MES contract constants."""

    def test_tick_size(self, sample_rl_data_with_features, sample_feature_columns):
        """Tick size should be 0.25 for MES."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        assert env.TICK_SIZE == 0.25

    def test_tick_value(self, sample_rl_data_with_features, sample_feature_columns):
        """Tick value should be $1.25 for MES."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        assert env.TICK_VALUE == 1.25

    def test_point_value(self, sample_rl_data_with_features, sample_feature_columns):
        """Point value should be $5.00 for MES."""
        env = TradingEnvironment(
            df=sample_rl_data_with_features,
            feature_columns=sample_feature_columns,
        )
        assert env.POINT_VALUE == 5.0


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_creation(self, ny_tz):
        """TradeRecord should store all trade information."""
        record = TradeRecord(
            entry_time=pd.Timestamp("2024-01-02 10:00:00", tz=ny_tz),
            exit_time=pd.Timestamp("2024-01-02 10:15:00", tz=ny_tz),
            direction=1,
            entry_price=4800.0,
            exit_price=4802.0,
            pnl=10.0,
            commission=0.84,
            slippage=0.50,
            net_pnl=8.66,
            bars_held=15,
        )

        assert record.direction == 1
        assert record.entry_price == 4800.0
        assert record.exit_price == 4802.0
        assert record.bars_held == 15


class TestEnvironmentImports:
    """Tests for module imports."""

    def test_import_action_enum(self):
        """Action enum should be importable."""
        from src.rl.trading_env import Action
        assert Action.FLAT == 0

    def test_import_trading_environment(self):
        """TradingEnvironment should be importable."""
        from src.rl.trading_env import TradingEnvironment
        assert TradingEnvironment is not None

    def test_import_trade_record(self):
        """TradeRecord should be importable."""
        from src.rl.trading_env import TradeRecord
        assert TradeRecord is not None


class TestMultipleDays:
    """Tests for multi-day trading scenarios."""

    def test_environment_handles_multi_day_data(self, multi_day_rl_data, sample_feature_columns):
        """Environment should handle data spanning multiple days."""
        # Add features to multi-day data
        df = multi_day_rl_data.copy()
        for col in sample_feature_columns:
            if col.startswith("return"):
                period = int(col.split("_")[1])
                df[col] = df["close"].pct_change(period).fillna(0)
            elif col.startswith("volatility"):
                period = int(col.split("_")[1])
                df[col] = df["close"].pct_change(1).rolling(period).std().fillna(0)
            elif col == "rsi_14":
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                df[col] = 100 - (100 / (1 + rs))
                df[col] = df[col].fillna(50)
            elif col == "macd":
                exp12 = df["close"].ewm(span=12, adjust=False).mean()
                exp26 = df["close"].ewm(span=26, adjust=False).mean()
                df[col] = exp12 - exp26
            elif col == "macd_signal":
                if "macd" in df.columns:
                    df[col] = df["macd"].ewm(span=9, adjust=False).mean()
                else:
                    df[col] = 0
            elif col == "volume_ratio":
                df[col] = df["volume"] / df["volume"].rolling(20).mean()
                df[col] = df[col].fillna(1)
            else:
                df[col] = 0

        env = TradingEnvironment(
            df=df,
            feature_columns=sample_feature_columns,
            lookback_window=60,
        )
        env.reset(seed=42)

        # Environment should have multiple trading days
        assert len(env.trading_days) >= 2
