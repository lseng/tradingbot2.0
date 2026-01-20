"""
Gymnasium-compatible Trading Environment for MES Futures.

This environment optimizes for PnL directly by:
1. Rewarding profitable trades (net of commissions/slippage)
2. Penalizing excessive risk and drawdowns
3. Supporting multi-horizon decision making (1h, 4h, EOD)

Actions:
    0: FLAT (close any position, stay out)
    1: LONG (buy if flat, hold if already long)
    2: SHORT (sell if flat, hold if already short)

Observation Space:
    - Price features (returns, volatility, momentum)
    - Multi-timeframe indicators
    - Position state (current position, unrealized PnL)
    - Time features (time of day, time to close)

Reward:
    - Realized PnL on position changes (net of costs)
    - Small penalty for holding positions (encourages decisive action)
    - Bonus for profitable days
    - Penalty for exceeding risk limits
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    slippage: float
    net_pnl: float
    bars_held: int


class TradingEnvironment(gym.Env):
    """
    MES Futures Trading Environment.

    Designed for intraday trading with:
    - 1-minute bar data
    - RTH hours only (9:30 AM - 4:00 PM NY)
    - Daily P&L tracking and risk limits
    - Realistic transaction costs
    """

    metadata = {"render_modes": ["human"]}

    # MES contract specs
    TICK_SIZE = 0.25
    TICK_VALUE = 1.25  # $1.25 per tick
    POINT_VALUE = 5.0  # $5.00 per point

    # Transaction costs
    COMMISSION_PER_SIDE = 0.42  # Per contract
    SLIPPAGE_TICKS = 1  # Average slippage in ticks

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        initial_balance: float = 1000.0,
        max_daily_loss: float = 100.0,
        max_position: int = 1,
        lookback_window: int = 60,  # 1 hour of 1-min bars
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV data and features (1-min bars, RTH only)
            feature_columns: List of feature column names to use as observation
            initial_balance: Starting account balance
            max_daily_loss: Maximum daily loss before stopping
            max_position: Maximum position size (contracts)
            lookback_window: Number of bars for observation window
            render_mode: Rendering mode
        """
        super().__init__()

        self.df = df.copy()
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.max_daily_loss = max_daily_loss
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.render_mode = render_mode

        # Ensure data is sorted by time
        self.df = self.df.sort_index()

        # Get unique trading days
        self.df['date'] = self.df.index.date
        self.trading_days = self.df['date'].unique()

        # Observation space: features + position info
        n_features = len(feature_columns)
        # Observation: [lookback_window x n_features] + [position, unrealized_pnl, daily_pnl, time_to_close]
        obs_dim = lookback_window * n_features + 4

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: FLAT, LONG, SHORT
        self.action_space = spaces.Discrete(3)

        # State variables (initialized in reset)
        self.current_idx = 0
        self.current_day_idx = 0
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0.0
        self.entry_idx = 0
        self.balance = initial_balance
        self.daily_pnl = 0.0
        self.trades: list = []
        self.done = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to start of a random trading day."""
        super().reset(seed=seed)

        # Pick a random starting day (leave room for at least one full day)
        if options and 'start_day_idx' in options:
            self.current_day_idx = options['start_day_idx']
        else:
            max_start = max(0, len(self.trading_days) - 2)
            self.current_day_idx = self.np_random.integers(0, max_start + 1)

        # Get first bar of the day
        current_day = self.trading_days[self.current_day_idx]
        day_mask = self.df['date'] == current_day
        day_indices = self.df[day_mask].index

        # Start after lookback window is available
        start_bar = self.lookback_window
        self.current_idx = self.df.index.get_loc(day_indices[min(start_bar, len(day_indices) - 1)])

        # Reset state
        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.balance = self.initial_balance
        self.daily_pnl = 0.0
        self.trades = []
        self.done = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=FLAT, 1=LONG, 2=SHORT

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()

        current_bar = self.df.iloc[self.current_idx]
        current_price = current_bar['close']
        current_time = self.df.index[self.current_idx]

        reward = 0.0

        # Map action to desired position
        desired_position = action - 1  # 0->-1, 1->0, 2->1 ... wait that's wrong
        # Actually: 0=FLAT->0, 1=LONG->1, 2=SHORT->-1
        if action == Action.FLAT:
            desired_position = 0
        elif action == Action.LONG:
            desired_position = 1
        elif action == Action.SHORT:
            desired_position = -1

        # Handle position changes
        if desired_position != self.position:
            # Close existing position if any
            if self.position != 0:
                reward += self._close_position(current_price, current_time)

            # Open new position if not flat
            if desired_position != 0:
                self._open_position(desired_position, current_price, current_time)

        # Track unrealized PnL (no holding penalty - let agent learn when to exit)
        if self.position != 0:
            unrealized = self._calculate_unrealized_pnl(current_price)
            # Only penalize if in significant drawdown (risk management)
            if unrealized < -20:  # More than $20 unrealized loss
                reward -= 0.1  # Small penalty for holding losing positions

        # Move to next bar
        self.current_idx += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # End of data
        if self.current_idx >= len(self.df) - 1:
            terminated = True
            self.done = True
            # Close any open position at end
            if self.position != 0:
                reward += self._close_position(
                    self.df.iloc[self.current_idx]['close'],
                    self.df.index[self.current_idx]
                )
        else:
            # Check if day changed
            next_bar = self.df.iloc[self.current_idx]
            if next_bar['date'] != current_bar['date']:
                # End of day - close position
                if self.position != 0:
                    reward += self._close_position(current_price, current_time)

                # Check daily loss limit
                if self.daily_pnl <= -self.max_daily_loss:
                    truncated = True
                    self.done = True

                # Move to next day
                self.current_day_idx += 1
                if self.current_day_idx >= len(self.trading_days):
                    terminated = True
                    self.done = True
                else:
                    self.daily_pnl = 0.0  # Reset daily PnL

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _open_position(self, direction: int, price: float, time: pd.Timestamp):
        """Open a new position."""
        self.position = direction
        # Account for slippage on entry
        slippage = self.SLIPPAGE_TICKS * self.TICK_SIZE
        if direction == 1:  # Long - pay more
            self.entry_price = price + slippage
        else:  # Short - receive less
            self.entry_price = price - slippage
        self.entry_idx = self.current_idx

    def _close_position(self, price: float, time: pd.Timestamp) -> float:
        """Close current position and return reward (net PnL)."""
        if self.position == 0:
            return 0.0

        # Account for slippage on exit
        slippage_cost = self.SLIPPAGE_TICKS * self.TICK_SIZE
        if self.position == 1:  # Closing long - receive less
            exit_price = price - slippage_cost
        else:  # Closing short - pay more
            exit_price = price + slippage_cost

        # Calculate PnL
        price_diff = exit_price - self.entry_price
        gross_pnl = self.position * price_diff * self.POINT_VALUE

        # Transaction costs (round trip)
        commission = self.COMMISSION_PER_SIDE * 2
        slippage = self.SLIPPAGE_TICKS * self.TICK_VALUE * 2

        net_pnl = gross_pnl - commission

        # Record trade
        trade = TradeRecord(
            entry_time=self.df.index[self.entry_idx],
            exit_time=time,
            direction=self.position,
            entry_price=self.entry_price,
            exit_price=exit_price,
            pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            net_pnl=net_pnl,
            bars_held=self.current_idx - self.entry_idx,
        )
        self.trades.append(trade)

        # Update state
        self.balance += net_pnl
        self.daily_pnl += net_pnl
        self.position = 0
        self.entry_price = 0.0

        return net_pnl

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for current position."""
        if self.position == 0:
            return 0.0
        price_diff = current_price - self.entry_price
        return self.position * price_diff * self.POINT_VALUE

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get lookback window of features
        start_idx = max(0, self.current_idx - self.lookback_window)
        end_idx = self.current_idx

        window_data = self.df.iloc[start_idx:end_idx][self.feature_columns].values

        # Pad if not enough history
        if len(window_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(window_data), len(self.feature_columns)))
            window_data = np.vstack([padding, window_data])

        # Flatten window features
        features_flat = window_data.flatten().astype(np.float32)

        # Add position info
        current_bar = self.df.iloc[min(self.current_idx, len(self.df) - 1)]
        current_price = current_bar['close']
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)

        # Time to market close (normalize to 0-1)
        current_time = self.df.index[min(self.current_idx, len(self.df) - 1)]
        if hasattr(current_time, 'hour'):
            minutes_since_open = (current_time.hour - 9) * 60 + current_time.minute - 30
            time_to_close = max(0, (390 - minutes_since_open) / 390)  # 390 min in RTH
        else:
            time_to_close = 0.5

        position_info = np.array([
            self.position,
            unrealized_pnl / 100.0,  # Normalize
            self.daily_pnl / 100.0,  # Normalize
            time_to_close,
        ], dtype=np.float32)

        obs = np.concatenate([features_flat, position_info])

        # Replace NaN/inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        return {
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'position': self.position,
            'num_trades': len(self.trades),
            'current_day': self.current_day_idx,
        }

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            current_bar = self.df.iloc[min(self.current_idx, len(self.df) - 1)]
            print(f"Bar {self.current_idx} | Price: {current_bar['close']:.2f} | "
                  f"Position: {self.position} | Balance: ${self.balance:.2f} | "
                  f"Daily PnL: ${self.daily_pnl:.2f} | Trades: {len(self.trades)}")

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades."""
        if not self.trades:
            return {'num_trades': 0}

        pnls = [t.net_pnl for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        return {
            'num_trades': len(self.trades),
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'win_rate': len(winners) / len(pnls) if pnls else 0,
            'avg_winner': np.mean(winners) if winners else 0,
            'avg_loser': np.mean(losers) if losers else 0,
            'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
            'max_drawdown': self._calculate_max_drawdown(pnls),
        }

    def _calculate_max_drawdown(self, pnls: list) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnls:
            return 0.0
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def create_env_from_data(
    df: pd.DataFrame,
    feature_columns: list,
    **kwargs
) -> TradingEnvironment:
    """
    Factory function to create trading environment.

    Args:
        df: DataFrame with OHLCV and features
        feature_columns: Feature column names
        **kwargs: Additional env parameters

    Returns:
        TradingEnvironment instance
    """
    return TradingEnvironment(df, feature_columns, **kwargs)
