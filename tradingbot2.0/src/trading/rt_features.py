"""
Real-Time Feature Engine for Live Trading.

Aggregates incoming ticks into 1-second bars and calculates features
for model inference. Designed for low-latency operation (< 5ms per bar).

Key responsibilities:
- Aggregate ticks to OHLCV bars (1-second resolution)
- Maintain rolling feature buffers (circular buffers)
- Calculate features efficiently using incremental updates
- Cache intermediate calculations
- Provide features ready for model inference

Reference: specs/live-trading-execution.md
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, List, Dict, Deque, Callable
import logging

from src.api import Quote

logger = logging.getLogger(__name__)

# MES Contract Constants
MES_TICK_SIZE = 0.25  # Minimum price movement
MES_TICK_VALUE = 1.25  # Dollar value per tick
MES_POINT_VALUE = 5.00  # Dollar value per point

# RTH Session
RTH_START = time(9, 30)
RTH_END = time(16, 0)
RTH_DURATION_MINUTES = 390


@dataclass
class OHLCV:
    """1-second OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    tick_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
        }


@dataclass
class FeatureVector:
    """
    Feature vector ready for model inference.

    Contains all features normalized and in the expected order.
    """
    features: np.ndarray
    feature_names: List[str]
    timestamp: datetime
    bar: OHLCV
    atr: float = 0.0  # For position sizing

    def as_tensor(self):
        """Convert to PyTorch tensor (lazy import)."""
        import torch
        return torch.tensor(self.features, dtype=torch.float32).unsqueeze(0)


@dataclass
class RTFeaturesConfig:
    """Configuration for real-time feature engine."""
    # Buffer sizes (in bars)
    max_bars: int = 500  # Keep last 500 bars (~8 min)

    # Feature lookback periods (in seconds/bars)
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 30, 60])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])

    # Technical indicator periods
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0

    # VWAP reset at session start
    vwap_reset_time: time = RTH_START

    # Cache settings
    cache_intermediate: bool = True


class BarAggregator:
    """
    Aggregates incoming ticks into 1-second OHLCV bars.

    Usage:
        aggregator = BarAggregator(on_bar_complete=handle_bar)
        for quote in quotes:
            aggregator.add_tick(quote)
    """

    def __init__(self, on_bar_complete: Optional[Callable[[OHLCV], None]] = None):
        """
        Initialize bar aggregator.

        Args:
            on_bar_complete: Callback when a bar is complete
        """
        self._on_bar_complete = on_bar_complete
        self._current_bar: Optional[OHLCV] = None
        self._current_second: Optional[int] = None

    def add_tick(self, quote: Quote) -> Optional[OHLCV]:
        """
        Add a tick to the aggregator.

        Args:
            quote: Quote from WebSocket

        Returns:
            Completed bar if this tick triggered a new second
        """
        price = quote.last_price
        volume = quote.volume
        timestamp = quote.timestamp

        # Normalize to second boundary
        tick_second = timestamp.replace(microsecond=0)

        if self._current_second is None:
            # First tick
            self._start_new_bar(tick_second, price, volume)
            return None

        elif tick_second > self._current_second:
            # New second - complete current bar
            completed_bar = self._complete_bar()

            # Start new bar
            self._start_new_bar(tick_second, price, volume)

            return completed_bar

        else:
            # Same second - update current bar
            self._update_bar(price, volume)
            return None

    def _start_new_bar(self, timestamp: datetime, price: float, volume: int) -> None:
        """Start a new bar."""
        self._current_second = timestamp
        self._current_bar = OHLCV(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            tick_count=1,
        )

    def _update_bar(self, price: float, volume: int) -> None:
        """Update current bar with new tick."""
        if self._current_bar is None:
            return

        self._current_bar.high = max(self._current_bar.high, price)
        self._current_bar.low = min(self._current_bar.low, price)
        self._current_bar.close = price
        self._current_bar.volume += volume
        self._current_bar.tick_count += 1

    def _complete_bar(self) -> Optional[OHLCV]:
        """Complete and return current bar."""
        bar = self._current_bar
        if bar and self._on_bar_complete:
            self._on_bar_complete(bar)
        return bar

    def flush(self) -> Optional[OHLCV]:
        """Flush any remaining bar (e.g., at end of session)."""
        if self._current_bar:
            return self._complete_bar()
        return None


class RealTimeFeatureEngine:
    """
    Real-time feature calculation engine.

    Maintains rolling buffers and calculates features incrementally
    for low-latency inference.

    Usage:
        engine = RealTimeFeatureEngine()

        # Add bars as they complete
        for bar in bars:
            features = engine.update(bar)
            if features:
                prediction = model.predict(features.as_tensor())
    """

    def __init__(self, config: Optional[RTFeaturesConfig] = None):
        """
        Initialize feature engine.

        Args:
            config: Feature configuration
        """
        self.config = config or RTFeaturesConfig()

        # Rolling buffers for OHLCV data
        self._bars: Deque[OHLCV] = deque(maxlen=self.config.max_bars)

        # Efficient numpy arrays for price/volume history
        self._prices: Deque[float] = deque(maxlen=self.config.max_bars)
        self._highs: Deque[float] = deque(maxlen=self.config.max_bars)
        self._lows: Deque[float] = deque(maxlen=self.config.max_bars)
        self._volumes: Deque[int] = deque(maxlen=self.config.max_bars)

        # EMA state (incremental calculation)
        self._ema_state: Dict[int, float] = {}

        # VWAP state (session reset)
        self._vwap_cumsum_pv: float = 0.0
        self._vwap_cumsum_v: float = 0.0
        self._vwap_reset_date: Optional[datetime] = None

        # ATR state (for position sizing)
        self._current_atr: float = 0.0

        # Feature names (generated once)
        self._feature_names: Optional[List[str]] = None

        logger.info(f"RealTimeFeatureEngine initialized with {self.config.max_bars} bar buffer")

    def update(self, bar: OHLCV) -> Optional[FeatureVector]:
        """
        Update with new bar and calculate features.

        Args:
            bar: Completed 1-second OHLCV bar

        Returns:
            FeatureVector if enough data, None otherwise
        """
        # Add to buffers
        self._bars.append(bar)
        self._prices.append(bar.close)
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._volumes.append(bar.volume)

        # Update incremental state
        self._update_ema_state(bar.close)
        self._update_vwap_state(bar)

        # Need minimum bars for features
        min_bars = max(self.config.ema_periods) if self.config.ema_periods else 200
        if len(self._bars) < min_bars:
            logger.debug(f"Need more bars: {len(self._bars)}/{min_bars}")
            return None

        # Calculate features
        features = self._calculate_features(bar)
        return features

    def _calculate_features(self, bar: OHLCV) -> FeatureVector:
        """Calculate all features for current state."""
        features = []
        names = []

        # Convert deques to numpy arrays for efficient calculation
        prices = np.array(self._prices)
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        volumes = np.array(self._volumes)

        # 1. Returns at different lookbacks
        for period in self.config.return_periods:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period-1]) / prices[-period-1]
                log_ret = np.log(prices[-1] / prices[-period-1])
            else:
                ret = 0.0
                log_ret = 0.0
            features.extend([ret, log_ret])
            names.extend([f'return_{period}s', f'log_return_{period}s'])

        # 2. EMAs (use incremental state)
        for period in self.config.ema_periods:
            ema = self._ema_state.get(period, prices[-1])
            close_to_ema = (prices[-1] - ema) / ema if ema != 0 else 0.0
            features.append(close_to_ema)
            names.append(f'close_to_ema_{period}')

        # EMA crossovers
        if 9 in self._ema_state and 21 in self._ema_state:
            ema_9_21_cross = (self._ema_state[9] - self._ema_state[21]) / self._ema_state[21]
            features.append(ema_9_21_cross)
            names.append('ema_9_21_cross')

        if 21 in self._ema_state and 50 in self._ema_state:
            ema_21_50_cross = (self._ema_state[21] - self._ema_state[50]) / self._ema_state[50]
            features.append(ema_21_50_cross)
            names.append('ema_21_50_cross')

        # 3. VWAP
        vwap = self._vwap_cumsum_pv / self._vwap_cumsum_v if self._vwap_cumsum_v > 0 else prices[-1]
        close_to_vwap = (prices[-1] - vwap) / vwap if vwap != 0 else 0.0
        features.append(close_to_vwap)
        names.append('close_to_vwap')

        # VWAP slope (approximation)
        if len(self._bars) >= 10:
            features.append((vwap - prices[-10]) / prices[-10])
        else:
            features.append(0.0)
        names.append('vwap_slope')

        # 4. Minutes to close
        minutes_to_close = self._calculate_minutes_to_close(bar.timestamp)
        features.append(minutes_to_close / RTH_DURATION_MINUTES)  # Normalized
        features.append(1.0 - minutes_to_close / RTH_DURATION_MINUTES)  # EOD urgency
        names.extend(['minutes_to_close_norm', 'eod_urgency'])

        # 5. Time of day (cyclical encoding)
        minutes_of_day = bar.timestamp.hour * 60 + bar.timestamp.minute
        features.append(np.sin(2 * np.pi * minutes_of_day / (24 * 60)))
        features.append(np.cos(2 * np.pi * minutes_of_day / (24 * 60)))
        names.extend(['time_sin', 'time_cos'])

        # 6. Day of week (cyclical)
        dow = bar.timestamp.weekday()
        features.append(np.sin(2 * np.pi * dow / 5))
        features.append(np.cos(2 * np.pi * dow / 5))
        names.extend(['dow_sin', 'dow_cos'])

        # Session period flags
        t = bar.timestamp.time()
        features.append(1.0 if (t >= time(9, 30) and t < time(10, 0)) else 0.0)
        features.append(1.0 if (t >= time(15, 0) and t <= time(16, 0)) else 0.0)
        features.append(1.0 if (t >= time(11, 30) and t < time(13, 0)) else 0.0)
        names.extend(['is_open_period', 'is_close_period', 'is_lunch_period'])

        # 7. Volatility features
        # ATR
        if len(prices) >= self.config.atr_period:
            true_ranges = []
            for i in range(1, min(self.config.atr_period + 1, len(prices))):
                tr = max(
                    highs[-i] - lows[-i],
                    abs(highs[-i] - prices[-i-1]) if i+1 <= len(prices) else 0,
                    abs(lows[-i] - prices[-i-1]) if i+1 <= len(prices) else 0,
                )
                true_ranges.append(tr)
            self._current_atr = np.mean(true_ranges) if true_ranges else 0.0
        atr_ticks = self._current_atr / MES_TICK_SIZE
        atr_pct = self._current_atr / prices[-1] if prices[-1] != 0 else 0.0
        features.extend([atr_ticks, atr_pct])
        names.extend(['atr_ticks', 'atr_pct'])

        # Bollinger Bands
        if len(prices) >= self.config.bb_period:
            bb_sma = np.mean(prices[-self.config.bb_period:])
            bb_std = np.std(prices[-self.config.bb_period:])
            bb_upper = bb_sma + self.config.bb_std * bb_std
            bb_lower = bb_sma - self.config.bb_std * bb_std
            bb_width = (bb_upper - bb_lower) / bb_sma if bb_sma != 0 else 0.0
            bb_range = bb_upper - bb_lower
            bb_position = (prices[-1] - bb_lower) / bb_range if bb_range != 0 else 0.5
            bb_position = np.clip(bb_position, 0, 1)
        else:
            bb_width = 0.0
            bb_position = 0.5
        features.extend([bb_width, bb_position])
        names.extend(['bb_width', 'bb_position'])

        # Realized volatility at different windows
        for window in [10, 30, 60, 300]:
            if len(prices) > window:
                returns = np.diff(prices[-window-1:]) / prices[-window-1:-1]
                vol = np.std(returns)
            else:
                vol = 0.0
            features.append(vol)
            names.append(f'volatility_{window}s')

        # 8. Momentum indicators
        # RSI
        if len(prices) >= self.config.rsi_period + 1:
            deltas = np.diff(prices[-self.config.rsi_period-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_gain > 0 else 50
            rsi_norm = (rsi - 50) / 50
        else:
            rsi_norm = 0.0
        features.append(rsi_norm)
        names.append('rsi_norm')

        # MACD
        if 12 in self._ema_state and 26 in self._ema_state:
            ema_12 = self._get_ema(prices, 12)
            ema_26 = self._get_ema(prices, 26)
            macd = ema_12 - ema_26
            macd_norm = macd / prices[-1] if prices[-1] != 0 else 0.0
        else:
            macd_norm = 0.0
        features.append(macd_norm)
        features.append(0.0)  # MACD histogram (simplified)
        names.extend(['macd_norm', 'macd_hist_norm'])

        # Stochastic
        if len(prices) >= self.config.rsi_period:
            low_min = np.min(lows[-self.config.rsi_period:])
            high_max = np.max(highs[-self.config.rsi_period:])
            denom = high_max - low_min
            if denom != 0:
                stoch_k = 100 * (prices[-1] - low_min) / denom
            else:
                stoch_k = 50
            stoch_k_norm = (stoch_k - 50) / 50
        else:
            stoch_k_norm = 0.0
        features.append(stoch_k_norm)
        features.append(stoch_k_norm)  # Simplified D
        names.extend(['stoch_k_norm', 'stoch_d_norm'])

        # 9. Microstructure features
        bar_direction = np.sign(bar.close - bar.open)
        features.append(bar_direction)
        names.append('bar_direction')

        total_range = bar.high - bar.low
        if total_range > 0:
            body = abs(bar.close - bar.open)
            body_ratio = body / total_range
            upper_wick = bar.high - max(bar.open, bar.close)
            upper_wick_ratio = upper_wick / total_range
            lower_wick = min(bar.open, bar.close) - bar.low
            lower_wick_ratio = lower_wick / total_range
        else:
            body_ratio = 1.0
            upper_wick_ratio = 0.0
            lower_wick_ratio = 0.0

        features.extend([body_ratio, upper_wick_ratio, lower_wick_ratio])
        names.extend(['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio'])

        # Gap from previous close
        if len(prices) >= 2:
            gap_ticks = (bar.open - prices[-2]) / MES_TICK_SIZE
        else:
            gap_ticks = 0.0
        features.append(gap_ticks)
        names.append('gap_ticks')

        # Range in ticks
        range_ticks = total_range / MES_TICK_SIZE
        features.append(range_ticks)
        names.append('range_ticks')

        # 10. Volume features
        for window in [10, 30, 60]:
            if len(volumes) >= window:
                vol_ma = np.mean(list(volumes)[-window:])
                vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1.0
            else:
                vol_ratio = 1.0
            features.append(vol_ratio)
            names.append(f'volume_ratio_{window}s')

        # Volume delta (simplified)
        features.append(0.0)  # volume_delta_norm
        features.append(0.0)  # obv_roc
        names.extend(['volume_delta_norm', 'obv_roc'])

        # 11. Multi-timeframe (simplified - using lagged minute data)
        # These would ideally be calculated from actual minute bars
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        names.extend(['htf_trend_1m', 'htf_momentum_1m', 'htf_vol_1m', 'htf_trend_5m', 'htf_momentum_5m'])

        # Store feature names
        self._feature_names = names

        return FeatureVector(
            features=np.array(features, dtype=np.float32),
            feature_names=names,
            timestamp=bar.timestamp,
            bar=bar,
            atr=self._current_atr,
        )

    def _update_ema_state(self, price: float) -> None:
        """Update EMA state incrementally."""
        for period in self.config.ema_periods:
            alpha = 2.0 / (period + 1)
            if period in self._ema_state:
                self._ema_state[period] = alpha * price + (1 - alpha) * self._ema_state[period]
            else:
                self._ema_state[period] = price

        # Also track for MACD
        for period in [12, 26, 9]:
            alpha = 2.0 / (period + 1)
            if period in self._ema_state:
                self._ema_state[period] = alpha * price + (1 - alpha) * self._ema_state[period]
            else:
                self._ema_state[period] = price

    def _update_vwap_state(self, bar: OHLCV) -> None:
        """Update VWAP state with session reset."""
        # Check for session reset
        bar_date = bar.timestamp.date()
        if self._vwap_reset_date != bar_date:
            # New session - reset VWAP
            self._vwap_cumsum_pv = 0.0
            self._vwap_cumsum_v = 0.0
            self._vwap_reset_date = bar_date
            logger.debug(f"VWAP reset for new session: {bar_date}")

        # Update VWAP
        typical_price = (bar.high + bar.low + bar.close) / 3
        self._vwap_cumsum_pv += typical_price * bar.volume
        self._vwap_cumsum_v += bar.volume

    def _calculate_minutes_to_close(self, timestamp: datetime) -> float:
        """Calculate minutes until RTH close."""
        t = timestamp.time()
        minutes_since_open = (
            (t.hour - RTH_START.hour) * 60 +
            (t.minute - RTH_START.minute) +
            t.second / 60.0
        )
        minutes_to_close = RTH_DURATION_MINUTES - max(0, minutes_since_open)
        return max(0, minutes_to_close)

    def _get_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA from price array."""
        if len(prices) < period:
            return prices[-1]

        alpha = 2.0 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def get_atr(self) -> float:
        """Get current ATR for position sizing."""
        return self._current_atr

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names or []

    def reset(self) -> None:
        """Reset all state (e.g., at start of new session)."""
        self._bars.clear()
        self._prices.clear()
        self._highs.clear()
        self._lows.clear()
        self._volumes.clear()
        self._ema_state.clear()
        self._vwap_cumsum_pv = 0.0
        self._vwap_cumsum_v = 0.0
        self._vwap_reset_date = None
        self._current_atr = 0.0
        logger.info("RealTimeFeatureEngine reset")
