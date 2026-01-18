"""
Circuit Breakers Module for MES Futures Scalping Bot.

Provides trading halts based on:
1. Consecutive losses (3 losses = 15-min, 5 losses = 30-min)
2. Daily loss limit hit (stop for day)
3. Max drawdown hit (manual review required)
4. Market conditions (volatility, spread, volume)

Key Parameters (from spec):
- 3 consecutive losses: 15-minute halt
- 5 consecutive losses: 30-minute halt
- Daily loss limit: Stop for day
- Max drawdown: Indefinite (manual review)
- Volatility > 3x normal: Reduce size or pause
- Spread > 2 ticks: Pause until normal
- Volume < 10% avg: Reduce size or pause
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List
import threading
import logging

logger = logging.getLogger(__name__)


class BreakerType(Enum):
    """Type of circuit breaker."""
    CONSECUTIVE_LOSSES = "consecutive_losses"
    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    HIGH_VOLATILITY = "high_volatility"
    WIDE_SPREAD = "wide_spread"
    LOW_VOLUME = "low_volume"
    MANUAL = "manual"


class BreakerAction(Enum):
    """Action to take when breaker triggers."""
    PAUSE = "pause"  # Temporary pause
    REDUCE_SIZE = "reduce_size"  # Continue with reduced size
    STOP_FOR_DAY = "stop_for_day"  # Stop until next session
    HALT = "halt"  # Halt indefinitely


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    # Consecutive loss breakers
    loss_3_pause_seconds: int = 900  # 15 minutes
    loss_5_pause_seconds: int = 1800  # 30 minutes

    # Market condition thresholds
    volatility_multiplier_threshold: float = 3.0  # ATR > 3x normal
    max_spread_ticks: int = 2  # Spread > 2 ticks = pause
    min_volume_pct: float = 0.10  # Volume < 10% avg = pause

    # Size reduction factors
    high_volatility_size_factor: float = 0.5  # 50% size in high vol
    low_volume_size_factor: float = 0.5  # 50% size in low vol


@dataclass
class CircuitBreakerState:
    """Current state of circuit breakers."""
    # Active breakers
    active_breakers: Dict[BreakerType, dict] = field(default_factory=dict)

    # Pause state
    is_paused: bool = False
    pause_until: Optional[datetime] = None
    pause_reason: Optional[str] = None

    # Size reduction
    size_multiplier: float = 1.0
    size_reduction_reasons: List[str] = field(default_factory=list)

    # Halt state
    is_halted: bool = False
    halt_reason: Optional[str] = None
    requires_manual_review: bool = False


class CircuitBreakers:
    """
    Circuit breaker system for trading risk control.

    Monitors various conditions and triggers trading halts or
    size reductions when thresholds are breached.

    Usage:
        breakers = CircuitBreakers()

        # After each trade
        breakers.record_loss()  # or record_win()

        # Before each trade
        if breakers.can_trade():
            size_mult = breakers.get_size_multiplier()
            # Execute trade with adjusted size

        # Update market conditions
        breakers.update_market_conditions(
            current_atr=3.5,
            normal_atr=1.0,
            spread_ticks=1,
            volume_pct=0.85
        )
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breakers.

        Args:
            config: Circuit breaker configuration (uses defaults if None)
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = threading.RLock()

        # Track consecutive losses
        self._consecutive_losses = 0

        logger.info("CircuitBreakers initialized")

    def can_trade(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed, False otherwise
        """
        with self._lock:
            # Check halt
            if self.state.is_halted:
                logger.warning(f"Trading halted: {self.state.halt_reason}")
                return False

            # Check pause
            if self.state.is_paused:
                if self.state.pause_until and datetime.now() >= self.state.pause_until:
                    # Pause expired
                    self._clear_pause()
                else:
                    remaining = 0
                    if self.state.pause_until:
                        remaining = (self.state.pause_until - datetime.now()).total_seconds()
                    logger.info(f"Trading paused: {self.state.pause_reason}, {remaining:.0f}s remaining")
                    return False

            return True

    def get_size_multiplier(self) -> float:
        """
        Get current position size multiplier.

        Returns:
            Multiplier (1.0 = full size, 0.5 = half size, etc.)
        """
        with self._lock:
            return self.state.size_multiplier

    def record_win(self) -> None:
        """Record a winning trade - resets consecutive loss counter."""
        with self._lock:
            self._consecutive_losses = 0
            logger.debug("Win recorded, consecutive losses reset to 0")

    def record_loss(self) -> None:
        """Record a losing trade - may trigger circuit breakers."""
        with self._lock:
            self._consecutive_losses += 1
            logger.info(f"Loss recorded, consecutive losses: {self._consecutive_losses}")

            # Check consecutive loss breakers
            if self._consecutive_losses >= 5:
                self._trigger_pause(
                    BreakerType.CONSECUTIVE_LOSSES,
                    self.config.loss_5_pause_seconds,
                    f"5 consecutive losses"
                )
            elif self._consecutive_losses >= 3:
                self._trigger_pause(
                    BreakerType.CONSECUTIVE_LOSSES,
                    self.config.loss_3_pause_seconds,
                    f"3 consecutive losses"
                )

    def trigger_daily_loss_stop(self, daily_loss: float, limit: float) -> None:
        """
        Trigger stop for day due to daily loss limit.

        Args:
            daily_loss: Current daily loss amount
            limit: Daily loss limit
        """
        with self._lock:
            self.state.active_breakers[BreakerType.DAILY_LOSS] = {
                "triggered_at": datetime.now(),
                "daily_loss": daily_loss,
                "limit": limit,
            }
            self.state.is_halted = True
            self.state.halt_reason = f"Daily loss limit: ${daily_loss:.2f} >= ${limit:.2f}"

            logger.error(f"CIRCUIT BREAKER: {self.state.halt_reason}")

    def trigger_max_drawdown_halt(self, drawdown: float, limit: float) -> None:
        """
        Trigger halt due to max drawdown - requires manual review.

        Args:
            drawdown: Current drawdown amount
            limit: Max drawdown limit
        """
        with self._lock:
            self.state.active_breakers[BreakerType.MAX_DRAWDOWN] = {
                "triggered_at": datetime.now(),
                "drawdown": drawdown,
                "limit": limit,
            }
            self.state.is_halted = True
            self.state.halt_reason = f"Max drawdown: ${drawdown:.2f} >= ${limit:.2f}"
            self.state.requires_manual_review = True

            logger.critical(f"CIRCUIT BREAKER - MANUAL REVIEW REQUIRED: {self.state.halt_reason}")

    def trigger_halt(self, reason: str) -> None:
        """
        Trigger immediate trading halt - requires manual intervention.

        1.16 FIX: Added for unprotected position scenario where stop order
        fails AND emergency exit fails, leaving position without protection.

        Args:
            reason: Reason for the halt
        """
        with self._lock:
            self.state.active_breakers[BreakerType.MANUAL] = {
                "triggered_at": datetime.now(),
                "reason": reason,
            }
            self.state.is_halted = True
            self.state.halt_reason = reason
            self.state.requires_manual_review = True

            logger.critical(f"CIRCUIT BREAKER - TRADING HALTED: {reason}")

    def update_market_conditions(
        self,
        current_atr: Optional[float] = None,
        normal_atr: Optional[float] = None,
        spread_ticks: Optional[float] = None,
        volume_pct: Optional[float] = None,
    ) -> None:
        """
        Update market conditions and check for breakers.

        Args:
            current_atr: Current ATR value
            normal_atr: Normal/baseline ATR value
            spread_ticks: Current bid-ask spread in ticks
            volume_pct: Current volume as percentage of average
        """
        with self._lock:
            size_reasons = []

            # Check volatility
            if current_atr is not None and normal_atr is not None and normal_atr > 0:
                vol_mult = current_atr / normal_atr
                if vol_mult > self.config.volatility_multiplier_threshold:
                    self.state.active_breakers[BreakerType.HIGH_VOLATILITY] = {
                        "current_atr": current_atr,
                        "normal_atr": normal_atr,
                        "multiplier": vol_mult,
                    }
                    size_reasons.append(
                        f"High volatility ({vol_mult:.1f}x) - "
                        f"{self.config.high_volatility_size_factor:.0%} size"
                    )
                elif BreakerType.HIGH_VOLATILITY in self.state.active_breakers:
                    del self.state.active_breakers[BreakerType.HIGH_VOLATILITY]

            # Check spread
            if spread_ticks is not None:
                if spread_ticks > self.config.max_spread_ticks:
                    self.state.active_breakers[BreakerType.WIDE_SPREAD] = {
                        "spread_ticks": spread_ticks,
                        "threshold": self.config.max_spread_ticks,
                    }
                    # Wide spread triggers pause, not size reduction
                    self._trigger_pause(
                        BreakerType.WIDE_SPREAD,
                        60,  # 1 minute pause, re-check
                        f"Wide spread: {spread_ticks} ticks > {self.config.max_spread_ticks} threshold"
                    )
                elif BreakerType.WIDE_SPREAD in self.state.active_breakers:
                    del self.state.active_breakers[BreakerType.WIDE_SPREAD]

            # Check volume
            if volume_pct is not None:
                if volume_pct < self.config.min_volume_pct:
                    self.state.active_breakers[BreakerType.LOW_VOLUME] = {
                        "volume_pct": volume_pct,
                        "threshold": self.config.min_volume_pct,
                    }
                    size_reasons.append(
                        f"Low volume ({volume_pct:.1%}) - "
                        f"{self.config.low_volume_size_factor:.0%} size"
                    )
                elif BreakerType.LOW_VOLUME in self.state.active_breakers:
                    del self.state.active_breakers[BreakerType.LOW_VOLUME]

            # Calculate size multiplier
            self._update_size_multiplier(size_reasons)

    def _trigger_pause(
        self,
        breaker_type: BreakerType,
        seconds: int,
        reason: str,
    ) -> None:
        """Trigger a temporary trading pause."""
        self.state.active_breakers[breaker_type] = {
            "triggered_at": datetime.now(),
            "pause_seconds": seconds,
            "reason": reason,
        }
        self.state.is_paused = True
        self.state.pause_until = datetime.now() + timedelta(seconds=seconds)
        self.state.pause_reason = reason

        logger.warning(f"CIRCUIT BREAKER - PAUSE {seconds}s: {reason}")

    def _clear_pause(self) -> None:
        """Clear pause state."""
        self.state.is_paused = False
        self.state.pause_until = None
        self.state.pause_reason = None

        # Clear consecutive loss breaker if it was the cause
        if BreakerType.CONSECUTIVE_LOSSES in self.state.active_breakers:
            del self.state.active_breakers[BreakerType.CONSECUTIVE_LOSSES]

        logger.info("Trading pause cleared")

    def _update_size_multiplier(self, reasons: List[str]) -> None:
        """Update size multiplier based on active breakers."""
        multiplier = 1.0

        if BreakerType.HIGH_VOLATILITY in self.state.active_breakers:
            multiplier = min(multiplier, self.config.high_volatility_size_factor)

        if BreakerType.LOW_VOLUME in self.state.active_breakers:
            multiplier = min(multiplier, self.config.low_volume_size_factor)

        self.state.size_multiplier = multiplier
        self.state.size_reduction_reasons = reasons

        if multiplier < 1.0:
            logger.info(f"Size multiplier updated to {multiplier:.0%}: {', '.join(reasons)}")

    def reset_daily(self) -> None:
        """
        Reset daily breakers. Call at start of new trading day.

        Does NOT reset max drawdown or manual halts.
        """
        with self._lock:
            self._consecutive_losses = 0

            # Clear daily loss breaker
            if BreakerType.DAILY_LOSS in self.state.active_breakers:
                del self.state.active_breakers[BreakerType.DAILY_LOSS]

            # Clear pause state
            self._clear_pause()

            # Reset halt if it was daily loss (not max drawdown)
            if self.state.is_halted and not self.state.requires_manual_review:
                self.state.is_halted = False
                self.state.halt_reason = None

            logger.info("Circuit breakers reset for new trading day")

    def manual_reset(self) -> bool:
        """
        Manually reset all breakers - requires human intervention.

        Returns:
            True if reset successful
        """
        with self._lock:
            self._consecutive_losses = 0
            self.state = CircuitBreakerState()

            logger.info("Circuit breakers manually reset")
            return True

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Dictionary with current status
        """
        with self._lock:
            return {
                "can_trade": self.can_trade(),
                "is_paused": self.state.is_paused,
                "pause_until": self.state.pause_until.isoformat() if self.state.pause_until else None,
                "pause_reason": self.state.pause_reason,
                "is_halted": self.state.is_halted,
                "halt_reason": self.state.halt_reason,
                "requires_manual_review": self.state.requires_manual_review,
                "size_multiplier": self.state.size_multiplier,
                "size_reduction_reasons": self.state.size_reduction_reasons,
                "consecutive_losses": self._consecutive_losses,
                "active_breakers": list(self.state.active_breakers.keys()),
            }


def check_market_conditions(
    atr: float,
    baseline_atr: float,
    spread_ticks: float,
    volume_ratio: float,
) -> Dict[str, bool]:
    """
    Quick check of market conditions for trading suitability.

    Args:
        atr: Current ATR
        baseline_atr: Normal/baseline ATR
        spread_ticks: Current spread in ticks
        volume_ratio: Current volume / average volume

    Returns:
        Dictionary with condition checks
    """
    return {
        "volatility_normal": atr <= baseline_atr * 3.0,
        "spread_acceptable": spread_ticks <= 2,
        "volume_adequate": volume_ratio >= 0.10,
        "tradeable": (
            atr <= baseline_atr * 3.0 and
            spread_ticks <= 2 and
            volume_ratio >= 0.10
        ),
    }
