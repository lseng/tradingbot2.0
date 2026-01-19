"""
Signal Generator for Live Trading.

Generates trading signals based on ML model predictions and current position state.
Integrates with risk manager to ensure all signals comply with risk limits.

Signal Types:
- LONG_ENTRY: Open long position
- SHORT_ENTRY: Open short position
- EXIT_LONG: Close long position
- EXIT_SHORT: Close short position
- REVERSE_TO_LONG: Close short, open long
- REVERSE_TO_SHORT: Close long, open short
- FLATTEN: Emergency close all positions
- HOLD: No action

Reversal Constraints (per specs/risk-management.md):
- Must have high-confidence opposite signal (> 75%)
- Cannot reverse more than 2x in same bar range
- Cooldown period after reversal: 30 seconds minimum
- Reversal counts as new trade for daily limits

Reference: specs/live-trading-execution.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum
import logging

from src.trading.position_manager import Position

logger = logging.getLogger(__name__)


@dataclass
class BarRange:
    """
    Tracks the price range of a bar for reversal constraint checking.

    Used to enforce the "Cannot reverse more than 2x in same bar range" rule
    from specs/risk-management.md.

    Attributes:
        high: Bar high price
        low: Bar low price
        timestamp: When this bar range was established
    """
    high: float
    low: float
    timestamp: datetime = field(default_factory=datetime.now)

    def contains_price(self, price: float, tolerance: float = 0.0) -> bool:
        """
        Check if a price is within this bar range.

        Args:
            price: Price to check
            tolerance: Additional tolerance in price units (default 0)

        Returns:
            True if price is within [low - tolerance, high + tolerance]
        """
        return (self.low - tolerance) <= price <= (self.high + tolerance)

    def overlaps(self, other: 'BarRange', tolerance: float = 0.0) -> bool:
        """
        Check if this bar range overlaps with another.

        Args:
            other: Another BarRange to check
            tolerance: Additional tolerance in price units

        Returns:
            True if the ranges overlap
        """
        return not (self.high + tolerance < other.low or
                    self.low - tolerance > other.high)


class SignalType(Enum):
    """Trading signal types."""
    HOLD = "hold"
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    REVERSE_TO_LONG = "reverse_to_long"
    REVERSE_TO_SHORT = "reverse_to_short"
    FLATTEN = "flatten"


@dataclass
class ModelPrediction:
    """
    Prediction output from ML model.

    Attributes:
        direction: Predicted direction (-1=short, 0=flat, 1=long)
        confidence: Prediction confidence (0-1)
        predicted_move: Expected price move in ticks
        volatility: Current volatility estimate for position sizing
        timestamp: Prediction timestamp
    """
    direction: int  # -1 (short), 0 (flat), 1 (long)
    confidence: float  # 0-1
    predicted_move: float = 0.0  # Expected ticks
    volatility: float = 0.0  # For position sizing
    timestamp: Optional[datetime] = None


@dataclass
class Signal:
    """
    Trading signal to execute.

    Attributes:
        signal_type: Type of signal to execute
        confidence: Model confidence for this signal
        stop_ticks: Suggested stop loss distance in ticks
        target_ticks: Suggested take profit distance in ticks
        predicted_class: Original model prediction class (0, 1, 2)
        reason: Human-readable reason for the signal
        timestamp: Signal generation timestamp
    """
    signal_type: SignalType
    confidence: float
    stop_ticks: float = 8.0  # Default 8 ticks = $10 risk
    target_ticks: float = 12.0  # Default 12 ticks = $15 target (1:1.5 R:R)
    predicted_class: int = 1  # 0=DOWN, 1=FLAT, 2=UP
    reason: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SignalConfig:
    """
    Configuration for signal generation.

    All thresholds are configurable to allow optimization.
    """
    # Confidence thresholds
    min_entry_confidence: float = 0.65  # Minimum for entries
    min_exit_confidence: float = 0.55  # Minimum for exits
    min_reversal_confidence: float = 0.75  # Higher bar for reversals

    # Stop/target defaults
    default_stop_ticks: float = 8.0  # 8 ticks = $10 risk per contract
    default_target_ticks: float = 12.0  # 12 ticks = $15 target

    # Volatility-adjusted stop multiplier
    atr_stop_multiplier: float = 1.5  # Stop = ATR * multiplier

    # Cooldown after exits (prevent overtrading)
    exit_cooldown_seconds: float = 30.0

    # Cooldown after reversals (per specs/risk-management.md: 30s minimum)
    reversal_cooldown_seconds: float = 30.0

    # Exit triggers
    exit_on_opposite_signal: bool = True  # Exit if model predicts opposite direction
    exit_on_flat_signal: bool = False  # Exit if model predicts FLAT

    # Reversal settings (per specs/risk-management.md)
    allow_reversals: bool = True  # Allow direct reversals
    require_flat_first: bool = False  # Require flat before reversal
    max_reversals_per_bar_range: int = 2  # Cannot reverse more than 2x in same bar range
    bar_range_tolerance: float = 0.25  # Tolerance for bar range overlap (1 tick for MES)


class SignalGenerator:
    """
    Generates trading signals from model predictions.

    Coordinates with:
    - ML model predictions (direction, confidence)
    - Position manager (current position state)
    - Risk manager (can_trade check)

    Usage:
        generator = SignalGenerator(config)

        # Generate signal
        signal = generator.generate(
            prediction=model_prediction,
            position=position_manager.position,
            risk_manager=risk_manager,
            current_atr=1.5,
        )

        if signal and signal.signal_type != SignalType.HOLD:
            # Execute signal
            pass
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Initialize signal generator.

        Args:
            config: Signal generation configuration
        """
        self.config = config or SignalConfig()
        self._last_exit_time: Optional[datetime] = None
        self._last_reversal_time: Optional[datetime] = None

        # Reversal bar-range constraint tracking (per specs/risk-management.md)
        self._current_bar_range: Optional[BarRange] = None
        self._reversals_in_bar_range: int = 0
        self._reversal_bar_ranges: List[BarRange] = []  # History for debugging

        logger.info(
            f"SignalGenerator initialized: "
            f"min_entry_conf={self.config.min_entry_confidence}, "
            f"min_exit_conf={self.config.min_exit_confidence}, "
            f"max_reversals_per_bar={self.config.max_reversals_per_bar_range}"
        )

    def generate(
        self,
        prediction: ModelPrediction,
        position: Position,
        risk_manager: 'RiskManager',
        current_atr: Optional[float] = None,
        eod_tighten_factor: Optional[float] = None,
    ) -> Optional[Signal]:
        """
        Generate trading signal from model prediction.

        Args:
            prediction: Model prediction output
            position: Current position state
            risk_manager: Risk manager for limit checks
            current_atr: Current ATR for volatility-adjusted stops
            eod_tighten_factor: EOD stop tightening factor (1.0 = no tightening,
                               lower = tighter stops). See EODManager.get_stop_tighten_factor()

        Returns:
            Signal to execute, or None if no action
        """
        # Check if trading is allowed
        if not risk_manager.can_trade():
            logger.debug("Trading not allowed by risk manager")
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Trading not allowed by risk manager",
            )

        # Check exit cooldown
        if self._in_exit_cooldown():
            logger.debug("In exit cooldown period")
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Exit cooldown active",
            )

        # Calculate stop/target based on ATR or defaults
        # 2.6 FIX: Apply EOD time-decay tightening
        stop_ticks, target_ticks = self._calculate_stops(current_atr, eod_tighten_factor)

        # Generate signal based on position state
        if position.is_flat:
            return self._generate_entry_signal(prediction, stop_ticks, target_ticks)
        elif position.is_long:
            return self._generate_long_signal(prediction, position, stop_ticks, target_ticks)
        elif position.is_short:
            return self._generate_short_signal(prediction, position, stop_ticks, target_ticks)
        else:
            # Should not happen
            logger.warning(f"Unexpected position state: {position}")
            return None

    def _generate_entry_signal(
        self,
        prediction: ModelPrediction,
        stop_ticks: float,
        target_ticks: float,
    ) -> Optional[Signal]:
        """Generate entry signal when flat."""
        # Check minimum confidence
        if prediction.confidence < self.config.min_entry_confidence:
            logger.debug(
                f"Entry rejected: confidence {prediction.confidence:.2%} "
                f"< {self.config.min_entry_confidence:.2%}"
            )
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=prediction.confidence,
                reason=f"Confidence below entry threshold ({prediction.confidence:.2%})",
            )

        # Generate signal based on direction
        if prediction.direction == 1:  # UP/LONG
            logger.info(
                f"LONG_ENTRY signal: confidence={prediction.confidence:.2%}, "
                f"stop={stop_ticks} ticks, target={target_ticks} ticks"
            )
            return Signal(
                signal_type=SignalType.LONG_ENTRY,
                confidence=prediction.confidence,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                predicted_class=2,  # UP
                reason="Model predicts UP",
            )

        elif prediction.direction == -1:  # DOWN/SHORT
            logger.info(
                f"SHORT_ENTRY signal: confidence={prediction.confidence:.2%}, "
                f"stop={stop_ticks} ticks, target={target_ticks} ticks"
            )
            return Signal(
                signal_type=SignalType.SHORT_ENTRY,
                confidence=prediction.confidence,
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                predicted_class=0,  # DOWN
                reason="Model predicts DOWN",
            )

        else:  # FLAT prediction
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=prediction.confidence,
                predicted_class=1,  # FLAT
                reason="Model predicts FLAT",
            )

    def _generate_long_signal(
        self,
        prediction: ModelPrediction,
        position: Position,
        stop_ticks: float,
        target_ticks: float,
    ) -> Optional[Signal]:
        """Generate signal when in long position."""
        # Check for exit signals
        if prediction.direction == -1:  # DOWN prediction while long
            if prediction.confidence >= self.config.min_reversal_confidence:
                if self.config.allow_reversals and not self.config.require_flat_first:
                    # Check bar-range constraint (per specs/risk-management.md)
                    can_reverse, reason = self._can_reverse_in_bar_range()
                    if not can_reverse:
                        logger.info(
                            f"Reversal blocked: {reason} (confidence={prediction.confidence:.2%})"
                        )
                        # Fall through to check for regular exit instead
                    else:
                        logger.info(
                            f"REVERSE_TO_SHORT signal: confidence={prediction.confidence:.2%}"
                        )
                        self._record_reversal()
                        return Signal(
                            signal_type=SignalType.REVERSE_TO_SHORT,
                            confidence=prediction.confidence,
                            stop_ticks=stop_ticks,
                            target_ticks=target_ticks,
                            predicted_class=0,  # DOWN
                            reason="Model predicts DOWN with high confidence (reversal)",
                        )

            if (self.config.exit_on_opposite_signal and
                    prediction.confidence >= self.config.min_exit_confidence):
                logger.info(
                    f"EXIT_LONG signal: confidence={prediction.confidence:.2%}"
                )
                self._record_exit_time()
                return Signal(
                    signal_type=SignalType.EXIT_LONG,
                    confidence=prediction.confidence,
                    predicted_class=0,  # DOWN
                    reason="Model predicts DOWN (exit long)",
                )

        elif prediction.direction == 0:  # FLAT prediction
            if self.config.exit_on_flat_signal and prediction.confidence >= self.config.min_exit_confidence:
                logger.info(
                    f"EXIT_LONG signal (flat prediction): confidence={prediction.confidence:.2%}"
                )
                self._record_exit_time()
                return Signal(
                    signal_type=SignalType.EXIT_LONG,
                    confidence=prediction.confidence,
                    predicted_class=1,  # FLAT
                    reason="Model predicts FLAT (exit long)",
                )

        # No signal - hold position
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=prediction.confidence,
            reason="Holding long position",
        )

    def _generate_short_signal(
        self,
        prediction: ModelPrediction,
        position: Position,
        stop_ticks: float,
        target_ticks: float,
    ) -> Optional[Signal]:
        """Generate signal when in short position."""
        # Check for exit signals
        if prediction.direction == 1:  # UP prediction while short
            if prediction.confidence >= self.config.min_reversal_confidence:
                if self.config.allow_reversals and not self.config.require_flat_first:
                    # Check bar-range constraint (per specs/risk-management.md)
                    can_reverse, reason = self._can_reverse_in_bar_range()
                    if not can_reverse:
                        logger.info(
                            f"Reversal blocked: {reason} (confidence={prediction.confidence:.2%})"
                        )
                        # Fall through to check for regular exit instead
                    else:
                        logger.info(
                            f"REVERSE_TO_LONG signal: confidence={prediction.confidence:.2%}"
                        )
                        self._record_reversal()
                        return Signal(
                            signal_type=SignalType.REVERSE_TO_LONG,
                            confidence=prediction.confidence,
                            stop_ticks=stop_ticks,
                            target_ticks=target_ticks,
                            predicted_class=2,  # UP
                            reason="Model predicts UP with high confidence (reversal)",
                        )

            if (self.config.exit_on_opposite_signal and
                    prediction.confidence >= self.config.min_exit_confidence):
                logger.info(
                    f"EXIT_SHORT signal: confidence={prediction.confidence:.2%}"
                )
                self._record_exit_time()
                return Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    confidence=prediction.confidence,
                    predicted_class=2,  # UP
                    reason="Model predicts UP (exit short)",
                )

        elif prediction.direction == 0:  # FLAT prediction
            if self.config.exit_on_flat_signal and prediction.confidence >= self.config.min_exit_confidence:
                logger.info(
                    f"EXIT_SHORT signal (flat prediction): confidence={prediction.confidence:.2%}"
                )
                self._record_exit_time()
                return Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    confidence=prediction.confidence,
                    predicted_class=1,  # FLAT
                    reason="Model predicts FLAT (exit short)",
                )

        # No signal - hold position
        return Signal(
            signal_type=SignalType.HOLD,
            confidence=prediction.confidence,
            reason="Holding short position",
        )

    def _calculate_stops(
        self,
        current_atr: Optional[float],
        eod_tighten_factor: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Calculate stop and target distances.

        2.6 FIX: Applies EOD time-decay tightening when factor provided.

        Args:
            current_atr: Current ATR in points
            eod_tighten_factor: EOD tightening factor (1.0 = no tightening,
                               0.7 = 30% tighter). Tightens stops as EOD approaches.

        Returns:
            Tuple of (stop_ticks, target_ticks)
        """
        if current_atr and current_atr > 0:
            # ATR-based stops (convert ATR points to ticks)
            stop_ticks = max(
                4.0,  # Minimum 4 ticks = $5
                (current_atr * self.config.atr_stop_multiplier) / 0.25  # Convert to ticks
            )
            # Target at 1.5x stop for 1:1.5 R:R
            target_ticks = stop_ticks * 1.5
        else:
            # Use defaults
            stop_ticks = self.config.default_stop_ticks
            target_ticks = self.config.default_target_ticks

        # 2.6 FIX: Apply EOD time-decay tightening
        if eod_tighten_factor is not None and eod_tighten_factor < 1.0:
            original_stop = stop_ticks
            stop_ticks = max(4.0, stop_ticks * eod_tighten_factor)  # Maintain minimum
            # Target is tightened proportionally
            target_ticks = target_ticks * eod_tighten_factor

            logger.debug(
                f"EOD tightening applied: stop {original_stop:.1f} -> {stop_ticks:.1f} ticks "
                f"(factor={eod_tighten_factor:.2f})"
            )

        return stop_ticks, target_ticks

    def _in_exit_cooldown(self) -> bool:
        """Check if in post-exit cooldown period."""
        if self._last_exit_time is None:
            return False

        elapsed = (datetime.now() - self._last_exit_time).total_seconds()
        return elapsed < self.config.exit_cooldown_seconds

    def _record_exit_time(self) -> None:
        """Record exit time for cooldown tracking."""
        self._last_exit_time = datetime.now()

    def generate_flatten_signal(self, reason: str = "EOD flatten") -> Signal:
        """
        Generate emergency flatten signal.

        Used for:
        - EOD flatten (4:25 PM NY)
        - Risk limit breach
        - Manual intervention

        Args:
            reason: Reason for flatten

        Returns:
            Flatten signal
        """
        logger.warning(f"FLATTEN signal: {reason}")
        return Signal(
            signal_type=SignalType.FLATTEN,
            confidence=1.0,
            reason=reason,
        )

    def reset_cooldown(self) -> None:
        """Reset exit cooldown (e.g., at start of new session)."""
        self._last_exit_time = None
        logger.debug("Exit cooldown reset")

    # ========== Reversal Bar-Range Constraint Methods ==========
    # Per specs/risk-management.md: "Cannot reverse more than 2x in same bar range"

    def update_bar_range(self, high: float, low: float, current_price: float) -> None:
        """
        Update the current bar range for reversal constraint checking.

        Should be called on each new bar (typically 1-second bars in live trading).
        If the new bar range overlaps with the current tracked range, keep tracking.
        If it's a new non-overlapping range, reset the reversal counter.

        Args:
            high: Current bar high price
            low: Current bar low price
            current_price: Current price (for range extension logic)
        """
        new_range = BarRange(high=high, low=low)

        if self._current_bar_range is None:
            # First bar - establish range
            self._current_bar_range = new_range
            self._reversals_in_bar_range = 0
            logger.debug(f"Bar range established: {low:.2f} - {high:.2f}")
            return

        # Check if new bar overlaps with current range
        if self._current_bar_range.overlaps(new_range, self.config.bar_range_tolerance):
            # Extend the range if needed
            extended_high = max(self._current_bar_range.high, high)
            extended_low = min(self._current_bar_range.low, low)
            self._current_bar_range = BarRange(
                high=extended_high,
                low=extended_low,
                timestamp=self._current_bar_range.timestamp
            )
            logger.debug(
                f"Bar range extended: {self._current_bar_range.low:.2f} - "
                f"{self._current_bar_range.high:.2f}, reversals={self._reversals_in_bar_range}"
            )
        else:
            # New non-overlapping range - reset counter
            old_range = self._current_bar_range
            self._reversal_bar_ranges.append(old_range)  # Keep history
            self._current_bar_range = new_range
            self._reversals_in_bar_range = 0
            logger.debug(
                f"New bar range: {low:.2f} - {high:.2f} (previous: "
                f"{old_range.low:.2f} - {old_range.high:.2f})"
            )

    def _in_reversal_cooldown(self) -> bool:
        """
        Check if in post-reversal cooldown period.

        Separate from exit cooldown per specs/risk-management.md:
        "Cooldown period after reversal: 30 seconds minimum"
        """
        if self._last_reversal_time is None:
            return False

        elapsed = (datetime.now() - self._last_reversal_time).total_seconds()
        return elapsed < self.config.reversal_cooldown_seconds

    def _can_reverse_in_bar_range(self) -> tuple[bool, str]:
        """
        Check if a reversal is allowed based on the bar-range constraint.

        Per specs/risk-management.md: "Cannot reverse more than 2x in same bar range"

        Returns:
            Tuple of (can_reverse, reason)
        """
        # Check reversal cooldown first
        if self._in_reversal_cooldown():
            elapsed = 0.0
            if self._last_reversal_time:
                elapsed = (datetime.now() - self._last_reversal_time).total_seconds()
            remaining = self.config.reversal_cooldown_seconds - elapsed
            return False, f"Reversal cooldown active ({remaining:.1f}s remaining)"

        # Check bar-range constraint
        if self._reversals_in_bar_range >= self.config.max_reversals_per_bar_range:
            return False, (
                f"Max {self.config.max_reversals_per_bar_range} reversals "
                f"reached in bar range "
                f"({self._current_bar_range.low:.2f} - {self._current_bar_range.high:.2f})"
                if self._current_bar_range else "Max reversals reached"
            )

        return True, ""

    def _record_reversal(self) -> None:
        """Record a reversal event for constraint tracking."""
        self._last_reversal_time = datetime.now()
        self._reversals_in_bar_range += 1
        self._record_exit_time()  # Reversals also trigger exit cooldown

        logger.info(
            f"Reversal recorded: count={self._reversals_in_bar_range} "
            f"in bar range {self._current_bar_range.low:.2f} - "
            f"{self._current_bar_range.high:.2f}"
            if self._current_bar_range else
            f"Reversal recorded: count={self._reversals_in_bar_range}"
        )

    def reset_reversal_state(self) -> None:
        """
        Reset reversal tracking state (e.g., at start of new session).

        This resets:
        - Reversal cooldown
        - Bar-range tracking
        - Reversal counter
        """
        self._last_reversal_time = None
        self._current_bar_range = None
        self._reversals_in_bar_range = 0
        self._reversal_bar_ranges.clear()
        logger.debug("Reversal state reset")

    def get_reversal_state(self) -> dict:
        """
        Get current reversal constraint state for monitoring/debugging.

        Returns:
            Dictionary with reversal tracking state
        """
        return {
            "reversals_in_bar_range": self._reversals_in_bar_range,
            "max_allowed": self.config.max_reversals_per_bar_range,
            "current_bar_range": (
                {"high": self._current_bar_range.high, "low": self._current_bar_range.low}
                if self._current_bar_range else None
            ),
            "in_reversal_cooldown": self._in_reversal_cooldown(),
            "last_reversal_time": self._last_reversal_time.isoformat() if self._last_reversal_time else None,
        }


def is_entry_signal(signal: Signal) -> bool:
    """Check if signal is an entry signal."""
    return signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)


def is_exit_signal(signal: Signal) -> bool:
    """Check if signal is an exit signal."""
    return signal.signal_type in (
        SignalType.EXIT_LONG,
        SignalType.EXIT_SHORT,
        SignalType.FLATTEN,
    )


def is_reversal_signal(signal: Signal) -> bool:
    """Check if signal is a reversal signal."""
    return signal.signal_type in (
        SignalType.REVERSE_TO_LONG,
        SignalType.REVERSE_TO_SHORT,
    )


def signal_to_direction(signal: Signal) -> int:
    """
    Get position direction from signal.

    Returns:
        1 for long, -1 for short, 0 for flat/exit
    """
    if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.REVERSE_TO_LONG):
        return 1
    elif signal.signal_type in (SignalType.SHORT_ENTRY, SignalType.REVERSE_TO_SHORT):
        return -1
    else:
        return 0
