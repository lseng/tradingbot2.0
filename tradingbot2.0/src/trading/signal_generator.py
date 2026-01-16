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

Reference: specs/live-trading-execution.md
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum
import logging

from src.trading.position_manager import Position

logger = logging.getLogger(__name__)


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

    # Exit triggers
    exit_on_opposite_signal: bool = True  # Exit if model predicts opposite direction
    exit_on_flat_signal: bool = False  # Exit if model predicts FLAT

    # Reversal settings
    allow_reversals: bool = True  # Allow direct reversals
    require_flat_first: bool = False  # Require flat before reversal


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

        logger.info(
            f"SignalGenerator initialized: "
            f"min_entry_conf={self.config.min_entry_confidence}, "
            f"min_exit_conf={self.config.min_exit_confidence}"
        )

    def generate(
        self,
        prediction: ModelPrediction,
        position: Position,
        risk_manager: 'RiskManager',
        current_atr: Optional[float] = None,
    ) -> Optional[Signal]:
        """
        Generate trading signal from model prediction.

        Args:
            prediction: Model prediction output
            position: Current position state
            risk_manager: Risk manager for limit checks
            current_atr: Current ATR for volatility-adjusted stops

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
        stop_ticks, target_ticks = self._calculate_stops(current_atr)

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
                    logger.info(
                        f"REVERSE_TO_SHORT signal: confidence={prediction.confidence:.2%}"
                    )
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
                    logger.info(
                        f"REVERSE_TO_LONG signal: confidence={prediction.confidence:.2%}"
                    )
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

    def _calculate_stops(self, current_atr: Optional[float]) -> tuple[float, float]:
        """
        Calculate stop and target distances.

        Args:
            current_atr: Current ATR in points

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
