"""
Core Risk Manager for MES Futures Scalping Bot.

This module enforces all risk limits to protect the $1,000 starting capital.
All limits are NON-NEGOTIABLE - the account CANNOT be blown up.

Key Parameters (from spec):
- Starting Capital: $1,000
- Max Daily Loss: $50 (5%)
- Max Daily Drawdown: $75 (7.5%)
- Max Per-Trade Risk: $25 (2.5%)
- Kill Switch: $300 cumulative loss (30%)
- Min Account Balance: $700
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from enum import Enum
import threading
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingStatus(Enum):
    """Current trading status."""
    ACTIVE = "active"
    PAUSED = "paused"  # Temporary pause (circuit breaker)
    STOPPED_FOR_DAY = "stopped_for_day"  # Daily limit hit
    HALTED = "halted"  # Permanent halt (kill switch)
    MANUAL_REVIEW = "manual_review"  # Requires human intervention


@dataclass
class RiskLimits:
    """
    Risk limit configuration.

    All values are in USD unless otherwise specified.
    These are the NON-NEGOTIABLE limits from the spec.
    """
    # Capital parameters
    starting_capital: float = 1000.0
    min_account_balance: float = 700.0

    # Daily limits
    max_daily_loss: float = 50.0  # 5% of starting capital
    max_daily_drawdown: float = 75.0  # 7.5% of starting capital

    # Per-trade limits
    max_per_trade_risk: float = 25.0  # 2.5% of starting capital

    # Circuit breakers
    max_consecutive_losses: int = 5  # Triggers 30-min pause
    consecutive_loss_pause_seconds: int = 1800  # 30 minutes

    # Kill switch
    kill_switch_loss: float = 300.0  # 30% of starting capital

    # Drawdown requiring manual review
    max_account_drawdown: float = 200.0  # 20% of starting capital

    # Confidence threshold
    min_confidence: float = 0.60  # No trade below 60% confidence

    # MES contract specifications
    tick_size: float = 0.25  # MES minimum price movement
    tick_value: float = 1.25  # Dollar value per tick
    point_value: float = 5.0  # Dollar value per point (4 ticks)
    commission_per_side: float = 0.42  # $0.20 comm + $0.22 fee
    round_trip_commission: float = 0.84  # Total per round-trip


@dataclass
class RiskState:
    """
    Current risk state tracking.

    This state is updated in real-time and persisted across restarts.
    Thread-safe through locking mechanism.
    """
    # Account state
    account_balance: float = 1000.0
    peak_balance: float = 1000.0  # For drawdown calculation

    # Daily tracking (reset at 9:30 AM NY each day)
    current_date: Optional[date] = None
    daily_pnl: float = 0.0
    daily_peak_equity: float = 0.0
    daily_drawdown: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0

    # Cumulative tracking
    cumulative_loss: float = 0.0  # For kill switch
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0

    # Consecutive loss tracking
    consecutive_losses: int = 0

    # Status
    status: TradingStatus = TradingStatus.ACTIVE
    pause_until: Optional[datetime] = None
    halt_reason: Optional[str] = None

    # Open position P&L
    open_pnl: float = 0.0


class RiskManager:
    """
    Core risk manager enforcing all trading limits.

    This class is thread-safe and designed for async compatibility.
    State persists across restarts via JSON file.

    Usage:
        limits = RiskLimits()
        manager = RiskManager(limits)

        # Before each trade
        if manager.can_trade():
            # Check specific trade
            if manager.approve_trade(risk_amount=20.0, confidence=0.75):
                # Execute trade
                pass

        # After trade closes
        manager.record_trade_result(pnl=15.0)
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        state_file: Optional[Path] = None,
        auto_persist: bool = True,
    ):
        """
        Initialize risk manager.

        Args:
            limits: Risk limit configuration (uses defaults if None)
            state_file: Path to persist state (optional)
            auto_persist: Whether to auto-save state after updates
        """
        self.limits = limits or RiskLimits()
        self.state = RiskState(
            account_balance=self.limits.starting_capital,
            peak_balance=self.limits.starting_capital,
        )
        self.state_file = state_file
        self.auto_persist = auto_persist
        self._lock = threading.RLock()

        # Load persisted state if available
        if state_file and state_file.exists():
            self._load_state()

        logger.info(
            f"RiskManager initialized: balance=${self.state.account_balance:.2f}, "
            f"daily_loss_limit=${self.limits.max_daily_loss:.2f}, "
            f"per_trade_risk=${self.limits.max_per_trade_risk:.2f}"
        )

    def can_trade(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed, False otherwise.
        """
        with self._lock:
            # Check status
            if self.state.status == TradingStatus.HALTED:
                logger.warning(f"Trading halted: {self.state.halt_reason}")
                return False

            if self.state.status == TradingStatus.STOPPED_FOR_DAY:
                logger.info("Trading stopped for the day")
                return False

            if self.state.status == TradingStatus.MANUAL_REVIEW:
                logger.warning("Manual review required before trading")
                return False

            # Check pause expiry
            if self.state.status == TradingStatus.PAUSED:
                if self.state.pause_until and datetime.now() >= self.state.pause_until:
                    self.state.status = TradingStatus.ACTIVE
                    self.state.pause_until = None
                    logger.info("Pause period ended, trading resumed")
                else:
                    remaining = (self.state.pause_until - datetime.now()).total_seconds()
                    logger.info(f"Trading paused, {remaining:.0f}s remaining")
                    return False

            # Check account balance
            if self.state.account_balance < self.limits.min_account_balance:
                logger.error(
                    f"Account balance ${self.state.account_balance:.2f} below minimum "
                    f"${self.limits.min_account_balance:.2f}"
                )
                return False

            # Check kill switch
            if self.state.cumulative_loss >= self.limits.kill_switch_loss:
                self._trigger_kill_switch(
                    f"Cumulative loss ${self.state.cumulative_loss:.2f} "
                    f"exceeds kill switch ${self.limits.kill_switch_loss:.2f}"
                )
                return False

            # Check daily loss limit
            if abs(self.state.daily_pnl) >= self.limits.max_daily_loss and self.state.daily_pnl < 0:
                self._stop_for_day("Daily loss limit reached")
                return False

            # Check daily drawdown
            if self.state.daily_drawdown >= self.limits.max_daily_drawdown:
                self._stop_for_day("Daily drawdown limit reached")
                return False

            return True

    def approve_trade(
        self,
        risk_amount: float,
        confidence: float,
        reason: str = "",
    ) -> bool:
        """
        Approve a specific trade based on risk parameters.

        Args:
            risk_amount: Dollar amount at risk for this trade
            confidence: Model confidence (0-1)
            reason: Optional reason for logging

        Returns:
            True if trade is approved, False otherwise.
        """
        with self._lock:
            # First check if trading is allowed at all
            if not self.can_trade():
                return False

            # Check confidence threshold
            if confidence < self.limits.min_confidence:
                logger.info(
                    f"Trade rejected: confidence {confidence:.1%} below "
                    f"minimum {self.limits.min_confidence:.1%}"
                )
                return False

            # Check per-trade risk limit
            if risk_amount > self.limits.max_per_trade_risk:
                logger.warning(
                    f"Trade rejected: risk ${risk_amount:.2f} exceeds "
                    f"max per-trade risk ${self.limits.max_per_trade_risk:.2f}"
                )
                return False

            # Check if this trade could breach daily loss limit
            potential_loss = abs(self.state.daily_pnl) + risk_amount
            if self.state.daily_pnl < 0 and potential_loss > self.limits.max_daily_loss:
                logger.warning(
                    f"Trade rejected: potential loss ${potential_loss:.2f} could "
                    f"breach daily limit ${self.limits.max_daily_loss:.2f}"
                )
                return False

            logger.info(
                f"Trade approved: risk=${risk_amount:.2f}, confidence={confidence:.1%}"
                f"{f', reason={reason}' if reason else ''}"
            )
            return True

    def record_trade_result(self, pnl: float, is_win: Optional[bool] = None) -> None:
        """
        Record the result of a completed trade.

        Args:
            pnl: Net P&L of the trade (positive for profit, negative for loss)
            is_win: Override win/loss determination (uses pnl sign if None)
        """
        with self._lock:
            # Determine win/loss
            if is_win is None:
                is_win = pnl > 0

            # Update account balance
            self.state.account_balance += pnl

            # Update peak balance
            if self.state.account_balance > self.state.peak_balance:
                self.state.peak_balance = self.state.account_balance

            # Update daily tracking
            self.state.daily_pnl += pnl
            self.state.trades_today += 1

            # Update daily peak and drawdown
            current_equity = self.state.account_balance + self.state.open_pnl
            if current_equity > self.state.daily_peak_equity:
                self.state.daily_peak_equity = current_equity
            self.state.daily_drawdown = max(
                0, self.state.daily_peak_equity - current_equity
            )

            # Update win/loss counts
            if is_win:
                self.state.wins_today += 1
                self.state.total_wins += 1
                self.state.consecutive_losses = 0
            else:
                self.state.losses_today += 1
                self.state.total_losses += 1
                self.state.consecutive_losses += 1

                # Track cumulative loss for kill switch
                if pnl < 0:
                    self.state.cumulative_loss += abs(pnl)

            self.state.total_trades += 1

            # Log result
            logger.info(
                f"Trade recorded: pnl=${pnl:+.2f}, balance=${self.state.account_balance:.2f}, "
                f"daily_pnl=${self.state.daily_pnl:+.2f}, consecutive_losses={self.state.consecutive_losses}"
            )

            # Check circuit breakers
            self._check_circuit_breakers()

            # Check risk limits
            self._check_risk_limits()

            # Persist state
            if self.auto_persist:
                self._persist_state()

    def update_open_pnl(self, pnl: float) -> None:
        """
        Update unrealized P&L from open positions.

        Args:
            pnl: Current unrealized P&L
        """
        with self._lock:
            self.state.open_pnl = pnl

            # Update daily drawdown based on equity
            current_equity = self.state.account_balance + self.state.open_pnl
            if current_equity > self.state.daily_peak_equity:
                self.state.daily_peak_equity = current_equity
            self.state.daily_drawdown = max(
                0, self.state.daily_peak_equity - current_equity
            )

    def reset_daily_state(self, current_date: Optional[date] = None) -> None:
        """
        Reset daily tracking state. Call at 9:30 AM NY each trading day.

        Args:
            current_date: Date to set (uses today if None)
        """
        with self._lock:
            self.state.current_date = current_date or date.today()
            self.state.daily_pnl = 0.0
            self.state.daily_peak_equity = self.state.account_balance
            self.state.daily_drawdown = 0.0
            self.state.trades_today = 0
            self.state.wins_today = 0
            self.state.losses_today = 0

            # Reset daily stop if not permanently halted
            if self.state.status == TradingStatus.STOPPED_FOR_DAY:
                self.state.status = TradingStatus.ACTIVE

            logger.info(
                f"Daily state reset for {self.state.current_date}, "
                f"balance=${self.state.account_balance:.2f}"
            )

            if self.auto_persist:
                self._persist_state()

    def get_remaining_daily_risk(self) -> float:
        """
        Get remaining risk budget for the day.

        Returns:
            Dollar amount that can still be risked today.
        """
        with self._lock:
            if self.state.daily_pnl >= 0:
                # No losses yet, full daily limit available
                return self.limits.max_daily_loss
            else:
                # Some losses, reduce available risk
                return max(0, self.limits.max_daily_loss - abs(self.state.daily_pnl))

    def get_metrics(self) -> dict:
        """
        Get current risk metrics for logging/display.

        Returns:
            Dictionary of current risk metrics.
        """
        with self._lock:
            return {
                "account_balance": self.state.account_balance,
                "peak_balance": self.state.peak_balance,
                "account_drawdown": self.state.peak_balance - self.state.account_balance,
                "account_drawdown_pct": (
                    (self.state.peak_balance - self.state.account_balance)
                    / self.state.peak_balance * 100
                ) if self.state.peak_balance > 0 else 0,
                "daily_pnl": self.state.daily_pnl,
                "daily_drawdown": self.state.daily_drawdown,
                "trades_today": self.state.trades_today,
                "wins_today": self.state.wins_today,
                "losses_today": self.state.losses_today,
                "win_rate_today": (
                    self.state.wins_today / self.state.trades_today * 100
                ) if self.state.trades_today > 0 else 0,
                "consecutive_losses": self.state.consecutive_losses,
                "cumulative_loss": self.state.cumulative_loss,
                "remaining_daily_risk": self.get_remaining_daily_risk(),
                "status": self.state.status.value,
                "total_trades": self.state.total_trades,
                "total_wins": self.state.total_wins,
                "total_losses": self.state.total_losses,
                "overall_win_rate": (
                    self.state.total_wins / self.state.total_trades * 100
                ) if self.state.total_trades > 0 else 0,
            }

    def _check_circuit_breakers(self) -> None:
        """Check and trigger circuit breakers based on consecutive losses."""
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            self._pause_trading(
                self.limits.consecutive_loss_pause_seconds,
                f"{self.state.consecutive_losses} consecutive losses"
            )
        elif self.state.consecutive_losses >= 3:
            # 3 losses = 15 minute pause
            self._pause_trading(
                900,  # 15 minutes
                f"{self.state.consecutive_losses} consecutive losses (warning)"
            )

    def _check_risk_limits(self) -> None:
        """Check all risk limits and update status accordingly."""
        # Check kill switch
        if self.state.cumulative_loss >= self.limits.kill_switch_loss:
            self._trigger_kill_switch(
                f"Cumulative loss ${self.state.cumulative_loss:.2f} "
                f"exceeds kill switch ${self.limits.kill_switch_loss:.2f}"
            )
            return

        # Check account drawdown requiring manual review
        account_drawdown = self.state.peak_balance - self.state.account_balance
        if account_drawdown >= self.limits.max_account_drawdown:
            self.state.status = TradingStatus.MANUAL_REVIEW
            self.state.halt_reason = (
                f"Account drawdown ${account_drawdown:.2f} exceeds "
                f"${self.limits.max_account_drawdown:.2f}, manual review required"
            )
            logger.error(self.state.halt_reason)
            return

        # Check daily loss limit
        if self.state.daily_pnl < 0 and abs(self.state.daily_pnl) >= self.limits.max_daily_loss:
            self._stop_for_day("Daily loss limit reached")
            return

        # Check daily drawdown limit
        if self.state.daily_drawdown >= self.limits.max_daily_drawdown:
            self._stop_for_day("Daily drawdown limit reached")

    def _pause_trading(self, seconds: int, reason: str) -> None:
        """Temporarily pause trading."""
        self.state.status = TradingStatus.PAUSED
        self.state.pause_until = datetime.now().replace(microsecond=0)
        from datetime import timedelta
        self.state.pause_until += timedelta(seconds=seconds)
        logger.warning(f"Trading paused for {seconds}s: {reason}")

    def _stop_for_day(self, reason: str) -> None:
        """Stop trading for the rest of the day."""
        self.state.status = TradingStatus.STOPPED_FOR_DAY
        self.state.halt_reason = reason
        logger.error(f"Trading stopped for day: {reason}")

    def _trigger_kill_switch(self, reason: str) -> None:
        """Permanently halt all trading."""
        self.state.status = TradingStatus.HALTED
        self.state.halt_reason = reason
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

    def halt(self, reason: str = "Manual halt requested") -> None:
        """
        Manually halt all trading immediately.

        This is the public interface for the kill switch, allowing operators
        to immediately stop all trading in emergency situations.

        Go-Live Checklist #12: Manual kill switch accessible and tested.

        Args:
            reason: Reason for the halt (logged for audit trail)

        Usage:
            manager.halt("Detected unusual market conditions")
            manager.halt("Emergency stop - operator decision")
        """
        with self._lock:
            self.state.status = TradingStatus.HALTED
            self.state.halt_reason = f"Manual halt: {reason}"
            logger.critical(f"MANUAL HALT TRIGGERED: {reason}")

            if self.auto_persist:
                self._persist_state()

    def reset_halt(self, new_balance: Optional[float] = None) -> bool:
        """
        Reset from a halted state after manual review.

        This requires manual intervention and should only be called
        after a human operator has reviewed the situation.

        Args:
            new_balance: Optional new account balance to set

        Returns:
            True if successfully reset, False if reset not allowed

        Usage:
            # After manual review
            manager.reset_halt(new_balance=800.0)
        """
        with self._lock:
            if self.state.status != TradingStatus.HALTED:
                logger.warning("reset_halt called but not in HALTED state")
                return False

            if new_balance is not None:
                self.state.account_balance = new_balance
                self.state.peak_balance = max(self.state.peak_balance, new_balance)

            # Reset cumulative loss when manually resetting
            self.state.cumulative_loss = 0.0
            self.state.consecutive_losses = 0
            self.state.status = TradingStatus.ACTIVE
            self.state.halt_reason = None

            logger.info(
                f"Trading MANUALLY RESET from halt, balance=${self.state.account_balance:.2f}, "
                f"cumulative_loss reset to $0.00"
            )

            if self.auto_persist:
                self._persist_state()

            return True

    def _persist_state(self) -> None:
        """Save current state to file."""
        if not self.state_file:
            return

        try:
            state_dict = {
                "account_balance": self.state.account_balance,
                "peak_balance": self.state.peak_balance,
                "current_date": self.state.current_date.isoformat() if self.state.current_date else None,
                "daily_pnl": self.state.daily_pnl,
                "daily_peak_equity": self.state.daily_peak_equity,
                "daily_drawdown": self.state.daily_drawdown,
                "trades_today": self.state.trades_today,
                "wins_today": self.state.wins_today,
                "losses_today": self.state.losses_today,
                "cumulative_loss": self.state.cumulative_loss,
                "total_trades": self.state.total_trades,
                "total_wins": self.state.total_wins,
                "total_losses": self.state.total_losses,
                "consecutive_losses": self.state.consecutive_losses,
                "status": self.state.status.value,
                "pause_until": self.state.pause_until.isoformat() if self.state.pause_until else None,
                "halt_reason": self.state.halt_reason,
            }

            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def _load_state(self) -> None:
        """Load state from file."""
        if not self.state_file or not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)

            self.state.account_balance = state_dict.get("account_balance", self.limits.starting_capital)
            self.state.peak_balance = state_dict.get("peak_balance", self.limits.starting_capital)

            if state_dict.get("current_date"):
                self.state.current_date = date.fromisoformat(state_dict["current_date"])

            self.state.daily_pnl = state_dict.get("daily_pnl", 0.0)
            self.state.daily_peak_equity = state_dict.get("daily_peak_equity", 0.0)
            self.state.daily_drawdown = state_dict.get("daily_drawdown", 0.0)
            self.state.trades_today = state_dict.get("trades_today", 0)
            self.state.wins_today = state_dict.get("wins_today", 0)
            self.state.losses_today = state_dict.get("losses_today", 0)
            self.state.cumulative_loss = state_dict.get("cumulative_loss", 0.0)
            self.state.total_trades = state_dict.get("total_trades", 0)
            self.state.total_wins = state_dict.get("total_wins", 0)
            self.state.total_losses = state_dict.get("total_losses", 0)
            self.state.consecutive_losses = state_dict.get("consecutive_losses", 0)
            self.state.status = TradingStatus(state_dict.get("status", "active"))

            if state_dict.get("pause_until"):
                self.state.pause_until = datetime.fromisoformat(state_dict["pause_until"])

            self.state.halt_reason = state_dict.get("halt_reason")

            logger.info(f"Loaded state from {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def manual_resume(self, new_balance: Optional[float] = None) -> bool:
        """
        Manually resume trading after review (requires human intervention).

        Args:
            new_balance: Optional new account balance to set

        Returns:
            True if successfully resumed, False if cannot resume (kill switch)
        """
        with self._lock:
            if self.state.status == TradingStatus.HALTED:
                logger.error("Cannot resume: kill switch triggered. Manual intervention required.")
                return False

            if new_balance is not None:
                self.state.account_balance = new_balance
                self.state.peak_balance = max(self.state.peak_balance, new_balance)

            self.state.status = TradingStatus.ACTIVE
            self.state.pause_until = None
            self.state.halt_reason = None
            self.state.consecutive_losses = 0

            logger.info(f"Trading manually resumed, balance=${self.state.account_balance:.2f}")

            if self.auto_persist:
                self._persist_state()

            return True
