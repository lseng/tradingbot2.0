"""
Position Sizing Module for MES Futures Scalping Bot.

Calculates position size based on:
- Account balance tier
- Risk per trade percentage
- Stop loss distance
- Model confidence level

Key Rules (from spec):
- $700-$1,000: 1 contract max, 2% risk
- $1,000-$1,500: 2 contracts max, 2% risk
- $1,500-$2,000: 3 contracts max, 2% risk
- $2,000-$3,000: 4 contracts max, 2% risk
- $3,000+: 5+ contracts max, 1.5% risk

Confidence multipliers:
- <60%: No trade (0x)
- 60-70%: 0.5x
- 70-80%: 1.0x
- 80-90%: 1.5x
- >90%: 2.0x (capped)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """
    Result of position size calculation.

    Contains the recommended position size along with all calculation details
    for transparency and debugging.
    """
    contracts: int  # Final position size (0 means no trade)
    base_contracts: int  # Before confidence scaling
    dollar_risk: float  # Total dollar risk for this position
    risk_per_contract: float  # Risk per contract based on stop
    stop_distance_ticks: float  # Stop distance in ticks
    confidence_multiplier: float  # Applied confidence multiplier
    eod_multiplier: float  # Applied EOD time-based multiplier (1.0=normal, 0.5=reduced, 0.0=no trade)
    reason: str  # Explanation of sizing decision
    max_contracts_for_tier: int  # Max allowed for current balance tier
    risk_pct_used: float  # Actual risk percentage of account


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    # MES contract specifications
    tick_size: float = 0.25
    tick_value: float = 1.25  # $1.25 per tick per contract
    point_value: float = 5.0  # $5.00 per point

    # Default risk parameters
    default_risk_pct: float = 0.02  # 2% default
    reduced_risk_pct: float = 0.015  # 1.5% for larger accounts

    # Balance tiers (upper bound, max contracts, risk pct)
    # Tuple: (balance_threshold, max_contracts, risk_pct)
    balance_tiers: Tuple[Tuple[float, int, float], ...] = (
        (1000.0, 1, 0.02),   # $700-$1,000: 1 contract, 2%
        (1500.0, 2, 0.02),   # $1,000-$1,500: 2 contracts, 2%
        (2000.0, 3, 0.02),   # $1,500-$2,000: 3 contracts, 2%
        (3000.0, 4, 0.02),   # $2,000-$3,000: 4 contracts, 2%
        (float('inf'), 10, 0.015),  # $3,000+: 5+ contracts, 1.5%
    )

    # Confidence multipliers
    # Tuple: (min_confidence_threshold, multiplier)
    # At or above threshold, use the multiplier. Below 60% = no trade.
    confidence_multipliers: Tuple[Tuple[float, float], ...] = (
        (0.60, 0.5),   # 60-70%: half size
        (0.70, 1.0),   # 70-80%: full size
        (0.80, 1.5),   # 80-90%: 1.5x size
        (0.90, 2.0),   # 90%+: 2x size (capped)
    )

    # Minimum confidence to trade
    min_confidence_threshold: float = 0.60

    # Minimum account balance to trade
    min_balance: float = 700.0


class PositionSizer:
    """
    Position sizing calculator for MES futures.

    Calculates appropriate position size based on:
    1. Account balance tier -> max contracts and risk %
    2. Stop loss distance -> risk per contract
    3. Model confidence -> size multiplier

    Usage:
        sizer = PositionSizer()
        result = sizer.calculate(
            account_balance=1000.0,
            stop_ticks=8,
            confidence=0.75
        )
        print(f"Trade {result.contracts} contracts, risk ${result.dollar_risk}")
    """

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """
        Initialize position sizer.

        Args:
            config: Position sizing configuration (uses defaults if None)
        """
        self.config = config or PositionSizingConfig()

    def calculate(
        self,
        account_balance: float,
        stop_ticks: float,
        confidence: float,
        max_risk_override: Optional[float] = None,
        eod_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """
        Calculate position size for a trade.

        Args:
            account_balance: Current account balance in dollars
            stop_ticks: Stop loss distance in ticks (e.g., 8 ticks = 2 points)
            confidence: Model confidence level (0-1)
            max_risk_override: Override max dollar risk (optional)
            eod_multiplier: EOD time-based multiplier from EODManager (1.0=normal,
                           0.5=reduced after 4PM, 0.0=no trading after 4:15PM)

        Returns:
            PositionSizeResult with calculated position size and details
        """
        # Check EOD multiplier - 0.0 means no trading allowed
        if eod_multiplier <= 0:
            return PositionSizeResult(
                contracts=0,
                base_contracts=0,
                dollar_risk=0.0,
                risk_per_contract=0.0,
                stop_distance_ticks=stop_ticks,
                confidence_multiplier=0.0,
                eod_multiplier=eod_multiplier,
                reason="No new positions allowed (EOD CLOSE_ONLY or AFTER_HOURS phase)",
                max_contracts_for_tier=0,
                risk_pct_used=0.0,
            )

        # Check minimum balance
        if account_balance < self.config.min_balance:
            return PositionSizeResult(
                contracts=0,
                base_contracts=0,
                dollar_risk=0.0,
                risk_per_contract=0.0,
                stop_distance_ticks=stop_ticks,
                confidence_multiplier=0.0,
                eod_multiplier=eod_multiplier,
                reason=f"Account balance ${account_balance:.2f} below minimum ${self.config.min_balance:.2f}",
                max_contracts_for_tier=0,
                risk_pct_used=0.0,
            )

        # Get confidence multiplier
        confidence_mult = self._get_confidence_multiplier(confidence)

        if confidence_mult == 0:
            return PositionSizeResult(
                contracts=0,
                base_contracts=0,
                dollar_risk=0.0,
                risk_per_contract=0.0,
                stop_distance_ticks=stop_ticks,
                confidence_multiplier=0.0,
                eod_multiplier=eod_multiplier,
                reason=f"Confidence {confidence:.1%} below minimum threshold (60%)",
                max_contracts_for_tier=0,
                risk_pct_used=0.0,
            )

        # Get balance tier parameters
        max_contracts, risk_pct = self._get_tier_params(account_balance)

        # Calculate risk per contract based on stop distance
        risk_per_contract = stop_ticks * self.config.tick_value

        if risk_per_contract <= 0:
            return PositionSizeResult(
                contracts=0,
                base_contracts=0,
                dollar_risk=0.0,
                risk_per_contract=0.0,
                stop_distance_ticks=stop_ticks,
                confidence_multiplier=confidence_mult,
                eod_multiplier=eod_multiplier,
                reason="Invalid stop distance (must be positive)",
                max_contracts_for_tier=max_contracts,
                risk_pct_used=0.0,
            )

        # Calculate dollar risk budget
        dollar_risk_budget = account_balance * risk_pct
        if max_risk_override is not None:
            dollar_risk_budget = min(dollar_risk_budget, max_risk_override)

        # Calculate base contracts (before confidence scaling)
        base_contracts = int(dollar_risk_budget / risk_per_contract)
        base_contracts = max(1, min(base_contracts, max_contracts))

        # Apply confidence multiplier
        scaled_contracts = int(base_contracts * confidence_mult)

        # Apply EOD multiplier (reduces position size near end of day)
        # e.g., 0.5 after 4:00 PM NY per spec
        eod_scaled_contracts = int(scaled_contracts * eod_multiplier)

        # Ensure at least 1 contract if confidence is above threshold
        # and cap at max for tier
        final_contracts = max(1, min(eod_scaled_contracts, max_contracts))

        # Calculate actual dollar risk
        actual_dollar_risk = final_contracts * risk_per_contract
        actual_risk_pct = actual_dollar_risk / account_balance

        # Build reason string, noting EOD reduction if applicable
        reason = (
            f"Calculated {final_contracts} contracts for ${account_balance:.0f} balance, "
            f"{stop_ticks} tick stop, {confidence:.1%} confidence"
        )
        if eod_multiplier < 1.0:
            reason += f" (EOD reduced by {eod_multiplier:.0%})"

        return PositionSizeResult(
            contracts=final_contracts,
            base_contracts=base_contracts,
            dollar_risk=actual_dollar_risk,
            risk_per_contract=risk_per_contract,
            stop_distance_ticks=stop_ticks,
            confidence_multiplier=confidence_mult,
            eod_multiplier=eod_multiplier,
            reason=reason,
            max_contracts_for_tier=max_contracts,
            risk_pct_used=actual_risk_pct,
        )

    def calculate_max_stop_for_risk(
        self,
        account_balance: float,
        target_risk_dollars: float,
        contracts: int = 1,
    ) -> float:
        """
        Calculate maximum stop distance (in ticks) for a given risk budget.

        Useful for determining stop placement based on risk constraints.

        Args:
            account_balance: Current account balance
            target_risk_dollars: Target risk in dollars
            contracts: Number of contracts

        Returns:
            Maximum stop distance in ticks
        """
        if contracts <= 0:
            return 0.0

        risk_per_contract = target_risk_dollars / contracts
        max_ticks = risk_per_contract / self.config.tick_value

        return max_ticks

    def calculate_contracts_for_risk(
        self,
        target_risk_dollars: float,
        stop_ticks: float,
    ) -> int:
        """
        Calculate number of contracts for a specific risk amount.

        Args:
            target_risk_dollars: Desired dollar risk
            stop_ticks: Stop loss distance in ticks

        Returns:
            Number of contracts (minimum 1 if valid inputs)
        """
        if stop_ticks <= 0:
            return 0

        risk_per_contract = stop_ticks * self.config.tick_value
        contracts = int(target_risk_dollars / risk_per_contract)

        return max(1, contracts)

    def get_tier_info(self, account_balance: float) -> dict:
        """
        Get information about the current balance tier.

        Args:
            account_balance: Current account balance

        Returns:
            Dictionary with tier information
        """
        max_contracts, risk_pct = self._get_tier_params(account_balance)

        # Find tier boundaries
        prev_threshold = self.config.min_balance
        for threshold, _, _ in self.config.balance_tiers:
            # Use < to match _get_tier_params (boundary belongs to next tier)
            if account_balance < threshold:
                return {
                    "balance": account_balance,
                    "tier_min": prev_threshold,
                    "tier_max": threshold,
                    "max_contracts": max_contracts,
                    "risk_pct": risk_pct,
                    "max_dollar_risk": account_balance * risk_pct,
                }
            prev_threshold = threshold

        # Highest tier
        return {
            "balance": account_balance,
            "tier_min": prev_threshold,
            "tier_max": float('inf'),
            "max_contracts": max_contracts,
            "risk_pct": risk_pct,
            "max_dollar_risk": account_balance * risk_pct,
        }

    def _get_tier_params(self, account_balance: float) -> Tuple[int, float]:
        """
        Get max contracts and risk percentage for balance tier.

        Args:
            account_balance: Current account balance

        Returns:
            Tuple of (max_contracts, risk_percentage)
        """
        for threshold, max_contracts, risk_pct in self.config.balance_tiers:
            # Use <= so boundary belongs to current tier (conservative risk management)
            # Per spec: $700-$1,000 = 1 contract, $1,000-$1,500 = 2 contracts
            # At exactly $1,000, use 1 contract (lower risk)
            if account_balance <= threshold:
                return max_contracts, risk_pct

        # Return last tier if above all thresholds
        return self.config.balance_tiers[-1][1], self.config.balance_tiers[-1][2]

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """
        Get position size multiplier based on confidence level.

        Args:
            confidence: Model confidence (0-1)

        Returns:
            Size multiplier (0 means no trade)
        """
        # Below minimum threshold = no trade
        if confidence < self.config.min_confidence_threshold:
            return 0.0

        # Find the appropriate multiplier for this confidence level
        # Iterate through thresholds in reverse to find highest applicable
        result_mult = self.config.confidence_multipliers[0][1]  # Default to first tier

        for threshold, multiplier in self.config.confidence_multipliers:
            if confidence >= threshold:
                result_mult = multiplier
            else:
                break

        return result_mult


def calculate_position_size(
    account_balance: float,
    stop_ticks: float,
    confidence: float,
    tick_value: float = 1.25,
    risk_pct: float = 0.02,
    max_contracts: int = 1,
) -> int:
    """
    Simple position size calculation function.

    For quick calculations without instantiating PositionSizer.

    Args:
        account_balance: Current account balance
        stop_ticks: Stop loss distance in ticks
        confidence: Model confidence (0-1)
        tick_value: Dollar value per tick (default $1.25 for MES)
        risk_pct: Risk percentage per trade (default 2%)
        max_contracts: Maximum contracts allowed

    Returns:
        Number of contracts to trade
    """
    # Check confidence threshold
    if confidence < 0.60:
        return 0

    # Calculate base contracts
    dollar_risk = account_balance * risk_pct
    risk_per_contract = stop_ticks * tick_value

    if risk_per_contract <= 0:
        return 0

    base_contracts = int(dollar_risk / risk_per_contract)
    base_contracts = max(1, min(base_contracts, max_contracts))

    # Apply confidence scaling
    if confidence < 0.70:
        multiplier = 0.5
    elif confidence < 0.80:
        multiplier = 1.0
    elif confidence < 0.90:
        multiplier = 1.5
    else:
        multiplier = 2.0

    final_contracts = max(1, int(base_contracts * multiplier))
    return min(final_contracts, max_contracts)
