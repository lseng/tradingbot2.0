"""
Transaction Cost Model for MES Futures

This module accurately models the costs of trading MES futures through TopstepX.
Correct cost modeling is critical for realistic backtest results - using wrong
costs can lead to strategies that appear profitable but lose money live.

MES (Micro E-mini S&P 500) Cost Structure:
- Commission: $0.20 per side
- Exchange Fee (CME): $0.22 per side
- Total Round-Trip: $0.84 per contract

NOTE: The legacy evaluation.py uses $5.00 commission which is WRONG by 6x.
This module provides the correct MES-specific costs.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MESCostConfig:
    """
    Configuration for MES futures transaction costs.

    These are the actual costs for trading MES through TopstepX.
    Costs are per CONTRACT, not per tick or per dollar.

    Attributes:
        commission_per_side: Broker commission per side (entry or exit)
        exchange_fee_per_side: CME exchange fee per side

    Example:
        For a 1-contract round-trip trade:
        - Entry: $0.20 + $0.22 = $0.42
        - Exit: $0.20 + $0.22 = $0.42
        - Total: $0.84
    """
    commission_per_side: float = 0.20  # TopstepX commission
    exchange_fee_per_side: float = 0.22  # CME exchange fee

    @property
    def per_side_cost(self) -> float:
        """Total cost per side (entry OR exit) per contract."""
        return self.commission_per_side + self.exchange_fee_per_side

    @property
    def round_trip_cost(self) -> float:
        """Total cost for a complete round-trip (entry + exit) per contract."""
        return self.per_side_cost * 2


class TransactionCostModel:
    """
    Transaction cost calculator for MES futures trading.

    This class calculates all trading costs including commission and exchange fees.
    It tracks cumulative costs for performance analysis and ensures costs are
    properly deducted from gross P&L to calculate net P&L.

    Why this matters:
    - A strategy with 55% win rate and 1:1 R:R looks profitable
    - But with $0.84 cost per trade and small profits, costs eat returns
    - Example: 10 tick target = $12.50 gross, net = $12.50 - $0.84 = $11.66
    - That's a 6.7% cost drag per trade

    Usage:
        model = TransactionCostModel()
        cost = model.calculate_round_trip_cost(contracts=2)  # $1.68
        net_pnl = gross_pnl - cost
    """

    def __init__(self, config: Optional[MESCostConfig] = None):
        """
        Initialize the transaction cost model.

        Args:
            config: Cost configuration. Uses MES defaults if not provided.
        """
        self.config = config or MESCostConfig()
        self._total_commission = 0.0
        self._total_trades = 0

    def calculate_entry_cost(self, contracts: int) -> float:
        """
        Calculate the cost to enter a position.

        Args:
            contracts: Number of contracts to enter

        Returns:
            Entry cost in dollars
        """
        if contracts <= 0:
            return 0.0
        return self.config.per_side_cost * contracts

    def calculate_exit_cost(self, contracts: int) -> float:
        """
        Calculate the cost to exit a position.

        Args:
            contracts: Number of contracts to exit

        Returns:
            Exit cost in dollars
        """
        if contracts <= 0:
            return 0.0
        return self.config.per_side_cost * contracts

    def calculate_round_trip_cost(self, contracts: int) -> float:
        """
        Calculate the total cost for a complete trade (entry + exit).

        This is the most commonly used method for backtest P&L calculations.

        Args:
            contracts: Number of contracts traded

        Returns:
            Total round-trip cost in dollars
        """
        if contracts <= 0:
            return 0.0
        return self.config.round_trip_cost * contracts

    def record_trade(self, contracts: int) -> float:
        """
        Record a completed trade and return its cost.

        This method tracks cumulative costs for performance reporting.

        Args:
            contracts: Number of contracts in the completed trade

        Returns:
            Round-trip cost for this trade
        """
        cost = self.calculate_round_trip_cost(contracts)
        self._total_commission += cost
        self._total_trades += 1
        return cost

    def get_total_commission(self) -> float:
        """Get total commission paid across all recorded trades."""
        return self._total_commission

    def get_total_trades(self) -> int:
        """Get total number of recorded trades."""
        return self._total_trades

    def get_average_cost_per_trade(self) -> float:
        """Get average commission per trade."""
        if self._total_trades == 0:
            return 0.0
        return self._total_commission / self._total_trades

    def reset(self) -> None:
        """Reset cumulative tracking (e.g., for new backtest run)."""
        self._total_commission = 0.0
        self._total_trades = 0

    def calculate_breakeven_ticks(self, contracts: int = 1, tick_value: float = 1.25) -> float:
        """
        Calculate how many ticks profit needed to break even after costs.

        Useful for understanding minimum viable trade size.

        Args:
            contracts: Number of contracts
            tick_value: Dollar value per tick (MES = $1.25)

        Returns:
            Number of ticks needed to cover round-trip costs

        Example:
            For 1 contract: $0.84 / $1.25 = 0.67 ticks
            So any trade making 1+ ticks is profitable after costs.
        """
        cost = self.calculate_round_trip_cost(contracts)
        return cost / (tick_value * contracts) if contracts > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"TransactionCostModel("
            f"per_side=${self.config.per_side_cost:.2f}, "
            f"round_trip=${self.config.round_trip_cost:.2f})"
        )


# Convenience function for quick cost calculation
def calculate_mes_cost(contracts: int) -> float:
    """
    Quick helper to calculate MES round-trip cost.

    Args:
        contracts: Number of contracts

    Returns:
        Total cost in dollars
    """
    return MESCostConfig().round_trip_cost * contracts
