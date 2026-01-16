"""
Risk Management Module for MES Futures Scalping Bot.

This module provides comprehensive risk controls to protect the $1,000 starting capital:
- Daily loss limits and drawdown tracking
- Position sizing based on account balance and confidence
- Stop loss strategies (ATR-based, fixed tick, structure-based)
- End-of-day flatten logic (HARD REQUIREMENT: flat by 4:30 PM NY)
- Circuit breakers for consecutive losses and market conditions

All limits are NON-NEGOTIABLE as capital preservation is the primary objective.
"""

from .risk_manager import RiskManager, RiskState, RiskLimits
from .position_sizing import PositionSizer, PositionSizeResult
from .stops import StopLossManager, StopType
from .eod_manager import EODManager, EODPhase
from .circuit_breakers import CircuitBreakers, CircuitBreakerState, BreakerType

__all__ = [
    # Core risk manager
    'RiskManager',
    'RiskState',
    'RiskLimits',
    # Position sizing
    'PositionSizer',
    'PositionSizeResult',
    # Stop loss
    'StopLossManager',
    'StopType',
    # EOD management
    'EODManager',
    'EODPhase',
    # Circuit breakers
    'CircuitBreakers',
    'CircuitBreakerState',
    'BreakerType',
]
