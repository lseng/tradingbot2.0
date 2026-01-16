"""
Trading constants and contract specifications.

This module defines all constants used throughout the trading system:
- Contract specifications (tick size, tick value, point value)
- Trading session times (RTH, ETH, EOD)
- Risk parameters
- Commission and fee structures

All values are sourced from TopstepX/CME specifications and the
risk-management.md specification.
"""

from dataclasses import dataclass
from datetime import time
from typing import Optional
from zoneinfo import ZoneInfo


# =============================================================================
# Timezone
# =============================================================================

NY_TIMEZONE = ZoneInfo("America/New_York")
UTC_TIMEZONE = ZoneInfo("UTC")


# =============================================================================
# MES (Micro E-mini S&P 500) Specifications
# =============================================================================

# Price movement
MES_TICK_SIZE = 0.25  # Minimum price movement in points
MES_TICK_VALUE = 1.25  # Dollar value per tick per contract ($)
MES_POINT_VALUE = 5.00  # Dollar value per point (4 ticks = 1 point)

# Commission and fees (TopstepX)
MES_COMMISSION_PER_SIDE = 0.20  # Commission per side ($)
MES_EXCHANGE_FEE_PER_SIDE = 0.22  # CME exchange fee per side ($)
MES_TOTAL_PER_SIDE = MES_COMMISSION_PER_SIDE + MES_EXCHANGE_FEE_PER_SIDE  # $0.42
MES_ROUND_TRIP_COST = MES_TOTAL_PER_SIDE * 2  # $0.84 per round-trip

# Margin (approximate, varies by broker)
MES_MARGIN_INTRADAY = 50.0  # Intraday margin per contract ($)
MES_MARGIN_OVERNIGHT = 1500.0  # Overnight margin per contract ($)


# =============================================================================
# ES (E-mini S&P 500) Specifications
# =============================================================================

ES_TICK_SIZE = 0.25  # Minimum price movement in points
ES_TICK_VALUE = 12.50  # Dollar value per tick per contract ($)
ES_POINT_VALUE = 50.00  # Dollar value per point (4 ticks = 1 point)

ES_COMMISSION_PER_SIDE = 0.59  # Commission per side ($)
ES_EXCHANGE_FEE_PER_SIDE = 0.47  # CME exchange fee per side ($)
ES_TOTAL_PER_SIDE = ES_COMMISSION_PER_SIDE + ES_EXCHANGE_FEE_PER_SIDE
ES_ROUND_TRIP_COST = ES_TOTAL_PER_SIDE * 2


# =============================================================================
# MNQ (Micro E-mini Nasdaq) Specifications
# =============================================================================

MNQ_TICK_SIZE = 0.25  # Minimum price movement in points
MNQ_TICK_VALUE = 0.50  # Dollar value per tick per contract ($)
MNQ_POINT_VALUE = 2.00  # Dollar value per point

MNQ_COMMISSION_PER_SIDE = 0.20
MNQ_EXCHANGE_FEE_PER_SIDE = 0.22
MNQ_TOTAL_PER_SIDE = MNQ_COMMISSION_PER_SIDE + MNQ_EXCHANGE_FEE_PER_SIDE
MNQ_ROUND_TRIP_COST = MNQ_TOTAL_PER_SIDE * 2


# =============================================================================
# NQ (E-mini Nasdaq) Specifications
# =============================================================================

NQ_TICK_SIZE = 0.25  # Minimum price movement in points
NQ_TICK_VALUE = 5.00  # Dollar value per tick per contract ($)
NQ_POINT_VALUE = 20.00  # Dollar value per point

NQ_COMMISSION_PER_SIDE = 0.59
NQ_EXCHANGE_FEE_PER_SIDE = 0.47
NQ_TOTAL_PER_SIDE = NQ_COMMISSION_PER_SIDE + NQ_EXCHANGE_FEE_PER_SIDE
NQ_ROUND_TRIP_COST = NQ_TOTAL_PER_SIDE * 2


# =============================================================================
# Contract Specification Dataclass
# =============================================================================

@dataclass(frozen=True)
class ContractSpec:
    """
    Immutable contract specification.

    Contains all relevant information about a futures contract including
    pricing, costs, and margin requirements.

    Attributes:
        symbol: Contract symbol (e.g., "MES", "ES")
        name: Full contract name
        tick_size: Minimum price increment in points
        tick_value: Dollar value per tick
        point_value: Dollar value per point
        commission_per_side: Commission per side in dollars
        exchange_fee_per_side: Exchange fee per side in dollars
        margin_intraday: Intraday margin requirement per contract
        margin_overnight: Overnight margin requirement per contract
        multiplier: Contract multiplier (optional, for reference)
    """
    symbol: str
    name: str
    tick_size: float
    tick_value: float
    point_value: float
    commission_per_side: float
    exchange_fee_per_side: float
    margin_intraday: float
    margin_overnight: float
    multiplier: Optional[float] = None

    @property
    def total_per_side(self) -> float:
        """Total cost per side (commission + exchange fee)."""
        return self.commission_per_side + self.exchange_fee_per_side

    @property
    def round_trip_cost(self) -> float:
        """Total round-trip cost (entry + exit)."""
        return self.total_per_side * 2

    @property
    def ticks_per_point(self) -> int:
        """Number of ticks in one point."""
        return int(self.point_value / self.tick_value)

    def price_to_ticks(self, price_move: float) -> float:
        """Convert a price movement to ticks."""
        return price_move / self.tick_size

    def ticks_to_price(self, ticks: float) -> float:
        """Convert ticks to price movement."""
        return ticks * self.tick_size

    def ticks_to_dollars(self, ticks: float, contracts: int = 1) -> float:
        """Convert ticks to dollar value."""
        return ticks * self.tick_value * contracts

    def dollars_to_ticks(self, dollars: float, contracts: int = 1) -> float:
        """Convert dollar value to ticks."""
        if contracts == 0:
            return 0.0
        return dollars / (self.tick_value * contracts)

    def breakeven_ticks(self, contracts: int = 1) -> float:
        """Calculate ticks needed to break even after costs."""
        total_cost = self.round_trip_cost * contracts
        return self.dollars_to_ticks(total_cost, contracts)


# Pre-defined contract specifications
MES_SPEC = ContractSpec(
    symbol="MES",
    name="Micro E-mini S&P 500",
    tick_size=MES_TICK_SIZE,
    tick_value=MES_TICK_VALUE,
    point_value=MES_POINT_VALUE,
    commission_per_side=MES_COMMISSION_PER_SIDE,
    exchange_fee_per_side=MES_EXCHANGE_FEE_PER_SIDE,
    margin_intraday=MES_MARGIN_INTRADAY,
    margin_overnight=MES_MARGIN_OVERNIGHT,
    multiplier=5.0,
)

ES_SPEC = ContractSpec(
    symbol="ES",
    name="E-mini S&P 500",
    tick_size=ES_TICK_SIZE,
    tick_value=ES_TICK_VALUE,
    point_value=ES_POINT_VALUE,
    commission_per_side=ES_COMMISSION_PER_SIDE,
    exchange_fee_per_side=ES_EXCHANGE_FEE_PER_SIDE,
    margin_intraday=500.0,
    margin_overnight=15000.0,
    multiplier=50.0,
)

MNQ_SPEC = ContractSpec(
    symbol="MNQ",
    name="Micro E-mini Nasdaq",
    tick_size=MNQ_TICK_SIZE,
    tick_value=MNQ_TICK_VALUE,
    point_value=MNQ_POINT_VALUE,
    commission_per_side=MNQ_COMMISSION_PER_SIDE,
    exchange_fee_per_side=MNQ_EXCHANGE_FEE_PER_SIDE,
    margin_intraday=60.0,
    margin_overnight=1800.0,
    multiplier=2.0,
)

NQ_SPEC = ContractSpec(
    symbol="NQ",
    name="E-mini Nasdaq",
    tick_size=NQ_TICK_SIZE,
    tick_value=NQ_TICK_VALUE,
    point_value=NQ_POINT_VALUE,
    commission_per_side=NQ_COMMISSION_PER_SIDE,
    exchange_fee_per_side=NQ_EXCHANGE_FEE_PER_SIDE,
    margin_intraday=600.0,
    margin_overnight=18000.0,
    multiplier=20.0,
)

# Contract lookup by symbol
CONTRACT_SPECS = {
    "MES": MES_SPEC,
    "ES": ES_SPEC,
    "MNQ": MNQ_SPEC,
    "NQ": NQ_SPEC,
}


def get_contract_spec(symbol: str) -> ContractSpec:
    """
    Get contract specification by symbol.

    Args:
        symbol: Contract symbol (e.g., "MES", "ES")

    Returns:
        ContractSpec for the given symbol

    Raises:
        ValueError: If symbol is not found
    """
    symbol = symbol.upper()
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Unknown contract symbol: {symbol}. "
                        f"Available: {list(CONTRACT_SPECS.keys())}")
    return CONTRACT_SPECS[symbol]


# =============================================================================
# Trading Session Times (NY Time)
# =============================================================================

# Regular Trading Hours (RTH) - Main session
RTH_START = time(9, 30)  # 9:30 AM NY
RTH_END = time(16, 0)    # 4:00 PM NY

# Extended Trading Hours (ETH) - Globex session
ETH_START = time(18, 0)  # 6:00 PM NY (previous day)
ETH_END = time(17, 0)    # 5:00 PM NY (next day, 15-min break)

# End of Day (EOD) Management
EOD_REDUCED_SIZE_TIME = time(16, 0)   # 4:00 PM - reduce position sizing 50%
EOD_CLOSE_ONLY_TIME = time(16, 15)    # 4:15 PM - no new positions
EOD_FLATTEN_START_TIME = time(16, 25) # 4:25 PM - begin aggressive exits
EOD_FLATTEN_TIME = time(16, 30)       # 4:30 PM - MUST be flat (HARD REQUIREMENT)

# Session durations
RTH_DURATION_MINUTES = 390  # 6.5 hours (9:30 AM - 4:00 PM)
ETH_DURATION_MINUTES = 1380  # 23 hours (6:00 PM - 5:00 PM next day)


# =============================================================================
# Risk Parameters (from risk-management.md)
# =============================================================================

# Account protection (for $1,000 starting capital)
DEFAULT_STARTING_CAPITAL = 1000.0
DEFAULT_MAX_DAILY_LOSS = 50.0  # 5% of capital
DEFAULT_MAX_DAILY_DRAWDOWN = 75.0  # 7.5% of capital
DEFAULT_MAX_PER_TRADE_RISK = 25.0  # 2.5% of capital
DEFAULT_MAX_CONSECUTIVE_LOSSES = 5
DEFAULT_KILL_SWITCH_THRESHOLD = 300.0  # 30% cumulative loss
DEFAULT_MIN_ACCOUNT_BALANCE = 700.0  # Minimum to trade

# Position sizing
DEFAULT_RISK_PER_TRADE_PCT = 0.025  # 2.5%
DEFAULT_MIN_CONFIDENCE = 0.60  # Minimum model confidence to trade

# Circuit breaker thresholds
CIRCUIT_BREAKER_3_LOSSES_PAUSE_MINUTES = 15
CIRCUIT_BREAKER_5_LOSSES_PAUSE_MINUTES = 30
HIGH_VOLATILITY_ATR_MULTIPLIER = 3.0  # ATR > 3x normal
WIDE_SPREAD_TICKS = 2  # Spread > 2 ticks
LOW_VOLUME_THRESHOLD_PCT = 0.10  # Volume < 10% of average

# Slippage assumptions
DEFAULT_SLIPPAGE_TICKS_NORMAL = 1.0  # 1 tick normal conditions
DEFAULT_SLIPPAGE_TICKS_LOW_LIQUIDITY = 2.0  # 2 ticks low liquidity
DEFAULT_SLIPPAGE_TICKS_HIGH_VOLATILITY = 2.0  # 2-4 ticks high volatility


# =============================================================================
# Feature Engineering Periods (for scalping - SECONDS not days)
# =============================================================================

# Returns lookback periods in seconds
SCALPING_RETURN_PERIODS = [1, 5, 10, 30, 60]

# EMA periods for 1-second data
SCALPING_EMA_PERIODS = [9, 21, 50, 200]

# Multi-timeframe aggregation
MULTI_TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
}

# RSI and other indicator periods
DEFAULT_RSI_PERIOD = 14
DEFAULT_ATR_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9


# =============================================================================
# Model Parameters
# =============================================================================

# Target variable
DEFAULT_LOOKAHEAD_SECONDS = 30  # Predict 30 seconds ahead
DEFAULT_THRESHOLD_TICKS = 3.0  # 3 ticks for UP/DOWN classification

# Inference requirements
MAX_INFERENCE_LATENCY_MS = 10.0  # < 10ms required
MAX_FEATURE_CALC_LATENCY_MS = 5.0  # < 5ms for features
MAX_TOTAL_LATENCY_MS = 15.0  # < 15ms end-to-end

# Training defaults
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_GRAD_CLIP = 1.0


# =============================================================================
# Walk-Forward Validation
# =============================================================================

DEFAULT_TRAINING_MONTHS = 6
DEFAULT_VALIDATION_MONTHS = 1
DEFAULT_TEST_MONTHS = 1
DEFAULT_STEP_MONTHS = 1
DEFAULT_MIN_TRADES_PER_FOLD = 100


# =============================================================================
# API Configuration
# =============================================================================

TOPSTEPX_BASE_URL = "https://api.topstepx.com"
TOPSTEPX_WS_MARKET_URL = "wss://rtc.topstepx.com/hubs/market"
TOPSTEPX_WS_TRADE_URL = "wss://rtc.topstepx.com/hubs/trade"

# Rate limiting
TOPSTEPX_RATE_LIMIT_REQUESTS = 50
TOPSTEPX_RATE_LIMIT_WINDOW_SECONDS = 30

# Token management
TOPSTEPX_TOKEN_EXPIRY_MINUTES = 90
TOPSTEPX_TOKEN_REFRESH_BEFORE_MINUTES = 10

# WebSocket
TOPSTEPX_MAX_WS_SESSIONS = 2
TOPSTEPX_WS_HEARTBEAT_SECONDS = 15
