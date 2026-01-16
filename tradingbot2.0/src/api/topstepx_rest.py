"""
TopstepX REST API Endpoints.

This module provides REST API methods for interacting with TopstepX including:
- Historical data retrieval
- Order placement and management
- Position tracking
- Account information

Important Limitations:
- Historical data limited to ~7-14 days for second bars, ~30 days for minute+
- Use DataBento for training data, TopstepX for live trading only
- No bracket orders (must place entry, stop, target separately)
- Position netting per contract

API Reference: https://gateway.docs.projectx.com/docs/intro
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from src.api.topstepx_client import TopstepXClient, TopstepXAPIError

logger = logging.getLogger(__name__)


class OrderType(IntEnum):
    """Order types supported by TopstepX."""
    LIMIT = 1
    MARKET = 2
    STOP = 3
    STOP_LIMIT = 4


class OrderSide(IntEnum):
    """Order sides."""
    BUY = 1
    SELL = 2


class OrderStatus(IntEnum):
    """Order status codes."""
    PENDING = 0
    WORKING = 1
    FILLED = 2
    CANCELLED = 3
    REJECTED = 4
    PARTIALLY_FILLED = 5


class TimeUnit(IntEnum):
    """Time units for historical bar requests."""
    SECOND = 1
    MINUTE = 2
    HOUR = 3
    DAY = 4
    WEEK = 5
    MONTH = 6


@dataclass
class BarData:
    """Historical bar data structure.

    Attributes:
        timestamp: Bar timestamp (UTC)
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Volume traded
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_api(cls, data: dict) -> "BarData":
        """Create BarData from API response."""
        timestamp = data.get("timestamp", "")
        if isinstance(timestamp, str):
            # Parse ISO format timestamp
            timestamp = timestamp.rstrip("Z")
            if "." in timestamp:
                timestamp = datetime.fromisoformat(timestamp)
            else:
                timestamp = datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, (int, float)):
            # Unix timestamp
            timestamp = datetime.utcfromtimestamp(timestamp / 1000)

        return cls(
            timestamp=timestamp,
            open=float(data.get("open", 0)),
            high=float(data.get("high", 0)),
            low=float(data.get("low", 0)),
            close=float(data.get("close", 0)),
            volume=int(data.get("volume", 0)),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class OrderResponse:
    """Order response from API.

    Attributes:
        order_id: Unique order identifier
        account_id: Account ID
        contract_id: Contract identifier
        side: Buy or Sell
        type: Order type
        size: Number of contracts
        price: Limit price (for limit/stop-limit orders)
        stop_price: Stop price (for stop/stop-limit orders)
        status: Order status
        filled_size: Number of contracts filled
        avg_fill_price: Average fill price
        created_at: Order creation timestamp
        custom_tag: User-defined order tag
        error_message: Error message if order rejected
    """
    order_id: str
    account_id: int
    contract_id: str
    side: OrderSide
    type: OrderType
    size: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: int = 0
    avg_fill_price: Optional[float] = None
    created_at: Optional[datetime] = None
    custom_tag: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "OrderResponse":
        """Create OrderResponse from API response."""
        created = data.get("createdAt")
        if isinstance(created, str):
            created = datetime.fromisoformat(created.rstrip("Z"))

        return cls(
            order_id=str(data.get("orderId", data.get("id", ""))),
            account_id=int(data.get("accountId", 0)),
            contract_id=str(data.get("contractId", "")),
            side=OrderSide(data.get("side", 1)),
            type=OrderType(data.get("type", 2)),
            size=int(data.get("size", 0)),
            price=data.get("price"),
            stop_price=data.get("stopPrice"),
            status=OrderStatus(data.get("status", 0)),
            filled_size=int(data.get("filledSize", 0)),
            avg_fill_price=data.get("avgFillPrice"),
            created_at=created,
            custom_tag=data.get("customTag"),
            error_message=data.get("errorMessage"),
        )

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_working(self) -> bool:
        """Check if order is still working (open)."""
        return self.status in (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_rejected(self) -> bool:
        """Check if order was rejected."""
        return self.status == OrderStatus.REJECTED


@dataclass
class PositionData:
    """Position data structure.

    Attributes:
        account_id: Account ID
        contract_id: Contract identifier
        size: Position size (positive = long, negative = short)
        avg_price: Average entry price
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss
        timestamp: Last update timestamp
    """
    account_id: int
    contract_id: str
    size: int
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: dict) -> "PositionData":
        """Create PositionData from API response."""
        return cls(
            account_id=int(data.get("accountId", 0)),
            contract_id=str(data.get("contractId", "")),
            size=int(data.get("size", data.get("qty", 0))),
            avg_price=float(data.get("avgPrice", data.get("price", 0))),
            unrealized_pnl=float(data.get("unrealizedPnl", 0)),
            realized_pnl=float(data.get("realizedPnl", 0)),
        )

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.size == 0

    @property
    def direction(self) -> int:
        """Get position direction: 1 (long), -1 (short), 0 (flat)."""
        if self.size > 0:
            return 1
        elif self.size < 0:
            return -1
        return 0


@dataclass
class AccountInfo:
    """Account information.

    Attributes:
        account_id: Account ID
        name: Account name/label
        balance: Current account balance
        available_margin: Available margin for trading
        open_pnl: Open (unrealized) P&L
        closed_pnl: Closed (realized) P&L for the day
        total_pnl: Total P&L (open + closed)
    """
    account_id: int
    name: str
    balance: float
    available_margin: float = 0.0
    open_pnl: float = 0.0
    closed_pnl: float = 0.0
    total_pnl: float = 0.0

    @classmethod
    def from_api(cls, data: dict) -> "AccountInfo":
        """Create AccountInfo from API response."""
        return cls(
            account_id=int(data.get("id", data.get("accountId", 0))),
            name=str(data.get("name", "")),
            balance=float(data.get("balance", data.get("accountBalance", 0))),
            available_margin=float(data.get("availableMargin", data.get("marginAvailable", 0))),
            open_pnl=float(data.get("openPnl", data.get("unrealizedPnl", 0))),
            closed_pnl=float(data.get("closedPnl", data.get("realizedPnl", 0))),
            total_pnl=float(data.get("totalPnl", 0)),
        )


class TopstepXREST:
    """TopstepX REST API client.

    Provides methods for interacting with TopstepX REST endpoints.

    Example:
        client = TopstepXClient(username="user", password="pass")
        await client.authenticate()

        rest = TopstepXREST(client)

        # Get account info
        info = await rest.get_account_info()

        # Get historical bars
        bars = await rest.get_historical_bars(
            contract_id="CON.F.US.MES.H26",
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 15),
            unit=TimeUnit.MINUTE,
            unit_number=1
        )

        # Place market order
        order = await rest.place_order(
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.BUY,
            size=1,
            order_type=OrderType.MARKET
        )
    """

    def __init__(self, client: TopstepXClient):
        """Initialize REST API client.

        Args:
            client: Authenticated TopstepXClient instance
        """
        self._client = client

    async def get_account_info(self, account_id: Optional[int] = None) -> AccountInfo:
        """Get account information.

        Args:
            account_id: Account ID (uses default if not specified)

        Returns:
            AccountInfo object

        Raises:
            TopstepXAPIError: On API error
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        response = await self._client.request("GET", f"/api/Account/{account_id}")
        return AccountInfo.from_api(response)

    async def get_positions(self, account_id: Optional[int] = None) -> list[PositionData]:
        """Get all open positions for account.

        Args:
            account_id: Account ID (uses default if not specified)

        Returns:
            List of PositionData objects

        Raises:
            TopstepXAPIError: On API error
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        response = await self._client.request("GET", f"/api/Position/{account_id}")

        positions = []
        if isinstance(response, list):
            for pos in response:
                positions.append(PositionData.from_api(pos))
        elif isinstance(response, dict):
            if "positions" in response:
                for pos in response["positions"]:
                    positions.append(PositionData.from_api(pos))

        return positions

    async def get_position(
        self,
        contract_id: str,
        account_id: Optional[int] = None
    ) -> Optional[PositionData]:
        """Get position for specific contract.

        Args:
            contract_id: Contract identifier
            account_id: Account ID (uses default if not specified)

        Returns:
            PositionData if position exists, None otherwise
        """
        positions = await self.get_positions(account_id)
        for pos in positions:
            if pos.contract_id == contract_id:
                return pos
        return None

    async def get_historical_bars(
        self,
        contract_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        unit: TimeUnit = TimeUnit.MINUTE,
        unit_number: int = 1,
        limit: int = 20000,
        live: bool = False,
    ) -> list[BarData]:
        """Get historical OHLCV bars.

        Note: TopstepX has limited historical data (~7-14 days for second bars,
        ~30 days for minute+). Use DataBento for training data.

        Args:
            contract_id: Contract identifier (e.g., "CON.F.US.MES.H26")
            start_time: Start time (UTC)
            end_time: End time (UTC), defaults to now
            unit: Time unit (SECOND, MINUTE, HOUR, etc.)
            unit_number: Number of units per bar (e.g., 5 for 5-minute bars)
            limit: Maximum bars to return (max 20,000)
            live: Whether to get live data

        Returns:
            List of BarData objects

        Raises:
            TopstepXAPIError: On API error
        """
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            # Default to 7 days back
            from datetime import timedelta
            start_time = end_time - timedelta(days=7)

        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "unit": int(unit),
            "unitNumber": unit_number,
            "limit": min(limit, 20000),
        }

        response = await self._client.request("POST", "/api/History/retrieveBars", json=payload)

        bars = []
        bar_list = response.get("bars", response) if isinstance(response, dict) else response
        if isinstance(bar_list, list):
            for bar in bar_list:
                bars.append(BarData.from_api(bar))

        logger.debug(f"Retrieved {len(bars)} bars for {contract_id}")
        return bars

    async def place_order(
        self,
        contract_id: str,
        side: OrderSide,
        size: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        account_id: Optional[int] = None,
        custom_tag: Optional[str] = None,
    ) -> OrderResponse:
        """Place a new order.

        Note: TopstepX does not support bracket orders. Entry, stop, and target
        orders must be placed separately.

        Args:
            contract_id: Contract identifier
            side: BUY or SELL
            size: Number of contracts
            order_type: Order type (MARKET, LIMIT, STOP, STOP_LIMIT)
            price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop price (required for STOP and STOP_LIMIT)
            account_id: Account ID (uses default if not specified)
            custom_tag: Optional user-defined tag for the order

        Returns:
            OrderResponse with order details

        Raises:
            TopstepXAPIError: On API error
            ValueError: If required prices not provided for order type
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        # Validate price requirements
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and price is None:
            raise ValueError(f"Limit price required for {order_type.name} orders")
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise ValueError(f"Stop price required for {order_type.name} orders")

        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": int(order_type),
            "side": int(side),
            "size": size,
        }

        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        if custom_tag:
            payload["customTag"] = custom_tag

        logger.info(f"Placing {order_type.name} {side.name} order: {size} {contract_id}")

        response = await self._client.request("POST", "/api/Order/place", json=payload)
        order = OrderResponse.from_api(response)

        if order.is_rejected:
            logger.error(f"Order rejected: {order.error_message}")
            raise TopstepXAPIError(f"Order rejected: {order.error_message}")

        logger.info(f"Order placed: {order.order_id}")
        return order

    async def cancel_order(
        self,
        order_id: str,
        account_id: Optional[int] = None,
    ) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order ID to cancel
            account_id: Account ID (uses default if not specified)

        Returns:
            True if order cancelled successfully

        Raises:
            TopstepXAPIError: On API error
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        payload = {
            "accountId": account_id,
            "orderId": order_id,
        }

        logger.info(f"Cancelling order: {order_id}")

        response = await self._client.request("POST", "/api/Order/cancel", json=payload)

        success = response.get("success", True) if isinstance(response, dict) else True
        if success:
            logger.info(f"Order cancelled: {order_id}")
        else:
            logger.warning(f"Failed to cancel order: {order_id}")

        return success

    async def cancel_all_orders(
        self,
        contract_id: Optional[str] = None,
        account_id: Optional[int] = None,
    ) -> int:
        """Cancel all pending orders.

        Args:
            contract_id: Optional contract to filter by
            account_id: Account ID (uses default if not specified)

        Returns:
            Number of orders cancelled

        Raises:
            TopstepXAPIError: On API error
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        payload = {"accountId": account_id}
        if contract_id:
            payload["contractId"] = contract_id

        logger.info(f"Cancelling all orders for account {account_id}")

        response = await self._client.request("POST", "/api/Order/cancelAll", json=payload)

        count = response.get("cancelledCount", 0) if isinstance(response, dict) else 0
        logger.info(f"Cancelled {count} orders")
        return count

    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        account_id: Optional[int] = None,
    ) -> list[OrderResponse]:
        """Get all orders.

        Args:
            status: Optional status filter
            account_id: Account ID (uses default if not specified)

        Returns:
            List of OrderResponse objects

        Raises:
            TopstepXAPIError: On API error
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        params = {"accountId": account_id}
        if status is not None:
            params["status"] = int(status)

        response = await self._client.request("GET", "/api/Order/list", params=params)

        orders = []
        order_list = response.get("orders", response) if isinstance(response, dict) else response
        if isinstance(order_list, list):
            for order in order_list:
                orders.append(OrderResponse.from_api(order))

        return orders

    async def get_order(
        self,
        order_id: str,
        account_id: Optional[int] = None,
    ) -> OrderResponse:
        """Get order by ID.

        Args:
            order_id: Order ID
            account_id: Account ID (uses default if not specified)

        Returns:
            OrderResponse object

        Raises:
            TopstepXAPIError: On API error
        """
        account_id = account_id or self._client.default_account_id
        if not account_id:
            raise ValueError("No account ID available")

        response = await self._client.request(
            "GET",
            f"/api/Order/{order_id}",
            params={"accountId": account_id}
        )

        return OrderResponse.from_api(response)

    async def flatten_position(
        self,
        contract_id: str,
        account_id: Optional[int] = None,
    ) -> Optional[OrderResponse]:
        """Flatten (close) position for a contract.

        Places a market order to close the entire position.

        Args:
            contract_id: Contract identifier
            account_id: Account ID (uses default if not specified)

        Returns:
            OrderResponse if position was closed, None if already flat

        Raises:
            TopstepXAPIError: On API error
        """
        position = await self.get_position(contract_id, account_id)

        if position is None or position.is_flat:
            logger.info(f"No position to flatten for {contract_id}")
            return None

        # Determine side to close position
        side = OrderSide.SELL if position.is_long else OrderSide.BUY
        size = abs(position.size)

        logger.info(f"Flattening position: {side.name} {size} {contract_id}")

        return await self.place_order(
            contract_id=contract_id,
            side=side,
            size=size,
            order_type=OrderType.MARKET,
            account_id=account_id,
            custom_tag="FLATTEN",
        )


# Contract ID helpers
def get_mes_contract_id(year: int, month: str) -> str:
    """Get MES contract ID.

    Args:
        year: 2-digit year (e.g., 26 for 2026)
        month: Expiry month code (H=March, M=June, U=September, Z=December)

    Returns:
        Contract ID string (e.g., "CON.F.US.MES.H26")
    """
    return f"CON.F.US.MES.{month}{year:02d}"


def get_current_mes_contract() -> str:
    """Get the current front-month MES contract ID.

    Returns:
        Contract ID for current front-month MES
    """
    now = datetime.utcnow()
    year = now.year % 100  # 2-digit year

    # Quarterly expiry months
    expiry_months = [
        (3, "H"),   # March
        (6, "M"),   # June
        (9, "U"),   # September
        (12, "Z"),  # December
    ]

    for exp_month, code in expiry_months:
        if now.month <= exp_month:
            return get_mes_contract_id(year, code)

    # Roll to next year's March contract
    return get_mes_contract_id(year + 1, "H")
