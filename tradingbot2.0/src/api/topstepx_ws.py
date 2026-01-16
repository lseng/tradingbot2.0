"""
TopstepX WebSocket Connections for Real-Time Data.

This module provides WebSocket clients for real-time market data and trade updates
via the TopstepX SignalR hubs.

Key Components:
- Market Hub: Real-time quotes (bid, ask, last, volume)
- Trade Hub: Order fills, position updates, account updates

WebSocket URLs:
- Market Hub: wss://rtc.topstepx.com/hubs/market
- Trade Hub: wss://rtc.topstepx.com/hubs/trade

Limitations:
- Maximum 2 concurrent WebSocket sessions
- SignalR protocol required

API Reference: https://gateway.docs.projectx.com/docs/intro
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional

import aiohttp

from src.api.topstepx_client import TopstepXClient, TopstepXConnectionError

logger = logging.getLogger(__name__)


class WebSocketState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    CLOSED = auto()


@dataclass
class Quote:
    """Real-time quote data.

    Attributes:
        contract_id: Contract identifier
        bid: Best bid price
        ask: Best ask price
        last: Last trade price
        bid_size: Bid size
        ask_size: Ask size
        volume: Total volume
        timestamp: Quote timestamp (local reception time)
        server_timestamp: Server-side timestamp from API (if available)
        reception_latency_ms: Time from server to local reception (if server_timestamp available)
    """
    contract_id: str
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    timestamp: Optional[datetime] = None
    server_timestamp: Optional[datetime] = None
    reception_latency_ms: Optional[float] = None

    @classmethod
    def from_signalr(cls, data: dict) -> "Quote":
        """Create Quote from SignalR message."""
        reception_time = datetime.utcnow()

        # Parse server timestamp if available
        server_timestamp = None
        latency_ms = None

        # Try different timestamp field names
        ts_value = data.get("timestamp", data.get("ts", data.get("time")))
        if ts_value is not None:
            try:
                if isinstance(ts_value, (int, float)):
                    # Unix timestamp (seconds or milliseconds)
                    if ts_value > 1e12:  # Milliseconds
                        server_timestamp = datetime.utcfromtimestamp(ts_value / 1000)
                    else:  # Seconds
                        server_timestamp = datetime.utcfromtimestamp(ts_value)
                elif isinstance(ts_value, str):
                    # ISO format string
                    server_timestamp = datetime.fromisoformat(ts_value.replace('Z', '+00:00').replace('+00:00', ''))
            except (ValueError, TypeError, OSError):
                pass

        # Calculate latency if we have server timestamp
        if server_timestamp:
            latency_ms = (reception_time - server_timestamp).total_seconds() * 1000
            # Clamp negative values (clock skew)
            if latency_ms < 0:
                latency_ms = 0.0

        return cls(
            contract_id=str(data.get("contractId", data.get("symbol", ""))),
            bid=float(data.get("bid", data.get("bidPrice", 0))),
            ask=float(data.get("ask", data.get("askPrice", 0))),
            last=float(data.get("last", data.get("lastPrice", data.get("price", 0)))),
            bid_size=int(data.get("bidSize", data.get("bidQty", 0))),
            ask_size=int(data.get("askSize", data.get("askQty", 0))),
            volume=int(data.get("volume", data.get("totalVolume", 0))),
            timestamp=reception_time,
            server_timestamp=server_timestamp,
            reception_latency_ms=latency_ms,
        )

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class OrderFill:
    """Order fill notification.

    Attributes:
        order_id: Order ID
        contract_id: Contract identifier
        side: 1 (buy) or 2 (sell)
        fill_price: Fill price
        fill_size: Filled quantity
        remaining_size: Remaining quantity
        is_complete: Whether order is completely filled
        timestamp: Fill timestamp
    """
    order_id: str
    contract_id: str
    side: int
    fill_price: float
    fill_size: int
    remaining_size: int = 0
    is_complete: bool = False
    timestamp: Optional[datetime] = None

    @classmethod
    def from_signalr(cls, data: dict) -> "OrderFill":
        """Create OrderFill from SignalR message."""
        return cls(
            order_id=str(data.get("orderId", "")),
            contract_id=str(data.get("contractId", "")),
            side=int(data.get("side", 0)),
            fill_price=float(data.get("fillPrice", data.get("price", 0))),
            fill_size=int(data.get("fillSize", data.get("qty", 0))),
            remaining_size=int(data.get("remainingSize", 0)),
            is_complete=bool(data.get("isComplete", data.get("filled", False))),
            timestamp=datetime.utcnow(),
        )


@dataclass
class PositionUpdate:
    """Position update notification.

    Attributes:
        account_id: Account ID
        contract_id: Contract identifier
        size: Position size (positive = long, negative = short)
        avg_price: Average entry price
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L
        timestamp: Update timestamp
    """
    account_id: int
    contract_id: str
    size: int
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None

    @classmethod
    def from_signalr(cls, data: dict) -> "PositionUpdate":
        """Create PositionUpdate from SignalR message."""
        return cls(
            account_id=int(data.get("accountId", 0)),
            contract_id=str(data.get("contractId", "")),
            size=int(data.get("size", data.get("qty", 0))),
            avg_price=float(data.get("avgPrice", data.get("price", 0))),
            unrealized_pnl=float(data.get("unrealizedPnl", 0)),
            realized_pnl=float(data.get("realizedPnl", 0)),
            timestamp=datetime.utcnow(),
        )

    @property
    def direction(self) -> int:
        """Get position direction: 1 (long), -1 (short), 0 (flat)."""
        if self.size > 0:
            return 1
        elif self.size < 0:
            return -1
        return 0


@dataclass
class AccountUpdate:
    """Account update notification.

    Attributes:
        account_id: Account ID
        balance: Current balance
        available_margin: Available margin
        open_pnl: Open P&L
        closed_pnl: Closed P&L
        timestamp: Update timestamp
    """
    account_id: int
    balance: float
    available_margin: float = 0.0
    open_pnl: float = 0.0
    closed_pnl: float = 0.0
    timestamp: Optional[datetime] = None

    @classmethod
    def from_signalr(cls, data: dict) -> "AccountUpdate":
        """Create AccountUpdate from SignalR message."""
        return cls(
            account_id=int(data.get("accountId", data.get("id", 0))),
            balance=float(data.get("balance", data.get("accountBalance", 0))),
            available_margin=float(data.get("availableMargin", 0)),
            open_pnl=float(data.get("openPnl", data.get("unrealizedPnl", 0))),
            closed_pnl=float(data.get("closedPnl", data.get("realizedPnl", 0))),
            timestamp=datetime.utcnow(),
        )


# Type aliases for callbacks
QuoteCallback = Callable[[Quote], None]
OrderFillCallback = Callable[[OrderFill], None]
PositionCallback = Callable[[PositionUpdate], None]
AccountCallback = Callable[[AccountUpdate], None]


class SignalRConnection:
    """Low-level SignalR WebSocket connection.

    Handles the SignalR protocol over WebSocket including:
    - Connection negotiation
    - Message framing (JSON with record separator)
    - Handshake protocol
    - Ping/pong heartbeats
    - Invocation tracking
    """

    # SignalR record separator
    RECORD_SEPARATOR = "\x1e"

    def __init__(
        self,
        url: str,
        access_token: str,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """Initialize SignalR connection.

        Args:
            url: WebSocket URL
            access_token: JWT access token for authentication
            session: Optional aiohttp session to use
        """
        self._url = url
        self._access_token = access_token
        self._session = session
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._state = WebSocketState.DISCONNECTED
        self._invocation_id = 0
        self._pending_invocations: dict[str, asyncio.Future] = {}
        self._message_handlers: dict[str, list[Callable]] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> WebSocketState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == WebSocketState.CONNECTED

    def on(self, method: str, handler: Callable) -> None:
        """Register handler for server-invoked method.

        Args:
            method: Method name (e.g., "GotQuote")
            handler: Callback function
        """
        if method not in self._message_handlers:
            self._message_handlers[method] = []
        self._message_handlers[method].append(handler)

    def off(self, method: str, handler: Optional[Callable] = None) -> None:
        """Unregister handler for server-invoked method.

        Args:
            method: Method name
            handler: Specific handler to remove, or None to remove all
        """
        if method in self._message_handlers:
            if handler is None:
                del self._message_handlers[method]
            else:
                self._message_handlers[method] = [
                    h for h in self._message_handlers[method] if h != handler
                ]

    async def connect(self) -> None:
        """Establish WebSocket connection and perform handshake."""
        if self._state in (WebSocketState.CONNECTED, WebSocketState.CONNECTING):
            return

        self._state = WebSocketState.CONNECTING

        try:
            # Create session if not provided
            if self._session is None:
                self._session = aiohttp.ClientSession()

            # Negotiate connection (SignalR requirement)
            negotiate_url = self._url.replace("wss://", "https://").replace("ws://", "http://")
            negotiate_url = f"{negotiate_url}/negotiate?negotiateVersion=1"

            headers = {"Authorization": f"Bearer {self._access_token}"}

            async with self._session.post(negotiate_url, headers=headers) as resp:
                if resp.status != 200:
                    raise TopstepXConnectionError(f"Negotiate failed: {resp.status}")
                negotiate_data = await resp.json()
                connection_token = negotiate_data.get("connectionToken", "")

            # Connect to WebSocket with connection token
            ws_url = f"{self._url}?id={connection_token}"

            self._ws = await self._session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=30,
            )

            # Send SignalR handshake
            handshake = {"protocol": "json", "version": 1}
            await self._send_message(handshake)

            # Wait for handshake response
            response = await self._receive_raw()
            if response is None or response.get("error"):
                error = response.get("error", "Unknown error") if response else "No response"
                raise TopstepXConnectionError(f"Handshake failed: {error}")

            self._state = WebSocketState.CONNECTED
            logger.info(f"Connected to {self._url}")

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Start ping task
            self._ping_task = asyncio.create_task(self._ping_loop())

        except Exception as e:
            self._state = WebSocketState.DISCONNECTED
            logger.error(f"Connection failed: {e}")
            raise TopstepXConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._state = WebSocketState.CLOSED

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        # Cancel pending invocations
        for future in self._pending_invocations.values():
            if not future.done():
                future.cancel()
        self._pending_invocations.clear()

        logger.info(f"Disconnected from {self._url}")

    async def invoke(self, method: str, *args: Any, timeout: float = 30.0) -> Any:
        """Invoke a server method and wait for response.

        Args:
            method: Method name
            *args: Arguments to pass
            timeout: Response timeout in seconds

        Returns:
            Response result

        Raises:
            TopstepXConnectionError: On connection or invocation error
            asyncio.TimeoutError: If response not received in time
        """
        if not self.is_connected:
            raise TopstepXConnectionError("Not connected")

        self._invocation_id += 1
        invocation_id = str(self._invocation_id)

        message = {
            "type": 1,  # Invocation
            "invocationId": invocation_id,
            "target": method,
            "arguments": list(args),
        }

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_invocations[invocation_id] = future

        try:
            await self._send_message(message)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Invoke timeout for {method}")
            raise
        finally:
            self._pending_invocations.pop(invocation_id, None)

    async def send(self, method: str, *args: Any) -> None:
        """Send a message without waiting for response.

        Args:
            method: Method name
            *args: Arguments to pass
        """
        if not self.is_connected:
            raise TopstepXConnectionError("Not connected")

        message = {
            "type": 1,  # Invocation (no response expected)
            "target": method,
            "arguments": list(args),
        }

        await self._send_message(message)

    async def _send_message(self, message: dict) -> None:
        """Send a SignalR message."""
        if self._ws is None or self._ws.closed:
            raise TopstepXConnectionError("WebSocket not connected")

        text = json.dumps(message) + self.RECORD_SEPARATOR
        await self._ws.send_str(text)

    async def _receive_raw(self) -> Optional[dict]:
        """Receive a single SignalR message."""
        if self._ws is None:
            return None

        try:
            msg = await self._ws.receive(timeout=30)

            if msg.type == aiohttp.WSMsgType.TEXT:
                # SignalR uses record separator for message framing
                text = msg.data.rstrip(self.RECORD_SEPARATOR)
                if text:
                    return json.loads(text)
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.warning("WebSocket closed by server")
                self._state = WebSocketState.DISCONNECTED
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {msg.data}")
                self._state = WebSocketState.DISCONNECTED

        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")

        return None

    async def _receive_loop(self) -> None:
        """Background task to receive and dispatch messages."""
        while self.is_connected:
            try:
                message = await self._receive_raw()
                if message:
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                if self._state == WebSocketState.CONNECTED:
                    self._state = WebSocketState.DISCONNECTED
                break

    async def _handle_message(self, message: dict) -> None:
        """Handle received SignalR message."""
        msg_type = message.get("type", 0)

        if msg_type == 1:
            # Invocation from server
            target = message.get("target", "")
            arguments = message.get("arguments", [])

            if target in self._message_handlers:
                for handler in self._message_handlers[target]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(*arguments)
                        else:
                            handler(*arguments)
                    except Exception as e:
                        logger.error(f"Handler error for {target}: {e}")

        elif msg_type == 3:
            # Completion (response to invocation)
            invocation_id = message.get("invocationId", "")
            if invocation_id in self._pending_invocations:
                future = self._pending_invocations[invocation_id]
                if not future.done():
                    if "error" in message:
                        future.set_exception(
                            TopstepXConnectionError(message["error"])
                        )
                    else:
                        future.set_result(message.get("result"))

        elif msg_type == 6:
            # Ping
            await self._send_message({"type": 6})  # Pong

        elif msg_type == 7:
            # Close
            logger.info("Server requested close")
            self._state = WebSocketState.DISCONNECTED

    async def _ping_loop(self) -> None:
        """Background task to send periodic pings."""
        while self.is_connected:
            try:
                await asyncio.sleep(15)
                if self.is_connected:
                    await self._send_message({"type": 6})  # Ping
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Ping error: {e}")


class TopstepXWebSocket:
    """High-level WebSocket client for TopstepX real-time data.

    Provides subscription management for market quotes and trade updates
    with automatic reconnection.

    Example:
        client = TopstepXClient(username="user", password="pass")
        await client.authenticate()

        ws = TopstepXWebSocket(client)

        # Register callbacks
        ws.on_quote(lambda q: print(f"Quote: {q.contract_id} @ {q.last}"))
        ws.on_fill(lambda f: print(f"Fill: {f.order_id} @ {f.fill_price}"))

        # Connect and subscribe
        await ws.connect()
        await ws.subscribe_quotes(["CON.F.US.MES.H26"])

        # Keep running
        await asyncio.sleep(3600)

        # Cleanup
        await ws.disconnect()
    """

    def __init__(
        self,
        client: TopstepXClient,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        """Initialize WebSocket client.

        Args:
            client: Authenticated TopstepXClient instance
            auto_reconnect: Whether to auto-reconnect on disconnect
            reconnect_delay: Initial reconnect delay in seconds
            max_reconnect_attempts: Max reconnection attempts (0 = infinite)
        """
        self._client = client
        self._auto_reconnect = auto_reconnect
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts

        self._market_connection: Optional[SignalRConnection] = None
        self._trade_connection: Optional[SignalRConnection] = None

        self._subscribed_contracts: set[str] = set()

        # Callbacks
        self._quote_callbacks: list[QuoteCallback] = []
        self._fill_callbacks: list[OrderFillCallback] = []
        self._position_callbacks: list[PositionCallback] = []
        self._account_callbacks: list[AccountCallback] = []

        # Reconnection state
        self._reconnect_attempts = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_run = False

    @property
    def is_connected(self) -> bool:
        """Check if market WebSocket is connected."""
        return (
            self._market_connection is not None
            and self._market_connection.is_connected
        )

    @property
    def market_state(self) -> WebSocketState:
        """Get market connection state."""
        if self._market_connection:
            return self._market_connection.state
        return WebSocketState.DISCONNECTED

    @property
    def trade_state(self) -> WebSocketState:
        """Get trade connection state."""
        if self._trade_connection:
            return self._trade_connection.state
        return WebSocketState.DISCONNECTED

    def on_quote(self, callback: QuoteCallback) -> None:
        """Register quote callback.

        Args:
            callback: Function called with Quote on each quote update
        """
        self._quote_callbacks.append(callback)

    def on_fill(self, callback: OrderFillCallback) -> None:
        """Register order fill callback.

        Args:
            callback: Function called with OrderFill on each fill
        """
        self._fill_callbacks.append(callback)

    def on_position(self, callback: PositionCallback) -> None:
        """Register position update callback.

        Args:
            callback: Function called with PositionUpdate on changes
        """
        self._position_callbacks.append(callback)

    def on_account(self, callback: AccountCallback) -> None:
        """Register account update callback.

        Args:
            callback: Function called with AccountUpdate on changes
        """
        self._account_callbacks.append(callback)

    async def connect(self) -> None:
        """Connect to WebSocket hubs.

        Establishes connections to both market and trade hubs.
        """
        if not self._client.access_token:
            raise TopstepXConnectionError("Client not authenticated")

        self._should_run = True

        # Connect to market hub
        await self._connect_market()

        # Connect to trade hub
        await self._connect_trade()

    async def _connect_market(self) -> None:
        """Connect to market hub."""
        if self._market_connection and self._market_connection.is_connected:
            return

        self._market_connection = SignalRConnection(
            url=self._client.config.ws_market_url,
            access_token=self._client.access_token,
        )

        # Register quote handler
        self._market_connection.on("GotQuote", self._handle_quote)
        self._market_connection.on("QuoteUpdate", self._handle_quote)

        await self._market_connection.connect()

        # Resubscribe to contracts
        if self._subscribed_contracts:
            await self._market_connection.invoke(
                "SubscribeQuotes",
                list(self._subscribed_contracts)
            )

        self._reconnect_attempts = 0

    async def _connect_trade(self) -> None:
        """Connect to trade hub."""
        if self._trade_connection and self._trade_connection.is_connected:
            return

        self._trade_connection = SignalRConnection(
            url=self._client.config.ws_trade_url,
            access_token=self._client.access_token,
        )

        # Register trade handlers
        self._trade_connection.on("OrderFilled", self._handle_fill)
        self._trade_connection.on("GotFill", self._handle_fill)
        self._trade_connection.on("PositionUpdate", self._handle_position)
        self._trade_connection.on("AccountUpdate", self._handle_account)

        await self._trade_connection.connect()

        # Subscribe to account updates
        if self._client.default_account_id:
            await self._trade_connection.invoke(
                "SubscribeAccount",
                self._client.default_account_id
            )

    async def disconnect(self) -> None:
        """Disconnect from WebSocket hubs."""
        self._should_run = False

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        if self._market_connection:
            await self._market_connection.disconnect()
            self._market_connection = None

        if self._trade_connection:
            await self._trade_connection.disconnect()
            self._trade_connection = None

        self._subscribed_contracts.clear()

    async def subscribe_quotes(self, contract_ids: list[str]) -> None:
        """Subscribe to quote updates for contracts.

        Args:
            contract_ids: List of contract IDs to subscribe to
        """
        if not self._market_connection or not self._market_connection.is_connected:
            raise TopstepXConnectionError("Market hub not connected")

        self._subscribed_contracts.update(contract_ids)

        await self._market_connection.invoke("SubscribeQuotes", contract_ids)
        logger.info(f"Subscribed to quotes: {contract_ids}")

    async def unsubscribe_quotes(self, contract_ids: list[str]) -> None:
        """Unsubscribe from quote updates.

        Args:
            contract_ids: List of contract IDs to unsubscribe from
        """
        if not self._market_connection or not self._market_connection.is_connected:
            return

        self._subscribed_contracts.difference_update(contract_ids)

        await self._market_connection.invoke("UnsubscribeQuotes", contract_ids)
        logger.info(f"Unsubscribed from quotes: {contract_ids}")

    def _handle_quote(self, data: Any) -> None:
        """Handle incoming quote update."""
        try:
            if isinstance(data, dict):
                quote = Quote.from_signalr(data)
            elif isinstance(data, list) and len(data) > 0:
                quote = Quote.from_signalr(data[0])
            else:
                return

            for callback in self._quote_callbacks:
                try:
                    callback(quote)
                except Exception as e:
                    logger.error(f"Quote callback error: {e}")

        except Exception as e:
            logger.error(f"Quote parse error: {e}")

    def _handle_fill(self, data: Any) -> None:
        """Handle incoming order fill."""
        try:
            if isinstance(data, dict):
                fill = OrderFill.from_signalr(data)
            elif isinstance(data, list) and len(data) > 0:
                fill = OrderFill.from_signalr(data[0])
            else:
                return

            logger.info(f"Order fill: {fill.order_id} @ {fill.fill_price}")

            for callback in self._fill_callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}")

        except Exception as e:
            logger.error(f"Fill parse error: {e}")

    def _handle_position(self, data: Any) -> None:
        """Handle incoming position update."""
        try:
            if isinstance(data, dict):
                position = PositionUpdate.from_signalr(data)
            elif isinstance(data, list) and len(data) > 0:
                position = PositionUpdate.from_signalr(data[0])
            else:
                return

            logger.debug(f"Position update: {position.contract_id} size={position.size}")

            for callback in self._position_callbacks:
                try:
                    callback(position)
                except Exception as e:
                    logger.error(f"Position callback error: {e}")

        except Exception as e:
            logger.error(f"Position parse error: {e}")

    def _handle_account(self, data: Any) -> None:
        """Handle incoming account update."""
        try:
            if isinstance(data, dict):
                account = AccountUpdate.from_signalr(data)
            elif isinstance(data, list) and len(data) > 0:
                account = AccountUpdate.from_signalr(data[0])
            else:
                return

            logger.debug(f"Account update: balance={account.balance}")

            for callback in self._account_callbacks:
                try:
                    callback(account)
                except Exception as e:
                    logger.error(f"Account callback error: {e}")

        except Exception as e:
            logger.error(f"Account parse error: {e}")

    async def _auto_reconnect_loop(self) -> None:
        """Background task for auto-reconnection."""
        while self._should_run and self._auto_reconnect:
            if not self.is_connected:
                if self._max_reconnect_attempts == 0 or self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
                    delay = min(delay, 60)  # Cap at 60 seconds

                    logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})")
                    await asyncio.sleep(delay)

                    try:
                        await self.connect()
                    except Exception as e:
                        logger.error(f"Reconnection failed: {e}")
                else:
                    logger.error("Max reconnection attempts reached")
                    break

            await asyncio.sleep(1)

    async def __aenter__(self) -> "TopstepXWebSocket":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
