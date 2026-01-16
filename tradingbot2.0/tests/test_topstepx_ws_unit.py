"""
Unit tests for TopstepX WebSocket module.

Tests cover:
- Quote, OrderFill, PositionUpdate, AccountUpdate dataclasses
- WebSocketState enum
- SignalRConnection initialization
- TopstepXWebSocket initialization and properties
- Callback registration
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.api.topstepx_ws import (
    WebSocketState,
    Quote,
    OrderFill,
    PositionUpdate,
    AccountUpdate,
    SignalRConnection,
    TopstepXWebSocket,
)


class TestWebSocketState:
    """Tests for WebSocketState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert hasattr(WebSocketState, 'DISCONNECTED')
        assert hasattr(WebSocketState, 'CONNECTING')
        assert hasattr(WebSocketState, 'CONNECTED')
        assert hasattr(WebSocketState, 'RECONNECTING')
        assert hasattr(WebSocketState, 'CLOSED')

    def test_states_are_unique(self):
        """Test that all states have unique values."""
        states = [
            WebSocketState.DISCONNECTED,
            WebSocketState.CONNECTING,
            WebSocketState.CONNECTED,
            WebSocketState.RECONNECTING,
            WebSocketState.CLOSED,
        ]
        values = [s.value for s in states]
        assert len(values) == len(set(values))


class TestQuote:
    """Tests for Quote dataclass."""

    def test_basic_creation(self):
        """Test basic Quote creation."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=5000.0,
            ask=5000.25,
            last=5000.0,
        )

        assert quote.contract_id == "CON.F.US.MES.H26"
        assert quote.bid == 5000.0
        assert quote.ask == 5000.25
        assert quote.last == 5000.0

    def test_default_values(self):
        """Test default optional values."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=5000.0,
            ask=5000.25,
            last=5000.0,
        )

        assert quote.bid_size == 0
        assert quote.ask_size == 0
        assert quote.volume == 0
        assert quote.timestamp is None

    def test_full_creation(self):
        """Test Quote with all values."""
        ts = datetime.now()
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=5000.0,
            ask=5000.25,
            last=5000.0,
            bid_size=10,
            ask_size=15,
            volume=100,
            timestamp=ts,
        )

        assert quote.bid_size == 10
        assert quote.ask_size == 15
        assert quote.volume == 100
        assert quote.timestamp == ts

    def test_spread_property(self):
        """Test spread calculation."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=5000.0,
            ask=5000.25,
            last=5000.0,
        )

        assert quote.spread == 0.25

    def test_mid_price_property(self):
        """Test mid price calculation."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=5000.0,
            ask=5000.50,
            last=5000.0,
        )

        assert quote.mid_price == 5000.25

    def test_from_signalr_basic(self):
        """Test creating Quote from SignalR message."""
        data = {
            "contractId": "CON.F.US.MES.H26",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
            "bidSize": 10,
            "askSize": 15,
            "volume": 100,
        }

        quote = Quote.from_signalr(data)

        assert quote.contract_id == "CON.F.US.MES.H26"
        assert quote.bid == 5000.0
        assert quote.ask == 5000.25
        assert quote.last == 5000.0
        assert quote.bid_size == 10
        assert quote.ask_size == 15
        assert quote.volume == 100
        assert quote.timestamp is not None

    def test_from_signalr_alternative_keys(self):
        """Test creating Quote with alternative SignalR keys."""
        data = {
            "symbol": "MES",
            "bidPrice": 5000.0,
            "askPrice": 5000.25,
            "lastPrice": 5000.0,
            "bidQty": 10,
            "askQty": 15,
            "totalVolume": 100,
        }

        quote = Quote.from_signalr(data)

        assert quote.contract_id == "MES"
        assert quote.bid == 5000.0
        assert quote.ask == 5000.25
        assert quote.last == 5000.0
        assert quote.bid_size == 10
        assert quote.ask_size == 15
        assert quote.volume == 100

    def test_from_signalr_missing_values(self):
        """Test creating Quote with missing values uses defaults."""
        data = {"contractId": "MES"}

        quote = Quote.from_signalr(data)

        assert quote.contract_id == "MES"
        assert quote.bid == 0.0
        assert quote.ask == 0.0
        assert quote.last == 0.0
        assert quote.bid_size == 0
        assert quote.ask_size == 0
        assert quote.volume == 0


class TestOrderFill:
    """Tests for OrderFill dataclass."""

    def test_basic_creation(self):
        """Test basic OrderFill creation."""
        fill = OrderFill(
            order_id="ORD123",
            contract_id="CON.F.US.MES.H26",
            side=1,
            fill_price=5000.0,
            fill_size=2,
        )

        assert fill.order_id == "ORD123"
        assert fill.contract_id == "CON.F.US.MES.H26"
        assert fill.side == 1
        assert fill.fill_price == 5000.0
        assert fill.fill_size == 2

    def test_default_values(self):
        """Test default optional values."""
        fill = OrderFill(
            order_id="ORD123",
            contract_id="CON.F.US.MES.H26",
            side=1,
            fill_price=5000.0,
            fill_size=2,
        )

        assert fill.remaining_size == 0
        assert fill.is_complete is False
        assert fill.timestamp is None

    def test_full_creation(self):
        """Test OrderFill with all values."""
        ts = datetime.now()
        fill = OrderFill(
            order_id="ORD123",
            contract_id="CON.F.US.MES.H26",
            side=1,
            fill_price=5000.0,
            fill_size=2,
            remaining_size=0,
            is_complete=True,
            timestamp=ts,
        )

        assert fill.is_complete is True
        assert fill.remaining_size == 0
        assert fill.timestamp == ts

    def test_from_signalr_basic(self):
        """Test creating OrderFill from SignalR message."""
        data = {
            "orderId": "ORD123",
            "contractId": "CON.F.US.MES.H26",
            "side": 1,
            "fillPrice": 5000.0,
            "fillSize": 2,
            "remainingSize": 0,
            "isComplete": True,
        }

        fill = OrderFill.from_signalr(data)

        assert fill.order_id == "ORD123"
        assert fill.contract_id == "CON.F.US.MES.H26"
        assert fill.side == 1
        assert fill.fill_price == 5000.0
        assert fill.fill_size == 2
        assert fill.is_complete is True
        assert fill.timestamp is not None

    def test_from_signalr_alternative_keys(self):
        """Test creating OrderFill with alternative SignalR keys."""
        data = {
            "orderId": "ORD123",
            "contractId": "CON.F.US.MES.H26",
            "side": 2,
            "price": 5000.0,
            "qty": 3,
            "filled": True,
        }

        fill = OrderFill.from_signalr(data)

        assert fill.fill_price == 5000.0
        assert fill.fill_size == 3
        assert fill.is_complete is True


class TestPositionUpdate:
    """Tests for PositionUpdate dataclass."""

    def test_basic_creation(self):
        """Test basic PositionUpdate creation."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="CON.F.US.MES.H26",
            size=2,
            avg_price=5000.0,
        )

        assert pos.account_id == 12345
        assert pos.contract_id == "CON.F.US.MES.H26"
        assert pos.size == 2
        assert pos.avg_price == 5000.0

    def test_default_values(self):
        """Test default optional values."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="CON.F.US.MES.H26",
            size=2,
            avg_price=5000.0,
        )

        assert pos.unrealized_pnl == 0.0
        assert pos.realized_pnl == 0.0
        assert pos.timestamp is None

    def test_direction_long(self):
        """Test direction property for long position."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="CON.F.US.MES.H26",
            size=2,
            avg_price=5000.0,
        )

        assert pos.direction == 1

    def test_direction_short(self):
        """Test direction property for short position."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="CON.F.US.MES.H26",
            size=-2,
            avg_price=5000.0,
        )

        assert pos.direction == -1

    def test_direction_flat(self):
        """Test direction property for flat position."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="CON.F.US.MES.H26",
            size=0,
            avg_price=0.0,
        )

        assert pos.direction == 0

    def test_from_signalr_basic(self):
        """Test creating PositionUpdate from SignalR message."""
        data = {
            "accountId": 12345,
            "contractId": "CON.F.US.MES.H26",
            "size": 2,
            "avgPrice": 5000.0,
            "unrealizedPnl": 50.0,
            "realizedPnl": 100.0,
        }

        pos = PositionUpdate.from_signalr(data)

        assert pos.account_id == 12345
        assert pos.contract_id == "CON.F.US.MES.H26"
        assert pos.size == 2
        assert pos.avg_price == 5000.0
        assert pos.unrealized_pnl == 50.0
        assert pos.realized_pnl == 100.0
        assert pos.timestamp is not None

    def test_from_signalr_alternative_keys(self):
        """Test creating PositionUpdate with alternative SignalR keys."""
        data = {
            "accountId": 12345,
            "contractId": "CON.F.US.MES.H26",
            "qty": 3,
            "price": 5000.0,
        }

        pos = PositionUpdate.from_signalr(data)

        assert pos.size == 3
        assert pos.avg_price == 5000.0


class TestAccountUpdate:
    """Tests for AccountUpdate dataclass."""

    def test_basic_creation(self):
        """Test basic AccountUpdate creation."""
        account = AccountUpdate(
            account_id=12345,
            balance=10000.0,
        )

        assert account.account_id == 12345
        assert account.balance == 10000.0

    def test_default_values(self):
        """Test default optional values."""
        account = AccountUpdate(
            account_id=12345,
            balance=10000.0,
        )

        assert account.available_margin == 0.0
        assert account.open_pnl == 0.0
        assert account.closed_pnl == 0.0
        assert account.timestamp is None

    def test_full_creation(self):
        """Test AccountUpdate with all values."""
        ts = datetime.now()
        account = AccountUpdate(
            account_id=12345,
            balance=10000.0,
            available_margin=5000.0,
            open_pnl=100.0,
            closed_pnl=200.0,
            timestamp=ts,
        )

        assert account.available_margin == 5000.0
        assert account.open_pnl == 100.0
        assert account.closed_pnl == 200.0
        assert account.timestamp == ts

    def test_from_signalr_basic(self):
        """Test creating AccountUpdate from SignalR message."""
        data = {
            "accountId": 12345,
            "balance": 10000.0,
            "availableMargin": 5000.0,
            "openPnl": 100.0,
            "closedPnl": 200.0,
        }

        account = AccountUpdate.from_signalr(data)

        assert account.account_id == 12345
        assert account.balance == 10000.0
        assert account.available_margin == 5000.0
        assert account.open_pnl == 100.0
        assert account.closed_pnl == 200.0
        assert account.timestamp is not None

    def test_from_signalr_alternative_keys(self):
        """Test creating AccountUpdate with alternative SignalR keys."""
        data = {
            "id": 12345,
            "accountBalance": 10000.0,
            "unrealizedPnl": 100.0,
            "realizedPnl": 200.0,
        }

        account = AccountUpdate.from_signalr(data)

        assert account.account_id == 12345
        assert account.balance == 10000.0
        assert account.open_pnl == 100.0
        assert account.closed_pnl == 200.0


class TestSignalRConnection:
    """Tests for SignalRConnection class."""

    def test_init_basic(self):
        """Test basic initialization."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        assert conn._url == "wss://rtc.topstepx.com/hubs/market"
        assert conn._access_token == "test_token"
        assert conn._session is None
        assert conn._ws is None

    def test_init_with_session(self):
        """Test initialization with session."""
        mock_session = MagicMock()
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
            session=mock_session,
        )

        assert conn._session == mock_session

    def test_initial_state(self):
        """Test initial connection state."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        assert conn.state == WebSocketState.DISCONNECTED
        assert conn.is_connected is False

    def test_state_property(self):
        """Test state property."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        conn._state = WebSocketState.CONNECTED
        assert conn.state == WebSocketState.CONNECTED
        assert conn.is_connected is True

    def test_on_register_handler(self):
        """Test registering message handlers."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        handler = MagicMock()
        conn.on("GotQuote", handler)

        assert "GotQuote" in conn._message_handlers
        assert handler in conn._message_handlers["GotQuote"]

    def test_on_multiple_handlers(self):
        """Test registering multiple handlers for same method."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        handler1 = MagicMock()
        handler2 = MagicMock()
        conn.on("GotQuote", handler1)
        conn.on("GotQuote", handler2)

        assert len(conn._message_handlers["GotQuote"]) == 2

    def test_off_remove_specific_handler(self):
        """Test unregistering specific handler."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        handler1 = MagicMock()
        handler2 = MagicMock()
        conn.on("GotQuote", handler1)
        conn.on("GotQuote", handler2)
        conn.off("GotQuote", handler1)

        assert handler1 not in conn._message_handlers["GotQuote"]
        assert handler2 in conn._message_handlers["GotQuote"]

    def test_off_remove_all_handlers(self):
        """Test unregistering all handlers for method."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        handler1 = MagicMock()
        handler2 = MagicMock()
        conn.on("GotQuote", handler1)
        conn.on("GotQuote", handler2)
        conn.off("GotQuote")

        assert "GotQuote" not in conn._message_handlers

    def test_record_separator(self):
        """Test record separator constant."""
        assert SignalRConnection.RECORD_SEPARATOR == "\x1e"

    def test_invocation_id_starts_at_zero(self):
        """Test invocation ID starts at zero."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        assert conn._invocation_id == 0

    def test_pending_invocations_empty(self):
        """Test pending invocations starts empty."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        assert conn._pending_invocations == {}


class TestTopstepXWebSocket:
    """Tests for TopstepXWebSocket class."""

    def test_init_basic(self, mock_topstepx_client):
        """Test basic initialization."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        assert ws._client == mock_topstepx_client
        assert ws._auto_reconnect is True
        assert ws._reconnect_delay == 5.0
        assert ws._max_reconnect_attempts == 10

    def test_init_custom_settings(self, mock_topstepx_client):
        """Test initialization with custom settings."""
        ws = TopstepXWebSocket(
            mock_topstepx_client,
            auto_reconnect=False,
            reconnect_delay=10.0,
            max_reconnect_attempts=5,
        )

        assert ws._auto_reconnect is False
        assert ws._reconnect_delay == 10.0
        assert ws._max_reconnect_attempts == 5

    def test_initial_state(self, mock_topstepx_client):
        """Test initial WebSocket state."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        assert ws._market_connection is None
        assert ws._trade_connection is None
        assert ws._subscribed_contracts == set()
        assert ws.is_connected is False

    def test_market_state_disconnected(self, mock_topstepx_client):
        """Test market state when disconnected."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        assert ws.market_state == WebSocketState.DISCONNECTED

    def test_trade_state_disconnected(self, mock_topstepx_client):
        """Test trade state when disconnected."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        assert ws.trade_state == WebSocketState.DISCONNECTED

    def test_on_quote_callback(self, mock_topstepx_client):
        """Test registering quote callback."""
        ws = TopstepXWebSocket(mock_topstepx_client)
        callback = MagicMock()

        ws.on_quote(callback)

        assert callback in ws._quote_callbacks

    def test_on_fill_callback(self, mock_topstepx_client):
        """Test registering fill callback."""
        ws = TopstepXWebSocket(mock_topstepx_client)
        callback = MagicMock()

        ws.on_fill(callback)

        assert callback in ws._fill_callbacks

    def test_on_position_callback(self, mock_topstepx_client):
        """Test registering position callback."""
        ws = TopstepXWebSocket(mock_topstepx_client)
        callback = MagicMock()

        ws.on_position(callback)

        assert callback in ws._position_callbacks

    def test_on_account_callback(self, mock_topstepx_client):
        """Test registering account callback."""
        ws = TopstepXWebSocket(mock_topstepx_client)
        callback = MagicMock()

        ws.on_account(callback)

        assert callback in ws._account_callbacks

    def test_multiple_quote_callbacks(self, mock_topstepx_client):
        """Test registering multiple quote callbacks."""
        ws = TopstepXWebSocket(mock_topstepx_client)
        callback1 = MagicMock()
        callback2 = MagicMock()

        ws.on_quote(callback1)
        ws.on_quote(callback2)

        assert len(ws._quote_callbacks) == 2

    def test_reconnect_state_initial(self, mock_topstepx_client):
        """Test initial reconnection state."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        assert ws._reconnect_attempts == 0
        assert ws._reconnect_task is None
        assert ws._should_run is False

    def test_is_connected_with_market_connection(self, mock_topstepx_client):
        """Test is_connected when market connection exists."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        # Mock a connected market connection
        mock_market_conn = MagicMock()
        mock_market_conn.is_connected = True
        ws._market_connection = mock_market_conn

        assert ws.is_connected is True

    def test_is_connected_market_not_connected(self, mock_topstepx_client):
        """Test is_connected when market connection exists but not connected."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        mock_market_conn = MagicMock()
        mock_market_conn.is_connected = False
        ws._market_connection = mock_market_conn

        assert ws.is_connected is False

    def test_market_state_with_connection(self, mock_topstepx_client):
        """Test market state when connection exists."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        mock_market_conn = MagicMock()
        mock_market_conn.state = WebSocketState.CONNECTED
        ws._market_connection = mock_market_conn

        assert ws.market_state == WebSocketState.CONNECTED

    def test_trade_state_with_connection(self, mock_topstepx_client):
        """Test trade state when connection exists."""
        ws = TopstepXWebSocket(mock_topstepx_client)

        mock_trade_conn = MagicMock()
        mock_trade_conn.state = WebSocketState.RECONNECTING
        ws._trade_connection = mock_trade_conn

        assert ws.trade_state == WebSocketState.RECONNECTING


class TestQuoteSpreadCalculations:
    """Tests for quote spread and price calculations."""

    def test_zero_spread(self):
        """Test spread when bid equals ask."""
        quote = Quote(
            contract_id="MES",
            bid=5000.0,
            ask=5000.0,
            last=5000.0,
        )

        assert quote.spread == 0.0

    def test_typical_spread(self):
        """Test typical 1-tick spread."""
        quote = Quote(
            contract_id="MES",
            bid=5000.0,
            ask=5000.25,
            last=5000.0,
        )

        assert quote.spread == 0.25

    def test_wide_spread(self):
        """Test wider spread."""
        quote = Quote(
            contract_id="MES",
            bid=5000.0,
            ask=5001.0,
            last=5000.0,
        )

        assert quote.spread == 1.0

    def test_mid_price_calculation(self):
        """Test mid price with typical spread."""
        quote = Quote(
            contract_id="MES",
            bid=5000.0,
            ask=5000.50,
            last=5000.0,
        )

        assert quote.mid_price == 5000.25

    def test_mid_price_zero_spread(self):
        """Test mid price with zero spread."""
        quote = Quote(
            contract_id="MES",
            bid=5000.0,
            ask=5000.0,
            last=5000.0,
        )

        assert quote.mid_price == 5000.0


class TestOrderFillSide:
    """Tests for order fill side interpretation."""

    def test_buy_side(self):
        """Test buy side (side=1)."""
        fill = OrderFill(
            order_id="ORD123",
            contract_id="MES",
            side=1,
            fill_price=5000.0,
            fill_size=1,
        )

        assert fill.side == 1  # Buy

    def test_sell_side(self):
        """Test sell side (side=2)."""
        fill = OrderFill(
            order_id="ORD123",
            contract_id="MES",
            side=2,
            fill_price=5000.0,
            fill_size=1,
        )

        assert fill.side == 2  # Sell


class TestPositionPnLTracking:
    """Tests for position P&L tracking."""

    def test_profitable_long_position(self):
        """Test long position with profit."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="MES",
            size=2,
            avg_price=5000.0,
            unrealized_pnl=100.0,  # Position is profitable
            realized_pnl=0.0,
        )

        assert pos.direction == 1  # Long
        assert pos.unrealized_pnl > 0

    def test_losing_short_position(self):
        """Test short position with loss."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="MES",
            size=-2,
            avg_price=5000.0,
            unrealized_pnl=-50.0,  # Position is losing
            realized_pnl=0.0,
        )

        assert pos.direction == -1  # Short
        assert pos.unrealized_pnl < 0

    def test_closed_position_with_realized_pnl(self):
        """Test closed position with realized P&L."""
        pos = PositionUpdate(
            account_id=12345,
            contract_id="MES",
            size=0,
            avg_price=0.0,
            unrealized_pnl=0.0,
            realized_pnl=150.0,  # Position was closed with profit
        )

        assert pos.direction == 0  # Flat
        assert pos.realized_pnl > 0


class TestAccountMargin:
    """Tests for account margin tracking."""

    def test_full_margin_available(self):
        """Test account with full margin available."""
        account = AccountUpdate(
            account_id=12345,
            balance=10000.0,
            available_margin=10000.0,
        )

        assert account.available_margin == account.balance

    def test_partial_margin_used(self):
        """Test account with partial margin used."""
        account = AccountUpdate(
            account_id=12345,
            balance=10000.0,
            available_margin=5000.0,  # Half margin used
        )

        assert account.available_margin < account.balance

    def test_account_with_open_pnl(self):
        """Test account with open P&L."""
        account = AccountUpdate(
            account_id=12345,
            balance=10000.0,
            available_margin=10500.0,  # More available due to open profit
            open_pnl=500.0,
        )

        assert account.open_pnl > 0
