"""
Tests for TopstepX WebSocket dataclasses and state management.

Tests cover:
- Quote dataclass and parsing
- OrderFill dataclass and parsing
- PositionUpdate dataclass and parsing
- AccountUpdate dataclass and parsing
- WebSocketState enum
- SignalRConnection state management
- TopstepXWebSocket state and callback management
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from src.api.topstepx_ws import (
    WebSocketState,
    Quote,
    OrderFill,
    PositionUpdate,
    AccountUpdate,
    SignalRConnection,
)


# =============================================================================
# WebSocketState Tests
# =============================================================================

class TestWebSocketState:
    """Tests for WebSocketState enum."""

    def test_disconnected_state(self):
        """Test DISCONNECTED state."""
        assert WebSocketState.DISCONNECTED is not None
        assert WebSocketState.DISCONNECTED.name == "DISCONNECTED"

    def test_connecting_state(self):
        """Test CONNECTING state."""
        assert WebSocketState.CONNECTING is not None
        assert WebSocketState.CONNECTING.name == "CONNECTING"

    def test_connected_state(self):
        """Test CONNECTED state."""
        assert WebSocketState.CONNECTED is not None
        assert WebSocketState.CONNECTED.name == "CONNECTED"

    def test_reconnecting_state(self):
        """Test RECONNECTING state."""
        assert WebSocketState.RECONNECTING is not None
        assert WebSocketState.RECONNECTING.name == "RECONNECTING"

    def test_closed_state(self):
        """Test CLOSED state."""
        assert WebSocketState.CLOSED is not None
        assert WebSocketState.CLOSED.name == "CLOSED"


# =============================================================================
# Quote Tests
# =============================================================================

class TestQuote:
    """Tests for Quote dataclass."""

    def test_quote_creation(self):
        """Test Quote creation with basic values."""
        quote = Quote(
            contract_id="MES",
            bid=6000.00,
            ask=6000.25,
            last=6000.00,
            bid_size=10,
            ask_size=15,
            volume=1000,
        )

        assert quote.contract_id == "MES"
        assert quote.bid == 6000.00
        assert quote.ask == 6000.25
        assert quote.last == 6000.00
        assert quote.bid_size == 10
        assert quote.ask_size == 15
        assert quote.volume == 1000

    def test_quote_spread(self):
        """Test Quote spread property."""
        quote = Quote(
            contract_id="MES",
            bid=6000.00,
            ask=6000.25,
            last=6000.00,
        )

        assert quote.spread == 0.25

    def test_quote_mid_price(self):
        """Test Quote mid_price property."""
        quote = Quote(
            contract_id="MES",
            bid=6000.00,
            ask=6000.50,
            last=6000.00,
        )

        assert quote.mid_price == 6000.25

    def test_quote_from_signalr_standard(self):
        """Test Quote from_signalr with standard field names."""
        data = {
            "contractId": "MES",
            "bid": 6000.00,
            "ask": 6000.25,
            "last": 6000.00,
            "bidSize": 10,
            "askSize": 15,
            "volume": 1000,
        }

        quote = Quote.from_signalr(data)

        assert quote.contract_id == "MES"
        assert quote.bid == 6000.00
        assert quote.ask == 6000.25
        assert quote.last == 6000.00
        assert quote.bid_size == 10
        assert quote.ask_size == 15
        assert quote.volume == 1000
        assert quote.timestamp is not None

    def test_quote_from_signalr_alternate_names(self):
        """Test Quote from_signalr with alternate field names."""
        data = {
            "symbol": "MES",
            "bidPrice": 6000.00,
            "askPrice": 6000.25,
            "lastPrice": 6000.00,
            "bidQty": 10,
            "askQty": 15,
            "totalVolume": 1000,
        }

        quote = Quote.from_signalr(data)

        assert quote.contract_id == "MES"
        assert quote.bid == 6000.00
        assert quote.ask == 6000.25
        assert quote.last == 6000.00
        assert quote.bid_size == 10
        assert quote.ask_size == 15
        assert quote.volume == 1000

    def test_quote_from_signalr_price_fallback(self):
        """Test Quote from_signalr with 'price' fallback for last."""
        data = {
            "contractId": "MES",
            "bid": 6000.00,
            "ask": 6000.25,
            "price": 6000.00,
        }

        quote = Quote.from_signalr(data)

        assert quote.last == 6000.00

    def test_quote_from_signalr_missing_fields(self):
        """Test Quote from_signalr with missing fields."""
        data = {}

        quote = Quote.from_signalr(data)

        assert quote.contract_id == ""
        assert quote.bid == 0
        assert quote.ask == 0
        assert quote.last == 0


# =============================================================================
# OrderFill Tests
# =============================================================================

class TestOrderFill:
    """Tests for OrderFill dataclass."""

    def test_order_fill_creation(self):
        """Test OrderFill creation with basic values."""
        fill = OrderFill(
            order_id="12345",
            contract_id="MES",
            side=1,
            fill_price=6000.00,
            fill_size=1,
            remaining_size=0,
            is_complete=True,
        )

        assert fill.order_id == "12345"
        assert fill.contract_id == "MES"
        assert fill.side == 1
        assert fill.fill_price == 6000.00
        assert fill.fill_size == 1
        assert fill.remaining_size == 0
        assert fill.is_complete is True

    def test_order_fill_from_signalr_standard(self):
        """Test OrderFill from_signalr with standard field names."""
        data = {
            "orderId": "12345",
            "contractId": "MES",
            "side": 1,
            "fillPrice": 6000.00,
            "fillSize": 1,
            "remainingSize": 0,
            "isComplete": True,
        }

        fill = OrderFill.from_signalr(data)

        assert fill.order_id == "12345"
        assert fill.contract_id == "MES"
        assert fill.side == 1
        assert fill.fill_price == 6000.00
        assert fill.fill_size == 1
        assert fill.remaining_size == 0
        assert fill.is_complete is True
        assert fill.timestamp is not None

    def test_order_fill_from_signalr_alternate_names(self):
        """Test OrderFill from_signalr with alternate field names."""
        data = {
            "orderId": "12345",
            "contractId": "MES",
            "side": 2,
            "price": 6000.00,
            "qty": 1,
            "remainingSize": 0,
            "filled": True,
        }

        fill = OrderFill.from_signalr(data)

        assert fill.fill_price == 6000.00
        assert fill.fill_size == 1
        assert fill.is_complete is True

    def test_order_fill_from_signalr_missing_fields(self):
        """Test OrderFill from_signalr with missing fields."""
        data = {}

        fill = OrderFill.from_signalr(data)

        assert fill.order_id == ""
        assert fill.contract_id == ""
        assert fill.side == 0
        assert fill.fill_price == 0
        assert fill.fill_size == 0


# =============================================================================
# PositionUpdate Tests
# =============================================================================

class TestPositionUpdate:
    """Tests for PositionUpdate dataclass."""

    def test_position_update_creation(self):
        """Test PositionUpdate creation with basic values."""
        pos = PositionUpdate(
            account_id=123,
            contract_id="MES",
            size=1,
            avg_price=6000.00,
            unrealized_pnl=10.0,
            realized_pnl=5.0,
        )

        assert pos.account_id == 123
        assert pos.contract_id == "MES"
        assert pos.size == 1
        assert pos.avg_price == 6000.00
        assert pos.unrealized_pnl == 10.0
        assert pos.realized_pnl == 5.0

    def test_position_update_direction_long(self):
        """Test PositionUpdate direction property for long position."""
        pos = PositionUpdate(
            account_id=123,
            contract_id="MES",
            size=1,
            avg_price=6000.00,
        )

        assert pos.direction == 1

    def test_position_update_direction_short(self):
        """Test PositionUpdate direction property for short position."""
        pos = PositionUpdate(
            account_id=123,
            contract_id="MES",
            size=-1,
            avg_price=6000.00,
        )

        assert pos.direction == -1

    def test_position_update_direction_flat(self):
        """Test PositionUpdate direction property for flat position."""
        pos = PositionUpdate(
            account_id=123,
            contract_id="MES",
            size=0,
            avg_price=0.0,
        )

        assert pos.direction == 0

    def test_position_update_from_signalr_standard(self):
        """Test PositionUpdate from_signalr with standard field names."""
        data = {
            "accountId": 123,
            "contractId": "MES",
            "size": 1,
            "avgPrice": 6000.00,
            "unrealizedPnl": 10.0,
            "realizedPnl": 5.0,
        }

        pos = PositionUpdate.from_signalr(data)

        assert pos.account_id == 123
        assert pos.contract_id == "MES"
        assert pos.size == 1
        assert pos.avg_price == 6000.00
        assert pos.unrealized_pnl == 10.0
        assert pos.realized_pnl == 5.0
        assert pos.timestamp is not None

    def test_position_update_from_signalr_alternate_names(self):
        """Test PositionUpdate from_signalr with alternate field names."""
        data = {
            "accountId": 123,
            "contractId": "MES",
            "qty": 1,
            "price": 6000.00,
        }

        pos = PositionUpdate.from_signalr(data)

        assert pos.size == 1
        assert pos.avg_price == 6000.00


# =============================================================================
# AccountUpdate Tests
# =============================================================================

class TestAccountUpdate:
    """Tests for AccountUpdate dataclass."""

    def test_account_update_creation(self):
        """Test AccountUpdate creation with basic values."""
        update = AccountUpdate(
            account_id=123,
            balance=50000.0,
            available_margin=45000.0,
            open_pnl=100.0,
            closed_pnl=500.0,
        )

        assert update.account_id == 123
        assert update.balance == 50000.0
        assert update.available_margin == 45000.0
        assert update.open_pnl == 100.0
        assert update.closed_pnl == 500.0

    def test_account_update_from_signalr_standard(self):
        """Test AccountUpdate from_signalr with standard field names."""
        data = {
            "accountId": 123,
            "balance": 50000.0,
            "availableMargin": 45000.0,
            "openPnl": 100.0,
            "closedPnl": 500.0,
        }

        update = AccountUpdate.from_signalr(data)

        assert update.account_id == 123
        assert update.balance == 50000.0
        assert update.available_margin == 45000.0
        assert update.open_pnl == 100.0
        assert update.closed_pnl == 500.0
        assert update.timestamp is not None

    def test_account_update_from_signalr_alternate_names(self):
        """Test AccountUpdate from_signalr with alternate field names."""
        data = {
            "id": 123,
            "accountBalance": 50000.0,
            "unrealizedPnl": 100.0,
            "realizedPnl": 500.0,
        }

        update = AccountUpdate.from_signalr(data)

        assert update.account_id == 123
        assert update.balance == 50000.0
        assert update.open_pnl == 100.0
        assert update.closed_pnl == 500.0


# =============================================================================
# SignalRConnection State Tests
# =============================================================================

class TestSignalRConnectionState:
    """Tests for SignalRConnection state management."""

    def test_signalr_connection_initial_state(self):
        """Test SignalRConnection initial state."""
        conn = SignalRConnection(
            url="wss://example.com/hub",
            access_token="test_token",
        )

        assert conn._state == WebSocketState.DISCONNECTED
        assert conn._ws is None
        assert conn._session is None

    def test_signalr_connection_url(self):
        """Test SignalRConnection URL storage."""
        conn = SignalRConnection(
            url="wss://example.com/hub",
            access_token="test_token",
        )

        assert conn._url == "wss://example.com/hub"

    def test_signalr_connection_token(self):
        """Test SignalRConnection token storage."""
        conn = SignalRConnection(
            url="wss://example.com/hub",
            access_token="test_token",
        )

        assert conn._access_token == "test_token"

    def test_signalr_connection_record_separator(self):
        """Test SignalRConnection record separator constant."""
        assert SignalRConnection.RECORD_SEPARATOR == "\x1e"


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestQuoteProcessing:
    """Tests for quote processing scenarios."""

    def test_quote_spread_zero_when_same(self):
        """Test spread is zero when bid equals ask."""
        quote = Quote(
            contract_id="MES",
            bid=6000.00,
            ask=6000.00,
            last=6000.00,
        )

        assert quote.spread == 0.0

    def test_quote_mid_price_equals_last_when_spread_zero(self):
        """Test mid_price when spread is zero."""
        quote = Quote(
            contract_id="MES",
            bid=6000.00,
            ask=6000.00,
            last=6000.00,
        )

        assert quote.mid_price == 6000.00

    def test_quote_large_spread(self):
        """Test quote with large spread."""
        quote = Quote(
            contract_id="MES",
            bid=5999.00,
            ask=6001.00,
            last=6000.00,
        )

        assert quote.spread == 2.0
        assert quote.mid_price == 6000.0


class TestPositionUpdateScenarios:
    """Tests for position update scenarios."""

    def test_large_long_position(self):
        """Test large long position."""
        pos = PositionUpdate(
            account_id=123,
            contract_id="MES",
            size=10,
            avg_price=6000.00,
        )

        assert pos.direction == 1

    def test_large_short_position(self):
        """Test large short position."""
        pos = PositionUpdate(
            account_id=123,
            contract_id="MES",
            size=-10,
            avg_price=6000.00,
        )

        assert pos.direction == -1
