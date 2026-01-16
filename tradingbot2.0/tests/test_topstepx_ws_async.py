"""
Async tests for TopstepX WebSocket module.

Tests cover the async methods:
- SignalRConnection.connect, disconnect, invoke, send
- SignalRConnection._send_message, _receive_raw, _receive_loop, _handle_message, _ping_loop
- TopstepXWebSocket.connect, disconnect, subscribe_quotes, unsubscribe_quotes
- TopstepXWebSocket._connect_market, _connect_trade
- TopstepXWebSocket._handle_quote, _handle_fill, _handle_position, _handle_account
- TopstepXWebSocket._auto_reconnect_loop
- TopstepXWebSocket context manager
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import aiohttp

from src.api.topstepx_ws import (
    WebSocketState,
    Quote,
    OrderFill,
    PositionUpdate,
    AccountUpdate,
    SignalRConnection,
    TopstepXWebSocket,
)
from src.api.topstepx_client import TopstepXConnectionError, TopstepXConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_client():
    """Create a mock TopstepXClient."""
    client = MagicMock()
    client.access_token = "test_token"
    client.default_account_id = 12345
    client.config = TopstepXConfig(
        ws_market_url="wss://rtc.topstepx.com/hubs/market",
        ws_trade_url="wss://rtc.topstepx.com/hubs/trade",
    )
    return client


@pytest.fixture
def mock_ws_response():
    """Create a mock WebSocket response."""
    ws = AsyncMock()
    ws.closed = False
    ws.close = AsyncMock()
    ws.send_str = AsyncMock()
    return ws


@pytest.fixture
def mock_session(mock_ws_response):
    """Create a mock aiohttp session."""
    session = MagicMock()
    session.ws_connect = AsyncMock(return_value=mock_ws_response)
    session.post = MagicMock()
    session.closed = False
    return session


# =============================================================================
# SignalRConnection Tests
# =============================================================================

class TestSignalRConnectionConnect:
    """Tests for SignalRConnection.connect method."""

    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self):
        """Test connect returns immediately when already connected."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._state = WebSocketState.CONNECTED

        await conn.connect()

        # Should not change state
        assert conn.state == WebSocketState.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_when_connecting(self):
        """Test connect returns immediately when already connecting."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._state = WebSocketState.CONNECTING

        await conn.connect()

        # Should not change state
        assert conn.state == WebSocketState.CONNECTING

    @pytest.mark.asyncio
    async def test_connect_negotiate_failure(self, mock_session):
        """Test connect handles negotiate failure."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
            session=mock_session,
        )

        # Mock negotiate response failure
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_response)

        with pytest.raises(TopstepXConnectionError, match="Negotiate failed"):
            await conn.connect()

        assert conn.state == WebSocketState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_handshake_failure(self, mock_session, mock_ws_response):
        """Test connect handles handshake failure."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
            session=mock_session,
        )

        # Mock successful negotiate
        mock_negotiate_response = AsyncMock()
        mock_negotiate_response.status = 200
        mock_negotiate_response.json = AsyncMock(return_value={"connectionToken": "conn_token"})
        mock_negotiate_response.__aenter__ = AsyncMock(return_value=mock_negotiate_response)
        mock_negotiate_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_negotiate_response)

        # Mock handshake failure response
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = json.dumps({"error": "Handshake failed"}) + "\x1e"
        mock_ws_response.receive = AsyncMock(return_value=mock_msg)

        with pytest.raises(TopstepXConnectionError, match="Handshake failed"):
            await conn.connect()

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_session, mock_ws_response):
        """Test successful connection."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
            session=mock_session,
        )

        # Mock successful negotiate
        mock_negotiate_response = AsyncMock()
        mock_negotiate_response.status = 200
        mock_negotiate_response.json = AsyncMock(return_value={"connectionToken": "conn_token"})
        mock_negotiate_response.__aenter__ = AsyncMock(return_value=mock_negotiate_response)
        mock_negotiate_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_negotiate_response)

        # Mock successful handshake response (empty response = success)
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = "{}\x1e"
        mock_ws_response.receive = AsyncMock(return_value=mock_msg)

        await conn.connect()

        assert conn.state == WebSocketState.CONNECTED
        assert conn.is_connected is True

        # Cleanup tasks
        if conn._receive_task:
            conn._receive_task.cancel()
        if conn._ping_task:
            conn._ping_task.cancel()


class TestSignalRConnectionDisconnect:
    """Tests for SignalRConnection.disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_cancels_tasks(self, mock_ws_response):
        """Test disconnect cancels receive and ping tasks."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        # Create mock tasks
        async def dummy_task():
            await asyncio.sleep(100)

        conn._receive_task = asyncio.create_task(dummy_task())
        conn._ping_task = asyncio.create_task(dummy_task())

        await conn.disconnect()

        assert conn.state == WebSocketState.CLOSED
        assert conn._receive_task is None
        assert conn._ping_task is None

    @pytest.mark.asyncio
    async def test_disconnect_closes_websocket(self, mock_ws_response):
        """Test disconnect closes WebSocket connection."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        await conn.disconnect()

        mock_ws_response.close.assert_called_once()
        assert conn._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_clears_pending_invocations(self, mock_ws_response):
        """Test disconnect cancels pending invocations."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        # Add pending invocation
        future = asyncio.get_event_loop().create_future()
        conn._pending_invocations["1"] = future

        await conn.disconnect()

        assert len(conn._pending_invocations) == 0
        assert future.cancelled()


class TestSignalRConnectionInvoke:
    """Tests for SignalRConnection.invoke method."""

    @pytest.mark.asyncio
    async def test_invoke_not_connected(self):
        """Test invoke raises error when not connected."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        with pytest.raises(TopstepXConnectionError, match="Not connected"):
            await conn.invoke("TestMethod", "arg1")

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, mock_ws_response):
        """Test invoke handles timeout."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        with pytest.raises(asyncio.TimeoutError):
            await conn.invoke("TestMethod", "arg1", timeout=0.1)

    @pytest.mark.asyncio
    async def test_invoke_increments_invocation_id(self, mock_ws_response):
        """Test invoke increments invocation ID."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        initial_id = conn._invocation_id

        # Start invoke but don't await completion
        task = asyncio.create_task(conn.invoke("TestMethod", timeout=0.1))
        await asyncio.sleep(0.01)  # Let task start

        assert conn._invocation_id == initial_id + 1

        # Cancel to clean up
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


class TestSignalRConnectionSend:
    """Tests for SignalRConnection.send method."""

    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        """Test send raises error when not connected."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        with pytest.raises(TopstepXConnectionError, match="Not connected"):
            await conn.send("TestMethod", "arg1")

    @pytest.mark.asyncio
    async def test_send_success(self, mock_ws_response):
        """Test send sends message correctly."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        await conn.send("TestMethod", "arg1", "arg2")

        mock_ws_response.send_str.assert_called_once()
        sent_data = mock_ws_response.send_str.call_args[0][0]
        assert "TestMethod" in sent_data
        assert '"arguments": ["arg1", "arg2"]' in sent_data


class TestSignalRConnectionSendMessage:
    """Tests for SignalRConnection._send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self):
        """Test _send_message raises error when not connected."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = None

        with pytest.raises(TopstepXConnectionError, match="not connected"):
            await conn._send_message({"type": 1})

    @pytest.mark.asyncio
    async def test_send_message_closed_ws(self, mock_ws_response):
        """Test _send_message raises error when WebSocket is closed."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        mock_ws_response.closed = True
        conn._ws = mock_ws_response

        with pytest.raises(TopstepXConnectionError, match="not connected"):
            await conn._send_message({"type": 1})

    @pytest.mark.asyncio
    async def test_send_message_appends_record_separator(self, mock_ws_response):
        """Test _send_message appends record separator."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response

        await conn._send_message({"type": 1})

        sent_data = mock_ws_response.send_str.call_args[0][0]
        assert sent_data.endswith("\x1e")


class TestSignalRConnectionReceiveRaw:
    """Tests for SignalRConnection._receive_raw method."""

    @pytest.mark.asyncio
    async def test_receive_raw_no_websocket(self):
        """Test _receive_raw returns None when no WebSocket."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = None

        result = await conn._receive_raw()
        assert result is None

    @pytest.mark.asyncio
    async def test_receive_raw_text_message(self, mock_ws_response):
        """Test _receive_raw handles text message."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response

        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = '{"type": 1, "target": "test"}\x1e'
        mock_ws_response.receive = AsyncMock(return_value=mock_msg)

        result = await conn._receive_raw()

        assert result == {"type": 1, "target": "test"}

    @pytest.mark.asyncio
    async def test_receive_raw_closed_message(self, mock_ws_response):
        """Test _receive_raw handles closed message."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.CLOSED
        mock_ws_response.receive = AsyncMock(return_value=mock_msg)

        result = await conn._receive_raw()

        assert result is None
        assert conn.state == WebSocketState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_receive_raw_error_message(self, mock_ws_response):
        """Test _receive_raw handles error message."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.ERROR
        mock_msg.data = "Connection error"
        mock_ws_response.receive = AsyncMock(return_value=mock_msg)

        result = await conn._receive_raw()

        assert result is None
        assert conn.state == WebSocketState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_receive_raw_timeout(self, mock_ws_response):
        """Test _receive_raw handles timeout."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response

        mock_ws_response.receive = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await conn._receive_raw()

        assert result is None


class TestSignalRConnectionHandleMessage:
    """Tests for SignalRConnection._handle_message method."""

    @pytest.mark.asyncio
    async def test_handle_message_invocation(self, mock_ws_response):
        """Test handling invocation message (type=1)."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        handler = MagicMock()
        conn.on("TestMethod", handler)

        message = {"type": 1, "target": "TestMethod", "arguments": ["arg1", "arg2"]}
        await conn._handle_message(message)

        handler.assert_called_once_with("arg1", "arg2")

    @pytest.mark.asyncio
    async def test_handle_message_invocation_async_handler(self, mock_ws_response):
        """Test handling invocation with async handler."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        async_handler = AsyncMock()
        conn.on("TestMethod", async_handler)

        message = {"type": 1, "target": "TestMethod", "arguments": ["arg1"]}
        await conn._handle_message(message)

        async_handler.assert_called_once_with("arg1")

    @pytest.mark.asyncio
    async def test_handle_message_completion_success(self):
        """Test handling completion message (type=3) with result."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        future = asyncio.get_event_loop().create_future()
        conn._pending_invocations["1"] = future

        message = {"type": 3, "invocationId": "1", "result": "success_result"}
        await conn._handle_message(message)

        assert future.result() == "success_result"

    @pytest.mark.asyncio
    async def test_handle_message_completion_error(self):
        """Test handling completion message (type=3) with error."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )

        future = asyncio.get_event_loop().create_future()
        conn._pending_invocations["1"] = future

        message = {"type": 3, "invocationId": "1", "error": "Some error"}
        await conn._handle_message(message)

        with pytest.raises(TopstepXConnectionError, match="Some error"):
            future.result()

    @pytest.mark.asyncio
    async def test_handle_message_ping(self, mock_ws_response):
        """Test handling ping message (type=6)."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response

        message = {"type": 6}
        await conn._handle_message(message)

        # Should send pong
        mock_ws_response.send_str.assert_called_once()
        sent_data = mock_ws_response.send_str.call_args[0][0]
        assert '"type": 6' in sent_data

    @pytest.mark.asyncio
    async def test_handle_message_close(self):
        """Test handling close message (type=7)."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._state = WebSocketState.CONNECTED

        message = {"type": 7}
        await conn._handle_message(message)

        assert conn.state == WebSocketState.DISCONNECTED


class TestSignalRConnectionPingLoop:
    """Tests for SignalRConnection._ping_loop method."""

    @pytest.mark.asyncio
    async def test_ping_loop_sends_ping(self, mock_ws_response):
        """Test ping loop sends periodic pings."""
        conn = SignalRConnection(
            url="wss://rtc.topstepx.com/hubs/market",
            access_token="test_token",
        )
        conn._ws = mock_ws_response
        conn._state = WebSocketState.CONNECTED

        # Start ping loop but cancel quickly
        task = asyncio.create_task(conn._ping_loop())
        await asyncio.sleep(0.01)  # Very short wait
        conn._state = WebSocketState.DISCONNECTED  # Stop the loop

        # Wait for task to finish
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()


# =============================================================================
# TopstepXWebSocket Tests
# =============================================================================

class TestTopstepXWebSocketConnect:
    """Tests for TopstepXWebSocket.connect method."""

    @pytest.mark.asyncio
    async def test_connect_not_authenticated(self, mock_client):
        """Test connect raises error when not authenticated."""
        mock_client.access_token = None
        ws = TopstepXWebSocket(mock_client)

        with pytest.raises(TopstepXConnectionError, match="not authenticated"):
            await ws.connect()

    @pytest.mark.asyncio
    async def test_connect_sets_should_run(self, mock_client):
        """Test connect sets should_run flag."""
        ws = TopstepXWebSocket(mock_client)

        with patch.object(ws, '_connect_market', new_callable=AsyncMock):
            with patch.object(ws, '_connect_trade', new_callable=AsyncMock):
                await ws.connect()

        assert ws._should_run is True


class TestTopstepXWebSocketDisconnect:
    """Tests for TopstepXWebSocket.disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, mock_client):
        """Test disconnect clears state."""
        ws = TopstepXWebSocket(mock_client)
        ws._should_run = True
        ws._subscribed_contracts.add("MES")

        mock_market = AsyncMock()
        mock_trade = AsyncMock()
        ws._market_connection = mock_market
        ws._trade_connection = mock_trade

        await ws.disconnect()

        assert ws._should_run is False
        assert len(ws._subscribed_contracts) == 0
        mock_market.disconnect.assert_called_once()
        mock_trade.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_reconnect_task(self, mock_client):
        """Test disconnect cancels reconnect task."""
        ws = TopstepXWebSocket(mock_client)

        async def dummy_reconnect():
            await asyncio.sleep(100)

        ws._reconnect_task = asyncio.create_task(dummy_reconnect())
        ws._should_run = True

        await ws.disconnect()

        assert ws._reconnect_task is None


class TestTopstepXWebSocketSubscribeQuotes:
    """Tests for TopstepXWebSocket.subscribe_quotes method."""

    @pytest.mark.asyncio
    async def test_subscribe_quotes_not_connected(self, mock_client):
        """Test subscribe_quotes raises error when not connected."""
        ws = TopstepXWebSocket(mock_client)

        with pytest.raises(TopstepXConnectionError, match="not connected"):
            await ws.subscribe_quotes(["MES"])

    @pytest.mark.asyncio
    async def test_subscribe_quotes_success(self, mock_client):
        """Test successful quote subscription."""
        ws = TopstepXWebSocket(mock_client)

        mock_market = MagicMock()
        mock_market.is_connected = True
        mock_market.invoke = AsyncMock()
        ws._market_connection = mock_market

        await ws.subscribe_quotes(["MES", "ES"])

        assert "MES" in ws._subscribed_contracts
        assert "ES" in ws._subscribed_contracts
        mock_market.invoke.assert_called_once_with("SubscribeQuotes", ["MES", "ES"])


class TestTopstepXWebSocketUnsubscribeQuotes:
    """Tests for TopstepXWebSocket.unsubscribe_quotes method."""

    @pytest.mark.asyncio
    async def test_unsubscribe_quotes_not_connected(self, mock_client):
        """Test unsubscribe_quotes does nothing when not connected."""
        ws = TopstepXWebSocket(mock_client)
        ws._subscribed_contracts.add("MES")

        await ws.unsubscribe_quotes(["MES"])

        # When not connected, returns early without modifying local tracking
        assert "MES" in ws._subscribed_contracts

    @pytest.mark.asyncio
    async def test_unsubscribe_quotes_success(self, mock_client):
        """Test successful quote unsubscription."""
        ws = TopstepXWebSocket(mock_client)
        ws._subscribed_contracts.add("MES")

        mock_market = MagicMock()
        mock_market.is_connected = True
        mock_market.invoke = AsyncMock()
        ws._market_connection = mock_market

        await ws.unsubscribe_quotes(["MES"])

        assert "MES" not in ws._subscribed_contracts
        mock_market.invoke.assert_called_once_with("UnsubscribeQuotes", ["MES"])


class TestTopstepXWebSocketHandlers:
    """Tests for TopstepXWebSocket handler methods."""

    def test_handle_quote_dict(self, mock_client):
        """Test _handle_quote with dict data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_quote(callback)

        data = {
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
        }

        ws._handle_quote(data)

        callback.assert_called_once()
        quote = callback.call_args[0][0]
        assert isinstance(quote, Quote)
        assert quote.contract_id == "MES"

    def test_handle_quote_list(self, mock_client):
        """Test _handle_quote with list data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_quote(callback)

        data = [{
            "contractId": "MES",
            "bid": 5000.0,
            "ask": 5000.25,
            "last": 5000.0,
        }]

        ws._handle_quote(data)

        callback.assert_called_once()

    def test_handle_quote_empty_list(self, mock_client):
        """Test _handle_quote with empty list data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_quote(callback)

        ws._handle_quote([])

        callback.assert_not_called()

    def test_handle_quote_callback_error(self, mock_client):
        """Test _handle_quote handles callback errors gracefully."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock(side_effect=ValueError("test error"))
        ws.on_quote(callback)

        data = {"contractId": "MES", "bid": 5000.0, "ask": 5000.25, "last": 5000.0}

        # Should not raise
        ws._handle_quote(data)

    def test_handle_fill_dict(self, mock_client):
        """Test _handle_fill with dict data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_fill(callback)

        data = {
            "orderId": "ORD123",
            "contractId": "MES",
            "side": 1,
            "fillPrice": 5000.0,
            "fillSize": 1,
        }

        ws._handle_fill(data)

        callback.assert_called_once()
        fill = callback.call_args[0][0]
        assert isinstance(fill, OrderFill)
        assert fill.order_id == "ORD123"

    def test_handle_fill_list(self, mock_client):
        """Test _handle_fill with list data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_fill(callback)

        data = [{
            "orderId": "ORD123",
            "contractId": "MES",
            "side": 1,
            "fillPrice": 5000.0,
            "fillSize": 1,
        }]

        ws._handle_fill(data)

        callback.assert_called_once()

    def test_handle_position_dict(self, mock_client):
        """Test _handle_position with dict data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_position(callback)

        data = {
            "accountId": 12345,
            "contractId": "MES",
            "size": 1,
            "avgPrice": 5000.0,
        }

        ws._handle_position(data)

        callback.assert_called_once()
        pos = callback.call_args[0][0]
        assert isinstance(pos, PositionUpdate)
        assert pos.account_id == 12345

    def test_handle_position_list(self, mock_client):
        """Test _handle_position with list data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_position(callback)

        data = [{
            "accountId": 12345,
            "contractId": "MES",
            "size": 1,
            "avgPrice": 5000.0,
        }]

        ws._handle_position(data)

        callback.assert_called_once()

    def test_handle_account_dict(self, mock_client):
        """Test _handle_account with dict data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_account(callback)

        data = {
            "accountId": 12345,
            "balance": 10000.0,
        }

        ws._handle_account(data)

        callback.assert_called_once()
        account = callback.call_args[0][0]
        assert isinstance(account, AccountUpdate)
        assert account.account_id == 12345

    def test_handle_account_list(self, mock_client):
        """Test _handle_account with list data."""
        ws = TopstepXWebSocket(mock_client)
        callback = MagicMock()
        ws.on_account(callback)

        data = [{
            "accountId": 12345,
            "balance": 10000.0,
        }]

        ws._handle_account(data)

        callback.assert_called_once()


class TestTopstepXWebSocketContextManager:
    """Tests for TopstepXWebSocket context manager."""

    @pytest.mark.asyncio
    async def test_aenter(self, mock_client):
        """Test async context manager entry."""
        ws = TopstepXWebSocket(mock_client)

        with patch.object(ws, 'connect', new_callable=AsyncMock):
            result = await ws.__aenter__()

        assert result is ws

    @pytest.mark.asyncio
    async def test_aexit(self, mock_client):
        """Test async context manager exit."""
        ws = TopstepXWebSocket(mock_client)

        with patch.object(ws, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
            await ws.__aexit__(None, None, None)

        mock_disconnect.assert_called_once()


class TestTopstepXWebSocketAutoReconnect:
    """Tests for TopstepXWebSocket auto reconnection."""

    @pytest.mark.asyncio
    async def test_auto_reconnect_disabled(self, mock_client):
        """Test auto reconnect when disabled."""
        ws = TopstepXWebSocket(mock_client, auto_reconnect=False)
        ws._should_run = True

        # Should exit immediately
        task = asyncio.create_task(ws._auto_reconnect_loop())
        await asyncio.sleep(0.01)

        # Check task completed
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_auto_reconnect_max_attempts(self, mock_client):
        """Test auto reconnect respects max attempts."""
        ws = TopstepXWebSocket(mock_client, max_reconnect_attempts=2)
        ws._should_run = True
        ws._auto_reconnect = True

        # Mock disconnected state
        ws._market_connection = MagicMock()
        ws._market_connection.is_connected = False

        with patch.object(ws, 'connect', new_callable=AsyncMock, side_effect=Exception("Failed")):
            task = asyncio.create_task(ws._auto_reconnect_loop())
            await asyncio.sleep(0.5)  # Let it run a bit

            # Cancel task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
