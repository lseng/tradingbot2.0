"""
Tests for TopstepX API Integration Module.

This module tests the TopstepX API client including:
- Authentication and token management
- Rate limiting
- REST API endpoints
- WebSocket connections
- Error handling and retry logic

Note: These are unit tests using mocks. Integration tests with real API
require credentials and should be run separately.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Configure pytest-asyncio to use auto mode
pytestmark = pytest.mark.asyncio(loop_scope="function")

from src.api.topstepx_client import (
    TopstepXClient,
    TopstepXConfig,
    TopstepXAPIError,
    TopstepXAuthError,
    TopstepXRateLimitError,
    RateLimiter,
)
from src.api.topstepx_rest import (
    TopstepXREST,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeUnit,
    BarData,
    OrderResponse,
    PositionData,
    AccountInfo,
    get_mes_contract_id,
    get_current_mes_contract,
)
from src.api.topstepx_ws import (
    TopstepXWebSocket,
    Quote,
    OrderFill,
    PositionUpdate,
    AccountUpdate,
    WebSocketState,
    SignalRConnection,
)


# ============================================================================
# TopstepXConfig Tests
# ============================================================================

class TestTopstepXConfig:
    """Tests for TopstepXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TopstepXConfig()
        assert config.base_url == "https://api.topstepx.com"
        assert config.ws_market_url == "wss://rtc.topstepx.com/hubs/market"
        assert config.ws_trade_url == "wss://rtc.topstepx.com/hubs/trade"
        assert config.rate_limit_requests == 50
        assert config.rate_limit_window == 30.0
        assert config.token_refresh_margin == 600.0
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TopstepXConfig(
            username="test@example.com",
            password="secret123",
            rate_limit_requests=100,
            max_retries=5,
        )
        assert config.username == "test@example.com"
        assert config.password == "secret123"
        assert config.rate_limit_requests == 100
        assert config.max_retries == 5

    def test_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict("os.environ", {
            "TOPSTEPX_USERNAME": "env_user",
            "TOPSTEPX_PASSWORD": "env_pass",
            "TOPSTEPX_DEVICE_ID": "test-device-123",
        }):
            config = TopstepXConfig.from_env()
            assert config.username == "env_user"
            assert config.password == "env_pass"
            assert config.device_id == "test-device-123"

    def test_device_id_auto_generated(self):
        """Test device ID is auto-generated if not provided."""
        config = TopstepXConfig()
        assert config.device_id is not None
        assert len(config.device_id) > 0


# ============================================================================
# RateLimiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_initial_acquire(self):
        """Test acquiring rate limit initially returns 0 wait time."""
        limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        wait = await limiter.acquire()
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        """Test multiple acquires under limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            wait = await limiter.acquire()
            assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_at_limit(self):
        """Test acquire returns wait time when at limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=1.0)
        await limiter.acquire()
        await limiter.acquire()
        wait = await limiter.acquire()
        assert wait > 0.0

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test reset clears rate limiter state."""
        limiter = RateLimiter(max_requests=2, window_seconds=1.0)
        await limiter.acquire()
        await limiter.acquire()
        limiter.reset()
        wait = await limiter.acquire()
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Test requests expire after window."""
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)
        await limiter.acquire()
        await limiter.acquire()
        await asyncio.sleep(0.15)
        wait = await limiter.acquire()
        assert wait == 0.0


# ============================================================================
# TopstepXClient Tests
# ============================================================================

class TestTopstepXClient:
    """Tests for TopstepXClient."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = TopstepXConfig(
            username="test@example.com",
            password="secret123",
        )
        client = TopstepXClient(config=config)
        assert client.config.username == "test@example.com"
        assert client.config.password == "secret123"

    def test_init_with_credentials(self):
        """Test initialization with direct credentials."""
        client = TopstepXClient(
            username="user@test.com",
            password="pass123",
        )
        assert client.config.username == "user@test.com"
        assert client.config.password == "pass123"

    def test_init_credentials_override_config(self):
        """Test credentials override config values."""
        config = TopstepXConfig(
            username="config_user",
            password="config_pass",
        )
        client = TopstepXClient(
            username="override_user",
            password="override_pass",
            config=config,
        )
        assert client.config.username == "override_user"
        assert client.config.password == "override_pass"

    def test_not_authenticated_initially(self):
        """Test client is not authenticated initially."""
        client = TopstepXClient(username="user", password="pass")
        assert not client.is_authenticated
        assert client.access_token is None
        assert client.user_id is None
        assert client.accounts == []

    def test_authenticate_sets_state(self):
        """Test authentication state is properly set after successful auth.

        Note: This test verifies the state management without actual HTTP calls.
        Integration tests with real API should verify full authentication flow.
        """
        client = TopstepXClient(username="user", password="pass")

        # Simulate successful authentication by setting internal state
        client._access_token = "test_token_123"
        client._user_id = 12345
        client._accounts = [{"id": 67890, "name": "TestAccount"}]
        client._token_expiry = datetime.utcnow() + timedelta(minutes=90)

        # Verify state is properly set
        assert client.is_authenticated
        assert client.access_token == "test_token_123"
        assert client.user_id == 12345
        assert len(client.accounts) == 1
        assert client.default_account_id == 67890

    @pytest.mark.asyncio
    async def test_authenticate_missing_credentials(self):
        """Test authentication fails without credentials."""
        client = TopstepXClient()
        with pytest.raises(ValueError, match="Username and password"):
            await client.authenticate()

    @pytest.mark.asyncio
    async def test_token_expiry(self):
        """Test token expiry detection."""
        client = TopstepXClient(username="user", password="pass")
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() - timedelta(minutes=1)
        assert not client.is_authenticated

    @pytest.mark.asyncio
    async def test_token_valid(self):
        """Test token validity detection."""
        client = TopstepXClient(username="user", password="pass")
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(minutes=30)
        assert client.is_authenticated

    @pytest.mark.asyncio
    async def test_close(self):
        """Test client close."""
        client = TopstepXClient(username="user", password="pass")
        await client.close()
        assert client._session is None


# ============================================================================
# OrderType and OrderSide Enum Tests
# ============================================================================

class TestEnums:
    """Tests for enum types."""

    def test_order_type_values(self):
        """Test OrderType enum values match API."""
        assert OrderType.LIMIT == 1
        assert OrderType.MARKET == 2
        assert OrderType.STOP == 3
        assert OrderType.STOP_LIMIT == 4

    def test_order_side_values(self):
        """Test OrderSide enum values match API."""
        assert OrderSide.BUY == 1
        assert OrderSide.SELL == 2

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING == 0
        assert OrderStatus.WORKING == 1
        assert OrderStatus.FILLED == 2
        assert OrderStatus.CANCELLED == 3
        assert OrderStatus.REJECTED == 4

    def test_time_unit_values(self):
        """Test TimeUnit enum values match API."""
        assert TimeUnit.SECOND == 1
        assert TimeUnit.MINUTE == 2
        assert TimeUnit.HOUR == 3
        assert TimeUnit.DAY == 4
        assert TimeUnit.WEEK == 5
        assert TimeUnit.MONTH == 6


# ============================================================================
# BarData Tests
# ============================================================================

class TestBarData:
    """Tests for BarData dataclass."""

    def test_from_api_dict(self):
        """Test creating BarData from API response dict."""
        data = {
            "timestamp": "2026-01-15T14:30:00Z",
            "open": 6050.25,
            "high": 6052.00,
            "low": 6049.50,
            "close": 6051.75,
            "volume": 1523,
        }
        bar = BarData.from_api(data)
        assert bar.open == 6050.25
        assert bar.high == 6052.00
        assert bar.low == 6049.50
        assert bar.close == 6051.75
        assert bar.volume == 1523

    def test_from_api_with_milliseconds(self):
        """Test parsing timestamp with milliseconds."""
        data = {
            "timestamp": "2026-01-15T14:30:00.123Z",
            "open": 6050.0,
            "high": 6051.0,
            "low": 6049.0,
            "close": 6050.5,
            "volume": 100,
        }
        bar = BarData.from_api(data)
        assert bar.timestamp.hour == 14
        assert bar.timestamp.minute == 30

    def test_to_dict(self):
        """Test converting BarData to dict."""
        bar = BarData(
            timestamp=datetime(2026, 1, 15, 14, 30, 0),
            open=6050.25,
            high=6052.00,
            low=6049.50,
            close=6051.75,
            volume=1523,
        )
        d = bar.to_dict()
        assert d["open"] == 6050.25
        assert d["close"] == 6051.75
        assert d["volume"] == 1523


# ============================================================================
# OrderResponse Tests
# ============================================================================

class TestOrderResponse:
    """Tests for OrderResponse dataclass."""

    def test_from_api(self):
        """Test creating OrderResponse from API response."""
        data = {
            "orderId": "ORD123",
            "accountId": 67890,
            "contractId": "CON.F.US.MES.H26",
            "side": 1,
            "type": 2,
            "size": 1,
            "status": 2,
            "filledSize": 1,
            "avgFillPrice": 6050.25,
        }
        order = OrderResponse.from_api(data)
        assert order.order_id == "ORD123"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert order.is_filled
        assert not order.is_working
        assert order.avg_fill_price == 6050.25

    def test_is_working(self):
        """Test is_working property."""
        order = OrderResponse(
            order_id="ORD123",
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=1,
            status=OrderStatus.WORKING,
        )
        assert order.is_working
        assert not order.is_filled

    def test_is_rejected(self):
        """Test is_rejected property."""
        order = OrderResponse(
            order_id="ORD123",
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            size=1,
            status=OrderStatus.REJECTED,
            error_message="Insufficient margin",
        )
        assert order.is_rejected
        assert not order.is_working


# ============================================================================
# PositionData Tests
# ============================================================================

class TestPositionData:
    """Tests for PositionData dataclass."""

    def test_from_api(self):
        """Test creating PositionData from API response."""
        data = {
            "accountId": 67890,
            "contractId": "CON.F.US.MES.H26",
            "size": 2,
            "avgPrice": 6050.25,
            "unrealizedPnl": 125.50,
            "realizedPnl": 50.00,
        }
        pos = PositionData.from_api(data)
        assert pos.account_id == 67890
        assert pos.contract_id == "CON.F.US.MES.H26"
        assert pos.size == 2
        assert pos.avg_price == 6050.25

    def test_is_long(self):
        """Test is_long property."""
        pos = PositionData(
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            size=2,
            avg_price=6050.25,
        )
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat
        assert pos.direction == 1

    def test_is_short(self):
        """Test is_short property."""
        pos = PositionData(
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            size=-2,
            avg_price=6050.25,
        )
        assert pos.is_short
        assert not pos.is_long
        assert not pos.is_flat
        assert pos.direction == -1

    def test_is_flat(self):
        """Test is_flat property."""
        pos = PositionData(
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            size=0,
            avg_price=0.0,
        )
        assert pos.is_flat
        assert not pos.is_long
        assert not pos.is_short
        assert pos.direction == 0


# ============================================================================
# AccountInfo Tests
# ============================================================================

class TestAccountInfo:
    """Tests for AccountInfo dataclass."""

    def test_from_api(self):
        """Test creating AccountInfo from API response."""
        data = {
            "id": 67890,
            "name": "TestAccount",
            "balance": 1000.00,
            "availableMargin": 500.00,
            "openPnl": 50.00,
            "closedPnl": 25.00,
        }
        info = AccountInfo.from_api(data)
        assert info.account_id == 67890
        assert info.name == "TestAccount"
        assert info.balance == 1000.00
        assert info.available_margin == 500.00


# ============================================================================
# TopstepXREST Tests
# ============================================================================

class TestTopstepXREST:
    """Tests for TopstepXREST class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        client = MagicMock(spec=TopstepXClient)
        client.default_account_id = 67890
        client.request = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_account_info(self, mock_client):
        """Test getting account info."""
        mock_client.request.return_value = {
            "id": 67890,
            "name": "TestAccount",
            "balance": 1000.00,
            "availableMargin": 500.00,
        }

        rest = TopstepXREST(mock_client)
        info = await rest.get_account_info()

        assert info.account_id == 67890
        assert info.balance == 1000.00
        mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_positions(self, mock_client):
        """Test getting positions."""
        mock_client.request.return_value = [
            {
                "accountId": 67890,
                "contractId": "CON.F.US.MES.H26",
                "size": 2,
                "avgPrice": 6050.25,
            }
        ]

        rest = TopstepXREST(mock_client)
        positions = await rest.get_positions()

        assert len(positions) == 1
        assert positions[0].size == 2

    @pytest.mark.asyncio
    async def test_get_position_found(self, mock_client):
        """Test getting specific position that exists."""
        mock_client.request.return_value = [
            {
                "accountId": 67890,
                "contractId": "CON.F.US.MES.H26",
                "size": 2,
                "avgPrice": 6050.25,
            }
        ]

        rest = TopstepXREST(mock_client)
        position = await rest.get_position("CON.F.US.MES.H26")

        assert position is not None
        assert position.size == 2

    @pytest.mark.asyncio
    async def test_get_position_not_found(self, mock_client):
        """Test getting position that doesn't exist."""
        mock_client.request.return_value = []

        rest = TopstepXREST(mock_client)
        position = await rest.get_position("CON.F.US.MES.H26")

        assert position is None

    @pytest.mark.asyncio
    async def test_get_historical_bars(self, mock_client):
        """Test getting historical bars."""
        mock_client.request.return_value = {
            "bars": [
                {
                    "timestamp": "2026-01-15T14:30:00Z",
                    "open": 6050.25,
                    "high": 6052.00,
                    "low": 6049.50,
                    "close": 6051.75,
                    "volume": 1523,
                }
            ]
        }

        rest = TopstepXREST(mock_client)
        bars = await rest.get_historical_bars(
            contract_id="CON.F.US.MES.H26",
            start_time=datetime(2026, 1, 1),
            end_time=datetime(2026, 1, 15),
        )

        assert len(bars) == 1
        assert bars[0].close == 6051.75

    @pytest.mark.asyncio
    async def test_place_market_order(self, mock_client):
        """Test placing a market order."""
        mock_client.request.return_value = {
            "orderId": "ORD123",
            "accountId": 67890,
            "contractId": "CON.F.US.MES.H26",
            "side": 1,
            "type": 2,
            "size": 1,
            "status": 1,
        }

        rest = TopstepXREST(mock_client)
        order = await rest.place_order(
            contract_id="CON.F.US.MES.H26",
            side=OrderSide.BUY,
            size=1,
            order_type=OrderType.MARKET,
        )

        assert order.order_id == "ORD123"
        assert order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_place_limit_order_requires_price(self, mock_client):
        """Test limit order requires price."""
        rest = TopstepXREST(mock_client)

        with pytest.raises(ValueError, match="Limit price required"):
            await rest.place_order(
                contract_id="CON.F.US.MES.H26",
                side=OrderSide.BUY,
                size=1,
                order_type=OrderType.LIMIT,
            )

    @pytest.mark.asyncio
    async def test_place_stop_order_requires_stop_price(self, mock_client):
        """Test stop order requires stop price."""
        rest = TopstepXREST(mock_client)

        with pytest.raises(ValueError, match="Stop price required"):
            await rest.place_order(
                contract_id="CON.F.US.MES.H26",
                side=OrderSide.SELL,
                size=1,
                order_type=OrderType.STOP,
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_client):
        """Test cancelling an order."""
        mock_client.request.return_value = {"success": True}

        rest = TopstepXREST(mock_client)
        result = await rest.cancel_order("ORD123")

        assert result is True

    @pytest.mark.asyncio
    async def test_flatten_position_long(self, mock_client):
        """Test flattening a long position."""
        mock_client.request.side_effect = [
            # get_position call
            [{
                "accountId": 67890,
                "contractId": "CON.F.US.MES.H26",
                "size": 2,
                "avgPrice": 6050.25,
            }],
            # place_order call
            {
                "orderId": "ORD123",
                "accountId": 67890,
                "contractId": "CON.F.US.MES.H26",
                "side": 2,  # SELL to close long
                "type": 2,
                "size": 2,
                "status": 1,
            }
        ]

        rest = TopstepXREST(mock_client)
        order = await rest.flatten_position("CON.F.US.MES.H26")

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.size == 2

    @pytest.mark.asyncio
    async def test_flatten_position_flat(self, mock_client):
        """Test flattening when already flat."""
        mock_client.request.return_value = []

        rest = TopstepXREST(mock_client)
        order = await rest.flatten_position("CON.F.US.MES.H26")

        assert order is None


# ============================================================================
# Contract ID Helper Tests
# ============================================================================

class TestContractIdHelpers:
    """Tests for contract ID helper functions."""

    def test_get_mes_contract_id(self):
        """Test MES contract ID generation."""
        assert get_mes_contract_id(26, "H") == "CON.F.US.MES.H26"
        assert get_mes_contract_id(26, "M") == "CON.F.US.MES.M26"
        assert get_mes_contract_id(26, "U") == "CON.F.US.MES.U26"
        assert get_mes_contract_id(26, "Z") == "CON.F.US.MES.Z26"

    def test_get_current_mes_contract(self):
        """Test current contract ID generation returns valid format."""
        contract = get_current_mes_contract()
        assert contract.startswith("CON.F.US.MES.")
        assert len(contract) == len("CON.F.US.MES.H26")


# ============================================================================
# Quote Tests
# ============================================================================

class TestQuote:
    """Tests for Quote dataclass."""

    def test_from_signalr(self):
        """Test creating Quote from SignalR message."""
        data = {
            "contractId": "CON.F.US.MES.H26",
            "bid": 6050.00,
            "ask": 6050.25,
            "last": 6050.25,
            "bidSize": 100,
            "askSize": 75,
            "volume": 50000,
        }
        quote = Quote.from_signalr(data)
        assert quote.contract_id == "CON.F.US.MES.H26"
        assert quote.bid == 6050.00
        assert quote.ask == 6050.25
        assert quote.spread == 0.25

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=6050.00,
            ask=6050.50,
            last=6050.25,
        )
        assert quote.mid_price == 6050.25

    def test_spread(self):
        """Test spread calculation."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=6050.00,
            ask=6050.50,
            last=6050.25,
        )
        assert quote.spread == 0.50


# ============================================================================
# OrderFill Tests
# ============================================================================

class TestOrderFill:
    """Tests for OrderFill dataclass."""

    def test_from_signalr(self):
        """Test creating OrderFill from SignalR message."""
        data = {
            "orderId": "ORD123",
            "contractId": "CON.F.US.MES.H26",
            "side": 1,
            "fillPrice": 6050.25,
            "fillSize": 1,
            "remainingSize": 0,
            "isComplete": True,
        }
        fill = OrderFill.from_signalr(data)
        assert fill.order_id == "ORD123"
        assert fill.fill_price == 6050.25
        assert fill.is_complete


# ============================================================================
# PositionUpdate Tests
# ============================================================================

class TestPositionUpdate:
    """Tests for PositionUpdate dataclass."""

    def test_from_signalr(self):
        """Test creating PositionUpdate from SignalR message."""
        data = {
            "accountId": 67890,
            "contractId": "CON.F.US.MES.H26",
            "size": 2,
            "avgPrice": 6050.25,
            "unrealizedPnl": 125.50,
        }
        pos = PositionUpdate.from_signalr(data)
        assert pos.account_id == 67890
        assert pos.size == 2
        assert pos.direction == 1


# ============================================================================
# AccountUpdate Tests
# ============================================================================

class TestAccountUpdate:
    """Tests for AccountUpdate dataclass."""

    def test_from_signalr(self):
        """Test creating AccountUpdate from SignalR message."""
        data = {
            "accountId": 67890,
            "balance": 1000.00,
            "availableMargin": 500.00,
            "openPnl": 50.00,
            "closedPnl": 25.00,
        }
        update = AccountUpdate.from_signalr(data)
        assert update.account_id == 67890
        assert update.balance == 1000.00


# ============================================================================
# WebSocketState Tests
# ============================================================================

class TestWebSocketState:
    """Tests for WebSocketState enum."""

    def test_states(self):
        """Test WebSocket states exist."""
        assert WebSocketState.DISCONNECTED is not None
        assert WebSocketState.CONNECTING is not None
        assert WebSocketState.CONNECTED is not None
        assert WebSocketState.RECONNECTING is not None
        assert WebSocketState.CLOSED is not None


# ============================================================================
# TopstepXWebSocket Tests
# ============================================================================

class TestTopstepXWebSocket:
    """Tests for TopstepXWebSocket class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client for testing."""
        client = MagicMock(spec=TopstepXClient)
        client.access_token = "test_token"
        client.default_account_id = 67890
        client.config = TopstepXConfig()
        return client

    def test_init(self, mock_client):
        """Test WebSocket initialization."""
        ws = TopstepXWebSocket(mock_client)
        assert not ws.is_connected
        assert ws.market_state == WebSocketState.DISCONNECTED
        assert ws.trade_state == WebSocketState.DISCONNECTED

    def test_register_callbacks(self, mock_client):
        """Test registering callbacks."""
        ws = TopstepXWebSocket(mock_client)

        quote_callback = MagicMock()
        fill_callback = MagicMock()
        position_callback = MagicMock()
        account_callback = MagicMock()

        ws.on_quote(quote_callback)
        ws.on_fill(fill_callback)
        ws.on_position(position_callback)
        ws.on_account(account_callback)

        assert quote_callback in ws._quote_callbacks
        assert fill_callback in ws._fill_callbacks
        assert position_callback in ws._position_callbacks
        assert account_callback in ws._account_callbacks

    def test_handle_quote(self, mock_client):
        """Test quote handling."""
        ws = TopstepXWebSocket(mock_client)

        received_quotes = []
        ws.on_quote(lambda q: received_quotes.append(q))

        # Simulate quote data
        data = {
            "contractId": "CON.F.US.MES.H26",
            "bid": 6050.00,
            "ask": 6050.25,
            "last": 6050.25,
        }
        ws._handle_quote(data)

        assert len(received_quotes) == 1
        assert received_quotes[0].contract_id == "CON.F.US.MES.H26"

    def test_handle_fill(self, mock_client):
        """Test fill handling."""
        ws = TopstepXWebSocket(mock_client)

        received_fills = []
        ws.on_fill(lambda f: received_fills.append(f))

        # Simulate fill data
        data = {
            "orderId": "ORD123",
            "contractId": "CON.F.US.MES.H26",
            "side": 1,
            "fillPrice": 6050.25,
            "fillSize": 1,
        }
        ws._handle_fill(data)

        assert len(received_fills) == 1
        assert received_fills[0].order_id == "ORD123"

    def test_handle_position(self, mock_client):
        """Test position update handling."""
        ws = TopstepXWebSocket(mock_client)

        received_positions = []
        ws.on_position(lambda p: received_positions.append(p))

        # Simulate position data
        data = {
            "accountId": 67890,
            "contractId": "CON.F.US.MES.H26",
            "size": 2,
            "avgPrice": 6050.25,
        }
        ws._handle_position(data)

        assert len(received_positions) == 1
        assert received_positions[0].size == 2

    @pytest.mark.asyncio
    async def test_connect_not_authenticated(self, mock_client):
        """Test connect fails when not authenticated."""
        mock_client.access_token = None

        ws = TopstepXWebSocket(mock_client)

        from src.api.topstepx_client import TopstepXConnectionError
        with pytest.raises(TopstepXConnectionError, match="not authenticated"):
            await ws.connect()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_api_error(self):
        """Test TopstepXAPIError creation."""
        error = TopstepXAPIError(
            "Test error",
            status_code=400,
            response={"error": "test"},
        )
        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.response == {"error": "test"}

    def test_auth_error(self):
        """Test TopstepXAuthError is subclass of APIError."""
        error = TopstepXAuthError("Auth failed")
        assert isinstance(error, TopstepXAPIError)

    def test_rate_limit_error(self):
        """Test TopstepXRateLimitError with retry_after."""
        error = TopstepXRateLimitError("Rate limited", retry_after=30.0)
        assert error.retry_after == 30.0


# ============================================================================
# Integration Test Helpers
# ============================================================================

class TestIntegrationHelpers:
    """Tests for integration helpers and edge cases."""

    def test_bar_data_ohlc_validity(self):
        """Test OHLC data validity."""
        bar = BarData(
            timestamp=datetime.now(),
            open=6050.25,
            high=6052.00,
            low=6049.50,
            close=6051.75,
            volume=1523,
        )
        # Standard OHLC validation
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.low <= bar.high

    def test_position_pnl_calculation(self):
        """Test position PnL fields."""
        pos = PositionData(
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            size=2,
            avg_price=6050.25,
            unrealized_pnl=125.50,
            realized_pnl=50.00,
        )
        assert pos.unrealized_pnl == 125.50
        assert pos.realized_pnl == 50.00


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_positions_list(self):
        """Test handling empty positions list."""
        data = []
        positions = [PositionData.from_api(p) for p in data]
        assert len(positions) == 0

    def test_empty_bars_list(self):
        """Test handling empty bars list."""
        data = {"bars": []}
        bars = [BarData.from_api(b) for b in data.get("bars", [])]
        assert len(bars) == 0

    def test_quote_with_zero_volume(self):
        """Test quote with zero volume."""
        quote = Quote(
            contract_id="CON.F.US.MES.H26",
            bid=6050.00,
            ask=6050.25,
            last=6050.25,
            volume=0,
        )
        assert quote.volume == 0

    def test_flat_position_direction(self):
        """Test flat position direction."""
        pos = PositionData(
            account_id=67890,
            contract_id="CON.F.US.MES.H26",
            size=0,
            avg_price=0.0,
        )
        assert pos.direction == 0
        assert pos.is_flat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
