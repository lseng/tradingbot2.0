"""
Extended tests for TopstepX client module.

Tests cover:
- TopstepXClient.authenticate
- TopstepXClient.request with various scenarios
- TopstepXClient._ensure_authenticated
- RateLimiter.wait_and_acquire
- Error handling scenarios
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import aiohttp

from src.api.topstepx_client import (
    TopstepXClient,
    TopstepXConfig,
    TopstepXAPIError,
    TopstepXAuthError,
    TopstepXRateLimitError,
    TopstepXConnectionError,
    RateLimiter,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client_config():
    """Create a test client configuration."""
    return TopstepXConfig(
        base_url="https://test.api.com",
        username="test_user",
        password="test_pass",
        rate_limit_requests=50,
        rate_limit_window=30.0,
    )


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    return session


# =============================================================================
# TopstepXConfig Tests
# =============================================================================

class TestTopstepXConfig:
    """Tests for TopstepXConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TopstepXConfig()

        assert config.base_url == "https://api.topstepx.com"
        assert config.ws_market_url == "wss://rtc.topstepx.com/hubs/market"
        assert config.ws_trade_url == "wss://rtc.topstepx.com/hubs/trade"
        assert config.rate_limit_requests == 50
        assert config.rate_limit_window == 30.0

    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        with patch.dict('os.environ', {}, clear=True):
            config = TopstepXConfig.from_env()

        assert config.base_url == "https://api.topstepx.com"
        assert config.username == ""
        assert config.password == ""

    def test_from_env_with_vars(self):
        """Test from_env with environment variables."""
        env_vars = {
            'TOPSTEPX_BASE_URL': 'https://custom.api.com',
            'TOPSTEPX_USERNAME': 'env_user',
            'TOPSTEPX_PASSWORD': 'env_pass',
            'TOPSTEPX_DEVICE_ID': 'custom_device_id',
        }

        with patch.dict('os.environ', env_vars):
            config = TopstepXConfig.from_env()

        assert config.base_url == "https://custom.api.com"
        assert config.username == "env_user"
        assert config.password == "env_pass"
        assert config.device_id == "custom_device_id"


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquire within rate limit."""
        limiter = RateLimiter(max_requests=10, window_seconds=30.0)

        wait_time = await limiter.acquire()

        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self):
        """Test acquire when exceeding rate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=30.0)

        # Make 2 requests (within limit)
        await limiter.acquire()
        await limiter.acquire()

        # Third request should return wait time
        wait_time = await limiter.acquire()

        # Should need to wait (or be 0 if enough time passed)
        assert isinstance(wait_time, float)

    @pytest.mark.asyncio
    async def test_wait_and_acquire(self):
        """Test wait_and_acquire waits and acquires."""
        limiter = RateLimiter(max_requests=100, window_seconds=30.0)

        # Should not wait with high limit
        await limiter.wait_and_acquire()

        assert len(limiter._request_times) > 0

    def test_reset(self):
        """Test reset clears request times."""
        limiter = RateLimiter(max_requests=10, window_seconds=30.0)
        limiter._request_times = [1.0, 2.0, 3.0]

        limiter.reset()

        assert len(limiter._request_times) == 0


# =============================================================================
# TopstepXClient Initialization Tests
# =============================================================================

class TestTopstepXClientInit:
    """Tests for TopstepXClient initialization."""

    def test_init_with_credentials(self):
        """Test initialization with username/password."""
        client = TopstepXClient(
            username="test_user",
            password="test_pass",
        )

        assert client.config.username == "test_user"
        assert client.config.password == "test_pass"

    def test_init_with_config(self, client_config):
        """Test initialization with config object."""
        client = TopstepXClient(config=client_config)

        assert client.config.base_url == "https://test.api.com"

    def test_init_credentials_override_config(self, client_config):
        """Test credentials override config values."""
        client = TopstepXClient(
            username="override_user",
            password="override_pass",
            config=client_config,
        )

        assert client.config.username == "override_user"
        assert client.config.password == "override_pass"

    def test_initial_state(self, client_config):
        """Test initial authentication state."""
        client = TopstepXClient(config=client_config)

        assert client._access_token is None
        assert client._token_expiry is None
        assert client._user_id is None
        assert client._accounts == []
        assert client.is_authenticated is False


# =============================================================================
# TopstepXClient Properties Tests
# =============================================================================

class TestTopstepXClientProperties:
    """Tests for TopstepXClient properties."""

    def test_is_authenticated_no_token(self, client_config):
        """Test is_authenticated with no token."""
        client = TopstepXClient(config=client_config)

        assert client.is_authenticated is False

    def test_is_authenticated_expired_token(self, client_config):
        """Test is_authenticated with expired token."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() - timedelta(hours=1)

        assert client.is_authenticated is False

    def test_is_authenticated_valid_token(self, client_config):
        """Test is_authenticated with valid token."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        assert client.is_authenticated is True

    def test_access_token_property(self, client_config):
        """Test access_token property."""
        client = TopstepXClient(config=client_config)
        client._access_token = "my_token"

        assert client.access_token == "my_token"

    def test_user_id_property(self, client_config):
        """Test user_id property."""
        client = TopstepXClient(config=client_config)
        client._user_id = 12345

        assert client.user_id == 12345

    def test_accounts_property_returns_copy(self, client_config):
        """Test accounts property returns copy."""
        client = TopstepXClient(config=client_config)
        client._accounts = [{"id": 1}]

        accounts = client.accounts
        accounts.append({"id": 2})

        # Original should not be modified
        assert len(client._accounts) == 1

    def test_default_account_id_with_accounts(self, client_config):
        """Test default_account_id with accounts."""
        client = TopstepXClient(config=client_config)
        client._accounts = [{"id": 12345}, {"id": 67890}]

        assert client.default_account_id == 12345

    def test_default_account_id_no_accounts(self, client_config):
        """Test default_account_id with no accounts."""
        client = TopstepXClient(config=client_config)

        assert client.default_account_id is None


# =============================================================================
# TopstepXClient.authenticate Tests
# =============================================================================

class TestTopstepXClientAuthenticate:
    """Tests for TopstepXClient.authenticate method."""

    @pytest.mark.asyncio
    async def test_authenticate_missing_credentials(self, client_config):
        """Test authenticate raises error with missing credentials."""
        client_config.username = ""
        client_config.password = ""
        client = TopstepXClient(config=client_config)

        with pytest.raises(ValueError, match="Username and password"):
            await client.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_success(self, client_config):
        """Test successful authentication."""
        client = TopstepXClient(config=client_config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "accessToken": "new_token",
            "userId": 12345,
            "accounts": [{"id": 1}],
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            result = await client.authenticate()

        assert result is True
        assert client._access_token == "new_token"
        assert client._user_id == 12345
        assert len(client._accounts) == 1

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, client_config):
        """Test authentication failure."""
        client = TopstepXClient(config=client_config)

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={
            "success": False,
            "errorMessage": "Invalid credentials",
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            with pytest.raises(TopstepXAuthError, match="Invalid credentials"):
                await client.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_connection_error(self, client_config):
        """Test authentication with connection error."""
        client = TopstepXClient(config=client_config)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            with pytest.raises(TopstepXConnectionError, match="Connection error"):
                await client.authenticate()


# =============================================================================
# TopstepXClient.request Tests
# =============================================================================

class TestTopstepXClientRequest:
    """Tests for TopstepXClient.request method."""

    @pytest.mark.asyncio
    async def test_request_success(self, client_config):
        """Test successful request."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            result = await client.request("GET", "/api/test")

        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_request_rate_limited(self, client_config):
        """Test request handles rate limiting."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        # First response is rate limited, second succeeds
        mock_response_limited = AsyncMock()
        mock_response_limited.status = 429
        mock_response_limited.headers = {"Retry-After": "0.1"}
        mock_response_limited.json = AsyncMock(return_value={})
        mock_response_limited.__aenter__ = AsyncMock(return_value=mock_response_limited)
        mock_response_limited.__aexit__ = AsyncMock(return_value=None)

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=[mock_response_limited, mock_response_success])
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            result = await client.request("GET", "/api/test")

        assert result == {"data": "success"}

    @pytest.mark.asyncio
    async def test_request_rate_limited_no_retry(self, client_config):
        """Test request rate limited without retry."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "30"}
        mock_response.json = AsyncMock(return_value={})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            with pytest.raises(TopstepXRateLimitError):
                await client.request("GET", "/api/test", retry=False)

    @pytest.mark.asyncio
    async def test_request_auth_expired_reauth(self, client_config):
        """Test request re-authenticates on 401."""
        client = TopstepXClient(config=client_config)
        client._access_token = "old_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        # First response is 401, second succeeds after re-auth
        mock_response_401 = AsyncMock()
        mock_response_401.status = 401
        mock_response_401.json = AsyncMock(return_value={})
        mock_response_401.__aenter__ = AsyncMock(return_value=mock_response_401)
        mock_response_401.__aexit__ = AsyncMock(return_value=None)

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=[mock_response_401, mock_response_success])
        mock_session.closed = False

        # Mock authenticate to set new token
        async def mock_authenticate():
            client._access_token = "new_token"
            return True

        with patch.object(client, '_get_session', return_value=mock_session):
            with patch.object(client, 'authenticate', side_effect=mock_authenticate):
                result = await client.request("GET", "/api/test")

        assert result == {"data": "success"}

    @pytest.mark.asyncio
    async def test_request_server_error_retry(self, client_config):
        """Test request retries on server error."""
        client_config.max_retries = 2
        client_config.initial_backoff = 0.01
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        # First response is 500, second succeeds
        mock_response_500 = AsyncMock()
        mock_response_500.status = 500
        mock_response_500.json = AsyncMock(return_value={"errorMessage": "Server error"})
        mock_response_500.__aenter__ = AsyncMock(return_value=mock_response_500)
        mock_response_500.__aexit__ = AsyncMock(return_value=None)

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=[mock_response_500, mock_response_success])
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            result = await client.request("GET", "/api/test")

        assert result == {"data": "success"}

    @pytest.mark.asyncio
    async def test_request_client_error(self, client_config):
        """Test request handles client error (4xx)."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"errorMessage": "Bad request"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            with pytest.raises(TopstepXAPIError, match="Bad request"):
                await client.request("GET", "/api/test")

    @pytest.mark.asyncio
    async def test_request_api_failure_flag(self, client_config):
        """Test request handles API success=False response."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "success": False,
            "errorMessage": "API failure",
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            with pytest.raises(TopstepXAPIError, match="API failure"):
                await client.request("GET", "/api/test")

    @pytest.mark.asyncio
    async def test_request_connection_error_retry(self, client_config):
        """Test request retries on connection error."""
        client_config.max_retries = 2
        client_config.initial_backoff = 0.01
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"data": "success"})
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=[
            aiohttp.ClientError("Connection failed"),
            mock_response_success,
        ])
        mock_session.closed = False

        with patch.object(client, '_get_session', return_value=mock_session):
            result = await client.request("GET", "/api/test")

        assert result == {"data": "success"}


# =============================================================================
# TopstepXClient._ensure_authenticated Tests
# =============================================================================

class TestTopstepXClientEnsureAuthenticated:
    """Tests for TopstepXClient._ensure_authenticated method."""

    @pytest.mark.asyncio
    async def test_ensure_authenticated_not_authenticated(self, client_config):
        """Test _ensure_authenticated calls authenticate when not authenticated."""
        client = TopstepXClient(config=client_config)

        # 10C.7 FIX: _ensure_authenticated now calls _authenticate_internal directly
        mock_authenticate = AsyncMock()
        with patch.object(client, '_authenticate_internal', mock_authenticate):
            await client._ensure_authenticated()

        mock_authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_authenticated_token_expiring(self, client_config):
        """Test _ensure_authenticated refreshes expiring token."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        # Token expires in 5 minutes (less than margin)
        client._token_expiry = datetime.utcnow() + timedelta(minutes=5)
        client.config.token_refresh_margin = 600  # 10 minutes

        # 10C.7 FIX: _ensure_authenticated now calls _authenticate_internal directly
        mock_authenticate = AsyncMock()
        with patch.object(client, '_authenticate_internal', mock_authenticate):
            await client._ensure_authenticated()

        mock_authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_authenticated_token_valid(self, client_config):
        """Test _ensure_authenticated does nothing with valid token."""
        client = TopstepXClient(config=client_config)
        client._access_token = "test_token"
        client._token_expiry = datetime.utcnow() + timedelta(hours=1)

        # 10C.7 FIX: _ensure_authenticated now calls _authenticate_internal directly
        mock_authenticate = AsyncMock()
        with patch.object(client, '_authenticate_internal', mock_authenticate):
            await client._ensure_authenticated()

        mock_authenticate.assert_not_called()


# =============================================================================
# TopstepXClient Context Manager Tests
# =============================================================================

class TestTopstepXClientContextManager:
    """Tests for TopstepXClient context manager."""

    @pytest.mark.asyncio
    async def test_aenter(self, client_config):
        """Test async context manager entry."""
        client = TopstepXClient(config=client_config)

        mock_authenticate = AsyncMock()
        with patch.object(client, 'authenticate', mock_authenticate):
            result = await client.__aenter__()

        assert result is client
        mock_authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit(self, client_config):
        """Test async context manager exit."""
        client = TopstepXClient(config=client_config)

        mock_close = AsyncMock()
        with patch.object(client, 'close', mock_close):
            await client.__aexit__(None, None, None)

        mock_close.assert_called_once()


# =============================================================================
# TopstepXClient.close Tests
# =============================================================================

class TestTopstepXClientClose:
    """Tests for TopstepXClient.close method."""

    @pytest.mark.asyncio
    async def test_close_with_session(self, client_config, mock_session):
        """Test close closes session."""
        client = TopstepXClient(config=client_config)
        client._session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_no_session(self, client_config):
        """Test close does nothing with no session."""
        client = TopstepXClient(config=client_config)
        client._session = None

        # Should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_close_already_closed_session(self, client_config, mock_session):
        """Test close with already closed session."""
        client = TopstepXClient(config=client_config)
        mock_session.closed = True
        client._session = mock_session

        # Should not try to close again
        await client.close()

        mock_session.close.assert_not_called()


# =============================================================================
# Error Classes Tests
# =============================================================================

class TestErrorClasses:
    """Tests for error classes."""

    def test_api_error(self):
        """Test TopstepXAPIError."""
        error = TopstepXAPIError("Test error", status_code=400, response={"key": "value"})

        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.response == {"key": "value"}
        assert str(error) == "Test error"

    def test_auth_error(self):
        """Test TopstepXAuthError."""
        error = TopstepXAuthError("Auth failed")

        assert isinstance(error, TopstepXAPIError)
        assert error.message == "Auth failed"

    def test_rate_limit_error(self):
        """Test TopstepXRateLimitError."""
        error = TopstepXRateLimitError("Rate limited", retry_after=30.0)

        assert isinstance(error, TopstepXAPIError)
        assert error.retry_after == 30.0

    def test_connection_error(self):
        """Test TopstepXConnectionError."""
        error = TopstepXConnectionError("Connection failed")

        assert isinstance(error, TopstepXAPIError)
        assert error.message == "Connection failed"
