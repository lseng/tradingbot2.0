"""
TopstepX Base Client with Authentication and Session Management.

This module provides the core client for interacting with the TopstepX API,
handling authentication, token refresh, rate limiting, and error handling.

Key Features:
- JWT token-based authentication with automatic refresh
- Rate limiting (50 requests per 30 seconds)
- Exponential backoff on errors
- Thread-safe session management
- Request/response logging

API Reference: https://gateway.docs.projectx.com/docs/intro
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class TopstepXAPIError(Exception):
    """Base exception for TopstepX API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class TopstepXAuthError(TopstepXAPIError):
    """Authentication failed."""
    pass


class TopstepXRateLimitError(TopstepXAPIError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float = 30.0):
        super().__init__(message)
        self.retry_after = retry_after


class TopstepXConnectionError(TopstepXAPIError):
    """Connection error."""
    pass


@dataclass
class TopstepXConfig:
    """Configuration for TopstepX API client.

    Attributes:
        base_url: API base URL
        ws_market_url: WebSocket market hub URL
        ws_trade_url: WebSocket trade hub URL
        username: Login username (email)
        password: Login password
        device_id: Unique device identifier (auto-generated if not provided)
        app_id: Application identifier
        app_version: Application version
        rate_limit_requests: Max requests per rate limit window
        rate_limit_window: Rate limit window in seconds
        token_refresh_margin: Seconds before expiry to trigger refresh
        request_timeout: Default request timeout in seconds
        max_retries: Maximum retry attempts on failure
        initial_backoff: Initial backoff delay in seconds
        max_backoff: Maximum backoff delay in seconds
    """
    base_url: str = "https://api.topstepx.com"
    ws_market_url: str = "wss://rtc.topstepx.com/hubs/market"
    ws_trade_url: str = "wss://rtc.topstepx.com/hubs/trade"
    username: str = ""
    password: str = ""
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    app_id: str = "tradingbot2.0"
    app_version: str = "1.0.0"
    rate_limit_requests: int = 50
    rate_limit_window: float = 30.0
    token_refresh_margin: float = 600.0  # Refresh 10 minutes before expiry
    request_timeout: float = 30.0
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 30.0

    @classmethod
    def from_env(cls) -> "TopstepXConfig":
        """Create config from environment variables.

        Environment variables:
            TOPSTEPX_USERNAME: Login username
            TOPSTEPX_PASSWORD: Login password
            TOPSTEPX_BASE_URL: Optional custom base URL
            TOPSTEPX_DEVICE_ID: Optional device ID
        """
        return cls(
            base_url=os.getenv("TOPSTEPX_BASE_URL", "https://api.topstepx.com"),
            username=os.getenv("TOPSTEPX_USERNAME", ""),
            password=os.getenv("TOPSTEPX_PASSWORD", ""),
            device_id=os.getenv("TOPSTEPX_DEVICE_ID", str(uuid.uuid4())),
        )


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Implements a sliding window rate limiter to enforce API rate limits.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire permission to make a request.

        Returns:
            Wait time in seconds before request should be made (0 if immediate)

        Raises:
            TopstepXRateLimitError: If rate limit would be exceeded and wait is too long
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Remove old requests outside window
            self._request_times = [t for t in self._request_times if t > cutoff]

            if len(self._request_times) < self.max_requests:
                # Can make request immediately
                self._request_times.append(now)
                return 0.0

            # Need to wait for oldest request to expire
            oldest = min(self._request_times)
            wait_time = oldest + self.window_seconds - now

            if wait_time > 0:
                return wait_time

            self._request_times.append(now)
            return 0.0

    async def wait_and_acquire(self) -> None:
        """Wait if necessary and acquire permission to make a request."""
        wait_time = await self.acquire()
        if wait_time > 0:
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            # Re-acquire after waiting
            await self.wait_and_acquire()

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._request_times.clear()


class TopstepXClient:
    """Base client for TopstepX API.

    Handles authentication, session management, rate limiting, and error handling.
    All HTTP requests should go through this client.

    Example:
        client = TopstepXClient(
            username="user@example.com",
            password="password123"
        )
        await client.authenticate()

        # Make authenticated request
        response = await client.request("GET", "/api/Account/info")

        # Clean up
        await client.close()
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[TopstepXConfig] = None,
    ):
        """Initialize the client.

        Args:
            username: Login username (overrides config)
            password: Login password (overrides config)
            config: Configuration object (uses env vars if not provided)
        """
        self.config = config or TopstepXConfig.from_env()

        # Override credentials if provided
        if username:
            self.config.username = username
        if password:
            self.config.password = password

        # Authentication state
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._user_id: Optional[int] = None
        self._accounts: list[dict] = []

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self._rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window
        )

        # Authentication lock to prevent concurrent auth attempts
        self._auth_lock = asyncio.Lock()

    @property
    def is_authenticated(self) -> bool:
        """Check if client is authenticated with valid token."""
        if not self._access_token or not self._token_expiry:
            return False
        return datetime.utcnow() < self._token_expiry

    @property
    def access_token(self) -> Optional[str]:
        """Get current access token."""
        return self._access_token

    @property
    def user_id(self) -> Optional[int]:
        """Get authenticated user ID."""
        return self._user_id

    @property
    def accounts(self) -> list[dict]:
        """Get list of trading accounts."""
        return self._accounts.copy()

    @property
    def default_account_id(self) -> Optional[int]:
        """Get default (first) account ID."""
        if self._accounts:
            return self._accounts[0].get("id")
        return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def authenticate(self) -> bool:
        """Authenticate with the TopstepX API.

        Obtains a JWT access token that expires after ~90 minutes.

        Returns:
            True if authentication successful

        Raises:
            TopstepXAuthError: If authentication fails
            ValueError: If credentials not configured
        """
        if not self.config.username or not self.config.password:
            raise ValueError("Username and password must be configured")

        async with self._auth_lock:
            session = await self._get_session()

            payload = {
                "userName": self.config.username,
                "password": self.config.password,
                "deviceId": self.config.device_id,
                "appId": self.config.app_id,
                "appVersion": self.config.app_version,
            }

            url = f"{self.config.base_url}/api/Auth/loginKey"

            try:
                async with session.post(url, json=payload) as response:
                    data = await response.json()

                    if response.status != 200 or not data.get("success", True):
                        error_msg = data.get("errorMessage", "Authentication failed")
                        logger.error(f"Authentication failed: {error_msg}")
                        raise TopstepXAuthError(
                            error_msg,
                            status_code=response.status,
                            response=data
                        )

                    self._access_token = data.get("accessToken")
                    self._user_id = data.get("userId")
                    self._accounts = data.get("accounts", [])

                    # Token expires in ~90 minutes, set expiry with margin
                    self._token_expiry = datetime.utcnow() + timedelta(minutes=90)

                    logger.info(f"Authenticated as user {self._user_id} with {len(self._accounts)} accounts")
                    return True

            except aiohttp.ClientError as e:
                logger.error(f"Connection error during authentication: {e}")
                raise TopstepXConnectionError(f"Connection error: {e}")

    async def _ensure_authenticated(self) -> None:
        """Ensure client is authenticated, refreshing token if needed."""
        if not self.is_authenticated:
            await self.authenticate()
        elif self._token_expiry:
            # Refresh token if close to expiry
            margin = timedelta(seconds=self.config.token_refresh_margin)
            if datetime.utcnow() + margin >= self._token_expiry:
                logger.info("Token expiring soon, refreshing...")
                await self.authenticate()

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        authenticated: bool = True,
        retry: bool = True,
    ) -> dict:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/api/Account/info")
            params: Query parameters
            json: JSON body for POST/PUT
            authenticated: Whether to include auth header
            retry: Whether to retry on transient errors

        Returns:
            JSON response data

        Raises:
            TopstepXAPIError: On API error
            TopstepXAuthError: On authentication error
            TopstepXRateLimitError: If rate limited
        """
        if authenticated:
            await self._ensure_authenticated()

        # Apply rate limiting
        await self._rate_limiter.wait_and_acquire()

        session = await self._get_session()
        url = f"{self.config.base_url}{endpoint}"

        headers = {}
        if authenticated and self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        backoff = self.config.initial_backoff
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries if retry else 1):
            try:
                async with session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                ) as response:
                    data = await response.json()

                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = float(response.headers.get("Retry-After", 30))
                        logger.warning(f"Rate limited, retry after {retry_after}s")
                        if retry:
                            await asyncio.sleep(retry_after)
                            continue
                        raise TopstepXRateLimitError(
                            "Rate limit exceeded",
                            retry_after=retry_after
                        )

                    # Handle auth errors
                    if response.status == 401:
                        logger.warning("Authentication expired, re-authenticating...")
                        await self.authenticate()
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        continue

                    # Handle server errors
                    if response.status >= 500:
                        error_msg = data.get("errorMessage", f"Server error {response.status}")
                        if retry and attempt < self.config.max_retries - 1:
                            logger.warning(f"Server error, retrying in {backoff}s: {error_msg}")
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, self.config.max_backoff)
                            continue
                        raise TopstepXAPIError(error_msg, status_code=response.status, response=data)

                    # Handle client errors
                    if response.status >= 400:
                        error_msg = data.get("errorMessage", f"Client error {response.status}")
                        raise TopstepXAPIError(error_msg, status_code=response.status, response=data)

                    # Check for API-level success flag
                    if isinstance(data, dict) and data.get("success") is False:
                        error_msg = data.get("errorMessage", "Request failed")
                        raise TopstepXAPIError(error_msg, response=data)

                    return data

            except aiohttp.ClientError as e:
                last_error = TopstepXConnectionError(f"Connection error: {e}")
                if retry and attempt < self.config.max_retries - 1:
                    logger.warning(f"Connection error, retrying in {backoff}s: {e}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.config.max_backoff)
                    continue
                raise last_error

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise TopstepXAPIError("Request failed after retries")

    async def __aenter__(self) -> "TopstepXClient":
        """Async context manager entry."""
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
