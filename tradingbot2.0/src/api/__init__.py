"""
TopstepX API Integration Module.

This module provides clients for interacting with the TopstepX (ProjectX Gateway) API
for futures trading data and execution.

Key Components:
- TopstepXClient: Base client with authentication and session management
- TopstepXREST: REST API endpoints for orders, positions, and historical data
- TopstepXWebSocket: WebSocket connections for real-time market data and trade updates

Usage:
    from src.api import TopstepXClient, TopstepXREST, TopstepXWebSocket

    # Initialize client
    client = TopstepXClient(username="user", password="pass")
    await client.authenticate()

    # Use REST endpoints
    rest = TopstepXREST(client)
    bars = await rest.get_historical_bars("CON.F.US.MES.H26")

    # Use WebSocket for real-time data
    ws = TopstepXWebSocket(client)
    await ws.connect()
    await ws.subscribe_quotes(["CON.F.US.MES.H26"])

Important Notes:
- TopstepX API has limited historical data (~7-14 days for second bars, ~30 days for minute+)
- Use DataBento for historical training data, TopstepX for live trading only
- Rate limit: ~50 requests per 30 seconds
- Token expires after ~90 minutes, auto-refresh implemented
"""

from src.api.topstepx_client import (
    TopstepXClient,
    TopstepXConfig,
    TopstepXAuthError,
    TopstepXRateLimitError,
    TopstepXAPIError,
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
)
from src.api.topstepx_ws import (
    TopstepXWebSocket,
    Quote,
    OrderFill,
    PositionUpdate,
    WebSocketState,
)

__all__ = [
    # Client
    "TopstepXClient",
    "TopstepXConfig",
    "TopstepXAuthError",
    "TopstepXRateLimitError",
    "TopstepXAPIError",
    # REST
    "TopstepXREST",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeUnit",
    "BarData",
    "OrderResponse",
    "PositionData",
    "AccountInfo",
    # WebSocket
    "TopstepXWebSocket",
    "Quote",
    "OrderFill",
    "PositionUpdate",
    "WebSocketState",
]
