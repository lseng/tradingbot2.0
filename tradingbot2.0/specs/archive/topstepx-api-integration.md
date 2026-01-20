# TopstepX API Integration Specification

## Overview

Integration with TopstepX (ProjectX Gateway) API for futures trading data and execution. This spec covers data acquisition capabilities for ML model training and live trading.

## API Reference

- **Official Docs**: https://gateway.docs.projectx.com/docs/intro
- **Base URL**: `https://api.topstepx.com`
- **WebSocket Market Hub**: `wss://rtc.topstepx.com/hubs/market`
- **WebSocket Trade Hub**: `wss://rtc.topstepx.com/hubs/trade`

---

## Historical Data Capabilities

### Endpoint
```
POST /api/History/retrieveBars
```

### Time Units Available
| Unit | Value | Description |
|------|-------|-------------|
| Second | 1 | 1-second bars |
| Minute | 2 | 1-minute bars (default) |
| Hour | 3 | Hourly bars |
| Day | 4 | Daily bars |
| Week | 5 | Weekly bars |
| Month | 6 | Monthly bars |

### Granularity
- `unitNumber` parameter allows custom intervals
- Examples:
  - `unit=2, unitNumber=5` → 5-minute bars
  - `unit=2, unitNumber=15` → 15-minute bars
  - `unit=3, unitNumber=4` → 4-hour bars

### Limits
- **Max bars per request**: 20,000
- **Rate limit**: ~50 requests per 30 seconds
- **Historical depth**: Unknown limit (needs testing)

### Request Format
```json
{
  "contractId": "CON.F.US.MES.H26",
  "live": false,
  "startTime": "2025-01-01T00:00:00.000Z",
  "endTime": "2025-01-15T00:00:00.000Z",
  "unit": 2,
  "unitNumber": 1,
  "limit": 20000
}
```

### Response Format
```json
{
  "success": true,
  "bars": [
    {
      "timestamp": "2025-01-15T14:30:00Z",
      "open": 6050.25,
      "high": 6052.00,
      "low": 6049.50,
      "close": 6051.75,
      "volume": 1523
    }
  ]
}
```

---

## Real-Time Data (WebSocket)

### Market Hub
- Protocol: SignalR
- URL: `wss://rtc.topstepx.com/hubs/market`

#### Subscribe to Quotes
```javascript
connection.invoke("SubscribeQuotes", [contractId]);
connection.on("GotQuote", (quote) => {
  // quote.bid, quote.ask, quote.last, quote.volume
});
```

### Trade Hub
- URL: `wss://rtc.topstepx.com/hubs/trade`
- Order fill notifications
- Position updates

---

## Contract IDs

### Format
```
CON.F.US.{SYMBOL}.{EXPIRY}
```

### Common Contracts
| Symbol | Description | Tick Size | Tick Value |
|--------|-------------|-----------|------------|
| MES | Micro E-mini S&P 500 | 0.25 | $1.25 |
| ES | E-mini S&P 500 | 0.25 | $12.50 |
| MNQ | Micro E-mini Nasdaq | 0.25 | $0.50 |
| NQ | E-mini Nasdaq | 0.25 | $5.00 |

### Expiry Codes
- H = March
- M = June
- U = September
- Z = December

---

## Authentication

### Login
```
POST /api/Auth/loginKey
```

```json
{
  "userName": "email@example.com",
  "password": "your_password",
  "deviceId": "unique_device_id",
  "appId": "tradingbot2.0",
  "appVersion": "1.0.0"
}
```

### Response
```json
{
  "accessToken": "jwt_token",
  "userId": 12345,
  "accounts": [
    {"id": 67890, "name": "150KTC-V2-XXXXXX"}
  ]
}
```

### Token Usage
```
Authorization: Bearer {accessToken}
```

### Token Expiry
- Tokens expire after ~90 minutes
- Implement refresh logic

---

## Data Requirements for ML

### Historical Data Limitations (Discovered)

| Data Type | TopstepX Max Depth | Notes |
|-----------|-------------------|-------|
| Second | ~7-14 days | Not suitable for backtesting |
| Minute | ~30 days | Current contract only |
| Hour/Day | ~30 days | Current contract only |

**Critical**: TopstepX API is for LIVE TRADING only, not historical backtesting.

### Training Data Strategy

**Use DataBento for historical data** (already configured):
- Tick-level data going back years
- Continuous contracts available
- Professional-grade quality

**Use TopstepX for**:
- Live trading execution
- Real-time quotes
- Recent data (< 7 days)

### Training Data Needs
1. **Historical bars** (via DataBento)
   - Minimum: 1 year of 1-minute data
   - Preferred: 2-3 years for regime testing

2. **Multiple timeframes**
   - 1-minute (primary)
   - 5-minute
   - 15-minute
   - 1-hour
   - Daily

3. **Session awareness**
   - RTH (Regular Trading Hours): 9:30 AM - 4:00 PM ET
   - ETH (Extended Trading Hours): 6:00 PM - 5:00 PM ET next day

### Data Pipeline Requirements
- [ ] Fetch historical data from DataBento (for training)
- [ ] Fetch real-time data from TopstepX (for live trading)
- [ ] Convert timestamps to consistent timezone
- [ ] Calculate OHLCV aggregations for higher timeframes
- [ ] Store in Parquet format

---

## Order Execution

### Place Order
```
POST /api/Order/place
```

```json
{
  "accountId": 12345,
  "contractId": "CON.F.US.MES.H26",
  "type": 2,
  "side": 1,
  "size": 1,
  "price": null,
  "stopPrice": null,
  "customTag": "ML_BOT_v1"
}
```

### Order Types
| Type | Value |
|------|-------|
| Limit | 1 |
| Market | 2 |
| Stop | 3 |
| StopLimit | 4 |

### Order Sides
| Side | Value |
|------|-------|
| Buy | 1 |
| Sell | 2 |

---

## Important Limitations

1. **No bracket orders** - Must place entry, stop, target separately
2. **Position netting** - All positions netted per contract
3. **Rate limits** - ~50 req/30s for REST
4. **WebSocket sessions** - Max 2 concurrent sessions
5. **No tick data** - Only bar data available via REST

---

## Acceptance Criteria

### Data Fetching
- [ ] Successfully authenticate with TopstepX API
- [ ] Fetch 1-minute historical bars for MES
- [ ] Determine maximum historical depth available
- [ ] Handle rate limits with exponential backoff
- [ ] Store data in Parquet format

### Real-Time Integration
- [ ] Connect to Market Hub WebSocket
- [ ] Subscribe to MES quotes
- [ ] Handle reconnection on disconnect

### Order Execution (Paper Trading)
- [ ] Place market orders
- [ ] Place limit orders
- [ ] Cancel pending orders
- [ ] Track order fills
