# Live Trading Execution Specification

## Overview

Live trading system that connects to TopstepX API, receives real-time market data, executes trades based on ML model predictions, and manages positions with strict risk controls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Trading Bot                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Data Stream │→ │ ML Model    │→ │ Signal Generator    │  │
│  │ (WebSocket) │  │ (Inference) │  │ (Entry/Exit Logic)  │  │
│  └─────────────┘  └─────────────┘  └──────────┬──────────┘  │
│                                                │             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────▼──────────┐  │
│  │ Position    │← │ Order       │← │ Risk Manager        │  │
│  │ Manager     │  │ Executor    │  │ (Size/Limits)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   TopstepX API          │
              │   - REST (Orders)       │
              │   - WebSocket (Data)    │
              └─────────────────────────┘
```

---

## Components

### 1. Market Data Stream

#### WebSocket Connection
```python
class MarketDataStream:
    """
    Connects to TopstepX Market Hub for real-time quotes.
    URL: wss://rtc.topstepx.com/hubs/market
    """

    async def connect(self):
        # SignalR connection
        pass

    async def subscribe(self, contract_id: str):
        # Subscribe to quotes
        pass

    def on_quote(self, callback: Callable):
        # Quote handler
        pass

    def on_bar(self, callback: Callable):
        # 1-second bar aggregation
        pass
```

#### Data Processing
- Aggregate ticks into 1-second bars
- Calculate real-time features
- Feed to ML model for inference

### 2. ML Model Inference

#### Model Loading
```python
class ScalpingModel:
    def __init__(self, model_path: str, config_path: str):
        self.model = torch.load(model_path)
        self.model.eval()
        self.scaler = load(scaler_path)

    def predict(self, features: np.ndarray) -> Prediction:
        with torch.no_grad():
            scaled = self.scaler.transform(features)
            tensor = torch.tensor(scaled, dtype=torch.float32)
            output = self.model(tensor)
            return self.parse_output(output)
```

#### Inference Requirements
- Latency: < 10ms per prediction
- Batch size: 1 (real-time)
- GPU optional (CPU sufficient for inference)

### 3. Signal Generator

#### Signal Types
| Signal | Action |
|--------|--------|
| LONG_ENTRY | Open long position |
| SHORT_ENTRY | Open short position |
| EXIT_LONG | Close long position |
| EXIT_SHORT | Close short position |
| REVERSE_TO_LONG | Close short, open long |
| REVERSE_TO_SHORT | Close long, open short |
| FLATTEN | Close all positions |

#### Signal Generation Logic
```python
def generate_signal(
    prediction: Prediction,
    current_position: Position,
    risk_manager: RiskManager
) -> Optional[Signal]:
    # Check if trading allowed
    if not risk_manager.can_trade():
        return None

    # Check confidence threshold
    if prediction.confidence < config.min_confidence:
        return None

    # Generate entry signals
    if current_position.is_flat():
        if prediction.direction == 1:
            return Signal.LONG_ENTRY
        elif prediction.direction == -1:
            return Signal.SHORT_ENTRY

    # Generate exit signals
    elif current_position.is_long():
        if prediction.direction == -1 and prediction.confidence > 0.75:
            return Signal.REVERSE_TO_SHORT
        elif should_exit_long(prediction):
            return Signal.EXIT_LONG

    # ... similar for short positions

    return None
```

### 4. Risk Manager

See `risk-management.md` for full specification.

#### Live Trading Checks
```python
class LiveRiskManager:
    def can_trade(self) -> bool:
        return all([
            self.daily_loss < self.max_daily_loss,
            self.account_balance > self.min_balance,
            self.consecutive_losses < self.max_consecutive,
            not self.is_eod_flatten_time(),
            not self.is_circuit_breaker_active(),
        ])

    def calculate_position_size(self, signal: Signal) -> int:
        # Dynamic sizing based on risk rules
        pass

    def get_stop_price(self, entry: float, direction: int) -> float:
        # Calculate stop based on ATR/config
        pass

    def get_target_price(self, entry: float, direction: int) -> float:
        # Calculate target based on R:R ratio
        pass
```

### 5. Order Executor

#### Order Placement
```python
class OrderExecutor:
    def __init__(self, api_client: TopstepXClient):
        self.client = api_client

    async def place_entry(
        self,
        contract_id: str,
        direction: int,
        size: int,
        stop_price: float,
        target_price: float
    ) -> EntryResult:
        # 1. Place entry order (market)
        entry_order = await self.client.place_order(
            contract_id=contract_id,
            side=Side.BUY if direction == 1 else Side.SELL,
            size=size,
            order_type=OrderType.MARKET,
            custom_tag="SCALPER_ENTRY"
        )

        # 2. Wait for fill
        fill = await self.wait_for_fill(entry_order.order_id)

        # 3. Place stop loss order
        stop_order = await self.client.place_order(
            contract_id=contract_id,
            side=Side.SELL if direction == 1 else Side.BUY,
            size=size,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            custom_tag="SCALPER_STOP"
        )

        # 4. Place take profit order
        target_order = await self.client.place_order(
            contract_id=contract_id,
            side=Side.SELL if direction == 1 else Side.BUY,
            size=size,
            order_type=OrderType.LIMIT,
            price=target_price,
            custom_tag="SCALPER_TARGET"
        )

        return EntryResult(
            entry_fill=fill,
            stop_order=stop_order,
            target_order=target_order
        )
```

#### Order Management
- Track all open orders
- Cancel orphaned orders on exit
- Handle partial fills
- Manage OCO (one-cancels-other) manually

### 6. Position Manager

#### Position Tracking
```python
@dataclass
class Position:
    contract_id: str
    direction: int  # 1=long, -1=short, 0=flat
    size: int
    entry_price: float
    entry_time: datetime
    stop_price: float
    target_price: float
    unrealized_pnl: float
    stop_order_id: str
    target_order_id: str

class PositionManager:
    def __init__(self):
        self.position: Optional[Position] = None

    def is_flat(self) -> bool:
        return self.position is None

    def update_from_fill(self, fill: Fill):
        # Update position state
        pass

    def calculate_pnl(self, current_price: float) -> float:
        if self.position is None:
            return 0.0
        direction = self.position.direction
        diff = current_price - self.position.entry_price
        return diff * direction * self.position.size * 5.0  # MES point value
```

---

## Trading Session Flow

### Startup Sequence
```
1. Load configuration
2. Load ML model
3. Initialize risk manager
4. Connect to TopstepX API
5. Authenticate
6. Subscribe to market data
7. Sync existing positions (if any)
8. Begin trading loop
```

### Main Loop
```python
async def trading_loop():
    while market_is_open():
        # 1. Receive market data
        bar = await data_stream.next_bar()

        # 2. Update features
        features = feature_engine.update(bar)

        # 3. Get ML prediction
        prediction = model.predict(features)

        # 4. Generate signal
        signal = signal_generator.generate(prediction, position_manager.position)

        # 5. Execute if signal
        if signal:
            await execute_signal(signal)

        # 6. Check EOD flatten
        if risk_manager.is_eod_flatten_time():
            await flatten_all_positions()

        # 7. Log state
        logger.log_tick(bar, prediction, position_manager.position)
```

### Shutdown Sequence
```
1. Cancel all pending orders
2. Flatten all positions (if any)
3. Disconnect from WebSocket
4. Save session state
5. Generate session report
```

---

## Error Handling

### Connection Errors
| Error | Action |
|-------|--------|
| WebSocket disconnect | Auto-reconnect with backoff |
| API timeout | Retry with exponential backoff |
| Auth failure | Re-authenticate, alert if fails |

### Order Errors
| Error | Action |
|-------|--------|
| Order rejected | Log, do not retry |
| Insufficient margin | Reduce size, retry |
| Rate limited | Wait, retry |
| Position mismatch | Sync with API, reconcile |

### Recovery
```python
async def recover_from_disconnect():
    # 1. Reconnect
    await connect_with_backoff()

    # 2. Sync positions with API
    api_positions = await client.get_positions()
    local_position = position_manager.position

    # 3. Reconcile
    if mismatch(api_positions, local_position):
        logger.alert("Position mismatch detected")
        # Use API as source of truth
        position_manager.sync_from_api(api_positions)

    # 4. Resume trading
```

---

## Logging

### Log Levels
| Level | Content |
|-------|---------|
| DEBUG | All predictions, feature values |
| INFO | Signals, orders, fills |
| WARNING | Risk events, connection issues |
| ERROR | Failures, exceptions |

### Log Format
```
2025-01-15 10:30:45.123 [INFO] SIGNAL: LONG_ENTRY conf=0.78 price=6050.25
2025-01-15 10:30:45.456 [INFO] ORDER: MARKET BUY 1 MES @ market
2025-01-15 10:30:45.789 [INFO] FILL: BOUGHT 1 MES @ 6050.50
2025-01-15 10:30:46.012 [INFO] ORDER: STOP SELL 1 MES @ 6048.00
2025-01-15 10:30:46.234 [INFO] ORDER: LIMIT SELL 1 MES @ 6054.00
2025-01-15 10:35:22.567 [INFO] FILL: SOLD 1 MES @ 6054.00 (TARGET HIT)
2025-01-15 10:35:22.789 [INFO] TRADE CLOSED: +$17.50 (14 ticks)
```

### Session Log File
```
logs/
├── trading_2025-01-15.log      # Full session log
├── trades_2025-01-15.csv       # Trade log
└── metrics_2025-01-15.json     # Session metrics
```

---

## Configuration

### Config File Structure
```yaml
# config/live_trading.yaml

api:
  base_url: "https://api.topstepx.com"
  ws_url: "wss://rtc.topstepx.com/hubs/market"
  account_id: ${TOPSTEPX_ACCOUNT_ID}

trading:
  contract: "CON.F.US.MES.H26"
  session_start: "09:30"  # NY time
  session_end: "16:30"    # NY time
  flatten_time: "16:25"   # Start flatten 5 min before

model:
  path: "models/scalper_v1.pt"
  config: "models/scalper_v1_config.json"
  min_confidence: 0.65

risk:
  starting_capital: 1000
  max_daily_loss: 50
  max_risk_per_trade: 0.025
  max_consecutive_losses: 5

execution:
  order_type: "market"  # or "limit"
  slippage_buffer_ticks: 1
```

---

## Acceptance Criteria

### Connectivity
- [ ] Successful TopstepX authentication
- [ ] Stable WebSocket connection (auto-reconnect)
- [ ] Real-time quote reception (< 100ms latency)

### Order Execution
- [ ] Market orders execute within 1 second
- [ ] Stop/target orders placed after entry fill
- [ ] Orders cancelled on position close
- [ ] No orphaned orders

### Risk Compliance
- [ ] Daily loss limit enforced
- [ ] EOD flatten at 4:25 PM guaranteed
- [ ] Position sizing correct
- [ ] Circuit breakers functional

### Reliability
- [ ] Graceful handling of disconnects
- [ ] Position sync after reconnect
- [ ] No duplicate orders
- [ ] Comprehensive logging

### Performance
- [ ] Inference latency < 10ms
- [ ] Order placement < 500ms
- [ ] Feature calculation < 5ms
- [ ] Memory stable over 8-hour session
