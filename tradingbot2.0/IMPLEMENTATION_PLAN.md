# Implementation Plan - MES Futures Scalping Bot

> Last Updated: 2026-01-15
> Status: ACTIVE - Phase 1 (Data Pipeline)
> Verified: All findings confirmed via codebase analysis

---

## Executive Summary

**Current State**: Basic daily direction prediction ML pipeline exists, but fundamentally misaligned with scalping requirements.

**Critical Issue**: The model predicts DAILY price direction using CSV/TXT data resampled to daily bars. The requirement is for SCALPING (seconds to minutes) using the 1-second parquet dataset.

**Capital Protection**: Starting capital is $1,000. The account CANNOT be blown up. All implementations must prioritize risk management.

**Data Asset**: 227MB parquet file exists at `data/historical/MES/MES_1s_2years.parquet` (33.2M rows) but is NOT being used by the current pipeline.

---

## Quick Reference: Key Numbers

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting Capital | $1,000 | TopstepX funded account |
| MES Tick Size | 0.25 points | Minimum price movement |
| MES Tick Value | $1.25 | Dollar value per tick |
| MES Point Value | $5.00 | $1.25 x 4 ticks |
| Max Daily Loss | $50 (5%) | Stop trading for day |
| Max Daily Drawdown | $75 (7.5%) | Stop trading for day |
| Max Per-Trade Risk | $25 (2.5%) | Position sizing limit |
| Kill Switch | $300 (30%) | Cumulative loss - halt permanently |
| Min Account Balance | $700 | Cannot trade below this |
| EOD Flatten Time | 4:30 PM NY | **HARD REQUIREMENT** |
| Commission (RT) | $0.84 | $0.20 comm + $0.22 fee x 2 sides |
| Min Confidence | 60% | No trade below this threshold |
| Target Inference | < 10ms | Real-time requirement |
| RTH Session | 9:30 AM - 4:00 PM NY | Regular trading hours |

---

## Codebase Gap Analysis (Verified 2026-01-15)

### What EXISTS (Implemented in `src/ml/`)

| Component | File | Status | Gap | Line Reference |
|-----------|------|--------|-----|----------------|
| Data Loader | `data/data_loader.py` | DAILY only | Needs parquet, 1-second, session filtering | Line 143-168: Binary target only |
| Feature Engineering | `data/feature_engineering.py` | DAILY periods | Needs SECONDS periods (1,5,10,30,60s) | Config line 16: `[1, 5, 10, 21]` days |
| Neural Networks | `models/neural_networks.py` | Binary class | Needs 3-class (softmax) | Lines 77, 102, 166, 213, 304: `sigmoid` output |
| Training Pipeline | `models/training.py` | Walk-forward exists | Needs 3-class loss, class weights | Uses `BCELoss` |
| Evaluation | `utils/evaluation.py` | WRONG costs | Needs $0.84 RT, tick-based slippage | Line 129: `commission: float = 5.0` |
| Config | `configs/default_config.yaml` | EXISTS but orphaned | Not loaded by any code | Line 88: `commission: 5.0` |

### What DOES NOT EXIST (0% implemented - Verified)

| Module | Directory | Priority | Status | Files Needed |
|--------|-----------|----------|--------|--------------|
| Risk Management | `src/risk/` | **P1 - CRITICAL** | NOT IMPLEMENTED | risk_manager.py, position_sizing.py, stops.py, eod_manager.py, circuit_breakers.py |
| Backtesting Engine | `src/backtest/` | **P1 - CRITICAL** | NOT IMPLEMENTED | engine.py, costs.py, slippage.py, metrics.py, logging.py |
| TopstepX API | `src/api/` | P2 - HIGH | NOT IMPLEMENTED | topstepx_client.py, topstepx_rest.py, topstepx_ws.py |
| Live Trading | `src/trading/` | P2 - HIGH | NOT IMPLEMENTED | live_trader.py, signal_generator.py, order_executor.py, position_manager.py, rt_features.py, recovery.py |
| DataBento Client | `src/data/` | P3 - MEDIUM | NOT IMPLEMENTED | databento_client.py |
| Shared Utilities | `src/lib/` | P3 - MEDIUM | NOT IMPLEMENTED | config.py, logging.py, time_utils.py |
| Tests | `tests/` | P3 - MEDIUM | **PARTIAL** (26 tests for parquet_loader) | Remaining test files |

### Critical Bug Summary

| Issue | Location | Current Value | Required Value | Impact |
|-------|----------|---------------|----------------|--------|
| Wrong commission | `evaluation.py:129` | $5.00 | $0.84 | 6x cost overestimate |
| Wrong commission (config) | `default_config.yaml:88` | $5.00 | $0.84 | Config also wrong |
| Wrong slippage model | `evaluation.py:130` | 0.0001 (1bp %) | 1 tick ($1.25) | Futures use ticks, not % |
| Wrong slippage (config) | `default_config.yaml:89` | 0.0001 (1bp %) | 1 tick ($1.25) | Config also wrong |
| Binary output | `neural_networks.py:77,102,166,213,304` | sigmoid (2-class) | softmax (3-class) | Cannot predict FLAT |
| Wrong time periods | `feature_engineering.py:37` | [1,5,10,21] days | [1,5,10,30,60] seconds | Daily vs scalping mismatch |
| Wrong time periods (config) | `default_config.yaml:16` | [1,5,10,21] days | [1,5,10,30,60] seconds | Config also wrong |
| Binary target | `data_loader.py:157` | Binary (up/down) | 3-class (UP/FLAT/DOWN) | Cannot predict FLAT zone |
| No parquet support | `data_loader.py:45-57` | CSV/TXT only | Need parquet | 227MB 1s data unused |
| Config not loaded | `train_futures_model.py` | CLI args only | Load from YAML | Config file orphaned |
| ~~Missing pyarrow~~ | `requirements.txt` | ~~Not listed~~ | ~~Add pyarrow>=14.0.0~~ | **FIXED** - Added to requirements.txt |
| ~~Missing pyyaml~~ | `requirements.txt` | ~~Not listed~~ | ~~Add pyyaml>=6.0~~ | **FIXED** - Added to requirements.txt |
| HybridNet not integrated | `train_futures_model.py` | Only feedforward/lstm | Add hybrid option | Unused architecture |

---

## Phase 1: CRITICAL - Data Pipeline (Week 1-2)

### 1.1 Parquet Data Loader for 1-Second Data
**Status**: COMPLETED (2026-01-15)
**File**: `src/ml/data/parquet_loader.py` (NEW)
**Spec**: `specs/databento-historical-data.md`, `specs/backtesting-engine.md`
**Data Asset**: `data/historical/MES/MES_1s_2years.parquet` (227MB, verified exists)

Current `FuturesDataLoader` only reads CSV/TXT and resamples to daily. Must support:
- [x] Load parquet format (`data/historical/MES/MES_1s_2years.parquet` - 227MB, 33.2M rows)
- [x] Parse timestamp (nanosecond int64 or datetime64) to proper datetime
- [x] Convert UTC to NY timezone (handle DST transitions)
- [x] Filter RTH (9:30 AM - 4:00 PM NY) vs ETH (6:00 PM - 9:30 AM NY)
- [x] Handle gaps (weekends, holidays, market closures)
- [x] Multi-timeframe aggregation (1s -> 5s, 15s, 1m, 5m, 15m)
- [x] Efficient chunked loading for memory management (dask or polars optional)
- [x] Session boundary detection (trading day start/end)

**Acceptance Criteria** (ALL MET):
- Load 33.2M rows in ~1 second (parquet loading very fast)
- Filter to RTH only, returns ~15.8M rows
- Full pipeline (load + filter + target + features + split) in ~53 seconds
- Memory usage < 4GB during loading
- No timezone errors across DST boundaries

### 1.2 Scalping Target Variable (3-Class)
**Status**: COMPLETED (2026-01-15)
**File**: `src/ml/data/parquet_loader.py` (implemented in `create_target_variable()` method)
**Spec**: `specs/ml-scalping-model.md` (Target Variable section)

Current target is binary (line 157: `(df['close'].shift(-lookahead) > df['close']).astype(int)`). Scalping requires 3-class:
- [x] 3-class classification: DOWN (0), FLAT (1), UP (2)
- [x] Configurable lookahead: 5, 10, 30, 60 seconds (default: 30s)
- [x] Configurable threshold: 2-4 ticks for MES (default: 3.0 ticks = $3.75)
- [x] Class balance analysis and logging
- [x] Lookahead bias prevention (strict future data isolation)

**Formula**:
```python
future_price = df['close'].shift(-lookahead_seconds)
tick_move = (future_price - df['close']) / 0.25  # MES tick size = 0.25 points
target = np.where(tick_move > threshold_ticks, 2,  # UP
         np.where(tick_move < -threshold_ticks, 0,  # DOWN
                  1))  # FLAT
```

**Acceptance Criteria** (ALL MET):
- Class distribution logged: DOWN=19.7%, FLAT=60.2%, UP=20.1%
- No NaN values in target column (rows with NaN targets dropped)
- Lookahead correctly applied (shift by seconds, not rows)

### 1.3 Feature Engineering for 1-Second Data
**Status**: COMPLETED (2026-01-15)
**File**: `src/ml/data/scalping_features.py` (NEW)
**Spec**: `specs/ml-scalping-model.md` (Feature Engineering section)

Implemented `ScalpingFeatureEngineer` class with all required features:
- [x] Returns at 1, 5, 10, 30, 60 SECONDS
- [x] EMAs: 9, 21, 50, 200 periods on 1-second data
- [x] Session-based VWAP (reset at 9:30 AM NY each day)
- [x] Minutes-to-close feature (0-390 for RTH, normalized)
- [x] Multi-timeframe features (1m, 5m trend/momentum - use lagged to avoid lookahead)
- [x] Microstructure: bar direction (+1/-1), upper/lower wick ratios, body ratio
- [x] Volume delta (buy vs sell volume if tick data available)
- [x] All features normalized for neural network input

**Acceptance Criteria** (ALL MET):
- No lookahead bias in any feature (verified via temporal unit test)
- All features normalized to [-1, 1] or [0, 1] range
- Feature correlation matrix generated
- Feature importance baseline established

---

## Phase 2: CRITICAL - Risk Management Module (Week 2-3)

**Status**: ENTIRE MODULE NOT IMPLEMENTED
**Directory**: `src/risk/` (NEW - directory does not exist)
**Spec**: `specs/risk-management.md`

### 2.1 Core Risk Manager
**File**: `src/risk/risk_manager.py` (NEW)

Per spec, this is NON-NEGOTIABLE for $1,000 account:
- [ ] Daily loss limit: $50 (5% of $1,000) -> stop trading
- [ ] Daily drawdown limit: $75 (7.5%) -> stop trading
- [ ] Per-trade max risk: $25 (2.5%)
- [ ] Max consecutive losses: 5 -> 30-min pause
- [ ] Kill switch: $300 cumulative loss -> halt permanently
- [ ] Minimum account balance: $700 -> cannot trade below this
- [ ] Account drawdown $200 (20%) -> stop, manual review required
- [ ] Thread-safe state tracking (async-compatible)

### 2.2 Position Sizing
**File**: `src/risk/position_sizing.py` (NEW)

- [ ] Calculate position size from: account balance, risk %, stop distance, tick value
- [ ] Scaling rules by account balance tier:

| Balance | Max Contracts | Risk % |
|---------|---------------|--------|
| $700-$1,000 | 1 | 2% |
| $1,000-$1,500 | 2 | 2% |
| $1,500-$2,000 | 3 | 2% |
| $2,000-$3,000 | 4 | 2% |
| $3,000+ | 5+ | 1.5% |

- [ ] Confidence-based multipliers:

| Confidence | Multiplier |
|------------|------------|
| < 60% | No trade (0x) |
| 60-70% | 0.5x |
| 70-80% | 1.0x |
| 80-90% | 1.5x |
| > 90% | 2.0x (capped) |

### 2.3 Stop Loss Strategy
**File**: `src/risk/stops.py` (NEW)

- [ ] ATR-based stops (recommended): Stop = Entry +/- (ATR x 1.5 multiplier)
- [ ] Fixed tick stops (e.g., 8 ticks = $10 risk per contract)
- [ ] Structure-based stops (swing high/low)
- [ ] Trailing stop logic (move stop to breakeven after X profit)
- [ ] EOD tightening rules (tighter stops near market close)

### 2.4 EOD Flatten Logic
**File**: `src/risk/eod_manager.py` (NEW)

**HARD REQUIREMENT**: All positions must be flat by 4:30 PM NY
- [ ] 4:00 PM NY: Reduce position sizing by 50%
- [ ] 4:15 PM NY: No new positions, close existing only
- [ ] 4:25 PM NY: Begin market order exits (aggressive)
- [ ] 4:30 PM NY: MUST be flat (no exceptions)
- [ ] Timezone-aware datetime handling (pytz or zoneinfo)

### 2.5 Circuit Breakers
**File**: `src/risk/circuit_breakers.py` (NEW)

- [ ] 3 consecutive losses -> 15-min pause (auto-resume)
- [ ] 5 consecutive losses -> 30-min pause (auto-resume)
- [ ] Daily loss limit hit -> stop for day (resume next day 9:30 AM)
- [ ] Max drawdown hit -> indefinite pause (manual review required)
- [ ] Volatility > 3x normal ATR -> reduce size 50% or pause
- [ ] Spread widening > 2 ticks ($2.50) -> pause until spread <= 1 tick
- [ ] Low volume (< 10% of avg) -> reduce size or pause

**Acceptance Criteria** (Circuit Breakers):
- All triggers tested in simulation with synthetic loss sequences
- Pause timers accurate to within 1 second
- State persists across bot restarts (daily loss tracked in file/DB)

---

## Phase 3: CRITICAL - Backtesting Engine (Week 3-4)

**Status**: NOT IMPLEMENTED (current `evaluation.py` is basic daily simulation with wrong costs)
**Directory**: `src/backtest/` (NEW - directory does not exist)
**Spec**: `specs/backtesting-engine.md`

### 3.1 Event-Driven Backtest Engine
**File**: `src/backtest/engine.py` (NEW)

- [ ] Bar-by-bar simulation on 1-second data
- [ ] Event loop: Update indicators -> Check exits -> Generate signals -> Risk check -> Execute
- [ ] Proper order fill simulation:
  - Conservative: Fill only if price touches order price
  - Optimistic: Fill at signal bar close
  - Realistic: Fill with slippage model
- [ ] EOD flatten enforcement at 4:30 PM NY (use risk module)
- [ ] Walk-forward optimization framework:
  - 6 months training
  - 1 month validation
  - 1 month test
  - Roll monthly
- [ ] Minimum 100 trades per fold for statistical significance
- [ ] Out-of-sample performance tracking

### 3.2 Transaction Cost Model
**File**: `src/backtest/costs.py` (NEW)

MES-specific costs (TopstepX):
- [ ] Commission: $0.20/side ($0.40 round-trip)
- [ ] Exchange fee: $0.22/side ($0.44 round-trip)
- [ ] Total round-trip: $0.84/contract
- [ ] Configurable for different brokers

**BUG FIX REQUIRED**: Current `evaluation.py` line 129 uses `commission: float = 5.0` - WRONG by 6x.

### 3.3 Slippage Model
**File**: `src/backtest/slippage.py` (NEW)

**BUG FIX REQUIRED**: Current `evaluation.py` line 131 uses `slippage: float = 0.0001` (1bp percentage) - WRONG for futures.
- [ ] Normal liquidity: 1 tick ($1.25)
- [ ] Low liquidity (thin orderbook, ETH): 2 ticks ($2.50)
- [ ] High volatility (news, FOMC): 2-4 ticks ($2.50-$5.00)
- [ ] Market orders: 1 tick assumed minimum
- [ ] Limit orders: 0 ticks if filled
- [ ] Volatility-adaptive model (ATR-based)

### 3.4 Trade & Equity Logging
**File**: `src/backtest/logging.py` (NEW)

- [ ] Trade log CSV columns:
  - entry_time, exit_time, direction, entry_price, exit_price
  - contracts, gross_pnl, commission, slippage, net_pnl
  - exit_reason (target, stop, eod_flatten, signal)
  - model_confidence, predicted_class
- [ ] Equity curve at bar-level resolution (every second if needed)
- [ ] Per-fold walk-forward results
- [ ] Drawdown tracking (depth, duration, recovery)

### 3.5 Performance Metrics
**File**: `src/backtest/metrics.py` (NEW)
**Status**: PARTIAL in `evaluation.py` - missing several key metrics

Missing metrics per spec:
- [ ] Calmar ratio: annualized_return / max_drawdown
- [ ] Sortino ratio: return / downside_deviation
- [ ] Max drawdown duration (bars/hours/days to recover)
- [ ] Consistency: win days %, consecutive wins/losses streaks
- [ ] Expectancy: (win_rate x avg_win) - (loss_rate x avg_loss)
- [ ] Per-trade metrics: avg win, avg loss, largest win, largest loss
- [ ] Trade frequency: trades per day/hour
- [ ] Risk-adjusted metrics: return per dollar risked
- [ ] Per-trade cost breakdown (commission, slippage, total)
- [ ] Time-of-day performance analysis (hourly P&L bucketing)

### 3.6 Backtesting Performance Requirements
**Spec Reference**: `specs/backtesting-engine.md` (Performance Requirements section)

- [ ] Process 1M bars in < 60 seconds
- [ ] Walk-forward fold completes in < 5 minutes
- [ ] Full optimization run in < 1 hour
- [ ] Memory usage stable (no leaks over long runs)

### 3.7 Backtesting Validation Tests
**Spec Reference**: `specs/backtesting-engine.md` (Validation section)

- [ ] Known strategy (e.g., trend following) produces expected positive results
- [ ] Random strategy produces ~0 expectancy (validates no lookahead bias)
- [ ] Transaction costs reduce returns by expected amount
- [ ] Results reproducible with same random seed

---

## Phase 4: HIGH - Model Architecture Updates (Week 4-5)

### 4.1 3-Class Classification
**Status**: NOT IMPLEMENTED (current is binary sigmoid output)
**File**: `src/ml/models/neural_networks.py` (modify)
**Spec**: `specs/ml-scalping-model.md`

Lines requiring modification:
- Line 77: `self.output_layer = nn.Linear(hidden_dims[-1], 1)` -> `nn.Linear(hidden_dims[-1], 3)`
- Line 102: `return torch.sigmoid(x)` -> `return F.softmax(x, dim=-1)`
- Line 166: `self.output_layer = nn.Linear(..., 1)` -> `nn.Linear(..., 3)`
- Line 213: `output = torch.sigmoid(...)` -> `output = F.softmax(..., dim=-1)`
- Line 304: `output = torch.sigmoid(...)` -> `output = F.softmax(..., dim=-1)`

Tasks:
- [ ] Change output layer: `Dense(1, sigmoid)` -> `Dense(3, softmax)`
- [ ] Change loss: `BCELoss` -> `CrossEntropyLoss` with class weights
- [ ] Handle class imbalance (FLAT class likely dominant ~60-70%):
  - Class weights inverse to frequency
  - Or focal loss for hard examples
- [ ] Update all model architectures: FeedForwardNet, LSTMNet, HybridNet
- [ ] Update training loop in `training.py` for 3-class

### 4.2 Model Output Interface
**Status**: NOT IMPLEMENTED
**File**: `src/ml/models/neural_networks.py` (add)

Per spec, model must return structured output:
```python
@dataclass
class ModelPrediction:
    direction: int  # -1 (short), 0 (flat), 1 (long)
    confidence: float  # 0-1, max of softmax probabilities
    predicted_move: float  # expected ticks (weighted by class probs)
    volatility: float  # for position sizing (from ATR or auxiliary head)
    timestamp: datetime
```

### 4.3 Inference Optimization
**Status**: NOT TESTED
**File**: `src/ml/models/neural_networks.py` (add benchmark)

- [ ] Verify inference latency < 10ms on CPU
- [ ] Benchmark on target hardware (M1 Mac, cloud VM)
- [ ] Optional: ONNX export for production deployment
- [ ] Batch inference for backtesting (process many samples at once)
- [ ] Feature computation included in latency budget

### 4.4 Transformer Architecture (Optional Enhancement)
**Status**: NOT IMPLEMENTED
**File**: `src/ml/models/transformer.py` (NEW, optional)

Per spec, consider attention-based model for sequence patterns:
- [ ] Multi-head self-attention for temporal dependencies
- [ ] Positional encoding for time awareness
- [ ] May capture regime changes better than LSTM
- [ ] Compare performance vs LSTM in walk-forward

---

## Phase 5: HIGH - TopstepX API Integration (Week 5-6)

**Status**: ENTIRE MODULE NOT IMPLEMENTED
**Directory**: `src/api/` (NEW - directory does not exist)
**Spec**: `specs/topstepx-api-integration.md`

### 5.1 API Client Base
**File**: `src/api/topstepx_client.py` (NEW)

- [ ] Base URL: `https://api.topstepx.com`
- [ ] Authentication: POST `/api/Auth/loginKey` with API key
- [ ] Token refresh logic (90-min expiry, refresh at 80 min proactively)
- [ ] Rate limiting: 50 requests/30 seconds (track and throttle)
- [ ] Exponential backoff on errors (1s, 2s, 4s, max 30s)
- [ ] Request/response logging (sanitize sensitive data)
- [ ] Session management (aiohttp or httpx)

### 5.2 REST Endpoints
**File**: `src/api/topstepx_rest.py` (NEW)

- [ ] Place order: POST `/api/Order/place` (market, limit, stop, stop-limit)
- [ ] Cancel order: POST `/api/Order/cancel`
- [ ] Get positions: GET `/api/Position/list`
- [ ] Get account info: GET `/api/Account/info`
- [ ] Retrieve historical bars: POST `/api/History/retrieveBars` (limited: ~7-14 days)
- [ ] Order types enum: MARKET, LIMIT, STOP, STOP_LIMIT
- [ ] Error handling for each endpoint

### 5.3 WebSocket Market Data
**File**: `src/api/topstepx_ws.py` (NEW)

- [ ] SignalR connection to `wss://rtc.topstepx.com/hubs/market`
- [ ] Subscribe to MES quotes (MESU5, MESZ5, etc.)
- [ ] Quote handler callback (bid, ask, last, size, timestamp)
- [ ] Auto-reconnect with exponential backoff (1s, 2s, 4s...)
- [ ] Max 2 concurrent WebSocket sessions (per spec)
- [ ] Heartbeat/ping to detect disconnects

### 5.4 WebSocket Trade Hub
**File**: `src/api/topstepx_ws.py` (extend)

- [ ] Connect to `wss://rtc.topstepx.com/hubs/trade`
- [ ] Order fill notifications (fill_price, fill_qty, order_id)
- [ ] Position update notifications (direction, size, avg_price)
- [ ] Account update notifications (balance, realized_pnl)
- [ ] Order rejection notifications

---

## Phase 6: HIGH - Live Trading System (Week 6-7)

**Status**: ENTIRE MODULE NOT IMPLEMENTED
**Directory**: `src/trading/` (NEW - directory does not exist)
**Spec**: `specs/live-trading-execution.md`

### 6.1 Main Trading Loop
**File**: `src/trading/live_trader.py` (NEW)

**Startup Sequence**:
- [ ] Load configuration from YAML + env vars
- [ ] Load ML model from checkpoint (e.g., `models/scalper_v1.pt`)
- [ ] Initialize risk manager with current account state
- [ ] Connect to TopstepX API and authenticate
- [ ] Subscribe to market data WebSocket
- [ ] Sync existing positions from API (API is source of truth)
- [ ] Begin trading loop

**Main Loop (per bar)**:
- [ ] Receive bar from WebSocket aggregator
- [ ] Update feature buffer (circular buffer)
- [ ] Run model inference (< 10ms)
- [ ] Generate signal from prediction
- [ ] Risk manager check (limits, EOD, circuit breakers)
- [ ] Execute if approved
- [ ] Log state to file and console

**Shutdown Sequence**:
- [ ] Cancel all pending orders
- [ ] Flatten all positions (market orders)
- [ ] Disconnect from WebSocket
- [ ] Save session state to disk
- [ ] Generate session report (trades, P&L, metrics)

### 6.2 Signal Generator
**File**: `src/trading/signal_generator.py` (NEW)

Signal types:
- [ ] LONG_ENTRY, SHORT_ENTRY
- [ ] EXIT_LONG, EXIT_SHORT
- [ ] REVERSE_TO_LONG, REVERSE_TO_SHORT
- [ ] FLATTEN (EOD or risk limit)
- [ ] HOLD (no action)

Logic:
- [ ] Confidence threshold check (min 60%)
- [ ] Risk manager integration (check limits before signal)
- [ ] Position-aware (no duplicate entries, handle reversals)
- [ ] Cooldown after exits (configurable seconds)

### 6.3 Order Executor
**File**: `src/trading/order_executor.py` (NEW)

- [ ] Place entry order (market for speed, limit for price)
- [ ] Wait for fill confirmation (timeout handling)
- [ ] Place stop loss order immediately after entry fill
- [ ] Place take profit order
- [ ] Manual OCO management (API doesn't support bracket natively)
- [ ] Track all open orders by ID
- [ ] Cancel orphaned orders on exit/flatten
- [ ] Handle partial fills (adjust stop/target quantities)

### 6.4 Position Manager
**File**: `src/trading/position_manager.py` (NEW)

```python
@dataclass
class Position:
    contract_id: str  # MES contract symbol
    direction: int  # 1 = long, -1 = short
    size: int  # contracts
    entry_price: float
    entry_time: datetime
    stop_price: float
    target_price: float
    stop_order_id: str
    target_order_id: str
    unrealized_pnl: float
    realized_pnl: float  # if partially closed
```

- [ ] Track open position state locally
- [ ] Calculate unrealized P&L in real-time (tick-by-tick)
- [ ] Sync with API on reconnect (API is source of truth)
- [ ] Position change notifications/callbacks

### 6.5 Real-Time Feature Engine
**File**: `src/trading/rt_features.py` (NEW)

- [ ] Aggregate incoming ticks to 1-second OHLCV bars
- [ ] Maintain rolling feature windows (EMA, RSI, VWAP, etc.)
- [ ] Calculate features in < 5ms (budget for 10ms total inference)
- [ ] Memory-efficient circular buffers (collections.deque or numpy)
- [ ] Feature caching (avoid redundant calculations)

### 6.6 Error Handling & Recovery
**File**: `src/trading/recovery.py` (NEW)

- [ ] WebSocket disconnect -> auto-reconnect with backoff
- [ ] Position mismatch -> sync from API (API wins, log discrepancy)
- [ ] Order rejection -> log reason, don't retry same order
- [ ] Insufficient margin -> reduce size, retry once
- [ ] Rate limited -> wait required time, retry
- [ ] Auth failure -> re-authenticate, alert if fails twice
- [ ] Unhandled exception -> flatten positions, halt, alert

### 6.7 Live Trading Performance Requirements
**Spec Reference**: `specs/live-trading-execution.md` (Performance section)

- [ ] WebSocket quote reception latency < 100ms
- [ ] Market orders execute within 1 second of signal
- [ ] Feature calculation < 5ms per bar
- [ ] Memory stable over 8-hour trading session (no leaks)
- [ ] Order placement round-trip < 500ms

---

## Phase 7: MEDIUM - DataBento Integration (Week 7-8)

**Status**: NOT IMPLEMENTED (parquet data exists but no update mechanism)
**Directory**: `src/data/` (NEW - directory does not exist)
**Spec**: `specs/databento-historical-data.md`

### 7.1 DataBento Client
**File**: `src/data/databento_client.py` (NEW)

- [ ] Initialize with API key from `DATABENTO_API_KEY` env var
- [ ] Fetch OHLCV data at multiple timeframes (1s, 1m, 1h, 1d)
- [ ] Handle continuous contracts (MES.FUT, ES.FUT, MNQ.FUT, NQ.FUT)
- [ ] Store in parquet format with year/month partitioning
- [ ] Schema validation on download

### 7.2 Data Download Script
**File**: `scripts/download_data.py` (NEW)

- [ ] Initial bulk download (3+ years of 1-second data)
- [ ] Incremental daily updates (append to parquet)
- [ ] Gap detection and backfill
- [ ] Data validation:
  - OHLC relationships (L <= O,C <= H for all bars)
  - No gaps during trading hours (weekdays 6 PM - 5 PM ET)
  - Volume sanity checks (no negative, no extreme outliers)
  - Timestamps in UTC
- [ ] Progress logging and resumption

### 7.3 Data Quality Acceptance Criteria
**Spec Reference**: `specs/databento-historical-data.md` (Data Quality section)

- [ ] All OHLC relationships valid: `low <= open <= high` and `low <= close <= high`
- [ ] No gaps during trading hours (alert on missing bars)
- [ ] Volume data present for all bars (no nulls)
- [ ] Timestamps correctly in UTC
- [ ] Continuous contract stitching produces smooth price series (no jumps at roll)

---

## Phase 8: MEDIUM - Testing (Ongoing)

**Status**: PARTIAL - tests/ directory created with 76 unit tests (26 for parquet_loader, 50 for scalping_features)
**Directory**: `tests/`

### 8.1 Unit Tests

- [x] `tests/test_parquet_loader.py` - parquet loading, session filtering, timezone handling (26 tests, all passing)
- [x] `tests/test_scalping_features.py` - scalping feature engineering for 1-second data (50 tests, all passing)
- [ ] `tests/test_feature_engineering.py` - feature calculations, no NaN leakage, no lookahead
- [ ] `tests/test_risk_manager.py` - all risk limits, position sizing, EOD flatten
- [ ] `tests/test_backtest.py` - cost model, slippage, order fill logic
- [ ] `tests/test_models.py` - model forward pass, output shape, 3-class output
- [ ] `tests/test_signal_generator.py` - signal logic, confidence thresholds
- [ ] `tests/conftest.py` - pytest fixtures, sample data

### 8.2 Integration Tests
**Directory**: `tests/integration/` (NEW)

- [ ] End-to-end backtest run on sample data (100 trades minimum)
- [ ] API mock tests (httpx mock or responses library)
- [ ] Walk-forward validation produces expected fold count
- [ ] Risk limits properly halt trading in simulation
- [ ] EOD flatten fires at correct time

### 8.3 Test Configuration

- [ ] `pytest.ini` or `pyproject.toml` pytest config
- [ ] Test coverage target: > 80%
- [ ] CI/CD integration (GitHub Actions)

---

## Phase 9: LOW - Optimizations & Polish (Week 8+)

### 9.1 Parameter Optimization Framework
**Directory**: `src/optimization/` (NEW)

- [ ] Grid search over: stop_ticks, target_ticks, confidence_threshold, risk_pct
- [ ] Random search for efficiency
- [ ] Bayesian optimization (Optuna)
- [ ] Genetic algorithm for complex parameter spaces
- [ ] Overfitting prevention: optimize on validation, test on holdout

### 9.2 Visualization & Reporting
**Status**: PARTIAL in `evaluation.py`

- [ ] Interactive equity curve plots (Plotly)
- [ ] Trade distribution histograms
- [ ] Drawdown visualization with duration markers
- [ ] Per-fold metrics dashboard
- [ ] Walk-forward equity stitching (combined OOS equity)

### 9.3 Configuration Management
**Status**: PARTIAL - `configs/default_config.yaml` exists but is orphaned (not loaded by any code)

- [ ] `src/lib/config.py` - Unified config loader
- [ ] Environment variable support for secrets (API keys)
- [ ] Config validation at startup (pydantic or marshmallow)
- [ ] Config versioning for reproducibility

### 9.4 Shared Utilities Library
**Directory**: `src/lib/` (NEW - directory does not exist)

- [ ] `src/lib/config.py` - Unified config loader (YAML + env vars)
- [ ] `src/lib/logging.py` - Structured logging with rotation
- [ ] `src/lib/time_utils.py` - NY timezone, session times, market calendar
- [ ] `src/lib/constants.py` - MES tick size, point value, session times

---

## Proposed Directory Structure

```
tradingbot2.0/
├── CLAUDE.md                  # Agent instructions
├── AGENTS.md                  # Operational guide (build/run/test)
├── IMPLEMENTATION_PLAN.md     # This file (task tracking)
├── loop.sh                    # Ralph autonomous loop script
│
├── specs/                     # Requirements specifications (6 files - verified)
│   ├── backtesting-engine.md
│   ├── databento-historical-data.md
│   ├── live-trading-execution.md
│   ├── ml-scalping-model.md
│   ├── risk-management.md
│   └── topstepx-api-integration.md
│
├── src/
│   ├── ml/                    # EXISTING - ML pipeline (11 files)
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py     # MODIFY for 3-class target (line 143)
│   │   │   ├── feature_engineering.py  # MODIFY for seconds
│   │   │   ├── parquet_loader.py  # NEW - Phase 1.1/1.2
│   │   │   └── scalping_features.py  # NEW - Phase 1.3 (ScalpingFeatureEngineer class)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── neural_networks.py # MODIFY for 3-class (lines 77,102,166,213,304)
│   │   │   ├── training.py        # MODIFY for 3-class loss
│   │   │   └── transformer.py     # NEW (optional)
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── evaluation.py      # Keep for reference (wrong costs at line 129)
│   │   ├── configs/
│   │   │   └── default_config.yaml  # EXISTS but orphaned (wrong values)
│   │   └── train_futures_model.py
│   │
│   ├── risk/                  # NEW - Risk management (0% implemented)
│   │   ├── __init__.py
│   │   ├── risk_manager.py
│   │   ├── position_sizing.py
│   │   ├── stops.py
│   │   ├── eod_manager.py
│   │   └── circuit_breakers.py
│   │
│   ├── backtest/              # NEW - Backtesting engine (0% implemented)
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── costs.py
│   │   ├── slippage.py
│   │   ├── metrics.py
│   │   └── logging.py
│   │
│   ├── api/                   # NEW - TopstepX integration (0% implemented)
│   │   ├── __init__.py
│   │   ├── topstepx_client.py
│   │   ├── topstepx_rest.py
│   │   └── topstepx_ws.py
│   │
│   ├── trading/               # NEW - Live trading (0% implemented)
│   │   ├── __init__.py
│   │   ├── live_trader.py
│   │   ├── signal_generator.py
│   │   ├── order_executor.py
│   │   ├── position_manager.py
│   │   ├── rt_features.py
│   │   └── recovery.py
│   │
│   ├── data/                  # NEW - External data (0% implemented)
│   │   ├── __init__.py
│   │   └── databento_client.py
│   │
│   └── lib/                   # NEW - Shared utilities (0% implemented)
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── time_utils.py
│       └── constants.py
│
├── tests/                     # Test suite (PARTIAL - 76 tests)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parquet_loader.py    # 26 tests (Phase 1.1/1.2)
│   ├── test_scalping_features.py # 50 tests (Phase 1.3)
│   ├── test_feature_engineering.py
│   ├── test_risk_manager.py
│   ├── test_backtest.py
│   ├── test_models.py
│   └── integration/
│       ├── __init__.py
│       ├── test_backtest_e2e.py
│       └── test_api_mock.py
│
├── scripts/                   # NEW - Entry points
│   ├── download_data.py
│   ├── run_backtest.py
│   └── run_live.py
│
├── data/
│   └── historical/
│       └── MES/
│           ├── MES_1s_2years.parquet           # EXISTS - 227MB (verified)
│           └── MES_full_1min_continuous_UNadjusted.txt  # EXISTS - 122MB
│
└── models/                    # Model checkpoints (gitignored)
    └── scalper_v1.pt
```

---

## Implementation Order (Critical Path)

```
Week 1-2: Phase 1 - Data Pipeline
├── 1.1 Parquet loader (enables everything else)
├── 1.2 3-class target variable
└── 1.3 1-second feature engineering

Week 2-3: Phase 2 - Risk Management
├── 2.1 Risk manager core
├── 2.2 Position sizing
├── 2.3 Stop loss strategies
├── 2.4 EOD flatten (HARD REQUIREMENT)
└── 2.5 Circuit breakers

Week 3-4: Phase 3 - Backtesting Engine
├── 3.1 Event-driven engine
├── 3.2 MES cost model ($0.84 RT)
├── 3.3 Tick-based slippage
├── 3.4 Trade logging
└── 3.5 Extended metrics

Week 4-5: Phase 4 - Model Updates
├── 4.1 3-class output layer
├── 4.2 ModelPrediction interface
├── 4.3 Inference optimization (< 10ms)
└── 4.4 (Optional) Transformer

Week 5-6: Phase 5 - TopstepX API
├── 5.1 Auth + rate limiting
├── 5.2 REST endpoints
├── 5.3 Market WebSocket
└── 5.4 Trade WebSocket

Week 6-7: Phase 6 - Live Trading
├── 6.1 Main trading loop
├── 6.2 Signal generator
├── 6.3 Order executor
├── 6.4 Position manager
├── 6.5 Real-time features
└── 6.6 Recovery handling

Week 7-8: Phase 7-8 - Data & Testing
├── 7.1 DataBento client
├── 7.2 Download script
├── 8.1 Unit tests
└── 8.2 Integration tests

Week 8+: Phase 9 - Optimization
├── 9.1 Parameter optimization
├── 9.2 Visualization
├── 9.3 Config management
└── 9.4 Utilities library
```

---

## R:R Ratios to Optimize

| R:R | Stop (ticks) | Target (ticks) | Stop ($) | Target ($) | Breakeven Win Rate |
|-----|--------------|----------------|----------|------------|---------------------|
| 1:1 | 8 | 8 | $10.00 | $10.00 | 50% + costs |
| 1:1.5 | 8 | 12 | $10.00 | $15.00 | 40% |
| 1:2 | 8 | 16 | $10.00 | $20.00 | 33% |
| 1:3 | 8 | 24 | $10.00 | $30.00 | 25% |

*Note: Add $0.84 commission + ~$1.25 slippage = $2.09 per trade to cost calculations*

---

## Acceptance Criteria: Go-Live Checklist

Before going live with real capital, the system must:

1. [ ] Walk-forward backtest shows consistent profitability (Sharpe > 1.0, Calmar > 0.5)
2. [ ] Out-of-sample accuracy > 52% on 3-class (better than random)
3. [ ] All risk limits enforced and verified in simulation
4. [ ] EOD flatten works 100% of the time (verified across DST boundaries)
5. [ ] Inference latency < 10ms (measured on target hardware)
6. [ ] No lookahead bias in features or targets (temporal unit tests pass)
7. [ ] Unit test coverage > 80%
8. [ ] Paper trading for minimum 2 weeks without critical errors
9. [ ] Position sizing matches spec for all account balance tiers
10. [ ] Circuit breakers tested and working (simulated loss scenarios)
11. [ ] API reconnection works (tested with network interruption)
12. [ ] Manual kill switch accessible and tested

---

## Notes

- The existing `src/ml/` code is a solid foundation but needs significant rework for scalping timeframes
- **76 tests exist** (26 for parquet_loader, 50 for scalping_features) - remaining modules need test coverage
- The 227MB 1-second parquet dataset is the primary asset but isn't being used
- TopstepX API is for **live trading only** (7-14 day historical limit)
- DataBento is for historical data (already have 2 years in parquet)
- The 1-minute TXT file contains **2,334,170 lines** spanning May 2019 - Dec 2025 (~6.5 years)
- HybridNet architecture exists in `neural_networks.py:218-306` but is NOT integrated into train script
- Risk management is NON-NEGOTIABLE given $1,000 starting capital
- EOD flatten at 4:30 PM NY is a hard requirement (day trading rules)
- Current `evaluation.py` uses wrong costs ($5.00 vs $0.84) and wrong slippage model

### Data File Row Count Note
**VERIFIED (2026-01-15)**: The parquet file contains **33,206,650 rows** (matching the spec's "33 million 1-second bars"). Previous estimates of ~8.7M were incorrect.
- Total rows: 33,206,650
- After RTH filtering: ~15.8M rows (RTH is ~6.5 hours of 23-hour session)
- Class distribution (with 30s lookahead, 3-tick threshold): DOWN=19.7%, FLAT=60.2%, UP=20.1%

---

## Change Log

| Date | Change |
|------|--------|
| 2026-01-15 | Initial comprehensive plan created from codebase analysis |
| 2026-01-15 | Verified data files exist: MES_1s_2years.parquet (227MB), MES_full_1min.txt (122MB) |
| 2026-01-15 | Confirmed: src/risk/, src/backtest/, src/api/, src/trading/, tests/ do NOT exist |
| 2026-01-15 | Confirmed: Config YAML exists but is not loaded by any code |
| 2026-01-15 | **Verification pass**: Added specific line numbers for all bugs found |
| 2026-01-15 | Added "Critical Bug Summary" table with exact line references |
| 2026-01-15 | Added "Line Reference" column to gap analysis table |
| 2026-01-15 | Verified all 6 spec files exist in specs/ directory |
| 2026-01-15 | Verified all 10 Python files exist in src/ml/ |
| 2026-01-15 | Confirmed neural_networks.py uses sigmoid at lines 77, 102, 166, 213, 304 |
| 2026-01-15 | Confirmed evaluation.py has wrong commission ($5.00) at line 129 |
| 2026-01-15 | Confirmed data_loader.py creates binary target at line 157 |
| 2026-01-15 | Confirmed default_config.yaml has daily periods [1,5,10,21] at line 16 |
| 2026-01-15 | **Second verification pass**: Corrected parquet row count from ~33M to ~8.7M rows |
| 2026-01-15 | Corrected slippage bug line number from 131 to 130 |
| 2026-01-15 | Added duplicate bugs in default_config.yaml (commission at line 88, slippage at line 89) |
| 2026-01-15 | Added feature_engineering.py line 37 for return_periods bug |
| 2026-01-15 | Expanded Critical Bug Summary table from 5 to 10 entries with config duplicates |
| 2026-01-15 | Added "Config not loaded" as explicit bug (train_futures_model.py uses CLI only) |
| 2026-01-15 | Confirmed src/data/, src/lib/ directories also do NOT exist |
| 2026-01-15 | Adjusted RTH row estimate to ~4M (6.5h of 23h session from 8.7M total) |
| 2026-01-15 | **Third verification pass**: All line numbers verified against source files |
| 2026-01-15 | Confirmed: evaluation.py:129 commission=$5.0, line 130 slippage=0.0001 |
| 2026-01-15 | Confirmed: neural_networks.py sigmoid at lines 77,102,166,213,304 |
| 2026-01-15 | Confirmed: data_loader.py:157 binary target, feature_engineering.py:37 daily periods |
| 2026-01-15 | Confirmed: default_config.yaml:16,88,89 all have incorrect values |
| 2026-01-15 | Verified directory structure: src/ml/ exists, all other src/* subdirs do NOT exist |
| 2026-01-15 | Verified: tests/ does NOT exist, specs/ exists with all 6 files |
| 2026-01-15 | Verified: data/historical/MES/ exists with both data files (227MB parquet, 122MB txt) |
| 2026-01-15 | **Plan verification pass**: Reviewed all 6 spec files for completeness |
| 2026-01-15 | Added Section 3.6: Backtesting Performance Requirements (from spec) |
| 2026-01-15 | Added Section 3.7: Backtesting Validation Tests (from spec) |
| 2026-01-15 | Added Section 6.7: Live Trading Performance Requirements (from spec) |
| 2026-01-15 | Added Section 7.3: Data Quality Acceptance Criteria (from spec) |
| 2026-01-15 | Enhanced Section 2.5: Added circuit breaker acceptance criteria and spread threshold |
| 2026-01-15 | Enhanced Section 7.2: Added specific OHLC validation rules |
| 2026-01-15 | Added "Data File Row Count Note" explaining 33M vs 8.7M discrepancy |
| 2026-01-15 | Confirmed: All 6 specs fully covered - risk, backtest, ml-model, topstepx, live-trading, databento |
| 2026-01-15 | **Fourth verification pass (9-agent parallel analysis)**: All claims independently verified |
| 2026-01-15 | Added: Missing pyarrow and pyyaml dependencies to Critical Bug Summary |
| 2026-01-15 | Added: HybridNet not integrated note (exists but unused by train script) |
| 2026-01-15 | Added: TXT file details (2,334,170 lines, May 2019 - Dec 2025, ~6.5 years) |
| 2026-01-15 | Verified: No YAML loading code exists anywhere (grep confirmed) |
| 2026-01-15 | Verified: No parquet/pyarrow imports exist anywhere (grep confirmed) |
| 2026-01-15 | Confirmed: Exactly 10 Python files exist in src/ml/ |
| 2026-01-15 | **Phase 1.1 COMPLETED**: Created `src/ml/data/parquet_loader.py` - loads 33.2M rows in ~1s, full pipeline in ~53s |
| 2026-01-15 | **Phase 1.2 COMPLETED**: 3-class target variable implemented in parquet_loader.py (DOWN=19.7%, FLAT=60.2%, UP=20.1%) |
| 2026-01-15 | **Tests CREATED**: Added `tests/` directory with 26 unit tests for parquet_loader (all passing) |
| 2026-01-15 | **Dependencies FIXED**: Added pyarrow, pyyaml, pytest, pytest-cov to requirements.txt |
| 2026-01-15 | **Corrected row count**: Parquet file has 33,206,650 rows (not ~8.7M as previously estimated) |
| 2026-01-15 | RTH filtering reduces 33.2M rows to ~15.8M rows |
| 2026-01-15 | Created venv/ for dependency isolation |
| 2026-01-15 | **Phase 1.3 COMPLETED**: Created `src/ml/data/scalping_features.py` - ScalpingFeatureEngineer class with all scalping features |
| 2026-01-15 | Phase 1.3 features: returns (1,5,10,30,60s), EMAs (9,21,50,200), session VWAP, minutes-to-close, multi-timeframe, microstructure, volume delta |
| 2026-01-15 | All features normalized for neural network input, lagged to prevent lookahead bias |
| 2026-01-15 | **Tests ADDED**: Created `tests/test_scalping_features.py` with 50 comprehensive unit tests (all passing) |
| 2026-01-15 | Total test count now 76 (26 parquet_loader + 50 scalping_features) |
