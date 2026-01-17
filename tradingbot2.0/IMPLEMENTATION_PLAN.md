# Implementation Plan - MES Futures Scalping Bot

> **Last Updated**: 2026-01-17 23:15 UTC
> **Status**: **4 P0 BUGS BLOCKING ALL TRADING** - 10.17 (original) + 10.21-10.23 (NEW)
> **Verified**: All bugs confirmed via direct code inspection at specific file:line references
> **BUGS_FOUND.md**: 9 historical deployment bugs - ALL FIXED (verified)
> **Test Coverage**: 2,508 test functions across 61 test files, 91% coverage
> **10.15 FIXED**: walk_forward.py attribute mismatch - fixed 2026-01-17

---

## Executive Summary

The codebase is substantially complete (Phases 1-9 done). However, **4 critical P0 bugs must be fixed before ANY trading can begin**:

| # | Bug ID | Issue | Effort | Impact |
|---|--------|-------|--------|--------|
| ~~1~~ | ~~**10.15**~~ | ~~walk_forward.py attribute errors crash optimization~~ | ~~15 min~~ | **FIXED 2026-01-17** |
| 2 | **10.17** | 8 of 10 features wrong + 5 HTF features have LOOKAHEAD BIAS | 4-6h | Model predictions INVALID |
| 3 | **10.21** | Feature ordering not validated between training/inference | 2h | Model receives scrambled inputs |
| 4 | **10.22** | Scaler mismatch in inference (silent failure) | 1h | Unscaled features → wrong predictions |
| 5 | **10.23** | OCO double-fill race condition | 3h | Position doubled or 2x losses |

Additionally, **9 P1 bugs** affect extended trading sessions (>90 min), and **4 P2 items** need attention during paper trading.

---

## TIER 0: P0 - MUST FIX BEFORE ANY TRADING

### ~~10.15 Walk-Forward Attribute Mismatch~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/optimization/walk_forward.py:554-557` |
| **Status** | **FIXED** - Changed `param.values[0]` → `param.choices[0]`, `param.low/high` → `param.min_value/max_value` |
| **Tests** | 16/16 walk_forward tests pass, 113/113 optimization tests pass |

---

### 10.17 Feature Parity Mismatch (CRITICAL - 4-6h)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/rt_features.py` vs `src/ml/data/scalping_features.py` |
| **Effort** | 4-6 hours |
| **Impact** | Model receives different inputs during live trading than training - **8 of 10 features wrong + 5 have LOOKAHEAD BIAS** (htf_vol_5m missing in both). Predictions INVALID. |

**Detailed Feature Comparison**:

| # | Feature | scalping_features.py (Training) | rt_features.py (Live) | Issue |
|---|---------|--------------------------------|----------------------|-------|
| 1 | **macd_hist_norm** | `(macd - signal) / close` (line 388-394) | Hardcoded `0.0` (line 456) | **ALWAYS ZERO** |
| 2 | **stoch_d_norm** | `stoch_k.rolling(3).mean()` (line 402) | Uses raw stoch_k (line 472) | **NO SMOOTHING** |
| 3 | **volume_delta_norm** | 30-bar lookback (line 472-475) | 60-bar lookback (line 768) | **WRONG LOOKBACK** |
| 4 | **obv_roc** | `obv.pct_change(30)` (line 480) | 14-bar ROC (line 810) | **WRONG LOOKBACK** |
| 5 | **vwap_slope** | `vwap.pct_change(10)` (line 219) | `(vwap - close[-10]) / close[-10]` (line 355) | **USES CLOSE NOT VWAP** |
| 6 | **htf_trend_1m** | 1-bar return, **lagged 1 bar** (line 507-512) | Full-period return, **NO LAG** (line 848-872) | **LOOKAHEAD BIAS** |
| 7 | **htf_trend_5m** | Same as htf_trend_1m for 5m | Same issue | **LOOKAHEAD BIAS** |
| 8 | **htf_momentum_1m** | 5-bar pct_change, **lagged** (line 508-513) | RSI-style gains/losses, **NO LAG** (line 874-898) | **DIFFERENT CALC + LOOKAHEAD** |
| 9 | **htf_momentum_5m** | Same as htf_momentum_1m for 5m | Same issue | **DIFFERENT CALC + LOOKAHEAD** |
| 10 | **htf_vol_1m** | Rolling std, **lagged 1 bar** (line 509-514) | std × 100, **NO LAG** (line 900-926) | **DIFFERENT SCALE + LOOKAHEAD** |
| 11 | **htf_vol_5m** | Not implemented | Not implemented | **MISSING IN BOTH FILES** |

**CRITICAL FINDING**: The HTF features (trend, momentum, vol) in `scalping_features.py` are properly **lagged by 1 timeframe** before forward-filling to prevent lookahead bias. In `rt_features.py`, these features use **current prices with NO LAG**, creating potential lookahead bias during live inference.

**Features That ARE Correct (2 of 10)**: `atr_pct`, `rsi_norm` (Note: `htf_vol_5m` not implemented in either file)

**Required Fix**: Update `src/trading/rt_features.py` to match `src/ml/data/scalping_features.py` EXACTLY for all 9 mismatched features.

---

### 10.21 Feature Ordering Not Validated (CRITICAL - 2h) [NEW]

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/rt_features.py:310-546` vs `src/ml/data/scalping_features.py` |
| **Effort** | 2 hours |
| **Impact** | Model receives features in wrong order = completely scrambled inputs = random predictions |

**Problem**: Feature names are generated dynamically in `_calculate_features()` method. The order depends on execution sequence through:
1. Return periods (lines 319-327)
2. EMA periods (lines 330-345)
3. VWAP (lines 347-358)
4. Minutes to close (lines 360-364)
5. Time features (lines 366-383)
6. Volatility (lines 385-426)
7. Momentum (lines 428-473)
8. Microstructure (lines 475-507)
9. Volume (lines 509-524)
10. Multi-timeframe (lines 526-535)

**Critical Issue**: If this order differs from how `scalping_features.py` generates features during training, the model will receive completely scrambled inputs. Line 541: `self._feature_names = names` is set AFTER all features calculated.

**Required Fix**:
1. Extract feature names list from trained model checkpoint
2. Add validation in `RealTimeFeatureEngine.__init__()` that compares generated feature order against expected order
3. Raise error if mismatch detected

---

### 10.22 Scaler Mismatch in Inference (CRITICAL - 1h) [NEW]

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/live_trader.py:573-576` |
| **Effort** | 1 hour |
| **Impact** | Silent failure - model receives unscaled features if scaler missing |

**Problem**: Feature scaling depends on scaler availability:
```python
if self._scaler:
    features = self._scaler.transform(feature_vector.features.reshape(1, -1))
    tensor = torch.tensor(features, dtype=torch.float32)
else:
    tensor = feature_vector.as_tensor()  # UNSCALED - SILENT FAILURE
```

**Issues**:
1. If model was trained with scaling but scaler is not loaded → model receives unscaled features
2. No validation that scaler's expected feature count matches feature_vector dimensions
3. No logging or error when scaler is missing but should be present

**Required Fix**:
1. Add `requires_scaler` flag to model config (saved during training)
2. Validate scaler exists during model loading if `requires_scaler=True`
3. Validate scaler feature count matches feature vector dimensions
4. Log warning if scaler is None but features appear to need scaling

---

### 10.23 OCO Double-Fill Race Condition (CRITICAL - 3h) [NEW]

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/order_executor.py:710-746` |
| **Effort** | 3 hours |
| **Impact** | CRITICAL - Both OCO orders could fill, resulting in doubled position or 2x losses |

**Problem**: When one OCO order fills, the code schedules an async cancellation task (line 743). However, between when the task is created and when `_cancel_oco_orders_with_timeout()` executes, the other order could also fill.

**Race Condition Scenario**:
```
1. Stop order fills at 8000
2. _handle_fill() called for stop (line 663)
3. _handle_oco_fill() schedules cancellation of target as async task
4. Before task runs, target also fills (WebSocket message arrives)
5. Both orders now filled - position doubled or 2x losses
```

**Evidence**: Lines 743-746 show non-blocking task creation:
```python
cancel_task = asyncio.create_task(
    self._cancel_oco_orders_with_timeout(orders_to_cancel, timeout=5.0)
)
self._pending_oco_cancellations.add(cancel_task)
```

**Required Fix**:
1. Add synchronous blocking cancellation for OCO orders (don't rely on async task)
2. Track fill state atomically with mutex/lock before scheduling cancellation
3. Add "other order filled during cancellation" detection and handling
4. Implement OCO state machine with clear transitions

---

## TIER 1: P1 - FIX BEFORE EXTENDED LIVE SESSIONS (>90 min)

| Task | Location | Issue | Status | Impact | Effort |
|------|----------|-------|--------|--------|--------|
| **10C.1** | `topstepx_client.py:338-346` | WebSocket auth token expiry | **FIXED** | Token refresh works (10-min margin before 90-min expiry) | N/A |
| **10C.2** | `live_trader.py:275` | No position sync after WebSocket reconnect | **NOT FIXED** | `_sync_positions()` only called at startup; stale position state after disconnect | 2-3h |
| **10C.3** | `topstepx_ws.py` | No rate limiting for WebSocket operations | **NOT FIXED** | REST has 50/30s limit; WS invoke/send have NO throttling | 3-4h |
| **10C.4** | `risk_manager.py` + `circuit_breakers.py` | Duplicate consecutive loss tracking | **CONFIRMED** | Both track independently; inconsistent pause state | 2-3h |
| **10C.5** | `live_trader.py` | `update_market_conditions()` never called | **CONFIRMED** | Method exists (circuit_breakers.py:232-298) but never called; volatility circuit breakers inactive | 1-2h |
| **10C.6** | `topstepx_ws.py:325-328` | WebSocket concurrent connect() race | **NEW** | Multiple connections possible; lost futures | 2h |
| **10C.7** | `topstepx_client.py:337-346` | Token refresh race condition | **NEW** | Multiple authentications simultaneously | 1h |
| **10C.8** | `topstepx_ws.py:334-335` | Session leak on reconnect | **NEW** | Old sessions never closed; resource exhaustion | 1h |
| **10C.9** | `order_executor.py:799-806` | Cancel order doesn't verify | **NEW** | Order removed from tracking without confirming cancellation | 2h |
| **10C.10** | `order_executor.py:393-407` | Stop order failure continues | **NEW** | Position created without stop loss protection | 1h |

### 10C.2 Details: Position Sync After Reconnect

**Problem**: `_sync_positions()` is only called during startup (line 275), not after WebSocket reconnect in `_auto_reconnect_loop()`.

**Fix**: Add reconnection callback to `TopstepXWebSocket` that calls `_sync_positions()`.

### 10C.3 Details: WebSocket Rate Limiting

**Problem**: REST API has `RateLimiter` (50 req/30s in topstepx_rest.py) but WebSocket `invoke`/`send` have no throttling.

**Fix**: Add rate limiter to WebSocket operations similar to REST.

### 10C.4 Details: Duplicate Tracking

**Problem**: Both modules track consecutive losses independently:
- Circuit breakers: lines 126, 173-191 (`_consecutive_losses`)
- Risk manager: `state.consecutive_losses`
- Both updated in `live_trader.py:484-489`

**Fix**: Consolidate tracking into CircuitBreakers only; RiskManager should delegate.

### 10C.5 Details: Dead Code

**Problem**: `CircuitBreakers.update_market_conditions()` (lines 232-298) handles HIGH_VOLATILITY, WIDE_SPREAD, LOW_VOLUME breakers but is NEVER called anywhere in the codebase.

**Fix**: Add periodic market condition updates in trading loop (every minute).

### 10C.6 Details: WebSocket Concurrent Connect Race [NEW]

**Problem**: The state check in `SignalRConnection.connect()` (line 327) is not atomic. Two concurrent `connect()` calls can both pass the check before either updates state to `CONNECTING` (line 330).

**Race Condition**:
```
Thread A: checks state = DISCONNECTED ✓
Thread B: checks state = DISCONNECTED ✓
Both proceed to create connections
```

**Fix**: Add mutex/lock around connection state check and update.

### 10C.7 Details: Token Refresh Race Condition [NEW]

**Problem**: `_auth_lock` is only held during `authenticate()` call, not during the refresh check in `_ensure_authenticated()`. Multiple concurrent requests could see the token is expiring and all call `authenticate()` simultaneously.

**Fix**: Hold `_auth_lock` during the entire expiry check + refresh flow.

### 10C.8 Details: Session Leak on Reconnect [NEW]

**Problem**: If `connect()` is called multiple times with `self._session = None`, new sessions are created but old sessions may not be properly closed. Sessions stored in `self._session` can be overwritten.

**Fix**: Ensure old session is closed before creating new one; add session lifecycle tracking.

### 10C.9 Details: Cancel Order Doesn't Verify [NEW]

**Problem**: `_cancel_order_safe()` calls `cancel_order()` and removes from local tracking, but doesn't verify if cancel was successful. If API returns error but exception is caught, order remains open on broker while removed from local tracking.

**Fix**: Verify cancel response status before removing from tracking; query order state if uncertain.

### 10C.10 Details: Stop Order Failure Continues [NEW]

**Problem**: In `execute_entry()` (lines 393-407), stop order is placed first, then target order. If stop order placement returns None (line 603), the code continues anyway and tries to track it. This can result in a position without stop loss protection.

**Fix**: Return early if stop order fails; do not proceed to target order without valid stop.

---

## TIER 2: P2 - ADDRESS DURING PAPER TRADING

| Task | Location | Issue | Status | Impact | Effort |
|------|----------|-------|--------|--------|--------|
| **10.16** | `trade_logger.py:253` | Slippage calculation | **NOT A BUG** | `net_pnl = gross - comm - slip` is CORRECT | N/A |
| **10.18** | `neural_networks.py` | Missing Transformer architecture | **CONFIRMED** | Only FeedForward + LSTM + HybridNet (spec requires Transformer) | 4-6h |
| **10.19** | `risk_manager.py:231-282` | EOD manager NOT queried in `approve_trade()` | **PARTIAL** | EOD check exists in trading loop but not in approve_trade() as defense-in-depth | 2h |
| **10.20** | `position_sizing.py:118-217` | EOD size multiplier NOT applied | **CONFIRMED** | `calculate()` doesn't use `EODManager.get_position_size_multiplier()` | 1h |

### 10.20 Details: EOD Size Multiplier

**Problem**:
- `PositionSizer.calculate()` has no time awareness (lines 118-217)
- `EODManager.get_position_size_multiplier()` exists (lines 281-292) but is not called
- Positions should be reduced 50% after 4:00 PM NY per spec

**Fix**: Add `eod_multiplier` parameter to `calculate()` and apply it to final position size.

---

## TIER 3: P3 - NICE TO HAVE

| Task | Location | Issue | Effort |
|------|----------|-------|--------|
| **10.10** | `src/ml/data/` | No memory estimation for large datasets | 2-3h |
| **10.11** | Various | Config versioning, HybridNet documentation | 1-2h |

---

## TEST COVERAGE STATUS

**9 modules lack dedicated unit test files** but are well-tested via multiple integration tests:

| Module | File | Dedicated Test | Actual Coverage |
|--------|------|----------------|-----------------|
| position_sizing.py | `src/risk/` | No dedicated file | 3 test files (test_risk_manager.py, test_go_live_thresholds.py, test_go_live_validation.py) |
| stops.py | `src/risk/` | No dedicated file | test_risk_manager.py |
| eod_manager.py | `src/risk/` | No dedicated file | 4 test files including dedicated test_eod_integration.py |
| costs.py | `src/backtest/` | No dedicated file | test_backtest.py |
| slippage.py | `src/backtest/` | No dedicated file | test_backtest.py |
| metrics.py | `src/backtest/` | No dedicated file | 7 test files |
| trade_logger.py | `src/backtest/` | No dedicated file | test_backtest.py |
| signal_generator.py | `src/trading/` | No dedicated file | 8 test files |
| rt_features.py | `src/trading/` | No dedicated file | 8 test files |

**Note**: While dedicated unit test files do not exist, these modules have substantial test coverage across multiple integration test files. P0/P1 bug fixes take priority over reorganizing test structure.

---

## IMPLEMENTATION SEQUENCE

```
═══════════════════════════════════════════════════════════════════════════════
 PHASE 1: P0 FIXES (BLOCKING - NO TRADING UNTIL COMPLETE) - ~12h total
═══════════════════════════════════════════════════════════════════════════════

Day 1 (Critical Path):
├── 10.15 Walk-forward attribute fix (15 min) ← QUICK WIN
├── 10.21 Feature ordering validation (2h)
├── 10.22 Scaler mismatch validation (1h)
└── 10.17 Feature parity - macd_hist_norm, stoch_d_norm (2h)

Day 2 (Critical Path):
├── 10.17 Feature parity - volume_delta_norm, obv_roc, vwap_slope (3h)
└── 10.23 OCO double-fill race condition fix (3h)

Day 3 (Critical Path):
├── 10.17 Feature parity - htf_trend/momentum/vol with proper lagging (3h)
├── Expand test_feature_parity.py with strict equality tests (1h)
└── Integration testing for all P0 fixes (2h)

═══════════════════════════════════════════════════════════════════════════════
 PHASE 2: P1 FIXES (REQUIRED FOR >90 MIN SESSIONS) - ~16h total
═══════════════════════════════════════════════════════════════════════════════

Day 4-5:
├── 10C.2 Position sync after reconnect (3h)
├── 10C.5 Call update_market_conditions() (2h)
├── 10C.6 WebSocket concurrent connect() race fix (2h)
├── 10C.7 Token refresh race condition fix (1h)
└── 10C.8 Session leak on reconnect (1h)

Day 6-7:
├── 10C.3 WebSocket rate limiting (3h)
├── 10C.4 Consolidate consecutive loss tracking (2h)
├── 10C.9 Cancel order verification (2h)
├── 10C.10 Stop order failure handling (1h)
└── Integration testing for all P1 fixes (2h)

═══════════════════════════════════════════════════════════════════════════════
 PHASE 3: PAPER TRADING + P2 FIXES - ~10h total
═══════════════════════════════════════════════════════════════════════════════

During Paper Trading:
├── 10.18 Transformer architecture (4-6h)
├── 10.19 EOD check in approve_trade() (2h)
├── 10.20 EOD size multiplier (1h)
└── Monitor and validate all fixes in live environment
```

---

## WHAT IS COMPLETE (Do NOT Re-implement)

### Phases Complete
- **Phase 1**: Data Pipeline - Parquet loader, scalping features, 3-class targets
- **Phase 2**: Risk Management - All 5 modules
- **Phase 3**: Backtesting - Engine, costs, slippage, metrics, trade_logger, visualization, go_live_validator
- **Phase 4**: ML Models - 3-class classification, FeedForward, LSTM, HybridNet
- **Phase 5**: TopstepX API - REST + WebSocket with auto-reconnect
- **Phase 6**: Live Trading - All 6 modules
- **Phase 7**: DataBento Client
- **Phase 8**: Testing Infrastructure - 2513 tests, 91% coverage
- **Phase 9**: Parameter Optimization - Grid, random, Bayesian optimizers

### All 11 Previous P0 Bugs VERIFIED FIXED (2026-01-16)
1. Risk manager `approve_trade()` IS called (lines 480-487 in live_trader.py)
2. WebSocket auto-reconnect task IS started (added asyncio.create_task in connect())
3. EOD phase method IS correct (`get_status().phase`)
4. Backtest slippage IS deducted (`net_pnl = gross_pnl - commission - slippage_cost`)
5. OCO cancellation race condition FIXED (timeout + verification added, lines 699-798)
6. Daily loss check IS in trading loop (`can_trade()` at lines 321-328)
7. Circuit breaker IS instantiated (import line 35, instance line 263)
8. Account drawdown check EXISTS (MANUAL_REVIEW at lines 330-338)
9. 7 features HAVE calculation methods (but parity issues remain - see 10.17)
10. OOS evaluation USES separate data (`holdout_objective_fn` parameter added)
11. LSTM tuple handling WORKS

### Core Functionality VERIFIED Working
- approve_trade() called at line 631 before orders
- Circuit breaker checked at line 529
- OCO management with timeout (lines 699-798)
- Position sync on startup (line 275)
- All 5 signal types implemented (LONG_ENTRY, SHORT_ENTRY, EXIT_LONG, EXIT_SHORT, FLATTEN)
- Token refresh works with 10-min margin before 90-min expiry
- Transaction costs: $0.84 RT
- Slippage model: Tick-based (1-4 ticks)

---

## CRITICAL FILES FOR IMPLEMENTATION

| Priority | File | Lines | Issue |
|----------|------|-------|-------|
| **P0** | `src/optimization/walk_forward.py` | 554-557 | Fix attribute names (10.15) |
| **P0** | `src/trading/rt_features.py` | 310-546, 355, 456, 472, 768, 810, 848-926 | Feature parity + ordering (10.17, 10.21) |
| **P0** | `src/ml/data/scalping_features.py` | Reference | Source of truth for features |
| **P0** | `src/trading/live_trader.py` | 573-576 | Scaler mismatch validation (10.22) |
| **P0** | `src/trading/order_executor.py` | 710-746 | OCO double-fill race (10.23) |
| **P1** | `src/trading/live_trader.py` | 275 | Add reconnect sync (10C.2) |
| **P1** | `src/api/topstepx_ws.py` | 325-328, 334-335 | Rate limiting + connect race + session leak (10C.3, 10C.6, 10C.8) |
| **P1** | `src/api/topstepx_client.py` | 337-346 | Token refresh race (10C.7) |
| **P1** | `src/risk/circuit_breakers.py` | 232-298 | Activate update_market_conditions() (10C.5) |
| **P1** | `src/trading/order_executor.py` | 393-407, 799-806 | Stop failure + cancel verify (10C.9, 10C.10) |
| **P2** | `src/risk/position_sizing.py` | 118-217 | Add EOD multiplier (10.20) |
| **P2** | `src/risk/risk_manager.py` | 231-282 | Add EOD check to approve_trade() (10.19) |
| **P2** | `src/ml/models/neural_networks.py` | 98-524 | Add Transformer architecture (10.18) |

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

## Test Coverage Summary

**Total Tests**: 2513 tests (all passing)
**Coverage**: 91% (target: >80%)

### Test Breakdown by Module
- Phase 1 Data Pipeline: 76 tests
- Phase 2 Risk Management: 77 tests
- Phase 3 Backtesting: 84 tests
- Phase 4 ML Models: 55 tests
- Phase 5 TopstepX API: 77 tests
- Phase 6 Live Trading: 77 tests
- Phase 7 DataBento: 43 tests
- Phase 8-9: Extended coverage + Optimization: 200+ tests
- Integration Tests: 88 tests
- Go-Live Validation: 170+ tests

---

## Specification Coverage

| Spec File | Coverage | Notes |
|-----------|----------|-------|
| `specs/ml-scalping-model.md` | 90% | Missing Transformer architecture |
| `specs/risk-management.md` | 100% | All risk limits enforced |
| `specs/backtesting-engine.md` | 100% | Full cost/slippage modeling |
| `specs/topstepx-api-integration.md` | 100% | REST + WebSocket complete |
| `specs/live-trading-execution.md` | 85% | Feature parity issue (10.17) |
| `specs/databento-historical-data.md` | 100% | Parquet loading complete |

---

## Data Asset

227MB parquet file at `data/historical/MES/MES_1s_2years.parquet` (33.2M rows) - fully integrated.

---

## Appendix A: Verified Fixes from 2026-01-16

All 11 previous P0 bugs have been verified fixed and should NOT be re-investigated.

---

## Appendix B: 2026-01-17 22:00 UTC Verification (Opus 4.5 + 8 Sonnet Subagents)

### Codebase Health Check
- **TODOs/FIXMEs**: None found in src/
- **Incomplete stubs**: None found (all `pass` statements are legitimate exception classes or abstract methods)
- **NotImplementedError**: None found
- **Custom exception classes**: 9 (properly implemented)
- **Abstract methods**: All have implementations in subclasses

### BUGS_FOUND.md Status (9 Deployment Bugs)
All bugs discovered during RunPod training have been **VERIFIED FIXED**:
1. Infinity values in feature scaling - FIXED (scalping_features.py:638)
2. Infinity in validation/test sets - FIXED (train_scalping_model.py:409-419)
3. CUDA OOM during test evaluation - FIXED (batched processing)
4. LSTM output tuple unpacking - FIXED
5. Container memory limit - WORKAROUND (--max-samples)
6. ScalpingFeatureEngineer constructor - FIXED (run_backtest.py:407)
7. Model loading config key mismatch - FIXED (run_backtest.py:256-266)
8. FeedForwardNet parameter names - FIXED (run_backtest.py:270-277)
9. PerformanceMetrics attribute names - FIXED (run_backtest.py:495-498)

### P0 Bugs Verified
| Bug | Status | Evidence |
|-----|--------|----------|
| **10.15** | CONFIRMED | walk_forward.py:554-557 uses param.values/low/high instead of choices/min_value/max_value |
| **10.17** | CONFIRMED | 8 of 10 features wrong + 5 HTF features have lookahead bias (see detailed table) |
| **10.21** | NEW | Feature ordering not validated - rt_features.py:310-546 generates order dynamically |
| **10.22** | NEW | Scaler mismatch silent failure - live_trader.py:573-576 fallback to unscaled |
| **10.23** | NEW | OCO double-fill race - order_executor.py:710-746 async task scheduling |

### P1 Bugs Verified
| Bug | Status | Evidence |
|-----|--------|----------|
| **10C.2** | CONFIRMED | _sync_positions only at startup line 275, not after reconnect |
| **10C.3** | CONFIRMED | REST has RateLimiter but WebSocket invoke/send have no throttling |
| **10C.4** | CONFIRMED | Duplicate tracking in circuit_breakers + risk_manager |
| **10C.5** | CONFIRMED | update_market_conditions() never called (method at circuit_breakers.py:232-298) |
| **10C.6** | NEW | WebSocket connect() race - topstepx_ws.py:325-328 non-atomic state check |
| **10C.7** | NEW | Token refresh race - topstepx_client.py:337-346 lock not held during check |
| **10C.8** | NEW | Session leak - topstepx_ws.py:334-335 old sessions not closed |
| **10C.9** | NEW | Cancel order no verify - order_executor.py:799-806 removes without confirmation |
| **10C.10** | NEW | Stop failure continues - order_executor.py:393-407 position without stop |

### P2 Bugs Verified
| Bug | Status | Evidence |
|-----|--------|----------|
| **10.18** | CONFIRMED | Only FeedForward, LSTM, HybridNet in neural_networks.py:98-524 |
| **10.19** | PARTIAL | EOD check in trading loop but not in approve_trade() |
| **10.20** | CONFIRMED | position_sizing.py doesn't use EODManager.get_position_size_multiplier() |

### Test Infrastructure Analysis
- **Total test files**: 61 (58 unit/extended + 4 integration)
- **Total test functions**: 2,508
- **Total test lines**: 45,389
- **Skipped tests**: 4 (intentional - require real data or complex mocking)
- **Tests without assertions**: 14 (plotting tests verify execution, not output)
- **Flaky patterns**: ~100+ async/time-dependent tests with hardcoded delays

### Test Coverage Gaps (31 modules without dedicated tests)
These modules are tested via integration tests but lack dedicated unit test files:
- **API**: topstepx_client.py, topstepx_rest.py, topstepx_ws.py
- **Backtest**: costs.py, engine.py, metrics.py, slippage.py, trade_logger.py
- **Optimization**: bayesian_optimizer.py, grid_search.py, optimizer_base.py, parameter_space.py, random_search.py, results.py
- **ML**: neural_networks.py
- **Library**: config.py, constants.py, logging_utils.py

### Feature Parity Test Analysis
- **File**: test_feature_parity.py exists
- **Thresholds**: CORRELATION_THRESHOLD = 0.5, MAE_THRESHOLD = 0.3 (RELAXED)
- **Issue**: Tests document formula differences but use relaxed thresholds
- **Recommendation**: Tighten thresholds after fixing 10.17

### src/lib Shared Utilities
All 6 modules are complete and well-designed:
- **constants.py**: Contract specs, risk params, API config
- **time_utils.py**: NY timezone handling, EOD phases, market calendar
- **config.py**: Hierarchical config (env > YAML > defaults)
- **logging_utils.py**: Structured trading logs with rotation
- **performance_monitor.py**: Latency tracking with thresholds
- **alerts.py**: Multi-channel notifications (console, email, Slack, Discord)
