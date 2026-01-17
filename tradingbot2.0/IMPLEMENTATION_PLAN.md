# Implementation Plan - MES Futures Scalping Bot

> **Last Updated**: 2026-01-17 UTC
> **Status**: **ALL P0 BUGS FIXED** | **ALL P1 BUGS FIXED** | **ALL P2 BUGS FIXED**
> **Verified**: All bugs confirmed via direct code inspection at specific file:line references
> **BUGS_FOUND.md**: 9 historical deployment bugs - ALL FIXED (verified)
> **Test Coverage**: 2,520 test functions across 61 test files, 91% coverage
> **10.15 FIXED**: walk_forward.py attribute mismatch - fixed 2026-01-17
> **10.17 FIXED**: Feature parity mismatch (8 features + HTF lagging) - fixed 2026-01-17
> **10.21 FIXED**: Feature ordering validation - fixed 2026-01-17
> **10.22 FIXED**: Scaler mismatch validation - fixed 2026-01-17
> **10.23 FIXED**: OCO double-fill race condition - fixed 2026-01-17
> **10.19 FIXED**: EOD check in approve_trade() - fixed 2026-01-17
> **10.20 FIXED**: EOD size multiplier in position sizing - fixed 2026-01-17
> **10.18 FIXED**: Transformer architecture implemented - fixed 2026-01-17

---

## Executive Summary

The codebase is substantially complete (Phases 1-9 done). **All P0 bugs have been fixed** and the system is ready for paper trading:

| # | Bug ID | Issue | Effort | Impact |
|---|--------|-------|--------|--------|
| ~~1~~ | ~~**10.15**~~ | ~~walk_forward.py attribute errors crash optimization~~ | ~~15 min~~ | **FIXED 2026-01-17** |
| ~~2~~ | ~~**10.17**~~ | ~~8 of 10 features wrong + 5 HTF features have LOOKAHEAD BIAS~~ | ~~4-6h~~ | **FIXED 2026-01-17** |
| ~~3~~ | ~~**10.21**~~ | ~~Feature ordering not validated between training/inference~~ | ~~2h~~ | **FIXED 2026-01-17** |
| ~~4~~ | ~~**10.22**~~ | ~~Scaler mismatch in inference (silent failure)~~ | ~~1h~~ | **FIXED 2026-01-17** |
| ~~5~~ | ~~**10.23**~~ | ~~OCO double-fill race condition~~ | ~~3h~~ | **FIXED 2026-01-17** |

Additionally, **all 9 P1 bugs have been FIXED** (2026-01-17), **ALL 3 P2 bugs FIXED** (10.18, 10.19, 10.20), and only **P3 items** remain for paper trading.

---

## TIER 0: P0 - MUST FIX BEFORE ANY TRADING

### ~~10.15 Walk-Forward Attribute Mismatch~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/optimization/walk_forward.py:554-557` |
| **Status** | **FIXED** - Changed `param.values[0]` → `param.choices[0]`, `param.low/high` → `param.min_value/max_value` |
| **Tests** | 16/16 walk_forward tests pass, 113/113 optimization tests pass |

---

### ~~10.17 Feature Parity Mismatch~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/rt_features.py` vs `src/ml/data/scalping_features.py` |
| **Status** | **FIXED** |
| **Changes** | 1. Fixed all 8 feature calculations in rt_features.py to match scalping_features.py |
| | 2. Fixed HTF features to use proper lagging (using `_calculate_htf_features()` with aggregated bars) |
| | 3. Added helper methods: `_get_macd_signal()`, `_get_stoch_d()`, `_get_vwap_slope()` |
| | 4. Added state tracking: `_macd_signal`, `_stoch_k_history`, `_vwap_history` |
| | 5. Changed `volume_delta_norm` lookback from 60 to 30 |
| | 6. Changed `obv_roc` lookback from 14 to 30 |
| | 7. Fixed `vwap_slope` to use VWAP history instead of price history |
| **Tests** | All feature parity tests pass with strict equality |

---

### ~~10.21 Feature Ordering Not Validated~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/rt_features.py:217-358` |
| **Status** | **FIXED** |
| **Changes** | 1. Added `expected_feature_names` parameter to `RealTimeFeatureEngine.__init__()` |
| | 2. Added `set_expected_feature_names()` for late binding |
| | 3. Added `_validate_feature_order()` that compares generated vs expected features |
| | 4. Validation runs on first feature generation, raises `RuntimeError` on mismatch |
| | 5. Updated `live_trader.py:_load_model()` to extract and pass feature names from checkpoint |
| **Tests** | 11/11 RealTimeFeatureEngine tests pass, 176/176 related tests pass |

---

### ~~10.22 Scaler Mismatch in Inference~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/live_trader.py:184-188, 573-656, 753-783` |
| **Status** | **FIXED** |
| **Changes** | 1. Added `_requires_scaler` and `_scaler_validated` flags to track scaler requirements |
| | 2. `_load_model()` now sets `_requires_scaler=True` when scaler is loaded from checkpoint |
| | 3. Added `_validate_scaler()` method that validates on first inference |
| | 4. Raises `RuntimeError` if scaler required but missing, or dimensions mismatch |
| | 5. Logs warning if running without scaler |
| **Tests** | 134/134 live_trader tests pass (tests updated to use proper mock scalers) |

---

### ~~10.23 OCO Double-Fill Race Condition~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/trading/order_executor.py:710-746` |
| **Status** | **FIXED** |
| **Changes** | 1. Added `_order_lock` for thread-safe `_open_orders` access |
| | 2. Added `_orders_being_cancelled` set to prevent duplicate cancellations |
| | 3. Added `_filled_oco_orders` set to detect dual fill race conditions |
| | 4. Added dual fill detection with CRITICAL logging |
| | 5. Improved `_cancel_order_safe` to verify cancellation success |
| | 6. Improved logging throughout OCO cancellation flow |
| **Tests** | All OCO and order executor tests pass |

---

## TIER 1: P1 - FIX BEFORE EXTENDED LIVE SESSIONS (>90 min) - ALL FIXED

| Task | Location | Issue | Status | Impact | Effort |
|------|----------|-------|--------|--------|--------|
| **10C.1** | `topstepx_client.py:338-346` | WebSocket auth token expiry | **FIXED** | Token refresh works (10-min margin before 90-min expiry) | N/A |
| **10C.2** | `live_trader.py:275` | No position sync after WebSocket reconnect | **FIXED 2026-01-17** | Added reconnect callback mechanism | N/A |
| **10C.3** | `topstepx_ws.py` | No rate limiting for WebSocket operations | **FIXED 2026-01-17** | Added rate limiting to invoke/send | N/A |
| **10C.4** | `risk_manager.py` + `circuit_breakers.py` | Duplicate consecutive loss tracking | **FIXED 2026-01-17** | Consolidated into CircuitBreakers, RiskManager delegates | N/A |
| **10C.5** | `live_trader.py` | `update_market_conditions()` never called | **FIXED 2026-01-17** | Added periodic calls in trading loop | N/A |
| **10C.6** | `topstepx_ws.py:325-328` | WebSocket concurrent connect() race | **FIXED 2026-01-17** | Added _connect_lock | N/A |
| **10C.7** | `topstepx_client.py:337-346` | Token refresh race condition | **FIXED 2026-01-17** | Extended lock scope in _ensure_authenticated | N/A |
| **10C.8** | `topstepx_ws.py:334-335` | Session leak on reconnect | **FIXED 2026-01-17** | Added session lifecycle tracking | N/A |
| **10C.9** | `order_executor.py:799-806` | Cancel order doesn't verify | **FIXED 2026-01-17** | Added retry verification before removing from tracking | N/A |
| **10C.10** | `order_executor.py:393-407` | Stop order failure continues | **FIXED 2026-01-17** | Added early return and emergency exit on stop failure | N/A |

### ~~10C.2 Details: Position Sync After Reconnect~~ ✅ FIXED (2026-01-17)

**Problem**: `_sync_positions()` is only called during startup (line 275), not after WebSocket reconnect in `_auto_reconnect_loop()`.

**Fix**: Added reconnect callback mechanism to `TopstepXWebSocket` that calls `_sync_positions()`.

### ~~10C.3 Details: WebSocket Rate Limiting~~ ✅ FIXED (2026-01-17)

**Problem**: REST API has `RateLimiter` (50 req/30s in topstepx_rest.py) but WebSocket `invoke`/`send` have no throttling.

**Fix**: Added rate limiting to invoke/send operations similar to REST.

### ~~10C.4 Details: Duplicate Tracking~~ ✅ FIXED (2026-01-17)

**Problem**: Both modules track consecutive losses independently:
- Circuit breakers: lines 126, 173-191 (`_consecutive_losses`)
- Risk manager: `state.consecutive_losses`
- Both updated in `live_trader.py:484-489`

**Fix**: Consolidated tracking into CircuitBreakers only; RiskManager now delegates to CircuitBreakers.

### ~~10C.5 Details: Dead Code~~ ✅ FIXED (2026-01-17)

**Problem**: `CircuitBreakers.update_market_conditions()` (lines 232-298) handles HIGH_VOLATILITY, WIDE_SPREAD, LOW_VOLUME breakers but is NEVER called anywhere in the codebase.

**Fix**: Added periodic market condition updates in trading loop (every minute).

### ~~10C.6 Details: WebSocket Concurrent Connect Race~~ ✅ FIXED (2026-01-17)

**Problem**: The state check in `SignalRConnection.connect()` (line 327) is not atomic. Two concurrent `connect()` calls can both pass the check before either updates state to `CONNECTING` (line 330).

**Race Condition**:
```
Thread A: checks state = DISCONNECTED ✓
Thread B: checks state = DISCONNECTED ✓
Both proceed to create connections
```

**Fix**: Added `_connect_lock` mutex around connection state check and update.

### ~~10C.7 Details: Token Refresh Race Condition~~ ✅ FIXED (2026-01-17)

**Problem**: `_auth_lock` is only held during `authenticate()` call, not during the refresh check in `_ensure_authenticated()`. Multiple concurrent requests could see the token is expiring and all call `authenticate()` simultaneously.

**Fix**: Extended `_auth_lock` scope to cover the entire expiry check + refresh flow in `_ensure_authenticated()`.

### ~~10C.8 Details: Session Leak on Reconnect~~ ✅ FIXED (2026-01-17)

**Problem**: If `connect()` is called multiple times with `self._session = None`, new sessions are created but old sessions may not be properly closed. Sessions stored in `self._session` can be overwritten.

**Fix**: Added session lifecycle tracking; old session is now properly closed before creating new one.

### ~~10C.9 Details: Cancel Order Doesn't Verify~~ ✅ FIXED (2026-01-17)

**Problem**: `_cancel_order_safe()` calls `cancel_order()` and removes from local tracking, but doesn't verify if cancel was successful. If API returns error but exception is caught, order remains open on broker while removed from local tracking.

**Fix**: Added retry verification before removing from tracking; now queries order state if uncertain.

### ~~10C.10 Details: Stop Order Failure Continues~~ ✅ FIXED (2026-01-17)

**Problem**: In `execute_entry()` (lines 393-407), stop order is placed first, then target order. If stop order placement returns None (line 603), the code continues anyway and tries to track it. This can result in a position without stop loss protection.

**Fix**: Added early return on stop order failure; emergency exit triggered if stop cannot be placed.

---

## TIER 2: P2 - ADDRESS DURING PAPER TRADING

| Task | Location | Issue | Status | Impact | Effort |
|------|----------|-------|--------|--------|--------|
| **10.16** | `trade_logger.py:253` | Slippage calculation | **NOT A BUG** | `net_pnl = gross - comm - slip` is CORRECT | N/A |
| ~~**10.18**~~ | ~~`neural_networks.py`~~ | ~~Missing Transformer architecture~~ | **FIXED 2026-01-17** | Transformer architecture implemented | N/A |
| ~~**10.19**~~ | ~~`risk_manager.py:272-327`~~ | ~~EOD manager NOT queried in `approve_trade()`~~ | **FIXED 2026-01-17** | N/A | N/A |
| ~~**10.20**~~ | ~~`position_sizing.py:118-251`~~ | ~~EOD size multiplier NOT applied~~ | **FIXED 2026-01-17** | N/A | N/A |

### ~~10.19 Details: EOD Check in approve_trade()~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/risk/risk_manager.py:272-327` |
| **Status** | **FIXED** |
| **Changes** | 1. Added `eod_manager` parameter to `RiskManager.__init__()` |
| | 2. Added `set_eod_manager()` method for late binding |
| | 3. Added EOD phase check in `approve_trade()` that rejects trades during CLOSE_ONLY, AGGRESSIVE_EXIT, MUST_BE_FLAT, AFTER_HOURS phases |
| | 4. Updated `live_trader.py` to link EODManager to RiskManager |
| **Tests** | All 2520 tests pass |

### ~~10.20 Details: EOD Size Multiplier~~ ✅ FIXED (2026-01-17)

| Attribute | Value |
|-----------|-------|
| **Location** | `src/risk/position_sizing.py:118-251` |
| **Status** | **FIXED** |
| **Changes** | 1. Added `eod_multiplier` field to `PositionSizeResult` dataclass |
| | 2. Added `eod_multiplier` parameter to `calculate()` method (default 1.0) |
| | 3. Apply EOD multiplier after confidence scaling, before tier capping |
| | 4. Returns 0 contracts when `eod_multiplier <= 0` (no trading allowed) |
| | 5. Updated `live_trader.py` to get EOD multiplier from `_eod_manager.get_position_size_multiplier()` |
| **Tests** | All 2520 tests pass |

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

**Total Tests**: 2520 tests (all passing)
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

### P1 Bugs Verified - ALL FIXED (2026-01-17)
| Bug | Status | Fix Applied |
|-----|--------|----------|
| **10C.2** | **FIXED** | Added reconnect callback mechanism |
| **10C.3** | **FIXED** | Added rate limiting to invoke/send |
| **10C.4** | **FIXED** | Consolidated into CircuitBreakers, RiskManager delegates |
| **10C.5** | **FIXED** | Added periodic calls in trading loop |
| **10C.6** | **FIXED** | Added _connect_lock |
| **10C.7** | **FIXED** | Extended lock scope in _ensure_authenticated |
| **10C.8** | **FIXED** | Added session lifecycle tracking |
| **10C.9** | **FIXED** | Added retry verification before removing from tracking |
| **10C.10** | **FIXED** | Added early return and emergency exit on stop failure |

### P2 Bugs Verified
| Bug | Status | Evidence |
|-----|--------|----------|
| **10.18** | **FIXED** | Transformer architecture implemented in neural_networks.py |
| **10.19** | PARTIAL | EOD check in trading loop but not in approve_trade() |
| **10.20** | CONFIRMED | position_sizing.py doesn't use EODManager.get_position_size_multiplier() |

### Test Infrastructure Analysis
- **Total test files**: 61 (58 unit/extended + 4 integration)
- **Total test functions**: 2,520
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
