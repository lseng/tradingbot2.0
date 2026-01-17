# Implementation Plan - MES Futures Scalping Bot

> **Last Updated**: 2026-01-17 UTC
> **Status**: **READY FOR PAPER TRADING** - All P0, P1, P2 bugs fixed
> **Test Coverage**: 2,547 tests passing, 91% coverage
> **Git Tag**: v0.0.64

---

## Current Status

| Priority | Status | Summary |
|----------|--------|---------|
| **P0** | ✅ ALL FIXED | 5 critical bugs blocking trading |
| **P1** | ✅ ALL FIXED | 9 extended session bugs (>90 min) |
| **P2** | ✅ ALL FIXED | 3 paper trading enhancements |
| **P3** | ⏳ PENDING | 2 nice-to-have items |

---

## TIER 3: P3 - NICE TO HAVE (Remaining Work)

| Task | Location | Issue | Effort |
|------|----------|-------|--------|
| **10.10** | `src/ml/data/` | No memory estimation for large datasets | 2-3h |
| **10.11** | Various | Config versioning, HybridNet documentation | 1-2h |

### 10.10 Details: Memory Estimation for Large Datasets

**Problem**: Data loaders load files directly without estimating memory requirements first. Can cause OOM on large datasets.

**Required Implementation**:
- Pre-load size estimation for parquet/CSV files
- System memory availability check
- Warning/blocking if dataset > available memory
- Chunked/streaming loading option

**Recommended Location**: `src/ml/data/memory_utils.py` (NEW FILE)

### 10.11 Details: Config Versioning and HybridNet Documentation

**Config Versioning** - Missing:
- No version field in `default_config.yaml`
- No schema validation for config changes
- No backwards compatibility handling

**HybridNet Documentation** - Current docstring is minimal (7 lines). Needs:
- Architectural diagram
- Usage examples comparing vs FeedForward/LSTM/Transformer
- Hyperparameter tuning guidance

---

## Completed Bug Fixes (Historical Reference)

### P0 Bugs (Fixed 2026-01-17)
| Bug | Issue | Fix |
|-----|-------|-----|
| 10.15 | walk_forward.py attribute mismatch | Changed param.values/low/high → choices/min_value/max_value |
| 10.17 | Feature parity (8 features wrong, HTF lookahead) | Fixed all calculations in rt_features.py |
| 10.21 | Feature ordering not validated | Added validation in RealTimeFeatureEngine |
| 10.22 | Scaler mismatch in inference | Added _requires_scaler flag and validation |
| 10.23 | OCO double-fill race condition | Added _order_lock, _filled_oco_orders tracking |

### P1 Bugs (Fixed 2026-01-17)
| Bug | Issue | Fix |
|-----|-------|-----|
| 10C.1 | Token expiry | Token refresh with 10-min margin |
| 10C.2 | No position sync after reconnect | Added reconnect callback mechanism |
| 10C.3 | No WebSocket rate limiting | Added rate limiting to invoke/send |
| 10C.4 | Duplicate consecutive loss tracking | Consolidated into CircuitBreakers |
| 10C.5 | update_market_conditions() never called | Added periodic calls in trading loop |
| 10C.6 | WebSocket concurrent connect() race | Added _connect_lock |
| 10C.7 | Token refresh race condition | Extended lock scope |
| 10C.8 | Session leak on reconnect | Added session lifecycle tracking |
| 10C.9 | Cancel order doesn't verify | Added retry verification |
| 10C.10 | Stop order failure continues | Added early return and emergency exit |

### P2 Bugs (Fixed 2026-01-17)
| Bug | Issue | Fix |
|-----|-------|-----|
| 10.18 | Missing Transformer architecture | Implemented TransformerNet |
| 10.19 | EOD not checked in approve_trade() | Added EOD phase check |
| 10.20 | EOD size multiplier not applied | Added eod_multiplier to position sizing |

---

## System Overview

### Phases Complete
- **Phase 1**: Data Pipeline - Parquet loader, scalping features, 3-class targets
- **Phase 2**: Risk Management - All 5 modules
- **Phase 3**: Backtesting - Engine, costs, slippage, metrics, trade_logger
- **Phase 4**: ML Models - FeedForward, LSTM, HybridNet, TransformerNet
- **Phase 5**: TopstepX API - REST + WebSocket with auto-reconnect
- **Phase 6**: Live Trading - All 6 modules
- **Phase 7**: DataBento Client
- **Phase 8**: Testing - 2,547 tests, 91% coverage
- **Phase 9**: Parameter Optimization - Grid, random, Bayesian

### Core Functionality Verified
- approve_trade() enforces all risk limits
- Circuit breakers active with market condition updates
- OCO management with thread-safe cancellation
- Position sync on startup and reconnect
- EOD flatten at 4:30 PM NY
- Token refresh with 10-min margin

---

## Quick Reference

| Parameter | Value |
|-----------|-------|
| Starting Capital | $1,000 |
| MES Tick Size | 0.25 points ($1.25) |
| Max Daily Loss | $50 (5%) |
| Max Per-Trade Risk | $25 (2.5%) |
| Kill Switch | $300 (30%) |
| EOD Flatten | 4:30 PM NY |
| Commission (RT) | $0.84 |
| Min Confidence | 60% |
| Target Inference | < 10ms |

---

## Test Coverage

**Total**: 2,547 tests passing (1 skipped)

| Category | Tests |
|----------|-------|
| Data Pipeline | 76 |
| Risk Management | 77 |
| Backtesting | 84 |
| ML Models | 55 |
| TopstepX API | 77 |
| Live Trading | 77 |
| DataBento | 43 |
| Optimization | 200+ |
| Integration | 88 |
| Go-Live Validation | 170+ |

---

## Specification Coverage

| Spec | Coverage |
|------|----------|
| ml-scalping-model.md | 100% |
| risk-management.md | 100% |
| backtesting-engine.md | 100% |
| topstepx-api-integration.md | 100% |
| live-trading-execution.md | 100% |
| databento-historical-data.md | 100% |

---

## Data Asset

`data/historical/MES/MES_1s_2years.parquet` - 227MB, 33.2M rows, fully integrated.
