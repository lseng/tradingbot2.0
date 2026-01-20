# Implementation Plan - 5-Minute Scalping System

> **Last Updated**: 2026-01-21 UTC (tag v0.0.93)
> **Status**: PHASES 1-3.6 COMPLETE - **ALL DIRECTION STRATEGIES FAILED** - PROJECT CONCLUDED
> **Primary Spec**: `specs/5M_SCALPING_SYSTEM.md`
> **Approach**: LightGBM/XGBoost (NOT neural networks)
> **Data**: 6.5-year 1-minute data aggregated to 5-minute bars
> **Data File**: `data/historical/MES/MES_full_1min_continuous_UNadjusted.txt` (122MB, 2.3M rows)

## Progress Update - 2026-01-21

### Committed Source Files

The following source files were committed to the repository:
- `specs/5M_SCALPING_SYSTEM.md` - Main specification document for the 5-minute scalping system
- `src/rl/` - Reinforcement Learning module (14 files, ~4,871 lines) - EXPERIMENTAL
- `scripts/simple_backtest.py` - Simple backtest runner script
- `scripts/train_recent_binary.py` - Binary classifier training on recent data
- `src/ml/train_binary_scalper.py` - Binary scalper model training

### Test Suite Fixes

**Fixed the `test_load_new_format_checkpoint` test failure:**
- **Bug**: Test expected 2 return values from `load_model()` but function now returns 5 values
- **Root Cause**: The `load_model()` function in `src/ml/models/training.py` was updated to return `(model, config, scaler_mean, scaler_scale, is_binary)` but 12 test locations were still expecting the old 2-value signature `(model, config)`
- **Resolution**: Updated all 12 test locations in `tests/ml/models/test_training.py` to handle the new 5-value return signature

**Fixed flaky test: `test_schema_detection_default_1m` in `tests/test_databento_extended.py`:**
- **Root Cause**: Random temp filenames could accidentally contain schema patterns
- **Fix**: Added explicit `prefix="test_default_"` parameter to `tempfile.NamedTemporaryFile()`

**Current Test Status:**
- All **3,140 tests pass** (2 skipped) in the main test suite
- **Note**: The `src/rl/` module has NO tests and is experimental

### Recent Commits (v0.0.92 - v0.0.93)

**v0.0.93** - Add binary model support, feature normalization, and archive specs:
- `scripts/run_backtest.py`: Added scaler_mean/scaler_scale support for feature normalization, is_binary flag for binary UP/DOWN models
- `src/ml/models/training.py`: Fixed duplicate num_classes argument in train_with_walk_forward()
- `BUGS_FOUND.md`: Documented bugs #10-#12
- Moved legacy specs to `specs/archive/`

### Project Status Summary

**All direction prediction strategies have failed.** The project is at a crossroads:

| Strategy | Win Rate | Profit Factor | Result |
|----------|----------|---------------|--------|
| Direction (24 features) | 38.8% | 0.28 | FAILED |
| Breakout Detection | 39.1% | 0.50 | FAILED |
| Mean-Reversion | 19.1% | 0.11 | FAILED |
| RL Pure Agent | 33.2% | 0.71 | FAILED (experimental) |
| RL Hybrid Agent | 43.2% | 0.82 | FAILED (experimental) |

### Remaining P1 Bugs (from BUGS_FOUND.md)

1. **Bug #11: LSTM Sequence Creation OOM on full dataset** - NOT FIXED
2. **Bug #13: Walk-Forward CV Memory Usage** - Needs investigation

---

## Phase 3.7: Reinforcement Learning Approach (EXPERIMENTAL)

**Status**: EXPERIMENTAL - needs tests and validation before use

### Overview

An experimental RL module was added at `src/rl/` as an alternative approach to direction prediction. This module uses Stable Baselines3 PPO agents trained on a custom trading environment.

### Module Contents

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Module init | 1 |
| `data_pipeline.py` | RL data loading and preprocessing | 342 |
| `enhanced_features.py` | Extended feature engineering for RL | 288 |
| `trading_env.py` | Gym environment for trading | 441 |
| `hybrid_env.py` | Hybrid RL+ML environment | 309 |
| `multi_horizon_model.py` | Multi-horizon prediction model | 454 |
| `regularized_model.py` | Regularized RL model | 406 |
| `train_rl_trader.py` | Pure RL agent training | 356 |
| `train_hybrid_agent.py` | Hybrid RL+ML agent training | 387 |
| `train_multi_horizon.py` | Multi-horizon agent training | 333 |
| `train_enhanced.py` | Enhanced agent training | 343 |
| `train_final_agent.py` | Final agent training script | 439 |
| `evaluate_agents.py` | Agent evaluation and comparison | 343 |
| `walk_forward.py` | Walk-forward validation for RL | 429 |
| **Total** | 14 files | ~4,871 |

### Trained Models

The following RL models were trained and saved to `models/`:
- `rl_trader.zip` (2.5 MB) - Pure RL agent
- `rl_trader_2024.zip` (2.5 MB) - Pure RL agent on 2024 data
- `hybrid_rl_agent.zip` (10.7 MB) - Hybrid RL+ML agent
- `hybrid_rl_10m.zip` (10.7 MB) - Hybrid agent (10-minute variant)
- `hybrid_rl_full.zip` (10.7 MB) - Full hybrid agent
- `final_enhanced_agent.zip` (53.0 MB) - Enhanced agent with all features

### Evaluation Results (from `models/agent_comparison.json`)

**Evaluation Period**: 2025-10-01 to 2025-12-31 (23,278 samples)

| Agent | Mean Reward | Mean P&L | Win Rate | Profit Factor | Profitable Episodes |
|-------|-------------|----------|----------|---------------|---------------------|
| Pure RL | -159.95 | -$118.62 | 33.2% | 0.71 | 15/50 (30%) |
| Hybrid RL+ML | -38.58 | -$13.02 | 43.2% | 0.82 | 19/50 (38%) |

**Key Findings:**
1. The Hybrid RL+ML approach performed better than pure RL (43.2% vs 33.2% win rate)
2. Neither approach achieved profitability (both have PF < 1.0)
3. High variance in results (std_reward > mean_reward for both)
4. This confirms the earlier finding: direction prediction on 5M MES is NOT viable

### Limitations

1. **No Tests**: The RL module has no test coverage - all other modules have comprehensive tests
2. **Experimental Code**: Not validated for production use
3. **Results Confirm Failure**: RL approach also failed to achieve profitability
4. **Resource Intensive**: Training RL agents requires significant compute time

### Recommendations

- Do NOT use in production without comprehensive testing
- The RL results reinforce the conclusion that direction prediction is not viable
- If pursuing RL further, add tests before any production use

---

## Executive Summary

The previous neural network approach (v0.0.83) did NOT achieve profitability. This plan pivots to a simpler, more robust approach using gradient boosted trees on 5-minute candles with strict confidence filtering.

| Phase | Description | Effort | Status |
|-------|-------------|--------|--------|
| **Phase 1** | Data Pipeline | 6-8 hrs | COMPLETE |
| **Phase 2** | Model Training | 4-6 hrs | COMPLETE (2.1, 2.2) |
| **Phase 3.1-3.4** | Backtesting & Volatility | 4-6 hrs | COMPLETE - **DIRECTION FAILED** |
| **Phase 3.5** | Breakout Detection Strategy | 4 hrs | COMPLETE - **FAILED** (WR=39%, PF=0.50) |
| **Phase 3.6** | Mean-Reversion Strategy | 4 hrs | COMPLETE - **FAILED** (WR=19%, PF=0.11) |
| **Phase 3.7** | RL Approach | N/A | EXPERIMENTAL - **FAILED** (WR=43%, PF=0.82) |
| **Phase 4** | Analysis & Iteration | 4-6 hrs | BLOCKED (no profitable strategy found) |
| **Phase 5** | Live Integration | 8-12 hrs | BLOCKED (no profitable strategy found) |

**Total Estimated Effort**: 26-38 hours

### Recommendations for Next Steps

Given the validation failure, the following options should be considered:

1. **Alternative Features**: Order flow imbalance, bid-ask spread dynamics, cross-market correlations
2. **Alternative Targets**: Volatility prediction, range-bound detection, breakout detection
3. **Alternative Timeframes**: Tick data with microstructure features, daily bars with fundamental factors
4. **Alternative Approach**: Market-making strategies, arbitrage strategies, systematic macro

**Do NOT proceed to Phase 5 (Live Integration)** with the current system - it would result in significant losses.

---

## Critical Findings Summary

### CRITICAL FINDING: Validation Backtest FAILED

**The 24-feature LightGBM approach does NOT produce a profitable trading signal.** The validation backtest on 2023 data revealed a fundamental issue: the features have no predictive power for 5-minute price direction.

**Validation Results (2023 data, min_confidence=60%):**
- Total Trades: **0** (model never produced confidence >= 60%)
- Model stopped at iteration 4 due to early stopping
- Validation AUC: **0.51** (barely above random)
- Prediction probability range: 0.42 - 0.51 (extremely narrow, centered on 0.5)

**With confidence threshold lowered to 50%:**
- Total Trades: 4,998 (19.6/day)
- **Win Rate: 38.8%** (worse than coin flip)
- **Profit Factor: 0.28** (losing $3.5 for every $1 won)
- **Net P&L: -$26,420** on validation set

**Feature Correlation Analysis:**
| Feature | Correlation with Target |
|---------|------------------------|
| close_vs_ema200 | +0.019 |
| close_vs_ema50 | +0.017 |
| macd_signal | +0.015 |
| All others | < 0.015 |

**Root Cause:** The MES futures market on 5-minute bars is highly efficient. Price movements approximate a random walk, making directional prediction with technical features essentially impossible.

---

## Summary of All Strategies Attempted

| Strategy | Phase | Result | Win Rate | PF | Trades | P&L |
|----------|-------|--------|----------|-----|--------|-----|
| Direction (24 features) | 3.3 | FAILED | 38.8% | 0.28 | 4,998 | -$26,420 |
| Breakout Detection | 3.5 | FAILED | 39.1% | 0.50 | 555 | -$3,900 |
| Mean-Reversion | 3.6 | FAILED | 19.1% | 0.11 | 356 | -$5,865 |
| Pure RL Agent | 3.7 | FAILED | 33.2% | 0.71 | 433 | -$5,931 |
| Hybrid RL+ML Agent | 3.7 | FAILED | 43.2% | 0.82 | 2,059 | -$651 |

### Key Findings

1. **Volatility IS predictable** (AUC 0.855) - but this cannot be monetized via direction trading
2. **Direction is NOT predictable** regardless of approach (ML, RL, Hybrid)
3. **Market efficiency**: The MES futures market on 5-minute bars is too efficient for technical analysis-based direction prediction
4. **RSI is NOT a reversal signal**: RSI extremes indicate momentum continuation, not reversal

### Recommendations

**Stop trying to predict direction.** All evidence indicates this is not viable with:
- Current data (MES futures, 5-minute bars)
- Current features (technical indicators)
- Current approaches (classification, regression, RL)

**Current Status**: This project has validated that the 5-minute MES scalping approach using technical features is NOT viable. Further work on direction prediction is not recommended.

---

## New Modules Created

**Scalping Module (`src/scalping/`):**
- `data_pipeline.py` - Data loading, aggregation, RTH filtering, temporal splits
- `features.py` - 24-feature generation engine for 5-minute scalping
- `model.py` - LightGBM classifier training and inference
- `walk_forward.py` - Walk-forward cross-validation with calibration metrics
- `backtest.py` - Simplified backtest engine with slippage, commission, and time stops
- `breakout.py` - Breakout detection strategy with consolidation features
- `mean_reversion.py` - Mean-reversion strategy with RSI and volatility filtering

**Scripts:**
- `scripts/run_validation_backtest.py` - Validation backtest runner
- `scripts/run_volatility_prediction.py` - Volatility prediction analysis
- `scripts/run_breakout_detection.py` - Breakout detection strategy runner
- `scripts/run_mean_reversion.py` - Mean-reversion strategy runner
- `scripts/simple_backtest.py` - Simple backtest runner
- `scripts/train_recent_binary.py` - Binary classifier training

**Tests:**
- `tests/scalping/` - Comprehensive test suite with **220 passing tests**

---

## Success Criteria (From Spec)

### Minimum Requirements (Must Have)
- [ ] Test set (2024-2025) win rate >= 55% on filtered trades
- [ ] Test set profit factor >= 1.2
- [ ] Average >= 3 trades per day
- [ ] No single day loses more than $100

### Target Requirements (Should Have)
- [ ] Test set win rate >= 58% on filtered trades
- [ ] Test set profit factor >= 1.5
- [ ] Average >= 5 trades per day
- [ ] Positive PnL in >= 60% of trading days

---

## Phase Details

### Phase 1: Data Pipeline (COMPLETE)

- [x] Load `data/historical/MES/MES_full_1min_continuous_UNadjusted.txt` (122MB, 2.3M rows)
- [x] Aggregate 1-minute bars to 5-minute OHLCV
- [x] Filter to RTH only (9:30 AM - 4:00 PM ET)
- [x] Generate all 24 features without lookahead bias
- [x] Create target variables (binary direction)
- [x] Create temporal train/val/test splits

### Phase 2: Model Training (COMPLETE)

- [x] LightGBM model setup and training
- [x] Walk-forward validation (42 tests, no data leakage)
- [ ] Hyperparameter tuning (blocked - no signal to optimize)

### Phase 3: Backtesting (COMPLETE - ALL FAILED)

- [x] Phase 3.1-3.4: Direction prediction and volatility analysis
- [x] Phase 3.5: Breakout detection strategy - FAILED
- [x] Phase 3.6: Mean-reversion strategy - FAILED
- [x] Phase 3.7: RL approach - EXPERIMENTAL, FAILED

### Phase 4-5: BLOCKED

No profitable strategy found. Live integration not recommended.

---

## Code Reusability

### Highly Reusable (Use As-Is)

| Module | Location | Use Case |
|--------|----------|----------|
| **Constants** | `src/lib/constants.py` | MES tick/dollar conversions, commission |
| **Time Utils** | `src/lib/time_utils.py` | RTH detection, EOD phase management |
| **Logging** | `src/lib/logging_utils.py` | Trade entry/exit logging |
| **Performance Monitor** | `src/lib/performance_monitor.py` | Latency tracking |
| **Backtest Metrics** | `src/backtest/metrics.py` | Sharpe, Sortino, profit factor, etc. |
| **Trade Logger** | `src/backtest/trade_logger.py` | Trade recording with MFE/MAE |
| **Slippage Model** | `src/backtest/slippage.py` | 1-tick normal slippage |
| **Cost Model** | `src/backtest/costs.py` | Commission tracking |

### Not Reusable

| Component | Reason |
|-----------|--------|
| Neural Networks (`src/ml/models/`) | Spec says AVOID |
| RL Module (`src/rl/`) | Experimental, no tests, failed to achieve profitability |

---

## Quick Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Timeframe | 5-minute bars | Aggregated from 1-min data |
| RTH Hours | 9:30 AM - 4:00 PM ET | DST-aware via NY timezone |
| Train Period | 2019-05 to 2022-12 | ~70K bars |
| Validation Period | 2023 | ~18K bars |
| Test Period | 2024-2025 | ~36K bars (HELD OUT) |
| Min Confidence | 60% | For trade entry |
| Profit Target | 6 ticks ($7.50) | |
| Stop Loss | 8 ticks ($10.00) | |
| Time Stop | 30 minutes (6 bars) | |
| Max Daily Loss | $100 | Circuit breaker |
| Commission | $0.84 round-trip | |
| Slippage | 1 tick ($1.25) | Normal conditions |

---

## Archived Content

<details>
<summary>Click to expand previous plan (Neural Network approach - NOT recommended)</summary>

### Previous Executive Summary (v0.0.83)

| Priority | Count | Status |
|----------|-------|--------|
| **P0** | 1 | COMPLETE |
| **Code Quality** | 3 | COMPLETE |
| **P1** | 17 | COMPLETE |
| **P2** | 4 | REMAINING |
| **P3** | 4 | LOW |

### Why This Approach Was Abandoned

1. Neural networks overfit despite comprehensive infrastructure
2. Complexity vs. robustness trade-off
3. 1-second data granularity too noisy
4. Spec recommendation: gradient boosted trees over neural networks

</details>
