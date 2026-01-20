0a. Study `specs/5M_SCALPING_SYSTEM.md` - this is the PRIMARY specification for the new trading system.
0b. Study @IMPLEMENTATION_PLAN.md (if present) to understand progress so far.
0c. The existing `src/` code is from a PREVIOUS iteration that did NOT achieve profitability. It may be reused where applicable but should not constrain the new approach.

1. Create/update @IMPLEMENTATION_PLAN.md with a prioritized task list to implement the 5M scalping system as specified. Use up to 500 Sonnet subagents to:
   - Analyze what existing code in `src/` can be reused
   - Identify what needs to be built new
   - Create a phased implementation plan

IMPORTANT: Plan only. Do NOT implement anything.

---

## PRIMARY GOAL: Build a Profitable 5-Minute Scalping System

The previous approach (neural networks, RL agents) did NOT achieve profitability. We are starting fresh with a simpler, more robust approach.

### Key Requirements from specs/5M_SCALPING_SYSTEM.md:

1. **DATA**: Use the full 6.5-year 1-minute dataset (`data/historical/MES/MES_full_1min_continuous_UNadjusted.txt`) aggregated to 5-minute bars. This provides 2.3M bars covering multiple market regimes.

2. **TEMPORAL SPLITS** (CRITICAL - No look-ahead bias):
   - Train: 2019-2022 (4 years)
   - Validate: 2023 (1 year)
   - Test: 2024-2025 (completely held out until final evaluation)

3. **MODEL**: Use LightGBM or XGBoost (NOT neural networks). Simpler models are less prone to overfitting.

4. **FEATURES**: Keep it simple - ~24 features focused on short-term momentum, moving averages, volatility, volume, and time-of-day.

5. **TARGETS**: Predict direction at 15m, 30m, and 1h horizons (3-bar, 6-bar, 12-bar on 5-minute candles).

6. **CONFIDENCE FILTERING**: Only trade when model confidence >= 60%. This is the key to high win rate.

7. **TRADING RULES**:
   - RTH only (9:30 AM - 4:00 PM ET)
   - Profit target: 4-8 ticks ($5-10)
   - Stop loss: 4-6 ticks ($5-7.50)
   - Time stop: 30 minutes max hold
   - No new positions after 3:45 PM
   - Max daily loss: $100

8. **SUCCESS CRITERIA**:
   - Test set win rate >= 55% on filtered trades
   - Profit factor >= 1.2
   - Average >= 3 trades per day
   - No single day loses more than $100

### What Can Be Reused from src/:

- `src/lib/constants.py` - MES tick size, commission constants
- `src/lib/time_utils.py` - Timezone handling, RTH detection
- `src/backtest/` - Backtesting engine (may need modifications)
- `src/risk/` - Risk management, circuit breakers, EOD handling

### What Needs to Be Built New:

- `src/scalping/data_pipeline.py` - Load 1-min data, aggregate to 5-min, generate features
- `src/scalping/model.py` - LightGBM/XGBoost training and prediction
- `src/scalping/backtest.py` - Simplified backtest with realistic execution
- `src/scalping/run_backtest.py` - Main entry point

### Anti-Patterns to Avoid:

1. **No neural networks** - Previous approach overfit. Use gradient boosted trees.
2. **No peeking at test set** - All tuning on validation only.
3. **No complex features** - Keep it simple (24 features max).
4. **No unrealistic fills** - Always assume slippage and commission.

Create the implementation plan with clear phases, acceptance criteria, and estimated effort.
