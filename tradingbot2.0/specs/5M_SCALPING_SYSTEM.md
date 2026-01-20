# 5-Minute Scalping System Specification

## Overview

Build a high-frequency scalping system for MES futures that generates multiple high-quality trades per day on 5-minute candles. The system must use the full 6.5-year historical dataset (2019-2025) with proper temporal train/test splits to ensure generalization.

## Goals

1. **Multiple trades per day** - Target 3-10 high-quality signals daily during RTH (Regular Trading Hours)
2. **High win rate** - Only trade when model confidence exceeds threshold (target >60% accuracy on filtered trades)
3. **5-minute timeframe** - All analysis and execution based on 5-minute candles
4. **Generalization** - Must perform well on completely unseen 2024-2025 data

## Data Requirements

### Source Data
- **File**: `data/historical/MES/MES_full_1min_continuous_UNadjusted.txt`
- **Format**: CSV with columns: datetime, open, high, low, close, volume
- **Range**: May 2019 - December 2025 (6.5 years, 2.3M 1-minute bars)

### Data Processing
1. Load 1-minute data
2. Aggregate to 5-minute OHLCV bars
3. Filter to RTH only (9:30 AM - 4:00 PM Eastern)
4. Handle any gaps or missing data
5. Expected output: ~400K 5-minute bars

### Temporal Splits (CRITICAL - No Look-Ahead Bias)
| Split | Date Range | Purpose | Approximate Bars |
|-------|------------|---------|------------------|
| Train | 2019-05-01 to 2022-12-31 | Model training | ~280K |
| Validate | 2023-01-01 to 2023-12-31 | Hyperparameter tuning, early stopping | ~70K |
| Test | 2024-01-01 to 2025-12-31 | Final evaluation (NEVER touch during development) | ~70K |

## Target Variables

Predict price direction over short horizons relevant to 5-minute scalping:

| Target | Definition | Use Case |
|--------|------------|----------|
| `target_3bar` | Price up/down after 3 bars (15 min) | Quick scalps |
| `target_6bar` | Price up/down after 6 bars (30 min) | Standard trades |
| `target_12bar` | Price up/down after 12 bars (1 hour) | Longer holds |

**Label Definition**:
- 1 = Close price higher than current close (long signal)
- 0 = Close price lower than current close (short signal)
- Minimum move threshold: 2 ticks ($2.50) to filter noise

## Feature Engineering

### Core Features (Keep It Simple)
Focus on features that work for short-term momentum:

1. **Returns** (5 features)
   - return_1bar, return_3bar, return_6bar, return_12bar, return_24bar

2. **Moving Average Relationships** (4 features)
   - close vs EMA-8, EMA-21, EMA-50, EMA-200

3. **Momentum** (5 features)
   - RSI-7, RSI-14
   - MACD, MACD signal, MACD histogram

4. **Volatility** (3 features)
   - ATR-14
   - Bollinger Band width
   - Bar range (high-low)

5. **Volume** (3 features)
   - Volume ratio (current vs 20-bar avg)
   - Volume trend (slope of volume)
   - VWAP deviation

6. **Time** (4 features)
   - time_of_day (normalized 0-1)
   - minutes_since_open
   - is_first_hour (9:30-10:30)
   - is_last_hour (3:00-4:00)

**Total: ~24 features** (intentionally simple to avoid overfitting)

## Model Architecture

### Primary Model: Gradient Boosted Trees (LightGBM or XGBoost)
- Simpler than neural networks
- Better interpretability
- Less prone to overfitting
- Fast inference for live trading

### Model Configuration
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'min_data_in_leaf': 100,
    'early_stopping_rounds': 50,
}
```

### Alternative: Simple Neural Network (if GBM underperforms)
- Input: 24 features
- Hidden: [64, 32]
- Dropout: 0.3
- Output: Sigmoid (probability)

## Confidence Filtering (CRITICAL)

Only generate trade signals when model confidence exceeds threshold:

```python
def should_trade(probability, threshold=0.60):
    """
    probability: model output (0-1)
    Returns: (should_trade, direction)
    """
    if probability >= threshold:
        return True, 'LONG'
    elif probability <= (1 - threshold):
        return True, 'SHORT'
    else:
        return False, None
```

### Confidence Tiers
| Tier | Probability Range | Expected Accuracy | Position Size |
|------|-------------------|-------------------|---------------|
| High | >70% or <30% | ~65%+ | Full size |
| Medium | 60-70% or 30-40% | ~58-62% | Half size |
| Low | 40-60% | ~50-55% | No trade |

## Trading Rules

### Entry Conditions
1. Model confidence >= 60% for direction
2. Current time is RTH (9:30 AM - 4:00 PM ET)
3. Not within 15 minutes of market close (no new positions after 3:45 PM)
4. No existing position in same direction
5. Daily loss limit not exceeded

### Exit Conditions
1. **Profit target**: 4-8 ticks ($5-10) depending on volatility
2. **Stop loss**: 4-6 ticks ($5-7.50)
3. **Time stop**: Exit after 30 minutes if neither target hit
4. **Signal reversal**: Exit if model flips direction with high confidence
5. **EOD**: Flatten all positions by 3:55 PM

### Risk Management
- **Max position**: 1 contract
- **Max daily loss**: $100 (80 ticks)
- **Max consecutive losses**: 3 (then pause 30 min)
- **Commission**: $0.42 per round trip

## Backtesting Requirements

### Realistic Execution Model
```python
class BacktestEngine:
    TICK_SIZE = 0.25
    TICK_VALUE = 1.25  # $1.25 per tick
    COMMISSION = 0.42  # per round trip
    SLIPPAGE = 1  # ticks

    def execute_trade(self, signal, current_bar):
        # Entry at next bar open + slippage
        entry_price = next_bar.open + (SLIPPAGE * TICK_SIZE * signal.direction)
        # ... execution logic
```

### Metrics to Report
1. **Overall**
   - Total trades
   - Win rate (%)
   - Profit factor
   - Total PnL ($)
   - Max drawdown ($)
   - Sharpe ratio (if possible)

2. **Per Day**
   - Average trades per day
   - Average daily PnL
   - Profitable days (%)

3. **By Confidence Tier**
   - Accuracy at each confidence level
   - PnL contribution by tier

4. **By Time of Day**
   - Performance in first hour
   - Performance in last hour
   - Performance mid-day

## Output Artifacts

### Models
- `models/5m_scalper_lgbm.pkl` - Trained LightGBM model
- `models/5m_scalper_config.json` - Model configuration and feature list

### Backtest Results
- `results/5m_backtest_summary.json` - Overall metrics
- `results/5m_backtest_trades.csv` - Individual trade log
- `results/5m_equity_curve.png` - Visualization

### Scripts
- `src/scalping/data_pipeline.py` - Data loading and feature engineering
- `src/scalping/model.py` - Model training and prediction
- `src/scalping/backtest.py` - Backtesting engine
- `src/scalping/run_backtest.py` - Main entry point

## Success Criteria

### Minimum Requirements (Must Have)
1. Test set (2024-2025) win rate >= 55% on filtered trades
2. Test set profit factor >= 1.2
3. Average >= 3 trades per day
4. No single day loses more than $100

### Target Requirements (Should Have)
1. Test set win rate >= 58% on filtered trades
2. Test set profit factor >= 1.5
3. Average >= 5 trades per day
4. Positive PnL in >= 60% of trading days

### Stretch Goals (Nice to Have)
1. Test set win rate >= 62% on high-confidence trades
2. Monthly positive returns for all months in test period
3. Max drawdown < $300

## Anti-Patterns to Avoid

1. **No peeking at test set** - All hyperparameter tuning on validation only
2. **No future leakage** - Features must only use past data
3. **No overfitting** - Keep model simple, use regularization
4. **No unrealistic fills** - Always assume slippage and commission
5. **No curve fitting** - Don't optimize for specific date ranges

## Development Phases

### Phase 1: Data Pipeline
- Load 6.5-year data
- Aggregate to 5-minute bars
- Create train/val/test splits
- Generate features and targets
- Validate data quality

### Phase 2: Model Training
- Train baseline model on train set
- Tune hyperparameters on validation set
- Evaluate feature importance
- Test confidence thresholds

### Phase 3: Backtesting
- Implement realistic backtest engine
- Run on validation set first
- Final evaluation on test set
- Generate comprehensive reports

### Phase 4: Analysis & Iteration
- Analyze results by time of day, volatility regime
- Identify failure modes
- Iterate if needed (but don't overfit!)

## Notes

- MES tick size: $0.25, tick value: $1.25
- MES commission: ~$0.42 round trip (broker dependent)
- RTH: 9:30 AM - 4:00 PM Eastern Time
- The 2024-2025 test period is completely held out - this is the TRUE measure of system performance
