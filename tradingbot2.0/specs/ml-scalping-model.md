# ML Scalping Model Specification

## Overview

A machine learning model that predicts short-term price action for MES futures scalping. The model must learn profitable entry/exit patterns from historical data and generalize to live trading.

## Objectives

1. **Predict price direction** over scalping timeframes (seconds to minutes)
2. **Identify optimal entry points** with high probability of profit
3. **Determine exit timing** - when to take profit or cut losses
4. **Avoid overfitting** through rigorous walk-forward validation

---

## Data Pipeline

### Input Data
```
data/historical/MES/
├── MES_1s_2years.parquet    # Primary: 3 years, 33M rows, 1-second bars
└── MES_full_1min_continuous.txt  # Secondary: 6.6 years, 1-minute bars
```

### Data Format (1-second parquet)
| Column | Type | Description |
|--------|------|-------------|
| timestamp | int64 (ns) | UTC timestamp |
| open | float | Open price |
| high | float | High price |
| low | float | Low price |
| close | float | Close price |
| volume | int | Volume |
| symbol | str | Contract symbol |

### Preprocessing
- [ ] Convert UTC timestamps to NY timezone for session filtering
- [ ] Filter to RTH only (9:30 AM - 4:00 PM NY) or include ETH based on testing
- [ ] Handle gaps (weekends, holidays)
- [ ] Normalize prices (returns, z-score, or min-max)

---

## Feature Engineering

### Price Action Features
| Feature | Description | Lookback |
|---------|-------------|----------|
| Returns | `(close - close_n) / close_n` | 1, 5, 10, 30, 60 seconds |
| Log returns | `log(close / close_n)` | 1, 5, 10, 30, 60 seconds |
| Price momentum | Cumulative returns | 1, 5, 15 minutes |
| Volatility | Rolling std of returns | 30s, 1m, 5m |
| Range | `(high - low) / close` | Per bar and rolling |

### Microstructure Features
| Feature | Description |
|---------|-------------|
| Bar direction | 1 if close > open, -1 if close < open, 0 if equal |
| Upper wick ratio | `(high - max(open, close)) / (high - low)` |
| Lower wick ratio | `(min(open, close) - low) / (high - low)` |
| Body ratio | `abs(close - open) / (high - low)` |
| Volume delta | Volume relative to rolling average |

### Technical Indicators (calculated on 1-second data)
| Indicator | Parameters |
|-----------|------------|
| EMA | 9, 21, 50, 200 periods |
| RSI | 14 periods |
| MACD | 12, 26, 9 |
| Bollinger Bands | 20 periods, 2 std |
| ATR | 14 periods |
| VWAP | Session-based |

### Time Features
| Feature | Encoding |
|---------|----------|
| Time of day | Cyclical (sin/cos of minutes since open) |
| Day of week | One-hot or cyclical |
| Minutes to close | Linear (for EOD flattening awareness) |

### Multi-Timeframe Features
- Aggregate 1-second data to 1-minute, 5-minute
- Include higher timeframe trend/momentum
- Alignment: use lagged HTF values to avoid lookahead

---

## Model Architecture

### Approach 1: Classification (Direction Prediction)
- **Target**: Price direction over next N seconds/minutes
  - Class 0: Price goes down by X ticks
  - Class 1: Price stays flat (within X ticks)
  - Class 2: Price goes up by X ticks
- **Output**: Softmax probabilities

### Approach 2: Regression (Price/Return Prediction)
- **Target**: Return over next N seconds
- **Output**: Predicted return magnitude

### Approach 3: Reinforcement Learning Style
- **Target**: Optimal action (buy, sell, hold)
- **Reward**: Realized P&L from action

### Recommended: Hybrid Approach
1. **Direction classifier** for entry signals
2. **Volatility regressor** for position sizing
3. **Exit classifier** for optimal exit timing

### Neural Network Options

#### Feed-Forward Network
```
Input (features) → Dense(256) → BN → ReLU → Dropout(0.3)
                → Dense(128) → BN → ReLU → Dropout(0.3)
                → Dense(64)  → BN → ReLU → Dropout(0.3)
                → Dense(num_classes) → Softmax
```

#### LSTM/GRU (Sequence Model)
```
Input (sequence of bars) → LSTM(128, return_sequences=True)
                        → LSTM(64)
                        → Dense(32) → ReLU
                        → Dense(num_classes) → Softmax
```

#### Transformer (Attention-Based)
```
Input (sequence) → Positional Encoding
               → Multi-Head Attention × N layers
               → Dense(num_classes) → Softmax
```

---

## Training Strategy

### Walk-Forward Validation (CRITICAL)
```
|----Train----|--Val--|----Train----|--Val--|----Train----|--Val--|--Test--|
   6 months    1 mo      6 months    1 mo      6 months    1 mo    1 mo

Fold 1: Train Jan-Jun 2023, Val Jul 2023
Fold 2: Train Jan-Jul 2023, Val Aug 2023
Fold 3: Train Jan-Aug 2023, Val Sep 2023
...continue rolling forward
Final Test: Last 1-3 months (never seen during training)
```

### Training Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 256-1024 | Larger for stability |
| Learning rate | 1e-4 to 1e-3 | With scheduler |
| Epochs | 50-100 | With early stopping |
| Early stopping patience | 10 epochs | Monitor val loss |
| Optimizer | AdamW | With weight decay |
| Loss | Cross-entropy (classification) or MSE (regression) |

### Regularization
- Dropout: 0.2-0.4
- Weight decay: 1e-5 to 1e-4
- Batch normalization
- Gradient clipping: 1.0

### Class Imbalance
- Calculate class weights from training data
- Or use focal loss for hard examples
- Oversample minority class if needed

---

## Output Requirements

### Model Outputs Per Prediction
| Output | Type | Description |
|--------|------|-------------|
| direction | int | -1 (short), 0 (flat), 1 (long) |
| confidence | float | 0-1 probability of prediction |
| predicted_move | float | Expected price movement in ticks |
| volatility | float | Expected volatility for position sizing |

### Model Artifacts
```
models/
├── scalper_v1.pt           # PyTorch model weights
├── scalper_v1_config.json  # Architecture config
├── feature_scaler.pkl      # Fitted scaler for features
└── training_metrics.json   # Walk-forward results
```

---

## Acceptance Criteria

### Model Performance
- [ ] Walk-forward validation shows consistent profitability
- [ ] Out-of-sample accuracy > 52% (better than random)
- [ ] Sharpe ratio > 1.0 in backtests
- [ ] No significant performance degradation in recent data

### Code Quality
- [ ] Modular feature engineering pipeline
- [ ] Reproducible training with seed control
- [ ] Model checkpointing and resume capability
- [ ] Inference latency < 10ms (for live trading)

### Overfitting Prevention
- [ ] Training and validation curves show no divergence
- [ ] Performance consistent across walk-forward folds
- [ ] No lookahead bias in feature calculation
- [ ] Model generalizes to unseen market regimes
