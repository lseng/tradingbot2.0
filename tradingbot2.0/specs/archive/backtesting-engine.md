# Backtesting Engine Specification

## Overview

A realistic backtesting engine that simulates the scalping strategy on historical data with proper handling of transaction costs, slippage, and market microstructure.

## Objectives

1. **Realistic simulation** - Account for all trading frictions
2. **Walk-forward testing** - Prevent overfitting
3. **Comprehensive metrics** - Evaluate strategy robustness
4. **Fast execution** - Run thousands of parameter combinations

---

## Data Requirements

### Primary Data Source
```
data/historical/MES/MES_1s_2years.parquet
- 33 million 1-second bars
- Jan 2023 → Dec 2025 (3 years)
- Columns: timestamp, open, high, low, close, volume
```

### Session Filtering
- **RTH (Regular Trading Hours)**: 9:30 AM - 4:00 PM NY
- **ETH (Extended Hours)**: 6:00 PM - 9:30 AM NY (optional)
- Convert UTC timestamps to NY timezone
- Handle DST transitions

---

## Transaction Costs

### MES Futures Costs
| Cost | Value | Notes |
|------|-------|-------|
| Commission | $0.20 per side | TopstepX typical |
| Exchange fee | $0.22 per side | CME fee |
| **Total round-trip** | **$0.84** | Per contract |

### Slippage Model
| Market Condition | Slippage (ticks) |
|------------------|------------------|
| Normal liquidity | 0.25 (1 tick) |
| Low liquidity | 0.50 (2 ticks) |
| High volatility | 0.50 - 1.00 |
| Market orders | 1 tick assumed |
| Limit orders | 0 ticks (if filled) |

### Slippage Implementation
```python
def apply_slippage(
    price: float,
    direction: int,  # 1=buy, -1=sell
    order_type: str,
    volatility: float,
    tick_size: float = 0.25
) -> float:
    if order_type == 'limit':
        return price  # No slippage on limits

    # Market order slippage
    base_slippage = tick_size  # 1 tick
    vol_adjustment = tick_size if volatility > threshold else 0

    slippage = (base_slippage + vol_adjustment) * direction
    return price + slippage
```

---

## Backtest Modes

### Mode 1: Simple Backtest
- Single pass through data
- Fixed parameters
- Quick validation

### Mode 2: Walk-Forward Optimization
```
For each fold:
  1. Train model on training window
  2. Optimize parameters on validation window
  3. Test on out-of-sample window
  4. Roll forward and repeat
```

### Mode 3: Monte Carlo Simulation
- Randomize trade order within constraints
- Bootstrap confidence intervals
- Assess strategy robustness

---

## Walk-Forward Configuration

### Time Windows
```python
walk_forward_config = {
    "training_months": 6,
    "validation_months": 1,
    "test_months": 1,
    "step_months": 1,  # Roll forward by this much
    "min_trades_per_fold": 100,
}
```

### Example Split (3 years of data)
```
Fold 1: Train Jan-Jun 2023 | Val Jul 2023 | Test Aug 2023
Fold 2: Train Feb-Jul 2023 | Val Aug 2023 | Test Sep 2023
Fold 3: Train Mar-Aug 2023 | Val Sep 2023 | Test Oct 2023
...
Fold N: Train Jun-Nov 2025 | Val Dec 2025 | Test (live)
```

---

## Simulation Engine

### Event-Driven Architecture
```python
class BacktestEngine:
    def __init__(self, data, strategy, risk_manager, config):
        self.data = data
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config

    def run(self):
        for bar in self.data:
            # 1. Update indicators/features
            self.strategy.on_bar(bar)

            # 2. Check for stop/target hits on current bar
            self.check_exits(bar)

            # 3. Generate signals
            signal = self.strategy.generate_signal()

            # 4. Apply risk management
            if self.risk_manager.can_trade():
                position = self.risk_manager.size_position(signal)
                self.execute_trade(position)

            # 5. EOD flatten check
            if self.is_eod(bar):
                self.flatten_all()

        return self.calculate_results()
```

### Order Fill Simulation
```python
def simulate_fill(
    order: Order,
    bar: Bar,
    fill_model: str = "conservative"
) -> Optional[Fill]:
    """
    Conservative: Only fill if price touches order price
    Optimistic: Fill at order price if within bar range
    Realistic: Fill with slippage based on bar direction
    """
    if order.type == OrderType.MARKET:
        # Fill at open of next bar + slippage
        fill_price = apply_slippage(bar.open, order.side)
        return Fill(price=fill_price, time=bar.timestamp)

    elif order.type == OrderType.LIMIT:
        if order.side == Side.BUY and bar.low <= order.price:
            return Fill(price=order.price, time=bar.timestamp)
        elif order.side == Side.SELL and bar.high >= order.price:
            return Fill(price=order.price, time=bar.timestamp)

    return None  # Not filled
```

---

## Performance Metrics

### Return Metrics
| Metric | Formula |
|--------|---------|
| Total Return | `(final_equity - initial_equity) / initial_equity` |
| CAGR | `(final/initial)^(1/years) - 1` |
| Daily Return | Mean of daily P&L |
| Monthly Return | Mean of monthly P&L |

### Risk Metrics
| Metric | Formula |
|--------|---------|
| Sharpe Ratio | `mean(returns) / std(returns) * sqrt(252)` |
| Sortino Ratio | `mean(returns) / downside_std * sqrt(252)` |
| Calmar Ratio | `CAGR / max_drawdown` |
| Max Drawdown | `max(peak - trough) / peak` |
| Max Drawdown Duration | Days from peak to recovery |

### Trade Metrics
| Metric | Description |
|--------|-------------|
| Total Trades | Number of round-trips |
| Win Rate | `wins / total_trades` |
| Profit Factor | `gross_profit / gross_loss` |
| Average Win | Mean P&L of winning trades |
| Average Loss | Mean P&L of losing trades |
| Largest Win | Maximum single trade P&L |
| Largest Loss | Minimum single trade P&L |
| Average Trade | Mean P&L across all trades |
| Expectancy | `(win_rate * avg_win) - (loss_rate * avg_loss)` |

### Consistency Metrics
| Metric | Description |
|--------|-------------|
| Win Days % | Percentage of profitable days |
| Best Day | Maximum daily P&L |
| Worst Day | Minimum daily P&L |
| Consecutive Wins | Max winning streak |
| Consecutive Losses | Max losing streak |

---

## Output Format

### Trade Log
```csv
trade_id,entry_time,exit_time,direction,entry_price,exit_price,contracts,gross_pnl,commission,slippage,net_pnl,exit_reason
1,2023-01-03 09:35:00,2023-01-03 09:42:00,LONG,3850.25,3852.50,1,11.25,0.84,1.25,9.16,TARGET
2,2023-01-03 10:15:00,2023-01-03 10:18:00,SHORT,3855.00,3856.25,1,-6.25,0.84,1.25,-8.34,STOP
```

### Equity Curve
```csv
timestamp,equity,drawdown,drawdown_pct
2023-01-03 09:30:00,1000.00,0.00,0.00
2023-01-03 09:42:00,1009.16,0.00,0.00
2023-01-03 10:18:00,1000.82,8.34,0.83
```

### Summary Report
```json
{
  "period": {
    "start": "2023-01-03",
    "end": "2025-12-26",
    "trading_days": 756
  },
  "returns": {
    "total_return_pct": 145.2,
    "cagr_pct": 35.8,
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.42,
    "calmar_ratio": 2.15
  },
  "risk": {
    "max_drawdown_pct": 16.6,
    "max_drawdown_duration_days": 23,
    "daily_var_95": 2.1,
    "worst_day_pct": -4.8
  },
  "trades": {
    "total_trades": 4521,
    "win_rate_pct": 54.2,
    "profit_factor": 1.45,
    "avg_trade_pnl": 0.32,
    "expectancy": 0.28
  },
  "costs": {
    "total_commission": 3797.64,
    "total_slippage": 2825.50,
    "cost_per_trade": 1.46
  }
}
```

---

## Optimization Framework

### Parameters to Optimize
| Parameter | Range | Step |
|-----------|-------|------|
| Stop loss (ticks) | 4-16 | 2 |
| Take profit (ticks) | 4-32 | 2 |
| ATR multiplier | 1.0-3.0 | 0.25 |
| Model confidence threshold | 0.5-0.8 | 0.05 |
| Position size risk % | 0.01-0.03 | 0.005 |

### Optimization Methods
1. **Grid Search** - Exhaustive parameter combinations
2. **Random Search** - Sample parameter space
3. **Bayesian Optimization** - Smart parameter exploration
4. **Genetic Algorithm** - Evolve optimal parameters

### Overfitting Prevention
- [ ] Optimize on validation set only
- [ ] Test on held-out test set
- [ ] Compare in-sample vs out-of-sample Sharpe
- [ ] Penalize parameter complexity
- [ ] Require consistent results across folds

---

## Acceptance Criteria

### Backtesting Accuracy
- [ ] Transaction costs correctly applied
- [ ] Slippage model realistic
- [ ] EOD flatten enforced
- [ ] Risk limits respected in simulation
- [ ] No lookahead bias

### Performance Requirements
- [ ] Process 1M bars in < 60 seconds
- [ ] Walk-forward fold in < 5 minutes
- [ ] Full optimization in < 1 hour

### Output Quality
- [ ] Complete trade log with all fields
- [ ] Equity curve at bar-level resolution
- [ ] Summary metrics match manual calculation
- [ ] Walk-forward results per fold

### Validation
- [ ] Known strategy produces expected results
- [ ] Random strategy produces ~0 expectancy
- [ ] Transaction costs reduce returns appropriately
- [ ] Results reproducible with same seed
