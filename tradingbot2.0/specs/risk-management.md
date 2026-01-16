# Risk Management Specification

## Overview

Risk management system to protect the $1,000 starting capital from catastrophic loss while allowing for aggressive scalping profits. **The account CANNOT be blown up.**

## Capital Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting capital | $1,000 | Fixed |
| MES tick size | 0.25 points | Minimum price movement |
| MES tick value | $1.25 | P&L per tick per contract |
| MES point value | $5.00 | 4 ticks = 1 point |
| Margin per contract | ~$50-100 | Varies by broker |

---

## Risk Limits (NON-NEGOTIABLE)

### Daily Limits
| Limit | Value | Action When Hit |
|-------|-------|-----------------|
| Max daily loss | $50 (5% of capital) | Stop trading for day |
| Max daily drawdown | $75 (7.5% of capital) | Stop trading for day |
| Max consecutive losses | 5 trades | Pause 30 minutes |

### Per-Trade Limits
| Limit | Value | Notes |
|-------|-------|-------|
| Max risk per trade | $25 (2.5% of capital) | Hard stop |
| Max position size | Dynamically calculated | Based on stop distance |
| Stop loss | ALWAYS required | No exceptions |

### Account Protection
| Limit | Value | Action |
|-------|-------|--------|
| Max account drawdown | $200 (20% of capital) | Stop trading, review strategy |
| Kill switch threshold | $300 (30% of capital) | Halt all trading permanently |
| Minimum account balance | $700 | Cannot trade below this |

---

## Position Sizing

### Formula
```python
def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,  # e.g., 0.02 for 2%
    stop_loss_ticks: int,
    tick_value: float = 1.25,  # MES
    max_contracts: int = 10
) -> int:
    """
    Calculate position size based on risk.

    Example:
    - Account: $1,000
    - Risk: 2% = $20
    - Stop: 8 ticks = $10 per contract
    - Position: $20 / $10 = 2 contracts
    """
    dollar_risk = account_balance * risk_per_trade_pct
    risk_per_contract = stop_loss_ticks * tick_value

    contracts = int(dollar_risk / risk_per_contract)
    return min(max(contracts, 1), max_contracts)
```

### Scaling Rules
| Account Balance | Max Contracts | Risk % |
|-----------------|---------------|--------|
| $700 - $1,000 | 1 | 2% |
| $1,000 - $1,500 | 2 | 2% |
| $1,500 - $2,000 | 3 | 2% |
| $2,000 - $3,000 | 4 | 2% |
| $3,000+ | 5+ | 1.5% |

### Confidence-Based Scaling
| Model Confidence | Position Multiplier |
|------------------|---------------------|
| < 60% | 0 (no trade) |
| 60% - 70% | 0.5x base size |
| 70% - 80% | 1.0x base size |
| 80% - 90% | 1.5x base size |
| > 90% | 2.0x base size |

---

## Stop Loss Strategy

### Initial Stop Placement
| Method | Description |
|--------|-------------|
| ATR-based | Stop = Entry ± (ATR × multiplier) |
| Fixed ticks | Stop = Entry ± N ticks |
| Structure-based | Stop beyond recent swing high/low |
| Volatility-adjusted | Wider stops in high volatility |

### Recommended: ATR-Based Stops
```python
def calculate_stop(
    entry_price: float,
    atr: float,
    direction: int,  # 1 for long, -1 for short
    atr_multiplier: float = 1.5
) -> float:
    stop_distance = atr * atr_multiplier
    if direction == 1:  # Long
        return entry_price - stop_distance
    else:  # Short
        return entry_price + stop_distance
```

### Stop Adjustment Rules (To Be Optimized by ML)
| Scenario | Options |
|----------|---------|
| Price moves in favor | Trail stop to breakeven, then trail with profit |
| Price stalls | Hold original stop or tighten |
| Volatility spikes | Widen stop or exit |
| Time decay | Tighten stop as EOD approaches |

---

## Take Profit Strategy

### Risk:Reward Ratios to Test
| R:R | Stop (ticks) | Target (ticks) | Win Rate Needed |
|-----|--------------|----------------|-----------------|
| 1:1 | 8 | 8 | > 50% |
| 1:1.5 | 8 | 12 | > 40% |
| 1:2 | 8 | 16 | > 33% |
| 1:3 | 8 | 24 | > 25% |

### Dynamic R:R (ML-Optimized)
- Model predicts expected move magnitude
- Set TP based on predicted move × confidence
- Wider targets when model is confident
- Tighter targets in choppy conditions

### Partial Profit Taking (Optional)
```
Position: 2 contracts
TP1: Close 1 contract at 1:1 R:R, move stop to breakeven
TP2: Close remaining at 1:2 R:R or trail
```

---

## Position Reversal Rules

### When to Reverse
| Condition | Action |
|-----------|--------|
| Strong opposite signal while in position | Consider reversal |
| Stop hit + immediate reversal signal | May reverse |
| Gradual fade with no signal | Just exit, don't reverse |

### Reversal Constraints
- Must have high-confidence opposite signal (> 75%)
- Cannot reverse more than 2x in same bar range
- Cooldown period after reversal: 30 seconds minimum
- Reversal counts as new trade for daily limits

---

## End of Day (EOD) Management

### Flatten Time: 4:30 PM NY (21:30 UTC)

### EOD Rules
| Time (NY) | Action |
|-----------|--------|
| 4:00 PM | Reduce position sizing by 50% |
| 4:15 PM | No new positions, only close existing |
| 4:25 PM | Begin market order exits |
| 4:30 PM | All positions MUST be flat |

### EOD Exit Strategy
```python
def should_flatten(current_time_ny: datetime) -> bool:
    market_close = current_time_ny.replace(hour=16, minute=30, second=0)
    time_to_close = (market_close - current_time_ny).total_seconds()

    return time_to_close <= 300  # 5 minutes or less
```

---

## Circuit Breakers

### Automatic Trading Halts
| Trigger | Duration | Condition to Resume |
|---------|----------|---------------------|
| 3 consecutive losses | 15 minutes | Automatic resume |
| 5 consecutive losses | 30 minutes | Automatic resume |
| Daily loss limit hit | Rest of day | Next trading day |
| Max drawdown hit | Indefinite | Manual review required |

### Market Condition Halts
| Condition | Action |
|-----------|--------|
| Volatility > 3x normal | Reduce size or pause |
| Spread widening | Pause until normal |
| Low volume | Reduce size |
| Major news events | Consider pausing |

---

## Risk Metrics Tracking

### Real-Time Metrics
| Metric | Calculation |
|--------|-------------|
| Daily P&L | Sum of closed trades |
| Open P&L | Current unrealized |
| Daily drawdown | Peak equity - current equity |
| Win rate (rolling) | Wins / total trades (last 20) |
| Average win/loss | Mean P&L of wins vs losses |

### Session Summary
```json
{
  "date": "2025-01-15",
  "trades": 12,
  "wins": 7,
  "losses": 5,
  "gross_pnl": 45.00,
  "commissions": 4.80,
  "net_pnl": 40.20,
  "max_drawdown": 25.00,
  "sharpe_daily": 1.8,
  "largest_win": 15.00,
  "largest_loss": -12.50
}
```

---

## Acceptance Criteria

### Risk Controls
- [ ] Daily loss limit enforced - trading stops at $50 loss
- [ ] Per-trade risk calculated correctly
- [ ] Position sizing scales with account balance
- [ ] EOD flatten at 4:30 PM NY guaranteed
- [ ] Circuit breakers trigger correctly

### Backtesting Validation
- [ ] No single day loses more than 5% in backtest
- [ ] Maximum drawdown < 20% over entire backtest
- [ ] Risk metrics logged for every trade
- [ ] Position sizing matches specification

### Code Requirements
- [ ] Risk manager as separate module
- [ ] All limits configurable via config file
- [ ] Override capability for emergencies
- [ ] Comprehensive logging of all risk decisions
