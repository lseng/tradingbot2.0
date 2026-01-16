# Futures Price Direction Prediction - Neural Network Model

A complete ML pipeline for predicting next-day price movement (up/down) for futures contracts using neural networks.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training with default settings
python train_futures_model.py --data /Users/leoneng/Downloads/MES_full_1min_continuous_UNadjusted.txt

# 3. Use LSTM instead of feed-forward
python train_futures_model.py --data /path/to/data.txt --model lstm --epochs 100
```

## Project Structure

```
ml/
├── data/
│   ├── data_loader.py        # Load and preprocess OHLCV data
│   └── feature_engineering.py # Technical indicator generation
├── models/
│   ├── neural_networks.py    # FeedForward, LSTM, Hybrid architectures
│   └── training.py           # Training loop, walk-forward validation
├── utils/
│   └── evaluation.py         # Metrics, backtesting, visualization
├── configs/
│   └── default_config.yaml   # Configuration template
├── train_futures_model.py    # Main training script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features Generated

| Category | Features |
|----------|----------|
| Returns | 1, 5, 10, 21-day returns (simple and log) |
| Moving Averages | SMA & EMA at 5, 10, 20, 50 periods, crossovers |
| Volatility | ATR, Bollinger Bands, realized volatility |
| Momentum | RSI, MACD, Stochastic oscillator |
| Volume | Volume ratios, OBV, VPT |
| Candlestick | Body size, wick ratios, gaps |
| Time | Day of week, month (cyclical encoding) |

## Model Architectures

### Feed-Forward Network (Default)
- 3 hidden layers (128-64-32 neurons)
- Batch normalization + ReLU + Dropout
- Best for: Fast training, interpretable

### LSTM Network
- 2-layer LSTM with 64 hidden units
- Captures temporal dependencies
- Best for: Sequence patterns

## Training Features

- **Walk-Forward Validation**: Time-series aware cross-validation
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Gradient Clipping**: Training stability
- **Checkpointing**: Resume training

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1
- AUC-ROC

### Trading
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Comparison vs Buy & Hold

## Command Line Options

```
--data PATH           Path to OHLCV data file
--model TYPE          'feedforward' or 'lstm'
--epochs N            Training epochs (default: 50)
--batch-size N        Batch size (default: 32)
--hidden-dims DIMS    Hidden layer sizes, e.g., '128,64,32'
--dropout RATE        Dropout rate (default: 0.3)
--learning-rate LR    Learning rate (default: 0.001)
--walk-forward-splits Number of validation folds (default: 5)
--seq-length N        LSTM sequence length (default: 20)
--output-dir DIR      Results directory (default: ./results)
--no-plot             Disable plotting
--seed N              Random seed (default: 42)
```

---

## Important Limitations & Risk Warnings

### Why ML Trading is Challenging

1. **Market Efficiency**: Markets incorporate information quickly, making persistent edges rare.

2. **Overfitting**: Models can memorize historical patterns that don't generalize. Walk-forward validation helps but doesn't guarantee out-of-sample success.

3. **Regime Changes**: Markets change. A model trained on bull markets may fail in bear markets.

4. **Transaction Costs**: Commissions, slippage, and market impact erode profits. High-frequency strategies are especially vulnerable.

5. **Survivorship Bias**: You only see strategies that "worked" historically.

6. **Data Snooping**: Testing many strategies on the same data inflates apparent performance.

### Best Practices to Reduce Overfitting

1. **Use walk-forward validation** (implemented here)
2. **Keep models simple** - fewer parameters = less overfitting
3. **Regularization** - dropout, L2 weight decay (implemented)
4. **Large training sets** - more data = better generalization
5. **Out-of-sample testing** - never peek at test data during development
6. **Paper trading** - test on live data before real trading

### Before Trading Real Money

1. **Paper trade for 6-12 months** minimum
2. **Start with tiny position sizes**
3. **Set strict risk limits** (max daily loss, max drawdown)
4. **Understand every trade** the model makes
5. **Have a kill switch** for unexpected behavior
6. **Never risk money you can't afford to lose**

### What This Model CAN'T Do

- Predict black swan events
- Account for breaking news
- Model liquidity/market impact
- Guarantee any specific return
- Replace human judgment

---

## Example Output

```
====================================================================
MODEL & STRATEGY EVALUATION REPORT
====================================================================

CLASSIFICATION METRICS
----------------------------------------
  Accuracy:   0.5234
  Precision:  0.5189
  Recall:     0.5412
  F1 Score:   0.5298
  AUC-ROC:    0.5301

TRADING METRICS
----------------------------------------
  Total Return:      12.45%
  Annualized Return: 8.23%
  Sharpe Ratio:      0.723
  Max Drawdown:      -15.32%
  Win Rate:          51.2%
  Profit Factor:     1.12
  Total Trades:      847

STRATEGY vs BUY & HOLD
----------------------------------------
  Strategy Return:   12.45%
  Buy & Hold Return: 18.73%
  Alpha (excess):    -6.28%
```

Note: In this example, the strategy underperformed buy-and-hold, which is common!

---

## License

Educational use only. Not financial advice.

## Acknowledgments

- Data: Databento (historical futures data)
- Framework: PyTorch
- Inspired by academic research in quantitative finance
