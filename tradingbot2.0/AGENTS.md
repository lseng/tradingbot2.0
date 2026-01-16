## Build & Run

Python ML project for futures price direction prediction.

```bash
# Install dependencies
pip install -r src/ml/requirements.txt

# Run training with default settings
python src/ml/train_futures_model.py --data data/historical/MES/MES_full_1min_continuous_UNadjusted.txt

# Run with LSTM model
python src/ml/train_futures_model.py --data data/historical/MES/MES_full_1min_continuous_UNadjusted.txt --model lstm --epochs 100

# Run backtest with default settings
python scripts/run_backtest.py

# Run backtest with specific model
python scripts/run_backtest.py --model models/scalper_v1.pt --verbose

# Run walk-forward validation
python scripts/run_backtest.py --walk-forward --output ./results/walkforward

# Run random baseline (should produce ~0 expectancy)
python scripts/run_backtest.py --random-baseline

# Run live trading (paper mode by default)
# Requires: export TOPSTEPX_API_KEY='your-api-key'
python scripts/run_live.py

# Run live trading with custom parameters
python scripts/run_live.py --capital 2000 --max-daily-loss 100 --min-confidence 0.70
```

## Validation

Run these after implementing to get immediate feedback:

- Tests: `pytest tests/` (858 tests)
- Typecheck: `mypy src/ml/`
- Lint: `ruff check src/ml/`

## Operational Notes

- Models available: `feedforward` (default), `lstm`
- Walk-forward validation for time-series aware cross-validation
- Results output to `./results/` directory by default

### Data Sources

```
data/historical/MES/
├── MES_1s_2years.parquet              # 3 years of 1-second data (227 MB, 33M rows)
│                                       # Range: Jan 2023 → Dec 2025
│                                       # Columns: timestamp, open, high, low, close, volume, symbol
│
└── MES_full_1min_continuous_UNadjusted.txt  # 6.6 years of 1-minute data (122 MB, 2.3M rows)
                                             # Range: May 2019 → Dec 2025
                                             # Format: timestamp,open,high,low,close,volume (CSV, no header)
```

### Project Structure

```
data/
└── historical/MES/        # Historical price data (DataBento)
src/
├── ml/                    # ML pipeline (training, models, features)
├── risk/                  # Risk management (position sizing, circuit breakers, EOD)
├── backtest/              # Backtesting engine (costs, slippage, metrics)
├── api/                   # TopstepX API integration (REST, WebSocket)
└── trading/               # Live trading (signal generation, order execution)
scripts/
├── run_backtest.py        # Backtest entry point
└── run_live.py            # Live trading entry point
tests/                     # 858 tests (unit + integration)
```

### Codebase Patterns

- PyTorch for neural network framework
- Pandas for data manipulation
- Walk-forward validation (time-series cross-validation)
- Early stopping and gradient clipping for training stability
