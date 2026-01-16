## Build & Run

Python ML project for futures price direction prediction.

```bash
# Install dependencies
pip install -r src/ml/requirements.txt

# Run training with default settings
python src/ml/train_futures_model.py --data /path/to/data.txt

# Run with LSTM model
python src/ml/train_futures_model.py --data /path/to/data.txt --model lstm --epochs 100
```

## Validation

Run these after implementing to get immediate feedback:

- Tests: `pytest src/ml/`
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
└── ml/
    ├── data/              # Data loading and feature engineering
    ├── models/            # Neural network architectures and training
    ├── utils/             # Evaluation metrics and visualization
    ├── configs/           # Configuration files
    └── train_futures_model.py  # Main entry point
```

### Codebase Patterns

- PyTorch for neural network framework
- Pandas for data manipulation
- Walk-forward validation (time-series cross-validation)
- Early stopping and gradient clipping for training stability
