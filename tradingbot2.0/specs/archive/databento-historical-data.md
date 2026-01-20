# DataBento Historical Data Specification

## Overview

DataBento provides professional-grade historical market data for backtesting and ML model training. This is our primary source for historical futures data since TopstepX only retains ~7-14 days of second-level data.

## API Reference

- **Docs**: https://databento.com/docs
- **Python SDK**: `databento` package
- **API Key**: Configured in `.env` as `DATABENTO_API_KEY`

---

## Data Requirements

### Contracts
| Symbol | DataBento Symbol | Description |
|--------|-----------------|-------------|
| MES | `MES.FUT` | Micro E-mini S&P 500 |
| ES | `ES.FUT` | E-mini S&P 500 |
| MNQ | `MNQ.FUT` | Micro E-mini Nasdaq |
| NQ | `NQ.FUT` | E-mini Nasdaq |

### Historical Depth
- **Minimum**: 1 year
- **Preferred**: 3-5 years
- **Purpose**: Walk-forward validation, regime testing

### Granularity Needed
| Timeframe | Use Case |
|-----------|----------|
| 1-second | High-frequency patterns (optional) |
| 1-minute | Primary training data |
| 5-minute | Multi-timeframe analysis |
| 15-minute | Swing patterns |
| 1-hour | Daily structure |
| Daily | Long-term trends |

---

## Data Schemas

DataBento provides multiple schemas:

| Schema | Description | Use Case |
|--------|-------------|----------|
| `trades` | Individual trades | Tick analysis |
| `ohlcv-1s` | 1-second OHLCV | High-frequency |
| `ohlcv-1m` | 1-minute OHLCV | Primary training |
| `ohlcv-1h` | 1-hour OHLCV | Multi-timeframe |
| `ohlcv-1d` | Daily OHLCV | Long-term |

---

## Implementation

### Python SDK Usage

```python
import databento as db

client = db.Historical(key="YOUR_API_KEY")

# Fetch 1-minute bars for MES
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",  # CME Globex
    symbols=["MES.FUT"],
    schema="ohlcv-1m",
    start="2022-01-01",
    end="2025-01-01",
)

# Convert to DataFrame
df = data.to_df()
```

### Continuous Contracts

DataBento handles roll logic automatically with `.FUT` suffix:
- `MES.FUT` → Continuous front-month MES
- No need to manually stitch quarterly contracts

---

## Data Pipeline Design

### 1. Initial Download
```
src/data/
├── downloaders/
│   └── databento_downloader.py
├── processors/
│   └── ohlcv_processor.py
└── storage/
    └── parquet_store.py
```

### 2. Storage Format
- **Format**: Parquet (columnar, compressed)
- **Partitioning**: By year/month for efficient queries
- **Location**: `data/historical/`

### 3. Update Strategy
- Initial bulk download: 3-5 years of data
- Daily incremental updates
- Gap detection and backfill

---

## Cost Considerations

DataBento pricing is based on data volume:
- Check current pricing at https://databento.com/pricing
- Start with 1-minute data (smaller than tick)
- Download once, store locally

---

## Acceptance Criteria

### Data Download
- [ ] Successfully authenticate with DataBento
- [ ] Download 3 years of 1-minute MES data
- [ ] Download 3 years of daily MES data
- [ ] Store in Parquet format

### Data Quality
- [ ] No gaps in trading hours
- [ ] Timestamps in UTC
- [ ] Volume data present
- [ ] OHLC relationship valid (L <= O,C <= H)

### Integration
- [ ] Load historical data for ML training
- [ ] Support multiple timeframe aggregation
- [ ] Efficient querying by date range
