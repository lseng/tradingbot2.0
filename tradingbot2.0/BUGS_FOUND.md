# Bugs Found During Training Deployment

These bugs were discovered during actual training runs on RunPod with full dataset.

## 1. Infinity Values in Feature Scaling (FIXED)

**File:** `src/ml/data/scalping_features.py:638`

**Issue:** Some technical indicators (e.g., division by volume, ratio calculations) produce infinity values when dividing by zero. StandardScaler crashes with `ValueError: Input X contains infinity or a value too large for dtype('float64')`.

**Fix:** Added infinity replacement before scaling:
```python
df_features[feature_names] = df_features[feature_names].replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna(subset=feature_names)
```

**Commit:** `6d0e400`

---

## 2. Infinity Values in Validation/Test Sets (FIXED)

**File:** `src/ml/train_scalping_model.py:409-419`

**Issue:** Same infinity issue affects validation and test sets when applying `scaler.transform()`.

**Fix:** Added infinity handling before transform for val/test sets.

**Commit:** `26a597c`

---

## 3. CUDA Out of Memory During Test Evaluation (FIXED)

**File:** `src/ml/train_scalping_model.py:593-613`

**Issue:** Test evaluation processes entire test set (2.2M samples) in one GPU batch. LSTM models require hidden states for entire batch, causing OOM: `Tried to allocate 134.54 GiB. GPU has 94.97 GiB`.

**Fix:** Process test set in batches (1024 samples) with periodic cache clearing:
```python
eval_batch_size = 1024
for i in range(0, len(X_test_t), eval_batch_size):
    batch = X_test_t[i:i+eval_batch_size].to(device)
    ...
    torch.cuda.empty_cache()
```

**Commit:** `06f3883`

---

## 4. LSTM Output Tuple Unpacking (FIXED)

**File:** `src/ml/train_scalping_model.py:604-606`

**Issue:** LSTM forward() returns `(logits, hidden_state)` tuple, but evaluation code passed tuple directly to softmax: `TypeError: softmax() received an invalid combination of arguments - got (tuple, dim=int)`.

**Fix:** Unpack LSTM output before softmax:
```python
output = model(batch)
logits = output[0] if isinstance(output, tuple) else output
```

**Commit:** `6130240`

---

## 5. Container Memory Limit (WORKAROUND)

**Issue:** RunPod container has 262GB cgroup memory limit even when host has 1.1TB. Feature engineering on full 6.2M samples peaks at ~250GB, causing OOM kill (exit code 137).

**Workaround:** Use `--max-samples 3000000` to limit training data to ~120GB peak memory.

**Proper Fix Needed:**
- Chunked feature engineering
- Memory-mapped data processing
- Or request higher container memory limit

---

## 6. ScalpingFeatureEngineer Constructor Mismatch (FIXED)

**File:** `scripts/run_backtest.py:407`

**Issue:** Backtest script creates `ScalpingFeatureEngineer()` without required `df` argument. The class requires `df` in constructor, not in method call.

**Error:**
```
TypeError: ScalpingFeatureEngineer.__init__() missing 1 required positional argument: 'df'
```

**Fix:** Pass df to constructor:
```python
# Before (wrong):
feature_engineer = ScalpingFeatureEngineer()
df = feature_engineer.add_all_features(df)

# After (correct):
feature_engineer = ScalpingFeatureEngineer(df)
df = feature_engineer.generate_all_features()
```

---

## 7. Model Loading Config Key Mismatch (FIXED)

**File:** `scripts/run_backtest.py:256-266`

**Issue:** `load_model()` function expected old checkpoint format with `config` key, but trained model saves with `model_config` key. Different key names throughout:
- Checkpoint uses `model_config` → code expected `config`
- Checkpoint uses `type` → code expected `model_type`
- Checkpoint uses `input_dim` (top-level) → code expected `input_size` in config
- Checkpoint uses `params.hidden_dims` → code expected `hidden_sizes`

**Error:**
```
TypeError: FeedForwardNet.__init__() got an unexpected keyword argument 'input_size'
```

**Fix:** Handle both old and new checkpoint formats:
```python
config = checkpoint.get('model_config', checkpoint.get('config', {}))
model_type = config.get('type', config.get('model_type', 'feedforward'))
input_size = checkpoint.get('input_dim', config.get('input_size', 50))
params = config.get('params', config)
hidden_dims = params.get('hidden_dims', params.get('hidden_sizes', [256, 128, 64]))
```

---

## 8. FeedForwardNet Parameter Names (FIXED)

**File:** `scripts/run_backtest.py:270-277`

**Issue:** Code passed `input_size` and `hidden_sizes` to `FeedForwardNet`, but the class uses `input_dim` and `hidden_dims`.

**Fix:** Use correct parameter names:
```python
model = FeedForwardNet(
    input_dim=input_size,
    hidden_dims=hidden_dims,
    dropout_rate=dropout_rate,
    num_classes=num_classes,
)
```

---

## 9. PerformanceMetrics Attribute Names (FIXED)

**File:** `scripts/run_backtest.py:495-498`

**Issue:** Code references wrong attribute names for PerformanceMetrics:
- `metrics.win_rate` → should be `metrics.win_rate_pct`
- `metrics.total_net_pnl` → should be `metrics.net_profit`
- `metrics.max_drawdown` → should be `metrics.max_drawdown_dollars`

**Error:**
```
AttributeError: 'PerformanceMetrics' object has no attribute 'win_rate'
```

**Fix:** Use correct attribute names matching PerformanceMetrics class.

---

## Recommended Tests to Add

1. Test feature scaling with infinity values in input
2. Test LSTM evaluation with large batch sizes
3. Test memory usage estimation before training
4. Integration test with full data pipeline
5. **Test checkpoint loading with both old and new formats**
6. **Test backtest script end-to-end with trained model**
7. **Test ScalpingFeatureEngineer API consistency**

---

## Notes for Ralph

When running `./loop.sh plan` next, consider:
- Adding unit tests for these edge cases
- Implementing checkpoint resumption so training can continue after crashes
- Adding memory estimation before training starts
- Implementing chunked feature engineering for large datasets
- **Standardizing checkpoint format keys across training/inference**
- **Ensuring PerformanceMetrics has consistent API**
- **Integration tests for backtest script with real models**
