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

## Recommended Tests to Add

1. Test feature scaling with infinity values in input
2. Test LSTM evaluation with large batch sizes
3. Test memory usage estimation before training
4. Integration test with full data pipeline

---

## Notes for Ralph

When running `./loop.sh plan` next, consider:
- Adding unit tests for these edge cases
- Implementing checkpoint resumption so training can continue after crashes
- Adding memory estimation before training starts
- Implementing chunked feature engineering for large datasets
