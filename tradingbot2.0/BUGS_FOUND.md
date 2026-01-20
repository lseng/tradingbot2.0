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

## 10. LSTM Sequence Creation Too Slow - BLOCKING (FIXED 2026-01-18)

**Priority:** P0 - COMPLETE (was BLOCKING)

**File:** `src/ml/models/training.py:43-96`

**Status:** FIXED using `numpy.lib.stride_tricks.sliding_window_view` - sequence creation now completes in seconds instead of 60+ minutes. Performance tests added in `tests/test_training.py:613-766`.

**Original Issue:** `SequenceDataset.__init__()` used a slow Python for-loop to create LSTM sequences. For the full dataset (6.2M training samples), this took 60+ minutes and never completed. GPU training never started.

**Evidence from RunPod testing (2026-01-17):**

Hardware: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM), 1.5TB RAM

| Time Elapsed | Stage | RAM Usage | GPU Usage |
|--------------|-------|-----------|-----------|
| 0.5 sec | Data loaded (15.8M rows) | 17 GB | 0% |
| ~5 min | Feature engineering done | 60 GB | 0% |
| ~15 min | Sequence creation started | 100 GB | 0% |
| ~30 min | Still creating sequences | 131 GB | 0% |
| ~45 min | Still creating sequences | 152 GB | 0% |
| ~60+ min | **STILL creating sequences** | 177 GB | **0%** |

Training was killed after 60+ minutes. GPU never utilized. Process was stuck in sequence creation.

**Root Cause - Slow Python loop:**

```python
# Current code in SequenceDataset.__init__() - SLOW
X_seq, y_seq = [], []
for i in range(len(features) - seq_length):  # 6.2M iterations!
    X_seq.append(features[i:i + seq_length])  # List append is O(1) amortized but slow
    y_seq.append(targets[i + seq_length])

self.X = np.array(X_seq)  # Copies everything AGAIN
self.y = np.array(y_seq)
```

**Why it's slow:**
1. Python for-loop with 6.2M iterations (no vectorization)
2. List.append() causes periodic memory reallocation
3. `np.array()` at the end copies all data again
4. Total memory: 6.2M × 60 timesteps × 56 features × 4 bytes = ~83 GB just for X_seq

**Required Fix - Use NumPy stride tricks:**

```python
from numpy.lib.stride_tricks import sliding_window_view

# Option 1: sliding_window_view (NumPy >= 1.20)
def create_sequences_fast(features, targets, seq_length):
    n_samples = len(features) - seq_length
    # Creates a VIEW in O(1) time, no copy
    X = sliding_window_view(features, window_shape=seq_length, axis=0)[:n_samples]
    y = targets[seq_length:seq_length + n_samples]
    return X.copy(), y  # Single copy at the end

# Option 2: as_strided (lower level, more control)
from numpy.lib.stride_tricks import as_strided

def create_sequences_strided(features, targets, seq_length):
    n_samples = len(features) - seq_length
    n_features = features.shape[1]

    # Create view with custom strides - O(1) operation
    X = as_strided(
        features,
        shape=(n_samples, seq_length, n_features),
        strides=(features.strides[0], features.strides[0], features.strides[1])
    )
    y = targets[seq_length:seq_length + n_samples]
    return X.copy(), y  # Single O(n) copy
```

**Performance improvement:**
- Before: 60+ minutes (never finishes for 6.2M samples)
- After: ~10-30 seconds

**Acceptance Criteria:**
1. Sequence creation completes in <60 seconds for 6M+ samples
2. Output shape identical: `(n_samples - seq_length, seq_length, n_features)`
3. Output values identical to current implementation (add correctness test)
4. Add benchmark test to prevent regression
5. Memory usage should not exceed 2x the final tensor size during creation

**Test to add:**
```python
def test_sequence_creation_performance():
    """Sequence creation should complete in <60s for 1M samples."""
    features = np.random.randn(1_000_000, 56).astype(np.float32)
    targets = np.random.randint(0, 3, 1_000_000)

    start = time.time()
    dataset = SequenceDataset(features, targets, seq_length=60)
    elapsed = time.time() - start

    assert elapsed < 60, f"Sequence creation took {elapsed:.1f}s, expected <60s"
    assert dataset.X.shape == (999_940, 60, 56)
```

---

## 11. LSTM Sequence Creation OOM on Full Dataset (NOT FIXED)

**Priority:** P1 - Prevents full-dataset training

**File:** `src/ml/models/training.py` - `SequenceDataset`

**Issue:** Even with Bug #10 fix (stride tricks), the full dataset still OOMs because the `.copy()` call materializes all sequences into memory at once.

**Evidence from RunPod testing (2026-01-18):**
- Container limit: ~175 GB usable RAM
- Memory needed for all sequences: ~145 GB (10.9M × 60 × 56 × 4 bytes)
- Plus DataFrames, scalers: ~30 GB
- Total needed: ~175-200 GB → **OOM kill (exit 137)**

**Workaround:** Use `--max-samples 2000000` to cap training data

**Proper Fix Needed:**
- Implement lazy DataLoader that generates sequences on-the-fly
- Or use memory-mapped arrays
- Or chunked sequence creation

---

## 12. Blackwell GPU (sm_120) Incompatible with PyTorch 2.4 (ENVIRONMENT)

**Priority:** P0 - Blocks GPU training on Blackwell GPUs

**Issue:** NVIDIA RTX PRO 6000 Blackwell (sm_120) is too new for PyTorch 2.4.x which only supports up to sm_90.

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Workaround:** Use older GPU generation:
- A100 (sm_80) ✅
- H100 (sm_90) ✅
- RTX 4090 (sm_89) ✅

**Proper Fix:** Upgrade to PyTorch nightly with Blackwell support when available.

---

## 13. Walk-Forward CV Memory Usage (FIXED 2026-01-20)

**Priority:** P1

**Issue:** Walk-forward cross-validation with 5 splits materialized ALL fold data upfront in a list, causing memory usage of O(n_folds * data_size). With 5 folds of millions of samples each, this could cause OOM when combined with LSTM sequence creation.

**Investigation findings:**
- `WalkForwardCV.generate_folds_from_arrays()` in `src/scalping/walk_forward.py` was creating ALL fold data copies upfront
- For 5 folds of 1M samples each with 56 features, this used ~2.2 GB just for fold data
- Combined with LSTM sequences, this exceeded available memory

**Fix (v0.0.97):**
- Added `_generate_folds_from_arrays_lazy()` generator that yields folds one at a time
- Added `_generate_fold_boundaries()` to calculate boundaries without copying data
- Updated `generate_folds_from_arrays()` with `lazy=True/False` parameter
- Updated `run()`, `run_with_dataframe()`, and `run_walk_forward_validation()` with `use_lazy_folds=True` default
- Added 12 new tests for lazy fold generation

**Memory improvement:**
- Before: O(n_folds * data_size) = ~2.2 GB for 5 folds of 1M samples
- After: O(data_size) = ~0.45 GB per fold, released after each iteration

---

## Recommended Tests to Add

1. Test feature scaling with infinity values in input
2. Test LSTM evaluation with large batch sizes
3. Test memory usage estimation before training
4. Integration test with full data pipeline
5. **Test checkpoint loading with both old and new formats**
6. **Test backtest script end-to-end with trained model**
7. **Test ScalpingFeatureEngineer API consistency**
8. **Test sequence creation performance (<60s for 1M samples)** - Bug #10
9. **Test sequence creation correctness (compare stride tricks vs loop output)** - Bug #10
10. **Test walk-forward CV memory usage** - Bug #11

---

## Notes for Ralph

### All P1 Memory Bugs Fixed (v0.0.97)

**Bug #10, #11, #13 are all FIXED:**
- Bug #10: LSTM Sequence Creation Too Slow - Fixed with NumPy stride tricks
- Bug #11: LSTM Sequence Creation OOM - Fixed with `LazySequenceDataset`
- Bug #13: Walk-Forward CV Memory Usage - Fixed with lazy fold generation

**These fixes enable full-dataset training without OOM by:**
1. Using lazy sequence generation (O(n * features) instead of O(n * seq * features))
2. Using lazy fold iteration (O(data_size) instead of O(n_folds * data_size))

### Other considerations:
- Adding unit tests for edge cases
- Implementing checkpoint resumption so training can continue after crashes
- Adding memory estimation before training starts
- Implementing chunked feature engineering for large datasets
- **Standardizing checkpoint format keys across training/inference**
- **Ensuring PerformanceMetrics has consistent API**
- **Integration tests for backtest script with real models**
