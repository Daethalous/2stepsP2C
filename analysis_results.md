# iTransformer Baseline Repo — Static Code Review

> [!NOTE]
> This is a **static review only** — no code was executed. Issues are ranked by severity.

---

## 🔴 Critical / Will Crash at Runtime

### 1. `InvertedTransformerModel.__init__` receives `embedding_dim` but expects `D`
**File:** [model_placeholder.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/model_placeholder.py#L142-L166) + [main.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/main.py#L427)

`main.py` line 427 passes `**model_params` from config, which contains `embedding_dim`, `num_layers`, `nhead`, `dim_feedforward`, `dropout`. But the constructor signature uses `D` (not `embedding_dim`):

```python
# main.py line 427
model = InvertedTransformerClass(T_in=T_in, S_out=S_out, N=N, **model_params)

# config.yaml model.params:
#   embedding_dim: 128  ← this key
#   num_layers: 3
#   ...

# model_placeholder.py constructor:
def __init__(self, T_in, S_out, N, D=None, ...)  # ← expects "D", not "embedding_dim"
```

**Result:** `TypeError: __init__() got an unexpected keyword argument 'embedding_dim'`

The constructor does read `embedding_dim` from the *global* `_CONFIG` dict as a fallback default, but `**model_params` will still crash because `embedding_dim` is not a valid keyword argument.

---

### 2. `run_single_horizon` uses stale `dataset_loader` — horizon change is ignored
**File:** [main.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/main.py#L383-L391)

```python
cfg["data"]["prediction_horizon"] = int(horizon)  # updates local copy
# ...
datasets = dataset_loader.build_datasets(scaler_obj)  # ← uses dataset_loader's OWN stored self.prediction_horizon
```

`dataset_loader` was constructed once in `main()` (line 615) and stores `self.prediction_horizon` from the *original* config. Changing `cfg` locally does NOT update `dataset_loader.prediction_horizon`. So when iterating over `horizon_list = [96, 192, 336, 720]`, all runs use the **same horizon** (the initial one from config).

---

### 3. `dataset_loader.py` line 158: timestamp column leaks back into numeric DataFrame
**File:** [dataset_loader.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/dataset_loader.py#L158)

```python
return numeric_df.reset_index(drop=False) if ts_col is not None else numeric_df.reset_index(drop=True)
```

When `ts_col is not None`, `drop=False` re-inserts the datetime index as a column. This datetime column will then fail `ensure_numpy`'s `select_dtypes(include=[np.number])` check OR, if pandas treats the re-inserted column as numeric (epoch timestamps), it corrupts the data array with timestamp values as a variate.

**Fix:** Should be `drop=True` in both cases (the timestamp was already used for sorting).

---

### 4. `dataset_loader.py` deprecated `fillna(method="ffill")`
**File:** [dataset_loader.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/dataset_loader.py#L377)

```python
df = df.fillna(method="ffill")  # ← FutureWarning in pandas 2.x, removed in pandas 3.x
```

With `pandas==1.5.3` (requirements.txt), this still works but is deprecated. With any newer pandas version, this will raise `TypeError`.

**Fix:** Use `df = df.ffill()` instead.

---

### 5. `Evaluator.__init__` consumes first batch via `_peek_batch_shape` — breaks iteration
**File:** [evaluation.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/evaluation.py#L152-L180)

```python
def _peek_batch_shape(self):
    iterator = iter(self.dataloader)
    first = next(iterator)  # ← consumes first batch
```

Creating `iter(self.dataloader)` and calling `next()` consumes the first batch. When `evaluate()` later iterates the dataloader, if the dataloader **does not shuffle**, that first batch is lost. With `shuffle=False` on the test set, this means evaluation skips the first batch of test data.

For non-shuffled test sets with `drop_last=False`, **one batch of test samples is silently dropped**.

---

### 6. `trainer.py` uses `history` dict for early stopping counter — type conflict
**File:** [trainer.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/trainer.py#L215-L217)

```python
self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
# ...
self.history["_early_stop_counter"] = 0  # ← int, not List[float]
```

The type annotation says `Dict[str, List[float]]` but `_early_stop_counter` stores an `int`. This is a type violation. Won't crash Python at runtime (no enforcement), but will cause issues if any downstream code iterates `self.history` expecting all values to be lists.

---

## 🟠 Likely Bugs / Semantic Errors

### 7. `BaseModel.forward` uses `self.device` property but BaseModel has **no parameters**
**File:** [model_placeholder.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/model_placeholder.py#L83-L105)

```python
def forward(self, x):
    model_device = self.device  # calls property
    if x.device != model_device:
        x = x.to(model_device)
```

The `device` property (line 120-126) falls back to `torch.device("cpu")` when there are no parameters (`StopIteration`). `BaseModel.__init__` registers no parameters. So `self.device` always returns CPU. The self-test at the bottom (line 367-370) creates `BaseModel` and passes a CUDA tensor — it will silently move to CPU, compute there, and return a CPU tensor, which mismatches the caller's expected device.

---

### 8. `minmax` inverse_transform uses wrong formula for constant channels
**File:** [scaler.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/scaler.py#L319-L329)

During `fit`, when `max - min < eps`, `diff` is set to `1.0`, and `self.min`/`self.max` keep original values. During `transform`: `(x - min) / 1.0`. During `inverse_transform`: `x * (max - min clamped to 1.0) + min`. This means the inverse doesn't exactly undo the transform for constant channels because the stored `self.max` isn't updated — the `denom` is recomputed from `maxv - minv` and clamped on-the-fly. This is correct but fragile: the clamping logic is duplicated in `transform`, `inverse_transform`, and `fit` with slightly different variable names, making it error-prone.

---

### 9. `Trainer._train_epoch` loss accumulation: weighted by batch_size but criterion uses `mean` reduction
**File:** [trainer.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/trainer.py#L338-L340)

```python
loss_item = float(loss.detach().cpu().item())  # MSELoss(reduction='mean') → per-element mean
total_loss += loss_item * batch_size           # re-weight by batch size
total_samples += batch_size
```

`MSELoss(reduction='mean')` computes `sum(err²) / (B * S_out * N)`. Then the code multiplies by `batch_size` (= B). The final `avg_loss = total_loss / total_samples` gives `sum(err²) / (total_B * S_out * N)`, which is correct for a weighted average. **This is actually fine** — just noting that it only works correctly because MSELoss divides by all elements, not just B.

---

### 10. `Evaluator` misses passing `config` to constructor
**File:** [main.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/main.py#L519)

```python
evaluator = Evaluator(model=inference_model, dataloader=dataloaders["test"], scaler=scaler_obj, device=device)
```

The `Evaluator.__init__` signature accepts `config` as an optional kwarg. Here it's not passed, so the Evaluator will try to load `config.yaml` from CWD (line 132). If the CWD is not the repo directory (e.g., if `main.py` is run from a parent directory), the evaluator may load the wrong config or use empty defaults, which changes behavior of `save_predictions`.

---

## 🟡 Warnings / Minor Issues  

### 11. `np.min` / `np.max` with `dtype=np.float64` — unsupported kwarg
**File:** [scaler.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/scaler.py#L195-L196)

```python
minv = np.min(arr, axis=0, dtype=np.float64)
maxv = np.max(arr, axis=0, dtype=np.float64)
```

`np.min` and `np.max` do **not** accept a `dtype` keyword in NumPy < 1.24. With `numpy==1.23.5` (from requirements.txt), this will raise:
```
TypeError: _wrapreduction() got an unexpected keyword argument 'dtype'
```

**Fix:** Cast the array first: `np.min(arr.astype(np.float64), axis=0)`

---

### 12. `torch.cuda.amp` deprecation warnings
**File:** [trainer.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/trainer.py#L133)

```python
self.scaler_amp = torch.cuda.amp.GradScaler()  # deprecated in torch 2.0+
```

PyTorch 2.0+ recommends `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`. With `torch==2.0.1` this will emit deprecation warnings.

---

### 13. `torch.load` without `weights_only` parameter
**File:** [main.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/main.py#L504), [utils.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/utils.py#L561)

```python
ckpt = torch.load(best_checkpoint_path, map_location=device)
```

PyTorch 2.0+ emits `FutureWarning: You are using torch.load without specifying weights_only`. In future PyTorch versions default will change to `weights_only=True`, which would break loading checkpoints that contain non-tensor data (config dicts, etc.).

---

### 14. `_sanitize_numpy_in_dict` aggressively converts everything to numpy
**File:** [utils.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/utils.py#L659-L676)

```python
def _sanitize_numpy_in_dict(d):
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        else:
            try:
                out[k] = np.asarray(v)  # converts strings, bools, dicts, lists, etc.
```

This converts string values to `np.array('some_string')`, which changes their type on save/load. For checkpoint metadata containing config dicts with string values, this corrupts the data.

---

### 15. Scaler `save()` method writes `.npz` but `main.py` saves with `.pkl` extension
**File:** [main.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/main.py#L664)

```python
scaler_save_path = str(Path(exp_dir) / "scaler.pkl")
if hasattr(scaler_obj, "save"):
    scaler_obj.save(scaler_save_path)
```

`Scaler.save()` uses `np.savez_compressed`, which creates an `.npz` file. But if the path has suffix `.pkl`, it will write the data as `.pkl` (since `Scaler.save` only appends `.npz` if suffix is empty). The resulting file is actually npz format but named `.pkl`, which is confusing and will break `Scaler.load()` if it expects `.npz`.

---

### 16. `model_placeholder.py` self-test: BaseModel on CUDA tensor silently moves to CPU
**File:** [model_placeholder.py](file:///c:/Computing/GitHub/2stepsP2C/outputs/iTransformer_baseline_repo/model_placeholder.py#L367-L370)

```python
base = BaseModel(T_in=t_in, S_out=s_out, N=n_vars)
yhat3 = base(x)  # x is on CUDA device, but BaseModel has no params → device="cpu"
```

The assertion `yhat3.shape == (batch_size, s_out, n_vars)` passes, but `yhat3` is on CPU while `x` was on CUDA. Any downstream code expecting same-device tensors would fail.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 **Critical** | main.py + model_placeholder.py | `**model_params` passes `embedding_dim` → `TypeError` |
| 2 | 🔴 **Critical** | main.py | Horizon change ignored — `dataset_loader` uses stale state |
| 3 | 🔴 **Critical** | dataset_loader.py | `reset_index(drop=False)` leaks timestamp column |
| 4 | 🟠 Medium | dataset_loader.py | Deprecated `fillna(method="ffill")` |
| 5 | 🟠 Medium | evaluation.py | `_peek_batch_shape` consumes first test batch |
| 6 | 🟡 Low | trainer.py | Type annotation mismatch for `_early_stop_counter` |
| 7 | 🟡 Low | model_placeholder.py | `BaseModel.device` always CPU (no params) |
| 8 | 🟡 Low | scaler.py | Duplicated clamping logic for constant channels |
| 9 | ✅ OK | trainer.py | Loss accumulation is actually correct |
| 10 | 🟡 Low | main.py | Evaluator missing `config` kwarg |
| 11 | 🔴 **Critical** | scaler.py | `np.min/np.max` with `dtype` kwarg unsupported in numpy 1.23.5 |
| 12 | 🟡 Low | trainer.py | `torch.cuda.amp` deprecation |
| 13 | 🟡 Low | main.py, utils.py | `torch.load` missing `weights_only` |
| 14 | 🟡 Low | utils.py | `_sanitize_numpy_in_dict` corrupts string values |
| 15 | 🟡 Low | main.py + scaler.py | `.pkl` extension but `.npz` format |
| 16 | 🟡 Low | model_placeholder.py | Self-test device mismatch |
