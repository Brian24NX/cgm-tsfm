# Arm A — Every File, Meticulously (with *why*)

*Read `01_CONCEPTS.md` first. Open each `.py` file beside this. Arm A = frozen Chronos embeddings + a classical sklearn regressor. Files covered: `config.py`, `data.py`, `encoders.py`, `handcrafted.py`, `regression.py`, and the three `run_*.py` scripts.*

The station order (from `00_START_HERE`): **`data.py` → `encoders.py` → `regression.py`**, driven by a `run_*.py` script, with `config.py` holding settings and `handcrafted.py` providing the comparison baseline.

---

## `config.py` — the settings file

**Job:** one place for every setting, so the rest of the code never hard-codes magic values. No logic here, just constants.

Key pieces and *why they exist*:
- `COGNITIVE_TARGETS = ["grids_cognitive_score", "symbols_cognitive_score", "prices_cognitive_score"]` — the **three scores** we predict, **one model each** (the advisor's instruction: single-target, three separate models). The names match the real CSV columns exactly.
- `PRICES_IS_INVERTED = True` — in the raw data the "Prices" score is **backwards** (higher = worse). This flag tells the loader to **negate** it so all three read "higher = better." One flag = easy to flip if we learn otherwise.
- `SUBJECT_COL = "subid"`, `SESSION_COL = "session"`, `GLUCOSE_COL = "Glucose_Before_Test"` — the column names in the CSV. `subid` is the **grouping key** for leakage-free CV (Concepts Part 9).
- `RANDOM_STATE = 42`, `OUTER_CV_SPLITS = 5`, `INNER_CV_SPLITS = 3` — reproducibility (fixed random seed) + the 5-fold outer / 3-fold inner nested-CV design.
- `DATA_DIR`, `COHORT_FILES` — where the real CSVs live and their filenames. (`COHORT_FILES` was pointed at `Updated_Cohort2...` — the corrected file that gives all 20 patients.)
- `CACHE_DIR` — where extracted embeddings are saved so we don't recompute them.
- `@dataclass WindowConfig` — the windowing settings: `min_readings=3` (drop sessions with fewer than 3 readings) and `max_readings=None` (keep the whole window; set it to e.g. 30 to cap at the most-recent 2.5 h).
- `@dataclass EncoderConfig` — which encoder: `encoder_type="chronos"`, `pretrained_model="amazon/chronos-bolt-small"`, `pooling="mean"`, `device="cpu"`, `batch_size=32`.

> A **`@dataclass`** is just a lightweight Python class for bundling named settings together (so we can pass "a window config" around as one object instead of loose variables).

---

## `data.py` — Station 1: get the data and cut it into windows

**Job:** produce a `CGMDataset` — the tidy object the rest of the pipeline consumes — either from the **real CSVs** or from a **synthetic** (fake) generator with the identical shape.

### `parse_glucose(raw)` — turn the stored text into numbers
In the CSV, `Glucose_Before_Test` is stored as a **string** that looks like a list: `"[140, 138, nan, 135]"`. This function converts that text into a clean array of floats:
```python
parsed = eval(s, {"__builtins__": {}}, {"nan": np.nan, "NaN": np.nan})
...
arr = arr[~np.isnan(arr)]           # strip missing readings
```
- It uses `eval` (with a **restricted namespace** — `{"__builtins__": {}}` disables dangerous built-ins) because Cohort 2 writes literal `nan` tokens inside the text, which the safer `ast.literal_eval` chokes on. This exactly mirrors the original `diabetes-fitbit` loader, so our numbers match theirs.
- `arr[~np.isnan(arr)]` **drops missing readings** (`NaN` = "not a number" = a gap in the sensor data). `~np.isnan(arr)` is a boolean mask meaning "keep the non-missing ones."
- Returns `None` for empty/unparseable cells — that's how a session with no glucose gets dropped later (this is what caused "19 vs 20 patients" before the data was fixed).

### `apply_window(glucose, window)` — enforce the window rules
```python
if glucose is None or glucose.size < window.min_readings:   # too short → drop
    return None
if window.max_readings is not None and glucose.size > window.max_readings:
    glucose = glucose[-window.max_readings:]                 # keep most-recent N
```
Drops windows shorter than `min_readings`; optionally keeps only the **most-recent** `max_readings` (the negative index `[-N:]` = "last N elements"). This is the code meaning of "**window**" (Concepts Part 4).

### `class CGMDataset` — the tidy container
A `@dataclass` holding four aligned things, one entry per session:
- `windows`: a **list of 1-D arrays** — each is one session's glucose readings (variable length).
- `subjects`: the `subid` per session — the **CV grouping key**.
- `sessions`: the session id per session.
- `targets`: a dict `{score_name → array of scores}` (NaN where a score is missing).

Helpers: `target_mask(target)` returns "which sessions have a non-missing value for this score" (so we only train on sessions that actually have that score); `summary()` prints the human-readable overview you've seen ("956 sessions, 20 subjects…").

### `within_subject_normalize(y, subjects, mode)` — the subject-baseline test
```python
for s in np.unique(subjects):
    idx = subjects == s
    vals = y[idx]
    out[idx] = vals - vals.mean()        # "center": subtract this subject's own mean
```
For each subject, subtract **their own average score** — so the model predicts *deviation from personal baseline* instead of the raw score. This is the direct test of "is the signal just per-child baseline?" (Concepts Part 9, end). `mode="none"` returns the scores unchanged. ⚠️ It uses each subject's own sessions to compute their mean ("oracle centering") — so it answers *"is there a within-child effect at all?"*, not *"can we predict a new child."*

### `load_real_data(...)` — read the real CSVs
Reads each cohort CSV with **pandas** (`pd.read_csv`), harmonizes the slightly-different column names between cohorts (e.g. `session_id → session`), applies `parse_glucose` to the glucose column, and builds a `CGMDataset` via the shared `_frame_to_dataset`. It **mirrors the original study's loader exactly**, so it runs unchanged on the real files. If the files aren't present it raises a clear error telling you to get them or use the synthetic generator.

### `generate_synthetic_data(...)` — fake data with the real shape
This is why the whole pipeline could be built and tested **before** the real data arrived. It fabricates ~20 subjects × ~32–67 sessions, each with:
- a **realistic glucose curve** — a "mean-reverting random walk" (`g[t]` wanders around the subject's personal mean, with noise), clipped to 40–400 mg/dL, with occasional dropped readings (like a real CGM):
  ```python
  g[t] = g[t-1] + 0.6*(subj_gluc_mean - g[t-1])/4 + rng.normal(0, subj_gluc_vol/2)
  ```
- **scores** built so that *most* of the variation is **between subjects** (a per-subject baseline `base[t]`), plus a **weak glucose effect**, plus noise — deliberately mimicking the real problem's structure.
- Two independent "signal knobs": `signal_strength` (a between-subject effect, removed by centering) and `within_subject_signal` (a within-subject effect, survives centering). Setting the second >0 lets us **prove the pipeline can detect a within-child signal when one exists** (centered R² jumps to ~0.93) — so a real ~0 means "no signal," not "broken detector."

> **Why this matters for your professor:** the synthetic generator is not busywork — it's how we validated that the *machinery* works and reproduces the real problem's "subject-baseline-dominated" character. Any *synthetic* number is a plumbing check, never evidence.

---

## `encoders.py` — Station 2: glucose window → 512-number embedding

**Job:** turn each window into a fixed-size embedding. Two encoders share one interface, plus a caching wrapper.

### `class MockEncoder` — the offline stand-in
Produces a small fixed vector of **hand-computed summary statistics** (mean, std, %low, slope…) per window — **no Chronos, no download**. It exists so the whole pipeline can run offline for smoke tests/CI. It is *not* a foundation model; it stands in for one so we can test plumbing.

### `class ChronosEncoder` — the real encoder
This is the scientific heart of Station 2. In `__init__` it loads the pretrained model:
```python
from chronos import BaseChronosPipeline
self.pipeline = BaseChronosPipeline.from_pretrained(
    self.cfg.pretrained_model, device_map=self.cfg.device, torch_dtype=dtype)
```
`BaseChronosPipeline` auto-detects Bolt vs T5 from the model name, so one code path serves both. `device_map` = `"cpu"` or `"cuda"` (GPU).

The `encode()` method is the recipe (this is the "input format" your professor cares about — see also `docs/06`):
```python
for i in range(0, len(windows), bs):                       # process in batches of bs
    batch = [torch.tensor(np.asarray(w, dtype=np.float32)) for w in windows[i:i+bs]]
    with torch.no_grad():                                  # no training → save memory
        emb, _ = self.pipeline.embed(batch)                # (B, num_patches+1, d_model)
        pooled = emb.mean(dim=1)                            # mean over the token axis → (B, d_model)
    out.append(pooled.to(torch.float32).cpu().numpy())
```
Line by line:
- We pass a **list of 1-D tensors** (one per window, variable length) — Chronos is **univariate**, so input is `(batch, length)` with **no channel dimension** (this is *why* the project is single-channel glucose). Chronos left-pads the batch internally.
- `self.pipeline.embed(batch)` returns Chronos's per-token hidden vectors, shape `(B, num_patches+1, 512)` — one 512-vector per patch/token, plus one extra summary token.
- `emb.mean(dim=1)` — **mean-pooling**: average over the token axis to collapse each series to **one** 512-vector → `(B, 512)`. (This averaging recipe is exactly what the ICML paper the advisor pointed to uses.)
- `torch.no_grad()` = "we're only running the model forward, not training it" (the encoder is **frozen**) — saves memory and time.
- `.cpu().numpy()` — move results off the GPU and convert to a NumPy array for the sklearn side.

**Why no normalization / no gap-filling?** Because Chronos scales each series internally and turns missing readings into a mask token. So we feed **raw mg/dL** — one of the practical advantages of using a TSFM (Concepts Part 3).

### `extract_embeddings(windows, cfg, cache_dir, use_cache)` — the caching wrapper
Running Chronos is the expensive step, and we reuse the same embeddings for all 3 scores and every model variation. So this function:
```python
key = f"{cfg.encoder_type}__{model}__{pooling}__{_windows_hash(windows)}"
cache_path = Path(cache_dir) / f"{key}.npy"
if use_cache and cache_path.exists():
    return np.load(cache_path)          # already computed → just load it
...
np.save(cache_path, emb)                # compute once, save for next time
```
It builds a cache filename from the model + a **hash of the exact windows** (so different data → different cache), and stores the result as a `.npy` file. First run computes; every later run is instant. This is why re-running the real analysis is fast.

---

## `handcrafted.py` — the OLD-way features (comparison baseline)

**Job:** compute Mack's **43 hand-crafted glucose statistics** per window — a **line-for-line port** of the original study's `features.py`, verified numerically identical. It exists so Chronos (raw-data learning) can be compared against feature engineering under the *same* evaluation.

`extract_glucose_features(arr)` returns a dict of 43 named numbers: basic stats (mean, std, min, max, median, IQR, skew, kurtosis), rate-of-change stats, clinical measures (% time-in-range 70–180, % low, % high, LBGI/HBGI hypo/hyper risk indices, MAGE, CONGA, J-index…), trend (slope, first/last/delta), and the last-5 raw readings. `extract_handcrafted_features(windows)` stacks them into an **`(N, 43)` matrix**, NaNs→0, aligned to the windows.

> **You don't need to know all 43 formulas.** The point to explain: *these are human-chosen glucose summaries; the model built on them is the "feature-engineering" arm; ours (Chronos) is the "learned-representation" arm; both go through the identical scorer so the comparison is fair.* Per the advisor's latest steer, this is now just a sanity-check reference, not the headline.

---

## `regression.py` — Stations 4+5 (Arm A): predict + score fairly

**Job:** given the `(N, 512)` embeddings and the scores, fit classical models and report RMSE/MAE/R² under leakage-free nested grouped CV. **This file is Arm A's brain.** It is written to take *any* `(N, d)` feature matrix — that's what makes Chronos-vs-hand-crafted a fair, one-line swap.

### `get_regression_models()` — the contenders + their tuning grids
```python
"Ridge": (Ridge(...), {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}),
"SVR":   (SVR(kernel="rbf"), {"model__C": [...], "model__gamma": [...]}),
"Linear":(LinearRegression(), {}),
```
Each entry is `(the model, the grid of knob-values to try)`. Ridge's `alpha` is the regularization strength; SVR's `C`/`gamma` control its flexibility; Linear has no knobs (`{}`). (XGBoost is added later if installed.) These mirror the original study's model list so both arms are comparable. *(See Concepts Part 7 for why each model.)*

### `nested_group_cv(...)` — the heart: honest evaluation of one model
This is the single most important function to understand. It runs 5-fold **grouped** CV, tuning inside each fold. Walk it:
```python
outer = _make_splitter(outer_splits, cv_scheme)     # GroupKFold(5) for "group"
for train_idx, test_idx in outer.split(X, y, groups):
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
        "subject leakage between train and test!"    # HARD guard: no child in both
    steps = [("scaler", StandardScaler())]           # standardize features...
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components)))   # ...optionally PCA...
    steps.append(("model", estimator))               # ...then the model
    pipe = Pipeline(steps)
    if param_grid:                                    # tune the knob with an INNER CV
        gs = GridSearchCV(pipe, param_grid, scoring="neg_mean_squared_error",
                          cv=_make_splitter(inner_splits, cv_scheme), n_jobs=-1)
        gs.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        best = gs.best_estimator_
    else:
        best = pipe.fit(X[train_idx], y[train_idx])
    pred = best.predict(X[test_idx])                  # predict the untouched test fold
    result.folds.append(FoldResult(
        rmse=np.sqrt(mean_squared_error(y[test_idx], pred)),
        mae=mean_absolute_error(y[test_idx], pred),
        r2=r2_score(y[test_idx], pred)))
```
Key ideas, in order:
1. **`GroupKFold`** splits by *subject*, so `train_idx` and `test_idx` never share a child. The `assert` is a **hard safety guard** — if leakage ever happened, the program crashes rather than reporting a fake score.
2. **`Pipeline`** chains "scale → (optional PCA) → model" into one object. This is the leakage-safe way: when the pipeline is `.fit()` on the training fold, the scaler and PCA learn *only from training data*, then get applied to the test fold. If we'd scaled the whole dataset up front, the test fold's statistics would leak into training. The pipeline prevents that automatically.
3. **`GridSearchCV`** does the **inner** cross-validation that tries each knob value and keeps the best — using an *inner* GroupKFold, so tuning also never peeks at the outer test fold. This is **nested CV**. `scoring="neg_mean_squared_error"` = "pick the knob with the lowest error." `n_jobs=-1` = use all CPU cores.
4. **`.predict(X[test_idx])`** on the held-out fold, then compute **RMSE/MAE/R²** on it. We store per-fold results and later report **mean ± std** across the 5 folds.

`cv_scheme="session"` swaps `GroupKFold` for a plain `KFold` that ignores subjects — the deliberately-leaky diagnostic (Concepts Part 9).

### `run_arm_a(dataset, embeddings, ...)` — do it for every score
```python
for target in targets:
    mask = dataset.target_mask(target)                        # sessions that have this score
    X, y, g = embeddings[mask], dataset.targets[target][mask], dataset.subjects[mask]
    y = within_subject_normalize(y, g, target_norm)           # no-op unless "center"/"zscore"
    # baseline first:
    base = nested_group_cv(X, y, g, DummyRegressor("mean"), {}, ...)   # the bar to beat
    for name, (est, grid) in models.items():
        res = nested_group_cv(X, y, g, est, grid, ...)        # each real model
    all_results[target] = {"n": ..., "n_subjects": ..., "rows": [...]}
```
For each of the 3 scores it: selects the sessions that have that score, (optionally centers the target), runs the **baseline** and each model through `nested_group_cv`, and collects a table of `{model → RMSE/MAE/R²}`. `eff_pca = min(pca_components, X.shape[1])` just clamps PCA to the number of features available. Returns a nested dict the printing/markdown functions format.

### `format_results(...)` — pretty-print the tables
Turns the results dict into the aligned text tables you've seen ("[grids_cognitive_score] … Ridge … R2 …"). No logic, just formatting.

---

## The three `run_*.py` scripts — the "main" entry points

These are the files you actually *run* (`python -m cgm_tsfm.<name> …`). Each parses command-line options (`argparse`), builds the dataset, extracts embeddings, calls `run_arm_a`, and prints/saves results.

### `run_demo.py`
The simplest end-to-end demo of Arm A. It runs `run_arm_a` under **both** `group` and `session` CV and prints them side by side — so you directly see the **subject-baseline diagnostic** (group R²≈0 vs session R²). Options: `--encoder mock|chronos`, `--real`, `--device cuda`. This is your best "show it live" command (fast, prints real numbers).

### `run_headtohead.py`
Runs the **comparison**: pushes the Chronos embeddings *and* the 43 hand-crafted features through the *same* `run_arm_a`, and (with `--with-arm-b`) also trains Arm B — then prints a per-target table with the best model per representation and a "verdict." Writes `results/headtohead_*.md`. This is the file that produces your headline result table.

### `run_sweep.py`
Tries **one setting at a time** to see what helps, holding the rest fixed: Chronos size (`--kind checkpoint`), window length (`window`), pooling mean/last (`pooling`), **PCA** size (`pca`), and within-subject norm (`targetnorm`). Each row = "this setting → best-model R² per target." Writes `results/sweep_*.md`. This is the "knobs" file (each knob answers a real question — see `PLAIN_ENGLISH_SUMMARY.md` Q13).

---

**Next:** `03_CODE_ARM_B.md` covers the `ben_adapter/` neural-network arm and the PyTorch/Lightning ideas.
