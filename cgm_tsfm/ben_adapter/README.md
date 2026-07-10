# Arm B — adapting Ben's code to single-channel CGM regression (Task 4)

This directory is the literal answer to the advisor's Task 4 ("adapt Ben's code for
regression output") + his channel answer ("adapt/generalize Ben's code to accept
one channel of CGM"). It is a faithful, minimal adaptation of two files in
`~/Desktop/mod-actigraphy-advanced`:

- `src/models/pretrained_models.py` → `model.py`
- `src/models/modules.py`           → `lightning_module.py`

Ben's code is **not modified** (it's a reference project). We reimplement the
relevant pieces here, with the changes below.

## `FoundationModelClassifier` → `FoundationModelRegressor` (`model.py`)

| Ben (`pretrained_models.py`) | Here | Reason |
|---|---|---|
| `num_channels = 2  # activity + light` (line 340) | `1` (glucose) | single-channel CGM |
| `forward(activity, light, activity_mask, light_mask, day_mask=None, ...)` (595) | `forward(glucose, glucose_mask=None)` | one signal + one mask |
| stacks 2 channels `torch.stack([activity, light], dim=2)` (545) | none | one channel |
| `_combine_channels` concat/add (401-422) | removed | nothing to combine |
| `(B, D, T)` days + causal transformer across days (509-593) | removed | each sample = ONE pre-test window → ONE score (D=1); no day sequence |
| `assert activity.ndim == 3` (620) | accepts `(B, T)` | window, not patient-days |
| head `LayerNorm→Linear→GELU→Dropout→Linear(hidden, num_classes)` (384-390) | same head, `num_targets=1` | regression scalar (single-target per the advisor) |
| `ChronosEncoder` via Bolt `.encode()` (39-113) | `ChronosTorchEmbedder` via `BaseChronosPipeline.embed()` + mean-pool | version-robust; identical recipe to Arm A; supports Bolt **and** T5 |
| encoder frozen inside model | encoder frozen; `requires_grad=False` enforced in `__init__` | zero-shot embeddings |

The encoder is **injected**, so the model is testable offline with
`MockTorchEmbedder` (no download, no training).

## `FinetuneLightningModule` → `CGMRegressionModule` (`lightning_module.py`)

| Ben (`modules.py`) | Here | Reason |
|---|---|---|
| `if task=="ptb_classification": num_classes=2, is_classification=True; else: regression` (354-359) | regression only; no task-string branch | Ben's check made *any* non-PTB task silently regress — fragile by design |
| `labels = labels.float()/365.0` … `preds = logits*365.0` (588-591) | z-score with **train-set target mean/std** (`register_buffer`), de-normalize preds | `/365` is gestational-age-specific; a general per-target normalizer fits any score/scale (incl. Prices' larger range) |
| `_shared_step` reads `activity/light/*_mask/day_mask` (550-554) | reads `glucose`, `glucose_mask`, `label` | single channel |
| MSE loss (391); MSE/RMSE/R² metrics (432-448) | MSE loss; MSE/RMSE/R² **+ MAE** | kept Ben's regression metrics; added MAE (matches Arm A / diabetes-fitbit) |
| optimizer over all params | optimizer over **trainable (head) params only** | encoder frozen |

## Verify

```bash
~/Desktop/mod-actigraphy-advanced/.venv/bin/python -m cgm_tsfm.ben_adapter.smoke_test
# forward -> (B, 1); train loss decreases -> "Arm B smoke test PASSED"
```

## Grouped-CV training (done)

- `datamodule.py` — `CGMEmbeddingDataModule`: serves one CV fold's train/val/test as
  `{"glucose", "label"}` batches. "glucose" is the PRECOMPUTED Chronos embedding (the
  frozen encoder is run once via `encoders.extract_embeddings` and cached), so only the
  head trains. It exposes train-only `target_mean/std` for the z-score.
- `train.py` — `run_arm_b(dataset, embeddings, ...)`: outer **`GroupKFold(5)` on `subid`
  (the same deterministic folds Arm A uses)** + a grouped `GroupShuffleSplit` val split
  for early stopping; trains one head per fold via a Lightning `Trainer`; aggregates to
  the same `{target: {n, n_subjects, rows}}` shape as `run_arm_a`. Difference vs Arm A: a
  fixed head architecture with early stopping instead of an inner `GridSearchCV` (keeps 15
  net-trainings tractable). Runs standalone (`python -m cgm_tsfm.ben_adapter.train`) or as
  the third contender in `run_headtohead --with-arm-b`.
- Reuses `FoundationModelRegressor` unchanged via `PassthroughEmbedder` (a no-op "encoder"
  that returns the precomputed embedding), so the exact same model/module train on either
  raw windows (real encoder, for future fine-tuning) or cached embeddings (fast path).

Validated: `ben_adapter/smoke_test.py` (wiring) + a full 3-target × 5-fold run on cached
Chronos embeddings.

## Not yet done (needs real data / compute)
- Full training on the **real** CSVs + real scores (blocked on data access), head-to-head vs Arm A.
- Optional: unfreeze the encoder (LoRA fine-tuning) — the raw-window path in
  `FoundationModelRegressor` + `ChronosTorchEmbedder` already supports it; only worth it if
  the frozen arm shows promise.
