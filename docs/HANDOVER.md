# Handover — CGM → cognition (TSFM arm)

**From:** Brian Zhou · **To:** Ben · **Date:** 2026-09-17
**Purpose:** everything you need to pick this project up, in one file. Read this before the
companion deck (`Handover_CGM_Cognition.pptx`, 16 slides with speaker notes — it is the same
content with charts).

---

## 0. The 60-second version

We ask: **does a child's glucose in the 2 hours before a cognitive test predict their score on
that test?** Data is 20 youth with Type 1 Diabetes wearing a Dexcom G6 (one reading / 5 min) who
took three short tests on a phone app ~5×/day for 10 days.

**The answer so far is no — nothing predicts better than a trivial mean-predictor.** That holds
across 13 separate attempts (encoder sizes, pooling, dimensionality reduction, glucose-regime
subgroups, a trainable head, window lengths from 15 min to 3 h). The pipeline is built,
validated, and reproducible; it is the finding that is flat, not the plumbing — see §4.

**What the project needs from you is not more modelling. It is two decisions** — see §8.

---

## 1. The study

| | |
|---|---|
| Participants | **20** (grant target was N=92; more is not expected) |
| Unit of analysis | **one test session** = one row. 956 sessions total; 32–67 per participant |
| Input | the CGM readings before that session (now fixed at the last **24 readings = 2.0 h**) |
| Targets | 3 separate regressions: **Grids**, **Symbols**, **Prices**. For all three, **lower = better** |
| Glucose | 68,206 readings, mean 177 mg/dL, 58% in range, **only 2.3% below 70** |

**Our lane is prediction only.** The lab splits this three ways: association / mixed-effects =
Phil, classical ML on hand-crafted features = Mack, DL/TSFM prediction = us. Mack's arm already
came out flat, which is why this one is framed as a fair comparison rather than a promised win.

**What the three scores are** (I derived Grids from the data — the column name explains nothing):

- **Grids** — mean placement error in grid cells, 6 objects on a 5×5 grid. 0 = all six perfect.
  **31% of sessions score exactly 0**, so the target has a hard floor and a long right tail.
  This matters: least-squares on a zero-inflated bounded target is the wrong likelihood.
- **Symbols** — a response time in seconds.
- **Prices** — a percentage in steps of 10, i.e. a count out of 10 items. Only 10 distinct values.

---

## 2. The pipeline

Input is single-channel. Frozen **Chronos-Bolt-small** (48M, `d_model` 512) encodes each window;
a regressor sits on top. Chronos is never fine-tuned.

```
glucose window (24 readings)
   → instance-normalised inside Chronos: (x − mean) / population_sd, loc/scale discarded
   → patched at 16 readings/token → 2 patch tokens + 1 [REG] token = 3 tokens
   → embed() → (B, 3, 512) → pool over the token axis → 512 features per session
   → Arm A  or  Arm B
```

**Arm A** (`cgm_tsfm/regression.py`) — `StandardScaler → [optional PCA] → Ridge | SVR(rbf) |
LinearRegression | DummyRegressor`, all inside one sklearn `Pipeline` so preprocessing is fitted
per-fold and cannot leak.

**Arm B** (`cgm_tsfm/ben_adapter/`) — **this is your architecture.** It is your frozen-encoder +
trainable-head design from `mod-actigraphy-advanced`, converted from 2-channel classification to
1-channel regression. Head is `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) →
Linear(256→1)`, **132,609 parameters**, AdamW (lr 1e-3, wd 0.01), MSE on the z-scored target,
batch 64, max 100 epochs, EarlyStopping(patience=8) on a grouped validation split.
`ben_adapter/README.md` has a line-by-line change map against your original.

**Evaluation** — subject-grouped nested CV: outer `GroupKFold(5)` on participant id, inner
`GroupKFold(3)` `GridSearchCV` on `neg_mean_squared_error`. An assert checks train/test
participants are disjoint on every fold. Metric is R² against a mean-predictor.

---

## 3. The result

Arm A, uniform 2 h window, 916 sessions, fold-averaged R² (`results/headtohead_real.md`):

| | Grids | Symbols | Prices |
|---|--:|--:|--:|
| **Mean-predictor (the baseline itself)** | −0.071 | −0.253 | −0.011 |
| Chronos 512-d + Ridge/SVR (Arm A) | −0.135 ± 0.033 | −0.290 ± 0.266 | −0.015 ± 0.019 |
| Hand-crafted 43 features + Ridge/SVR | −0.088 ± 0.039 | −0.257 ± 0.316 | −0.012 ± 0.013 |
| Chronos + MLP head (Arm B) | −0.329 ± 0.128 | −0.581 ± 0.406 | −0.170 ± 0.090 |

> ### ⚠️ The one thing to internalise before reading any number here
> **0 is not the neutral point.** With only 4 held-out participants per fold, R² is computed
> against each *test* fold's own mean while the model saw the *training* fold's mean. A predictor
> that ignores the input entirely therefore scores **−0.071 / −0.253 / −0.011**, not 0. Read every
> result as a **gap to that row**, not as a distance from zero. Arm A *ties* the trivial baseline;
> it does not collapse below it.

Two more things worth knowing up front:

- **Symbols is unstable.** Its fold-to-fold SD (±0.266) is larger than the value itself. Quote
  Grids and Prices with confidence; hedge on Symbols.
- **Arm B is the worst arm, and that is expected.** 132,609 parameters on ~612 training rows per
  fold is 217 parameters per example. This is a capacity problem, not a bug.

---

## 4. Why we believe it is the data and not the code

**(a) Known-answer check** (`results/rigor_real.md`). Feed the *same* embeddings and the *same*
folds a question we know is answerable — predict a property of the glucose itself:

| glucose target | best R² |
|---|--:|
| glucose SD (variability) | **+0.416** |
| mean glucose (absolute level) | −0.114 |
| % time > 180 | −0.079 |

Variability is recovered well, so the features do carry real information about the curve.
**Absolute level is not** — because Chronos instance-normalises each series before the encoder
sees it and we discard the returned `loc`/`scale`. Report this as **partial**, not as a clean
pass. If you want level back, concatenate the raw `loc`/`scale` onto the feature vector.

**(b) Shuffle test** — reassign the scores to the wrong sessions 200× and recompute the full
grouped CV each time. `p` = share of scrambled runs that did at least as well:

| | Grids | Symbols | Prices |
|---|--:|--:|--:|
| p | 1.000 | 1.000 | 0.801 |

Scrambled scores do as well as the real ones. Note the scrambled runs centre on ≈ **−0.05**, not
0 — same small-test-fold effect as above.

**(c) Before any model at all.** The largest absolute Pearson correlation between any glucose
summary statistic and any score is **0.098**. There is very little there to find, and
**62–92% of the score variance is within-participant**, i.e. session-to-session, not "who you
are". (That last figure *corrects* an earlier claim in this repo that between-participant
differences dominate — they do not.)

---

## 5. Three real errors found in our own work — so you don't re-trip them

1. **Pooling averaged over padding** (`encoders.py`, fixed). `emb.mean(dim=1)` averaged the token
   axis *including* left-padding positions, so a short session's embedding depended on which
   other sessions shared its batch. A 13-reading window encoded alone vs beside a 288-reading one
   gave **cosine 0.219**. Across 956 sessions, **861 (90%) had cosine < 0.9 against correctly
   pooled vectors; median 0.37.** Fixed by bucketing windows by token count so every batch is
   padding-homogeneous; verified bit-identical to solo encoding (max diff 2.4e-7) and invariant to
   batch size and order. Regression test: `cgm_tsfm/test_pooling_invariance.py`.
   The embedding cache key carries `EMBEDDING_RECIPE_VERSION` so stale `.npy` can never be reused.
   *Note:* under the current uniform-window default every session is exactly 3 tokens, so the bug
   cannot bite — it only mattered with variable lengths, and still would if you raise
   `max_readings` without `require_full`.
   **Fixing it made the numbers slightly worse**, because the buggy blurring was acting as an
   accidental smoother. `01_FACT_SHEET.md` §"Why fixing a bug made Arm B worse" has the full argument.
2. **The window was never uniform.** `max_readings` only *capped* long sessions, so every earlier
   "2 h window" run actually had lengths from 3 to 24 readings. Now `require_full=True` drops
   short sessions instead: **916/956 kept (95.8%), all 20 participants**, every window exactly
   24 readings. Only 1 session in 956 has an internal gap, so it is uniform in real time too.
3. **PCA silently discarded appended features.** Adding hand-crafted columns to the 512-d vector
   and then reducing with PCA(32) dropped them almost entirely (loading 1.0000 → 0.0853). Put
   appended features *after* the reduction, not before.

---

## 6. What we do NOT claim

- **Not** that no glucose→cognition effect exists. With 20 participants and 4 per test fold, Arm A's
  own fold-to-fold SD is ±0.019 (Prices) to ±0.266 (Symbols) — only a large effect would have been
  visible above that.
- **Not** that Chronos is unsuitable for CGM. It recovers glucose variability at R² 0.416 here.
- **Not** that 2 h is the right window on evidence. It was chosen on physiological grounds
  (the K01 hypothesis) and **pre-committed**. Windows 15 min–3 h were all measured: across the
  48 cells (8 windows × 3 scores × 2 sweeps) the differences between windows are 0.005–0.06 against
  a fold-to-fold spread of ±0.34–0.44 on that comparison — roughly 80× smaller, so the data cannot
  distinguish them. Picking the best-scoring cell would have been reading noise.
- **Not** that gradient boosting was tried. `regression.py` wraps the xgboost import in
  `try/except` and xgboost is **not installed**, so it silently never ran.

---

## 7. What to read, and what to ignore

This repo grew across a mid-project pivot, so some documents are superseded. **Authoritative, in
this order:**

| File | What it is |
|---|---|
| **this file** | the handover |
| `results/README.md` | the results narrative, current |
| `results/headtohead_real.md`, `rigor_real.md`, `sweep_real.md`, `subgroups_real.md` | auto-generated, current |
| `results/uniform_window_real.md` | the window investigation end to end |
| `docs/bootcamp/01_FACT_SHEET.md` | **every number in the project with its derivation.** Start here for any specific figure |
| `docs/bootcamp/15_FINDINGS_TO_REPORT.md` | the three errors above, in detail |
| `cgm_tsfm/` | the code is the spec |

**Treat as historical:**

- `README.md`, `PLAIN_ENGLISH_SUMMARY.md`, `PROGRESS_NOTES.md`, `docs/00_PROJECT_OVERVIEW.md`,
  `docs/02_ROADMAP.md`, `docs/04_OPEN_QUESTIONS.md` — written before / during the real-data run.
  Directionally right, numerically stale.
- `docs/bootcamp/02`–`14` — these are **my** study notes (an ML/DL course welded to this project,
  written to fix gaps the previous lead identified in me). Useful as reference, not as project
  status. Several still quote pre-uniform-window numbers and say so in a banner at the top.
- `results/archive_prebugfix_2026-07-07/`, `results/archive_variablewindow_2026-08/` — kept only
  so the earlier numbers are reproducible.

---

## 8. Run it

Environment is already built and GPU-validated on `cpsl-mds`. **Lab policy: everything —  repo,
env, HF downloads, caches, data — lives under `/data_1_8TB_ssd/brian_workspace`, never `/home`
(≤300 GB, shared).** See `docs/05_SERVER_SETUP.md`.

```bash
PY=/data_1_8TB_ssd/brian_workspace/envs/cgm/bin/python
cd /data_1_8TB_ssd/brian_workspace/cgm-tsfm

# offline smoke test, no download, no data needed
$PY -m cgm_tsfm.run_demo --encoder mock

# the headline result (Arm A + Arm B, 3 targets, grouped CV)
$PY -m cgm_tsfm.run_headtohead --encoder chronos --real --with-arm-b

# the two credibility checks
$PY -m cgm_tsfm.run_rigor --real

# the batch-invariance regression test — run this after touching encoders.py
$PY -m cgm_tsfm.test_pooling_invariance

# reproduce the old variable-length behaviour
$PY -m cgm_tsfm.run_headtohead --encoder chronos --real --variable-window
```

**The data is not in git and must not be.** It is patient data:
`data/Merged_glucose_data/Cohort1_scores_merged_with_glucose.csv` and
`Updated_Cohort2_scores_with_glucose.csv`, gitignored, copied directly to the server. Use the
`Updated_` Cohort 2 file — the original has 44/240 empty-glucose rows and loses a participant
(19 instead of 20).

---

## 9. What is actually open — the decisions you inherit

**These two were raised with the previous lead and never answered. Everything else waits on them.**

1. **Is a rigorous "low prediction accuracy + characterisation" the deliverable, or do we keep
   hunting for signal?** We have exhausted the cheap options (§3, §4). Going further means
   changing the problem, not the model: a two-part / zero-inflated likelihood for Grids, or
   bringing in insulin / meals / activity — which the previous lead gated on "TSFM exhausted on
   glucose alone first." That gate is now met.
2. **Is N=20 the sample for this phase, or is more coming toward the K01's 92?** This determines
   whether the current result is interim or a stopping point. Nothing in the modelling will move
   the needle at N=20.

Smaller, genuinely useful items:

- **Grids needs a different likelihood.** 31% exact zeros with a hard floor is not a Gaussian
  target. This is the one modelling change with a real argument behind it — though note the
  previous lead assigned mixed-effects/association work to Phil and told me to stay on prediction,
  so it may not be ours.
- **Concatenate Chronos's `loc`/`scale` onto the features** so absolute glucose level is available
  at all (§4a). Cheap, and closes a real gap.
- **Arm B hyperparameters were never swept.** The harness exists; the capacity argument in §3
  suggests shrinking the head rather than growing it.
- **Ask Phil how the 20th participant's glucose was backfilled.** He has the script; I have not
  seen it.
- **Confirm whether "compute2" in the original task list is a different machine from `cpsl-mds`.**

---

## 10. A note on the deck

`Handover_CGM_Cognition.pptx` (gitignored, sent separately) is this document with charts and
speaker notes on every slide. Slide 5 is your architecture. Slides 8–11 are the credibility
evidence. Slide 14 is §5 above.
