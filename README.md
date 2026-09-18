# CGM → Cognition via Time-Series Foundation Models

> ## 📌 2026-09 — project handover. **New to this project? Read [`docs/HANDOVER.md`](docs/HANDOVER.md) first.**
>
> The real data is **in** (20 participants, 956 sessions) and the real runs are **done**. Headline:
> nothing predicts the cognitive scores better than a trivial mean-predictor, across 13 separate
> attempts — full numbers and the evidence that this is the data rather than the pipeline are in
> [`docs/HANDOVER.md`](docs/HANDOVER.md) and [`results/README.md`](results/README.md).
> **Wording below this banner that says the data has not arrived, or that results are synthetic
> placeholders, is out of date.**

> 🟢 **New here / want the plain-English version?** Read **[`PLAIN_ENGLISH_SUMMARY.md`](PLAIN_ENGLISH_SUMMARY.md)** — a no-jargon catch-up on what we built and what to tell the advisor. Start there; come back here for the technical details/commands.

Predict a youth-with-T1D's cognitive-test score (Grids / Symbols / Prices) from the
continuous-glucose-monitor (CGM) window recorded before the test, using a **frozen
Amazon Chronos** foundation model as the feature encoder. This is the **raw-data /
TSFM arm** of Brian Zhou's research project (supervised by the advisor).

**Start here:** `docs/00_PROJECT_OVERVIEW.md` (what & why, and how the advisor's answers
reshaped the plan) →
`docs/01_PIPELINE_DESIGN.md` (technical design) → `docs/02_ROADMAP.md` (plan) →
`docs/04_OPEN_QUESTIONS.md` (decisions needed from the advisor).

**Results:** `results/README.md` stitches the auto-generated sweep + head-to-head
tables into one narrative. The `_real` tables are the actual findings (20 participants,
916 sessions, uniform 2 h window); the `_synthetic` ones are a pipeline harness check only.

## Layout

```
cgm_tsfm/
  config.py        paths, the 3 targets, windowing, encoder & CV settings
  data.py          real-schema loader (mirrors diabetes-fitbit) + synthetic generator
  encoders.py      frozen Chronos embedding extractor (Bolt/T5) + mock encoder + cache
  handcrafted.py   Mack's 43 hand-crafted glycemic features (faithful port; optional sanity-check baseline)
  regression.py    Arm A: nested GroupKFold regression (Ridge/SVR/…) + metrics
  run_demo.py      end-to-end Arm A demo (synthetic or real)
  run_headtohead.py  ONE COMMAND: Chronos vs the 43 features (+ optional Arm B), grouped CV; writes results/headtohead_*.md
  run_sweep.py     sweep checkpoint / window / pooling / PCA / within-subject target-norm (Arm A)
  ben_adapter/     Arm B: single-channel regressor + Lightning module (adapted from Ben)
    model.py, lightning_module.py, datamodule.py, train.py (grouped-CV training),
    smoke_test.py, README.md (change-mapping vs Ben)
docs/              overview, design, roadmap, meeting brief, open questions, server setup (05), Chronos dimensions (06)
requirements.txt
```

## Setup

On the lab server (`cpsl-mds`) the environment is **already set up and GPU-validated** — see **`docs/05_SERVER_SETUP.md`** (activate the prefix env and go; **Task 1 is done**). To build a fresh environment elsewhere (laptop **or** a new machine):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # GPU server: install a CUDA build of torch first — see docs/05
```

## Quick start

```bash
PY=python                              # venv active; add `--device cuda` on a GPU server

# 1) Fully offline smoke test (no download): synthetic data + mock encoder
$PY -m cgm_tsfm.run_demo --encoder mock

# 2) Real Chronos embeddings on synthetic data (downloads chronos-bolt-small once)
$PY -m cgm_tsfm.run_demo --encoder chronos --model amazon/chronos-bolt-small --device cpu

# 3) HEAD-TO-HEAD in one command: Chronos embeddings vs Mack's 43 features
#    (both through identical subject-grouped nested CV)
$PY -m cgm_tsfm.run_headtohead --encoder chronos

# 3b) Three-way: add Arm B (Chronos + trainable MLP head, grouped-CV training)
$PY -m cgm_tsfm.run_headtohead --encoder chronos --with-arm-b
#     …or run Arm B on its own:
$PY -m cgm_tsfm.ben_adapter.train --encoder chronos

# 3c) Sweep checkpoint / window / pooling / PCA (Arm A). --kind pca isolates the PCA axis.
#     Dumps every table to results/sweep_<synthetic|real>.md for the writeup (--out to override).
$PY -m cgm_tsfm.run_sweep --kind all
#     Apply the best PCA size in the head-to-head (fit per-fold; Arm A contenders):
$PY -m cgm_tsfm.run_headtohead --encoder chronos --pca 32

# 3d) Test the subject-baseline hypothesis: predict deviation from each subject's
#     own mean score (within-subject centering). Also a sweep axis: --kind targetnorm.
$PY -m cgm_tsfm.run_headtohead --encoder chronos --target-norm center

# 4) Arm B forward/backward smoke test (mock, no download)
$PY -m cgm_tsfm.ben_adapter.smoke_test

# 5) THE REAL RUN — once the merged CSVs are in data/Merged_glucose_data/
$PY -m cgm_tsfm.run_headtohead --encoder chronos --real --with-arm-b
```

`run_demo` prints, per target, RMSE/MAE/R² for each model vs a mean-predictor
baseline, under both subject-grouped CV and session-level CV (the subject-baseline
diagnostic). `run_headtohead` runs our TSFM approach (Arm A, +Arm B with `--with-arm-b`)
under grouped CV; it can also show Mack's 43 hand-crafted features alongside as an
optional sanity-check. **Per the 2026-07 decision, the deliverable is our own TSFM
approach — not a head-to-head vs Mack** (see `PLAIN_ENGLISH_SUMMARY.md`).

## Status

Pipeline built and validated end-to-end on synthetic data (both arms), including a
real `chronos-bolt-small` run **on GPU**. **Task 1 done** — environment built + GPU-validated
on `cpsl-mds` (`docs/05_SERVER_SETUP.md`). **Blocked only on the real CSVs** (coming after
labmate Phil's correlation analysis) to produce the first real TSFM results. See `docs/`.

## The three reference codebases (studied, not vendored here)

- `~/Desktop/mod-actigraphy-advanced` — Ben's actigraphy TSFM code (architecture template for Arm B).
- `representations-in-tsfms` (ICML'25) — the Chronos input-format reference (Task 3).
- `diabetes-fitbit` (the advisor) — the real study: data schema, the 3 targets, the grouped-CV protocol, Mack's feature-engineering arm.
