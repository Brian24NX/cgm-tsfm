# Diabetes / T1D Cognition Project

Brian Zhou's research project, supervised by **the advisor**. This is a separate project from `mod-actigraphy-advanced` (a labmate Ben's codebase, kept at `~/Desktop/mod-actigraphy-advanced`) — Ben's repo is being studied only as an architectural reference, not developed here.

> ## 🔗 New session / on the lab server? START HERE
> If you're a fresh Claude Code session (e.g. running on the lab server, or after a context reset), you will **not** have the prior chat history or Brian's laptop-local `~/.claude` memories — **this repo is the source of truth.** Get oriented fast:
> 1. **`PLAIN_ENGLISH_SUMMARY.md`** — plain-language overview of what's built and why.
> 2. **`docs/00_PROJECT_OVERVIEW.md`** + **`docs/02_ROADMAP.md`** — design + status/what's next.
> 3. The `cgm_tsfm/` package is the runnable pipeline (Arm A + Arm B); `results/README.md` explains the outputs.
>
> **Current state (2026-07):** the full pipeline is built and validated on *synthetic* data; it was developed on Brian's laptop and pushed to the private GitHub repo **`Brian24NX/cgm-tsfm`**. **We are mid-migration to the `cpsl-mds` lab server** so training runs on GPU. **Immediate task: follow `docs/05_SERVER_SETUP.md`.**
> **⚠️ Server storage (the advisor's policy):** on `cpsl-mds`, keep the repo, conda env, model downloads, data, and caches under **`/data_1_8TB_ssd/brian_workspace`** (the ~1 TB SSD) — NOT `/home` (≤300 GB, shared). Use a conda **prefix** env and set `HF_HOME`/pip/conda caches to the SSD (details in docs/05).
> Then verify with the mock + synthetic runs, and do real runs once the data CSVs are placed in `data/Merged_glucose_data/` (copied **directly to the server**, never via git — it's patient data). The one real blocker for scientific results is getting the data file from the advisor.

## What this project is

Studies how short-term blood glucose fluctuations affect real-time cognitive performance (working memory, processing speed) in youth with Type 1 Diabetes (T1D), using continuous glucose monitors (CGM) and a mobile cognitive-testing app, analyzed with time-series/ML methods. Full study design is in `Ray K01.pdf` (the driving NIH K01 grant proposal): Aim 1 tests whether glucose fluctuations predict cognitive performance in T1D youth (N=92, ages 9-16, Dexcom G6 CGM + "PARC" app cognitive tests 5x/day for 10 days); Aim 2 compares T1D youth to non-T1D controls during stable glucose.

## People

- **The advisor** — supervises Brian directly, assigns tasks (see the task-assignment screenshot), expects an open questions log for transparency rather than assumptions.
- **Dr. Ray** (likely) — K01 grant PI.
- **Dr. Hassenstab** — co-mentor, built the PARC cognitive-testing app.

## Files in this folder

- `Ray K01.pdf` — the grant proposal (study design, aims, methods).
- `7669_Exploring_Representations.pdf` — TSFM interpretability paper (MOMENT/Chronos/MOIRAI internals; general background, not CGM-specific).
- `2605.22759v2.pdf` — Google's "SensorFM" wearable foundation model paper (Fitbit/Pixel Watch; general background, not CGM-specific).
- the advisor's task-assignment screenshot (gitignored) — the assigned task list (task 1, compute2 env setup, is currently deferred).
- The advisor's written answers to the initial questions (gitignored .docx) — drove the 2026-07-03 pivot below.
- `PROGRESS_NOTES.md` — working notes (pre-answer framing partly superseded; carries a banner pointing to `docs/`). Ben's-code analysis still valid.
- `README.md` — top-level entry point (project state + how to run).
- `docs/` — authoritative writeups: `00_PROJECT_OVERVIEW.md` (the pivot), `01_PIPELINE_DESIGN.md`, `02_ROADMAP.md`, `04_OPEN_QUESTIONS.md`.
- `cgm_tsfm/` — the runnable pipeline package (Arm A frozen-embeddings regression; `ben_adapter/` = Arm B single-channel regression adapted from Ben). Validated end-to-end on synthetic data incl. a real Chronos-Bolt run.
- `requirements.txt` — deps (also satisfied by `mod-actigraphy-advanced/.venv`).

## Current status (pivoted 2026-07-03 after the advisor's answers)

The advisor's written answers reframed the project. Corrected scope: **1 channel (CGM only), single-target, three separate models (Grids / Symbols / Prices), raw-data/TSFM (Chronos) arm only.** The "two CSVs" in the task list were *Ben's* watch data (a different project) — not needed; Task 3 = understand the TSFM input format, done against the ICML `representations-in-tsfms` repo. A full runnable pipeline now exists in `cgm_tsfm/` (both arms), validated on synthetic data with a real Chronos-Bolt run. **The only blocker for real results is access to the merged CSVs** (`data/Merged_glucose_data/Cohort{1,2}_...glucose.csv`), which live in the advisor's `diabetes-fitbit` repo (gitignored). Note: Mack's hand-crafted-feature arm there found **no predictive signal** (negative R²), so the TSFM arm is framed as a fair comparison, not a promised accuracy win. Full narrative in `docs/00_PROJECT_OVERVIEW.md`; open decisions in `docs/04_OPEN_QUESTIONS.md`.

## Related reference codebases (studied; not developed here)

- `~/Desktop/mod-actigraphy-advanced` — **Ben's** codebase applying TSFMs (Chronos-Bolt, TimesFM) to actigraphy for preterm-birth prediction. The **architecture template** for Arm B. Its `.venv` also has every dependency `cgm_tsfm/` needs (chronos 2.3.0, torch, lightning, sklearn).
- `diabetes-fitbit` — **the advisor's** real study repo. `analysis/glucose_cognitive_ml/` defines the real data schema, the three targets, and the grouped-CV protocol, and holds Mack's feature-engineering arm + the `glucose_cognitive_next_steps.ipynb` that documents its null results. The real data (`data/Merged_glucose_data/`) is gitignored.
- `github.com/moment-timeseries-foundation-model/representations-in-tsfms` — the **ICML'25** paper repo the advisor pointed to for understanding the Chronos input format (Task 3). Its `HookedChronos` + SVM experiment is the template converted to regression in `cgm_tsfm/`.
