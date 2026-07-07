# Meeting Brief — Updates for Liuyi

*One-pager for the supervisor meeting. Prepared by Brian. Backing detail in the other `docs/` files.*

> **🆕 Post-meeting update (2026-07).** The decisions in §4 are now settled: **focus on our own TSFM approach — no head-to-head vs Mack** (item 4); **Task 1 / lab server is done** on `cpsl-mds` (item 6); Chronos version decided (`chronos-bolt-small`, item 5). Data is on the way after labmate **Phil**'s correlation analysis of the 20 patients. Full current status: [`PLAIN_ENGLISH_SUMMARY.md`](../PLAIN_ENGLISH_SUMMARY.md).

---

## 1. I understood the pivot from your answers

- The "two CSVs" = **Ben's watch data (sleep + light)**, a different project — not our data. Ben's code is a *template*, not built for us. ✔ Won't chase that data.
- **Task 3 = understand the TSFM input format**, done against the **ICML'25 `representations-in-tsfms` repo** (Chronos-specific, ships data, you reproduced it). Data availability is not a blocker. ✔
- Scope locked: **1 channel (CGM only), single-target, three separate models (Grids / Symbols / Prices), raw-data/TSFM arm only** (Mack owns feature-engineering). ✔

## 2. What I did

- **Studied all three codebases** (Ben's `mod-actigraphy-advanced`, the ICML `representations-in-tsfms`, and your `diabetes-fitbit`) and extracted the exact Chronos input recipe:
  `raw glucose window → BaseChronosPipeline.embed() → (B, patches+1, d_model) → mean-pool over tokens → (B, d_model)`. No normalization needed (Chronos mean-scales internally); NaN-tolerant (good for gappy CGM).
- **Built a runnable pipeline** (`cgm_tsfm/`) with the **real data schema** (`subid`, `session`, `Glucose_Before_Test`, the 3 scores), a synthetic generator so it runs today, and an embedding cache. It's **drop-in for the real CSVs**.
- **Arm A** (frozen Chronos embeddings + Ridge/SVR, nested **GroupKFold by `subid`**, RMSE/MAE/R²) — mirrors your `diabetes-fitbit` protocol exactly, so results are directly comparable to Mack's.
- **Arm B (Task 4)** — adapted Ben's `FoundationModelClassifier`→ single-channel `FoundationModelRegressor` and his `FinetuneLightningModule`→ regression `CGMRegressionModule` (replaced the `/365` gestational-age hack with a general per-target z-score; replaced the fragile task-name string check). Smoke-tested: the head learns.
- **Ran it end-to-end with real `chronos-bolt-small`**: 938 windows → 512-d embeddings → grouped nested CV on all 3 targets. Works.

## 3. What I found (calibrating expectations honestly)

- I saw in your `next_steps` notebook that **Mack's feature-engineering arm has no predictive power** (negative R², chance AUROC), and that **session-CV beats leave-one-patient-out → the signal is mostly subject-baseline**, not generalizable glucose→cognition.
- My pipeline **reproduces that exact diagnostic** on synthetic data (group-CV R²≈0, session-CV R²>0), so we can trust it to tell "no signal" apart from "a bug."
- Practical lesson from the real Chronos run: **plain linear regression on 512-d embeddings overfits badly** (R²≈−4); Ridge/SVR + grouped CV (both built in) are essential.
- ⚠️ My synthetic numbers are **not** evidence about Chronos-vs-features — the synthetic signal was defined via summary stats, which favors hand-crafted features. **The real comparison needs the real data.**

## 4. What I need from you (decisions)

1. **The data** — *on the way.* After labmate **Phil** finishes the correlation analysis of the 20 patients, Liuyi sends the merged CSVs; then I run the real results immediately. (Just a timing check-in.)
2. **The window / raw stream.** `Glucose_Before_Test` is pre-clipped upstream — is the **raw continuous CGM stream** available, and what **lookback** is baked in (the K01 hypothesis targets ~2 h pre-test)? *(Non-blocking — I default to the full clipped window + a 2.5 h sensitivity check.)*
3. **Prices + score meaning.** Confirm `prices_cognitive_score` is inverted (I negate it), and what do Grids/Symbols/Prices actually measure (units)? *(Non-blocking.)*
4. ✅ **Success criteria — RESOLVED:** focus on our own TSFM approach; **no head-to-head vs Mack.**
5. ✅ **Encoder — DECIDED (by me):** `chronos-bolt-small` (frozen) as primary, + `bolt-base` as a sensitivity check.
6. ✅ **Task 1 (server) — DONE:** built + GPU-validated on `cpsl-mds` (`docs/05_SERVER_SETUP.md`); confirm only whether "compute2" is a different machine.

## 5. Immediate next step (the moment I get the data)

`python -m cgm_tsfm.run_headtohead --encoder chronos --real --with-arm-b` → the first **real TSFM results** (Arm A + Arm B), per target, under subject-grouped nested CV. *(Per the 2026-07 decision we lead with our own approach; the runner can still print Mack's 43 features alongside as an optional sanity-check, but that's no longer the deliverable.)*
