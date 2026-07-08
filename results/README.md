# Results — CGM → Cognition (raw-data / TSFM arm)

Narrative index stitching together the two auto-generated results files in this folder:

**Real (20 patients — the actual findings):**
- [`headtohead_real.md`](headtohead_real.md) — Arm A + Arm B on the real data, grouped CV.
- [`headtohead_real_centered.md`](headtohead_real_centered.md) — the within-subject (personal-baseline-removed) test.
- [`sweep_real.md`](sweep_real.md) — PCA sweep on the real embeddings.

**Synthetic (pipeline harness check only):**
- [`sweep_synthetic.md`](sweep_synthetic.md) — Arm A hyperparameter **sweeps** (encoder checkpoint / window / pooling / PCA / within-subject target-norm).
- [`headtohead_synthetic.md`](headtohead_synthetic.md) — the **3-way representation comparison** (Chronos+classical vs hand-crafted+classical vs Chronos+trainable-head).

Full project context is in [`../docs/00_PROJECT_OVERVIEW.md`](../docs/00_PROJECT_OVERVIEW.md); the pipeline design (and how these map to the reference repos) is in [`../docs/01_PIPELINE_DESIGN.md`](../docs/01_PIPELINE_DESIGN.md).

> ✅ **Real results (20 patients, 956 sessions) are now in — see the "Real results" section just below.** Per the 2026-07 decision the deliverable is **our own TSFM approach**; hand-crafted features appear only as a sanity check. The `_synthetic` files (Sections 1–2, further down) remain as the pipeline **harness check** — their numbers are not scientific findings (the synthetic signal is summary-stat-based).

---

## The question these results answer

The K01's Aim-1 exploratory plan compares **"raw-data learning" vs "feature engineering"** for predicting three ARC/PARC cognitive scores (**Grids, Symbols, Prices**) from the CGM window before each test. Liuyi's collaborator Mack already built the feature-engineering arm and found **no predictive power** (negative R², chance AUROC), with diagnostics pointing to the signal being **subject-baseline** rather than a generalizable glucose→cognition effect.

This folder holds the **raw-data / TSFM arm**: frozen Amazon **Chronos** embeddings of the glucose window fed to a regressor, evaluated under the **same subject-grouped nested CV** as Mack's arm so the comparison is fair — the only thing that differs is the *representation*.

### How to read the tables
- **Metric = R² vs a mean-predictor** (higher = better). **R² < 0 means worse than just predicting the mean** — i.e., no usable signal. R² is reported as `mean ± std` across the 5 outer subject-grouped folds.
- **Three representations** in the head-to-head: `Chronos (512d) + Ridge/SVR` (Arm A), `Hand-crafted (43) + Ridge/SVR` (Mack's features, ported verbatim), `Chronos + MLP head` (Arm B, trainable).
- **Two diagnostics** worth watching on real data:
  - *group-CV vs session-CV*: if group≈0 but session>0, the signal is subject-baseline, not generalizable.
  - *within-subject target-norm*: centers each score on the subject's own mean, so the model must predict *deviation from personal baseline* — a direct test of the subject-baseline hypothesis (oracle centering; a characterization, not a new-subject predictor).

---

## ✅ Real results (20 patients, 956 sessions) — the headline

*First real run, 2026-07, on `cpsl-mds` (GPU). Metric = R² vs the mean-predictor under subject-grouped CV; **0 = no better than guessing the average**, higher = better. Files: `headtohead_real.md`, `headtohead_real_centered.md`, `sweep_real.md`.*

**Head-to-head — grouped CV ("predict a NEW kid"):**

| target | Chronos + Ridge/SVR (Arm A) | Chronos + MLP (Arm B) | hand-crafted (sanity) |
|---|--:|--:|--:|
| grids   | −0.089 | −0.218 | −0.065 |
| symbols | −0.056 | −0.281 | −0.038 |
| prices  | −0.012 | −0.168 | −0.009 |

**Within-subject centered — grouped CV ("predict a kid's deviation from their OWN mean"):**

| target | Chronos + Ridge/SVR | hand-crafted |
|---|--:|--:|
| grids   | −0.018 | −0.004 |
| symbols | −0.022 | −0.004 |
| prices  | −0.002 | −0.002 |

**Conclusion (honest).** Across all three cognitive scores, **no representation beats the mean baseline** — not frozen Chronos embeddings (Arm A), not the trainable head (Arm B), not the 43 hand-crafted features (all R² ≈ 0 to −0.28). Removing each subject's personal baseline (centering) does **not** surface a within-subject effect either (all ≈ 0). This is a rigorous, leakage-free **null result** — *no generalizable, or even within-subject, short-term glucose→cognition signal is detectable in these 20 patients* — corroborating the earlier feature-based attempt with a stronger method. It is an honest scientific answer, not a pipeline failure.

**Supporting diagnostics.**
- *Group vs session* (`run_demo --real`): session-CV is only marginally less negative than group-CV, and even the mean-baseline is slightly negative under group CV — both signatures of **between-subject variance dominating**.
- *Linear blow-up*: plain `LinearRegression` on raw 512-d embeddings → R² ≈ −3 (both schemes) → regularization / dimensionality reduction is mandatory (as designed).
- *PCA sweep* (`sweep_real.md`): reducing 512→8–32 improves mean R² only marginally (−0.052 → −0.045) and stays negative — it cleans up the math, it doesn't create signal.

> ⚠️ **N = 20 subjects is small.** This is an honest characterization on the data we have, not proof that no effect exists anywhere. A different pre-test window, more subjects, or subgroup analyses (e.g. hypoglycemia windows) could change it — see `../docs/02_ROADMAP.md` and `../docs/04_OPEN_QUESTIONS.md`.

---

## 1. Head-to-head (synthetic harness check) — do learned representations beat hand-crafted features?

Full table: [`headtohead_synthetic.md`](headtohead_synthetic.md). Distilled (best model per representation, R²):

| target | Chronos + Ridge/SVR | Hand-crafted + Ridge/SVR | Chronos + MLP head |
|---|--:|--:|--:|
| grids | −0.728 | −0.779 | −1.090 |
| symbols | −0.225 | −0.363 | −0.589 |
| prices | −0.185 | −0.200 | −0.410 |

**Reading (synthetic):** all three are below baseline → *no generalizable signal from any representation*, which is the expected outcome given how the synthetic data is built. Note the ordering — Chronos+classical ≥ hand-crafted+classical ≥ Chronos+MLP-head — is an artifact here (the synthetic signal favors neither; the MLP head simply overfits more on a signal-less high-dim input). **On real data this table is the headline result**: whichever representation has the least-negative / most-positive R² per target is the answer to "raw-data vs feature-engineering."

---

## 2. Sweeps — which encoder / window / pooling / reduction to use

Full tables: [`sweep_synthetic.md`](sweep_synthetic.md). One-factor-at-a-time; each cell is the best-model mean-R² (3-target average in parentheses):

- **Checkpoint** (window=full, mean pooling): Bolt tiny/mini/small/base are **flat** (mean R² ≈ −0.38 to −0.39); `chronos-t5-small` is clearly **worse** (−0.458). → default to **Chronos-Bolt**; bigger isn't better here, and T5 is the one to avoid.
- **Window** (bolt-small): `full` (−0.379) ≈ best; 2–3 h windows slightly worse. *Synthetic artifact* — the synthetic labels are generated from the full trace, so clipping drops signal by construction; don't over-read. On real data this axis tells you the physiologically useful lookback.
- **Pooling** (bolt-small, full): `mean` (−0.379) ≥ `last` (−0.389) — marginal, consistent with the ICML recipe. Use **mean**.
- **PCA** (bolt-small, full, mean): reducing 512-d → **~16–32 components improves** mean R² (−0.379 → **−0.347**) and removes the `LinAlgWarning` ill-conditioning; degrades again by 128. → apply **PCA≈32** before Ridge/SVR on the real embeddings (re-pick the exact k with `--kind pca --real`).
- **Target-norm** (bolt-small, full, mean): raw −0.379 → **within-subject center −0.029** → zscore −0.014. Centering strips the (unpredictable) between-subject baseline; the residual within-subject R² sits at ~0 — the same place Mack landed. This is the diagnostic that separates "no signal at all" from "signal buried under baseline."

---

## How these were generated

Interpreter: `~/Desktop/mod-actigraphy-advanced/.venv/bin/python` (has chronos/torch/sklearn). From the project root:

```bash
python -m cgm_tsfm.run_sweep      --kind all                       # → results/sweep_synthetic.md
python -m cgm_tsfm.run_headtohead --encoder chronos --with-arm-b   # → results/headtohead_synthetic.md
```

Both write markdown automatically (metadata header, per-target tables, verdicts, synthetic caveat). See [`../README.md`](../README.md) for all commands.

## Regenerating on real data

The moment `data/Merged_glucose_data/Cohort{1,2}_...glucose.csv` are in place:

```bash
python -m cgm_tsfm.run_sweep      --kind all --real                                    # → results/sweep_real.md
python -m cgm_tsfm.run_sweep      --kind pca --real                                    # pick best PCA k for real embeddings
python -m cgm_tsfm.run_headtohead --encoder chronos --real --with-arm-b --pca 32       # → results/headtohead_real.md
python -m cgm_tsfm.run_headtohead --encoder chronos --real --pca 32 --target-norm center --out results/headtohead_real_withinsubj.md
```

**What to conclude on real data:**
- If the head-to-head shows Chronos R² **clearly above** hand-crafted (and above baseline) → learned representations capture temporal structure the 43 features miss. *Raw-data learning wins.*
- If the within-subject-centered table turns **positive** while raw scores stay ~0 → there *is* a within-subject glucose→cognition signal that the subject baseline was hiding. *A genuine effect, worth reporting.*
- If everything stays ~0 even after centering and PCA → the effect is genuinely weak, corroborating Mack's arm with a stronger method. *Still a publishable, honest answer to the K01 question.*

## Caveats (carry into any writeup)
- **Synthetic ≠ real** — current numbers are a harness check only.
- **Oracle within-subject centering** — uses a subject's own sessions; answers "is there a within-subject signal?", not "can we predict a new subject."
- **Subject-baseline dominance** — expect between-subject variance to dwarf within-subject; grouped CV + centering are how we see past it.
- **High-dim embeddings need reduction** — plain linear/MLP on raw 512-d overfits badly; PCA (or strong regularization) is required.
