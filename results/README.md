# Results — CGM → Cognition (raw-data / TSFM arm)

Narrative index stitching together the two auto-generated results files in this folder:

- [`sweep_synthetic.md`](sweep_synthetic.md) — Arm A hyperparameter **sweeps** (encoder checkpoint / window / pooling / PCA / within-subject target-norm).
- [`headtohead_synthetic.md`](headtohead_synthetic.md) — the **3-way representation comparison** (Chronos+classical vs hand-crafted+classical vs Chronos+trainable-head).

Full project context is in [`../docs/00_PROJECT_OVERVIEW.md`](../docs/00_PROJECT_OVERVIEW.md); the pipeline design (and how these map to the reference repos) is in [`../docs/01_PIPELINE_DESIGN.md`](../docs/01_PIPELINE_DESIGN.md).

> ⚠️ **Everything below is SYNTHETIC.** These runs validate the pipeline end-to-end and exercise every knob; the numbers are **not** scientific findings (the synthetic signal is summary-stat-based and baseline-dominated by construction). The `_synthetic` files are placeholders — the meaningful `_real` versions are one command each (see [§Regenerating](#regenerating-on-real-data)). When real results land, update this narrative accordingly.

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

## 1. Head-to-head — do learned representations beat hand-crafted features?

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
