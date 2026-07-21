# Results — CGM → Cognition (raw-data / TSFM arm)

Narrative index stitching together the two auto-generated results files in this folder:

**Real (20 patients — the actual findings):**
- [`headtohead_real.md`](headtohead_real.md) — Arm A + Arm B on the real data, grouped CV.
- [`headtohead_real_centered.md`](headtohead_real_centered.md) — the within-subject (personal-baseline-removed) test.
- [`sweep_real.md`](sweep_real.md) — full sweep: encoder size / window length / pooling / PCA / within-subject target-norm.
- [`subgroups_real.md`](subgroups_real.md) — signal search by glucose regime (hypo / hyper / in-range).

**Synthetic (pipeline harness check only):**
- [`sweep_synthetic.md`](sweep_synthetic.md) — Arm A hyperparameter **sweeps** (encoder checkpoint / window / pooling / PCA / within-subject target-norm).
- [`headtohead_synthetic.md`](headtohead_synthetic.md) — the **3-way representation comparison** (Chronos+classical vs hand-crafted+classical vs Chronos+trainable-head).

Full project context is in [`../docs/00_PROJECT_OVERVIEW.md`](../docs/00_PROJECT_OVERVIEW.md); the pipeline design (and how these map to the reference repos) is in [`../docs/01_PIPELINE_DESIGN.md`](../docs/01_PIPELINE_DESIGN.md).

> ✅ **Real results (20 patients, 956 sessions) are now in — see the "Real results" section just below.** Per the 2026-07 decision the deliverable is **our own TSFM approach**; hand-crafted features appear only as a sanity check. The `_synthetic` files (Sections 1–2, further down) remain as the pipeline **harness check** — their numbers are not scientific findings (the synthetic signal is summary-stat-based).

---

## The question these results answer

The K01's Aim-1 exploratory plan compares **"raw-data learning" vs "feature engineering"** for predicting three ARC/PARC cognitive scores (**Grids, Symbols, Prices**) from the CGM window before each test. The advisor's collaborator Mack already built the feature-engineering arm and found **no predictive power** (negative R², chance AUROC), with diagnostics pointing to the signal being **subject-baseline** rather than a generalizable glucose→cognition effect.

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

**Conclusion (honest, plain terms).** Across all three cognitive scores, the TSFM approach **does not predict the score more accurately than a trivial "guess the average" baseline** (R² ≈ 0 to −0.28) — and neither does the trainable head (Arm B) or the 43 hand-crafted features. Removing each subject's personal baseline (centering) doesn't surface a within-subject effect either. In short: **low prediction accuracy, and it holds up robustly.** This corroborates the earlier feature-based attempt with a stronger method — an honest scientific answer, not a pipeline failure.

**We searched hard for signal (per the advisor's steer) — it's absent everywhere:**
- *Glucose-regime subgroups* ([`subgroups_real.md`](subgroups_real.md)): restricting to **hypoglycemia** (min<70; **19 subjects**) or **hyperglycemia** (max>250; **20 subjects**) sessions — where an effect is most physiologically plausible — still gives R² ≈ 0 to −0.19. These subgroups are **well-powered**, so "no signal in the excursion regimes" is a real finding, not a tiny-N artifact.
- *Encoder size* ([`sweep_real.md`](sweep_real.md)): bolt tiny→base and t5-small are all flat (mean R² −0.05 to −0.06; **`base` is *worse***) → a bigger model won't rescue it.
- *Window length*: capping to the most-recent 2 h / 2.5 h / 3 h is **worse** than the full window (itself still ≈0) → shortening the lookback doesn't reveal signal.
- *Pooling / PCA*: mean ≥ last; PCA 8–32 improves mean R² only marginally (−0.052 → −0.045) and stays negative.
- *Group-vs-session CV* + *plain-Linear blow-up (R²≈−3)*: confirm between-subject variance dominates and that regularization is mandatory (both as designed).

**Score direction:** all three scores are "lower = better" (error / response-time-type). We keep them in raw orientation — this does **not** change any R²/RMSE above (invariant to the target's sign).

> ⚠️ **N = 20 (likely all we get).** An honest characterization on the available data, not proof that no effect exists anywhere. **The one lever not yet pulled:** a *principled uniform* pre-test window (e.g. exactly 2 h) cut from the **raw continuous CGM stream** — capping the pre-clipped arrays (above) can't emulate that. Obtaining the raw stream is the open question for Phil. See `../docs/02_ROADMAP.md` and `../docs/04_OPEN_QUESTIONS.md`.

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
