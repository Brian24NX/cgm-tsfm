# Results — CGM → Cognition (raw-data / TSFM arm)

Narrative index stitching together the two auto-generated results files in this folder:

**Real (20 participants — the actual findings):**
- [`headtohead_real.md`](headtohead_real.md) — Arm A + Arm B on the real data, grouped CV.
- [`headtohead_real_centered.md`](headtohead_real_centered.md) — the within-subject (personal-baseline-removed) test.
- [`sweep_real.md`](sweep_real.md) — full sweep: encoder size / window length / pooling / PCA / within-subject target-norm.
- [`subgroups_real.md`](subgroups_real.md) — signal search by glucose regime (hypo / hyper / in-range).
- [`rigor_real.md`](rigor_real.md) — the shuffle test (do scrambled scores do just as well?) + the known-answer check (can the pipeline predict a property of the glucose itself?).
- [`incremental_value_real.md`](incremental_value_real.md) — **2026-09-18:** giving the model the child's own earlier scores produces the project's first positive R² (Symbols +0.413). Also shows, with a per-fold residual test, that glucose contributes −0.01 of it.

**Synthetic (pipeline harness check only):**
- [`sweep_synthetic.md`](sweep_synthetic.md) — Arm A hyperparameter **sweeps** (encoder checkpoint / window / pooling / PCA / within-subject target-norm).
- [`headtohead_synthetic.md`](headtohead_synthetic.md) — the **3-way representation comparison** (Chronos+classical vs hand-crafted+classical vs Chronos+trainable-head).

Full project context is in [`../docs/00_PROJECT_OVERVIEW.md`](../docs/00_PROJECT_OVERVIEW.md); the pipeline design (and how these map to the reference repos) is in [`../docs/01_PIPELINE_DESIGN.md`](../docs/01_PIPELINE_DESIGN.md).

> ✅ **Real results (20 participants, 916 sessions) are now in — see the "Real results" section just below.** Per the 2026-07 decision the deliverable is **our own TSFM approach**; hand-crafted features appear only as a sanity check. The `_synthetic` files (Sections 1–2, further down) remain as the pipeline **harness check** — their numbers are not scientific findings (the synthetic signal is summary-stat-based).

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

## ✅ Real results (20 participants, 916 sessions, uniform 2 h) — the headline

> ### 📏 Regenerated 2026-09 — every session now uses the SAME 2 hours of glucose
>
> The pipeline default changed. Previously each session was given whatever glucose happened to sit between it and the child's previous test — anywhere from 3 readings (15 min) to 288 (24 h). That meant the *length* of the input varied, and length itself carried time-of-day information (a long window means a morning test, r = −0.506).
>
> Now every session gets exactly the **24 readings (2.0 h) immediately before the test**, and sessions with less than that are dropped: **916 of 956 sessions kept (95.8%), all 20 participants**. Only 1 session in 956 has an internal gap, so this is uniform in real time, not just in count.
>
> **2 h was chosen on physiological grounds and committed to in advance** — the K01 hypothesis concerns roughly the 2 h before a test. It was *not* picked for scoring best; it did not. Windows from 15 min to 3 h were all measured and the differences between them are far smaller than the fold-to-fold spread ([`uniform_window_real.md`](uniform_window_real.md)).
>
> **The conclusion did not change.** Pre-change files: [`archive_variablewindow_2026-08/`](archive_variablewindow_2026-08/). Pass `--variable-window` to any runner to reproduce them.
>
> ### 🔧 Earlier: regenerated 2026-08-01 after a pooling bug fix
>
> A batch-invariance bug was found in `cgm_tsfm/encoders.py`: pooling averaged over *padding* positions, so a session's embedding depended on which other sessions shared its batch. 861 of 956 sessions (90%) had materially wrong features (median cosine similarity 0.37 against correctly-pooled vectors). The same bug was present in Arm B's `ChronosTorchEmbedder`.
>
> **Both are fixed** (windows are bucketed by token count; verified bit-identical to encoding each window alone) and **every table below was regenerated**. The embedding cache key is now versioned (`v2`) so stale embeddings cannot be silently reused. Pre-fix outputs: [`archive_prebugfix_2026-07-07/`](archive_prebugfix_2026-07-07/).
>
> **The conclusion is unchanged — everything is still at or below zero.** Full write-up: [`../docs/bootcamp/15_FINDINGS_TO_REPORT.md`](../docs/bootcamp/15_FINDINGS_TO_REPORT.md).

*Metric = R² vs the mean-predictor under subject-grouped CV; **0 = no better than guessing the average**, higher = better. Files: `headtohead_real.md`, `headtohead_real_centered.md`, `sweep_real.md`.*

**Head-to-head — grouped CV ("predict a NEW kid"):**

| target | sessions | Chronos + Ridge/SVR (Arm A) | Chronos + MLP (Arm B) | hand-crafted (sanity) |
|---|--:|--:|--:|--:|
| grids   | 911 | −0.135 | −0.329 | −0.088 |
| symbols | 882 | −0.290 | −0.581 | −0.257 |
| prices  | 897 | −0.015 | −0.170 | −0.012 |

⚠️ **`symbols` is unstable — the fold-to-fold spread is ±0.266, larger than the number itself.** On 882 sessions from 20 children that column should be quoted with much less confidence than the other two.

**Within-subject centered — grouped CV ("predict a kid's deviation from their OWN mean"):**

| target | Chronos + Ridge/SVR | hand-crafted |
|---|--:|--:|
| grids   | −0.042 | −0.011 |
| symbols | −0.045 | −0.001 |
| prices  | −0.001 | −0.001 |

**⚠️ Important framing: 0 is not the neutral point here.** Measured under the same folds, the mean-predictor *itself* scores **−0.071 / −0.253 / −0.011**, because `r2_score` compares each test fold against *that fold's own* mean while the model was trained on the training fold's mean. With only 4 participants held out those differ. The honest measure is the **gap**:

| target | our model | guessing | gap |
|---|--:|--:|--:|
| grids | −0.135 | −0.071 | **−0.064** |
| symbols | −0.290 | −0.253 | **−0.036** |
| prices | −0.015 | −0.011 | **−0.004** |

So the model still sits at, or just below, guessing. Roughly half of the apparent drop from the previous numbers is the baseline moving, not the model getting worse.

**Why Arm B got worse after the fix.** Verified across 3 seeds, so it is not run-to-run noise (grids −0.218 ± 0.018 before vs −0.323 ± 0.011 after; the gap is ~6× the seed spread). The buggy pooling blurred sessions toward one another, which accidentally regularized a 132,609-parameter head trained on ~612 rows. Correct, sharper features give it more genuine detail to overfit. The capacity problem was always there; the fix made it visible.

**Conclusion (honest, plain terms).** Across all three cognitive scores, the TSFM approach **does not predict the score more accurately than a trivial "guess the average" baseline** (R² ≈ 0 to −0.41) — and neither does the trainable head (Arm B) or the 43 hand-crafted features. Removing each participant's personal baseline (centering) doesn't surface a within-participant effect either. In short: **low prediction accuracy, and it holds up robustly.** This corroborates the earlier feature-based attempt with a stronger method — an honest scientific answer, not a pipeline failure.

**We searched hard for a relationship (per the advisor's steer) — it's absent everywhere:**
- *Glucose-regime subgroups* ([`subgroups_real.md`](subgroups_real.md)): restricting to sessions containing **hypoglycemia** (any reading <70 — **111 sessions** from **19 participants**) or **hyperglycemia** (any reading >250 — **314 sessions** from **19 participants**) still gives R² of −0.03 to −0.68. ⚠️ **These subgroups shrank a lot under the 2 h window** (from 201 and 449), because 2 h of glucose contains far fewer excursions than 24 h. That is a real cost of the uniform window: fewer of the cases where an effect is most plausible. Read these rows as noisy.
- *Encoder size* ([`sweep_real.md`](sweep_real.md)): bolt tiny→base and t5-small are all flat (mean R² −0.134 to −0.150), and **`base` is still the *worst*** → a bigger model won't rescue it. That is a 24× parameter range (8.7M → 205M) spanning 0.016 in mean R².
- *Window length*: measured properly, 15 min through 3 h, uniform ([`uniform_window_real.md`](uniform_window_real.md)). Every window gives the same answer, and the differences between them are ~80× smaller than the fold-to-fold spread, so the data cannot tell them apart. Short windows *look* better only because the model runs out of input and falls back on predicting the average.
- *Pooling*: mean (−0.147) and the summary token (−0.139) are close — as expected once the padding bug was fixed.
- *PCA*: reducing the 512 numbers still leaves everything negative.
- *Plain-Linear blow-up (R²≈−3)*: confirms regularization is mandatory at 512 features on ~764 training rows (as designed).

**Score direction:** all three scores are "lower = better" (error / response-time-type). We keep them in raw orientation — this does **not** change any R²/RMSE above (invariant to the target's sign).

> ⚠️ **N = 20 (likely all we get).** An honest characterization on the available data, not proof that no effect exists anywhere. **The one thing we have not been able to try:** giving every session the *same span* of glucose (e.g. exactly 2 h) cut from the uncut recording — capping the pre-clipped arrays (above) cannot emulate that, because the short sessions do not contain 2 h. Whether re-cutting is possible depends on Phil's merge script. See `../docs/02_ROADMAP.md` and `../docs/04_OPEN_QUESTIONS.md`.

---

## Rigor checks — is the low accuracy *real*? ([`rigor_real.md`](rigor_real.md))

Two checks (same embeddings, same grouped CV) to make the "low prediction accuracy" conclusion convincing — the concrete evidence the advisor asked for.

**1. The shuffle test (200 rounds).** Randomly reassign the scores to the wrong sessions so any real relationship is destroyed, then re-run the whole evaluation. If the real result stands out from the scrambled pile, something is there.

| score | real R² | scrambled: mean ± sd | scrambled 95th | p |
|---|--:|--:|--:|--:|
| grids | −0.142 | −0.053 ± 0.016 | −0.024 | **1.000** |
| symbols | −0.359 | −0.053 ± 0.017 | −0.024 | **1.000** |
| prices | −0.066 | −0.053 ± 0.016 | −0.025 | **0.801** |

**Scrambled scores do as well as the real ones** — in fact the real result sits at the worse end of the scrambled range. Note also that the scrambled runs average about −0.05 rather than 0, for the same reason the mean-predictor does: R² is measured against each test fold's own average.

*(This check deliberately uses one fixed model — `StandardScaler → PCA(32) → Ridge(alpha=10)`, no tuning — so the 200 repeats stay comparable and affordable. That is why its real R² of −0.107 differs from the −0.114 in the head-to-head, which tunes alpha per fold.)*

**2. Asking the pipeline to predict something we know is in the data.** Same embeddings, same folds, but predict a property *of the glucose itself*:

| target | best R² |
|---|--:|
| glucose **variability** (SD) | **0.416** |
| mean glucose | −0.114 |
| % time above 180 | −0.079 |

**Variability is recovered clearly**, so the embeddings carry real information and the machinery is not broken.

**Be precise about the weak spot — do not call this check "passed."** The code only prints a success line if all three exceed 0.5 (`run_rigor.py:109`); only one comes close, so that line never printed and is absent from `rigor_real.md`. Absolute level is recovered poorly, and there is a known reason: **Chronos-Bolt applies instance normalization — it subtracts each series' own mean *and* divides by its own standard deviation** (`chronos_bolt.py:95-134`), so absolute height is removed before the encoder sees the data. (Several older docs in this repo say Chronos "mean-scales"; that describes Chronos-**T5**, not the model we use.) `embed()` returns those two removed numbers as `loc` and `scale`, and `encoders.py:104` currently discards them.

**What this means (important + honest):**
- The flat cognition result reflects the data, not a broken pipeline: the known-answer check recovers variability readily, and scrambled scores do just as well as real ones.
- But the frozen Chronos embedding is **largely blind to absolute glucose level** — awkward, because hypoglycemia and hyperglycemia are defined by absolute thresholds. Reassuringly, the **43 hand-crafted features do encode absolute level** (mean, time-in-range) and were *also* flat, so the finding holds for level-aware representations too.
- **Concrete next step:** keep `loc` and `scale` alongside the embedding. Tested and it does not change the conclusion, but it removes an obvious criticism.
- **A further honest comparison:** under the same protocol, **a single number — the average glucose — does as well as all 512 Chronos numbers** (−0.075 / −0.050 / −0.004 vs −0.114 / −0.093 / −0.069 with Ridge). Nothing beats guessing the average either way; the point is that the richer representation is not buying anything here.

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
