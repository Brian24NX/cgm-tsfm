# 01 · The Fact Sheet — every number, with proof

> **Purpose.** Liuyi said he wants to point at any detail and get *"an exact and right and knowledgeable answer with confidence."* This page is that. Every number here was recomputed from the actual CSVs and code in this repo on 2026-07-30 — none of it is remembered or guessed. Where a number can be *derived*, the derivation is shown, because "why is it 956?" is exactly the kind of question you will get.
>
> **How to use it.** Don't read it once. Cover the right column and quiz yourself. If you can produce these numbers cold, you cannot be caught on data.

---

## A. Study design (from the K01 grant, `Ray K01.pdf`)

| Fact | Value | Note |
|---|---|---|
| Condition studied | Type 1 Diabetes (T1D), youth | |
| Planned sample in the grant | **N = 92**, ages 9–16 | This is the *grant's* target |
| What **we actually have** | **20 participants** | Liuyi: more data *"is not guaranteed. I would not hope for that."* |
| CGM device | Dexcom G6 | |
| CGM sampling interval | **1 reading per 5 minutes** | → 12 readings/hour, 288/day |
| Cognitive app | **ARC** (Ambulatory Research in Cognition), built by Dr. Hassenstab | `ctrlab.org/projects/arc-smartphone-app/` |
| Testing schedule | ~5 short tests/day for ~10 days | |
| The three tests | **Grids, Symbols, Prices** | |
| Score direction | **All three: lower = better** | Liuyi stated this explicitly. Memorize it. |
| Aim 1 | Do glucose fluctuations predict cognitive performance in T1D youth? | ← *our* question |
| Aim 2 | T1D vs non-T1D controls during stable glucose | not our task |

**Your lane** (Liuyi, verbatim): *"'mixed-effects/association analysis' is what **Phil** is doing; 'ML prediction' is what **Mack** is doing. And what we want from you is **DL/TSFM prediction**."*

---

## B. The data files

| File | Rows | Participants | Unusable glucose |
|---|--:|--:|--:|
| `Cohort1_scores_merged_with_glucose.csv` | 740 | 14 | 23 |
| `Cohort2_scores_with_glucose.csv` *(superseded)* | 240 | 6 | **44** |
| `Updated_Cohort2_scores_with_glucose.csv` *(**used**)* | 240 | 6 | **1** |

**Which files the pipeline actually loads:** Cohort1 + **Updated_**Cohort2. Set in `cgm_tsfm/config.py:46-54` (`COHORT_FILES`).

**Why the "Updated_" file exists.** The original Cohort2 had 44 of 240 rows with empty glucose. Those gaps wiped out one participant entirely, leaving **19**. The corrected file has only 1 empty row, so all 6 Cohort2 participants survive → **20**.

> ⚠️ **Live disagreement — be ready for this.** Liuyi wrote: *"I did not work with 20 patients before, I only worked with 19 patients before"* and *"As far as I know, the CGM monitoring is continuous, i.e., always on. So, from my understanding, there is no need for imputation."* — note the two hedges; he is not asserting this as established fact. Our 19→20 recovery came from a file traced to MK, and **Phil has the code that did it**. You have not yet seen that code. The honest answer is: *"We use 20 because the corrected Cohort2 file fills in glucose that was blank in the original. I have not seen the code that filled it, so I can't yet tell you whether it was recovered from the raw stream or interpolated. I need to ask Phil for that script."* **Do not claim it is clean.** This is the single most likely place to be caught overstating.

### The row count, derived

```
   740   Cohort1 rows
+  240   Updated_Cohort2 rows
= 980    raw rows (one row = one cognitive test session)
-  24    rows whose Glucose_Before_Test cell is empty / unparseable  (23 + 1)
-   0    rows with fewer than 3 valid readings (min_readings=3) — none in practice
= 956    sessions used
```
Verified: `load_real_data()` returns exactly **956**.

| Split | Sessions | Participants |
|---|--:|--:|
| Cohort 1 (IDs `2xxxxx`) | 717 | 14 |
| Cohort 2 (IDs `3xxxxx`) | 239 | 6 |
| **Total** | **956** | **20** |

**Sessions per participant:** min **32**, median **47.5**, max **67**, mean **47.8**. (No participant is a tiny-N outlier — worth saying, since it means no one participant dominates.)

### Columns, and the one that matters
The glucose lives in **`Glucose_Before_Test`** — a *string* holding a Python-style list, e.g. `"[142, 145, nan, 151, ...]"`. Cohort2 embeds literal `nan` tokens, which is why `parse_glucose` (`data.py:26-48`) uses `eval` with a restricted namespace mapping `nan`→`np.nan` rather than `ast.literal_eval`. NaNs are then **dropped**, not filled.

Cohort naming differs and is harmonized in `data.py:195-199`: `session_id`→`session`, `Unix Time`/`combined time`→`combined_time`.

---

## C. The glucose data itself

**68,206 individual glucose readings** across 956 sessions.

| Statistic | Value |
|---|--:|
| Mean | **177.24** mg/dL |
| SD | 80.17 mg/dL |
| Median | 157 mg/dL |
| Min / Max | **40 / 400** mg/dL |
| 25th / 75th percentile | 116 / 222 |

**The 40 and 400 are the Dexcom G6 reporting limits, not real physiology.** Zero readings fall outside `[40, 400]`; all values are integers; all 361 integers from 40 to 400 appear. **1,470 readings (2.16%) sit exactly at 400** and 99 (0.15%) exactly at 40 — those are *censored* values meaning "≥400" and "≤40". If asked about outliers: there are none by construction; instead there is **censoring at both rails**, and 2.2% of readings pinned at the ceiling is a real limitation for a model trying to read the shape of a high excursion.

### Time in range (per reading, clinical thresholds)

| Range | % of readings |
|---|--:|
| < 54 (severe low) | 0.58% |
| **< 70 (low / hypoglycemia)** | **2.31%** |
| **70–180 (in range)** | **58.14%** |
| **> 180 (high / hyperglycemia)** | **39.54%** |
| > 250 (very high) | 17.66% |

*Context: clinical guidance targets >70% time in range; this cohort sits at 58%, i.e. these children are hyperglycemic a lot. Low readings are rare (2.3%) — which matters, because hypoglycemia is where a cognitive effect is most expected, and we barely have any.*

### Per session

| Sessions containing… | Count | % |
|---|--:|--:|
| any reading < 70 | 201 | 21.0% |
| any reading > 250 | 449 | 47.0% |
| entirely within 70–180 | 143 | 15.0% |

### Window length — the ugly, important one

| Percentile | Readings | Hours |
|---|--:|--:|
| min | **3** | 0.25 |
| 5th | 24 | 2.0 |
| 25th | 28 | 2.3 |
| **median** | **36** | **3.0** |
| 75th | 115 | 9.6 |
| 95th | 200 | 16.7 |
| max | **288** | **24.0** |

Mean 71.4 readings, SD 64.1. **This is a 96× spread between the shortest and longest input** (3 → 288 readings). 14 sessions have under 1 hour of data.

**Why so variable:** the stored array appears to be *all glucose since that participant's previous test* (the `Prev_Test_Time` column supports this), not a fixed lookback. An overnight gap → ~24 h; two tests an hour apart → 12 readings.

Liuyi's response when you raised this: *"This is a very good point, actually. I did not consider this point before."* — but he also corrected your wording: **do not call it a "confound."** His definition: a confound involves **two** variables; this is *one* variable taking different values. He *did* accept "confound" for time-of-day vs the glucose series, because those are two different variables.

---

## D. The three scores — what they actually are

This is worth memorizing because Liuyi asked what they measure and only answered "lower = better." The rest below was **reverse-engineered from the data** (see `11_WHAT_THE_SCORES_ARE.md` for the proof).

| | Grids | Symbols | Prices |
|---|--:|--:|--:|
| Sessions with a score | **951** | **920** | **936** |
| Missing | 5 | 36 | 20 |
| Mean | **0.478** | **1.846** | **41.59** |
| SD | **0.500** | **0.560** | **17.17** |
| Min | 0.000 | 0.150 | 0 |
| Max | 2.390 | 4.190 | 90 |
| Distinct values | 146 | 838 | **10** |
| What it is | mean placement error, in grid cells, over **6 items on a 5×5 grid** | a **response time in seconds** | a **percentage in steps of 10** (0,10,…,90) |
| Floor pile-up | **31.0% of sessions score exactly 0** | none | — |

**Between-participant share of variance** (how much of the score spread is "who you are" vs "which session"):

| Score | Between-participant | Within-participant |
|---|--:|--:|
| Grids | 0.153 | 0.847 |
| Symbols | **0.376** | 0.624 |
| Prices | 0.077 | 0.923 |

*Note this carefully — it corrects a claim in our older docs.* We had written that between-participant differences "dominate." They don't: **62–92% of the variance is within-participant**, i.e. session-to-session. So the reason grouped cross-validation is hard is *not* mainly that participants differ; it is that most of the variation is session-level noise that glucose does not explain.

**Cohort differences** (all three are worse in Cohort 2, remembering lower = better):

| Score | Cohort 1 | Cohort 2 |
|---|--:|--:|
| Grids | 0.447 | 0.580 |
| Symbols | 1.831 | 1.919 |
| Prices | 40.42 | 45.41 |

Also: Cohort 2's Grids/Symbols are partly **rounded to 2 decimals** while Cohort 1 is full precision (Cohort 1 has `0.33333333`, Cohort 2 has `0.33`). Harmless for modelling, but know it — it shows you looked.

---

## E. How strong is the raw signal? (the honest ceiling)

Correlation between a simple glucose statistic and a score, across all 956 sessions:

| Score | Strongest single correlation | r | r² |
|---|---|--:|--:|
| Grids | number of readings *(!)* | −0.077 | 0.006 |
| Symbols | fraction of readings < 70 | **−0.092** | **0.0085** |
| Prices | max glucose | +0.056 | 0.003 |

**The single strongest relationship anywhere in this dataset explains 0.85% of the variance.** Every other pairing is under 0.6%. Say it plainly: *before any model, the raw association between glucose and score is essentially flat.* No model can manufacture a signal that isn't there — this is the number that makes the low accuracy unsurprising rather than suspicious.

*(Note the Grids "signal" is with window **length**, not glucose at all — an artefact of the variable window, and a good example of why the variable lookback is worth fixing.)*

---

## F. The model: Chronos

| Fact | Value |
|---|---|
| Family | Amazon **Chronos** (a time-series foundation model) |
| Default checkpoint | **`amazon/chronos-bolt-small`** |
| Embedding width (`d_model`) | **512** |
| Pooling | **mean** over the token axis |
| Frozen? | **Yes** — `torch.no_grad()`, no gradients, weights never change |
| Feature matrix shape | **(956, 512)** |
| Hand-crafted comparison | **(956, 43)** features |

Checkpoint sizes tried: `bolt-tiny` (d=256), `bolt-mini` (384), `bolt-small` (512), `bolt-base` (768), `chronos-t5-small` (512).

*(Verified architecture internals — layers, patch size, how scaling works — are in `08_CHRONOS.md`.)*

---

## G. The evaluation protocol

| Setting | Value | Where |
|---|---|---|
| Outer folds | **5**, `GroupKFold` on `subid` | `config.py:40` |
| Inner folds | **3**, `GroupKFold` inside `GridSearchCV` | `config.py:41` |
| Random seed | **42** | `config.py:39` |
| Scaler | `StandardScaler`, fit on **train fold only** | `regression.py:116` |
| Inner tuning metric | `neg_mean_squared_error` | `regression.py:124` |
| Leakage guard | `assert` train/test participants disjoint, every fold | `regression.py:113` |

**What the 5 folds actually look like** (20 participants → 4 held out per fold):

| Fold | Test participants | Test sessions | Train sessions |
|---|--:|--:|--:|
| 1 | 4 | 189 | 767 |
| 2 | 4 | 193 | 763 |
| 3 | 4 | 191 | 765 |
| 4 | 4 | 191 | 765 |
| 5 | 4 | 192 | 764 |

**Models in the grid** (`regression.py:38-47`):
- `Ridge`, α ∈ {0.1, 1, 10, 100, 1000} → 5 settings
- `SVR` (RBF kernel), C ∈ {0.1, 1, 10} × γ ∈ {scale, auto} → 6 settings
- `LinearRegression`, no settings
- `DummyRegressor(strategy="mean")` — the comparison baseline

> **XGBoost is in the code but did NOT run.** `regression.py:172-179` wraps the import in `try/except`, and **xgboost is not installed in this environment**. So it was silently skipped in every result you have. If Liuyi asks "did you try gradient boosting?", the true answer is **no** — the code would have, but the package is absent. Don't let a code-reading catch you here.

**Number of model fits per Arm A run:** per target, 5 (baseline) + 80 (Ridge) + 95 (SVR) + 5 (Linear) = **185**; × 3 targets = **555 fits**.
*Ridge: 5 settings × 3 inner folds = 15, +1 refit = 16 per outer fold, ×5 = 80. SVR: 6 × 3 = 18, +1 = 19, ×5 = 95.*

---

## H. Arm B: the trainable head

| Setting | Value |
|---|---|
| Architecture | `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)` |
| **Trainable parameters** | **132,609** |
| Optimizer | **AdamW**, lr **1e-3**, weight decay **0.01** |
| Loss | **MSE**, on z-scored targets |
| Max epochs | **100** |
| Early stopping | monitor `val/loss`, **patience 8**, mode min |
| Batch size | **64** |
| Validation split | **20%** of the outer-train participants, `GroupShuffleSplit` |
| Encoder | **frozen** — only the head trains |
| Trainings per run | **5 folds × 3 targets = 15** |

**Parameter breakdown** (this is the "how does the MLP head work" answer, in numbers):

| Layer | Shape | Parameters |
|---|---|--:|
| LayerNorm weight | (512,) | 512 |
| LayerNorm bias | (512,) | 512 |
| Linear 1 weight | (256, 512) | 131,072 |
| Linear 1 bias | (256,) | 256 |
| Linear 2 weight | (1, 256) | 256 |
| Linear 2 bias | (1,) | 1 |
| | **Total** | **132,609** |

**Epoch arithmetic** (be able to do this live):
```
956 sessions, 5 grouped folds
  → test ≈ 191,  outer-train ≈ 765
  → 20% of outer-train for validation → val ≈ 153,  train ≈ 612
  batch_size 64 → ceil(612 / 64) = 10 batches
  ⇒ 1 epoch = 10 optimizer steps
  ⇒ 100 epochs = at most 1,000 steps (early stopping usually ends it far sooner)
```
**132,609 parameters trained on ~612 rows = 217× more parameters than training examples.** That single ratio explains why Arm B is the *worst* performer: it has vastly more freedom than the data can constrain. Dropout, weight decay and early stopping are the three things holding it back from memorizing outright.

---

## I. The results (real data, 20 participants, 956 sessions)

> ### 🔧 REGENERATED 2026-08-01 after the pooling fix
>
> The batch-invariance bug described in `15_FINDINGS_TO_REPORT.md` has now been **fixed in the code** (`cgm_tsfm/encoders.py`, `ben_adapter/model.py`) and **every result file below was regenerated** from corrected embeddings. Pre-fix outputs are preserved in `results/archive_prebugfix_2026-07-07/`.
>
> **The conclusion did not change: every number is still at or below zero.** What moved:
>
> | | grids | symbols | prices |
> |---|--:|--:|--:|
> | Arm A — before → **after** | −0.089 → **−0.114** | −0.056 → **−0.052** | −0.012 → **−0.012** |
> | Arm B — before → **after** | −0.218 → **−0.313** | −0.281 → **−0.412** | −0.168 → **−0.360** |
> | Centered — before → **after** | −0.018 → **−0.027** | −0.022 → **−0.028** | −0.002 → **−0.001** |
>
> **Arm B's drop is real, not run-to-run noise.** Measured across 3 seeds on each embedding set: on the old embeddings grids = −0.218 ± 0.018, on the corrected ones −0.323 ± 0.011. The gap (≈0.105) is roughly 6× the seed spread. Same for prices (−0.144 ± 0.005 → −0.341 ± 0.041).
>
> **Why fixing a bug made Arm B *worse* — have this answer ready.** The buggy pooling averaged each session's real tokens together with padding positions. That blurring acted as an accidental smoother, shrinking the differences between sessions. A 132,609-parameter head on ~612 training rows benefits from blurred inputs, because there is less spurious detail to memorize. Give it sharper, genuinely per-session features and it overfits harder. This is not a regression — it is the capacity problem showing through more clearly once the features are correct.
>
> **Numbers in the teaching chapters (02–13) may still quote the pre-fix values in worked examples.** Those examples remain arithmetically valid; this page and `results/` are authoritative for current values.

**R² against a "predict the average" comparison. 0 = no better than guessing the average. Negative = worse than that.**

Grouped cross-validation — *"predict a child the model has never seen"*:

| Score | Chronos + Ridge/SVR (Arm A) | Chronos + MLP head (Arm B) | 43 hand-crafted features |
|---|--:|--:|--:|
| Grids | **−0.114** | −0.313 | −0.065 |
| Symbols | **−0.052** | −0.412 | −0.038 |
| Prices | **−0.012** | −0.360 | −0.009 |

After removing each child's own average score — *"predict this child's good vs bad sessions"*:

| Score | Chronos + Ridge/SVR | hand-crafted |
|---|--:|--:|
| Grids | −0.027 | −0.004 |
| Symbols | −0.028 | −0.004 |
| Prices | −0.001 | −0.002 |

**Every cell is ≤ 0.** Nothing beats guessing the average.

### ⚠️ The most important nuance in this whole document: 0 is NOT the neutral point

Measured directly (`DummyRegressor(strategy="mean")` under the same 5 grouped folds, 2026-07-30):

| Score | R² of the mean-predictor **itself** |
|---|--:|
| Grids | **−0.0775 ± 0.0988** |
| Symbols | **−0.0503 ± 0.0396** |
| Prices | **−0.0044 ± 0.0028** |

Compare against Arm A (−0.114 / −0.052 / −0.012). **Our model is within about 0.01–0.04 R² of a constant predictor — that is essentially a tie, not a collapse.** (On Symbols the model is now *marginally better* than the constant predictor: −0.052 vs −0.050.)

**Why the baseline is negative at all.** `r2_score` measures each test fold against *that fold's own* mean. But the model is trained on the *training* fold's mean. With only 4 participants held out, those two means differ. Fold 1 worked out exactly: train mean 0.519256, test mean 0.311700, offset 0.207556, test variance 0.172094 → R² = −(0.207556²)/0.172094 = **−0.250326**, matching the measured value to 6 decimals.

So under leave-participants-out cross-validation with 4 participants per fold, a slightly-negative R² *is* the neutral point.

**Say it this way:** *"Our R² is negative, but so is the mean-predictor's — it scores −0.078 on Grids under the same folds. The reason is that R² compares against each test fold's own average, and with only four children held out, their average differs from the training average. So the honest statement is that our model ties the trivial baseline. It does not beat it, and it is not catastrophically broken either."*

This is a far stronger and more accurate framing than "our R² is negative", and it protects you from *"negative R² means your model is broken"*.

> **A second measured correction:** sklearn's `GroupKFold` defaults to `shuffle=False`, so **`RANDOM_STATE = 42` does not affect the outer split at all.** The outer folds are deterministic regardless of seed. The seed does affect the `GroupShuffleSplit` validation split in Arm B and the models' own randomness. Don't claim the seed controls the folds.
>
> **A third:** the per-target folds are *not* the same folds with rows removed — each target re-splits after masking its own missing scores. Per-target test-fold sizes: grids 188/191/192/190/190, symbols 170/190/188/187/185, prices 187/189/187/187/186.

### Where we looked for a signal and didn't find one

**Model size** (`sweep_real.md`) — bigger is *not* better:

| Checkpoint | d | mean R² |
|---|--:|--:|
| bolt-tiny | 256 | −0.051 |
| bolt-mini | 384 | −0.049 |
| bolt-small | 512 | −0.059 |
| **bolt-base** | 768 | **−0.067** ← worst (tied) |
| t5-small | 512 | −0.067 |

*The 24× parameter range from tiny (8.7M) to base (205M) spans just 0.018 in mean R², and the largest is the worst. A bigger model is not the answer.*

**Window length** — shortening it makes things *worse*: 2.0 h −0.077, 2.5 h −0.076, 3.0 h −0.071, full −0.059.
**Pooling** — mean (−0.059) and the summary token (−0.060) are now essentially identical. *This is itself evidence the fix worked: before the fix, mean pooling was contaminated by padding and the two differed more.*
**PCA** — 16/32 components help (−0.059 → −0.045); 128 gives it back.
**Glucose regime subgroups** (`subgroups_real.md`):

| Subgroup | Sessions | Participants | Grids | Symbols | Prices |
|---|--:|--:|--:|--:|--:|
| all | 956 | 20 | −0.114 | −0.052 | −0.012 |
| any reading < 70 | 201 | 19 | −0.224 | −0.109 | −0.086 |
| any reading > 250 | 449 | 20 | −0.094 | −0.134 | −0.087 |
| any excursion | 577 | 20 | −0.118 | −0.246 | −0.018 |
| entirely 70–180 | 143 | 19 | −0.153 | −0.289 | −0.110 |

> ⚠️ Liuyi's correction (Word comment C0): in the subgroup table, say **sessions**, not "patients". The 201/449 are *sessions*; the 19/20 are participants. Get the unit right.

### The two checks that make the low accuracy credible (`rigor_real.md`)

**1. Shuffle test** — scramble the scores 200× so any real link is destroyed, re-run everything, and see where the true result lands in that pile of nonsense results:

| Score | Real R² | Shuffled: mean ± SD | Shuffled 95th | p |
|---|--:|--:|--:|--:|
| Grids | −0.107 | −0.048 ± 0.015 | −0.025 | **1.000** |
| Symbols | −0.096 | −0.052 ± 0.015 | −0.028 | **0.995** |
| Prices | −0.069 | −0.050 ± 0.014 | −0.027 | **0.910** |

The real result is not merely inside the scrambled range — it is at the **bad end** of it. (Model used for this test: `StandardScaler → PCA(32) → Ridge(α=10)`, fixed, no tuning. That's why the real R² here, −0.107, differs from the −0.114 in the main table, which tuned α per fold. **Know that difference** — it looks like an inconsistency if you can't explain it.)

**2. Same-embeddings sanity check** — ask the identical pipeline to predict something about glucose itself, which it *should* be able to do:

| Target | Best R² |
|---|--:|
| Glucose **variability** (SD) | **0.462** |
| Mean glucose | 0.042 |
| % time > 180 | 0.068 |

Variability is predicted well → the embeddings carry real information and the machinery works. So the flat cognition result is about the data, not a broken pipeline.

**But be honest about the weak spot:** mean glucose scores only 0.042. That is *expected* — Chronos-Bolt subtracts each series' own mean and divides by its own standard deviation, discarding absolute height — but it means our Chronos result speaks to glucose **shape**, not **level**. The saving grace: the 43 hand-crafted features *do* encode absolute level (mean, time-in-range) and they were flat too. So "no relationship" holds for level-aware features as well.

*Two things moved after the pooling fix, and only one has a confirmed explanation:*
- *Glucose **variability** is unchanged at **0.462** before and after. That information is genuinely in the curve's shape, so correcting the padding didn't touch it.*
- *Mean glucose slipped from 0.070 to **0.042**. **I do not have a verified explanation for that specific change.** A tempting story is that the padding contamination leaked window length, which then stood in for glucose level — but the numbers don't support it: window length correlates with mean glucose at only r = −0.071 (0.5% of the variance), far too weak to account for a 2.8-point R² move. Both values simply say the same thing — absolute level is poorly recovered — and the gap between them is unexplained. If asked, say that, rather than inventing a mechanism.*

*(Window length does correlate strongly with glucose **SD**, r = +0.384. That's a separate and genuinely useful fact: longer windows have more room to swing.)*

> The code prints a "check passed" line only if **all** three of those are > 0.5 (`run_rigor.py:109`). Only one is. **So that line never printed** — and it isn't in `rigor_real.md`. Never claim the check "passed" outright; say variability is recovered well and level is not, and explain why.

---

## J. Environment (as of 2026-07-30)

| Component | Version |
|---|---|
| Python env | `/data_1_8TB_ssd/brian_workspace/envs/cgm` |
| chronos | 2.3.1 |
| torch | 2.6.0+cu124 |
| lightning | 2.6.5 |
| scikit-learn | 1.9.0 |
| numpy / pandas / scipy | 2.4.6 / 3.0.3 / 1.17.1 |
| torchmetrics | 1.9.0 |
| **xgboost** | **not installed** |
| GPU | **NVIDIA RTX 6000 Ada, 48 GB**, CUDA available |

Note: `EncoderConfig.device` defaults to `"cpu"` (`config.py:85`). The GPU is used only when a run passes `--device cuda`.

---

## K. Ten numbers to have on instant recall

1. **20** participants — 14 + 6.
2. **956** sessions, from 980 raw rows minus 24 empty-glucose rows.
3. **5 minutes** per reading; **68,206** readings total.
4. Window: **3 to 288** readings, median **36** (= 3 hours).
5. Glucose mean **177** mg/dL; **58%** in range; only **2.3%** below 70.
6. Chronos-bolt-small → **512** numbers per session; matrix **(956, 512)**.
7. **5** outer grouped folds, **4** participants held out each time.
8. MLP head = **132,609** trainable parameters on **~612** training rows.
9. Best real R² = **−0.012** (Prices, Arm A). Everything is ≤ 0. The mean-predictor itself scores **−0.078 / −0.050 / −0.004**, so we *tie* it.
10. Shuffle-test p = **1.000 / 0.995 / 0.910**. Glucose-variability check = **0.462**; mean glucose only **0.042**.

*(Items 9 and 10 are post-fix values, regenerated 2026-08-01.)*

---

*Every figure above was recomputed from `data/Merged_glucose_data/*.csv` and `cgm_tsfm/` on 2026-07-30. If a number here disagrees with an older document in this repo, this page is the newer one — and `11_WHAT_THE_SCORES_ARE.md` explains the one substantive correction (within- vs between-participant variance).*
