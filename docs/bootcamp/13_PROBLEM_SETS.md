# 13 · Problem Sets — the workbook

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this is.** Nine problem sets plus a final exam. Everything is solvable with pen and paper. Nothing says "run the code and see."
>
> **Why a workbook.** Reading the other chapters gets you *recognition* ("yes, I've seen that"). Only working problems gets you *recall* ("R² of −0.089 means our squared error was 8.9% larger than guessing the average, and here is why the baseline itself scores −0.078"). In a meeting you need recall.

---

## How to use this

1. **Self-test first.** Every problem set hides its answers in a collapsed `Solutions` block at the end. Do not open it until you have written an answer down. An answer you *nearly* had is worth ten answers you read.
2. **Write the arithmetic out.** Half of these are 3-line calculations. If you can only do them "in principle," you cannot do them at a whiteboard.
3. **Say the answer out loud in plain words** after checking it. The supervisor's questions arrive as English, not as equations.
4. **Pace: two problem sets a day for five days**, then two days of consolidation.

| Day | Do | Also |
|---|---|---|
| 1 | PS1, PS2 | read `01_FACT_SHEET.md`, `03_PREPROCESSING.md` |
| 2 | PS3, PS4 | read `05_EVALUATION.md` |
| 3 | PS5, PS6 | read `04_LINEAR_MODELS.md`, `06_DEEP_LEARNING_CORE.md` |
| 4 | PS7, PS8 | read `07_TRANSFORMERS.md`, `08_CHRONOS.md`, `09_THE_TWO_ARMS.md` |
| 5 | PS9 | read `10_RIGOR_AND_STATS.md` |
| 6 | Redo, cold, every problem you got wrong | — |
| 7 | Final exam, timed, closed-book | — |

### Difficulty tags

| Tag | Means |
|---|---|
| `[warm-up]` | You should get this in under 30 seconds. If not, you are not ready to move on. |
| `[core]` | The bread and butter. These are the questions most likely to be asked. |
| `[hard]` | Requires combining two ideas, or spotting a subtlety. |
| `[interview-killer]` | If you can answer this cleanly, you will look like you own the project. |
| `[supervisor-asked]` | This exact question has already been asked in a meeting. Non-negotiable. |

### Scoring guide — what "ready" means

| Problem set | Ready at | Why that bar |
|---|---|---|
| PS1, PS2, PS3 | **90%** | Pure arithmetic about your own data. Getting these wrong is the one unforgivable category. |
| PS4, PS5, PS6, PS7 | **75%** | Mechanism questions. You need the shape of the answer plus one number. |
| PS8 | **90%** | These are *your results*. There is no excuse. |
| PS9 | judgement | No single right answer. You are ready when you can defend a position using a verified number. |
| Final exam | **12/15** | 12–15 ready · 9–11 re-drill PS4–PS7 · under 9 go back to the chapters. |

### Which numbers in here are real

Every number tagged **verified** was recomputed from `data/Merged_glucose_data/*.csv`, `cgm_tsfm/*.py`, the installed `chronos` package, and the `amazon/chronos-bolt-small` checkpoint on 2026-07-30. A few small tables are marked **illustrative** — those are constructed for the exercise (e.g. a made-up list of validation losses) and are not logged output. Never quote an illustrative number as a result.

Two places where this workbook **corrects** `01_FACT_SHEET.md`; both are flagged in the solutions and both are worth knowing, because a code-reader could otherwise catch you:

- The Arm B validation split holds out **4 of 16 participants (about 190–210 sessions)**, not "20% of 765 rows = 153". Training folds are **557–576 sessions → 9 batches per epoch**, not 10.
- The Chronos parameter split is **20,015,872 used / 27,702,144 unused**. The figure 27,703,168 double-counts a 1,024-parameter table.

### Conventions used throughout

- **session** = one cognitive test occasion = one row of the data = one training example. **participant** = one child. 956 sessions, 20 participants. Never write "patients" when you mean sessions.
- `ceil(x)` = round up. `n` = number of rows. `p` = number of features.
- All three scores are **lower = better**.
- R² is always measured against a "predict the average" comparison: 0 = no better than guessing the average, negative = worse than that.

---
---

# PS1 — Data and shapes

**1.1** `[warm-up]` The two cohort files have 740 and 240 rows. 23 and 1 of those rows have an empty glucose cell. The pipeline also drops any session with fewer than 3 valid readings. Derive the number of sessions the pipeline actually uses, and state in one sentence what one row is.

**1.2** `[warm-up]` The CGM samples once every 5 minutes. Convert: (a) 36 readings to hours; (b) 3 readings to minutes; (c) 288 readings to hours; (d) 71.35 readings to hours; (e) a 2-hour lookback to readings.

**1.3** `[core]` There are 68,206 glucose readings across the 956 sessions. Compute (a) the mean readings per session; (b) the total wall-clock time those readings cover, in days; (c) the same per participant. (d) Does (c) square with the study design of roughly 10 days of monitoring?

**1.4** `[core]` The number of tokens Chronos produces for a series of length `T` is `ceil(min(T, 2048)/16) + 1`. Compute it for T = 3, 16, 17, 36, 45, 71, 288, 2048, 3000.

**1.5** `[core]` Chronos-bolt-small has a context length of 2048 readings. (a) How long is that in days of CGM? (b) Does the cap ever bind on our data? Justify with a number.

**1.6** `[core]` `[supervisor-asked]` Fill in the shape at every stage, for a batch of 32 sessions whose longest member has 288 readings.

| Stage | Shape |
|---|---|
| raw CSV rows | ? |
| windows after parsing (a Python list) | ? |
| one batch, left-padded | ? |
| after patching (values + observed-flags) | ? |
| after the input patch embedding | ? |
| after appending [REG] | ? |
| after the 6 encoder layers | ? |
| after mean-pooling over tokens | ? |
| the full feature matrix for all sessions | ? |
| X and y for the grids model | ? |
| after `StandardScaler` | ? |
| after `PCA(32)` | ? |
| the predictions for one test fold | ? |

**1.7** `[core]` The encoder runs with `batch_size = 32`. How many batches does it take to embed all 956 sessions, and how many sessions are in the last one?

**1.8** `[core]` Here is one hour of made-up CGM, 12 readings in mg/dL:

`65, 72, 88, 140, 175, 182, 210, 265, 300, 255, 190, 160`

Compute (a) how long this window is in minutes; (b) the mean; (c) the percentage below 70; (d) the percentage in 70–180 inclusive; (e) the percentage above 180; (f) the percentage above 250. Then compare (c)–(e) with the real cohort figures.

**1.9** `[hard]` (a) How many individual numbers are in the Chronos feature matrix? (b) How many raw glucose readings does that matrix summarise? (c) Divide. What does the ratio tell you about calling the embedding a "compression"?

**1.10** `[hard]` Grids has 951 scored sessions, symbols 920, prices 936, out of 956. (a) How many are missing for each, and what percentage? (b) Why do the three models see different numbers of rows? (c) Name one consequence for comparing the three R² numbers.

**1.11** `[interview-killer]` The window length runs from 3 readings to 288. (a) What is the ratio? (b) What does the stored array actually represent? (c) Using a verified number from our own data, name one concrete way this could manufacture a correlation that has nothing to do with glucose.

<details>
<summary><b>Solutions — PS1</b></summary>

**1.1**
```
   740   Cohort1 rows
+  240   Updated_Cohort2 rows
=  980   raw rows
-   24   empty / unparseable glucose  (23 + 1)
-    0   fewer than 3 valid readings  (none in practice)
=  956   sessions used            [verified]
```
Cross-check per cohort: 740 − 23 = **717** sessions from 14 participants; 240 − 1 = **239** from 6 participants; 717 + 239 = 956, and 14 + 6 = **20 participants**.

One row = **one cognitive-test session** — one occasion on which one child took the tests, paired with the glucose readings recorded before it. A child who tested 47 times contributes 47 rows. (Verified session counts per participant range from 32 to 67.)

**1.2** 12 readings per hour, 288 per day.
(a) 36 × 5 = 180 min = **3.0 h**. (b) 3 × 5 = **15 min**. (c) 288 × 5 = 1440 min = **24.0 h**. (d) 71.35 × 5 = 356.75 min = **5.95 h**. (e) 2 h = 120 min ÷ 5 = **24 readings**.

**1.3**
(a) 68,206 ÷ 956 = **71.35 readings** per session (the fact sheet rounds this to 71.4).
(b) 68,206 × 5 = 341,030 min = 5,683.8 h = **236.8 days**.
(c) 236.8 ÷ 20 = **11.8 days per participant**.
(d) Yes. The design is about 10 days of monitoring per child, and the windows are non-overlapping stretches between consecutive tests, so their total should be a bit more than the testing period. 11.8 days is exactly the right order. This is a good sanity check to be able to do out loud — it shows the windows tile the monitoring period rather than double-counting it.

**1.4**

| T | ceil(T/16) patches | + [REG] | tokens |
|--:|--:|--:|--:|
| 3 | 1 | +1 | **2** |
| 16 | 1 | +1 | **2** |
| 17 | 2 | +1 | **3** |
| 36 | 3 | +1 | **4** |
| 45 | 3 | +1 | **4** |
| 71 | 5 | +1 | **6** |
| 288 | 18 | +1 | **19** |
| 2048 | 128 | +1 | **129** |
| 3000 | 128 (capped) | +1 | **129** |

All verified by calling `embed()` on single series of exactly those lengths.

**Subtlety.** Two traps live here. First, forgetting the `+1` — the [REG] token is appended *last* and is not derived from any reading. Second, using floor instead of ceiling: 36 readings do not make 2 patches, they make 3, because the patcher left-pads the series up to a multiple of 16. And note T = 45 gives the same 4 tokens as T = 36, because both round up to 3 patches.

**1.5**
(a) 2048 × 5 = 10,240 min = 170.67 h = **7.11 days**.
(b) **No.** Our longest window is 288 readings (24 hours), which is 14% of the 2048 cap. The `min(T, 2048)` in the formula never activates on our data. Say this crisply if asked "do you truncate?" — the answer is *the model would, at 7.1 days, but our longest input is 1 day*.

**1.6**

| Stage | Shape | Note |
|---|---|---|
| raw CSV rows | 980 rows | one row per session, glucose stored as a string holding a list |
| windows after parsing | `list` of **956** 1-D arrays | lengths 3 to 288 — **not** a rectangular array |
| one batch, left-padded | **(32, 288)** | padded with `NaN` to the longest member of the batch |
| after patching | **(32, 18, 32)** | 18 patches of 16 readings; last axis = 16 values then 16 observed-flags |
| after input patch embedding | **(32, 18, 512)** | one 512-vector per patch |
| after appending [REG] | **(32, 19, 512)** | the extra token goes last |
| after the encoder | **(32, 19, 512)** | encoder is shape-preserving |
| after mean-pooling over tokens | **(32, 512)** | average over axis 1 |
| full feature matrix | **(956, 512)** | |
| grids X and y | **(951, 512)** and **(951,)** | 5 sessions have no grids score |
| after `StandardScaler` | unchanged, e.g. **(765, 512)** for a training fold | it rescales columns, it does not reshape |
| after `PCA(32)` | **(765, 32)** | |
| predictions for one test fold | **(191,)** | one number per test session |

The one-sentence version, which is what the supervisor actually wants: *"One session goes in as a variable-length list of glucose numbers and comes out as 512 numbers; stacked up that is a 956 × 512 table, and each row has one score attached."*

**1.7** 956 ÷ 32 = 29.875 → **30 batches**; the last holds 956 − (29 × 32) = 956 − 928 = **28 sessions**.

**1.8**
(a) 12 × 5 = **60 minutes**.
(b) Sum = 2102; 2102 ÷ 12 = **175.17 mg/dL**.
(c) Below 70: just `65` → 1/12 = **8.33%**.
(d) In 70–180: `72, 88, 140, 175, 160` → 5/12 = **41.67%**.
(e) Above 180: `182, 210, 265, 300, 255, 190` → 6/12 = **50.00%**.
(f) Above 250: `265, 300, 255` → 3/12 = **25.00%**.
Check: 1 + 5 + 6 = 12.

Comparison with the real cohort (verified): below 70 **2.31%**, in range **58.14%**, above 180 **39.54%**, above 250 **17.66%**, mean **177.24**. So this made-up hour has about the right mean but far too much hypoglycemia — which is the point worth remembering: *lows are rare in this cohort, and lows are where a cognitive effect is most expected*.

**1.9**
(a) 956 × 512 = **489,472** numbers.
(b) **68,206** readings.
(c) 489,472 ÷ 68,206 = **7.18**. The embedding is **7.2× larger than the raw data it describes** — it is an *expansion*, not a compression. That matters twice: it is why plain linear regression blows up (512 features from 765 rows), and it is why "we used a foundation model to extract features" should not be described as summarising or reducing the data.

**1.10**
(a) Grids **5** missing (0.52%), symbols **36** (3.77%), prices **20** (2.09%).
(b) Each target is masked independently in the code (`dataset.target_mask(target)`), so a session with a symbols score but no grids score is used for symbols and dropped for grids. There are three separate models, one per target — not one model with three outputs.
(c) The three R² values are computed on **different row sets and therefore different cross-validation folds**. They are comparable in spirit but not literally the same test set, and the fold sizes differ (verified: grids test folds 188–192, symbols 170–190, prices 186–189). Do not claim the three numbers are paired.

**1.11**
(a) 288 ÷ 3 = **96×**.
(b) It is **all glucose since that participant's previous test**, not a fixed lookback. Two tests an hour apart give 12 readings; a test after an overnight gap gives about 288.
(c) **Verified:** the strongest single correlation with the grids score anywhere in the dataset is with the **number of readings** (r = −0.077), not with any glucose value. Window length is a proxy for time-since-last-test, which is a proxy for time of day and for whether the child slept. So a model that reads window length is reading the clock, not the glucose.

**Subtlety and a wording landmine.** The supervisor has already corrected the word "confound" here: a confound involves *two* variables, and window length is *one* variable taking different values across sessions. He accepted "confound" for time-of-day versus the glucose series, because those are two different variables. Say "the input is not a fixed physiological lookback" or "window length is a nuisance variable" and you are safe.

</details>

---
---

# PS2 — Normalization by hand

Use this 30-minute glucose series throughout, called **S**:

```
S = 135, 155, 155, 165, 165, 185      (6 readings, mg/dL)
```

**2.1** `[warm-up]` `[supervisor-asked]` In two sentences: what is normalization doing inside Chronos, and why does a model like this need it at all?

**2.2** `[warm-up]` Chronos computes `loc` as the NaN-aware mean of the series. Compute `loc` for S.

**2.3** `[core]` Chronos computes `scale` as the **population** standard deviation (divide by n, not n−1). Compute it for S, showing the deviations and their squares.

**2.4** `[core]` Compute the standardized series `(x − loc)/scale`. Give the values as fractions and as decimals, then verify the result has mean 0 and population sd 1.

**2.5** `[core]` Now compute the **sample** standard deviation (divide by n−1) for S. (a) What is it? (b) By what factor do the standardized values change if you use it by mistake? (c) Which one does Chronos use?

**2.6** `[core]` A missing reading appears: `S' = 135, 155, NaN, 155, 165, 165, 185` (7 slots). (a) What `loc` and `scale` would Chronos compute? (b) What does *our* pipeline do to that NaN before Chronos ever sees it, and what does that silently break?

**2.7** `[core]` A session records a perfectly flat series: `180, 180, 180, 180`. (a) What are `loc` and `scale`? (b) The code does `scale = where(scale == 0, eps, scale)` with `eps = 1e-5`. What is the standardized series? (c) State the uncomfortable consequence in one sentence.

**2.8** `[core]` Arm B z-scores the target using the training fold's mean and standard deviation. Pretend those are the overall grids values, mean 0.478 and sd 0.500. (a) Z-score a session whose grids score is 1.228. (b) The head outputs −0.400 for another session; convert that back to a grids score. (c) Why z-score the target at all? Support it with the ratio of the prices variance to the grids variance.

**2.9** `[hard]` Leak or not? Decide for each, and say what our code actually does.
(a) Fit one `StandardScaler` on all 956 rows, then run grouped cross-validation.
(b) Chronos standardizes each session using only that session's own readings.
(c) Subtract each participant's own mean score from their sessions, then run grouped cross-validation.
(d) Choose the Ridge alpha by looking at the outer test fold's R².
(e) Compute the target's mean and sd from the training split only, and use them to z-score train, validation and test.

**2.10** `[hard]` Two ordering questions. (a) Why must `PCA` be fitted inside the fold loop rather than once on all 956 rows? (b) Chronos runs *outside* the loop — the embeddings for all 956 sessions are extracted once and cached. Is that a leak?

**2.11** `[interview-killer]` Give the three verified numbers that prove Chronos throws away absolute glucose level, and then give the one number that stops that from sinking the whole study.

<details>
<summary><b>Solutions — PS2</b></summary>

**2.1** It puts every series on the same scale by subtracting the series' own mean and dividing by its own spread, so what reaches the transformer is *shape* — the pattern of ups and downs — rather than raw units. It is needed because Chronos was pre-trained across wildly different time series (electricity, traffic, retail, weather) whose units and magnitudes have nothing in common; a model that had to cope with raw magnitudes would waste all its capacity on units. The technical name in the code is **instance normalization**, "instance" meaning *each series separately*, and it happens inside `chronos_bolt.py` before any patching.

The consequence to state in the same breath, because it is the interesting half: this is why our embeddings predict glucose **variability** at R² 0.462 but mean glucose only at 0.070.

**2.2** loc = (135 + 155 + 155 + 165 + 165 + 185)/6 = 960/6 = **160.0**.

**2.3**

| x | x − loc | (x − loc)² |
|--:|--:|--:|
| 135 | −25 | 625 |
| 155 | −5 | 25 |
| 155 | −5 | 25 |
| 165 | +5 | 25 |
| 165 | +5 | 25 |
| 185 | +25 | 625 |
| | sum 0 | **1350** |

Population variance = 1350/6 = 225. `scale` = √225 = **15.0**. (Verified against `torch.nanmean((x−loc)²).sqrt()`, which is exactly what the code computes.)

**2.4** Divide each deviation by 15:

`−25/15, −5/15, −5/15, +5/15, +5/15, +25/15` = **−5/3, −1/3, −1/3, +1/3, +1/3, +5/3** ≈ −1.667, −0.333, −0.333, +0.333, +0.333, +1.667.

Check: they sum to 0, so the mean is 0. Sum of squares = 2(25/9) + 2(1/9) + ... = (625 + 25 + 25 + 25 + 25 + 625)/225 = 1350/225 = 6, and 6/6 = 1, so the population sd is 1.

**2.5**
(a) 1350/5 = 270; √270 = **16.4317**.
(b) The ratio is √(6/5) = 1.0954, so every standardized value would come out **8.7% smaller in magnitude** (−1.667 becomes −1.522).
(c) **Population** (divide by n, `ddof = 0`).

**Subtlety — this is a deliberate trap.** Almost every statistics course teaches the n−1 version as "the" standard deviation, so the reflex answer is wrong. Two extra things worth knowing: `numpy`'s `np.std` defaults to `ddof = 0` (population) while `pandas`' `.std()` defaults to `ddof = 1` (sample), which is a classic source of two slightly different numbers for the same column; and the gap matters most for short series, which is exactly what we have — at n = 6 it is 9%, at n = 288 it is 0.17%.

**2.6**
(a) NaN-aware means the missing slot is skipped entirely, so the mean and the mean-of-squared-deviations are both computed over the 6 valid readings: **loc = 160.0, scale = 15.0** — identical to S. (Verified.)
(b) Our `parse_glucose` **drops** NaNs rather than passing them through: `arr = arr[~np.isnan(arr)]`. So Chronos receives a 6-element array, not a 7-slot array with a hole. What that silently breaks is the **time axis**: the gap is closed up, so readings that were 10 minutes apart are presented as if they were 5 minutes apart, and everything after the gap shifts earlier. Chronos has no timestamps — it only sees position — so it cannot know. The NaN-awareness of the scaler therefore matters for us only for the padding Chronos adds itself, not for real dropouts.

That is an honest weakness to volunteer, and it pairs with the supervisor's own position that "CGM is continuous, so there is no need for imputation."

**2.7**
(a) loc = **180**, scale = **0** (every deviation is zero).
(b) `scale` is replaced by 1e-5, so every standardized value is 0/1e-5 = **0.0** — the input becomes all zeros.
(c) **Chronos cannot tell a flat 180 from a flat 90 or a flat 300**: all three standardize to the same all-zero input and therefore produce the identical embedding. Level information is not merely attenuated for flat series, it is completely gone.

**2.8**
(a) (1.228 − 0.478)/0.500 = 0.750/0.500 = **1.5**.
(b) (−0.400 × 0.500) + 0.478 = −0.200 + 0.478 = **0.278**.
(c) Because the three targets live on completely different scales: prices sd 17.17 (variance 294.8) versus grids sd 0.500 (variance 0.25) — a ratio of **1,179×**, call it 1,200×. An MSE loss on raw prices would start about 1,200 times larger than on raw grids, so a single learning rate of 1e-3 would be far too small for one and far too large for the other. Z-scoring the target makes one set of hyperparameters work for all three, and it makes the starting loss about 1.0 for every target (predicting 0 for a unit-variance target gives MSE 1).

**2.9**
(a) **Leak** (mild but real). The column means and sds handed to the test fold were computed partly from the test fold. Our code avoids it: the scaler is a `Pipeline` step, so `GridSearchCV` and the outer loop refit it on each training split.
(b) **Not a leak.** It uses only that one row's own inputs, no label and no other row. It is a valid per-row transform that you could apply to a brand-new session at prediction time. This is the distinction to hold onto: *leakage is about information crossing between rows or from labels to features, not about "doing arithmetic on the data."*
(c) **Not feature leakage, but it does use held-out labels.** A held-out participant's personal mean is computed from their own sessions, which all live in the test fold, so it is not available prospectively. Our own code calls this out as **oracle centering** and the results file calls it "a characterization, not a new-subject predictor." The correct framing: it answers *"is there a within-participant signal at all?"* and the answer was still no (−0.018 / −0.022 / −0.002 verified).
(d) **Leak** — the textbook one. It turns the test set into a validation set and the reported number becomes optimistic. This is exactly why there is an inner 3-fold `GridSearchCV`: alpha is chosen inside the training split, and the outer fold is used once, for reporting only.
(e) **Not a leak.** This is what `train.py` does (`target_mean = dm.target_mean` computed from `y_tr`). Using all 956 rows for those statistics *would* be a leak.

**2.10**
(a) PCA's components are *estimated from data* — they are the directions of greatest variance in whatever rows you show it. Fitting on all 956 rows lets the test fold help choose the axes the model then uses, which is leakage of exactly the same kind as (a) above. In our code PCA is a `Pipeline` step, so it sees only the training split of each fold.
(b) **Not a leak in the label sense**, for two reasons that you should give in order: the encoder is frozen and never sees a score, and each session's embedding depends on that session's own readings. **But** there is one real caveat, and volunteering it is what makes the answer strong: our encoder pools over *all* tokens without applying the padding mask, so a session's embedding does depend slightly on which other sessions happen to share its batch of 32. See PS7.10 and PS9.1. It cannot leak the train/test split (the batches are formed once, before any splitting), so it adds noise rather than optimism.

**2.11** The three (all verified, all from the *same* embeddings and the *same* grouped cross-validation):

| Ask the pipeline to predict | R² |
|---|--:|
| Glucose variability (SD) | **0.462** |
| Mean glucose | **0.070** |
| Fraction of readings above 180 | **0.075** |

Variability is recovered well; level is barely recovered at all. That is exactly what instance normalization predicts: shape survives, height is divided out. So a Chronos-based result speaks to glucose **shape**, not glucose **level** — and that is a genuine limitation of this arm, not a detail.

The number that saves the argument: the **43 hand-crafted features** *do* encode level directly (`gluc_mean`, time-in-range, and so on), and they were flat too — **−0.065 / −0.038 / −0.009**. So "no usable relationship" holds for level-aware features as well, and the Chronos limitation does not explain away the finding.

</details>

---
---

# PS3 — Metrics by hand

Use these five sessions (a made-up test fold of grids scores) for 3.1–3.5:

```
actual        y = 0.0, 0.4, 0.8, 0.6, 0.2
predicted   yhat = 0.3, 0.5, 0.5, 0.5, 0.2
```

**3.1** `[warm-up]` Compute the residuals, then MSE, RMSE and MAE.

**3.2** `[core]` Compute ȳ, SS_tot, SS_res and R².

**3.3** `[core]` Replace every prediction with the constant **0.6**. Recompute MSE, RMSE, MAE and R².

**3.4** `[core]` Replace every prediction with the constant **0.4**. What is R², exactly? Then write the general formula for R² when the prediction is a constant `c`, and check it reproduces your answer to 3.3.

**3.5** `[core]` Show that R² does not change if you (a) add 10 to every actual value and every prediction, or (b) negate every actual value and every prediction. Then say why our decision to keep all three scores in their raw "lower = better" orientation cannot possibly change any reported R².

**3.6** `[core]` `[supervisor-asked]` What exactly is the "mean baseline"? Under our grouped cross-validation, what R² does it score — and is that 0?

**3.7** `[hard]` Construct a case where RMSE improves but R² gets worse. Then find the same phenomenon in our own results table.

**3.8** `[hard]` Our grids row reports RMSE 0.505 and R² −0.089, and the overall grids sd is 0.500. But 1 − (0.505/0.500)² = −0.020, not −0.089. Explain the gap.

**3.9** `[hard]` 295 of the 951 scored grids sessions are exactly 0.000, and the grids mean is 0.478. (a) What error does a mean-predictor make on each of those sessions? (b) What share of the total variance to be explained sits in that pile at zero? (c) What does that tell you about which metric to quote?

**3.10** `[interview-killer]` The permutation test shuffled the scores 200 times and got an average R² of **−0.049** (grids). Why is that not 0? Produce a number, not just a direction.

<details>
<summary><b>Solutions — PS3</b></summary>

**3.1**

| y | ŷ | e = y − ŷ | e² | \|e\| |
|--:|--:|--:|--:|--:|
| 0.0 | 0.3 | −0.3 | 0.09 | 0.3 |
| 0.4 | 0.5 | −0.1 | 0.01 | 0.1 |
| 0.8 | 0.5 | +0.3 | 0.09 | 0.3 |
| 0.6 | 0.5 | +0.1 | 0.01 | 0.1 |
| 0.2 | 0.2 | 0.0 | 0.00 | 0.0 |
| | | | **0.20** | **0.8** |

MSE = 0.20/5 = **0.04**. RMSE = √0.04 = **0.20**. MAE = 0.8/5 = **0.16**.

RMSE is in the units of the score, which is what makes it sayable out loud: *"we are off by about 0.2 grid cells."*

**3.2** ȳ = 2.0/5 = **0.4**. Deviations −0.4, 0.0, +0.4, +0.2, −0.2 → squares 0.16, 0, 0.16, 0.04, 0.04 → SS_tot = **0.40**. SS_res = **0.20** (from 3.1). R² = 1 − 0.20/0.40 = **0.50**.

Read that as: the model removed half the squared error that guessing 0.4 every time would have left.

**3.3** Errors: −0.6, −0.2, +0.2, 0.0, −0.4 → squares 0.36, 0.04, 0.04, 0, 0.16 → SS_res = **0.60**.
MSE = 0.12, RMSE = √0.12 = **0.3464**, MAE = (0.6+0.2+0.2+0+0.4)/5 = **0.28**.
R² = 1 − 0.60/0.40 = **−0.50**.

So a constant prediction that is off-centre by 0.2 gives R² = −0.5. Negative R² needs no exotic model.

**3.4** The constant 0.4 *is* ȳ, so SS_res = SS_tot = 0.40 and R² = **exactly 0**.

General formula. For a constant prediction `c`, split the sum:
```
Σ(y − c)²  =  Σ(y − ȳ)²  +  n(ȳ − c)²
```
(the cross term vanishes because Σ(y − ȳ) = 0). Therefore
```
R²  =  1 − [SS_tot + n(ȳ − c)²]/SS_tot  =  − n(ȳ − c)² / SS_tot
```
Check against 3.3: −5 × (0.4 − 0.6)² / 0.40 = −5(0.04)/0.40 = **−0.50**. Matches.

**Keep that formula.** It is the single most useful line in this workbook: *any* constant predictor scores R² ≤ 0, and it hits 0 only if the constant is the test set's own mean. It explains 3.6, 3.10, PS5.5 and PS8.3.

**3.5** Let the transform be y → ay + b applied to actuals and predictions alike.
- Numerator: Σ((ay+b) − (aŷ+b))² = a²Σ(y − ŷ)².
- Denominator: the new mean is aȳ + b, so Σ((ay+b) − (aȳ+b))² = a²Σ(y − ȳ)².
- The a² cancels and the b never appears. R² is **unchanged**. (a) is a = 1, b = 10. (b) is a = −1, b = 0.

Why this settles the score-direction question: negating prices would flip every coefficient of a linear model and flip every prediction, so both sums scale by (−1)² = 1. R², RMSE and MAE are all identical. Only the *interpretation* changes ("higher is better" versus "lower is better"). Our `config.py` therefore keeps `PRICES_IS_INVERTED = False` and reads all three as lower = better.

**Subtlety.** The invariance requires the *predictions* to transform the same way, which they do for any model that is equivariant under shifting and scaling the target — an unpenalized-intercept Ridge is, and the mean baseline is exactly. A model with a non-negativity constraint on its output, or one that penalized the intercept, would not be, and then the claim fails. Ours is fine because `sklearn`'s Ridge leaves the intercept unpenalized.

**3.6** The mean baseline is `DummyRegressor(strategy="mean")`: it ignores the features entirely and predicts one constant. Crucially, under cross-validation that constant is the **training fold's** mean, applied to the test fold.

**It does not score 0.** Verified, grouped 5-fold, per fold:

| target | per-fold R² of the mean baseline | average |
|---|---|--:|
| grids | −0.2503, −0.1269, −0.0010, −0.0062, −0.0030 | **−0.0775** |
| symbols | −0.0301, −0.0614, −0.0173, −0.0197, −0.1229 | **−0.0503** |
| prices | −0.0026, −0.0001, −0.0060, −0.0080, −0.0054 | **−0.0044** |

By formula 3.4, that is −(ȳ_test − ȳ_train)² divided by the test fold's variance — pure evidence that the four held-out children's average score differs from the sixteen training children's average. Grids fold 1 is the extreme case at −0.25.

**Now the punchline, and it is the most useful sentence you can memorize about our results.** Arm A scores −0.089 / −0.056 / −0.012. The mean baseline scores −0.0775 / −0.0503 / −0.0044. So the *model's own* extra cost over doing nothing is only **0.012 / 0.006 / 0.008**. The negative sign is dominated by the protocol, not by the model going haywire.

**Subtlety — a real trap.** "R² = 0 means the model predicts the mean" is almost right and completely misleading. R² = 0 means the model matches a predictor that knew the *test set's* mean. Nobody has that. Under leave-participants-out cross-validation the achievable ceiling for a no-signal model is slightly below zero, and you have to know by how much.

**3.7** Two test folds, same model quality claim:

| fold | y | population variance | RMSE | R² |
|---|---|--:|--:|--:|
| A | 0, 1, 2, 3, 4 | (4+1+0+1+4)/5 = **2.00** | 1.0 | 1 − 1/2 = **+0.50** |
| B | 1.8, 1.9, 2.0, 2.1, 2.2 | (0.04+0.01+0+0.01+0.04)/5 = **0.02** | 0.5 | 1 − 0.25/0.02 = **−11.5** |

RMSE halved; R² collapsed. R² is a *ratio to how spread out the truth is*, so a fold where everyone scores nearly the same is brutally hard to get a positive R² on, no matter how small the absolute errors.

In our own table (verified): **prices** reports RMSE 17.236 with R² −0.012, our **best** R² of all nine cells, while **grids** reports RMSE 0.505 with R² −0.089. Prices' RMSE is 34 times larger and its R² is 7 times better, because the prices sd is 17.17 and the grids sd is 0.500. **RMSE cannot be compared across the three targets; R² can.** That is the whole reason we lead with R².

**3.8** Two reasons, in order of size.

1. **The R² denominator is each test fold's own variance, not the pooled variance.** A fold contains only 4 participants, and part of the pooled spread is *between* participants, so within-fold spread is smaller. Verified: pooled grids sd = 0.4999 but the mean within-fold sd = **0.4852**. Redo the calculation with that: 1 − (0.505/0.4852)² = 1 − 1.0833 = **−0.083** — most of the gap closes immediately. Sanity check with the baseline row: its mean RMSE is 0.503, and 1 − (0.503/0.4852)² = **−0.075**, against the verified averaged baseline R² of −0.0775.
2. **Averaging ratios is not the ratio of averages.** We report mean(RMSE over 5 folds) and mean(R² over 5 folds) separately, and mean(1 − MSE_k/Var_k) ≠ 1 − mean(MSE)/mean(Var). The residual couple of thousandths is this.

The lesson to state: *you cannot reconstruct our R² from our RMSE and the overall sd, because R² is fold-local.* Knowing that is the difference between quoting numbers and understanding them.

**3.9**
(a) 0.478 − 0.000 = **0.478** on each of those 295 sessions.
(b) Their contribution to the total squared error is 295 × 0.478² = 295 × 0.228484 = **67.4**. The total to be explained is 951 × 0.4999² ≈ 951 × 0.2499 = **237.6**. So the pile of exact zeros carries 67.4/237.6 = **28.4%** of all the variance in the grids target.
(c) Grids is not a smooth continuous score — it is 31% floor plus a tail. Almost a third of the "variance to explain" is the single question *"did this child score a perfect zero or not?"*, which is arguably a classification problem, and no feature we have tells us which. Practically: quote **RMSE alongside R²** for grids, mention the floor, and note that MAE (which does not square the 0.478) paints a gentler picture than RMSE does. Do not present grids as if it were normally distributed.

**3.10** Two components add up, and both are below zero.

*Component 1 — the baseline offset.* Even a perfect constant predictor loses −n(ȳ_test − ȳ_train)²/SS_tot (formula 3.4). Under a **global** permutation of the scores (which is what `run_rigor.py` does: `rng.permutation(yt)`), the participant structure is destroyed, so this reduces to pure sampling noise. With σ² = 0.25, n_test = 191, n_train = 765, N = 956:

```
Var(ȳ_test)            = (σ²/n_test)·(N−n_test)/(N−1) = (0.25/191)(765/955) = 0.001048
ȳ_test − ȳ_train        = (N/n_train)(ȳ_test − ȳ)
Var(ȳ_test − ȳ_train)  = (956/765)² × 0.001048 = 1.5618 × 0.001048 = 0.001638
R² contribution        = −0.001638 / 0.25 ≈ −0.0065
```

*Component 2 — the cost of fitting features that carry no signal.* The fixed model in the permutation test is `StandardScaler → PCA(32) → Ridge(alpha=10)`. The standard result for least-squares fitting of p directions to n rows of pure noise is that expected out-of-sample MSE inflates by roughly (1 + p/n), i.e. R² ≈ −p/n. With p = 32 and n ≈ 765:

```
−32 / 765 ≈ −0.042
```

*Total:* −0.0065 − 0.042 ≈ **−0.049**. Observed chance means (verified): **−0.049 / −0.052 / −0.051**, with sd about 0.015.

**Say it in one sentence:** *"Chance is not zero here — chance is about −0.05, because a model with 32 free directions and 765 rows pays about p/n for fitting noise, plus a little for the training mean not being the test mean. Our real result, −0.106, sits at the bad end of that chance distribution, which is why p is 1.000."*

**Honesty flag.** The p/n step is an approximation: ridge shrinkage means the *effective* number of fitted directions is below 32, and the PCA step is refit per fold. The point is the sign and the scale, not the third decimal. What is exactly verified is the observed chance distribution.

</details>

---
---

# PS4 — Cross-validation

**4.1** `[warm-up]` Why group the folds by participant at all? What question does grouped splitting answer that random session-level splitting does not?

**4.2** `[core]` Eight participants with these session counts:

| participant | A | B | C | D | E | F | G | H |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| sessions | 40 | 35 | 60 | 45 | 50 | 32 | 67 | 47 |

Build **4** grouped folds using the rule `GroupKFold` actually uses — sort participants from most sessions to fewest, then give each in turn to whichever fold currently holds the fewest sessions. Report the members, the test session count and the training session count for each fold. Then explain why the folds are not all the same size.

**4.3** `[core]` Our real run has 20 participants and 5 folds. (a) How many participants are held out per fold? (b) The verified test-fold sizes are 189, 193, 191, 191, 192. What is the naive expectation, and why is the reality different?

**4.4** `[core]` Find the leak in each protocol, and name the fix.
- **P1** Shuffle all 956 sessions, take 80% for training and 20% for testing.
- **P2** Grouped folds, but a single `StandardScaler` is fitted on all 956 rows first.
- **P3** Grouped folds; for each fold, try all five alphas on the test fold and report the best R².

**4.5** `[core]` Count the model fits in one Arm A run. The grid is: mean baseline (no parameters), Ridge with 5 alphas, SVR with 3 C values × 2 gamma values, LinearRegression with nothing to tune. Outer 5 folds, inner 3 folds, and `GridSearchCV` refits the winner on the full training split. Give the per-target total and the total for all three targets.

**4.6** `[core]` Where does XGBoost appear in that count? Answer carefully. Then compute what it *would* have added, given the grid `n_estimators {100, 300} × max_depth {3, 5} × learning_rate {0.03, 0.1}`.

**4.7** `[hard]` In nested cross-validation, exactly which data chose the alpha, and exactly which data produced the reported R²? How many different alphas can a single reported number be an average over? What is nested CV actually estimating?

**4.8** `[hard]` Arm B needs a third split, for early stopping. The verified numbers over the 5 outer folds are:

| fold | test | outer-train | train | val | batches at size 64 |
|--:|--:|--:|--:|--:|--:|
| 1 | 189 | 767 | 557 | 210 | 9 |
| 2 | 193 | 763 | 568 | 195 | 9 |
| 3 | 191 | 765 | 561 | 204 | 9 |
| 4 | 191 | 765 | 576 | 189 | 9 |
| 5 | 192 | 764 | 557 | 207 | 9 |

The code asks for `GroupShuffleSplit(test_size=0.2)`. Twenty percent of 765 is 153. Why is the validation set about 200 instead?

**4.9** `[hard]` The between-participant share of score variance is grids 0.153, symbols 0.376, prices 0.077. If we switched from grouped to session-level splitting, which target would gain the most R², and why would that gain not be evidence of a glucose-to-cognition effect?

**4.10** `[interview-killer]` With 20 participants you could do leave-one-participant-out cross-validation — 20 folds instead of 5. What changes, and is it better?

<details>
<summary><b>Solutions — PS4</b></summary>

**4.1** Because our rows are not independent: one child contributes 32 to 67 sessions, and their sessions resemble each other. If the same child appears in both training and test, the model can learn *"child 2xxxxx scores about 0.30"* and score well without any glucose-to-cognition relationship at all. Grouped splitting puts every one of a child's sessions on the same side of the line, so the reported number answers the question the grant actually asks: **"given a child we have never seen, can the glucose before a test predict the score on that test?"** Random session-level splitting answers a much weaker question: "given other sessions from this same child, can we interpolate?"

Our code keeps both and treats the gap between them as a diagnostic (`cv_scheme="group"` versus `"session"`), and there is an `assert` on every fold that the training and test participant sets are disjoint.

**4.2** Total sessions = 40+35+60+45+50+32+67+47 = **376**. Sorted descending: G 67, C 60, E 50, H 47, D 45, A 40, B 35, F 32.

| step | assign | to fold (currently smallest) | fold totals after |
|---|---|---|---|
| 1 | G 67 | F1 (0) | 67 / 0 / 0 / 0 |
| 2 | C 60 | F2 (0) | 67 / 60 / 0 / 0 |
| 3 | E 50 | F3 (0) | 67 / 60 / 50 / 0 |
| 4 | H 47 | F4 (0) | 67 / 60 / 50 / 47 |
| 5 | D 45 | F4 (47) | 67 / 60 / 50 / 92 |
| 6 | A 40 | F3 (50) | 67 / 60 / 90 / 92 |
| 7 | B 35 | F2 (60) | 67 / 95 / 90 / 92 |
| 8 | F 32 | F1 (67) | 99 / 95 / 90 / 92 |

| fold | members | test sessions | train sessions |
|---|---|--:|--:|
| 1 | G, F | 99 | 277 |
| 2 | C, B | 95 | 281 |
| 3 | E, A | 90 | 286 |
| 4 | H, D | 92 | 284 |

Check: 99 + 95 + 90 + 92 = 376. Equal folds would be 94 each. They cannot be equal, because **a participant is indivisible** — the algorithm balances greedily and gets within 5% here.

**4.3**
(a) 20 ÷ 5 = **4 participants** held out per fold.
(b) Naive expectation 956/5 = **191.2** per fold. Reality 189–193, because participants carry 32 to 67 sessions each and cannot be split. **Trap:** saying "each fold has 191 sessions" as if it were exact. The right phrasing is *"about 191, from 189 to 193, and 4 participants exactly."*

**4.4**
- **P1** — the same child lands in both halves. The model can score well by recognising children. Fix: group by participant. (We run this deliberately as a diagnostic; if grouped R² ≈ 0 while session-level R² > 0, the "signal" was participant identity.)
- **P2** — feature-statistics leakage: the column means and sds used to transform the test fold were computed partly *from* the test fold. Mild in effect, but indefensible in a write-up. Fix: put the scaler inside the `Pipeline` so it refits per fold, which is what `regression.py` does.
- **P3** — tuning on the test set. You are choosing the model using the very data you then report on, so the reported R² is optimistically biased and is no longer an estimate of out-of-sample performance. Fix: **nested** CV — choose alpha with an inner 3-fold grouped split of the training data, and touch the outer test fold exactly once.

**4.5** Per target:

```
Baseline    no grid          →  1 fit per outer fold × 5           =   5
Ridge       5 alphas         →  5×3 inner = 15, +1 refit = 16, ×5  =  80
SVR         3 C × 2 gamma=6  →  6×3 inner = 18, +1 refit = 19, ×5  =  95
Linear      no grid          →  1 fit per outer fold × 5           =   5
                                                            total  = 185
```
× 3 targets = **555 fits** per Arm A run. (Baseline and Linear skip `GridSearchCV` entirely, because the code branches on `if param_grid:`.)

**4.6** **It does not appear at all.** The XGBoost import in `regression.py` is inside a `try/except`, and xgboost is **not installed** in this environment, so the whole model was silently skipped in every run we have — no error, no warning. If asked "did you try gradient boosting?", the true answer is **no**.

Had it run: 2 × 2 × 2 = 8 settings → 8 × 3 = 24 inner fits + 1 refit = 25 per outer fold → 125 per target → 375 more fits, taking the run to 930.

**4.7** Alpha is chosen **inside the outer training split**, by splitting those ~765 sessions into 3 grouped inner folds and picking the alpha with the best average negative-MSE. The reported R² comes **only** from the outer test fold, which was never used to choose anything.

A single reported number is the average over 5 outer folds, and each fold ran its own inner search — so it can be an average over **up to 5 different alphas**, i.e. up to 5 different models. That is not a bug; it is the point. Nested CV estimates the performance of the **whole procedure** ("standardize, tune alpha by inner grouped CV, fit"), not the performance of one specific fitted model. If you want one model to ship, you re-run the tuning on all the data afterwards — and you do not get to quote the nested R² as that model's accuracy.

**4.8** Because `GroupShuffleSplit`'s `test_size` counts **groups, not rows**. There are 16 participants in an outer-training split, and `ceil(0.2 × 16) = 4`, so **4 of 16 participants (25%) go to validation** and 12 remain for training. Those 4 participants happen to carry 189–210 sessions.

This corrects the nominal arithmetic in `01_FACT_SHEET.md` ("val ≈ 153, train ≈ 612, 10 batches"). Verified: **train 557–576, val 189–210, 9 batches per epoch in every one of the 15 trainings** (5 folds × 3 targets; per-target train sizes run 538–576, all of which round up to 9).

Being able to say this is worth a lot: it is exactly the kind of detail a supervisor finds by reading the code, and "the fact sheet rounds; the splitter divides participants, so the real training folds are 557 to 576 and there are 9 batches" is a much stronger answer than repeating 612.

**4.9** **Symbols**, by a distance. 37.6% of the symbols variance is "which child this is" — the largest share of the three — so a split that lets the model see some of a child's sessions during training hands it that 37.6% almost for free. Grids (15.3%) and prices (7.7%) have far less to cash in.

Why it is not evidence of a glucose effect: the model would be exploiting **participant identity**, which the embedding partly carries because a child's glucose habits (their typical variability, their excursion pattern) are stable. Predicting "this looks like child X's glucose, and child X is slow at symbols" is not predicting the effect of a glucose fluctuation on cognition. It is why the group-versus-session gap is a *diagnostic* rather than a result.

**4.10** What changes:

1. **20 folds, each testing one child** (32 to 67 sessions).
2. **The R² denominator changes meaning.** A single child's own score variance is *within*-participant variance only — 84.7% of the grids variance, 62.4% of symbols, 92.3% of prices. So the numbers are computed against a different reference and are **not comparable** to the 5-fold ones. Do not put them in the same table.
3. **The baseline penalty gets bigger and noisier.** Formula 3.4 says the mean-predictor loses −(ȳ_child − ȳ_train)²/Var_child. A child whose average differs from the group's, or whose own scores barely vary, produces a wildly negative R². Grids fold 1 already shows this at 5 folds (−0.2503 for the baseline alone); at 20 folds it would be worse.
4. **More training data per fold** (about 908 instead of 765 sessions) — the one genuine gain.

**Is it better? No, not for reporting.** More folds is not more information: we still have only 20 independent participants, and the extra folds buy a noisier estimate against a shifting denominator. Five folds gives ~191 test sessions per fold, which is a stable enough denominator to average. Leave-one-participant-out is useful as a *diagnostic* — it shows you which children the model fails on — not as the headline number.

</details>

---
---

# PS5 — Linear models and regularization

**5.1** `[warm-up]` Write the prediction formula for a Ridge model on our Chronos features and count the unknowns it has to determine.

**5.2** `[core]` Fit an ordinary least-squares line by hand. The feature is `x = (session mean glucose − 170)/40` and the target is `y = grids score − 0.478`. Four points:

```
(−2, −0.30)   (−1, −0.10)   (+1, +0.20)   (+2, +0.20)
```
Give the slope, the intercept, the four predictions, the residuals and the training R². Then say why the training R² is not something we would ever report.

**5.3** `[core]` With centred x and y, the ridge slope has the closed form `w(α) = Σxy / (Σx² + α)`. Compute `w` for α = 0, 1, 10, 90, 990 and α → ∞. Which α exactly halves the slope here, and why that one?

**5.4** `[core]` Our alpha grid is {0.1, 1, 10, 100, 1000}, applied *after* `StandardScaler`. Roughly where does that grid sit — barely-any shrinkage, or heavy? Use the fact that after standardizing, each of the 512 columns has Σx² ≈ n.

**5.5** `[core]` As α → ∞, what does Ridge predict, and what does R² tend to? Prove it.

**5.6** `[hard]` Plain `LinearRegression` on the 512 features with about 764 training rows scores R² ≈ **−3**. (a) Is the system underdetermined? (b) What does the (1 + p/n) rule predict for a no-signal problem of this shape? (c) Reconcile (b) with −3.

**5.7** `[hard]` In the sweep, PCA with 8 to 32 components improved the mean R² from −0.052 to −0.045, and PCA with 128 components gave the improvement back. Explain both halves with one formula. Then say why no amount of PCA could have made it positive.

**5.8** `[hard]` SVR with an RBF kernel: what do `C` and `gamma` control, and what is it doing that Ridge is not? Then check something about our grid: with `StandardScaler` in front, how different are `gamma="scale"` and `gamma="auto"` on 512 features?

**5.9** `[interview-killer]` If there is no usable signal, which alpha should the inner tuner prefer, and what does that predict about the tuned model's R² compared with a fixed, lightly-regularized one? Check your prediction against two numbers we actually report.

<details>
<summary><b>Solutions — PS5</b></summary>

**5.1**
```
predicted_score  =  b  +  w1·x1  +  w2·x2  +  ...  +  w512·x512
```
**513 unknowns**: 512 weights plus one intercept `b`. If every weight came out exactly zero, the model would return `b` for every session — and the fitted `b` is the training fold's mean score. *A Ridge model with all-zero weights is the mean baseline.* That equivalence is worth carrying around; it makes 5.5 obvious.

**5.2** x̄ = 0 and ȳ = (−0.30 − 0.10 + 0.20 + 0.20)/4 = 0, so both are already centred and the intercept is 0.

```
Σxy = (−2)(−0.30) + (−1)(−0.10) + (1)(0.20) + (2)(0.20)
    = 0.60 + 0.10 + 0.20 + 0.40 = 1.30
Σx² = 4 + 1 + 1 + 4 = 10
slope = 1.30/10 = 0.13        intercept = ȳ − slope·x̄ = 0
```

| x | y | ŷ = 0.13x | residual |
|--:|--:|--:|--:|
| −2 | −0.30 | −0.26 | −0.04 |
| −1 | −0.10 | −0.13 | +0.03 |
| +1 | +0.20 | +0.13 | +0.07 |
| +2 | +0.20 | +0.26 | −0.06 |

SS_res = 0.0016 + 0.0009 + 0.0049 + 0.0036 = **0.0110**. SS_tot = Σy² = 0.09 + 0.01 + 0.04 + 0.04 = **0.18**. Training R² = 1 − 0.0110/0.18 = **0.939**.

Why we would never report it: a training R² of 0.94 from 4 points and 1 free slope is almost guaranteed. The fit was chosen *to minimise exactly this quantity*, so it is a measure of how flexible the model is, not of how well it will do on a new child. Every number in our results tables is computed on sessions from participants the model never saw.

**5.3**

| α | Σx² + α | w = 1.30/(Σx²+α) |
|--:|--:|--:|
| 0 | 10 | **0.1300** |
| 1 | 11 | **0.1182** |
| 10 | 20 | **0.0650** |
| 90 | 100 | **0.0130** |
| 990 | 1000 | **0.0013** |
| → ∞ | → ∞ | **→ 0** |

α = **10** halves the slope, because α = Σx² = 10 there: shrinkage bites in proportion to α relative to the data's own "signal energy" Σx². That is the whole intuition for what alpha means — it is not an absolute quantity, it is compared against how much variation the feature has.

Note also that the intercept is untouched at every alpha: `sklearn` does not penalize the intercept.

**5.4** After `StandardScaler` each column has population variance 1, so Σx² = n ≈ 765 for every column, and trace(XᵀX) = n·p = 765 × 512 = 391,680, giving an **average eigenvalue of n = 765**.

Compare the grid to 765:

| α | α / 765 | reading |
|--:|--:|---|
| 0.1 | 0.00013 | essentially plain least squares |
| 1 | 0.0013 | essentially plain least squares |
| 10 | 0.013 | light |
| 100 | 0.13 | moderate |
| 1000 | 1.31 | **heavy** — shrinks an average direction by more than half |

So the grid spans "basically OLS" to "mostly the mean", which is the right span to search. Note the honest caveat: with 512 correlated columns the relevant quantities are the individual eigenvalues of XᵀX, not the average, and the small ones get shrunk far harder than the big ones. That is precisely how ridge helps — it kills the unstable directions first.

**5.5** As α → ∞ every one of the 512 coefficients goes to 0 (from 5.3, `w = Σxy/(Σx² + α) → 0`), while the unpenalized intercept stays at the training fold's mean of y. **The model becomes the mean baseline.**

So R² tends to the *baseline's* R², which by formula 3.4 is
```
R² → − n(ȳ_test − ȳ_train)² / SS_tot   ≤  0
```
with equality only if the training fold's mean happens to equal the test fold's mean. Verified values: **−0.0775 (grids), −0.0503 (symbols), −0.0044 (prices)**.

**Subtlety — a trap that catches most people.** The reflex answer is "R² → 0". It does not, under grouped cross-validation. Getting this right is what lets you explain why our whole results table is negative without sounding like you are making excuses.

**5.6**
(a) **No.** 764 rows and 513 unknowns means there is a unique least-squares solution (764 > 513). But there are only **1.49 rows per unknown**, which is the real problem — a unique solution can still be a terrible one.
(b) For a no-signal problem, expected out-of-sample MSE inflates by about (1 + p/n), so R² ≈ −p/n = −513/764 ≈ **−0.67**.
(c) Observed is −3, roughly 4.5× worse. The (1 + p/n) rule assumes the columns are well-behaved and roughly independent. Ours are a **mean-pooled embedding of a single channel**, so the 512 columns are strongly correlated: XᵀX has many near-zero eigenvalues, inverting it amplifies them, and the fitted coefficients become enormous with signs that cancel. On the training fold those huge cancelling coefficients look fine; on a new participant, whose features are shifted slightly, the cancellation fails and predictions fly off. Hence −3 rather than −0.67.

That is the whole argument for regularization, and note that in our project it is a **finding we can demonstrate**, not an assumption we imported.

**5.7** One formula: the noise-fitting cost is about **−p/n**, where p is the number of directions actually fitted.

- PCA to 8–32 components: p drops from 512 to 32, so the cost drops from −512/765 ≈ −0.67 to −32/765 ≈ −0.042. Combined with ridge shrinkage, that is worth a few hundredths of R² — matching the observed −0.052 → −0.045.
- PCA to 128: p/n climbs back to 128/765 ≈ −0.17, so you pay more again.

Why it can never turn positive: PCA is an unsupervised rotate-and-truncate. It **never looks at the score**. Discarding directions can only reduce the variance of your estimate; it cannot create a relationship that is not in the features. Verified support: the strongest correlation between any simple glucose statistic and any score in this dataset is r = −0.092, i.e. **r² = 0.0085** — 0.85% of the variance. There is nothing for PCA to concentrate.

**5.8** `C` controls how much the fit is allowed to chase individual training points: large C = small tolerance for error = **less** regularization. `gamma` is the width of the RBF kernel: large gamma = each training point influences only its immediate neighbourhood, so the fitted function becomes wiggly and local; small gamma = broad, smooth influence. Together they trade off fitting the training points against smoothness. Unlike Ridge, which fits a single hyperplane in the 512 features, SVR-RBF fits a smooth **nonlinear** surface — a weighted sum of bumps centred on training points.

The check: `gamma="scale"` = 1/(p · Var(X)) and `gamma="auto"` = 1/p. With `StandardScaler` immediately in front, Var(X) ≈ 1, so both are ≈ **1/512 = 0.00195**. The two settings are nearly identical, which means our 6 SVR settings are effectively about **3 distinct models**, and half the SVR search was nearly a no-op. Spotting that is exactly the kind of code-level detail that reads as ownership.

Result context: SVR won for symbols and prices, Ridge for grids — but every winner was ≤ 0, so "linear versus nonlinear" was not the binding constraint. Nonlinearity was not what was missing.

**5.9** With no usable signal, **any** nonzero coefficient adds error on the inner validation folds, so the tuner should prefer **large alpha** — the most shrinkage the grid offers. And by 5.5, large alpha pushes the model toward the training-fold mean, whose R² is the baseline value (−0.0775 / −0.0503 / −0.0044). So the prediction is: **a tuned model should land closer to zero than a fixed, lightly-regularized one.**

Check against our two reported numbers for grids:

| model | R² (grids) |
|---|--:|
| tuned per fold, alpha ∈ {0.1…1000}, no PCA (`headtohead_real.md`) | **−0.089** |
| fixed `StandardScaler → PCA(32) → Ridge(alpha=10)` (`rigor_real.md`) | **−0.106** |
| mean baseline (verified) | **−0.0775** |

The tuned model is closer to the baseline, as predicted, and the baseline is the floor that a maximally-shrunk model approaches.

**Honesty flag.** The two models also differ by the PCA(32) step, and the selected alphas were not logged, so this is a *mechanism* consistent with the two numbers, not a controlled experiment. Say it that way. And this is the answer to "your two documents disagree" — they do not; they are two different models on the same folds.

</details>

---
---

# PS6 — Training loop arithmetic

**6.1** `[warm-up]` `[supervisor-asked]` Define in one sentence each: forward pass, loss, backward pass, optimizer step, batch, epoch.

**6.2** `[warm-up]` `[supervisor-asked]` Training fold has 557 sessions; batch size 64. Compute (a) batches per epoch; (b) the size of the last batch; (c) optimizer steps in one epoch; (d) steps if all 100 epochs run; (e) how many times each training row is used in 100 epochs.

**6.3** `[core]` The verified training-fold sizes across the 5 outer folds are 557, 568, 561, 576, 557. (a) Batches per epoch for each. (b) Total optimizer steps for one target if every fold ran the full 100 epochs. (c) For all three targets.

**6.4** `[core]` `01_FACT_SHEET.md` says "val ≈ 153, train ≈ 612, 10 batches". The verified split gives val 189–210 and train 557–576, i.e. **9** batches. Explain the discrepancy in one sentence.

**6.5** `[core]` `[supervisor-asked]` Early stopping: `monitor="val/loss"`, `patience=8`, `mode="min"`, `min_delta=0`. Here is a fold's validation loss by epoch (**illustrative** — constructed for this exercise, not logged output):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| val/loss | 1.020 | 0.960 | 0.930 | 0.905 | 0.898 | 0.902 | 0.899 | 0.905 | 0.893 |

| epoch | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| val/loss | 0.897 | 0.901 | 0.899 | 0.904 | 0.902 | 0.910 | 0.908 | 0.895 | 0.900 |

(a) Which epoch had the best validation loss? (b) After which epoch does training stop? (c) How many optimizer steps did this fold take, at 9 batches per epoch? (d) **Which epoch's weights get evaluated on the test fold in our code?**

**6.6** `[core]` Count the head's parameters layer by layer: `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`. It must total 132,609.

**6.7** `[core]` (a) Parameters per training row, using the verified training-fold range 557–576. (b) How many sessions would we need for one parameter per row, and how does that compare to our whole dataset?

**6.8** `[hard]` Dropout 0.1 on the 256-unit hidden layer. (a) How many units are zeroed per forward pass on average? (b) What happens at evaluation time, and why does PyTorch not need to rescale then? (c) Name one observable consequence.

**6.9** `[hard]` AdamW, `weight_decay=0.01`. (a) Which of the 132,609 parameters does the decay act on in our configuration? (b) How does AdamW's decay differ from adding an L2 penalty to the loss? (c) What is slightly non-standard about our setup?

**6.10** `[hard]` The target is z-scored, so a model predicting 0 for everything has a loss near 1.0. In the illustrative table above the loss reaches 0.893. (a) Roughly what fraction of the validation variance is that? (b) The test R² for Arm B grids is −0.218. Reconcile (a) and (b). (c) Was 150 optimizer steps too few?

**6.11** `[interview-killer]` Arm B is the worst of the three representations: −0.218 / −0.281 / −0.168 against Arm A's −0.089 / −0.056 / −0.012. Build the explanation from numbers, and end with a one-sentence version.

<details>
<summary><b>Solutions — PS6</b></summary>

**6.1**
- **Forward pass** — push a batch of inputs through the network and get predictions out.
- **Loss** — one number measuring how wrong those predictions are; ours is mean squared error on the z-scored score.
- **Backward pass** — compute, for every trainable parameter, the derivative of the loss with respect to it (which direction to nudge it).
- **Optimizer step** — actually nudge every parameter using those derivatives; ours uses AdamW at learning rate 1e-3.
- **Batch** — the group of examples processed together before one step; ours is 64 sessions.
- **Epoch** — **one complete pass over every training row**, i.e. as many steps as there are batches.

The compressed answer for a meeting: *"An epoch is one pass over the training data. Our training fold has about 560 sessions and the batch size is 64, so an epoch is 9 optimizer steps."*

**6.2** (a) ceil(557/64) = **9**. (b) 8 full batches use 512 rows, so the last holds 557 − 512 = **45**. (c) **9 steps**. (d) 9 × 100 = **900** steps. (e) **100 times** — once per epoch.

Note (e): "100 epochs" sounds like a lot of training and is actually 900 parameter updates. That is a useful reframe when someone asks whether the network was trained enough.

**6.3**
(a) ceil of 557, 568, 561, 576, 557 over 64 → **9, 9, 9, 9, 9**. (576/64 is exactly 9; the others round up.)
(b) 5 folds × 9 × 100 = **4,500** steps per target.
(c) × 3 targets = **13,500** steps for a full Arm B run — if nothing stopped early. Early stopping typically ends folds far sooner. This is why Arm B is cheap enough to run on a CPU: the frozen encoder is applied once and cached (`PassthroughEmbedder`), so those 13,500 steps only ever touch a 132,609-parameter head.

**6.4** Because `GroupShuffleSplit(test_size=0.2)` splits **participants, not rows**: `ceil(0.2 × 16) = 4` of the 16 outer-training participants go to validation, which is 25% of participants and about 190–210 sessions, leaving 557–576 for training — and ceil(557/64) through ceil(576/64) are all **9**.

**6.5**
Track the best-so-far and the wait counter. The counter compares each epoch to the **best so far**, not to the previous epoch, and resets only on a genuine improvement.

| epoch | val/loss | best so far | improved? | wait |
|--:|--:|--:|---|--:|
| 1 | 1.020 | 1.020 | — | 0 |
| 2 | 0.960 | 0.960 | yes | 0 |
| 3 | 0.930 | 0.930 | yes | 0 |
| 4 | 0.905 | 0.905 | yes | 0 |
| 5 | 0.898 | 0.898 | yes | 0 |
| 6 | 0.902 | 0.898 | no | 1 |
| 7 | 0.899 | 0.898 | **no** (0.899 > 0.898) | 2 |
| 8 | 0.905 | 0.898 | no | 3 |
| 9 | **0.893** | **0.893** | yes | 0 |
| 10 | 0.897 | 0.893 | no | 1 |
| 11 | 0.901 | 0.893 | no | 2 |
| 12 | 0.899 | 0.893 | no | 3 |
| 13 | 0.904 | 0.893 | no | 4 |
| 14 | 0.902 | 0.893 | no | 5 |
| 15 | 0.910 | 0.893 | no | 6 |
| 16 | 0.908 | 0.893 | no | 7 |
| 17 | 0.895 | 0.893 | **no** (0.895 > 0.893) | **8 → stop** |

(a) Epoch **9**, at 0.893. (b) Training stops **after epoch 17**, when the wait counter reaches the patience of 8. Epoch 18 never runs. (c) 17 × 9 = **153 optimizer steps**. (d) **Epoch 17's weights** — see below.

**Subtlety — two traps in one table.** Epoch 7 (0.899) looks like an improvement because it is better than epoch 6 (0.902); it is not, because 0.899 > the best of 0.898. Epoch 17 (0.895) looks like a clear improvement after the drift up to 0.910; it is not, because 0.895 > the best of 0.893. If you compare to the previous epoch instead of the best, you will get both wrong and conclude training never stops.

**And the answer to (d) is a real weakness worth volunteering.** Our trainer is constructed with `enable_checkpointing=False` and only an `EarlyStopping` callback. Lightning's `EarlyStopping` **stops** training; it does not restore the best weights (unlike Keras's `restore_best_weights`). Then `trainer.test(lm, dm)` runs on the in-memory model. So we evaluate the **epoch-17** weights — 8 epochs past the best validation loss — not the epoch-9 weights. That is a one-line fix (add a `ModelCheckpoint(monitor="val/loss")` and test with `ckpt_path="best"`), and it plausibly costs Arm B some of its R². Say it before someone finds it.

The one-sentence definition for a meeting: *"Early stopping watches the error on held-out validation sessions after every pass over the data, and halts once it has failed to improve for 8 consecutive passes — it is how we keep a 132,609-parameter head from memorizing 560 rows."*

**6.6**

| layer | shape | parameters |
|---|---|--:|
| `LayerNorm(512)` weight | (512,) | 512 |
| `LayerNorm(512)` bias | (512,) | 512 |
| `Linear(512→256)` weight | (256, 512) | 256 × 512 = 131,072 |
| `Linear(512→256)` bias | (256,) | 256 |
| `GELU` | — | 0 |
| `Dropout(0.1)` | — | 0 |
| `Linear(256→1)` weight | (1, 256) | 256 |
| `Linear(256→1)` bias | (1,) | 1 |
| | **total** | **132,609** |

Running total: 1,024 + 131,328 + 257 = **132,609**. Note that GELU and Dropout have no parameters — an activation is a fixed formula and dropout is a random mask.

**6.7**
(a) 132,609 / 557 = **238.1**; 132,609 / 576 = **230.2**. So about **230 to 240 parameters per training example**. (`01_FACT_SHEET.md` quotes 217× using its nominal 612 rows; the verified range is 230–240. Either number makes the point; know which is which.)
(b) 132,609 sessions, which is 132,609 / 956 = **138.7×** our entire dataset. To train this head with one parameter per example we would need 139 studies the size of ours.

That single ratio is the most economical explanation of Arm B's result, and it is the answer to "why not use a bigger network?"

**6.8**
(a) 10% of 256 = **25.6 units** zeroed on average per forward pass (a fresh random mask each time).
(b) At evaluation dropout is **off** — every unit passes through. No rescaling is needed at test time because PyTorch uses **inverted dropout**: during training the surviving activations are divided by (1 − p) = 0.9, so the expected magnitude already matches what evaluation will see.
(c) The same input gives **different outputs** in training mode and evaluation mode. Practical consequences: training loss and validation loss are not measured under the same conditions, so validation loss can legitimately look *better* than training loss early on; and any comparison of the two must account for it. It also means a single forward pass in training mode is a noisy estimate of the model — which is the point, as the noise is what discourages memorizing.

**6.9**
(a) In our configuration, **all of them**. `configure_optimizers` collects `[p for p in self.model.parameters() if p.requires_grad]` into a single parameter group, so the decay hits the two `Linear` weight matrices, both `Linear` biases, and the `LayerNorm` weight and bias.
(b) A classic L2 penalty adds `wd · w` to the *gradient*, which then passes through Adam's per-parameter adaptive scaling and gets divided by a running estimate of gradient magnitude — so parameters with small gradients get decayed much harder than intended, and the effective decay strength depends on the loss surface. **AdamW decouples it**: it subtracts `lr · wd · w` from the weight directly, after the adaptive step, so the decay is uniform and predictable.
(c) Standard practice is to **exclude** biases and normalization parameters from weight decay, because shrinking a LayerNorm gain or a bias toward zero is not a meaningful complexity penalty. We do not exclude them. At 132,609 parameters with 1,537 of them being biases and norms, the effect is small — but it is a real deviation and cheap to fix.

**6.10**
(a) The target is z-scored with the training fold's mean and sd, so a model outputting 0 has MSE ≈ 1 and a loss of 0.893 corresponds to explaining roughly **1 − 0.893 ≈ 11%** of the variance — *if* the validation set's spread matches the training set's. (The z-scoring uses training statistics, so on validation data MSE_z = MSE / σ²_train, and the conversion is exact only when σ_val ≈ σ_train.)
(b) The 11% was measured on the **validation participants**, which the early-stopping rule was explicitly selecting for — so it is optimistic by construction, in the same way a training score is. The test fold contains four entirely different children, and there the model scores **−0.218**: worse than guessing their average. The gap between "11% on validation" and "−22% on test" *is* the overfitting.
(c) **No — steps were not the constraint.** 153 steps was already enough to drive the validation loss down and then start drifting up; more steps would make it worse, not better. The constraint is **data**: 560 rows for 132,609 parameters.

**6.11** The argument, in order:

1. **Capacity against data.** 132,609 trainable parameters on 557–576 training rows: about **235 parameters per example**. Arm A's Ridge has 513 unknowns on the same rows — 256× fewer.
2. **Regularization strength.** Arm A's inner loop can *choose* to be almost the mean baseline, by picking alpha = 1000 (see PS5.9), and it is scored on inner grouped folds while doing so. Arm B has **no hyperparameter search at all** — its only brakes are dropout 0.1, weight decay 0.01, and early stopping with patience 8. It cannot decide to become the baseline.
3. **Our implementation makes it worse.** The tested weights are the epoch-at-stop weights, up to 8 epochs past the best validation loss (6.5d).
4. **The economics of no signal.** From PS3.10 and PS5.7, the cost of fitting p directions to n rows of noise is about −p/n. A flexible nonlinear head effectively fits far more directions than 32, so it pays far more. Every extra degree of freedom is a pure loss when there is nothing to find.

Verified outcome: **−0.218 / −0.281 / −0.168** for Arm B against **−0.089 / −0.056 / −0.012** for Arm A, against a mean baseline of **−0.0775 / −0.0503 / −0.0044**.

One sentence: *"When there is no signal to find, extra capacity buys you a worse number, not a better one — and the neural head has 235 parameters per training session with no way to shrink itself back to the average."*

</details>

---
---

# PS7 — Transformer and Chronos mechanics

**7.1** `[warm-up]` `[supervisor-asked]` What is a patch, what is a token, and how do they relate in Chronos-Bolt? Give the size of each in our units.

**7.2** `[warm-up]` State the token-count formula and evaluate it for T = 17, 36, 71, 288.

**7.3** `[core]` Inside the patcher, a series is left-padded up to a multiple of 16. Compute the padding for T = 3, 17, 36, 45, 288. Why is it added on the **left**?

**7.4** `[core]` The [REG] token. (a) Where does it sit in the sequence? (b) Where does its 512-dim vector come from? (c) How many parameters does it own? (d) What share of the mean-pooled embedding does it contribute for a 36-reading session encoded alone, and for a 288-reading session?

**7.5** `[core]` Compute one attention output by hand. Three tokens, 2-dimensional queries, keys and values. Take token 3's query.

```
q3 = (2, 1)
k1 = (1, 0)    k2 = (0, 1)    k3 = (1, 1)
v1 = (1, 0)    v2 = (0, 1)    v3 = (2, 2)
d_k = 2,  so the scaling factor is 1/sqrt(2) = 0.7071
```
Hint table: `exp(0.7071) = 2.0281`, `exp(1.4142) = 4.1132`, `exp(2.1213) = 8.3420`.

Give the raw scores, the scaled scores, the softmax weights and the output vector.

**7.6** `[core]` Redo 7.5 **without** the 1/sqrt(d_k) factor. Hint: `exp(1) = 2.7183`, `exp(2) = 7.3891`, `exp(3) = 20.0855`. Which token gains weight, and what is the scaling factor actually for?

**7.7** `[core]` Now suppose token 2 is a **padding** patch, so its attention mask is 0. Recompute 7.5. How is masking implemented, mechanically?

**7.8** `[hard]` Derive the parameter budget of `chronos-bolt-small` from its configuration: `d_model 512`, `num_layers 6`, `num_decoder_layers 6`, `num_heads 8`, `d_kv 64`, `d_ff 2048`, `input_patch_size 16`, `prediction_length 64`, 9 quantiles, `relative_attention_num_buckets 32`. Use these two facts about T5: attention and feed-forward projections have **no bias**, and T5's LayerNorm has a **weight only, no bias**. Compute:
(a) one encoder layer; (b) the whole encoder stack; (c) the input patch embedding — a residual block `Linear(32→2048) → relu → Linear(2048→512)` plus a skip `Linear(32→512)`, all with biases; (d) the output patch embedding — the same shape but `512→2048→(64×9)` with a skip `512→576`; (e) one decoder layer and the whole decoder stack; (f) the [REG] embedding table; (g) the grand total; (h) the fraction that `embed()` actually uses.

**7.9** `[hard]` Someone writes: "`embed()` uses 20,015,872 parameters; the unused decoder side is 27,703,168." Add those two numbers. What is wrong, and what is the correct split?

**7.10** `[hard]` Our encoder passes **batches of 32 windows** to `embed()`, which left-pads every series in the batch to the longest one, and then we pool with `emb.mean(dim=1)` — no attention mask applied. Consider a 36-reading session in a batch whose longest member has 288 readings. (a) How many tokens does the batch have? (b) How many of that session's tokens are pure padding? (c) What share of its pooled embedding comes from padding tokens?

**7.11** `[interview-killer]` Given 7.10, is our "no leakage" claim still true? Answer precisely, and say what you would do about it.

<details>
<summary><b>Solutions — PS7</b></summary>

**7.1**
- A **patch** is 16 consecutive glucose readings — **80 minutes** at one reading per 5 minutes — bundled with 16 observed/missing flags, so 32 numbers in total.
- A **token** is one 512-dimensional vector in the sequence the transformer processes. Chronos turns each patch into one token with a small two-layer feed-forward block (the input patch embedding), then appends one extra token, **[REG]**, which is not derived from any reading.
- The relation: **one patch in, one token out**, plus one extra token at the end. Patching is what makes a transformer affordable on long series — 288 readings become 18 tokens instead of 288, and attention cost falls by roughly 16² = 256×.

The plain-words version: *"It chops the glucose trace into 80-minute chunks, turns each chunk into one 512-number vector, and lets the model compare chunks to each other."*

**7.2** `tokens = ceil(min(T, 2048)/16) + 1`. T = 17 → 2 + 1 = **3**; T = 36 → 3 + 1 = **4**; T = 71 → 5 + 1 = **6**; T = 288 → 18 + 1 = **19**. All verified by calling `embed()`.

**7.3** `pad = (16 − T mod 16) mod 16`:

| T | T mod 16 | pad | patches |
|--:|--:|--:|--:|
| 3 | 3 | **13** | 1 |
| 17 | 1 | **15** | 2 |
| 36 | 4 | **12** | 3 |
| 45 | 13 | **3** | 3 |
| 288 | 0 | **0** | 18 |

**On the left, and this matters.** The most recent reading must stay the last value of the last patch, because the model's whole notion of "now" is *the end of the sequence*. Pad on the right and you would tell the model that the newest 13 readings were missing. Note also that Chronos has no timestamps — position in the sequence is the only time information it has.

**7.4**
(a) **Last**, appended after all the patch tokens.
(b) From a tiny embedding table (`self.shared`, vocabulary 2 × 512), looked up at index 1. It is a learned constant vector, identical for every session.
(c) 2 × 512 = **1,024** parameters.
(d) Under mean pooling each token contributes 1/(P+1) of the result. A 36-reading session encoded alone has 4 tokens, so [REG] contributes **1/4 = 25%**. A 288-reading session has 19 tokens, so **1/19 = 5.3%**.

That last point is worth stating out loud: **the same fixed vector makes up a quarter of a short session's features and a twentieth of a long one's**, purely because of window length. It is a second reason (on top of PS1.11) why the variable lookback is the weakest part of the input, and it is a direct consequence of choosing mean pooling over the token axis.

What [REG] is *for*: it gives the encoder a position that is not tied to any patch, which the attention layers can use as a scratchpad or summary slot — analogous to BERT's `[CLS]`. Chronos-Bolt's config turns it on with `use_reg_token: true`.

**7.5**

```
raw scores      q3·k1 = 2·1 + 1·0 = 2
                q3·k2 = 2·0 + 1·1 = 1
                q3·k3 = 2·1 + 1·1 = 3

scaled (÷√2)    2/1.4142 = 1.4142      1/1.4142 = 0.7071      3/1.4142 = 2.1213

exp             4.1132                 2.0281                 8.3420
sum             4.1132 + 2.0281 + 8.3420 = 14.4833

softmax         4.1132/14.4833 = 0.2839
                2.0281/14.4833 = 0.1400
                8.3420/14.4833 = 0.5760      (sums to 1.0)

output = 0.2839·(1,0) + 0.1400·(0,1) + 0.5760·(2,2)
       x: 0.2839 + 0      + 1.1521  = 1.4360
       y: 0      + 0.1400 + 1.1521  = 1.2921
```
Output ≈ **(1.436, 1.292)**.

In words: token 3's query matched token 3's own key best, so the output is dominated by v3, with a minority contribution from v1 and a small one from v2. That is all attention is — a similarity-weighted average of the value vectors.

**7.6** Unscaled scores 2, 1, 3 → exps 7.3891, 2.7183, 20.0855 → sum 30.1929 → weights **0.2447, 0.0900, 0.6652**.

```
output x = 0.2447 + 0      + 1.3305 = 1.5752
output y = 0      + 0.0900 + 1.3305 = 1.4205
```
Output ≈ **(1.575, 1.421)**.

**Token 3 gains weight** (0.5760 → 0.6652) and token 2 loses it (0.1400 → 0.0900): removing the divisor makes the softmax **sharper**. That is exactly what the factor is for. Dot products of d_k-dimensional vectors grow like d_k, so without scaling the scores get large as the model widens, the softmax saturates toward a hard pick-one, and its gradient goes to almost zero — training stalls. Dividing by √d_k keeps the scores around order 1 regardless of width. With d_kv = 64 in our model the divisor is 8, not 1.41.

**7.7** Masking is implemented by **adding −∞ (a large negative number) to the score before the softmax**, so `exp` of it is 0. Token 2 drops out and the remaining weights renormalize over tokens 1 and 3:

```
sum   = 4.1132 + 8.3420 = 12.4552
w1    = 4.1132/12.4552 = 0.3302        w3 = 8.3420/12.4552 = 0.6698
output x = 0.3302 + 1.3396 = 1.6698
output y = 0      + 1.3396 = 1.3396
```
Output ≈ **(1.670, 1.340)**. Note the mask is applied to the *keys*: other tokens do not attend **to** the padding. It does not stop the padding position from producing an output vector of its own — which is the crux of 7.10.

**7.8** With `num_heads × d_kv = 8 × 64 = 512 = d_model`, each of Q, K, V, O is a bias-free 512×512 matrix = 262,144 parameters.

**(a) One encoder layer**
```
self-attention  q,k,v,o     4 × 262,144      = 1,048,576
attention LayerNorm (weight only)            =       512
feed-forward    wi 512×2048 = 1,048,576
                wo 2048×512 = 1,048,576      = 2,097,152
feed-forward LayerNorm                       =       512
                                       total = 3,146,752
```
**(b) Encoder stack** = 6 × 3,146,752 = 18,880,512, plus the relative-attention bias table that only layer 0 owns (32 buckets × 8 heads = **256**), plus the stack's final LayerNorm (**512**) = **18,881,280**.

**(c) Input patch embedding** — in 32 (16 values + 16 flags), hidden 2048, out 512, with a skip connection:
```
Linear(32→2048)    32×2048 + 2048        =    67,584
Linear(2048→512) 2048×512 +  512         = 1,049,088
skip Linear(32→512) 32×512 +  512        =    16,896
                                   total = 1,133,568
```
**(d) Output patch embedding** — 512 → 2048 → 64 × 9 = 576:
```
Linear(512→2048)   512×2048 + 2048       = 1,050,624
Linear(2048→576)  2048×576 +  576        = 1,180,224
skip Linear(512→576) 512×576 + 576       =   295,488
                                   total = 2,526,336
```
**(e) One decoder layer** adds cross-attention:
```
self-attention  4 × 262,144  = 1,048,576   + LayerNorm 512
cross-attention 4 × 262,144  = 1,048,576   + LayerNorm 512
feed-forward                 = 2,097,152   + LayerNorm 512
                       total = 4,195,840
```
Decoder stack = 6 × 4,195,840 = 25,175,040 + 256 (relative bias) + 512 (final LayerNorm) = **25,175,808**.

**(f) [REG] / shared embedding table** = 2 × 512 = **1,024**.

**(g) Grand total**
```
        1,024   shared / [REG] table
    1,133,568   input patch embedding
   18,881,280   encoder stack
   25,175,808   decoder stack
    2,526,336   output patch embedding
   ----------
   47,718,016   total          [verified by counting the loaded checkpoint]
```
**(h)** `embed()` runs only the input patch embedding, the [REG] lookup and the encoder:
```
1,024 + 1,133,568 + 18,881,280 = 20,015,872
20,015,872 / 47,718,016 = 41.9%
```
So **we use 41.9% of the model and never execute the other 58.1%** — the decoder and the quantile output head exist to forecast, and we never forecast. That is a genuinely good thing to be able to say: our use of Chronos is as a *feature extractor*, and more than half the downloaded weights are dead code for our purpose.

**7.9** 20,015,872 + 27,703,168 = **47,719,040**, which is **1,024 more** than the true total of 47,718,016.

The double-counted 1,024 is the **[REG] / `shared` embedding table**. It is genuinely used by `embed()` (that is where the [REG] vector comes from), *and* it is registered as the decoder stack's `embed_tokens` because Chronos passes the same tensor to both stacks. Count "everything the encoder path touches" and "everything on the decoder side" separately, and you count it twice.

Correct split:

| | parameters |
|---|--:|
| used by `embed()` — input patch embedding + [REG] table + encoder | **20,015,872** |
| unused — decoder stack (25,175,808) + output patch embedding (2,526,336) | **27,702,144** |
| total | **47,718,016** |

**Subtlety — this is a trap in our own fact sheet.** The figure 27,703,168 appears there. It is what you get from `total − (encoder + input_patch_embedding)`, i.e. it silently assigns the shared table to the unused side while the 20,015,872 figure also claims it. Knowing which 1,024 parameters are being argued over is the difference between reciting a number and understanding the model.

**7.10**
(a) Every series in the batch is padded to 288 readings, so the whole batch has **19 tokens** — including the 36-reading session.
(b) 288 − 36 = 252 padding readings. Patches are cut from the left in blocks of 16: 252 = 15 × 16 + 12, so **patches 1 through 15 are entirely padding**, and patch 16 straddles the boundary (12 pad readings, 4 real). So **15 tokens are pure padding**, 3 contain real data, and 1 is [REG].
(c) 15/19 = **78.9%** of that session's pooled embedding is the average of padding-derived token vectors.

**Verified empirically.** Encode a 6-reading series alone and then in a batch alongside a 288-reading series: the mean-pooled 512-vectors differ by up to **0.87** in absolute value per coordinate. This is not a rounding effect — a short session's features genuinely change depending on its batch-mates.

Why the padding tokens are not simply zero: where the mask is 0, the patch values are zeroed, but the input patch embedding still adds its **biases**, so a padding patch maps to a constant nonzero vector; the encoder then processes it (its query is free to attend to the real, unmasked keys) and emits an output vector. Masking prevents other tokens attending **to** the padding; it does not delete the padding's own output. Then `mean(dim=1)` averages it in.

**7.11** Answer in three parts, precisely.

**Labels: no leak, unambiguously.** The encoder is frozen, runs under `torch.no_grad()`, and never sees a cognitive score. No fitted quantity crosses from test to train.

**But the strict claim "each session's embedding is a function of its own glucose alone" is false.** It depends on the longest series in its batch of 32. Batches are consecutive rows of the concatenated cohort files, which are ordered by participant, so a session's batch-mates are usually its own participant's other sessions — meaning a little participant-level information can enter a session's features by the back door.

**Does that invalidate the grouped cross-validation result? No, and here is the precise reason:** the embeddings are extracted **once, before any split**, and the batch composition is identical whether a session ends up in training or in test. So the contamination cannot carry information *about the split*, which is what leakage means. Its effect is to make short sessions' features noisier and partly arbitrary — it degrades the measurement rather than inflating the score. Given that the result is "no better than guessing the average", a noise-adding bug is a live alternative explanation for part of the flatness, which is exactly why it should be fixed before the finding is written up as final.

**What to do about it**, in order of cost: (1) pool with the attention mask — `(emb * mask).sum(1) / mask.sum(1)` — a one-line change; or (2) encode one series at a time (slower, and it changes the token count per session, which reintroduces the [REG] share problem from 7.4d); or (3) both of those *and* a uniform window length, which removes the whole class of problem at once. Option (3) is why the "fixed 2-hour window from the raw stream" item in the roadmap matters for more than one reason.

</details>

---
---

# PS8 — Reading our actual results

All numbers below are verified. Grouped 5-fold cross-validation, R² against a "predict the average" comparison, all three scores lower = better.

| target | Chronos + Ridge/SVR (Arm A) | Chronos + MLP head (Arm B) | 43 hand-crafted features | mean baseline |
|---|--:|--:|--:|--:|
| grids | −0.089 | −0.218 | −0.065 | −0.0775 |
| symbols | −0.056 | −0.281 | −0.038 | −0.0503 |
| prices | −0.012 | −0.168 | −0.009 | −0.0044 |

| target | real R² (fixed model) | shuffled mean ± sd | shuffled 95th | p |
|---|--:|--:|--:|--:|
| grids | −0.106 | −0.049 ± 0.015 | −0.027 | 1.000 |
| symbols | −0.127 | −0.052 ± 0.016 | −0.028 | 1.000 |
| prices | −0.085 | −0.051 ± 0.014 | −0.025 | 0.995 |

**8.1** `[warm-up]` Which representation had the best R² for each target? What is the best number anywhere in the table, and how should you describe "best" here?

**8.2** `[warm-up]` `[supervisor-asked]` Explain "R² = −0.089" to a clinician who has never heard of R². Give the plain version and the precise version.

**8.3** `[core]` Why is R² negative at all, if the mean baseline is supposed to be the zero point?

**8.4** `[core]` Rank the three targets by Arm A R² and explain the ordering. The between-participant shares of score variance are grids 0.153, symbols 0.376, prices 0.077.

**8.5** `[core]` Why is Arm B the worst? Two sentences.

**8.6** `[core]` `[supervisor-asked]` Someone says "your pipeline must be broken." Give the two pieces of evidence that it is not, and state precisely what each does **and does not** show.

**8.7** `[core]` Why is the shuffled mean −0.05 rather than 0?

**8.8** `[core]` Why does `rigor_real.md` report −0.106 for grids while `headtohead_real.md` reports −0.089, on the same data and the same folds?

**8.9** `[hard]` "The glucose-SD R² of 0.462 proves the embeddings contain what we need." What does 0.462 actually establish, and what is the companion number that complicates it?

**8.10** `[hard]` The subgroup table:

| subgroup | sessions | participants | grids | symbols | prices |
|---|--:|--:|--:|--:|--:|
| all | 956 | 20 | −0.089 | −0.056 | −0.012 |
| any reading < 70 | 201 | 19 | −0.187 | −0.094 | −0.090 |
| any reading > 250 | 449 | 20 | −0.077 | −0.121 | −0.087 |
| entirely 70–180 | 143 | 19 | −0.175 | −0.296 | −0.111 |

The hypoglycemia subgroup is *worse* than the full set. Give two competing explanations, say which you can rule out, and name the wording mistake this table has already caused.

**8.11** `[hard]` The encoder sweep: bolt-tiny (d = 256) −0.049, bolt-mini (384) −0.052, bolt-small (512) −0.052, bolt-base (768) −0.063, chronos-t5-small (512) −0.057. What is the pattern, what mechanism explains it, and what does it rule out?

**8.12** `[interview-killer]` Write the three sentences you would say out loud if asked "so what did you find?"

<details>
<summary><b>Solutions — PS8</b></summary>

**8.1** The **43 hand-crafted features** have the least-negative R² for all three targets: −0.065 vs −0.089 (grids), −0.038 vs −0.056 (symbols), −0.009 vs −0.012 (prices). The best single number anywhere is **−0.009** (prices, hand-crafted).

How to describe it: **"best" here means "least far below the average-guess."** Every cell is at or below zero, so nothing beat guessing the average, and the honest statement is that the 512 Chronos features and the 43 hand-crafted features are **indistinguishable from each other and from doing nothing**, with the hand-crafted set marginally closer to the baseline. Do not say the hand-crafted features "won" without that qualifier — a supervisor will hear it as "your method lost," and the correct claim is "neither method found anything."

**8.2** **Plain:** *"Our predictions were about 9% further from the truth, in squared-error terms, than if we had thrown the glucose away and written down the group's average score for every child."*

**Precise:** the mean squared error of our predictions was 1.089 times the variance of the held-out children's scores. R² = 1 − MSE/variance, so a value of 0 means "exactly as good as knowing the held-out group's average," positive means better, negative means worse.

**8.3** Because the baseline that scores exactly 0 is a predictor that knows the **test fold's own mean**, and nobody has that. Our mean baseline predicts the **training fold's** mean, and the four held-out children's average score is not the sixteen training children's average score. From PS3.4, a constant predictor `c` scores −n(ȳ_test − c)²/SS_tot.

Verified: the mean baseline itself scores **−0.0775 / −0.0503 / −0.0044**. Compare with Arm A at −0.089 / −0.056 / −0.012 and the model's *own* extra cost is only **0.012 / 0.006 / 0.008**.

**Say this, because it reframes the whole result:** *"Most of the negative sign is the price of predicting children the model has never seen, not the model misbehaving. Against the same-protocol baseline our model is within about a hundredth of R² — it neither helps nor meaningfully hurts."*

**8.4** Ranking: prices −0.012 (best), symbols −0.056, grids −0.089 (worst).

It does **not** track between-participant variance monotonically — symbols has the largest between-participant share (0.376) and sits in the middle. What it does track is the **baseline penalty**, which is what dominates:

| target | between-participant share | mean-baseline R² | Arm A R² |
|---|--:|--:|--:|
| prices | 0.077 | −0.0044 | −0.012 |
| symbols | 0.376 | −0.0503 | −0.056 |
| grids | 0.153 | −0.0775 | −0.089 |

Prices is closest to zero because its participant means are nearly identical (7.7% between-participant), so the training mean is almost the right constant for any fold and the penalty is tiny. Grids is worst because its fold means differ most in practice — driven by a single fold whose baseline R² is −0.2503.

The honest conclusion: **the ordering of the three numbers is mostly an artifact of how much the fold means differ, not a statement about which cognitive test is more predictable from glucose.** Do not build a story about grids being "harder."

**8.5** It has **132,609 trainable parameters fitted to 557–576 training sessions — about 235 parameters per example** — and unlike Arm A it has no hyperparameter search, so it cannot choose to shrink itself back toward the average; its only brakes are dropout 0.1, weight decay 0.01 and early stopping. When there is no signal, extra flexibility is a pure cost, so more capacity produces a worse number, not a better one.

**8.6**

**Evidence 1 — positive control.** The same embeddings, the same grouped cross-validation, asked to predict a property of the glucose itself: **glucose variability (SD) R² = 0.462**.
- *Shows:* the encoder produces features that carry real information about the series, and the regression, scaling and cross-validation machinery all work. A broken pipeline would score near zero here too.
- *Does not show:* anything about cognition. It also does not show that the features carry glucose **level** — mean glucose scores only 0.070.

**Evidence 2 — permutation test.** 200 shuffles of the scores, everything re-run: chance mean −0.049 / −0.052 / −0.051 with sd about 0.015, against real values of −0.106 / −0.127 / −0.085 → **p = 1.000 / 1.000 / 0.995**.
- *Shows:* our accuracy is not above chance, stated statistically rather than by eyeballing a negative number. And that the real result sits at the **bad end** of the chance range, not outside it.
- *Does not show:* that no effect exists anywhere. It shows that with 20 participants, this representation, and this lookback definition, we cannot detect one.

**One more honesty point.** `run_rigor.py` prints a "check passed" line only if **all three** glucose properties score above 0.5. Only one does (0.462 does not even clear 0.5). So that line never printed, and it is not in `rigor_real.md`. Never say the positive control "passed" outright — say **variability is recovered well and level is not**, and explain why (instance normalization).

**8.7** Chance is set by the protocol and the model's capacity, not by the number zero. Two pieces: the training-mean-versus-test-mean penalty (about −0.007 once shuffling has destroyed the participant structure), plus the cost of fitting 32 PCA directions to ~765 rows of pure noise, which is about −p/n = −32/765 ≈ −0.042. Together ≈ **−0.049**, matching the observed chance means. Consequence: **comparing our R² to 0 is the wrong test; the right comparison is to the chance distribution** — and we lose that comparison too.

**8.8** They are **two different models on the same folds**, not an inconsistency:

| document | model | grids R² |
|---|---|--:|
| `headtohead_real.md` | `StandardScaler → Ridge/SVR`, alpha tuned per fold from {0.1, 1, 10, 100, 1000}, no PCA | −0.089 |
| `rigor_real.md` | `StandardScaler → PCA(32) → Ridge(alpha = 10)`, fixed, no tuning | −0.106 |

The permutation test needs *one fixed model*, because re-running an inner grid search 200 times per target would be far more expensive and would make the chance distribution depend on the tuner's behaviour. The tuned model lands closer to zero because with no signal the tuner prefers heavy shrinkage, which pushes predictions toward the training mean and hence toward the baseline value of −0.0775 (see PS5.9).

Know this difference cold — quoting both numbers without being able to separate them looks like carelessness.

**8.9** What 0.462 establishes: the embeddings encode **glucose variability**, and the whole pipeline (frozen encoder → scaler → PCA → ridge → grouped CV → R²) recovers a real relationship when one is present. That is a meaningful negative-result credential, and most flat findings do not have it.

The companion numbers that complicate it: **mean glucose 0.070** and **fraction of readings above 180 only 0.075**. Because Chronos instance-normalizes each series (PS2), absolute level is largely divided out. So our Chronos arm speaks to glucose **shape**, not glucose **level** — and if the true physiological effect were driven by level (being at 300 versus 100), our features would be poorly suited to find it.

**Subtlety — this is the trap.** The wrong version of this answer is "the positive control passed, so the features are fine." The strong version names the limitation *and then closes it*: the **43 hand-crafted features do encode level** (`gluc_mean`, time-in-range, min, max) and they were also flat (−0.065 / −0.038 / −0.009). So the level blind-spot does not explain away the finding.

**8.10** The two explanations:
1. **There is genuinely nothing to find**, even in the regime where an effect is most physiologically plausible.
2. **It is a small-sample artifact.** With 201 sessions the test folds shrink to about 40 sessions, so each fold's R² denominator is estimated from very little and the baseline penalty from PS3.4 grows relative to the within-fold spread. A more negative number is what you would expect from noise alone.

**Which can you rule out? Neither, from the R² alone** — and saying so is the correct answer. What you *can* say is that (2) is a real and sufficient explanation for the *magnitude*, so the subgroup rows should not be read as "the effect is even more absent in hypoglycemia"; they should be read as "restricting to excursion regimes did not surface anything, and these rows are noisier than the full-sample row." Note also that the physiological framing cuts against us: only **2.31% of readings are below 70**, so we barely have the condition of interest at all.

**The wording mistake:** calling the 201 and 449 "patients." They are **sessions**; the 19 and 20 are **participants**. This has already been corrected once in writing. Get the unit right every time.

**8.11** The pattern: **flat, and bigger is worse** — bolt-base at d = 768 is the worst of the five (−0.063), and the smallest model, bolt-tiny at d = 256, is nominally the best (−0.049). All five sit in a band of 0.014, which is about one standard deviation of the chance distribution — so they are not really distinguishable from each other either.

The mechanism is the same −p/n arithmetic as everywhere else in this workbook: with no signal to capture, a wider embedding means more directions to fit noise in, so R² drifts further below zero as d grows. Wider is strictly worse when there is nothing to find.

What it rules out: **"use a bigger foundation model" is not a plausible fix.** That is the most common suggestion at this point in a project, and we have already tested it across a 3× range of widths and two model families. Same for the neighbouring sweeps: shortening the window makes it worse (2.0 h −0.076, 2.5 h −0.075, 3.0 h −0.068, full −0.052), and mean pooling beats last-token pooling (−0.052 vs −0.060).

**8.12** A usable three-sentence version:

> *"Across 956 test sessions from 20 children, glucose in the window before a test did not predict the score any better than simply guessing the group average — R² ran from −0.01 to −0.09 for the frozen-Chronos features, and the 43 hand-crafted features were the same. Two checks say that is the data rather than the code: a 200-shuffle permutation test puts our accuracy inside the chance range, p ≈ 1.0, and the same embeddings predict glucose variability at R² 0.46, so the machinery does recover a real signal when one is there. The strongest raw correlation between any glucose statistic and any score in this dataset is r = −0.09, which is 0.85% of the variance — so the flat result is what the data looked like before we modelled anything."*

Then stop. Do not volunteer "maybe with more data" unless asked; if asked, the answer is that 20 participants is likely all there is, and the one untried lever is a uniform pre-test window cut from the raw CGM stream.

</details>

---
---

# PS9 — Diagnosis and design

These have no single right answer. Write your answer, then compare with the model answer for *whether you used a verified number to support it*. That is the grading criterion.

**9.1** `[core]` Rank the weakest links in the pipeline and justify each with a number or a code fact. Separate "threatens the science" from "is an implementation defect."

**9.2** `[core]` For each statement, decide whether our results support it, and give the supported version if not.
(a) "There is no relationship between short-term glucose fluctuations and cognition in T1D youth."
(b) "The Chronos embeddings carry real information about the glucose series."
(c) "A larger foundation model would help."
(d) "Hand-crafted features beat Chronos."

**9.3** `[core]` Propose the next experiment. Justify it with a verified number, and name the honest objection to it.

**9.4** `[core]` What would you do with five more minutes of compute? With five more days?

**9.5** `[core]` What size of effect could this study detect? Answer separately for a between-participant effect and a within-participant one.

**9.6** `[hard]` Rewrite these in plain words a clinician would accept.
(a) "Frozen TSFM embeddings mean-pooled over the token axis, fed to an L2-regularized linear head under nested subject-grouped cross-validation, yield R² ≤ 0 across all three ARC subtests."
(b) "The permutation test yields p = 1.000, indicating the observed R² does not exceed the empirical null distribution."
(c) "The positive control confirms non-degenerate representation learning, though mean-level information is attenuated by instance normalization."

**9.7** `[hard]` Design a falsifiable test of the claim "this pipeline could detect a within-participant glucose-to-cognition effect if one existed." Be specific about what you would run and what result would falsify it.

**9.8** `[hard]` The supervisor asks: "Did you try gradient boosting?" Answer.

**9.9** `[interview-killer]` One paragraph: what is the scientific contribution of a study where nothing beats guessing the average?

<details>
<summary><b>Solutions — PS9</b></summary>

**9.1** Two categories, and keeping them apart is most of the answer.

**Threatens the science:**
1. **The input is not a fixed physiological lookback.** Window length runs 3 to 288 readings, a 96× spread, because the stored array is everything since that child's previous test. Supporting number: the strongest correlation with grids in the whole dataset is with the **number of readings** (r = −0.077), not with any glucose value. So the single most predictive thing about the input is the clock.
2. **20 participants.** Grouped cross-validation means the effective independent sample for any between-participant effect is **20**, not 956. See 9.5.
3. **Chronos discards level.** Verified: glucose SD recovered at R² 0.462, mean glucose at 0.070. If the real effect is driven by absolute level, these features are the wrong instrument. (Partly answered by the hand-crafted arm, which does encode level and was also flat.)
4. **Grids is 31% floor.** 295 of 951 sessions score exactly 0.000, carrying 28.4% of the target's variance in a single spike. Treating it as a smooth continuous outcome is questionable.

**Implementation defects — cheap to fix, and you should fix them before writing up:**
5. **Mask-free mean pooling over padded tokens.** For a 36-reading session batched with a 288-reading one, **78.9%** of the pooled embedding comes from padding tokens, and the verified difference between encoding a short series alone versus in a mixed batch reaches **0.87 per coordinate**. One-line fix.
6. **Arm B evaluates the wrong weights.** `enable_checkpointing=False` plus Lightning's `EarlyStopping` means the tested model is the epoch-at-stop model, up to 8 epochs past its best validation loss.
7. **XGBoost never ran.** The import is in a `try/except` and the package is absent, so a model that appears in the code was silently skipped in every run.

**How to rank them out loud:** *"For the conclusion, the binding constraints are the variable lookback and 20 participants. For the code, there are two defects I would fix first — mask-free pooling and testing the post-early-stopping weights — because until they are fixed I cannot fully rule out that some of the flatness is my own noise."*

**9.2**
(a) **Not supported.** We have 20 participants, one representation family, one lookback definition, and 2.31% of readings below 70. Supported version: *"In this sample, with these features and this protocol, we could not predict the three scores better than guessing the group average, and a 200-shuffle permutation test places our accuracy inside the chance range."*
(b) **Supported** — glucose SD at R² 0.462 under the same embeddings and the same grouped folds. Note the limit: it is information about *shape*, not level (mean glucose 0.070).
(c) **Not supported, and actively contradicted.** bolt-base at d = 768 was the worst of five checkpoints (−0.063) while bolt-tiny at d = 256 was nominally the best (−0.049).
(d) **Technically true, practically misleading.** All three hand-crafted numbers are less negative (−0.065 / −0.038 / −0.009 vs −0.089 / −0.056 / −0.012), but every number is ≤ 0 and the differences are a fraction of the chance sd (0.015). Supported version: *"Both representations are indistinguishable from guessing the average; the hand-crafted features are marginally closer to it, which is consistent with their lower dimension — 43 features fit less noise than 512."*

**9.3** **Cut a uniform, fixed-length pre-test window — say exactly 2 hours, 24 readings — from the raw continuous CGM stream, for every session.**

Justification with a verified number: the strongest single correlation with grids anywhere in the dataset is with **window length** (r = −0.077), which beats every actual glucose statistic. Length is currently a nuisance variable varying 96-fold across sessions, and it also determines the [REG] token's share of the embedding (25% for a 4-token session, 5.3% for a 19-token one). Making every input identical in length removes a whole family of artifacts at once, and it makes the input match what the grant actually hypothesizes — the glucose shortly before the test.

**The honest objection, and you must raise it yourself:** we already tried capping the stored arrays to the most recent 2 h / 2.5 h / 3 h, and it made things **worse** (−0.076 / −0.075 / −0.068 versus −0.052 for the full window). The reply is that capping is not the same operation: capping can only discard readings that were stored, and it leaves short sessions short — a session with 12 stored readings stays 12 long under a 24-reading cap. Cutting from the raw stream would give all 956 sessions exactly 24 readings. The blocker is access to the raw stream, which is an open question for Phil, not a modelling decision.

**9.4**
**Five minutes:** fix the mask-free mean pooling (`(emb * mask).sum(1) / mask.sum(1)`), re-extract the embeddings from cache, and re-run Arm A. Two things to look at: whether short sessions' embeddings move materially, and whether the R² changes at all. Either answer is useful — if nothing changes, you have retired a worry; if something changes, every earlier number needs a footnote.

**Five days:** obtain the raw CGM stream, build the uniform-window dataset (9.3), and re-run the full head-to-head plus both rigor checks on it. Secondary: install xgboost so the model that is already in the code actually runs, and add a `ModelCheckpoint` so Arm B tests its best weights.

Note what is *not* on either list: bigger encoders (ruled out by the sweep), more elaborate heads (Arm B is already the worst performer), and more hyperparameter tuning (the tuner's preferred direction is toward the baseline).

**9.5** Split the question, because the answer is completely different on each side.

**Between-participant effect** (children who run higher tend to score worse): the effective sample is **20**, one per child. A correlation of r ≈ 0.09 needs on the order of (2/0.09)² ≈ **500 independent observations** to be two standard errors from zero. We have 20. There is essentially **no power** for a between-participant effect of the size the raw correlations suggest — and no amount of session-level modelling changes that, because sessions from the same child are not independent evidence about a between-child relationship.

**Within-participant effect** (this child's bad sessions follow bad glucose): here the effective sample is closer to 956 − 20 = **936**, which is above the ~500 needed for r ≈ 0.09. So a within-participant correlation of that size **should** have shown up. It did not: the within-participant centered results are **−0.018 / −0.022 / −0.002**. That is the more informative half of our finding, and it is the one to lead with — *the within-participant association really does look near zero, and we had enough sessions to see it if it were r ≈ 0.09.*

Also worth quoting: the chance distribution has sd ≈ 0.015 in R² units, so an effect would need to move R² by roughly 2 sd ≈ 0.03 above the chance mean to be visible. An r of 0.09 delivers r² = 0.0085. That is inside the noise even before cross-validation costs are paid.

**9.6**
(a) *"We turned each child's pre-test glucose trace into 512 numbers using a time-series model that was already trained and that we did not change, averaged those numbers, and used them to predict the test score with a simple linear formula. We always tested on children the model had never seen. For all three tests, the predictions were no better than just guessing the group's average score."*
(b) *"We scrambled the scores 200 times so that any real link between glucose and score was destroyed, and re-ran the whole analysis each time. Our real result was no better than the scrambled ones — in fact it sat at the worse end of them. So whatever the model latched onto was not a real relationship."*
(c) *"As a check that the method works at all, we asked the same 512 numbers to predict something about the glucose itself. They predict how much the glucose swings quite well (R² 0.46), so the features and the code are doing their job. They barely predict the average glucose level (0.07), because the model rescales every trace before it looks at it, which throws away how high or low the glucose sits."*

The general rule: replace every noun-phrase-of-noun-phrases with a verb, and give a number instead of an adjective.

**9.7** **The test:** `cgm_tsfm/data.py`'s `generate_synthetic_data` has two independent signal knobs. `within_subject_signal` scales an effect driven by the session's own recent glucose trend `(g[-1] − g[0])`, which is roughly zero-mean *within* a participant and therefore survives within-participant centering — unlike `signal_strength`, which drives a between-participant effect that centering removes.

**What to run:** generate synthetic data with `within_subject_signal` set to a small nonzero value (and `signal_strength = 0`), extract Chronos embeddings, and run Arm A with `target_norm="center"` under the same grouped 5-fold protocol. Sweep the knob upward from very small.

**What falsifies the claim:** if R² stays at or below zero even for a large `within_subject_signal`, then the pipeline **cannot** detect the class of effect it was built to look for, and the flat real result is uninterpretable — it would tell you nothing about the biology. Conversely, if R² becomes clearly positive at some effect size, you have both validated the pipeline *and* measured its detection threshold, which converts "we found nothing" into "we can rule out effects above size X." That second sentence is a much stronger scientific claim than the first, and this experiment is how you earn it.

Why this is better than the existing positive control: glucose SD is a property *of the input*, so recovering it tests the encoder and the regressor but not the *labelling* path or the centering. This tests end to end, on the exact quantity the grant hypothesizes.

**9.8** *"No. There is an XGBoost branch in `regression.py`, but the import sits inside a `try/except` and xgboost is not installed in this environment, so it was skipped in every run — silently, with no error. I can install it; the grid is 8 settings, which would add 375 fits to a run. Based on the sweep I do not expect it to change the conclusion: a tree ensemble on 512 correlated embedding dimensions with about 765 training rows faces the same rows-per-parameter problem, and every capacity increase we tested made things worse rather than better — bolt-base at 768 dimensions was the worst checkpoint, and the trainable neural head was the worst model overall. But it is a fair question and it belongs in the run."*

Note the shape of that answer: **admit the gap first, quantify the fix, then give a reasoned expectation clearly labelled as an expectation.** Never claim you tried something the code did not execute.

**9.9** Model answer:

> The contribution is a **decision-quality answer to a pre-registered exploratory question**, delivered with the two credentials that most flat findings lack. The K01's Aim 1 asks whether raw-data learning or feature engineering better predicts real-time cognition from CGM; a colleague's feature-engineering arm already came back flat, and the natural response — "you used the wrong representation" — is exactly the branch this work closes. We ran a modern time-series foundation model through the *same* participant-grouped folds, the same three targets and the same metric, so the comparison is fair by construction rather than by argument, and it came back flat too. What makes that believable rather than merely disappointing is the pair of checks: a 200-shuffle permutation test placing our accuracy inside the chance range (p ≈ 1.0), and a positive control showing the identical pipeline recovers glucose variability at R² 0.46, so the machinery demonstrably works when there is something to find. And the finding comes with a diagnosis rather than a shrug — 20 participants, a lookback that varies 96-fold because it is defined by the previous test rather than by physiology, only 2.31% of readings below 70 where an effect is most expected, and a strongest-anywhere raw correlation of r = −0.09 explaining 0.85% of the variance. That converts "it did not work" into a bounded claim plus a concrete next step: a uniform pre-test window cut from the raw CGM stream. Reporting this saves the field from re-running the same experiment, and reporting it *with* the diagnosis tells the next person which part to fix.

</details>

---
---

# Final exam — 15 mixed questions

Closed book. Aim for 25 minutes. Write short answers; the key is below.

1. Derive 956 from the raw file sizes, in three lines.
2. A session has 45 glucose readings. How many tokens does Chronos produce, how much left-padding does the patcher add, and how long is the window in hours?
3. Give `loc` and `scale` for `150, 150, 160, 170, 170` as Chronos computes them. (Watch the divisor.)
4. What is the shape of the feature matrix, and what is one row of it?
5. A test fold has 5 sessions with scores `0.0, 0.4, 0.8, 0.6, 0.2`. A model predicts `0.6` for all of them. Give MSE, RMSE and R².
6. What does the mean baseline predict under grouped cross-validation, and what R² does it score on our grids data?
7. What is an epoch, and how many optimizer steps is one epoch in Arm B?
8. Early stopping with patience 8: validation loss best at epoch 9, then rises and dips but never beats epoch 9. Which epoch does training stop after, and which epoch's weights do *we* test?
9. Count the parameters of `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`.
10. As Ridge's alpha goes to infinity, what does R² approach? Give the value for grids.
11. Why did the 200 label shuffles average −0.049 instead of 0?
12. What does the glucose-SD R² of 0.462 prove, and what does the mean-glucose R² of 0.070 tell you?
13. Our two documents report −0.089 and −0.106 for grids. Why?
14. Name the two implementation defects in the pipeline that you would fix before writing up.
15. Someone says "so glucose has no effect on cognition in children with T1D." Correct them in two sentences.

<details>
<summary><b>Answer key — final exam</b></summary>

1. 740 + 240 = 980 raw rows; minus 24 rows with empty or unparseable glucose (23 in Cohort1, 1 in Updated_Cohort2); minus 0 rows with fewer than 3 valid readings = **956 sessions**, from 20 participants (14 + 6).

2. `ceil(45/16) = 3` patches, + 1 for [REG] = **4 tokens**. Padding = `(16 − 45 mod 16) mod 16` = 16 − 13 = **3 readings**, added on the **left**. Window = 45 × 5 = 225 min = **3.75 hours**. (Caveat worth adding: 4 tokens is what you get encoding it alone; in a batch it is padded to the batch's longest member.)

3. Sum = 800, so **loc = 160**. Deviations −10, −10, 0, +10, +10 → squares 100, 100, 0, 100, 100 = 400. Population variance = 400/5 = 80, so **scale = √80 = 8.944**. (The trap: 400/4 = 100 → 10 is the *sample* sd, which Chronos does not use.)

4. **(956, 512)**. One row is one **cognitive-test session** — the mean-pooled Chronos embedding of the glucose readings recorded before that one test. Not one child: a child contributes 32 to 67 rows.

5. Errors −0.6, −0.2, +0.2, 0.0, −0.4 → squares 0.36, 0.04, 0.04, 0, 0.16 → SS_res = 0.60. **MSE = 0.12**, **RMSE = 0.3464**. ȳ = 0.4, SS_tot = 0.40, so **R² = 1 − 0.60/0.40 = −0.50**.

6. It predicts the **training fold's** mean score for every test session, ignoring the features entirely. Verified average over the 5 grouped folds for grids: **−0.0775** (per fold: −0.2503, −0.1269, −0.0010, −0.0062, −0.0030). Not 0 — because the held-out children's average is not the training children's average.

7. One complete pass over every training row. Arm B's training folds hold 557–576 sessions at batch size 64, so `ceil(557/64) = 9` … `ceil(576/64) = 9` → **9 optimizer steps per epoch**, in every one of the 15 trainings.

8. The wait counter hits 8 at epoch **17**, so training stops after epoch 17 and epoch 18 never runs. Because `enable_checkpointing=False` and Lightning's `EarlyStopping` does not restore the best checkpoint, we test the **epoch-17** weights — 8 epochs past the best validation loss. That is a defect, not a design choice.

9. LayerNorm 512 + 512 = 1,024; Linear(512→256) 131,072 + 256 = 131,328; GELU 0; Dropout 0; Linear(256→1) 256 + 1 = 257. Total = **132,609**.

10. All 512 coefficients go to 0 while the unpenalized intercept stays at the training mean, so the model becomes the mean baseline and R² approaches the **baseline's** R² — **−0.0775** for grids. Not 0.

11. Chance is below zero for two reasons that add: the training-mean-versus-test-mean penalty (about −0.007 once shuffling destroys the participant structure) plus the cost of fitting 32 PCA directions to ~765 rows of noise, about −p/n = −32/765 ≈ −0.042. Total ≈ −0.049. Therefore the correct comparison for a real R² is the chance distribution, not 0.

12. 0.462 proves the embeddings encode real information about the glucose series and that the whole pipeline recovers a genuine relationship when one exists — so the flat cognition result is about the data, not broken code. 0.070 for mean glucose tells you the embeddings barely encode absolute **level**, because Chronos instance-normalizes each series; our Chronos result therefore speaks to glucose shape. The rescue is that the 43 hand-crafted features do encode level and were flat too (−0.065 / −0.038 / −0.009).

13. Different models, same data and same folds. `headtohead_real.md` tunes alpha per fold from {0.1, 1, 10, 100, 1000} with no PCA → −0.089. `rigor_real.md` uses one fixed `StandardScaler → PCA(32) → Ridge(alpha=10)` so the permutation test has a single model to repeat 200 times → −0.106. With no signal the tuner prefers heavy shrinkage, which pushes the tuned model toward the baseline (−0.0775) and hence closer to zero.

14. (i) **Mask-free mean pooling**: `emb.mean(dim=1)` averages over padding-derived tokens, so for a 36-reading session batched with a 288-reading one, 78.9% of the pooled vector comes from padding, and the same short series encoded alone versus in a mixed batch differs by up to 0.87 per coordinate. (ii) **Arm B tests the post-early-stopping weights** rather than the best-validation-loss weights. Honourable mention: xgboost is in the code but not installed, so it never ran.

15. *"That is stronger than what we can say. What we found is that in this sample — 20 children, 956 sessions — the glucose in the window before a test does not predict the score better than guessing the group average, and that holds for a foundation-model representation, for 43 hand-crafted glycemic features, and after removing each child's personal baseline. The design limits are real: the lookback varies 96-fold because it is defined by the previous test rather than by physiology, and only 2.31% of readings are below 70, which is where an effect is most expected — so the honest claim is that we could not detect an effect of the size the raw correlations suggest, not that no effect exists."*

</details>

---

## Where to go next

- Numbers you could not recall: **`01_FACT_SHEET.md`**.
- Concepts that did not click: **`02_ML_FROM_ZERO.md`** (X, y, overfitting, parameters vs hyperparameters), **`03_PREPROCESSING.md`** (parsing, windowing, normalization), **`04_LINEAR_MODELS.md`** (Ridge, SVR), **`05_EVALUATION.md`** (R², cross-validation), **`06_DEEP_LEARNING_CORE.md`** (the head, the training loop), **`07_TRANSFORMERS.md`** (attention, patches, tokens), **`08_CHRONOS.md`** (the checkpoint internals), **`09_THE_TWO_ARMS.md`** (Arm A vs Arm B), **`10_RIGOR_AND_STATS.md`** (permutation test, positive control).
- Rapid-fire recall under time pressure: **`12_EXAM_DRILL.md`**.

If you have three hours left before the meeting, do this: PS1, PS3 and PS6 cold (they are the arithmetic you will be asked to do live), then read the solutions to PS8 twice, then say 8.12 out loud until it comes without effort.
