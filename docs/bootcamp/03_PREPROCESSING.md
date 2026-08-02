# 03 · Normalization and Preprocessing — the chapter Liuyi asked for

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you.**
> 1. A complete, correct answer to *"what is normalization and why must you process data before training?"* — the question you could not answer.
> 2. The single fact the rest of the repo gets wrong: Chronos-Bolt **centers and divides by standard deviation**; it does not divide by average magnitude.
> 3. One table and one diagram showing that this project has **four different normalizations at four different stages** — the thing no existing doc says in one place.
> 4. Worked arithmetic for every claim, so you can do it on a whiteboard.
> 5. The leakage rule, plus the exact line of code that enforces it.

Cross-references: exact dataset numbers are in [`01_FACT_SHEET.md`](01_FACT_SHEET.md); the general ML vocabulary (loss, R², cross-validation) is in [`02_ML_FROM_ZERO.md`](02_ML_FROM_ZERO.md); Chronos internals are in [`08_CHRONOS.md`](08_CHRONOS.md); the two model arms are in [`09_THE_TWO_ARMS.md`](09_THE_TWO_ARMS.md). Every number below was recomputed on 2026-07-30 from the real CSVs, the installed `chronos` 2.3.1 source, and `cgm_tsfm/`. Nothing is remembered.

---

## 0. The 30-second answer

**Normalization means rescaling numbers so that different quantities can be compared on the same footing.** The most common recipe is the z-score: subtract the average, divide by the standard deviation. Afterwards every quantity has average 0 and spread 1, and the original units (mg/dL, seconds, percent) are gone.

**Why you must do it before training:** almost every learning algorithm treats "1 unit" as meaning the same thing in every column. Glucose is measured in hundreds of mg/dL; time-in-range is a number between 0 and 1. Hand both to the same algorithm untouched and glucose silently gets hundreds of times more influence — not because it matters more, but because of the unit it happens to be written in. Preprocessing removes that accident. **And the rule that makes it honest:** compute the average and the standard deviation from the **training data only**. If you compute them from all the data, information about the held-out participants leaks into training, and your reported accuracy becomes a lie.

---

## 1. THE CENTRAL POINT: four normalizations, four stages

This project normalizes data **four separate times**, in four different places, with four different sets of statistics. Older docs in this repo describe them one at a time and contradict each other. Here they are together.

```
  RAW GLUCOSE                                            THE SCORE
  956 arrays of mg/dL, e.g. [116, 115, 112, ...]         one number per session
        |                                                       |
        |  we do NOTHING here: no scaling, no gap-filling        |
        v                                                       |
 +-------------------------------------------------+            |
 | STAGE 1  inside Chronos-Bolt  (chronos_bolt.py) |            |
 |   InstanceNorm: per SERIES, over TIME           |            |
 |   loc = nan-aware MEAN,  scale = pop SD (ddof=0)|            |
 |   x <- (x - loc) / scale                        |            |
 |   loc & scale are RETURNED -- and we throw them |            |
 |   away at encoders.py:104                       |            |
 +-------------------------------------------------+            |
        |  512 numbers per session                              |
   +----+------------------------------+                        |
   v  ARM A (sklearn)                  v  ARM B (neural head)   |
 +--------------------------+  +---------------------------+    |
 | STAGE 2  StandardScaler  |  | STAGE 4  LayerNorm(512)   |    |
 | per FEATURE, across the  |  | per EXAMPLE, across the   |    |
 | TRAIN ROWS only          |  | 512 features              |    |
 | regression.py:116        |  | NO dataset statistics ->  |    |
 | in a Pipeline -> no leak |  | cannot leak. model.py:147 |    |
 +--------------------------+  +---------------------------+    |
   |                              |     +---------------------- +
   |                              |     v
   |                              |  +-------------------------------+
   |                              |  | STAGE 3 (Arm B only)          |
   |                              |  | target z-scored with TRAIN    |
   |                              |  | mean/std (lightning:67), then |
   |                              |  | predictions converted BACK to |
   |                              |  | real units before any metric  |
   |                              |  | (lightning:70)                |
   v                              v  +-------------------------------+
  RMSE / MAE / R2 in the score's own units
```

### The table to memorize

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| **Name** | Instance normalization | Standardization (z-score) | Target z-score | Layer normalization |
| **What it acts on** | one glucose series | the 512 Chronos features | the cognitive score | the 512 features of one example |
| **Direction it averages over** | time (within one session) | rows (across sessions) | rows (across sessions) | features (within one row) |
| **Who does it** | Chronos-Bolt, internally | `sklearn.StandardScaler` | our Lightning module | `torch.nn.LayerNorm` |
| **Statistics used** | that series' own mean + SD | training-fold mean + SD per feature | training-set mean + SD of the score | that one example's own mean + SD |
| **Uses dataset statistics?** | No — per series | **Yes** | **Yes** | No |
| **Could it leak?** | No | Yes if done wrong — prevented by `Pipeline` | Yes if done wrong — train-only by construction | Never |
| **Where in the code** | `chronos/chronos_bolt.py:95-134`, called at line 288 | `cgm_tsfm/regression.py:116` | `ben_adapter/lightning_module.py:67,70`; stats from `datamodule.py:57-61` | `ben_adapter/model.py:147` |
| **Do we control it?** | No | Yes | Yes | Yes |

**Say it in your own words:** "There are four normalizations. Chronos normalizes each glucose series over time by itself. We standardize the 512 features across the training rows. Arm B also z-scores the score. And the neural head layer-normalizes each example across its own features."

**Drill**
- *Q: How many times is data normalized in your pipeline?* Four. Stage 1 inside Chronos over time; Stage 2 across training rows per feature; Stage 3 on the target in Arm B; Stage 4 LayerNorm inside the head.
- *Q: Which of the four could leak information from held-out participants?* Only Stages 2 and 3, because only they use statistics computed across rows. Stage 2 is inside an sklearn `Pipeline`, so its statistics are refit on each training fold. Stage 3 takes its mean and standard deviation from the training split only (`datamodule.py:57-61`).
- *Q: Which normalization do you not control at all?* Stage 1. It is baked into the frozen Chronos weights. If we wanted to disable it we would have to modify the model.

---

## 1A. The factual correction: how Chronos-Bolt actually scales

**Several documents in this repository are wrong about this and you must not repeat them.** The wrong claim appears at `docs/06_CHRONOS_INPUT_FORMAT.md:132` and `:143`, `docs/01_PIPELINE_DESIGN.md:46`, `cgm_tsfm/encoders.py:10`, `results/README.md:78`, and `PLAIN_ENGLISH_SUMMARY.md:153`. They all say Chronos "mean-scales each series internally (divides by its own average magnitude)".

That describes **Chronos-T5**, a different model family. `chronos/chronos.py:177,180`:

```python
scale = torch.nansum(torch.abs(context) * attention_mask, dim=-1) / torch.nansum(attention_mask, dim=-1)
scaled_context = context / scale.unsqueeze(dim=-1)      # no subtraction anywhere
```

That is `scale = mean(|x|)` with **no centering**. We do not use that model by default.

**We use `amazon/chronos-bolt-small`, and Bolt uses instance normalization — it centers AND scales.** `chronos/chronos_bolt.py:95-134`:

```python
class InstanceNorm(nn.Module):                    # eps = 1e-5, use_arcsinh = False
    def forward(self, x, loc_scale=None):
        if loc_scale is None:
            loc   = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        scaled_x = (x - loc) / scale
        return scaled_x.to(orig_dtype), (loc, scale)
```

Read it slowly. `loc` is the **NaN-aware arithmetic mean over time** (`dim=-1` is the time axis). `scale` is the square root of the mean squared deviation, i.e. the **population standard deviation, ddof = 0** — divide by n, not n−1. The transform is exactly `(x − loc) / scale`: the z-score, along time, within one series. Two guards: a perfectly flat series has `scale == 0` so line 113 substitutes `eps = 1e-5`; an all-missing series gets `loc = 0.0, scale = 1.0`. `use_arcsinh` is `False` in our checkpoints, so nothing else happens. It is called once per forward pass at line 288, and the comment where it is constructed (line 214) says it plainly: *"instance normalization, also referred to as 'scaling' in Chronos and GluonTS"*.

### Verified by running it

The first eight readings of session 1 of the real data: `116, 115, 112, 107, 101, 97, 94, 93`.

```
sum = 835,  n = 8,  mean = 104.375
variance(ddof=0) = 76.984375  ->  sqrt = 8.774074   <- population SD
                                  ddof=1 would be    9.37988
```

`pipeline.embed()` returned `loc = 104.375000` and `scale = 8.774073600769043`. That matches ddof=0 to seven digits and is clearly **not** ddof=1. If asked "which standard deviation", the answer is **population, dividing by n**, and you have checked it. Three more checks that were run:

| Input | returned `loc` | returned `scale` | mean-pooled embedding vs the original |
|---|--:|--:|---|
| a 16-reading window | 110.0625 | 13.7680 | — |
| the same window **+ 60 mg/dL** | 170.0625 | 13.7680 | **identical**, max element-wise difference `0.0` |
| the same window with **doubled amplitude** | 110.0625 | 27.5361 | **identical**, max element-wise difference `0.0` |
| a flat series (all 150) | 150.0 | **1e-05** | — (this is the `eps` guard firing) |

**Say it in your own words:** "Chronos-Bolt z-scores each series along time using its own mean and its own population standard deviation. Older notes in our repo describe Chronos-T5, which only divides by average magnitude and does not subtract the mean. I checked the source and ran it."

**Drill**
- *Q: Do you normalize the glucose before feeding Chronos?* No. Chronos-Bolt does it internally, per series. We pass raw mg/dL.
- *Q: Exactly what does it compute?* `(x − mean) / population_sd`, along time, for each series independently, at `chronos_bolt.py:95-134`.
- *Q: What if a child's glucose never moves during a session?* The standard deviation is 0, so line 113 substitutes `eps = 1e-5`. I confirmed it: a constant series returns `scale = 1e-05` exactly.
- *Q: Is that the same as what Chronos-T5 does?* No. T5 divides by the mean absolute value and never subtracts a center (`chronos.py:177`). Bolt centers and scales.

---

## 1B. The consequence: the embedding cannot see absolute glucose

This is the most important downstream fact in the whole project, and it follows directly from the arithmetic above.

If the transform is `(x − mean) / sd`, then **the mean and the standard deviation are divided out of the input**. Anything the encoder produces afterwards describes the **shape** of the series — its ups and downs, its rhythm, its turning points — expressed in units of its own spread. It is largely blind to **how high the glucose actually was** (the mean is subtracted away) and to **how much it actually swung** in mg/dL (the standard deviation is divided away).

`embed()` literally hands those two numbers back to the caller. Its docstring says so: *"the loc_scale, i.e., the mean and std of the original time series."* And our code throws them on the floor: `cgm_tsfm/encoders.py:104` is `emb, _ = self.pipeline.embed(batch)` — that `_` **is** `(loc, scale)`, discarded. The same discard happens in Arm B at `ben_adapter/model.py:72`.

### This explains a measured result

We ran the identical pipeline — same embeddings, same folds, same models — but asked it to predict a property of the glucose itself instead of a cognitive score (`results/rigor_real.md`):

| What we asked the 512 numbers to predict | Best R² |
|---|--:|
| glucose **variability** (the standard deviation of the session's readings) | **0.462** |
| **mean** glucose | **0.070** |
| percent of readings above 180 | 0.075 |

Variability is recovered fairly well; absolute level is barely recovered at all. That is not a bug — it is exactly what the formula predicts. The encoder reads shape, and shape still carries hints about how jagged a series is even after its overall size is normalized away, but it carries almost nothing about where the series sat on the mg/dL axis.

**Why this matters scientifically.** The clinical hypothesis is about hypoglycemia and hyperglycemia — that is, about **absolute level**. Our Chronos features are the wrong tool for that part of the question, by construction. The reason the overall conclusion still stands is that the comparison arm — 43 hand-crafted features including `gluc_mean`, `gluc_time_in_range`, `gluc_time_below`, `gluc_max` — *does* encode absolute level explicitly, and it was also no better than guessing the average (grids −0.065, symbols −0.038, prices −0.009). So "no usable relationship" holds for level-aware features too. **The obvious improvement to propose if asked "what would you do next?":** stop discarding `loc` and `scale`; concatenate them onto the 512 as two extra features. That is a two-line change to `encoders.py` and it restores absolute level to the feature set at zero cost.

**Say it in your own words:** "Because Chronos subtracts each series' mean and divides by its standard deviation, my 512 features describe the *shape* of the glucose curve, not how high it was. That is why the same features predict glucose variability at R² 0.462 but mean glucose at only 0.070. Chronos actually returns the mean and the standard deviation and my code discards them — adding them back is the first fix I would make."

**Drill**
- *Q: Your model can't tell a child at 250 mg/dL from one at 100. Isn't that fatal?* It is a real limitation for the level part of the hypothesis. Two things keep the conclusion intact: the hand-crafted arm does encode level and was also flat, and the fix is available — Chronos returns the mean and the standard deviation, we just discard them.
- *Q: Which of the three "predict glucose from its own embedding" numbers is the weak one and why?* Mean glucose, 0.070. Because the mean is subtracted out at `chronos_bolt.py:111` before the encoder ever sees the series.
- *Q: Then how did variability score 0.462 if the standard deviation is divided out?* Dividing by the standard deviation removes the *magnitude* of the swings, but the normalized shape still differs systematically between calm and volatile series — for example how many turning points there are per patch, and how the values distribute inside a patch. Also, series length is not normalized away, and length correlates with volatility here. So a fair amount of the information survives. Do not overclaim: 0.462 is good, not near-perfect.

---

## 2. Why preprocessing exists at all: three failure modes with numbers

The general rule: **most algorithms assume "one unit" means the same amount of importance in every column.** Here are the three ways that assumption breaks, each with arithmetic from our actual feature set. Our comparison arm has **43 hand-crafted features** on wildly different scales, measured across the 956 real sessions:

| Feature | mean | standard deviation | min | max |
|---|--:|--:|--:|--:|
| `gluc_gvp` (a percent) | 273.275 | **204.543** | 0.00 | 1120.60 |
| `gluc_mean` (mg/dL) | 181.696 | 69.540 | 45.33 | 400.00 |
| `gluc_j_index` (a squared quantity) | 54.298 | 38.637 | 2.62 | 181.96 |
| `gluc_n_readings` (a count) | 71.345 | 64.134 | 3.00 | 288.00 |
| `gluc_slope` (mg/dL per reading) | 0.031 | **2.931** | −17.55 | 12.72 |
| `gluc_time_in_range` (a fraction) | 0.550 | 0.355 | 0.00 | 1.00 |
| `gluc_time_below` (a fraction) | 0.026 | **0.085** | 0.00 | 1.00 |

**The largest column standard deviation is 2,396 times the smallest** (`gluc_gvp` 204.543 vs `gluc_time_below` 0.085). The whole matrix spans −326 to 1120.6. `gluc_j_index` is defined as `0.001 * (mean + sd)^2` (`handcrafted.py:108`), which is why it lands in the tens while `gluc_slope` sits near zero.

### (a) Distance-based models: the big column swallows the small one

SVR with an RBF kernel does not look at your features directly. It computes a **distance** between every pair of sessions and turns that into a similarity: `similarity = exp(-gamma * squared_distance)`. Squared distance is the sum over features of the squared difference.

Two sessions, using two features. Session A: `gluc_mean` 120, `gluc_time_in_range` 1.00. Session B: `gluc_mean` 260, `gluc_time_in_range` 0.20.

```
RAW      squared distance = (260-120)^2 + (0.20-1.00)^2 = 19600 + 0.64 = 19600.64
         gluc_mean's share         = 19600 / 19600.64 = 99.9967 %
         gluc_time_in_range's share =  0.64 / 19600.64 =  0.0033 %

SCALED   divide each by its own SD (69.5397 and 0.35477):
         squared distance = (140/69.5397)^2 + (0.80/0.35477)^2 = 4.0522 + 5.0859 = 9.1381
         shares: gluc_mean 44.35 %,  gluc_time_in_range 55.65 %

KERNEL   with gamma = 0.5:  raw exp(-0.5*19600.64) = 0.000e+00  <- underflows to exactly 0
                            scaled exp(-0.5*9.1381) = 0.0104    <- a usable similarity
```

Session B spent 80 percentage points less time in the healthy range — clinically enormous — and raw, it contributes **three thousandths of one percent** of the distance. The model cannot see it. Worse, the raw kernel returns exactly zero for essentially every pair, meaning "these two sessions are infinitely far apart", and the model degenerates into predicting a constant.

**Honest footnote you should volunteer.** sklearn's default `gamma="scale"` is `1 / (n_features * X.var())`, so it partly rescues you automatically. Measured on our matrices: `gamma="scale"` is `2.08e-06` on the raw 43 features and `2.33e-02` after standardizing — sklearn silently shrank gamma by a factor of 11,180 to compensate. That is why our measured "SVR without a scaler" numbers were not catastrophic. Do not claim the disaster is guaranteed; claim the mechanism, and note that sklearn hides part of it.

### (b) Penalty-based models: one alpha, unequal shrinkage. This is the load-bearing reason for us.

Ridge regression does not just minimize error. It minimizes

```
sum over rows of (prediction - truth)^2   +   alpha * sum over features of coefficient^2
```

There is **one** `alpha` for **all** coefficients (`regression.py:43`: `alpha in {0.1, 1, 10, 100, 1000}`). The penalty is on the coefficient values, and the size of a coefficient depends entirely on the **unit of its feature**. A feature measured in hundreds needs only a tiny coefficient to move the prediction; a feature measured in fractions needs a large one. The penalty then hits the large coefficient hard and leaves the tiny one alone — so the penalty is applied unequally, purely because of units.

Measured demonstration. A simulated dataset where **both** features genuinely matter, `y = 0.004*gluc_mean + 0.5*time_in_range + noise`, 400 rows:

| | `gluc_mean` coefficient | `time_in_range` coefficient |
|---|--:|--:|
| **Unscaled**, alpha = 1 | 0.003716 | 0.516733 |
| **Unscaled**, alpha = 100 | 0.003834 | 0.092937 |
| change | **+3.2 %** (not shrunk at all) | **−82.0 %** (crushed) |
| **Standardized**, alpha = 1 | 0.239046 | 0.123382 |
| **Standardized**, alpha = 100 | 0.193079 | 0.101866 |
| change | −19.2 % | −17.4 % |

The mechanism in one line: at alpha = 1 the penalty term is `0.003716^2 + 0.516733^2`. The `time_in_range` coefficient contributes `0.516733² / 0.003716² = 19,337` times more to the penalty than the `gluc_mean` coefficient does. So when you raise alpha, essentially all of the shrinkage lands on `time_in_range`. After standardizing, both features shrink by roughly the same 17–19 percent, which is what "one penalty for all" was supposed to mean.

**The unit-dependence, made explicit.** Fit Ridge with alpha = 100 twice on the same information — once with glucose in mg/dL, once in mmol/L (divide by 18.0182):

```
mg/dL   predictions: 0.980850  1.071809  0.884268
mmol/L  predictions: 0.980332  1.069805  0.885088
difference:          0.000518  0.002004 -0.000820      <- the answer changed
```

After `StandardScaler`, the two are identical to the last decimal place (difference exactly `0.`). Plain ordinary least squares with no penalty is also identical either way (difference exactly `0.`) — **it is the penalty, not the linear model, that introduces the unit-dependence.** That is the crisp version of this whole section.

**Measured on our real data.** Same folds, same grids, only the scaler removed:

| Features | Score | Ridge with scaler | Ridge **without** |
|---|---|--:|--:|
| 43 hand-crafted | grids | −0.0654 | −0.0790 |
| 43 hand-crafted | symbols | −0.0379 | −0.0678 |
| 43 hand-crafted | prices | −0.0086 | −0.0394 |
| Chronos 512 | grids | −0.0885 | −0.0706 |
| Chronos 512 | symbols | −0.0918 | −0.0539 |
| Chronos 512 | prices | −0.0724 | −0.0082 |

Read that honestly, both ways. On the hand-crafted features — the ones on wildly mismatched units — **the scaler helps every time**, and by a lot on prices (−0.039 to −0.009). On the Chronos embeddings the scaler slightly **hurts**. Why: every Chronos output already lives inside [−1.108, 1.197], so no dimension dominates by magnitude; but their per-column standard deviations range from `1.4e-06` to `0.316`, a ratio of 226,000. `StandardScaler` stretches those near-constant, near-useless dimensions up to standard deviation 1, amplifying noise. That is a genuinely useful thing to be able to say: **scaling is a fix for mismatched units, not a free win, and on already-comparable features it can cost you.**

### (c) Gradient descent: a long narrow valley

Arm B trains a neural head with AdamW at learning rate 1e-3 (`01_FACT_SHEET.md` section H). Gradient descent takes a step proportional to the slope, using **one** learning rate for **all** directions. If one direction is 1000 times steeper than another, no single step size works: a step small enough to be stable in the steep direction is far too small to make progress in the flat one.

```
  UNSCALED: a long thin valley             STANDARDIZED: nearly circular
  (one feature in the hundreds,
   one in fractions)

 w2 ^   .-------------------------.        w2 ^     .-'''-.
    |  |  * <- start              |           |   .'   x   `.
    |  |   ,-----------------.    |           |  |   *---->  |
    |  |  |    x <- minimum   |   |           |   `.       .'
    |  |   `-----------------'    |           |     `-,,,-'
    |   `-----------------------'             |
    +--------------------------------> w1     +------------------> w1
    Steps zig-zag across the narrow           The steepest direction
    direction and creep along the long        points at the minimum.
    one. Hundreds of steps to move.           One learning rate fits all.
```

The technical name for "how stretched the valley is" is the **condition number**: the ratio of the largest to the smallest curvature of the loss surface. Measured on our real 43-feature matrix:

```
raw 43 features:  largest curvature 7.0253e+04, smallest 1.4566e-11 -> ratio 4.8230e+15
standardized:     largest           15.2008,    smallest 3.4792e-04 -> ratio 4.3690e+04
```

`4.82e15` is past the precision of 64-bit floating point — the raw matrix is numerically almost singular. Standardizing takes it to `4.37e04`, eleven orders of magnitude better, which is merely difficult.

*(Note: Arm B's head never sees raw features; it sees Chronos embeddings and immediately LayerNorms them. This failure mode is therefore mostly about the hand-crafted arm and about why the LayerNorm in stage 4 is there at all.)*

### Where scaling is NOT needed

**Tree-based models do not care.** A decision tree asks questions of the form "is `gluc_mean` greater than 180?" It picks the threshold by sorting that one column and trying split points. Rescaling a column by any increasing function (multiply by 18, take a log, whatever) changes the *number* in the threshold but not the *order* of the rows, so the same split is found and the same predictions come out. This covers decision trees, random forests, and gradient boosting including XGBoost.

**Be honest here, because a code-reader will catch you.** `regression.py:171-179` wraps the XGBoost import in `try: ... except Exception: pass`, and **xgboost is not installed in this environment**. So XGBoost was silently skipped in every result we have. If asked "did you try gradient boosting?", the true answer is **no** — the code would have if the package were present, but it was absent. Do not say you tried it.

**Say it in your own words:** "Three reasons. Distance models let the biggest-unit column dominate the distance. Penalized models apply one penalty to all coefficients, so the shrinkage depends on the units — on my simulated check, raising alpha from 1 to 100 crushed the fraction-valued feature by 82 percent and left the mg/dL feature untouched. And gradient descent needs roughly equal curvature in every direction; on my 43 raw features the curvature ratio was 4.8e15, which standardizing brought down to 4.4e4. Tree models are the exception — they only use the ordering, so scaling is irrelevant to them."

**Drill**
- *Q: Which reason matters most for your project?* The penalty one. Ridge is our main Arm A model and its inner search tunes alpha over five orders of magnitude. Without scaling, that single alpha means different things for different features, and the fitted answer depends on whether glucose is written in mg/dL or mmol/L. I measured that: the predictions differ by up to 0.002 unscaled and are bit-identical after scaling.
- *Q: Prove that scaling actually changed a result of yours.* Ridge on the 43 hand-crafted features predicting prices: R² −0.0394 without the scaler, −0.0086 with it. Same folds, same alpha grid.
- *Q: Give me a case where scaling made things worse.* Ridge on the Chronos 512 predicting prices: −0.0082 without the scaler, −0.0724 with it. The embedding dimensions are already all inside [−1.1, 1.2], but their column standard deviations span a factor of 226,000, so standardizing inflates near-constant noise dimensions to full size.
- *Q: Would you have to scale for a random forest?* No. Trees split on thresholds within one column and only the ordering of values matters, and any increasing rescaling preserves ordering.

---

## 3. The z-score, worked by hand

```
                x - mu
        z  =  ----------
                sigma

mu    = the mean (arithmetic average) of the numbers
sigma = the standard deviation = sqrt( average of the squared deviations from mu )
```

Real numbers: the first five readings of session 1 of Cohort 1, in mg/dL.

```
x = 116, 115, 112, 107, 101

step 1  mean:        116+115+112+107+101 = 551  ->  551/5 = 110.2       <- mu
step 2  deviations:  +5.8   +4.8   +1.8   -3.2   -9.2   (they sum to exactly 0)
step 3  square and average:  33.64+23.04+3.24+10.24+84.64 = 154.80
                             154.80 / 5 = 30.96            <- variance, ddof=0
                             sqrt(30.96) = 5.5642          <- sigma,    ddof=0
step 4  divide:      +5.8/5.5642 = +1.0424      -3.2/5.5642 = -0.5751
                     +4.8/5.5642 = +0.8627      -9.2/5.5642 = -1.6534
                     +1.8/5.5642 = +0.3235

check: these five z-values sum to 0 and have standard deviation exactly 1.
```

**The ddof trap.** If instead you divide the squared deviations by `n − 1 = 4` you get variance 38.70 and sigma 6.2209, and the z-values become `+0.9323, +0.7716, +0.2893, −0.5144, −1.4789`. Both are called "the standard deviation". Know which one your tool uses:

| Tool | divides by | our line |
|---|---|---|
| Chronos `InstanceNorm` | **n** (ddof=0) | `chronos_bolt.py:112` |
| `sklearn.StandardScaler` | **n** (ddof=0) | `regression.py:116` |
| our Arm B target statistics | **n** (ddof=0) — explicitly `unbiased=False` | `datamodule.py:61` |
| `torch.nn.LayerNorm` | **n** (ddof=0) | `model.py:147` |
| `numpy.std` default | **n** (ddof=0) | |
| `pandas.Series.std` default | **n − 1** (ddof=1) | |
| `gluc_std` in our hand-crafted features | **n − 1** (`ddof=1`) | `handcrafted.py:57` |

Everything in our normalization path uses ddof=0. The one place we use ddof=1 is a *feature value*, not a normalization. That is a fine answer and a good detail to have.

### What the z-score does and does not change

| Changed | Not changed |
|---|---|
| the units (mg/dL becomes "standard deviations") | the **shape** of the distribution — it is still skewed the same way |
| the location (mean becomes 0) | the **ordering** of the values — the biggest stays biggest |
| the spread (standard deviation becomes 1) | **correlations** between features |
| | **outliers** — a value 5 standard deviations out is still 5 standard deviations out |

That last row is the one people get wrong. Z-scoring is not outlier handling. Our glucose sits on hard rails at 40 and 400 mg/dL (Dexcom G6 reporting limits). Using the whole reading pool (mean 177.24, standard deviation 80.17), the 400 rail is at `z = (400 − 177.24)/80.17 = +2.78` and the 40 rail is at `z = −1.71`. Both are still there after z-scoring, still pinned, still censored. **1,470 readings (2.16%) sit exactly at 400** — meaning "at least 400", the true value unknown — and 99 (0.15%) exactly at 40. Z-scoring does nothing about that. It is a genuine limitation of the input, not something preprocessing fixes.

**Say it in your own words:** "Z-scoring subtracts the mean and divides by the standard deviation, so the numbers come out centered at 0 with spread 1 and the units disappear. It does not change the shape of the distribution or remove outliers — our readings censored at the 400 mg/dL rail are still censored afterwards."

**Drill**
- *Q: Compute the z-score of 116 in that set.* `(116 − 110.2) / 5.5642 = +1.0424`.
- *Q: n or n−1?* Every normalization in this project uses n, ddof=0 — Chronos, StandardScaler, our target statistics, and LayerNorm. `pandas.std` defaults to n−1, which is a common source of small mismatches.
- *Q: Does z-scoring remove outliers?* No. It preserves ordering and shape. 2.16% of our readings are pinned at the 400 mg/dL ceiling and remain pinned at z = +2.78.
- *Q: What is the mean and standard deviation of z-scored data, exactly?* 0 and 1, by construction. I verified: the five values above sum to 0 and have standard deviation exactly 1.0.

---

## 4. The alternatives, and which we use

| Method | Formula | What it gives you | When to use it | Weakness |
|---|---|---|---|---|
| **Standardization (z-score)** | `(x − mean) / sd` | mean 0, spread 1 | default for penalized linear models, SVR, neural nets — **this is what we use** | outliers pull the mean and inflate the standard deviation |
| **Min-max** | `(x − min) / (max − min)` | everything in [0, 1] | when you need a bounded range, e.g. image pixels | a single extreme value squashes everything else into a tiny band |
| **Robust / median-IQR** | `(x − median) / (Q75 − Q25)` | centered, scaled, ignores tails | when outliers are real and you want to keep them without letting them set the scale | the resulting spread is not 1, so "1 unit" is less interpretable |
| **Log** | `log(x)` | compresses a long right tail; turns multiplication into addition | positive, right-skewed quantities spanning orders of magnitude | undefined at 0 and below; changes the meaning of "a unit difference" |
| **Quantile / rank** | replace each value by its position in the sorted order, then map to a chosen distribution | forces any shape into a target shape | badly-behaved features you cannot fix otherwise | discards the actual distances between values — a monotone but non-linear distortion |

The same five glucose readings, `116, 115, 112, 107, 101`, through each:

```
raw          116       115       112       107       101
z-score      +1.0424   +0.8627   +0.3235   -0.5751   -1.6534
min-max       1.0000    0.9333    0.7333    0.4000    0.0000
robust       +0.5000   +0.3750    0.0000   -0.6250   -1.3750    (median 112, IQR = 115-107 = 8)
log           4.7536    4.7449    4.7185    4.6728    4.6151
```

**What we use, everywhere:** the z-score. Chronos-Bolt internally (stage 1), `StandardScaler` (stage 2), the Arm B target transform (stage 3), LayerNorm along a different axis (stage 4). Four stages, one recipe, four different sets of statistics. **We use no min-max, no robust scaling, no log, and no quantile transform anywhere in the pipeline.** Would robust scaling be defensible here? Arguably yes, for the hand-crafted features: `gluc_gvp` runs to 1120.6 and `gluc_time_below` is 0 for most sessions and 1 for a few, so those columns' means and standard deviations are pulled around by a handful of sessions. We did not try it — an honest "not tested", not a "wouldn't help".

**Say it in your own words:** "The main options are z-score, min-max, median-IQR, log, and quantile. We use the z-score at all four stages. Robust median-IQR scaling would be a reasonable alternative for the hand-crafted features because a few of them have long tails, but I did not test it."

**Drill**
- *Q: Why not min-max?* One extreme value sets the whole range. `gluc_gvp` has a maximum of 1120.6 against a mean of 273.3, so min-max would compress the typical sessions into the bottom quarter of [0, 1].
- *Q: When would a log help?* For a positive, strongly right-skewed quantity. `gluc_j_index` is a squared quantity ranging 2.62 to 181.96, so a log would be defensible there. Not needed for glucose itself, which is bounded to [40, 400] and only mildly skewed.
- *Q: What is the downside of a quantile transform?* It keeps ordering but destroys the actual distances. A jump from 390 to 400 mg/dL and one from 100 to 110 could end up the same size, which is wrong physiologically.

---

## 5. The vocabulary trap: "standardization" vs "normalization"

**The honest answer is that usage varies, and you should say so before giving definitions** — different textbooks, libraries and papers use these words differently, so if you assert one definition as universal, someone will produce a counterexample. The most common convention: **standardization** = the z-score, subtract the mean and divide by the standard deviation, giving centered, spread-1, **unbounded** output; **normalization** = min-max rescaling into a fixed range, usually [0, 1], giving **bounded** output.

Two complications make the words genuinely ambiguous. First, **"normalization" is also the umbrella term** for the whole family — BatchNorm, LayerNorm, InstanceNorm and RMSNorm are all called normalization and none of them is min-max; Chronos' own source calls its z-score step `InstanceNorm` and its comment at line 214 calls it *"scaling"*. Second, **sklearn uses "normalize" for something else entirely**: `sklearn.preprocessing.normalize` rescales each **row** to unit length, which is neither definition above.

**What we mean in this project:** whenever this repo says "normalize", it means **the z-score**, at whichever of the four stages is under discussion. So the safe way to answer is: *"Usage varies, so let me be precise. In this project every normalization is a z-score — subtract a mean, divide by a standard deviation. The four stages differ only in which numbers the mean and standard deviation are computed over. If you mean min-max rescaling into [0, 1], we do not use that anywhere."*

**Say it in your own words:** "The words are used inconsistently in the literature. Standardization usually means the z-score; normalization usually means min-max, but it is also the umbrella name for BatchNorm and LayerNorm. In this project everything is a z-score, and I say which statistics each one uses."

**Drill**
- *Q: Is LayerNorm standardization or normalization?* By formula it is standardization — a z-score. By name and convention it belongs to the "normalization layers" family. This is exactly why I state the formula instead of relying on the word.
- *Q: Does `StandardScaler` bound its output?* No. It is unbounded. A value 4 standard deviations out comes through as 4.
- *Q: What does `sklearn.preprocessing.normalize` do?* It scales each row to unit length. It is neither the z-score nor min-max, and we do not use it.

---

## 6. Data leakage — the single most important rule

**The rule: any number computed from the data must be computed from the training portion only.** Means, standard deviations, minima and maxima, PCA directions, which features to keep, hyperparameters. All of it.

**Why fitting a scaler on all the data is cheating.** Compute the mean and standard deviation of a feature across all 956 sessions, then split into train and test: those two numbers now contain a contribution from every test row, so every training row you feed the model has been shifted and scaled using information about the test set. The model has been told something about the answers it is about to be graded on. That is not the situation it faces at deployment time, where a genuinely new child's data has not been seen by anything.

### The mechanism, with exact arithmetic

Six values. The first five are training; the last one, 300, is a held-out test row.

```
train only:       mean{100,110,120,130,140}     = 120.0   sd = 14.1421
leaky (all six):  mean{100,110,120,130,140,300} = 150.0   sd = 68.3130

the test value 300, standardized:
   under TRAIN-only stats:  (300 - 120.0) / 14.1421 = +12.73  <- correctly extreme
   under LEAKY stats:       (300 - 150.0) / 68.3130 =  +2.20  <- looks ordinary
```

The leaky scaler moved the center up by 30 and inflated the spread by a factor of 4.8, **because of the test row itself**. The model then sees a test point that has been quietly pulled toward the region it was trained on. The test point helped decide how it would be measured.

### The picture

```
  WRONG                                          RIGHT
  =====                                          =====
  all 956 sessions                               all 956 sessions
        |                                              |
        v                                              v
  fit scaler  <=== sees test rows              split into folds
        |                                              |
        v                                      +-------+-------+
  transform everything                      TRAIN            TEST
        |                                        |               |
        v                                   fit scaler           |
  split into folds                               |               v
        |                                   transform ---> transform with the
    train / score                                |         TRAIN statistics
                                            train / score        |
  Reported number is optimistic:                                 v
  test rows influenced the numbers                         honest score
  used to prepare training.
```

### How much does it actually inflate the score?

Measured, on pure noise — 60 rows, 400 random features, a random target, no real relationship at all:

| What was fit on all the data | Test R² |
|---|--:|
| scaler fit on all 60 rows, then Ridge | −0.1358 |
| scaler fit on the 40 training rows only, then Ridge | −0.1206 |
| **top 10 features chosen by correlation with all 60 targets**, then Ridge | **+0.3979** |
| top 10 features chosen using the 40 training targets only | −0.4793 |

**Be precise about the size of the effect, because overstating it is a trap.** Leaking only a scaler's mean and standard deviation is usually *mild* — here it moved R² from −0.121 to −0.136, and the sign of the change is not even guaranteed. What is **catastrophic** is leaking anything that touched the **target**: choosing the top 10 features by their correlation with all the targets produced R² = **+0.3979 on pure noise**, versus −0.4793 when done honestly. A reported R² of 0.40 from data containing no relationship whatsoever. That is how fake results get published.

### Why sklearn's `Pipeline` makes it structurally impossible

`regression.py:116-120` builds `Pipeline([("scaler", StandardScaler()), (optional "pca", PCA(...)), ("model", estimator)])`.

A `Pipeline` is a single estimator. When cross-validation calls `pipe.fit(X_train, y_train)`, the scaler's `fit` is called on `X_train` **only** — it is physically not given the test rows. When `pipe.predict(X_test)` is called, the scaler only `transform`s, using the statistics it already stored. There is no code path in which the scaler sees test data. The same protection covers the `PCA` step: its component directions are computed from each fold's training rows only. And because `GridSearchCV` is wrapped around the whole `pipe` (line 123), the alpha search also refits the scaler inside each inner fold. That is why "use a Pipeline" is real engineering advice and not style advice: it converts a rule you have to remember into a rule the code enforces.

### The second leak, which matters more here: participants

Splitting by row would put **the same child** in both training and testing. 956 sessions come from only **20 participants**, roughly 48 sessions each. A model could learn "child 7 usually scores 0.31" from child 7's training sessions and get credit for predicting child 7's test sessions — without knowing anything about glucose. So the outer split is `GroupKFold` on `subid` (`regression.py:81`), holding out 4 participants at a time, and `regression.py:112-114` checks it on every fold:

```python
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
    "subject leakage between train and test!"
```

An `assert` that runs on all 5 outer folds of every run. If a participant ever appeared on both sides, the run would crash rather than report a flattering number. Be able to point at this line; it is the strongest single piece of evidence that the protocol is honest.

**Say it in your own words:** "Leakage is using information from the held-out data to prepare the model. The scaler's mean and standard deviation are computed on the training fold only, and they are inside an sklearn Pipeline so the code cannot do otherwise. The folds hold out whole participants, not rows, and there is an assert on every fold that the participant sets are disjoint. On pure noise, picking features using all the targets produced a fake R² of 0.40 — that is what the protocol prevents."

**Drill**
- *Q: Where exactly is your scaler fitted?* Inside the `Pipeline` built at `regression.py:116`, so it is fitted on each outer fold's training rows, and refitted again inside each inner fold during the alpha search.
- *Q: If you had fitted it on all 956 rows first, would your conclusion change?* Almost certainly not, and this is important: our R² values are all at or below 0, so a mild optimistic bias would not lift them above 0. Leaking a scaler is mild. Leaking anything target-related is not, and we do not do that.
- *Q: Why group by participant instead of shuffling rows?* Because 956 sessions come from 20 participants. Shuffling rows lets the model memorize each child's usual score and score well without using glucose at all. Grouped folds ask the real question: predict a child never seen before.
- *Q: What would catch it if you got the split wrong?* The `assert` at `regression.py:113`, on every one of the 5 outer folds.
- *Q: Is PCA a leakage risk?* Yes, if fitted on everything — its directions are computed from the data. Ours is a step inside the same `Pipeline` (`regression.py:118`), so it is fitted per training fold.

---

## 7. LayerNorm vs BatchNorm vs StandardScaler

Three things that all compute `(x − mean) / sd` and differ **entirely** in which numbers the mean and standard deviation are taken over. This is a standard exam question and no other document in this repo answers it. Picture the data as a table: rows are examples in a batch, columns are the 512 features.

```
                    feature 0  feature 1  ...  feature 511
     example 0          a          b      ...      c
     example 1          d          e      ...      f
     example 2          g          h      ...      i

  StandardScaler: DOWN a column, over the whole TRAINING SET (512 pairs, stored)
  BatchNorm:      DOWN a column, over the CURRENT BATCH only (+ running average)
  LayerNorm:      ACROSS a row, over that ONE example's 512 features (nothing stored)
```

| | **StandardScaler** | **BatchNorm** | **LayerNorm** |
|---|---|---|---|
| Which axis | per feature, across rows | per feature, across rows in the batch | per example, across features |
| Which statistics | mean and sd of the training set, computed once and stored | mean and sd of the current batch during training; a running average at inference | mean and sd of that single example |
| Uses dataset statistics? | **Yes** | **Yes** (batch, plus running averages) | **No** |
| Can it leak? | **Yes if fit on all data.** Prevented by putting it in a `Pipeline` | **Yes**, subtly: batch statistics mix examples together, so one example's prediction depends on the others in its batch | **Never.** Nothing is shared between examples |
| Depends on batch size? | No | **Yes** — unstable with small batches | No |
| Train and inference behave the same? | Yes | **No** — batch statistics in training, running averages at inference. A classic source of bugs | Yes |
| Learnable parameters | None | 2 per feature (scale, shift) | 2 per feature (scale, shift) |
| Typical use | classical ML feature matrices | convolutional networks on images, large batches | transformers, sequence models, small batches |
| **Here** | `regression.py:116`, Arm A | **not used anywhere** | `model.py:147`, first layer of Arm B's head |

### Our LayerNorm, exactly

The head, `ben_adapter/model.py:146-152`, is `nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))` — LayerNorm is literally the first thing that touches the embedding.

Verified behaviour on a real embedding row: `LayerNorm(x)` matched `(x − x.mean()) / sqrt(x.var(unbiased=False) + 1e-5)` to `2.4e-07`, i.e. exactly, up to float32 rounding. Its two learnable vectors are initialized to weight 1 and bias 0. Row 0 of our real embedding matrix has mean −0.006384 and standard deviation 0.126794; after LayerNorm its mean is 1.4e-08 and its standard deviation is 0.999689. Of the head's **132,609** trainable parameters, LayerNorm accounts for **1,024** (512 scales plus 512 shifts).

**Why LayerNorm and not BatchNorm here?** Two reasons, and the second is the good one. First, batch size 64 against roughly 612 training rows means only 10 batches per epoch, and batch statistics from 64 rows are noisy. Second, and more important: BatchNorm makes each example's output depend on which other examples happened to share its batch. With whole participants held out and only 20 participants, that is exactly the kind of cross-contamination we spent effort eliminating. **LayerNorm uses no information from any other example, so it cannot leak, by construction.** That is the answer to give.

**Say it in your own words:** "All three are the same formula on different axes. StandardScaler normalizes each feature across the training rows and stores those numbers. BatchNorm normalizes each feature across the current batch, so one example's output depends on its batch-mates. LayerNorm normalizes each example across its own 512 features and uses no dataset statistics at all, so it can never leak. We use StandardScaler in Arm A and LayerNorm in Arm B, and BatchNorm nowhere."

**Drill**
- *Q: Which of the three cannot leak, and why?* LayerNorm. It only uses the one example's own 512 numbers. No statistic crosses between examples.
- *Q: Where is BatchNorm in your code?* Nowhere. LayerNorm at `model.py:147` is the only normalization layer in the head.
- *Q: LayerNorm has learnable parameters — doesn't that undo the normalization?* It can rescale and reshift afterwards, and yes, in principle it could learn to restore something like the original scale. But it starts at weight 1 and bias 0, and crucially it learns **one** scale per feature shared by all examples, not per example. The per-example centering and scaling is not undone.
- *Q: Why is BatchNorm bad at small batch sizes?* Because it estimates a mean and a standard deviation from the batch itself. With 64 rows those estimates are noisy, and the noise differs every step, which destabilizes training.

---

## 8. Our target-side option: within-participant centering

The four stages above are about the inputs, plus the Arm B target z-score. There is one more target-side transform, and it changes the **question being asked**, not just the numbers. `cgm_tsfm/data.py:105-142`, `within_subject_normalize` with `mode="center"`, loops over participants and does `out[idx] = vals - vals.mean()` — for each participant, subtract that participant's own average score from each of their sessions. Applied at `regression.py:185`, before cross-validation.

| Target | The question | Why you would ask it |
|---|---|---|
| raw score | "Given glucose readings from a child I have never seen, predict their score." | This is what a deployable tool would have to do. |
| score minus that child's own average | "Given glucose readings, predict whether **this** child did better or worse than their own usual." | This is closer to the actual scientific hypothesis: does a glucose swing move *this* child's performance? Baseline differences between children — schooling, age, practice, motivation — are removed. |

**The results.** Same features, same folds, same models (`results/headtohead_real_centered.md`):

| Score | raw target | after within-participant centering |
|---|--:|--:|
| Grids | −0.089 | −0.018 |
| Symbols | −0.056 | −0.022 |
| Prices | −0.012 | −0.002 |

Centering moved every number closer to 0 — closer to "exactly as good as guessing the average" — but **not one of them went above 0**. So even the easier, more scientifically-targeted question does not get a usable answer from these features. (Why they move toward zero: subtracting each child's average removes the between-participant part of the variation, which was the part the model got most wrong on held-out children. Removing a source of error moves R² up toward 0, but it does not add any correctly-predicted variation, which is what would push R² above 0.)

**The catch you must volunteer before you are asked.** Computing a participant's average uses **that participant's own sessions**, including the ones in the held-out fold. In a grouped fold the participant is entirely held out, so in a real prospective setting their personal average would be unknown — you would need to test them repeatedly first. `data.py:120-125` says this itself: *"This uses each subject's own sessions to compute their mean/std (ORACLE centering). Under leave-subjects-out / GroupKFold CV a test subject is entirely held out, so their personal mean would be unknown in a real prospective setting."* So the centered numbers **characterize** whether a within-participant relationship exists at all; they do not describe a tool that could be deployed on a new child. Say it that way.

**Say it in your own words:** "Centering means subtracting each child's own average score, so the model has to predict that child's good sessions versus their bad ones instead of predicting a new child. It moved R² from −0.089, −0.056, −0.012 to −0.018, −0.022, −0.002 — closer to break-even but still not better than guessing the average. And it uses the held-out child's own sessions to compute their average, so it describes whether the relationship exists rather than predicting anything prospectively."

**Drill**
- *Q: Isn't centering per participant a form of leakage?* Yes, in the strict sense, and I flag it rather than hide it. The participant's own average is computed from all their sessions including held-out ones. That is why I present the centered numbers as answering "is there a within-participant relationship at all?" and not as a deployable result.
- *Q: If it leaks, why is R² still below 0?* Because the leaked information — one number per child — helps remove between-child variation but tells the model nothing about which of that child's sessions were good. The remaining question is genuinely hard, and the features do not answer it.
- *Q: The centered numbers are better. Isn't that a signal?* No. Better here means "closer to 0", and 0 means exactly as good as predicting the average. −0.018 is still on the wrong side of 0.

---

## 9. Does normalizing the target change R²? No.

Short answer: **no**, and you should be able to prove it in two lines. R² is defined as

```
             sum of (truth - prediction)^2            SS_residual
   R^2 = 1 - ------------------------------- = 1 -  -------------
             sum of (truth - mean_of_truth)^2          SS_total
```

Now apply any affine transform to the truth, `y -> a*y + b`, and the **same** transform to the predictions (which is what our code does — it z-scores the target for training and converts predictions back at `lightning_module.py:70`). Each residual becomes `a*(truth − prediction)`, so `SS_residual` is multiplied by `a²`. The mean of the truth becomes `a*mean + b`, so each deviation becomes `a*(truth − mean)` and `SS_total` is multiplied by `a²` too. The ratio is unchanged, so **R² is unchanged.** Measured, on 50 rows:

| Transform of y | R² | RMSE |
|---|--:|--:|
| raw | 0.517426 | 1.836300 |
| y z-scored | **0.517426** | 0.694676 |
| y negated (`y -> -y`) | **0.517426** | 1.836300 |
| y shifted by +1000 | **0.517426** | 1.836300 |

R² is identical to six decimals in all four cases. RMSE changes only when the *scale* changes (`1.836300 / 0.694676 = 2.6434`, exactly the standard deviation of y that was divided out) — shifting and negating leave it alone, because RMSE has the units of y.

### Two practical consequences

**1. The Arm B target z-score is a training convenience, not a result-changer.** `lightning_module.py:67` does `labels_norm = (labels - self.target_mean) / self.target_std` so the MSE loss lands on an order-of-1 scale for any of the three scores — Grids has standard deviation 0.4999, Prices 17.1696, a factor of 34 apart, which would otherwise mean three wildly different effective learning rates. Line 70, `preds = logits * self.target_std + self.target_mean`, converts predictions straight back before any metric is computed, so every reported RMSE, MAE and R² is in the score's own units.

**2. The score-direction question does not change a single reported number.** All three scores are lower-is-better (Liuyi stated this). Negating a score to make it higher-is-better is the affine transform with `a = −1, b = 0`. As the table shows, R² is unchanged and RMSE is unchanged. `config.py:16-20` says exactly this, and `PRICES_IS_INVERTED = False`, so we keep all three raw. **The sign choice affects interpretation only.** If someone suggests the flat results are an artifact of the sign, that is arithmetically impossible.

**Say it in your own words:** "No. R² is a ratio of two sums of squares that both pick up the same scale factor, so any shifting, rescaling, or negation of the target cancels out. I measured it: 0.517426 in all four cases. That is also why the lower-is-better question does not change any number I have reported — negating the target leaves R² and RMSE identical."

**Drill**
- *Q: Then why z-score the target at all?* Only to put the loss on an order-of-1 scale so one learning rate works for all three scores, whose standard deviations differ by a factor of 34. Predictions are converted back before metrics.
- *Q: What about RMSE?* RMSE has the units of y, so it changes if the scale changes: 1.8363 raw becomes 0.6947 on z-scored y, the ratio being exactly the standard deviation removed. It does not change under shifting or negation. That is why we always report RMSE in the score's native units.
- *Q: Grids mean 0.4782, sd 0.4999. Z-score a raw score of 0.333.* `(0.333 − 0.4782) / 0.4999 = −0.2905`. And back: `−0.2905 × 0.4999 + 0.4782 = 0.33298`, which recovers 0.333.
- *Q: A raw grids score of 1.0?* `(1.0 − 0.4782) / 0.4999 = +1.0438`. Roughly one standard deviation above average, and since lower is better, that is a poor session.

---

## 10. Missing data: what we actually do, and the one place it bites

### The exact accounting

```
   740   rows in Cohort1_scores_merged_with_glucose.csv
+  240   rows in Updated_Cohort2_scores_with_glucose.csv
=  980   raw rows   (one row = one cognitive-test session)
-   24   rows whose Glucose_Before_Test cell is empty or unparseable   (23 + 1)
-    0   rows with fewer than 3 valid readings (min_readings = 3) — none in practice
=  956   sessions used,  from 20 participants,  68,206 individual readings
```

Then, **separately**, scores can be missing. Handled per target by a mask — `data.py:87-89` returns `~np.isnan(self.targets[target])`, applied at `regression.py:183-184` — which is why:

| Score | Sessions with a score | Missing |
|---|--:|--:|
| Grids | **951** | 5 |
| Symbols | **920** | 36 |
| Prices | **936** | 20 |

**The three targets do not use the same set of sessions.** A picky question here is "how many sessions did your model train on?" and the correct answer is "951, 920 or 936 depending on which of the three scores — 956 sessions have glucose, but the scores are missing at different rates." Get that right.

### Individual NaN readings inside a window: dropped

`cgm_tsfm/data.py:47`, inside `parse_glucose`, is one line: `arr = arr[~np.isnan(arr)]   # strip missing readings`.

**This contradicts what our docs claim, and you must be able to say so.** `docs/01_PIPELINE_DESIGN.md:47` says "NaN-tolerant. Missing readings map to a mask token — no gap-filling needed." `PLAIN_ENGLISH_SUMMARY.md:153` and `docs/understanding/04_WALKTHROUGH_AND_QA.md:49` say the same. Chronos-Bolt **is** NaN-tolerant — but **that tolerance is never exercised, because we delete the NaNs before Chronos ever sees them.**

**What Chronos-Bolt would have done with them** (`chronos_bolt.py:277-303`), worth knowing precisely because it is the mechanism we bypass. Line 280 builds a 0/1 observed-mask from where the values are not NaN. Line 288 runs `InstanceNorm`, whose mean and standard deviation are **NaN-aware**, so missing points do not corrupt them. Lines 296-297 cut both the values and the mask into patches of **16** (`input_patch_size = 16`, `input_patch_stride = 16`, from the `bolt-small` config). Line 298 sets missing values to **0.0** — and because this happens *after* standardization, 0.0 means **"at the series mean"**; a missing reading is filled with the series average, in normalized space. Line 300 **concatenates** the 16 values and the 16 mask flags, giving `16 + 16 = 32` inputs per patch, which is why the input layer's width is `input_patch_size * 2 = 32` (line 201) — the model is told *which* points were observed, so it can discount the filled ones. Line 303 sets `attention_mask = patched_mask.sum(dim=-1) > 0`, so a patch with **zero** observed points is dropped from attention entirely.

**Why our dropping is worse than leaving the NaNs in.** Glucose comes at one reading per 5 minutes, and **Chronos receives no timestamps** — it assumes the points it is given are evenly spaced. Delete one reading and the two neighbours become 10 minutes apart, but the model still reads them as 5. A 30-minute sensor gap silently becomes six consecutive readings that look like normal 5-minute steps. Every rate of change across that gap is wrong by the gap's length. Feeding the NaNs through instead would have let the mask flags tell the model where the holes were.

**How much does it matter?** We do not know, and you should say so. Liuyi's own position is *"the CGM monitoring is continuous, i.e., always on. So there is no need for imputation"* — under which reading there are few real gaps and the effect is small. Against that, the original Cohort2 file had 44 of 240 rows with **entirely** empty glucose, and the `Updated_` file we use has only 1, so something filled those in and we have not seen the code that did it (see `01_FACT_SHEET.md` section B). **Do not claim the timing is clean.**

### A verified problem in our own pooling, found while writing this chapter

This one is ours, not Chronos', and it is worth disclosing before someone finds it.

`encoders.py:101-108` batches 32 windows at a time. Chronos pads the shorter ones with NaN on the left so they all reach the length of the longest in the batch. Those pad positions become whole patches with zero observed points, and Chronos correctly sets their `attention_mask` to 0 — but they **still appear as rows in the encoder output**, and `encoders.py:108` is `pooled = emb.mean(dim=1)`, which averages over *all* rows including the padding ones.

Measured directly. An 8-reading series alone produces 2 tokens, both valid. The same series batched with a 200-reading series produces 14 tokens with `attention_mask = [0,0,0,0,0,0,0,0,0,0,0,0,1,1]` — only the last 2 valid. Averaging over all 14 differs from averaging over the correct 2 by up to **0.924** per element. Averaging over only the valid tokens reproduces the alone-case to `1.8e-07`, i.e. exactly.

On the real data, under the actual batching (batch size 32, dataset order):

**915 of 956 sessions (95.7%)** have padding tokens in their pooled embedding. On average **60.9%** of the pooled tokens are padding; the median is **73.7%**, the worst session **89.5%**, and for 699 sessions (73.1%) over half the pooled tokens are padding. So a session's 512 numbers depend on which other sessions happened to share its batch. That is a real flaw.

**And here is the part that matters most: fixing it does not change the conclusion.** Recomputing all 956 embeddings with the mask applied to the pooling (mean over valid tokens only) and re-running the identical grouped cross-validation:

| Score | as-shipped Ridge / SVR | corrected pooling Ridge / SVR |
|---|--:|--:|
| Grids | −0.0885 / −0.1208 | −0.1137 / −0.1277 |
| Symbols | −0.0918 / −0.0562 | −0.0934 / −0.0523 |
| Prices | −0.0724 / −0.0117 | −0.0687 / −0.0117 |

Every value is still at or below 0. The corrected embeddings differ from the shipped ones by 0.122 on average per element (maximum 1.215), 915 of 956 rows changed, and **nothing crosses zero.** That is the ideal thing to be able to say about a flaw you found in your own work: here it is, here is exactly how big it is, and here is the measurement showing it does not rescue the result.

**Say it in your own words:** "Twenty-four of the 980 rows had no usable glucose, leaving 956 sessions. Scores are missing separately, so grids, symbols and prices actually use 951, 920 and 936 sessions. Individual missing readings are deleted at `data.py:47` — so Chronos's NaN handling is never used, and deleting a reading breaks the regular 5-minute spacing that Chronos assumes, because it gets no timestamps. I also found that our mean-pooling averages in padding tokens for 96% of sessions; I recomputed the embeddings with the mask applied and the R² values are still all at or below zero."

**Drill**
- *Q: How many sessions does your model use?* 956 have usable glucose. Per score it is 951 grids, 920 symbols, 936 prices, because scores go missing independently.
- *Q: Your docs say Chronos is NaN-tolerant so you need no gap-filling. Is that true?* Chronos-Bolt is NaN-tolerant — it patches the values together with a 0/1 observed-mask, 16 plus 16 equals 32 inputs per patch, and drops fully-unobserved patches from attention. But we never use it: `data.py:47` deletes the NaNs first. Those three docs are wrong about our pipeline and I would correct them.
- *Q: Why does deleting a NaN matter if Chronos would fill it with the mean anyway?* Because Chronos gets no timestamps and assumes even spacing. Deleting a reading silently makes two neighbours 10 minutes apart while the model reads them as 5. Filling keeps the time axis correct; deleting corrupts it.
- *Q: What does Chronos fill a missing value with?* 0.0, set at line 298 *after* the standardization at line 288 — so it is the series mean in normalized space. And the accompanying mask flag tells the model it was filled.
- *Q: Is there anything wrong with your feature extraction?* Yes, one thing I found and measured: mean-pooling averages in left-padding tokens for 915 of 956 sessions, 60.9% of pooled tokens on average. I recomputed with the attention mask applied — the embeddings changed substantially (mean absolute change 0.122) and every R² stayed at or below 0.

---

## The three hardest questions on this topic

### Hard question 1

> *"You told me Chronos normalizes the glucose for you. Show me exactly what it computes, and tell me what that costs you."*

**Model answer.** It applies instance normalization, in `chronos/chronos_bolt.py`, class `InstanceNorm`, lines 95 to 134, called at line 288. For each series independently it computes `loc` as the NaN-aware arithmetic mean over time and `scale` as the population standard deviation, dividing by n rather than n−1, then returns `(x − loc) / scale`. If the series is perfectly flat the standard deviation is zero and line 113 substitutes `eps = 1e-5`; if the whole series is missing, `loc` becomes 0 and `scale` becomes 1.

I verified it by running it. On the first eight readings of session 1 — 116, 115, 112, 107, 101, 97, 94, 93 — the mean is 104.375 and the population standard deviation is 8.774074, and `embed()` returned `loc = 104.375` and `scale = 8.774073600769043`. The n−1 version would have been 9.37988, so it is definitely ddof = 0.

What it costs me: because both the mean and the standard deviation are divided out, the 512 numbers describe the *shape* of the glucose curve and are largely blind to how high it actually was and how much it actually swung in mg/dL. That shows up in a measurement. Asking the same embeddings to predict a property of the glucose itself, they recover the session's variability at R² 0.462 but the session's mean glucose at only 0.070. Chronos actually hands the mean and the standard deviation back to the caller — the docstring calls them the mean and std of the original series — and my code discards them at `encoders.py:104`, where the second return value is assigned to an underscore. Adding them back as two extra features is the first improvement I would make.

I should also flag that several older documents in this repository describe this wrongly, saying Chronos "divides by its own average magnitude". That is Chronos-**T5**'s scheme, at `chronos.py:177`, where `scale = mean(|x|)` with no centering. Bolt centers and scales. Those docs need fixing.

### Hard question 2

> *"You said your results are honest because there is no leakage. Convince me — and tell me how much leakage would have mattered."*

**Model answer.** Three defences, in increasing strength.

First, the scaler is inside an `sklearn.Pipeline`, built at `regression.py:116`. A Pipeline is one estimator: when cross-validation calls `fit`, the scaler's `fit` is given only the training rows, and on the test rows it only calls `transform` with the statistics it already stored. There is no code path that shows the scaler test data. The same holds for the optional PCA step at line 118. And because `GridSearchCV` wraps the whole Pipeline at line 123, the alpha search refits the scaler inside each inner fold too.

Second, the folds hold out whole **participants**, not rows — `GroupKFold` on `subid`, 5 outer folds, 4 of the 20 participants held out each time. That matters because 956 sessions come from 20 participants, about 48 each, so a row-level split would let the model memorize each child's usual score and score well without using glucose.

Third, and this is the one I would point at: `regression.py:112-114` asserts on **every** fold that the training and test participant sets are disjoint. It is an `assert`, so a violation crashes the run rather than reporting a flattering number.

On how much it would have mattered, I want to be precise rather than dramatic. Leaking only a scaler's mean and standard deviation is usually mild. I measured it on pure noise — 60 rows, 400 random features, random target: R² was −0.1358 with the scaler fitted on all rows and −0.1206 fitted on training rows only. Small, and not even reliably in the flattering direction. What is catastrophic is leaking anything that touched the **target**: choosing the top 10 features by their correlation with all 60 targets gave R² **+0.3979** on data with no relationship in it at all, against −0.4793 when the choice used training targets only. We never do target-aware selection. And since all our reported values sit at or below 0, a mild optimistic bias could not have lifted them above 0 — the direction of the argument works in my favour here.

### Hard question 3

> *"Your model doesn't work. How much of that is your preprocessing?"*

**Model answer.** Let me separate what I can measure from what I cannot.

What I can measure. First, scaling is not the cause. On the 43 hand-crafted features, removing the `StandardScaler` makes Ridge slightly worse on all three scores; on the Chronos embeddings it makes it slightly better. Nothing crosses zero either way. Second, I found a genuine flaw in my own pooling: `encoders.py:108` averages left-padding tokens into the embedding for 915 of 956 sessions, 60.9% of the pooled tokens on average. I recomputed all 956 embeddings with the attention mask applied to the pooling — the embeddings changed substantially, mean absolute change 0.122 per element with 915 rows affected — and every R² stayed at or below 0: grids −0.114, symbols −0.052, prices −0.012 at best. So fixing it does not rescue the result. Third, deleting individual missing readings at `data.py:47` distorts the time axis, because Chronos gets no timestamps and assumes even 5-minute spacing. I have not measured that one and I will not pretend to have.

What I think the real reason is, and it is not preprocessing. Before any model, the strongest correlation anywhere between a simple glucose statistic and a score is r = −0.092, which is 0.85% of the variance. Everything else is under 0.6%. No amount of preprocessing manufactures a relationship that is not in the data. A shuffle check confirms it: scrambling the scores 200 times and re-running, the real result lands at the *bad* end of the scrambled pile, with p values of 1.000, 1.000 and 0.995.

Where I think preprocessing genuinely does cost me something specific: Chronos removes absolute level, and absolute level is what the clinical hypothesis is about. That is the one real preprocessing limitation on the result, it is measurable — mean glucose recovered at R² 0.070 from the same embeddings — and it has a concrete fix, which is to stop discarding `loc` and `scale`. The reason it does not change my overall conclusion is that the hand-crafted comparison arm encodes absolute level explicitly, with `gluc_mean`, `gluc_time_in_range`, `gluc_time_below`, `gluc_max`, and it was also no better than guessing the average.

---

## Chapter summary in ten sentences

1. Normalization means rescaling numbers so quantities measured in different units can be compared, and the standard recipe is the z-score: subtract a mean, divide by a standard deviation.
2. You must do it before training because most algorithms treat "one unit" as equally important in every column, so a feature measured in hundreds silently outweighs one measured in fractions.
3. There are three concrete failure modes: distance-based models let the biggest-unit column own the distance (99.9967% versus 0.0033% in my two-feature example); penalized models like Ridge apply one alpha to all coefficients so the shrinkage depends on units (82% shrinkage on the fraction-valued feature versus none on the mg/dL feature); and gradient descent needs comparable curvature in every direction (my raw 43 features had a curvature ratio of 4.8e15, standardizing brought it to 4.4e4).
4. Tree-based models are the exception because they only use the ordering of values — and I should say honestly that our XGBoost never ran, since the package is not installed in this environment.
5. This project normalizes four separate times: Chronos-Bolt's internal instance normalization along time, `StandardScaler` across training rows per feature, the Arm B target z-score, and LayerNorm across the 512 features within one example.
6. Chronos-Bolt centers **and** scales — `(x − mean) / population_sd` at `chronos_bolt.py:95-134`, verified numerically — and several older docs in this repo wrongly describe Chronos-T5's mean-magnitude scheme instead; the consequence is that our embeddings are largely blind to absolute glucose level, which is why they recover glucose variability at R² 0.462 but mean glucose at only 0.070.
7. The one rule that makes any of this honest is that every statistic must come from the training portion only, which our `Pipeline` at `regression.py:116` enforces structurally and the `assert` at `regression.py:113` checks for participants on every fold.
8. LayerNorm, BatchNorm and `StandardScaler` are the same formula on three different axes; LayerNorm is the only one that uses no statistics from other examples, which is why it can never leak and why it is the one inside our neural head.
9. Normalizing the target does not change R² at all, because both sums of squares in the ratio pick up the same scale factor — verified at 0.517426 under z-scoring, negation and shifting — which is also why the lower-is-better score direction cannot change any number we report.
10. Our missing-data handling is: 24 of 980 rows dropped for empty glucose leaving 956 sessions, scores masked per target so grids, symbols and prices use 951, 920 and 936 sessions, and individual missing readings deleted at `data.py:47` — which means Chronos' NaN handling is never exercised and the regular 5-minute spacing is silently broken, a real limitation I can name but have not measured.
