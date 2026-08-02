# 05 · Evaluation — how we decided the model does not beat guessing the average

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you (5 lines).**
> 1. A plain-English answer to Liuyi's exact question: *"I cannot understand what this 'predict the average' baseline does."*
> 2. Every metric we report (MSE, RMSE, MAE, R²) hand-worked on five numbers you can redo on paper.
> 3. The one explanation that carries the whole project: **why R² can be negative, and what negative means.**
> 4. The reason our split is by *participant* and not by *session* — the single most important design decision in the study.
> 5. Pre-committed criteria for what would make us believe a real glucose→cognition relationship exists.

Companion chapters: `01_FACT_SHEET.md` (every number with proof), `02_ML_FROM_ZERO.md` (what a model is at all), `04_LINEAR_MODELS.md` (Ridge, SVR, why regularization), `10_RIGOR_AND_STATS.md` (the scrambled-score check and the glucose-property control).

---

## How to talk about the result (read this before anything else)

Liuyi banned these habits. Follow the right-hand column exactly.

| Do not say | Say instead |
|---|---|
| "null result" | **"lower accuracy"**, or **"no better than guessing the average"** |
| "no signal beyond chance" | **"the real number lands inside the range we get from scrambled scores"** |
| "power" | **"the subgroup is big enough (201 sessions) that the flat result isn't a small-sample artifact"** |
| "sensitivity" | **"how small a relationship we could have detected"** |
| "patients" (for rows) | **"sessions"** — 956 sessions from 20 **participants** |

⚠️ Two auto-generated files in the repo violate this. `results/rigor_real.md` prints *"no signal beyond chance"* and `results/README.md` says *"well-powered"* and *"a real null"*. **That is the code's wording, not yours.** If Liuyi opens those files, say: *"Those strings are printed by `run_rigor.py`. I'd phrase it as 'the real number lands inside the scrambled range.' I'll change the printed text."* Owning it beats being caught by it.

---

## 1. Why you need a comparison point, and what the "predict the average" baseline is

### 1.1 The problem with a bare error number

Suppose I tell you: *"my model predicts the Grids score with an average error of 0.41."* Is that good? You cannot answer. You have no idea whether 0.41 is small or large for this score. You need something to compare against — not another fancy model, but the **dumbest possible predictor**, so that "better than dumb" becomes a meaningful sentence.

### 1.2 What the baseline actually does

> **The "predict the average" baseline throws the glucose data in the bin, computes the average score of the children it was trained on, and prints that same number for every single test session.**

It has no inputs. It cannot look at glucose. It does not know which child it is predicting or what time of day it is. It is a constant. In code it is one line of scikit-learn (`regression.py:192`):

```python
DummyRegressor(strategy="mean")
```

Fit on a training set, it stores `y_train.mean()`. Asked to predict, it returns that stored number, repeated once per test row. That is the whole model — hence "Dummy."

Why bother? Because **the average is genuinely hard to beat.** It is the single number with the smallest total squared error on the data it came from. Any model that does not beat it has, in a precise sense, learned nothing usable from glucose.

### 1.3 Worked example on the real Grids data

Real numbers, computed 2026-07-30 with `GroupKFold(n_splits=5)` on the 951 sessions that have a Grids score. Fold 1:

```
Training set: 16 participants, 763 sessions with a Grids score
  average Grids score of those 763 sessions = 0.519256

Test set:     4 participants, 188 sessions
  their actual average                      = 0.311700

The baseline's prediction for all 188 test sessions:  0.519256
                                                      0.519256
                                                      0.519256   ... x188
```

That is it. One number, 188 times. It got the level wrong by 0.2076 because those four particular children happened to be better at Grids than the training group; section 2.5 shows exactly what that costs. Here is the baseline's own score in each of the 5 grouped folds:

| Fold | test sessions | train mean | test mean | baseline R² | baseline RMSE |
|---|--:|--:|--:|--:|--:|
| 1 | 188 | 0.5193 | 0.3117 | **−0.2503** | 0.4639 |
| 2 | 191 | 0.4373 | 0.6410 | **−0.1269** | 0.6071 |
| 3 | 192 | 0.4812 | 0.4666 | −0.0010 | 0.4511 |
| 4 | 190 | 0.4700 | 0.5114 | −0.0062 | 0.5274 |
| 5 | 190 | 0.4833 | 0.4579 | −0.0030 | 0.4633 |
| | | | **mean** | **−0.0775 ± 0.0988** | 0.503 ± 0.059 |

**Hold on to that −0.0775.** Chronos + Ridge scored **−0.089 ± 0.123** on the same data. The gap between our 512-number foundation-model pipeline and a constant is about **0.011 of R²**. Not a catastrophe — a tie. Same story on the other two:

| Score | mean baseline R² (grouped, 5 folds) | Chronos + Ridge/SVR | gap |
|---|--:|--:|--:|
| Grids | −0.0775 ± 0.0988 | −0.089 ± 0.123 | 0.011 |
| Symbols | −0.0503 ± 0.0396 | −0.056 ± 0.060 | 0.006 |
| Prices | −0.0044 ± 0.0028 | −0.012 ± 0.010 | 0.008 |

*(Baseline column computed 2026-07-30, `DummyRegressor(strategy="mean")`, same `GroupKFold(5)` on the same per-target masked arrays. Model column from `results/headtohead_real.md`, generated 2026-07-07.)*

This table is the most useful thing in the chapter for a hostile question: it reframes "your R² is negative, your model is broken" into "the constant predictor is also negative here, for a reason I can explain, and we are within a hundredth of it."

**Say it in your own words:** *"The baseline ignores the glucose completely and just predicts the average score of the children it trained on — one constant number for every test session. It's the bar. If glucose carried usable information, we'd beat it. We don't."*

**The 30-second spoken script for Liuyi (rehearse this out loud):**

> *"The 'predict the average' baseline is scikit-learn's `DummyRegressor` with `strategy='mean'`. It's a model with no inputs. During training it computes one number: the average cognitive score across the training participants' sessions. At test time it prints that number for every session, no matter what the glucose looked like. On Grids fold one that number is 0.519, repeated 188 times. We include it because an error number on its own is uninterpretable — 'average error 0.41' means nothing until you know what error you'd get for free. And R², the metric we report, is defined against exactly this predictor: R² = 0 means you tied it, positive means you beat it, negative means you did worse."*

### 1.4 Drill

**Q: Does the baseline use the training data at all?**
A: Yes, one number from it: the mean of the training targets. It uses none of the features.

**Q: The baseline predicts the same number every time. Isn't it trivially bad?**
A: No. It is optimal among all constant predictors, and it is optimal overall if the features carry no information. Our result is that on this data, the features carry no usable information, so the constant is the right answer.

**Q: Which mean, exactly — the training mean or the whole-dataset mean?**
A: The **training** fold's mean. Using the whole dataset's mean would peek at the test set. This distinction is not cosmetic; it is the entire reason the baseline's R² is −0.078 rather than 0.000 (section 2.5).

---

## 2. The metrics, hand-worked

### 2.1 The five-number example we will reuse

Five real-looking Grids values (the score is mean placement error in grid cells; **lower is better** for all three tests):

```
true scores  y = [ 0.000, 0.333, 0.500, 1.000, 2.390 ]
sum = 4.223          mean = 4.223 / 5 = 0.8446
```

### 2.2 MSE, RMSE, MAE

Definitions, with n = number of test sessions, yᵢ = true score, ŷᵢ = predicted score:

```
MSE  = (1/n) Σ (yᵢ − ŷᵢ)²        "mean squared error"      units: score²
RMSE = sqrt(MSE)                  "root mean squared error" units: score
MAE  = (1/n) Σ |yᵢ − ŷᵢ|          "mean absolute error"     units: score
```

**Units matter and get asked about.** MSE for Grids is in *squared grid cells*, meaningless to a human. RMSE and MAE are in *grid cells* — comparable to the score itself and to its SD of 0.500. So when reporting to a clinician, say RMSE or MAE, never MSE. We use MSE only internally, for tuning (section 6).

Four candidate models on the five numbers above:

| Model | predictions ŷ | MAE | MSE | RMSE |
|---|---|--:|--:|--:|
| Perfect | 0, 0.333, 0.5, 1.0, 2.39 | 0.0000 | 0.0000 | 0.0000 |
| Off by +0.2 each | 0.2, 0.533, 0.7, 1.2, 2.59 | 0.2000 | 0.0400 | 0.2000 |
| Predict the mean | 0.8446 ×5 | 0.6803 | 0.7012 | 0.8374 |
| Bad model | 1.5, 1.2, 0.2, 0.1, 0.3 | 1.1314 | 1.6540 | 1.2861 |

Two rows hand-worked, so you can reproduce them live:

```
PREDICT THE MEAN (0.8446 for all five)
  errors  (y − ŷ) :  −0.8446  −0.5116  −0.3446  +0.1554  +1.5454
  |errors|        :   0.8446   0.5116   0.3446   0.1554   1.5454   sum = 3.4016
  errors²         :   0.71335  0.26173  0.11875  0.02415  2.38826  sum = 3.50624
  MAE = 3.4016/5 = 0.68032   MSE = 3.50624/5 = 0.70125   RMSE = sqrt = 0.83741

BAD MODEL (1.5, 1.2, 0.2, 0.1, 0.3)
  errors          :  −1.5   −0.867   +0.3   +0.9   +2.09
  |errors| sum = 5.657                     -> MAE = 1.1314
  errors²  = 2.25 + 0.751689 + 0.09 + 0.81 + 4.3681 = 8.269789
  MSE = 1.653958   RMSE = 1.28606
```

### 2.3 When MAE and RMSE disagree: outliers

Squaring punishes large errors much harder than small ones. Two models, five sessions each:

```
Model E errors: [0.5, 0.5, 0.5, 0.5, 0.5]     -> MAE 0.500   MSE 0.25   RMSE 0.500
Model F errors: [0.0, 0.0, 0.0, 0.0, 2.5]     -> MAE 0.500   MSE 1.25   RMSE 1.118
```

**Identical MAE. RMSE 2.24× worse for F.** MAE says "these two models are the same." RMSE says "F is much worse." Both are right about different things: MAE measures the *typical* error; RMSE measures error while treating one big miss as far worse than several small ones.

Which do you want here? **Both, side by side — which is what `FoldResult` stores (`regression.py:51-54`).** Grids has a floor pile-up (31% of sessions score exactly 0.000) and a long right tail up to 2.39, so a model that mostly does fine but blows up on the rare high-error sessions would look acceptable in MAE and bad in RMSE. Seeing both catches that. (The published head-to-head tables print R² and RMSE only; MAE is computed and stored per fold but was not written into `results/headtohead_real.md`.)

### 2.4 R² — the important one

```
                SS_res       Σ (yᵢ − ŷᵢ)²
   R²  =  1 −  --------  =  1 − -----------------
                SS_tot       Σ (yᵢ − ȳ)²
```

- **SS_res** = "residual sum of squares" = how wrong *your model* is, squared and added up.
- **SS_tot** = "total sum of squares" = how wrong *the average* is, squared and added up.

So R² is literally **the fraction of the average-predictor's error that your model removed**. Plain reading: *the share of the score's variation your model accounts for, measured against just guessing the average.*

**The hand-worked table.** Same five numbers, mean 0.8446, so SS_tot = **3.5062** for every row:

| Model | predictions | SS_res | R² = 1 − SS_res/3.5062 |
|---|---|--:|--:|
| Perfect | exactly y | 0.0000 | **+1.0000** |
| Off by +0.2 each | y + 0.2 | 0.2000 | **+0.9430** |
| Predict the mean | 0.8446 ×5 | 3.5062 | **0.0000** |
| Bad model | 1.5, 1.2, 0.2, 0.1, 0.3 | 8.2698 | **−1.3586** |

```
SS_tot  = 0.71335 + 0.26173 + 0.11875 + 0.02415 + 2.38826 = 3.5062
off by +0.2 : SS_res = 5 x 0.2² = 0.2000   R2 = 1 − 0.2/3.5062    = +0.9430
mean        : SS_res = SS_tot = 3.5062     R2 = 1 − 3.5062/3.5062 =  0.0000
bad model   : SS_res = 2.25+0.751689+0.09+0.81+4.3681 = 8.269789
                                           R2 = 1 − 8.2698/3.5062 = −1.3586
```

Three things fall straight out of this table and you should state them in this order:

1. **R² = 0 is not "zero accuracy."** It is exactly the accuracy of predicting the average. That is why the row "predict the mean" gives 0.0000 — SS_res and SS_tot become the same sum.
2. **R² has no lower bound.** SS_res can be as large as you like. R² = 1 − (big)/(fixed) goes to minus infinity.
3. **R² is unitless.** It compares two sums of squares in the same units, so the units cancel. This is why you can put Grids (cells), Symbols (seconds) and Prices (percent) in one table — and why you must *also* report RMSE if anyone needs to know how wrong the model is in real units.

### 2.5 Why R² can be negative — the explanation the whole project rests on

**Negative R² means your model made larger squared errors than a constant would have.** That is all it means — not a bug, not an impossible value, not a sign flip. The number line:

```
   worse than guessing the average          |          better than guessing
 <-----------------------------------------(0)------------------------------>
 −3.0        −1.36        −0.089           0.0          +0.50          +1.0
  |            |            |               |             |              |
  |            |            |               |             |          perfect: every
  |            |            |               |             |          prediction exact
  |            |            |               |        removes half the average's error
  |            |            |          EXACTLY as good as predicting the average
  |            |       our real Grids result (Chronos + Ridge)
  |       the toy "bad model" in the table above
 plain LinearRegression on the raw 512 Chronos features (section 3)

Everything LEFT of 0 means: "you would have done better printing one number."
```

**Where does the negativity come from, mechanically?** Let `off` = (training mean − test mean), and take the constant predictor first:

```
SS_res  =  Σ (y_test − train_mean)²
        =  Σ (y_test − test_mean)²  +  n x off²
        =  SS_tot                   +  n x off²

so       R²  =  − off² / var(y_test)
```

The offset term is pure penalty and it can never help. Now the real Grids fold 1, every number verified:

```
train_mean = 0.519256      test_mean = 0.311700      off = 0.207556      off² = 0.043079
test variance = 0.172094   (test SD = 0.414842)      n_test = 188
SS_tot = 32.3536           SS_res = 40.4526

R² = 1 − 40.4526 / 32.3536 = −0.250326
and the shortcut:  − 0.043079 / 0.172094 = −0.250326      <-- identical, as it must be
```

So the constant predictor scored **−0.25 on that fold purely because those four children were better at Grids than the sixteen it trained on.** Nothing was overfitted; no model was involved. That is the honest core of the finding: under grouped splits with only 4 test participants, **R² = 0 is not the neutral point — slightly-below-0 is**, and our models land next to it.

### 2.6 The subtlety: which mean is the reference?

`sklearn.metrics.r2_score(y_true, y_pred)` computes SS_tot from **the mean of the `y_true` array it was handed** — i.e. each test fold's own mean. It does **not** use the global mean of all 956 sessions, and it has no way to know it.

Consequences you must be able to state:

- The baseline predicts the **train** mean; R² is measured against the **test** mean. Those differ, so **the baseline's R² is not 0** — it is 0 minus the offset penalty. Hence −0.0775 for Grids, not 0.0000.
- Small test folds make the two means differ more, so **negative values are easier to hit on small folds.** With 4 participants and ~190 sessions per fold, the offset penalty is not negligible.
- Fold R² values are therefore **not** all measured against the same yardstick. Averaging them is a convention, not an exact quantity. RMSE has no such reference, which is one reason we report both.
- If a fold's test scores happened to be nearly constant, SS_tot would be tiny and R² would swing wildly. On our data the per-fold test spreads are stable, so this is not driving our numbers — but knowing it separates "I ran sklearn" from "I know what sklearn did."

### 2.7 RMSE and R² can disagree — and did

Real Grids results, grouped CV (`results/headtohead_real.md` plus the baseline row computed above):

| Representation | R² | RMSE |
|---|--:|--:|
| Chronos (512) + Ridge | −0.089 ± 0.123 | 0.505 ± 0.059 |
| Hand-crafted (43) + Ridge | −0.065 ± 0.084 | 0.500 ± 0.061 |
| mean baseline | −0.078 ± 0.099 | 0.503 ± 0.059 |

By **R²**: hand-crafted (−0.065) > baseline (−0.078) > Chronos (−0.089). By **RMSE**: hand-crafted (0.500) < baseline (0.503) < Chronos (0.505). Same ordering here, but notice how tiny it is — **five thousandths of a grid cell** separates best from worst. R² makes the gaps look bigger than they are because it divides by a small SS_tot.

Why the two can disagree in general: RMSE compares a model to nothing, so it is an absolute error scale. R² compares a model to that fold's own mean, so a fold with unusually uniform scores (small SS_tot) can produce a very negative R² from an ordinary RMSE. Reporting only R² invites "your model is terrible"; reporting only RMSE invites "is 0.505 good?". **Report both, and say this: "the RMSE is 0.505 grid cells against a score whose SD is 0.500 — the model's typical error is the size of the whole spread of the score."** That lands harder than any R² value.

**Say it in your own words:** *"MSE is squared error, RMSE puts it back in the score's own units, MAE is the plain average error and ignores outliers, and R² is the share of the average-predictor's error that we removed — so R² = 0 means we tied the average and negative means we did worse."*

### 2.8 Drill

**Q: What does R² = 0.30 mean, in words a parent would understand?**
A: "Knowing the glucose lets us cut about 30% of the guessing error, compared to just assuming every child scores average."

**Q: Our R² is −0.089. How much worse than the average is that?**
A: Our squared errors add up to about 8.9% more than the average predictor's would on the same test sessions — measured against each fold's own mean. And the mean predictor itself scores −0.078 here, so we are about 1 percentage point behind a constant.

**Q: Can R² be below −1?**
A: Yes. The toy "bad model" above is −1.3586, and plain LinearRegression on our raw 512 features reaches about −3. There is no floor.

**Q: RMSE 0.505 versus score SD 0.500 — say what that means.**
A: The model's typical miss is as large as the entire spread of the score. Predicting nothing at all would be about equally accurate.

---

## 3. Overfitting, seen through the evaluation numbers

**Overfitting** in plain words: **the model memorized the training data instead of learning a rule, so it looks brilliant on data it has seen and useless on data it has not.** You detect it by the *gap* between the training score and the held-out score, not by any single number. A model can score near-perfectly on its training rows and below a constant on new participants; that gap is the diagnosis. (Honest caveat: our pipeline records only held-out scores — `regression.py` never scores the training split — so we quote the held-out side.)

### The LinearRegression blow-up, and why it happens

We deliberately keep unregularized `LinearRegression` in the model list (`regression.py:46`) **as a teaching exhibit**. On the raw 512-dimensional Chronos features under grouped CV it reaches **R² ≈ −3** (`docs/01_PIPELINE_DESIGN.md:71`, `results/README.md:64`) — roughly four times worse than printing a constant.

The mechanism, in numbers you already have:

- 512 features, ~765 training sessions — about 1.5 rows per free parameter.
- Ordinary least squares has no penalty, so it can use enormous positive and negative coefficients that cancel out on the training rows. Those cancellations are tuned to the training rows' exact noise; on a new participant they do not hold and the predictions fly off.
- The result is enormous prediction variance. Section 9 shows why prediction variance unrelated to the truth costs you R² directly.

The fix is `Ridge`, which penalizes the size of the coefficients (see `04_LINEAR_MODELS.md`). With alpha tuned per fold it shrinks toward predicting close to the mean, which is why its R² lands next to the baseline rather than at −3.

**The point to make out loud:** the −3 is not an embarrassment, it is *evidence* that the evaluation catches bad models. A protocol that scored every model the same would be suspicious. Ours gives −3 to the model that deserves −3.

**Say it in your own words:** *"Unregularized linear regression on 512 features memorizes the training children and scores about −3 on new ones. That's overfitting, it's why we use Ridge, and it shows our evaluation actually detects bad models."*

### Drill

**Q: How do you know Ridge at −0.089 is not also overfitting?**
A: Two ways. First, the inner loop picks alpha by held-out error, and if it picked a large alpha it is because shrinking helped. Second, an overfitting model produces wildly varying predictions; Ridge's predictions sit close to the mean, which is the opposite signature.

**Q: You have 512 features and 20 participants. Isn't overfitting inevitable?**
A: For an unpenalized model, yes — that is the −3. Regularization and PCA are exactly the tools for the case where features outnumber rows. Note the relevant count is 765 training **sessions**, not 20 participants, but the 20 participants do limit how much genuinely independent information exists.

**Q: Would more data fix the −3?**
A: It would shrink it, but it would not create a relationship. PCA down to 8–32 components moved the mean R² only from −0.052 to −0.045.

---

## 4. From one train/test split to k-fold cross-validation

**The simplest thing: one split.** Hold back some data, train on the rest, score once on the held-back part. The problem: **you get one number and it depends on which rows you happened to hold back.** With 20 participants, a single 4-participant test set could easily be the four with unusual baselines. You would not know.

### k-fold cross-validation

**Plain words: cut the data into k equal blocks; train k times, each time holding out a different block as the test set; average the k scores.** Every row is a test row exactly once, and every row is a training row k−1 times.

```
k = 5

               block 1   block 2   block 3   block 4   block 5
 round 1        TEST      train     train     train     train
 round 2        train     TEST      train     train     train
 round 3        train     train     TEST      train     train
 round 4        train     train     train     TEST      train
 round 5        train     train     train     train     TEST
                  |         |         |         |         |
                score1    score2    score3    score4    score5
                  \______________ average ______________/
                                    |
                          the reported number
```

Why bother: (1) **less luck** — one unlucky split cannot dominate, you average 5; (2) **every row is tested** — all 956 sessions contribute instead of ~190; (3) **you get a spread for free** — the 5 numbers show how stable the result is (section 7).

Cost: 5× the compute, and the 5 training sets overlap heavily (any two share about 75% of their sessions), so the 5 scores are **not independent measurements**. That matters in section 7.

Why k = 5 and not 10 or 20? With 20 participants, k = 5 holds out 4 per fold — the smallest test group we were willing to score. k = 20 (leave-one-participant-out) would give test sets of ~48 sessions from one child, where the offset penalty in section 2.5 dominates and per-fold R² becomes almost unreadable. k = 5 also matches the protocol the hand-crafted-feature arm used, which is what makes the two arms comparable.

**Say it in your own words:** *"We split the data into five blocks, train five times holding out a different block each time, and average. That way every session gets tested once and the answer doesn't hinge on one lucky split."*

### Drill

**Q: Does cross-validation produce a model, or a number?**
A: A number — an estimate of how well the procedure generalizes. Five different models get fitted and thrown away. If you needed a deployable model you would refit on everything afterwards.

**Q: Why not just use a big test set once?**
A: With 20 participants there is no "big" test set. Any single split wastes most of the data and gives one noisy number.

**Q: Is the seed 42 what determines the folds?**
A: Not for our outer split. `GroupKFold` in scikit-learn 1.9 defaults to `shuffle=False`, so it is deterministic — it orders participants by session count and assigns greedily to balance fold sizes. Seed 42 controls the models (Ridge, PCA) and the session-level `KFold(shuffle=True)` diagnostic variant, not the grouped split. This is a good detail to volunteer; it shows you read past the config file.

---

## 5. Grouped cross-validation — the most important design decision

### 5.1 The leak, concretely

Each participant contributed many sessions — minimum 32, median 47.5, maximum 67. If you split **by session** at random, child P07's Monday session lands in training and their Tuesday session lands in test.

Now the model has a shortcut. It does not need to learn anything about glucose; it only needs to learn *"sessions that look like P07's belong to a child who scores about 0.31."* Chronos embeddings of a child's own CGM traces are recognizable, because glucose patterns are personal. The model becomes a participant-identifier plus a lookup table of per-child averages, test scores go up, and **nothing was learned about glucose and cognition.** That is **leakage**: information about the test set reaching the model through the back door.

```
Child P07 contributed 47 sessions:   s1  s2  s3  s4  s5  ...  s47


  SESSION-LEVEL SPLIT (shuffle the rows)              *** LEAKY ***
  ---------------------------------------------------------------
    train:   s1      s3  s4      s6  s7  ...      <- P07 is here
    test:        s2          s5          s8 ...   <- P07 is ALSO here
  The model met P07 in training. It can recall P07's typical score.
  Question it answers: "predict another session from a child we already know."


  GROUPED SPLIT (whole participants move together)    *** HONEST ***
  ---------------------------------------------------------------
    train:   P01 P02 P03 P04 ... P16      16 participants, ~765 sessions
    test:    P17 P18 P19 P20               4 participants, ~191 sessions
  P07 is entirely on one side. All 47 of its sessions travel together.
  Question it answers: "predict a child we have never seen."
```

### 5.2 Which question do you actually want answered?

| Scheme | The question it answers | Is it the study's question? |
|---|---|---|
| Session-level `KFold` | "Given other sessions from this same child, predict this session." | **No.** And it flatters the model. |
| **`GroupKFold` on `subid`** | "Given 16 other children, predict a child we have never seen." | **Yes.** This is Aim 1. |

Both are implemented (`_make_splitter`, `regression.py:79-84`; `--cv session` switches). The gap between them is a diagnostic: if grouped ≈ 0 while session-level > 0, the apparent accuracy was per-child baseline, not a glucose effect. **Note carefully: we have not recorded a session-level number on the real data** — the reproduced diagnostic in `docs/00_PROJECT_OVERVIEW.md:62` is from synthetic data. If asked, say: *"That variant is implemented and one flag away, but I have not run it on the real data, so I won't quote a number."* Do not improvise one.

### 5.3 The real folds

20 participants, 5 folds, 4 held out each time. All 956 sessions:

| Fold | test participants | test sessions | train sessions |
|---|--:|--:|--:|
| 1 | 4 | 189 | 767 |
| 2 | 4 | 193 | 763 |
| 3 | 4 | 191 | 765 |
| 4 | 4 | 191 | 765 |
| 5 | 4 | 192 | 764 |
| | **20** | **956** | |

The fold sizes are not identical (189 to 193) because `GroupKFold` balances *sessions* while keeping participants whole, and participants have 32 to 67 sessions each. It cannot do both perfectly.

**A precision point worth volunteering.** Those are the counts for all 956 sessions. Each target drops its missing scores *before* splitting (`regression.py:183-184`), so the per-target folds differ:

```
grids   (n=951):  188 191 192 190 190     (5 sessions have no Grids score)
symbols (n=920):  170 190 188 187 185     (36 missing)
prices  (n=936):  187 189 187 187 186     (20 missing)
```

Because `GroupKFold` balances fold sizes using each participant's session count, re-splitting a masked array can shift both the fold sizes and which participants land in which fold. So the Symbols folds are not the Grids folds with 31 rows removed. Always say which target's folds you are quoting.

### 5.4 The guard in the code

`regression.py:113`:

```python
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
    "subject leakage between train and test!"
```

This runs on **every outer fold of every model of every target** — 5 folds × 4 models × 3 targets on a standard Arm A run. If a participant ever appeared on both sides, the run would crash rather than quietly report an inflated number. That is the answer to "how do you know there is no leakage": not by inspection, but because the code refuses to continue.

**Say it in your own words:** *"We split by child, not by session. If a child's sessions were on both sides, the model could just memorize that child's usual score instead of learning anything about glucose. Splitting by child forces it to answer the real question: predict a child we've never seen."*

### 5.5 Drill

**Q: Why is session-level splitting wrong here rather than just optimistic?**
A: Because it answers a different question. The study asks whether glucose predicts cognition, which requires generalizing to new children. Session-level splitting lets per-child baseline masquerade as a glucose effect.

**Q: Only 15% of the Grids variance is between participants. Doesn't that mean leakage barely matters?**
A: It means leakage buys less than people assume — that is a real correction to our older documents (`01_FACT_SHEET.md`, section D). It also means the difficulty is not mainly that children differ; it is that 62–92% of the variance is session-to-session variation that glucose does not explain. But 15% of variance is still a free 0.15 of R² for a model that recognizes children, and 0.15 is larger than any effect we are looking for.

**Q: Why not leave-one-participant-out?**
A: Test sets of one child (~48 sessions) make the train/test mean offset from section 2.5 dominate, so per-fold R² becomes very noisy. 5 folds of 4 is also the protocol the hand-crafted arm used, which keeps the comparison fair. (On id collisions: Cohort 1 ids are `2xxxxx` and Cohort 2 ids are `3xxxxx`, so they cannot overlap, and the `assert` would catch it if they did.)

---

## 6. Nested cross-validation — two loops, two jobs

### 6.1 The cheat we are avoiding

Ridge needs a setting called alpha (how hard to shrink the coefficients). We try five values: 0.1, 1, 10, 100, 1000. Suppose we fit all five on the training data and report whichever scores best **on the test fold**. That is cheating, and here is the honest way to say why: **the test fold stopped being held out the moment it was used to make a choice.** You picked the setting that flatters this particular test set. Reported as a generalization estimate, that number is biased upward — with five settings and small folds, easily by several hundredths of R², which is larger than every effect in this study.

### 6.2 The fix: an inner loop that never sees the test fold

```
OUTER loop  —  GroupKFold(5) on participant.  Job: produce an HONEST number.
INNER loop  —  GroupKFold(3) on the training participants only.  Job: CHOOSE alpha.
```

The test fold is opened exactly once per outer fold, after all choices are locked.

```
=========================== OUTER FOLD 3 of 5 ============================

  20 participants
      |
      +-- 16 TRAIN participants  (765 sessions) ----+     +-- 4 TEST participants
                                                    |     |   (191 sessions)
                                                    |     |   SEALED. Not touched
                                                    |     |   until the last step.
      INNER: GroupKFold(3) on those 16 participants |     |
      -----------------------------------------------     |
        inner 1 / 2 / 3:  train ~11 parts | validate ~5   |
                                                          |
        for alpha in {0.1, 1, 10, 100, 1000}:             |
            fit on each inner-train, score each inner-    |
            validation with neg_mean_squared_error,       |
            average the 3 numbers                         |
        = 5 alphas x 3 folds = 15 fits                    |
                                                          |
        pick the best alpha, refit it on all 16 train      |
        participants  = 1 more fit  (16 for this fold)     |
                        |                                  |
                        +----------------> predict ------- +
                                                |
                                    score ONCE: R², RMSE, MAE
                                    (fold 3's contribution)

Repeat for outer folds 1,2,4,5. Average the 5 R² values. Report mean ± sd.
==========================================================================
```

The inner splitter is **also** grouped (`regression.py:122`; `groups=groups[train_idx]` is passed to `GridSearchCV.fit` at lines 127-128). Choosing alpha on a session-level inner split would pick an alpha suited to a leaky task.

### 6.3 The fit count — be able to derive this live

```
Baseline  : empty grid -> no GridSearchCV -> 1 fit per outer fold  x5  =   5
Ridge     : 5 alphas x 3 inner folds = 15, +1 refit = 16 per fold  x5  =  80
SVR       : 3 C x 2 gamma = 6 settings x 3 inner = 18, +1 = 19     x5  =  95
Linear    : empty grid -> 1 fit per outer fold                     x5  =   5
                                                            per target = 185
                                                        x 3 targets   = 555 fits
```

**555 model fits per Arm A run.** `LinearRegression` and the baseline have empty grids, so they take the `else` branch at `regression.py:130-131` and skip the inner loop — nothing to tune.

⚠️ **XGBoost.** `regression.py:171-179` tries to import `XGBRegressor` and adds it with a 2×2×2 = 8-setting grid, wrapped in `try/except Exception: pass`. **xgboost is not installed in this environment** (`01_FACT_SHEET.md`, section J), so it was silently skipped in every result we have. If asked "did you try gradient boosting?", the true answer is **no** — the code would have, the package is absent, and no error was printed. Say that before someone finds it by reading the file.

### 6.4 Why the inner loop optimizes MSE while we report R²

The inner `GridSearchCV` uses `scoring="neg_mean_squared_error"` (`regression.py:124`). The outer loop reports R², RMSE and MAE. Different metrics, on purpose, for two reasons:

1. **Within one fold they rank identically.** On a fixed set of rows, R² = 1 − MSE/var(y) and var(y) is a constant, so minimizing MSE and maximizing R² pick the same winner. The name `neg_mean_squared_error` exists only because scikit-learn requires "higher is better" for every scorer, so it negates MSE.
2. **Across folds they are not interchangeable, and MSE is safer for tuning.** R²'s denominator changes from inner fold to inner fold (section 2.6), so averaging R² mixes yardsticks. MSE has a fixed meaning in score units, so averaging it is cleaner.

The general principle: **the quantity you optimize and the quantity you report are allowed to differ.** Optimize something smooth and convenient (squared error); report something interpretable (RMSE in score units, R² against the average). Arm B does the same more visibly — it trains on MSE over z-scored targets and is reported in raw-score R².

**Say it in your own words:** *"Two loops. The inner loop picks alpha using only training children, the outer loop scores on children neither loop has seen. Picking alpha on the test fold would mean the test fold isn't held out any more, and the number would be flattering rather than honest."*

### 6.5 Drill

**Q: Different outer folds may pick different alphas. Isn't that inconsistent?**
A: It is expected and correct. Each outer fold evaluates the whole *procedure* "tune alpha on training data, then predict," not one fixed model. If you wanted a single deployable model, you would tune once on everything at the end — but you would not report that model's training-time score.

**Q: The inner loop is 3-fold, not 5. Why?**
A: It runs inside 16 participants. Three grouped folds leaves about 5 participants per inner validation set. Going to 5 inner folds would leave about 3 participants per validation set, making the alpha choice noisier, and would raise the fit count by two-thirds.

**Q: Does the scaler leak?**
A: No. `StandardScaler` is the first step of a `Pipeline` (`regression.py:116-120`), so it is fitted on each training split only and applied to the test split. Same for PCA when enabled — that is stated explicitly in the docstring at lines 104-106.

---

## 7. Reading "mean ± sd across 5 folds" honestly

Every number we report looks like `−0.089±0.123`. Say precisely what that is:

> **It is the average of 5 fold scores and the standard deviation of those same 5 fold scores — a description of how much the answer moved between folds. It is not a confidence interval and not a margin of error.**

Three reasons it cannot be treated as a confidence interval: **n = 5**, so the standard deviation is itself very uncertain; **the folds are not independent**, since any two outer training sets share about 75% of their sessions, and independence is the assumption behind dividing by √n; and **each fold's R² is measured against a different yardstick**, its own test mean (section 2.6), so you are averaging quantities with different denominators.

**How large is the fold-to-fold movement, really?** Use the verified baseline numbers, where we have every fold:

```
Grids, mean baseline, per-fold R²:
    −0.2503   −0.1269   −0.0010   −0.0062   −0.0030
    mean −0.0775      sd 0.0988      range −0.250 to −0.001
```

The worst fold is **250×** the best fold, and both are the same model — a constant. All of that movement is which four children happened to be held out. Chronos + Ridge on Grids has sd 0.123, so its fold values move over a comparable band.

**How to state uncertainty honestly:**

- ✅ *"Grids came out at −0.089 on average, and the five folds ranged over roughly a ±0.12 band — so fold-to-fold movement is larger than any difference between our models. The spread is driven by which four children are held out; with 20 participants that is unavoidable. For scale, the mean baseline itself moves from −0.001 to −0.250 across the same folds."*
- ❌ *"−0.089 ± 0.123, so the true value is between −0.21 and +0.03."* That is a confidence-interval reading and it is not supported.
- ❌ *"Chronos (−0.089) is worse than hand-crafted features (−0.065)."* The gap is 0.024, one fifth of the fold spread. Say **"the two are indistinguishable at this sample size."**

If someone demands an interval: *"With 5 correlated folds and 20 participants I would not quote one. I would bootstrap over participants, which we have not done. What I can say is that the fold-to-fold spread is about ±0.12 and every fold is at or below zero."*

**Say it in your own words:** *"The ± is the spread across the five folds, not a confidence interval. With five folds and four children each, that spread is big — so I only treat differences bigger than the spread as meaning anything."*

### Drill

**Q: −0.089 ± 0.123. Could the true value be positive?**
A: I would not claim that from this. The spread crosses zero, but every one of the checks points the other way: the mean is negative on all three scores, the within-participant version is also negative, and the real number sits inside the scrambled-score range. The honest statement is "at or below the mean baseline."

**Q: Why not report the standard error, 0.123/√5 = 0.055?**
A: √n assumes independent samples. The five training sets overlap by about 75%, so 0.055 would understate the real uncertainty.

**Q: Would 10 folds shrink the spread?**
A: No, it would widen it. Each fold would hold out 2 participants instead of 4, so test sets get smaller, the train/test mean offset grows, and per-fold R² gets noisier. The real limit is 20 participants, and no choice of k changes that.

---

## 8. Multiple comparisons — the trap we happened to avoid

### 8.1 The idea, in plain words

**If you look at enough things, one of them will look good by luck.** Flip 20 coins five times each and something will come up all heads. Nothing was learned; you just looked 20 times. The same applies to models: every configuration you try is another look, and reporting the best one as if it were the only one you tried is reporting a lucky draw.

### 8.2 How many looks did we take?

Honest count of R² cells in the real-data result files:

| Source | configurations × targets | cells |
|---|---|--:|
| `sweep_real.md` — 5 checkpoints, 4 windows, 2 poolings, 6 PCA values, 3 target norms | 20 × 3 | 60 |
| `subgroups_real.md` — 5 glucose regimes | 5 × 3 | 15 |
| `headtohead_real.md` — 3 representations | 3 × 3 | 9 |
| `headtohead_real_centered.md` — 2 representations | 2 × 3 | 6 |
| | | **≈ 90** |

Each of those cells is itself the **best of up to four models** (baseline, Ridge, SVR, Linear), each tuned over 5 or 6 settings. The sweeps are one-factor-at-a-time rather than a full grid, and the reference configuration (bolt-small, full window, mean pooling, no PCA, raw targets) repeats in five tables, so the number of *distinct* configurations is smaller than 90 — but the number of *looks* is around 90 either way.

**Why that matters.** If each look had a 1-in-20 chance of clearing a "looks promising" bar by luck alone, 90 looks would be expected to produce **about four or five lucky passes.** Present one of those as the finding and you have published noise.

### 8.3 Why it does not bite us — and where it will

**It does not bite us here, for a simple reason: not one of the ~90 cells came out above zero.** The most favorable anywhere is Prices with within-participant centering at **−0.002**. Multiple comparisons inflate the *best* result you find; when the best is still at or below the bar, there is nothing to inflate. The sweeps are evidence in the other direction — 90 attempts, flat every time, including where an effect is most physiologically plausible (sessions with a reading under 70: Grids −0.187, Symbols −0.094, Prices −0.090).

**Where it will bite: any future search for a subgroup where the effect "shows up."** Slice by age, sex, cohort, time of day, overnight versus daytime, and you will eventually find a slice with a positive R². That slice will be small, the positive value will sit inside the fold-to-fold spread, and it will not replicate. Name all four defenses:

1. **Write the hypothesis down before running it.** "Sessions with a reading below 54" decided in advance is a test; decided afterwards it is a story.
2. **Count the looks and adjust the bar.** With 20 comparisons a 1-in-20 threshold becomes 1-in-400 (Bonferroni), or use false-discovery-rate control. See `10_RIGOR_AND_STATS.md`.
3. **Re-run the scrambled-score check inside the subgroup.** A small subgroup has its own chance range; the whole-data one does not apply.
4. **Hold participants back.** With 20 there is barely room, which is itself the honest limitation to state.

**Say it in your own words:** *"We looked at roughly ninety configurations. If you look ninety times, a few will look good by luck — but every single one of ours came out at or below the guessing baseline, so there's no lucky winner to be fooled by. The risk arrives the moment we start hunting for a subgroup where it does work."*

### 8.4 Drill

**Q: You tried five checkpoints and six PCA values and reported the best. Isn't that exactly the trap?**
A: It would be, if the best were positive. The best PCA setting moved the mean R² from −0.052 to −0.045 — still below the bar, so there is no winner to have cherry-picked. We report the whole sweep table, not just its best row, which is the other half of the defense.

**Q: Should you correct for multiple comparisons in the write-up?**
A: For the scrambled-score p-values across three scores, yes — and it changes nothing, because they are 1.000, 1.000 and 0.995. Any correction only pushes them further from significance.

**Q: Doesn't running 90 configurations look like fishing?**
A: It would if we reported one. We reported all of them, in tables, and the finding is the pattern: nothing works anywhere. Liuyi explicitly asked us to search hard for signal; documenting where we looked is what makes "we didn't find it" credible.

---

## 9. What "worse than the baseline" actually implies, mechanically

A negative R² is not mystical. Decompose the model's squared error on a test fold. Write `off` = (mean prediction − mean true score), and let `pred` be the model's outputs:

```
MSE  =  off²                     <-- the level is shifted
     +  var(y_test)              <-- unavoidable: the score's own spread
     +  var(pred)                <-- the model's wobble COSTS you
     −  2·cov(pred, y_test)      <-- the model's wobble PAYS you, only if correlated

R²   =  −( off²  +  var(pred)  −  2·cov(pred, y_test) ) / var(y_test)
```

Read the last line slowly. **A model's variation is a liability by default.** It only becomes an asset through the covariance term — only if the model moves *in the same direction as the truth*. If `cov ≈ 0`, every bit of wobble is pure added error, and R² is negative by exactly the amount of wobble you added.

That is our situation. The strongest raw correlation between any simple glucose statistic and any score in this dataset is **r = −0.092**, giving **r² = 0.0085**, i.e. **0.85% of the variance** (`01_FACT_SHEET.md`, section E). So the covariance available to a model built on simple summaries is nearly zero. A foundation model could in principle find more than a summary statistic can — that is the whole reason to try one — and the finding is that it did not, while the wobble term is whatever the model chooses to produce.

So "worse than the baseline" means three things at once. **(1) The model learned a real relationship in the training participants** — Ridge fit non-zero coefficients, so something in the 512 numbers did correlate with the score among those 16 children. **(2) That relationship does not hold in the held-out children** — it was a property of those 16, not of glucose and cognition. **(3) So the model's predictions move, and the movement is unrelated to the truth** — movement without correlation is added error, so the model does not merely fail to help, it **actively hurts** relative to printing a constant.

Plus the offset term, which is not the model's fault at all: with only 4 test participants, their average score differs from the training average, and that alone cost the constant predictor −0.0775 on Grids (section 2.5).

**The two-part sentence to say, because it is both honest and precise:**

> *"Most of our negative R² is the mean baseline's own negative R² — the four held-out children just have a different average score than the sixteen we trained on, and that costs about −0.08 on Grids before any model is involved. On top of that, the model adds roughly 0.01 of its own harm: it learned something in the training children that does not transfer, so its predictions move around for reasons unrelated to the true score, and that movement is added error."*

**Say it in your own words:** *"Negative R² means the model's predictions wobble in a direction that has nothing to do with the truth, so the wobble is pure added error. A flat line would have been better."*

### Drill

**Q: If the model is worse than a constant, why not just clamp it to predict the mean?**
A: You can, and that is essentially what a very large Ridge alpha does — shrink the coefficients toward zero until the model is nearly constant. The point of reporting the unclamped number is scientific: the finding is "glucose does not predict the score," not "we found the best possible predictor."

**Q: Is negative R² evidence of a bug?**
A: It is worth checking, and we checked. The same embeddings and the same grouped CV predict glucose **variability** at R² = 0.462, so the machinery extracts real information when it is present (`10_RIGOR_AND_STATS.md`). One honest caveat: mean glucose scores only 0.070, because Chronos rescales each series internally and largely discards absolute level. So our Chronos result speaks to glucose *shape*. The saving grace is that the 43 hand-crafted features do encode absolute level and were flat too.

**Q: Could the sign of the target be inverted, turning −0.089 into +0.089?**
A: No. R², RMSE and MAE are all invariant to negating the target — flipping the sign flips predictions and truth together. All three scores are "lower = better" and we keep them raw (`config.py`, `PRICES_IS_INVERTED = False`). Sign only affects how you word the interpretation.

---

## 10. What would make us believe there IS a relationship

Pre-commit to this **before** looking, and say so — committing in advance is what separates a test from a story.

| # | Criterion | The literal bar on our data |
|---|---|---|
| 1 | **Positive mean R² on held-out participants** | Grouped-CV mean R² clearly above 0 — not −0.002, which is a tie. |
| 2 | **Above the mean baseline in the same folds** | Above **−0.078** (Grids), **−0.050** (Symbols), **−0.004** (Prices) — the constant predictor's own score, which is the true neutral point under grouped CV. |
| 3 | **Consistent in sign across all 5 folds** | All five fold R² values positive. One good fold out of five is which children got held out. |
| 4 | **Larger than the fold-to-fold spread** | Mean R² greater than its own sd. For Grids that means beating about 0.12. Anything smaller is inside the noise. |
| 5 | **Survives the scrambled-score check** | Above the 95th percentile of 200 label shuffles: **−0.027** (Grids), **−0.028** (Symbols), **−0.025** (Prices) under the fixed `PCA(32) → Ridge(α=10)` model (`results/rigor_real.md`). |
| 6 | **Visible in real units** | A meaningful drop in RMSE. Grids baseline RMSE is 0.503 grid cells against a score SD of 0.500; a model at 0.50 has changed nothing a clinician could use. |
| 7 | **Not an artifact of the variable window** | The strongest raw correlation with Grids is with the *number of readings* (r = −0.077), not glucose. Any positive result must survive fixing the lookback to a uniform length. |
| 8 | **Replicates somewhere** | Same direction on a second target, or on participants never used in development. With 20 participants there is little room — a limitation to state, not hide. |

Where we actually stand: criterion 1 fails on every score, criterion 2 fails by roughly 0.01, criterion 5 gives p = 1.000 / 1.000 / 0.995, and criterion 6 shows a 0.505 versus 0.503 RMSE difference. Nothing is close.

**What the criteria are for:** they turn "we didn't find anything" from an excuse into a measurement. Anyone can say a result is negative. Saying *"here is exactly what would have convinced me, here are the numbers, and here is how far short they fall"* is the version that holds up.

**Say it in your own words:** *"To believe there's a real relationship I'd want a positive R² on held-out children, positive in all five folds, bigger than the fold-to-fold spread, above the mean baseline in the same folds, and above the range you get from scrambled scores. We hit none of those, and we're not close."*

### Drill

**Q: Isn't it convenient to define the criteria after the fact?**
A: It would be, which is why they are written here as the bar for the *next* run — the subgroup and window-length work. For the results we already have, the honest framing is that they fail the most basic version of the test: positive R² on held-out participants.

**Q: What if we got R² = 0.05 with sd 0.15?**
A: That fails criteria 3 and 4. I would call it inconclusive and report it as inconclusive, not as a finding.

**Q: What single change would most improve the chance of finding something real?**
A: Giving every session the same span of glucose. Right now the input looks like everything since that child's previous test, so it ranges from 3 readings to 288 — a 96× spread — and the length itself carries time-of-day information. That is the one thing we have not been able to try, and it needs the uncut recording; Phil's script would tell us whether re-cutting is possible.

---

## The five hardest questions on evaluation

**1. "I cannot understand what this 'predict the average' baseline does."** *(Liuyi's actual words. This is the one to have memorized.)*

> It is scikit-learn's `DummyRegressor(strategy="mean")` — a model with no inputs at all. During training it computes one number, the average cognitive score across the training participants' sessions. At test time it prints that number for every session, ignoring the glucose completely. On Grids fold one that number is 0.519, repeated for all 188 test sessions. We include it for two reasons. First, an error number alone is uninterpretable: "average error 0.41" means nothing until you know what error you get for free. Second, R² is *defined* against this predictor — R² = 1 − (our squared error)/(the average predictor's squared error). So R² = 0 means we tied it, positive means we beat it, negative means we did worse. Our three scores come out at −0.089, −0.056 and −0.012, and the baseline itself comes out at −0.078, −0.050 and −0.004 in the same folds. We are tied with a constant, within about one hundredth.

**2. "Your R² is negative. Is your code broken?"**

> Negative R² just means our squared errors exceeded the average predictor's. Two things produce it here, and I can separate them. Most of it is not the model: because we hold out whole children, the four test children have a different average score than the sixteen training children, and that offset alone gives the constant predictor R² = −0.078 on Grids. I can show the arithmetic — training mean 0.5193, test mean 0.3117, offset 0.2076, test variance 0.1721, and −0.2076²/0.1721 = −0.2503, which is exactly fold one's baseline R². The remaining 0.01 is the model's own contribution: it learned a relationship in the training children that does not transfer, so its predictions vary for reasons unrelated to the true score, and that variation is added error. As for broken code: the same embeddings and the same folds predict glucose variability at R² = 0.462, so the pipeline extracts real information when it is there.

**3. "Why split by participant? Wouldn't you get better numbers splitting by session?"**

> Yes, and that is exactly why we do not. Each child contributed 32 to 67 sessions. Split by session and the same child appears in training and test, so the model can identify the child from their glucose pattern and recall their typical score instead of learning anything about glucose. Between-participant differences are 15% of Grids variance and 38% of Symbols variance, so that shortcut is worth up to 0.38 of R² — larger than any effect we are looking for. Grouped splitting moves all of a child's sessions together, so the model has to answer the study's actual question: predict a child we have never seen. There is an assertion at `regression.py:113` that crashes the run if a participant ever lands on both sides.

**4. "You reported the best of many configurations. How is that not fishing?"**

> It would be, if the best were positive. We ran about ninety R² cells on real data — five checkpoints, four window lengths, two poolings, six PCA values, three target normalizations, five glucose-regime subgroups, three representations, and a within-participant version — across three scores. If each look had a one-in-twenty chance of a lucky pass, ninety looks would produce four or five by chance. But every single cell came out at or below zero; the most favorable anywhere is −0.002. There is no lucky winner to have cherry-picked, and we publish the whole sweep table rather than its best row. Where the risk is real is any future hunt for a subgroup where the effect appears, and for that I would pre-register the hypothesis, correct the threshold for the number of comparisons, and re-run the scrambled-score check inside the subgroup.

**5. "−0.089 ± 0.123 — the interval crosses zero. So you cannot rule out a positive effect."**

> Correct, and I would not claim to rule it out. But the ± is not a confidence interval: it is the standard deviation of five fold scores, the folds share about 75% of their training sessions, and each fold's R² is measured against a different test mean. I would not turn it into an interval without bootstrapping over participants, which we have not done. What I can say is that everything points one way. Three scores, all negative. Both representations, negative. The within-participant version, negative. Every glucose-regime subgroup, negative. And the real result sits inside the range from 200 scrambled-score runs, at the bad end of it. The precise claim is: on 20 participants and 956 sessions, this approach is no more accurate than guessing the average — not that no effect exists anywhere.

---

## Ten-sentence summary

1. An error number means nothing without a comparison point, so every result is measured against `DummyRegressor(strategy="mean")` — a model with no inputs that prints the training set's average score for every test session.
2. R² = 1 − SS_res/SS_tot is literally the fraction of that average predictor's squared error that our model removed, so R² = 0 means we tied it, +1 means perfect, and negative means we did worse — there is no lower bound.
3. We report RMSE and MAE alongside R² because R² is unitless while RMSE and MAE live in the score's own units, and because MAE ignores outliers that RMSE punishes.
4. Unregularized `LinearRegression` on the raw 512 Chronos features reaches R² ≈ −3, which both demonstrates overfitting and proves our evaluation can detect a bad model.
5. We use 5-fold cross-validation so that all 956 sessions get tested once and the answer does not hinge on one lucky split.
6. The folds are grouped by participant, not by session, because each child contributed 32 to 67 sessions and a session-level split would let the model recognize the child and recall their average score instead of learning about glucose; `regression.py:113` asserts this on every fold.
7. Hyperparameters are chosen in a 3-fold grouped inner loop on the training participants only, so the outer 5-fold estimate stays honest — 555 model fits per Arm A run, and XGBoost is in the code but was never installed, so it never ran.
8. "mean ± sd across 5 folds" is a spread across folds, not a confidence interval: the mean baseline alone moves from −0.001 to −0.250 across the same folds, so we do not read differences smaller than about 0.1 as real.
9. We ran roughly ninety configurations, which is enough looks that something should have looked good by luck — but every cell came out at or below zero, so there is no lucky winner, and the multiple-comparison risk arrives only when someone starts hunting for a subgroup that works.
10. The honest headline: on 20 participants and 956 sessions, the Chronos pipeline is no more accurate than guessing the average — the models sit within about 0.01 of R² of a constant predictor, which itself sits slightly below zero because holding out whole children shifts the average.

---

*Cross-references: `01_FACT_SHEET.md` (every number with its derivation), `02_ML_FROM_ZERO.md` (features, targets, what fitting means), `04_LINEAR_MODELS.md` (Ridge, SVR, the alpha grid, why regularization is mandatory), `10_RIGOR_AND_STATS.md` (the 200-shuffle check, the glucose-property control, correcting for many comparisons).*

*Provenance: numbers come from `results/headtohead_real.md`, `results/headtohead_real_centered.md`, `results/sweep_real.md`, `results/subgroups_real.md`, `results/rigor_real.md`, and `cgm_tsfm/regression.py`. The mean-baseline R² tables in section 1.3, the per-target fold sizes in section 5.3, and the fold-1 decomposition in section 2.5 were computed on 2026-07-30 with `DummyRegressor(strategy="mean")` under the same `GroupKFold(n_splits=5)` protocol, reproducible from `cgm_tsfm.data.load_real_data()` in about two seconds.*
