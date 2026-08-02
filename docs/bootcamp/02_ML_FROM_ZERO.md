# 02 · Machine Learning From Zero

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you.**
> 1. You can explain, without notes, what our program actually *does*: turn 512 numbers into one predicted test score.
> 2. You can answer "why can't you just score the model on the data you trained it on?" with a concrete example, not a slogan.
> 3. You can say which of our numbers is a parameter, which is a hyperparameter, and which set of data chose it.
> 4. You can explain both failure directions — plain linear regression at R² ≈ −3, and Ridge at α=1000 collapsing to the average — and name which one we live in.
> 5. You can say why a fancier model (our neural head, −0.168 to −0.281) lost to the simple one (−0.012 to −0.089) without sounding defensive about it.

Companion pages: every number here is cross-checked against **`01_FACT_SHEET.md`**. How the glucose becomes numbers is **`03_PREPROCESSING.md`**. Ridge and SVR in detail are **`04_LINEAR_MODELS.md`**. R², RMSE, cross-validation mechanics are **`05_EVALUATION.md`**. The neural head is **`06_DEEP_LEARNING_CORE.md`**. Rapid-fire practice is **`12_EXAM_DRILL.md`**.

---

## The build order

Read top to bottom. Each box depends on the ones above it.

```
 1. What ML is  ──►  2. X and y  ──►  3. regression vs classification
                                              │
                                              ▼
 4. train / test split  ──►  5. generalization, train / validation / test
                                              │
                                              ▼
 6. parameter vs hyperparameter  ──►  7. overfitting & underfitting
                                              │
                                              ▼
 8. curse of dimensionality  ──►  9. regularization (one idea, many names)
                                              │
                                              ▼
10. loss vs metric  ──►  11. inductive bias / no free lunch
                                              │
                                              ▼
                    12. what a model cannot do
```

---

## 1 · What machine learning is

**Plain words.** In ordinary programming you write the rule and the computer applies it. In machine learning you supply examples of input paired with the right answer, and a fitting procedure searches for a rule that reproduces those answers. **Supervised learning** is that setup — "supervised" just means every training example comes with its answer attached.

The useful mental picture is **function fitting**. You want a function `f` that takes an input and returns a number. You do not write `f` by hand. You pick a *shape* for `f` (a weighted sum, a decision tree, a small network), then adjust the numbers inside that shape until it fits the examples.

Ours is:

```
f( 512 numbers describing one child's glucose before one test )  ->  one predicted score
```

**Tiny worked example.** Suppose `f` is a weighted sum (this is exactly what Ridge is). With only three features it looks like:

```
predicted_grids = bias + w1*x1 + w2*x2 + w3*x3
```

Say the fitting procedure lands on `bias = 0.478`, `w1 = 0.02`, `w2 = -0.01`, `w3 = 0.00`, and a session's features are `x1 = 1.5, x2 = -0.4, x3 = 2.0`:

```
predicted = 0.478 + 0.02(1.5) + (-0.01)(-0.4) + 0.00(2.0)
          = 0.478 + 0.030 + 0.004 + 0
          = 0.512
```

Notice the bias, 0.478. That is the average grids score across all 956 sessions. If every weight were exactly zero, `f` would return 0.478 no matter what the glucose looked like — that is the "guess the average" model, and it is the thing everything else in this project is measured against. Our real weights turned out so small that the model barely departs from that.

**In this repo.** `cgm_tsfm/regression.py` builds a three-step pipeline — `StandardScaler` → optional `PCA` → the model — then calls `.fit(X_train, y_train)` and `.predict(X_test)`. `.fit` is the search. `.predict` is applying the found rule.

One thing that is *not* being fitted: Chronos. Its weights are downloaded already trained and never change. `cgm_tsfm/encoders.py` runs it under `torch.no_grad()`. Chronos is a fixed translator from glucose readings to 512 numbers; the only thing we learn is what to do with those 512 numbers.

**Say it in your own words:** "We give the program 956 examples of glucose-before-a-test paired with the score that followed, and it searches for a rule that turns the glucose into the score."

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. Is Chronos being trained?**
A. No. It is frozen — loaded pre-trained, run inside `torch.no_grad()`, weights never updated. Only the regressor on top of its output is fitted.

**Q. What is the "supervised" part?**
A. Every one of our 956 training examples arrives with the answer attached: the child's actual score on that session. Nothing has to be labelled by hand later.

**Q. If all the learned weights came out zero, what would the model predict?**
A. The training set's average score, every time — 0.478 for grids, 1.846 for symbols, 41.59 for prices. That is the comparison baseline, not a bug.

</details>

---

## 2 · Features (X) and target (y); one row = one example

**Plain words.**
- A **feature** is one input number. All the features for all the examples stacked up form the matrix conventionally called **X**.
- The **target** (also called the **label**) is the number you are trying to predict. All the targets stacked up form the vector conventionally called **y**.
- **One row of X is one example**, and it lines up with one entry of y. Row 7 of X and entry 7 of y describe the same thing.

For us, one example = **one cognitive test session**. Not one child. Not one day. One session. A child who tested 47 times contributes 47 rows. Getting this unit right matters — the supervisor has already corrected this project once for saying "patients" where the number referred to sessions.

**The actual shapes.**

```
                          X  (Chronos features)                         y  (one target)
         col 0     col 1     col 2    ...   col 511                    grids score
       +---------+---------+---------+-----+---------+               +-------------+
row  0 | -0.31   |  1.02   |  0.44   | ... | -0.07   |               |    0.000    |  <- session  1, child 2xxxxx
row  1 |  0.85   | -0.12   | -1.33   | ... |  0.29   |               |    0.667    |  <- session  2, child 2xxxxx
row  2 |  0.14   |  0.63   |  0.08   | ... | -0.41   |               |    0.333    |  <- session  3, child 2xxxxx
 ...   |  ...    |  ...    |  ...    | ... |  ...    |               |     ...     |
row955 |  0.02   |  0.77   |  0.10   | ... | -0.55   |               |    0.500    |  <- session 48, child 3xxxxx
       +---------+---------+---------+-----+---------+               +-------------+

  shape (956, 512)  =  489,472 individual numbers                     shape (956,)
```

The 512 columns are not glucose values. They are abstract numbers Chronos produces. Nobody can point at column 137 and say what it means — that is a real cost of this approach, and `03_PREPROCESSING.md` walks through where they come from.

**The hand-crafted alternative** is the same picture with 43 columns instead of 512: `(956, 43)`. Those columns *do* have names — `gluc_mean`, `gluc_std`, `gluc_time_below`, and so on (`cgm_tsfm/handcrafted.py`).

**Tiny worked example — why the row counts differ per target.** We have 956 sessions, but not every session has all three scores:

```
grids   : 951 sessions have a score   ->  X used is (951, 512),  y is (951,)   [5 missing]
symbols : 920 sessions have a score   ->  X used is (920, 512),  y is (920,)   [36 missing]
prices  : 936 sessions have a score   ->  X used is (936, 512),  y is (936,)   [20 missing]
```

We fit **three separate models**, one per target, each on the rows where that target exists. We do not train one model with three outputs.

**In this repo.** `CGMDataset.target_mask()` (`cgm_tsfm/data.py:87-89`) returns a true/false flag per session — true where that score is present. `regression.py:183-184` applies it to X, y and the participant-id array together, so the three stay aligned.

**Say it in your own words:** "One row is one test session, so X is 956 rows by 512 columns, and y is a single column of 956 scores — and I fit a separate model for each of the three tests."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. What is one row?**
A. One cognitive test session. 956 of them, from 20 participants — roughly 48 sessions per participant.

**Q. What is the exact shape of X for symbols?**
A. (920, 512). 920 of the 956 sessions have a symbols score; 36 are missing.

**Q. Why is 956 not 980?**
A. 980 raw rows across the two cohort files (740 + 240). 24 of those rows had an empty or unparseable glucose cell, so they were dropped. 980 − 24 = 956.

**Q. Do 512 columns mean the model has 512 pieces of information about the glucose?**
A. No. The median session has only 36 glucose readings. Chronos re-expresses those in 512 numbers, so the columns are heavily related to one another. That is why reducing 512 down to 8 components with PCA costs nothing (mean R² −0.052 → −0.045).

</details>

---

## 3 · Regression vs classification

**Plain words.**
- **Regression** predicts a number on a continuous scale. "How long did this take, in seconds?"
- **Classification** predicts which of a fixed set of categories something belongs to. "Did this child fall below 70 mg/dL: yes or no?"

The difference is not cosmetic. It changes the loss you fit with, the metric you report, and even the kind of mistake that counts as bad.

**Ours is regression** because all three targets are numbers on a scale, and the distance between two values is meaningful:

| Target | What it is | Range in our data | Why regression |
|---|---|---|---|
| grids | mean placement error, in grid cells | 0.000 – 2.390 | continuous; 0.4 is genuinely between 0.3 and 0.5 |
| symbols | a response time in seconds | 0.150 – 4.190 | continuous; 838 distinct values in 920 sessions |
| prices | a percentage in steps of 10 | 0 – 90 | only 10 distinct values, but they are ordered and evenly spaced |

All three are **lower = better**. Memorize that. They are error and response-time measures, not accuracy scores. (What each test actually asks the child to do was worked out from the data — see `01_FACT_SHEET.md` section D.)

**The honest nuance on prices.** It takes only 10 values: 0, 10, 20, … 90. Someone could argue for treating those as 10 categories. We do not, and there is a good reason: if the true score is 40 and you predict 30, that is a small mistake; if you predict 90, that is a large one. Classification treats both as simply "wrong". Regression keeps the ordering and the spacing, which is the information we want to use.

**Tiny worked example.** The "guess the average" model for prices predicts **41.59** for every single session. As a regression prediction, that is defensible and it is our baseline. As a classification prediction it is not even a legal answer, because 41.59 is not one of the 10 categories.

**In this repo.** Everything in the Arm A path is regression machinery: `DummyRegressor(strategy="mean")`, `Ridge`, `SVR`, `LinearRegression`, and the metrics `r2_score` / `mean_squared_error` / `mean_absolute_error`. In Arm B, `CGMRegressionModule` uses `nn.MSELoss()` and torchmetrics' `R2Score`. There is no accuracy or AUROC anywhere, because there are no categories.

**Say it in your own words:** "It's regression — the three scores are numbers on a scale, so being close counts, and I report R² and RMSE rather than accuracy."

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. Prices has only 10 distinct values. Why not classify?**
A. Because the values are ordered and evenly spaced. Regression is rewarded for predicting 30 when the truth is 40; classification would score that identically to predicting 90.

**Q. Which direction is a good score?**
A. Lower, for all three. They are error and time measures.

**Q. If you flipped the sign of all the scores, would R² change?**
A. No. R², RMSE and MAE are unchanged by negating the target. Only the interpretation of a coefficient's sign changes. That is why `config.py:28` keeps all three in their raw orientation and just documents "lower = better".

</details>

---

## 4 · The train/test split, and why you cannot score on training data

**Plain words.** You split your examples into a **training set**, which the fitting procedure is allowed to see, and a **test set**, which it never sees until you report the final number. The test score is your estimate of how the model will do on data it has never met.

**The memorize-the-answers analogy.** Give a student a 100-question practice exam *with* the answer key and let them study for a week. Then set them the same 100 questions. They score 100%. You have learned nothing about whether they understand the material — they may have simply memorized "question 34 → C". To find out what they know, you have to ask questions that were not in the packet.

A model with lots of freedom does exactly this. It can store enough about the specific training rows to reproduce their answers, without capturing anything that transfers.

**Our concrete example.** `LinearRegression` with no restraint, fed all 512 features, fits its training folds closely. On the held-out participants it scored **R² ≈ −3** (`results/README.md:64`). R² = −3 means the squared error is **four times** what you get by ignoring the features entirely and guessing the average. For grids, whose spread is 0.500, that is a typical error of about 1.0 grid cells on scores that mostly sit between 0 and 2.39. The model looked fine in training and was worse than useless outside it. That single result is why every model in our grid has some form of restraint on it.

**We split by participant, not by row.** This is the part people get wrong. Consider what would happen with a plain random split of the 956 rows:

```
RANDOM row split (wrong for us)
  train:  child A session 12,  child A session 13,  child A session 15, ...
  test:   child A session 14,  child B session 3, ...
                 ^^^^^^^^^^^^
    The model saw 46 other sessions from child A. It can learn "child A tends
    to score about 0.41" and get session 14 nearly right — with no glucose
    knowledge at all. The test score would be flattered by that.

GROUPED split by participant (what we do)
  train:  all 765-ish sessions from 16 participants
  test:   all 191-ish sessions from the OTHER 4 participants
    No child appears on both sides. To do well, the model must have learned
    something about glucose, not about individuals.
```

This matters here specifically because sessions from the same child are not independent: between **7.7% and 37.6%** of each score's spread is explained purely by *which participant* it came from (grids 0.153, symbols 0.376, prices 0.077).

**In this repo.** `regression.py:111-114` runs the outer loop and, on every fold, asserts the training participant ids and test participant ids share no members:

```python
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
    "subject leakage between train and test!"
```

The alternative — a plain random-row split — is deliberately still available as `cv_scheme="session"` (`regression.py:79-84`), precisely so the two can be compared side by side.

**Say it in your own words:** "We hold out four whole children at a time, because if the same child appeared in training and testing the model could just remember that child's usual score instead of learning anything about glucose."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Why not just report the training score?**
A. Because a flexible model can reproduce training answers without learning a transferable rule. Plain LinearRegression on our 512 features is the demonstration: fine in training, R² ≈ −3 on held-out participants.

**Q. What exactly does "grouped" mean in GroupKFold?**
A. The grouping key is `subid`, the participant id (`config.py:31`). All sessions from one participant go to the same side of the split. 20 participants, 5 folds, so 4 participants are held out each time.

**Q. How many sessions are in each test fold?**
A. 189, 193, 191, 191, 192 across the five folds — about 191 each, with 763 to 767 sessions training.

**Q. What is the assert for?**
A. It is a tripwire. If a coding change ever let a participant appear on both sides, the run crashes instead of quietly producing a flattering number.

</details>

---

## 5 · Generalization, and the three-way train / validation / test split

**Plain words.** **Generalization** is how well the model does on data that played no role whatsoever in producing it. "No role whatsoever" is stricter than it sounds, and that strictness is why there are three sets, not two.

| Set | Who sees it | What it decides |
|---|---|---|
| **Training** | the fitting procedure | the model's internal numbers — Ridge's 512 coefficients, the network's weights |
| **Validation** | you (or the search code) | choices *about* the model — which α, which C, how many epochs before stopping |
| **Test** | nobody, until the end | nothing. It only gets reported. |

The rule that makes this click: **the moment you use a set to choose anything, it stops being a test set.** If you try α = 0.1, 1, 10, 100, 1000, look at the test score for each, and report the best one, your reported number is the best of five attempts, not an honest estimate of new-data performance. You need a third set to absorb that choosing.

```
                        956 sessions from 20 participants
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  P01 P02 P03 P04 │ P05 P06 P07 P08 │ P09 P10 P11 P12 │ P13 ... │ P17 ... │
  └──────────────────────────────────────────────────────────────────────────┘
        fold 1              fold 2            fold 3        fold 4    fold 5

  OUTER LOOP, fold 1 of 5  (GroupKFold, 5 splits, on participant id)
  ┌──────────────────────────┬───────────────────────────────────────────────┐
  │  TEST: 4 participants    │  OUTER-TRAIN: 16 participants                 │
  │  ~191 sessions           │  ~765 sessions                                │
  │  touched once, at the    │                                               │
  │  very end                │   ── Arm A: split again, 3 inner folds ──►    │
  │                          │      inner-train ~510   inner-val ~255        │
  │                          │      used to pick alpha / C / gamma           │
  │                          │      then refit on all ~765 with the winner   │
  │                          │                                               │
  │                          │   ── Arm B: hold out ~3 of the 16 kids ──►    │
  │                          │      train ~612 sessions, validate on the rest│
  │                          │      used to decide when to stop training     │
  └──────────────────────────┴───────────────────────────────────────────────┘
              repeat for folds 2..5, then average the 5 test scores
```

**Where validation lives in each arm — know both, they are different.**

*Arm A (Ridge / SVR).* Validation is the **inner 3-fold split**, run by `GridSearchCV` (`regression.py:121-129`). It is also grouped by participant, so about 5 of the 16 training participants sit out each inner fold. The search tries every hyperparameter setting, scores each on the inner held-out part, picks the winner, refits on the whole outer-training set, and only then predicts the outer test fold. This nested arrangement is why we can tune and still report an honest number.

*Arm B (neural head).* There is no hyperparameter grid. Validation is a **20% grouped holdout** of the outer-training participants (`GroupShuffleSplit`, `train.py:92-93`) and its job is **early stopping**: after every pass through the training data, the validation loss is measured; if it has not improved for **8 consecutive epochs**, training halts (`patience=8`, monitoring `val/loss`, `train.py:53`). The maximum is **100 epochs**, but early stopping usually ends it well before that. There is an assert here too, checking all three sets — train, validation, test — share no participants (`train.py:95-96`).

**Tiny worked example — the fit count.** This is worth being able to do out loud, because it shows you know a search is happening.

```
Per outer fold, per target:
  Ridge   5 alphas x 3 inner folds = 15 fits,  + 1 refit on the winner = 16
  SVR     (3 C values x 2 gammas) = 6 settings x 3 = 18 fits, + 1 refit = 19
  Linear  no settings to try                                          =  1
  Baseline(mean)  no settings to try                                  =  1

  x 5 outer folds  ->  80 + 95 + 5 + 5 = 185 model fits per target
  x 3 targets      ->  555 model fits for one Arm A run
```

**Arm B's epoch arithmetic**, same spirit:

```
~612 training sessions, batch size 64  ->  ceil(612 / 64) = 10 batches
  1 epoch = 10 weight updates
  100 epochs = at most 1,000 weight updates  (early stopping usually cuts it short)
```

**Say it in your own words:** "Training data sets the model's internal numbers, validation data picks the settings and decides when to stop, and the test data is four children the model has never seen — I only look at it once, at the end."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. What does early stopping watch, and for how long?**
A. `val/loss` on the 20% grouped validation split, with patience 8 — eight epochs of no improvement and it stops. Never the test fold.

**Q. Why is the tuning loop nested inside the outer loop instead of run once outside it?**
A. If α were chosen once using all the data, the test folds would have influenced that choice and the reported R² would be optimistic. Nesting keeps each outer test fold completely untouched by the tuning that produced the model tested on it.

**Q. How many participants are in each of the three sets on one Arm B fold?**
A. 4 test, and of the remaining 16 roughly 3 for validation and 13 for training — 20 total, no overlap, asserted in code.

**Q. Someone tunes α by checking the test R² and reports the best. What is wrong?**
A. The test set has become a validation set. The reported number is the maximum over five attempts, which is biased upward. It is no longer an estimate of performance on new children.

</details>

---

## 6 · Parameter vs hyperparameter vs weight vs bias

**Plain words.**
- A **parameter** is a number the fitting procedure discovers from the training data.
- A **hyperparameter** is a number *you* set before fitting, which controls how the fitting behaves. It is chosen with validation data, not learned from training data.
- A **weight** is a specific kind of parameter: a multiplier applied to an input.
- A **bias** (also **intercept**) is a specific kind of parameter: a constant added on, independent of the input.

```
predicted = bias  +  w1*x1  +  w2*x2  + ... +  w512*x512
            ^^^^     ^^         ^^              ^^
          parameter          all parameters (weights)

Ridge(alpha=10)
      ^^^^^^^^  hyperparameter — not learned; picked by the inner search
```

**The crisp table for our project.**

| Thing | Kind | Value / count here | Set by | Where |
|---|---|---|---|---|
| Ridge coefficients | parameters (weights) | **512** per target | `.fit` on training data | `regression.py:42` |
| Ridge intercept | parameter (bias) | **1** | `.fit` on training data | same |
| Ridge `alpha` | hyperparameter | one of {0.1, 1, 10, 100, 1000} | inner 3-fold search | `regression.py:43` |
| SVR `C` | hyperparameter | one of {0.1, 1, 10} | inner 3-fold search | `regression.py:45` |
| SVR `gamma` | hyperparameter | `scale` or `auto` | inner 3-fold search | same |
| PCA component count | hyperparameter | none / 8 / 16 / 32 / 64 / 128 | chosen by us from a sweep | `regression.py:118` |
| Neural head weights + biases | parameters | **132,609** | gradient descent on training data | `model.py:146-152` |
| Learning rate | hyperparameter | **1e-3** | fixed by us | `train.py:73` |
| Weight decay | hyperparameter | **0.01** | fixed by us | `train.py` / `lightning_module.py:90` |
| Dropout rate | hyperparameter | **0.1** | fixed by us | `model.py:137` |
| Hidden width | hyperparameter | **256** | fixed by us | `model.py:136` |
| Batch size | hyperparameter | **64** | fixed by us | `train.py:70` |
| Early-stopping patience | hyperparameter | **8** | fixed by us | `train.py:53` |
| Max epochs | hyperparameter | **100** | fixed by us | `train.py:68` |
| Random seed | not either — a setting for reproducibility | **42** | fixed | `config.py:39` |
| Chronos's own weights | parameters **of a different model** | frozen, never updated | Amazon's pre-training | `encoders.py` |

That last row is worth saying out loud in a defence. Chronos has parameters, millions of them, but they are not *our* parameters. We never compute a gradient for them.

**Tiny worked example — count the neural head's parameters.** Do this on a whiteboard if asked.

```
LayerNorm over 512 inputs   : 512 scales + 512 shifts        =     1,024
Linear  512 -> 256          : 256*512 weights + 256 biases   =   131,328
GELU                        : no parameters                  =         0
Dropout(0.1)                : no parameters                  =         0
Linear  256 -> 1            : 1*256 weights + 1 bias         =       257
                                                              ---------
                                                        total =   132,609
```

Compare: Ridge on the same 512 inputs fits **513** parameters. The neural head fits **132,609** — about **258 times** as many — on the same roughly 612 training sessions.

**Say it in your own words:** "Ridge's 512 coefficients are parameters, learned from the training data; Ridge's alpha is a hyperparameter, picked by the inner validation folds — one is discovered, the other is a dial we set."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. How many parameters does Ridge fit for grids?**
A. 513 — one coefficient per Chronos feature (512) plus one intercept.

**Q. Is dropout a parameter or a hyperparameter?**
A. Hyperparameter. The rate is 0.1 and we set it. Dropout itself contains no learned numbers.

**Q. Which data chose α = 10 on a given fold?**
A. The inner 3-fold grouped split of that fold's 16 training participants — never the 4 test participants.

**Q. Is the random seed a hyperparameter?**
A. No. It does not control model capacity or fitting behaviour in any meaningful way; it fixes the pseudo-random choices so the run reproduces. Ours is 42.

</details>

---

## 7 · Overfitting, underfitting, capacity, and the bias–variance tradeoff

**Plain words.**
- **Capacity** is how many different rules a model is able to express. More parameters and more flexibility mean more capacity.
- **Overfitting** is fitting the particular quirks of the training rows — noise included — so the model does well in training and badly on new data.
- **Underfitting** is a model too restricted to express even the pattern that is genuinely there, so it does badly in both places.

```
error
  ^
  |*                                                    * = error on NEW data (validation)
  | *                                                   o = error on TRAINING data
  |  *                                              *
  |   *                                          *
  |    o *                                   *
  |     o  *                             *
  |      o    *                      *
  |       o      *               *
  |        o         *      *  *
  |         o            *              <-- lowest new-data error: stop here
  |          o o
  |              o o o
  |                    o o o o o o o o o o o
  +----------------------------------------------------------> more capacity
     UNDERFITTING              |                  OVERFITTING
     both errors high          |        training error keeps falling,
                          the sweet spot        new-data error rises
```

The two axes people use for that horizontal line are (a) model capacity and (b) number of training epochs. Both produce the same U shape on the validation curve, which is why early stopping and shrinking the model are two ways of doing the same job.

**Our overfitting end, with real numbers.** `LinearRegression`, 512 features, no restraint: **R² ≈ −3** on held-out participants. Unpack that:

```
R^2 = 1 - (model's squared error) / (average-guesser's squared error)

R^2 = -3   =>   model's squared error = 4 x the average-guesser's
           =>   model's typical error = 2 x the average-guesser's

For grids (spread 0.500):  average-guesser RMSE ~ 0.500
                           this model's RMSE    ~ 1.0 grid cells
                           on scores that run 0.000 to 2.390
```

That is a model whose typical miss is comparable to the whole range of the outcome. It is the textbook picture of a model with more freedom than the data can pin down.

**Our underfitting end, also real.** Ridge's α grid goes up to 1000. A large α forces the coefficients toward zero. At the extreme, all coefficients are effectively zero and the model returns the training mean — it *becomes* the average-guesser, with R² just barely below 0 (barely, because the training mean is not exactly the test fold's mean). The uncomfortable finding of this project is that **the restrained end is where the best numbers live**. Best real R² anywhere is **−0.012** (prices, Arm A). The tuning search keeps choosing heavy restraint because there is nothing worth spending flexibility on.

**The bias–variance sketch, honestly.** The expected squared error of a prediction decomposes roughly as:

```
expected squared error  =  bias^2   +   variance   +   noise
                           ^^^^^^       ^^^^^^^^       ^^^^^
                    how wrong the    how much the   part nobody
                    model's shape    fitted model   can predict,
                    is on average    swings if you  ever
                                     change the
                                     training rows
```

More capacity usually lowers bias and raises variance. Less capacity does the reverse. Tuning is a search for the point where the sum is smallest.

Now the part specific to us. The **noise** term here is enormous. The strongest relationship between any simple glucose statistic and any score in this dataset is r = −0.092, i.e. r² = 0.0085 — under 1% of the score's spread. So at most about 1% of the error is potentially removable and roughly 99% is not. When the noise term is that dominant, the bias/variance balance barely matters: no setting of the dial reaches a positive R². This is why the low accuracy is a fact about the data rather than a tuning failure.

**Say it in your own words:** "Plain linear regression on 512 features scored about −3 on new children — four times the squared error of just guessing the average — so we needed to restrain the model; but the best restrained result is still −0.012, which tells me the limit isn't the model, it's the data."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. A model has negative R². Is that overfitting or underfitting?**
A. Negative R² alone does not tell you. It only says "worse than guessing the average". You diagnose by comparing training and new-data error: low training error with high test error is overfitting; high in both is underfitting. Our plain LinearRegression at −3 is the first; Ridge at α=1000 is the second.

**Q. What does R² = −3 mean in plain error terms?**
A. Four times the squared error of the average-guesser, so about twice the typical error. For grids that is roughly 1.0 grid cells versus the average-guesser's 0.500.

**Q. The tuning search keeps picking large α. What is it telling you?**
A. That the features are not earning their flexibility. The validation folds prefer a model that nearly ignores them, which is the same message the raw correlations give.

**Q. Could we escape overfitting by adding more features?**
A. No, that moves the wrong way — more features means more capacity and more variance on the same 764 training rows. The direction that helped, marginally, was fewer: PCA down to 8–32 components (−0.052 → −0.045).

</details>

---

## 8 · The curse of dimensionality

**Plain words.** As you add feature columns, the space those features live in grows explosively, and your fixed number of rows spreads out until it is almost empty. Two consequences: the number of free parameters climbs while the number of examples does not, and notions like "nearby points" stop being informative because in high dimensions nearly everything is far from everything else.

**Quantified for us — the number to have ready.**

```
Chronos features                   : 512
Training rows per outer fold       : 763 to 767   (call it 764)
                    ratio          : 764 / 512  ~=  1.5 rows per feature
Ridge parameters                   : 513         ->  ~1.5 rows per parameter

Neural head parameters             : 132,609
Training rows                      : ~612
                    ratio          : 132,609 / 612  ~=  217 parameters per row
```

About one and a half rows per feature is thin for a linear model. Two hundred and seventeen parameters per row is not thin, it is upside down — the model has vastly more freedom than the data can constrain, and that is the single clearest explanation for why Arm B is the worst performer we have (−0.168 to −0.281).

**A second angle, specific to this data.** The median session contains **36 glucose readings** (3.0 hours at one reading per 5 minutes). The shortest has 3, the longest 288. Chronos turns each of those into 512 numbers. Those 512 cannot contain more information than the 36 they came from — they are a re-expression, not an enrichment. So the 512 columns are heavily inter-related, and the evidence is direct: squeezing them to 8 PCA components does not lose accuracy, it slightly improves it.

| PCA components | mean R² across the 3 targets |
|---|--:|
| none (all 512) | −0.052 |
| 8 | **−0.045** |
| 16 | −0.046 |
| 32 | **−0.045** |
| 64 | −0.053 |
| 128 | −0.051 |

**A third angle.** More dimensions did not help even when we bought them from a bigger model:

| Chronos checkpoint | feature width | mean R² |
|---|--:|--:|
| bolt-tiny | 256 | −0.049 |
| bolt-mini | 384 | −0.052 |
| bolt-small | 512 | −0.052 |
| bolt-base | 768 | **−0.063** (worst) |
| t5-small | 512 | −0.057 |

The widest features gave the worst result. That is the curse showing up as a measurement rather than as theory.

**In this repo.** PCA sits *inside* the pipeline (`regression.py:116-120`), between the scaler and the model, so it is fitted on each fold's training rows only and never sees the test rows. Putting PCA outside the loop — fitting it once on all 956 rows — would let the test rows influence the features, which is exactly the kind of quiet mistake that inflates a result.

**Say it in your own words:** "We have 512 features and only about 764 training rows per fold, roughly one and a half rows per feature — and the neural head is far worse at 217 parameters per training row, which is why it loses."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. State the rows-to-features ratio.**
A. About 764 training rows to 512 features per outer fold, so roughly 1.5:1. Ridge fits 513 parameters, so also about 1.5 rows per parameter.

**Q. Why does compressing 512 features down to 8 not hurt?**
A. Because the 512 come from a median of 36 readings and are highly related to one another. There were never 512 independent facts to lose.

**Q. Why must PCA be fitted inside each fold?**
A. Fitting it on all 956 rows would let information from the test participants shape the feature space. Inside the pipeline it only ever sees that fold's training rows.

**Q. Did the larger Chronos model help?**
A. No. bolt-base at 768 features was the worst of the five checkpoints at −0.063, against −0.049 for the smallest.

</details>

---

## 9 · Regularization: one idea with many names

**Plain words.** Regularization is anything you deliberately add that stops the model from using its full freedom, in exchange for a better chance on new data. Every technique below is that same trade. They differ in *what* they restrict, not in why.

| Technique | What it restricts | The dial | Used here? |
|---|---|---|---|
| **L2 / Ridge** | the sum of *squared* coefficients — shrinks all of them toward zero, none exactly to zero | `alpha` ∈ {0.1, 1, 10, 100, 1000} | **Yes**, `regression.py:43` |
| **L1 / Lasso** | the sum of *absolute* coefficients — can push some to exactly zero, dropping features | `alpha` | **No.** Not in our grid. Do not claim it. |
| **Weight decay** | the size of the network's weights during optimization — the same idea as L2, applied inside the optimizer step | `weight_decay = 0.01` in AdamW | **Yes**, `lightning_module.py:90` |
| **Dropout** | how much of the network can be relied on at once — during training, randomly zeroes a fraction of the hidden units each step | `p = 0.1` | **Yes**, `model.py:150` |
| **Early stopping** | how long fitting is allowed to continue — halts when validation loss stops improving | `patience = 8`, max 100 epochs | **Yes**, `train.py:53` |
| **Dimensionality reduction (PCA)** | how many feature directions the model may use at all | components ∈ {8, 16, 32, 64, 128} | **Yes**, `regression.py:118` |
| **Smaller model** | capacity, directly | hidden width 256; bolt-tiny instead of bolt-base | Explored in the sweeps |
| **SVR's C** | how much the fit is allowed to chase individual training points (smaller C = more restraint) | C ∈ {0.1, 1, 10} | **Yes**, `regression.py:45` |

**Tiny worked example — what α actually does.** Ridge does not just minimize error. It minimizes error *plus a penalty on its own coefficients*:

```
minimize:   sum over training rows of (actual - predicted)^2   +   alpha * sum of (w_i)^2
            \_______________________________________________/       \_________________/
                        fit the data                                  stay small
```

- α = 0.1 → the penalty is almost nothing; behaves close to plain linear regression, the one that scored ≈ −3.
- α = 1000 → the penalty dominates; coefficients are crushed toward zero and predictions converge on the training mean.
- The grid spans four orders of magnitude on purpose, so the search can land anywhere between those extremes.

Our search kept landing near the restrained end, and Ridge or SVR — never plain Linear — is what the head-to-head reports as the best model for every target.

**One thing that is *not* regularization.** Grouped cross-validation. It does not restrain the model at all; it changes how honestly you *measure* it. Do not put it in this family — that is exactly the sort of slip a picky examiner will catch.

**Say it in your own words:** "Ridge's alpha, the network's weight decay, dropout, early stopping and PCA are all the same move — take away some of the model's freedom so it can't chase noise — and we use every one of them somewhere in the pipeline."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Name three regularizers in this repo with their exact settings.**
A. Ridge α searched over {0.1, 1, 10, 100, 1000}; AdamW weight decay 0.01; dropout 0.1. Also early stopping at patience 8 and PCA at 8–32 components.

**Q. Is weight decay the same thing as L2?**
A. They are the same idea — penalize large weights. For plain gradient descent they are equivalent. AdamW deliberately applies the decay separately from the adaptive gradient step rather than folding it into the loss, so they are not bit-for-bit identical. Say the plain version and add the AdamW caveat only if pushed.

**Q. Did we use Lasso?**
A. No. The grid is Ridge, SVR, LinearRegression and the mean baseline. XGBoost is in the code but the package is not installed in this environment, so it never ran either.

**Q. Why does stopping training early count as restraint?**
A. Because a network's freedom grows as training proceeds — early epochs capture broad structure, late epochs start memorizing individual rows. Cutting training short caps how far into memorization it can get.

</details>

---

## 10 · Loss vs metric

**Plain words.**
- The **loss** is the number the fitting procedure actively pushes down. It has to be something a fitting algorithm can work with — smooth, differentiable for gradient methods.
- A **metric** is a number you compute afterwards to describe the result to a human. It has no such constraint.

They are often different, and that is fine as long as you know which is which.

**Ours.**

```
TRAIN on:    Mean Squared Error (MSE)
             Arm A: Ridge/SVR minimize squared error internally
             Arm B: nn.MSELoss() on z-scored targets   (lightning_module.py:44,68)

TUNE on:     neg_mean_squared_error                     (regression.py:124)

REPORT:      R^2, RMSE, MAE                             (regression.py:135-139)
```

**Why the tuning metric is *negative* MSE.** `GridSearchCV` is written to *maximize* whatever score you hand it. MSE is a lower-is-better quantity. Negating it turns it into a higher-is-better quantity, so the same maximizing code works. There is nothing statistical in that minus sign; it is an interface convention. Being able to say that in one sentence is worth more than it sounds, because it is a very easy thing to be asked and to fumble.

**Why not tune on R² directly?** Within one inner fold, minimizing MSE and maximizing R² pick the same model, because R² is just MSE rescaled by that fold's own spread. Across folds they diverge: R² divides by each fold's variance, so a fold that happens to contain a narrow range of scores gets its errors amplified. Tuning on MSE keeps the objective on one consistent scale across the inner folds.

**Tiny worked example — R² from RMSE, on our real numbers.** Take prices, Arm A.

```
spread of prices across all sessions   : sd = 17.17     ->  variance = 294.8
our model's reported RMSE              : 17.236         ->  MSE      = 297.1

R^2 = 1 - 297.1 / 294.8 = 1 - 1.008 = -0.008
reported R^2                           : -0.012
```

Those agree to within a rounding-level gap, and you should know *why* the gap exists rather than pretend it doesn't: R² is computed separately on each of the 5 test folds against **that fold's own mean and spread**, then the five values are averaged. RMSE is averaged the same way. Mean-of-R² and mean-of-RMSE are therefore not algebraically tied to each other. If asked, say exactly that.

**The three reported metrics, and what each is good for.**

| Metric | Plain meaning | Units | Character |
|---|---|---|---|
| **R²** | fraction of the score's spread the model accounts for, compared with guessing the average. 1 = perfect, 0 = no better than the average, negative = worse | none | comparable across the three targets, which is why it is our headline |
| **RMSE** | typical error, with big misses weighted heavily (square, average, square-root) | same as the score | interpretable but not comparable across targets — prices' 17.2 and grids' 0.505 are not on the same scale |
| **MAE** | average size of the error, all misses weighted equally | same as the score | less swayed by a handful of large misses |

**Tiny worked example — RMSE vs MAE on four errors.** Errors of 1, 1, 1, and 5:

```
MAE  = (1 + 1 + 1 + 5) / 4                = 2.00
RMSE = sqrt((1 + 1 + 1 + 25) / 4) = sqrt(7) = 2.65
```

Same errors, different summaries. RMSE is larger because squaring gives the single error of 5 disproportionate weight. That is also why fitting on MSE means the model works hardest on its biggest misses.

**Say it in your own words:** "We fit by minimizing mean squared error, and we report R² and RMSE — the minus sign in `neg_mean_squared_error` is only there because the search code maximizes its score and MSE is lower-is-better."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Why `neg_mean_squared_error` and not `mean_squared_error`?**
A. `GridSearchCV` maximizes. MSE is lower-is-better, so scikit-learn negates it to fit that convention.

**Q. What does R² = 0 mean, exactly?**
A. The model's squared error equals that of predicting the training mean for every session. It adds nothing. Every real R² in this project is at or below 0, best being −0.012.

**Q. Compute R² for prices from its RMSE.**
A. RMSE 17.236 → MSE 297.1; the spread 17.17 → variance 294.8; 1 − 297.1/294.8 ≈ −0.008, against the reported −0.012. The small gap is because both R² and RMSE are computed per fold and then averaged, so the two averages are not algebraically linked.

**Q. Why report RMSE at all if R² is the headline?**
A. Because RMSE is in the score's own units, so it says how wrong we are in a way a clinician can read: about 17 percentage points on prices, about 0.5 grid cells on grids.

</details>

---

## 11 · Inductive bias, and why fancier is not automatically better

**Plain words.** **Inductive bias** is the set of assumptions a model brings *before* it sees any data — the built-in preference for some rules over others. Every model has one. It has to: infinitely many rules fit any finite set of examples, so something other than the data has to break the tie. Which is also the reason a model with no assumptions cannot generalize at all.

Our three models make three different bets:

| Model | Its assumption about the answer | When that pays off |
|---|---|---|
| **Ridge** | the score is a weighted sum of the features, with small weights | when the relationship is roughly straight-line and every feature contributes a little |
| **SVR with an RBF kernel** | sessions whose 512 features are close together should get similar scores | when there is a smooth, local structure in feature space |
| **MLP head** | the score is some smooth non-linear function of the 512 features | when there are genuine interactions and enough data to find them |

**"No free lunch"** is the formal result that averaged over all possible problems, no learning algorithm beats any other. In practice it means: a method wins only when its assumptions happen to match the problem. There is no method that is better everywhere, so "we used a neural network" is not by itself an argument.

**Our evidence, in numbers.**

| Target | Arm A: Chronos + Ridge/SVR | Arm B: Chronos + neural head | 43 hand-crafted features + Ridge/SVR |
|---|--:|--:|--:|
| grids | **−0.089** | −0.218 | −0.065 |
| symbols | **−0.056** | −0.281 | −0.038 |
| prices | **−0.012** | −0.168 | −0.009 |

The most flexible model is the worst on all three targets, by a wide margin. This is not mysterious and you should not be defensive about it. The head fits 132,609 parameters on about 612 training sessions — 217 parameters per row — while trying to learn a relationship that, judged by every raw correlation we can compute, is close to flat. Its extra flexibility has nothing to spend itself on except noise, and dropout, weight decay and early stopping are the only reason it is merely bad rather than catastrophic.

Two more data points in the same direction. The larger Chronos checkpoint (bolt-base, 768 features) was the *worst* of the five at −0.063. And the 43 simple hand-crafted features actually edge out the 512 Chronos features on all three targets. When there is nothing to find, added sophistication reliably costs you.

**Say it in your own words:** "Every model brings assumptions, and a model only wins when its assumptions fit the problem — ours has 217 parameters per training row chasing a nearly flat relationship, so the flexible neural head lost to plain Ridge, which is the expected outcome, not a surprise."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Why did the neural network do worse than Ridge?**
A. 132,609 parameters against about 612 training sessions — 217 to 1 — with no real relationship in the data to spend that flexibility on. Ridge's assumption of a small-coefficient weighted sum is far closer to "there is almost nothing here", which is the truth in this dataset.

**Q. Does that mean neural networks are bad for this kind of problem?**
A. No. It means they lose at this sample size on this relationship. With 92 participants as the grant planned, or with a real effect present, the comparison could go the other way. The claim is about our data, not about the method in general.

**Q. Name Ridge's inductive bias in one sentence.**
A. That the score is a straight-line weighted sum of the features and that the weights should be small.

**Q. Isn't it embarrassing that 43 hand-made features beat 512 learned ones?**
A. It is informative rather than embarrassing. Both are at or below 0, so neither works; the ordering among things that all fail is a weak signal. But it does say the learned features bought us nothing here, and it is worth reporting plainly.

</details>

---

## 12 · What a model cannot do

**Plain words.** A model can only re-express relationships that are already present in the numbers you gave it. It cannot invent one. If the input carries almost no information about the output, no architecture, no amount of tuning and no amount of compute will produce accurate predictions. It will produce confident-looking predictions that are wrong, which is worse.

**The ceiling, measured.** Before any model, we correlated simple glucose statistics against each score across all 956 sessions:

| Score | Strongest single correlation | r | r² = share of spread explained |
|---|---|--:|--:|
| grids | number of readings in the window | −0.077 | 0.006 (0.6%) |
| symbols | fraction of readings below 70 mg/dL | **−0.092** | **0.0085 (0.85%)** |
| prices | maximum glucose | +0.056 | 0.003 (0.3%) |

**The single strongest relationship anywhere in this dataset explains 0.85% of the spread in the score.** Every other pairing is under 0.6%. Say that number before you say any R², because it converts the low accuracy from suspicious into expected.

Two details to hold alongside it, because they are the difference between reciting and understanding:

- The sign on that strongest correlation runs *opposite* to the physiological expectation. Symbols is a response time and lower is better, so r = −0.092 says more time below 70 mg/dL went with slightly *faster* responses. At r² = 0.0085 that is not worth interpreting — but you should notice it rather than let someone else notice it for you.
- The grids "signal" is with the *length* of the glucose window, not with glucose at all. Our windows run from 3 to 288 readings, a 96-fold spread, because the stored array appears to hold everything since that child's previous test rather than a fixed lookback. So that 0.006 is probably an artefact of how the data was assembled, not biology.

**Three more numbers that box in the ceiling.**

- **62% to 92% of each score's spread is within a participant**, not between participants (between-participant share: grids 0.153, symbols 0.376, prices 0.077). So most of what we are trying to predict is session-to-session variation, which is the hardest kind.
- **Only 2.31% of all 68,206 readings are below 70 mg/dL**, and only **201 of 956 sessions** contain even one such reading. Low glucose is where a cognitive effect is most expected, and we have very little of it.
- **2.16% of readings sit exactly at 400 mg/dL**, the Dexcom G6 ceiling. Those are recordings of "at least 400", not measurements, so the shape of a high excursion is partly unreadable.

**Why this is a finding about the data and not a broken program.** Two checks, both in `results/rigor_real.md`:

*The scrambled-scores check.* We shuffled the scores at random 200 times, destroying any real link, and re-ran the whole thing each time. If our pipeline were finding something, the real result would sit clearly better than the scrambled pile. It does not — it sits *inside* it, at the bad end: p = **1.000** for grids, **1.000** for symbols, **0.995** for prices. Our accuracy is what you get from scores paired at random.

*The same-embeddings check.* We asked the identical pipeline, using the identical 512 features, to predict something about the glucose itself — something it ought to be able to do. (This is the manoeuvre sometimes called a positive control: give the machinery a task you know is solvable, so a failure on the real task can't be blamed on the machinery.)

| Target given to the same pipeline | Best R² |
|---|--:|
| glucose variability (standard deviation) | **0.462** |
| mean glucose | 0.070 |
| percentage of time above 180 mg/dL | 0.075 |

Variability comes out clearly, so the features carry real information and the machinery works. Be precise about the weak spot rather than glossing it: mean glucose scores only 0.070, because Chronos rescales every series internally and largely discards absolute height. So our Chronos result speaks to glucose *shape*, not *level*. The reason that does not sink the conclusion is that the 43 hand-crafted features **do** encode absolute level — mean glucose, time in range — and they were flat too (−0.009 to −0.065). "No usable relationship" holds for level-aware features as well.

**Say it in your own words:** "The strongest relationship between any glucose statistic and any score in this dataset explains under one percent of the variation, so there was nothing for a model to find — and I can show that the machinery works, because the same features predict glucose variability at R² 0.46."

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Suppose a better architecture appeared tomorrow. What is the realistic ceiling on this data?**
A. Somewhere around the raw association. The strongest single glucose-to-score correlation is r = −0.092, r² = 0.0085 — under 1% of the spread. A better model cannot manufacture a relationship that is not in the numbers.

**Q. How do you know the pipeline isn't just broken?**
A. Two checks. Scrambling the scores 200 times gives results indistinguishable from ours (p = 1.000, 1.000, 0.995). And the same features and the same cross-validation predict glucose variability at R² 0.462, so the machinery extracts real information when it is present.

**Q. Doesn't the weak mean-glucose result (0.070) undermine the whole thing?**
A. It limits the scope of the Chronos-based conclusion to glucose shape rather than level, and I say so. It does not undermine the overall answer, because the hand-crafted features do encode absolute level and were also at or below 0.

**Q. Is the honest claim "there is no glucose-to-cognition effect"?**
A. No. The honest claim is narrower: in these 20 participants and 956 sessions, with these windows and these three scores, no method we tried predicted the score more accurately than guessing the average. That is a description of what this data supports, not a statement about the effect in general.

</details>

---

## Recap · ten sentences that cover this whole chapter

1. Supervised machine learning means handing a program examples of input paired with the right answer and letting it search for a rule; ours takes 512 numbers describing one child's glucose before one test and returns one predicted score.
2. X is the feature matrix, (956, 512) for the Chronos version and (956, 43) for the hand-crafted version, and y is a single column of scores — one row is one test session, not one child.
3. It is regression, not classification, because all three scores are numbers on a scale where being close counts, and all three run lower = better.
4. You cannot score a model on the data it was fitted to, because a flexible model can reproduce training answers without learning anything transferable — plain linear regression on our 512 features looked fine in training and scored R² ≈ −3 on held-out children.
5. We split by whole participant, four held out at a time across five folds, because 7.7% to 37.6% of each score's spread is explained purely by which child it came from, so a random-row split would let the model recall a child's usual score instead of learning about glucose.
6. Three sets, three jobs: training data fixes the model's internal numbers, validation data picks settings like Ridge's alpha and decides when to stop training, and the test fold is reported and nothing else — using it to choose anything would destroy the honesty of the estimate.
7. Parameters are learned from training data (Ridge's 512 coefficients plus 1 intercept; the neural head's 132,609 weights and biases) while hyperparameters are dials we or the search set (alpha, C, gamma, learning rate 1e-3, weight decay 0.01, dropout 0.1, patience 8).
8. Overfitting and underfitting are the two ends of one dial, and our numbers show both ends: unrestrained linear regression at −3, and heavily restrained Ridge collapsing to the average-guesser at roughly 0 — with only about 1.5 training rows per feature, and 217 parameters per row in the neural head, restraint is mandatory.
9. Ridge's alpha, weight decay, dropout, early stopping and PCA are all one idea — take away model freedom — and we train on mean squared error while reporting R², RMSE and MAE, with the minus sign in `neg_mean_squared_error` there only because the search code maximizes its score.
10. Every model carries assumptions and only wins when they match the problem, which is why the flexible neural head (−0.168 to −0.281) lost to plain Ridge (−0.012 to −0.089); and none of them could win, because the strongest relationship between any glucose statistic and any score in this dataset explains 0.85% of the variation.

---

## The 3 most likely exam questions from this chapter

**1. "Why can't you evaluate on your training data? Show me it matters."**
Because a model with enough freedom reproduces training answers without learning a transferable rule — the answer-key student. In this project it matters concretely: `LinearRegression` on all 512 Chronos features fits its training folds closely and scores **R² ≈ −3** on held-out participants, which is four times the squared error of guessing the average, roughly 1.0 grid cells of typical error on scores that run 0 to 2.39. That result is why every model in our grid carries some restraint, and why our split holds out four whole children at a time rather than random rows.

**2. "What is the difference between your validation set and your test set, and which one chose alpha?"**
The validation data chooses things *about* the model; the test data is only reported. Alpha was chosen by the **inner 3-fold grouped split** of each fold's 16 training participants (`GridSearchCV`, scoring `neg_mean_squared_error`), never by the 4 test participants. In Arm B the equivalent is a 20% grouped holdout of the training participants, used for early stopping at patience 8 on `val/loss`. The reason for nesting is that if alpha had been picked using the test folds, the reported R² would be the best of five attempts rather than an estimate of performance on new children.

**3. "You used a foundation model and a neural network and got nothing. How do I know it isn't just your code?"**
Three separate pieces of evidence. First, the ceiling: the strongest correlation between any simple glucose statistic and any score across all 956 sessions is r = −0.092, so r² = 0.0085 — under 1% of the score's variation — before any model is involved. Second, scrambling the scores 200 times and re-running everything gives results indistinguishable from the real one (p = 1.000, 1.000, 0.995), meaning our accuracy is what random pairing produces. Third, the same 512 features under the same cross-validation predict glucose variability at **R² = 0.462**, so the machinery does extract real information when there is some to extract. The one honest caveat is that Chronos rescales each series and so is partly blind to absolute glucose level (mean glucose only R² 0.070) — which is why it matters that the 43 hand-crafted features, which do encode absolute level, were also at or below 0.

---

*Numbers in this chapter are taken from `01_FACT_SHEET.md` and from `cgm_tsfm/`, `results/headtohead_real.md`, `results/sweep_real.md` and `results/rigor_real.md`. Where a claim could not be verified against those sources it was left out. If anything here disagrees with `01_FACT_SHEET.md`, the fact sheet wins.*
