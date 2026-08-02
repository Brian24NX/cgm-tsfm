# 04 · Linear Models, Ridge, SVR and PCA — from absolute zero

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> ### What this chapter buys you
> 1. You can derive linear regression on a napkin, invert a 3×3 matrix by hand, and say what "fitting" means.
> 2. You can explain **why plain linear regression scored R² ≈ −3 on our data while Ridge scored −0.089** — the single best story in the whole project.
> 3. You can define condition number, alpha, epsilon, gamma, C, and explained-variance ratio without hand-waving.
> 4. You can defend the choice of *simple* models as the correct scientific decision, not laziness.
> 5. You can answer "why not PCA everything?" with a measured counter-example where PCA destroyed a perfect prediction.

**Prerequisites:** `02_ML_FROM_ZERO.md` (what a model, feature, target, and R² are), `03_PREPROCESSING.md` (StandardScaler, why it is fit on the training fold only). **Numbers:** `01_FACT_SHEET.md`. **Cross-validation and metrics in depth:** `05_EVALUATION.md`.

**Where we are in the project.** A frozen Chronos time-series model has already turned each glucose window into 512 numbers. That step is finished and unchangeable. This chapter is entirely about the *last* box in the pipeline:

```
   CGM readings before a test          Chronos (frozen)        THIS CHAPTER        predicted
   [142, 145, 151, ... ]      ──────▶  512 numbers    ──────▶  a regressor ──────▶ score
   (3 to 288 readings)                 per session             (Ridge / SVR)
```

**Provenance rule for this chapter.** Numbers marked **[M]** were measured on the real data on 2026-07-30 for this chapter, using the cached embeddings in `.cache/embeddings/chronos__amazon_chronos-bolt-small__mean__4b2c75e042223011.npy` (shape `(956, 512)`) and the real `GroupKFold(5)` splits. They are reproducible. Numbers marked **[F]** come from `01_FACT_SHEET.md`. Anything not marked is arithmetic you can redo on paper.

---

## 1. Linear regression from absolute zero

### 1.1 The model

One session gives us a list of numbers, `x₁ … x₅₁₂` — one row of the feature matrix. Call the cognitive score `y`. A linear model is this rule:

```
  ŷ  =  w₁x₁ + w₂x₂ + … + w₅₁₂x₅₁₂  +  b          compactly:   ŷ = w · x + b
```

`ŷ` ("y-hat") is the **prediction**. The `w`'s are **weights** or **coefficients** — one per feature, saying how much that feature pushes the prediction up or down. `b` is the **intercept**, the prediction when every feature is zero. That is the entire model; nothing is hidden. To predict a new session you multiply 512 numbers by 512 weights, add them up, add `b`, done.

**Glued to our project.** `x` is the 512-number Chronos summary of the glucose trace before one test. `y` is that session's Grids, Symbols, or Prices score. We fit **three separate models**, one per score — not one model with three outputs. All three scores are "lower = better" **[F]**.

### 1.2 What "fitting" means

Fitting = choosing `w` and `b`. We have 956 sessions **[F]**, each with a known `x` and known `y`. **Least squares** defines "closest to the known answers" as minimising the sum of squared gaps:

```
  Loss(w, b)  =  Σ over sessions i of ( yᵢ − ŷᵢ )²  =  Σ ( yᵢ − (w·xᵢ + b) )²
```

Each gap `yᵢ − ŷᵢ` is a **residual**. We square it for two reasons: a squared gap is always positive, so overshooting by 2 does not cancel undershooting by 2; and squaring punishes large mistakes disproportionately (off by 4 costs 16, off by 1 costs 1).

### 1.3 The picture, with one feature

Drop to one feature so it fits on a page. Four sessions, feature `x` = (mean glucose − 170) / 10, target `y` = Grids-like score.

```
  y
  6 |                            ● D  (x=+1, y=6)
  5 |                     .·´
  4 |           .·´                ŷ = 3 + 1.5x  ← the fitted line
  3 |-·´-----------------●-C-(x=+1,-y=3)-------  ← baseline: always predict ȳ = 3
  2 |      ● B  (x=−1, y=2)
  1 | ● A  (x=−1, y=1)
  0 +------------------------------------------ x
        −1            0            +1
```

Fitting = sliding and tilting that dotted line until the sum of squared vertical gaps to the four dots is as small as it can be. The flat line at `y = 3` is the **baseline** we compare against: it ignores glucose entirely and always predicts the average score. In code that baseline is `DummyRegressor(strategy="mean")` (`regression.py:192`).

### 1.4 The normal equation

You do not have to search for the best `w`. There is a formula. Stack the data into a matrix `X` with one row per session, prepending a column of 1s so the intercept becomes just another weight. Set the derivative of the loss to zero and you get the **normal equation**:

```
   XᵀX β  =  Xᵀy          ⟹        β̂ = (XᵀX)⁻¹ Xᵀy
```

**In plain words:** build a small square table `XᵀX` that records how every feature co-varies with every other feature; build a short list `Xᵀy` that records how every feature co-varies with the score; then *undo* the first table (invert it) to strip out the double-counting caused by features overlapping with each other. What survives is the unique set of weights that minimises squared error.

`XᵀX` is `(d+1) × (d+1)` where `d` is the number of features. **For our project d = 512, so `XᵀX` is 513 × 513.** Remember that shape — it is the crime scene for Section 3.

### 1.5 Worked example, done fully by hand

Four sessions, two features, small numbers on purpose. Read `x₁` as "glucose level, centred" and `x₂` as "glucose variability, centred", each coded −1 (low) or +1 (high).

| session | x₁ | x₂ | y (score) |
|---|--:|--:|--:|
| A | −1 | −1 | 1 |
| B | −1 | +1 | 2 |
| C | +1 | −1 | 3 |
| D | +1 | +1 | 6 |

**Step 1 — build `XᵀX`.** With the intercept column of 1s, `X` is 4×3. Each entry is a dot product of two columns:

```
  (XᵀX)₁₁ = 1²+1²+1²+1²                     = 4      ← intercept · intercept
  (XᵀX)₁₂ = 1(−1)+1(−1)+1(+1)+1(+1)         = 0      ← intercept · x₁
  (XᵀX)₁₃ = 1(−1)+1(+1)+1(−1)+1(+1)         = 0
  (XᵀX)₂₂ = (−1)²+(−1)²+1²+1²               = 4
  (XᵀX)₂₃ = (−1)(−1)+(−1)(1)+(1)(−1)+(1)(1) = 1−1−1+1 = 0      ← x₁ · x₂
  (XᵀX)₃₃ = 4
                     ⎡ 4  0  0 ⎤
              XᵀX =  ⎢ 0  4  0 ⎥      (diagonal — the easy case)
                     ⎣ 0  0  4 ⎦
```

**Step 2 — build `Xᵀy`.**

```
  intercept row:  1(1) + 1(2) + 1(3) + 1(6)            = 12
  x₁ row:        (−1)(1) + (−1)(2) + (+1)(3) + (+1)(6) = −1 − 2 + 3 + 6 =  6
  x₂ row:        (−1)(1) + (+1)(2) + (−1)(3) + (+1)(6) = −1 + 2 − 3 + 6 =  4
```

**Step 3 — solve.** `XᵀX` is diagonal, so inverting it is just dividing by each diagonal entry:

```
   b  = 12 / 4 = 3.0
   w₁ =  6 / 4 = 1.5
   w₂ =  4 / 4 = 1.0

   ⟹   ŷ = 3.0 + 1.5·x₁ + 1.0·x₂
```

**Step 4 — check it.**

| session | ŷ = 3 + 1.5x₁ + x₂ | y | residual |
|---|--:|--:|--:|
| A | 3 − 1.5 − 1 = **0.5** | 1 | +0.5 |
| B | 3 − 1.5 + 1 = **2.5** | 2 | −0.5 |
| C | 3 + 1.5 − 1 = **3.5** | 3 | −0.5 |
| D | 3 + 1.5 + 1 = **5.5** | 6 | +0.5 |

Sum of squared residuals **SSE = 4 × 0.25 = 1.0**. The baseline that always predicts the mean (`ȳ = 3`) gets SSE = 4 + 1 + 0 + 9 = **14.0**. So `R² = 1 − 1/14 = 13/14 = 0.9286`. All verified numerically. Hold on to `SSE = 1.0` and `TSS = 14.0`; Section 4 reuses them.

**One thing to notice, because it matters later.** The off-diagonal entries of `XᵀX` came out exactly 0, meaning the two features were **uncorrelated** in this sample. Consequence: `w₁ = 1.5` whether or not `x₂` is in the model — each feature got its own clean answer. Real Chronos dimensions are *not* uncorrelated: mean absolute pairwise correlation across the 512 dimensions is **0.276**, median **0.245**, maximum **0.953** **[M]**. That is the whole difference between Section 1 and Section 3.

### 1.6 The geometric picture

Two geometries; people mix them up. Know both.

**Picture A (intuitive).** Each session is a dot in a space with one axis per feature plus one for the score. The model is a flat sheet — a line with 1 feature, a plane with 2, a **hyperplane** with 512. Fitting tilts the sheet as close to the dots as possible, measuring distance vertically only.

**Picture B (the one that explains Section 3).** Think of the 764 predictions as one vector in 764-dimensional space. Every choice of `w, b` produces one such vector, and all reachable ones form a flat subspace — the **column space** of `X` — of dimension at most 513. Least squares drops a perpendicular from the true `y` onto that subspace. That is why residuals come out orthogonal to every feature.

```
        y (the truth, in 764-dim space)
         \
          \   residual  ⟂  column space
           \
   ─────────●─────────────────────  column space of X (≤ 513 dimensions)
            ŷ  (nearest reachable point)
```

**Why this matters:** a 513-dimensional subspace inside a 764-dimensional space is enormous. It can get *very* close to almost any `y`, including a `y` made of pure noise. That is overfitting, seen geometrically.

> **Say it in your own words:** Linear regression picks one weight per feature so that the weighted sum lands as close as possible to the real score, and there is a formula that finds those weights in one shot.

**Drill**
- *Q: What exactly does the model output?* A single number: the weighted sum of the 512 Chronos values plus an intercept. That number is our predicted Grids/Symbols/Prices score for that session.
- *Q: Why squared error and not absolute error?* Squaring makes the loss differentiable everywhere and yields the closed-form normal equation; it also penalises large errors disproportionately. Absolute error (which gives median regression) has no closed form and needs an iterative solver.
- *Q: In your worked example, why is `w₁` unchanged when you drop `x₂`?* Because the two features are exactly uncorrelated in that sample, so `XᵀX` is diagonal and the features do not compete for credit. Real embedding dimensions are correlated, so they do compete.
- *Q: How many numbers does the fitted model store for us?* 512 weights plus 1 intercept = 513, per target, per fold.

---

## 2. "Linear" is less limiting than it sounds

Beginners hear "linear" and picture a straight line, then object that biology is not straight lines. The objection misreads the word.

**Linear means linear in the parameters, not in the raw measurements.** The requirement is that `ŷ` is a weighted sum of *something*. What that something is, is up to you.

All of these are still linear regression:

```
   ŷ = w₁·(mean glucose) + w₂·(mean glucose)²  + b        ← a curve in glucose
   ŷ = w₁·log(glucose SD) + b                            ← a log relationship
   ŷ = w₁·x₁ + w₂·x₂ + w₃·(x₁ · x₂) + b                  ← an interaction
   ŷ = w · (512 Chronos numbers) + b                      ← OURS
```

The last line is the point. Chronos is a deep, heavily non-linear transformation of the glucose series — attention layers, patch embeddings, residual streams. It maps a raw trace to 512 numbers. Our regressor is linear **on top of that**, so the *composition* is a highly non-linear function of glucose. This is exactly the "frozen encoder + linear probe" design.

**How to say it to the advisor:** "The non-linearity lives in Chronos. The last layer is deliberately linear, because a linear probe answers a clean question — *is the information present in the representation in a linearly decodable form?* — and it is the only model class our 956 sessions can constrain honestly."

**The honest caveat.** If the true relationship between glucose and cognition is non-linear *in a way Chronos did not already encode*, a linear probe would miss it. That is why we also ran an RBF-kernel SVR (Section 8), which is non-linear in the 512 features, and a small neural head (Arm B, `01_FACT_SHEET.md` §H). Neither beat guessing the average. So "we only tried linear things" is not a valid criticism of this project.

> **Say it in your own words:** Linear refers to how the weights combine, not to the shape of the curve — Chronos supplies the curvature and the regressor just weights it.

**Drill**
- *Q: Is `ŷ = w·log(x) + b` a linear model?* Yes. It is linear in `w`. Non-linear in `x`, which is allowed.
- *Q: Then what would make a model non-linear in the parameters?* Anything where parameters multiply or nest, e.g. `ŷ = w₁·exp(w₂ x)`, or a neural network with hidden layers. Those need iterative optimisation.
- *Q: So did we test non-linear relationships at all?* Yes — SVR with an RBF kernel is non-linear in the 512 features, and Arm B's MLP head has a GELU non-linearity. Both did the same or worse.

---

## 3. What goes wrong with 512 features and ~764 training rows

This is the section to have cold. It is the most technically interesting result we have.

### 3.1 The arithmetic of the problem

Five outer folds, grouped by participant, so 4 of the 20 participants are held out each time **[F]**. Train-fold sizes are **767, 763, 765, 765, 764** sessions **[F]**.

```
  unknowns to estimate  =  512 weights + 1 intercept  =  513
  training rows         ≈  764
  ratio                 =  764 / 513  =  1.49 rows per unknown
                           (range across folds: 763/513 = 1.487  to  767/513 = 1.495)
```

For comparison, a rule of thumb in applied statistics is to want 10–20 rows per estimated parameter. We have 1.49.

**And it gets worse inside the tuning loop.** `GridSearchCV` splits each ~764-row training fold again, 3 ways, so each hyperparameter is scored on roughly two-thirds of it. Measured inner-training sizes across all 5 × 3 = 15 inner folds: **min 488, max 523, mean 509.9** — and **5 of the 15 inner folds have fewer rows (488–491) than the 512 features** **[M]**. In those five, the design matrix is literally wider than it is tall: there are more unknowns than equations and infinitely many weight vectors fit the training data perfectly.

### 3.2 Condition number, in plain words

`XᵀX` is 513 × 513 and we must invert it. Whether that inversion is trustworthy is measured by the **condition number**.

**Plain-words definition:** it answers *"if I nudge the inputs by 1 part in a million, how far can the answer move?"* A condition number of 1,000 means a relative wobble in the data can be amplified up to 1,000× in the weights. 1 is perfect; infinite means the matrix cannot be inverted at all.

Formally, for a symmetric positive matrix it is `κ = λ_max / λ_min`, largest eigenvalue over smallest. Geometrically `XᵀX` stretches space a lot along some directions and barely at all along others, and `κ` is the ratio of the most- to the least-stretched. Inverting flips that, so barely-stretched directions get enormously amplified.

**Our measured numbers**, per outer training fold, on the standardized real 512-d features **[M]**:

| Outer fold | rows | λ_min of XᵀX | λ_max of XᵀX | κ(XᵀX) |
|---|--:|--:|--:|--:|
| 1 | 767 | 0.0275 | 106,380 | 3.86 × 10⁶ |
| 2 | 763 | 0.0257 | 103,990 | 4.05 × 10⁶ |
| 3 | 765 | 0.0266 | 105,120 | 3.95 × 10⁶ |
| 4 | 765 | 0.0267 | 106,340 | 3.98 × 10⁶ |
| 5 | 764 | 0.0246 | 102,990 | 4.20 × 10⁶ |
| **mean** | | | | **4.01 × 10⁶** |

Four million. The matrix is *technically* invertible — `numpy.linalg.matrix_rank` returns the full 512 **[M]** and no eigenvalue is below 0.024 — but **148 of the 512 eigenvalues are under 1.0** **[M]**: roughly 150 directions in embedding space along which our 764 sessions carry almost no information. Least squares still assigns confident weights there, and those weights are noise.

**Precision point worth making out loud:** the problem is *ill-conditioning*, not *singularity*. Nothing crashes. The answer is simply not reproducible.

### 3.3 The collinearity demonstration, by hand

Why near-duplicate features are poison. Take two standardized features correlating at 0.999 — close to what happens when two Chronos dimensions encode nearly the same thing.

```
              ⎡ 1      0.999 ⎤
      XᵀX  =  ⎣ 0.999  1     ⎦

  det = 1 − 0.999² = 1 − 0.998001 = 0.001999          ← nearly zero

  eigenvalues of [[a,b],[b,a]] are a+b and a−b:
      λ_max = 1.999 ,  λ_min = 0.001   ⟹   κ = 1999
```

Now fit twice, on data differing by a hair. Suppose the two features' correlations with the score are:

```
  case 1:  Xᵀy = (0.100, 0.100)
  case 2:  Xᵀy = (0.101, 0.099)     ← a 1% wiggle, e.g. one session's score changed slightly
```

The inverse of a 2×2 is `(1/det)·[[d, −b], [−c, a]]`:

```
  case 1:  w₁ = (1·0.100 − 0.999·0.100)/0.001999 = 0.000100/0.001999 =  0.0500
           w₂ = same                                                  =  0.0500

  case 2:  w₁ = (1·0.101 − 0.999·0.099)/0.001999 = 0.002099/0.001999 =  1.0500
           w₂ = (−0.999·0.101 + 1·0.099)/0.001999 = −0.001899/0.001999 = −0.9500
```

Verified numerically. **A 1% change in the data flipped the coefficients from (+0.05, +0.05) to (+1.05, −0.95).** The fit barely changed — the two features almost cancel — but the story the model tells reversed completely: feature 2 went from "mildly helpful" to "strongly harmful". Any interpretation of these coefficients is worthless. Now scale from 2 features at κ = 1,999 to 512 features at κ = 4 × 10⁶.

### 3.4 The observed damage: R² ≈ −3

Measured on the real embeddings under the exact protocol (outer `GroupKFold(5)` on participants, `StandardScaler` fit on the training fold only) **[M]**:

| Target | plain `LinearRegression`, R² per fold | mean R² | mean RMSE |
|---|---|--:|--:|
| Grids | −4.351, −1.580, −3.510, −2.109, −3.608 | **−3.031** | 0.951 |
| Symbols | −2.768, −2.600, −3.350, −2.226, −2.370 | **−2.663** | 1.036 |
| Prices | −2.279, −3.620, −3.079, −4.186, −2.844 | **−3.202** | 35.00 |

Against Ridge with the α grid, same run **[M]**:

| Target | plain Linear | **Ridge** | mean-predictor baseline |
|---|--:|--:|--:|
| Grids | −3.031 | **−0.089** | −0.078 |
| Symbols | −2.663 | **−0.092** | −0.050 |
| Prices | −3.202 | **−0.072** | −0.004 |
| Grids RMSE | 0.951 | **0.505** | 0.503 |
| Prices RMSE | 35.00 | **17.75** | 17.18 |

Read the Grids row aloud: plain least squares has **roughly double the RMSE of simply announcing the average score every time**. R² = −3 means squared error four times the baseline's. That is the model confidently inventing structure in the ~150 near-empty directions and being wrong about it on four unseen participants.

Ridge on the *identical* features recovers almost all of it, landing within 0.011–0.068 R² of the baseline. **Nothing changed except the penalty term.** That is the cleanest evidence in the project that regularization on high-dimensional foundation-model embeddings is mandatory, not optional.

### 3.5 The `LinAlgWarning`, precisely

The repo mentions a `LinAlgWarning`. Be exact, because a code-reading supervisor will check. It is emitted from `sklearn/linear_model/_ridge.py:264`, at `dual_coef = linalg.solve(K, y, assume_a="pos")`, saying *"An ill-conditioned matrix detected: slice 0 has rcond = 1.1556e-07"*. Across the run above it fired **9 times** with `rcond` between **1.09 × 10⁻⁷ and 1.17 × 10⁻⁷** **[M]**. `rcond` is the reciprocal condition number, so it implies κ ≈ 9 × 10⁶ — same order as the 4 × 10⁶ measured directly.

Three details that make the answer airtight:
1. **It comes from Ridge, not `LinearRegression`.** `LinearRegression` uses `scipy.linalg.lstsq`, which silently returns a minimum-norm solution and never warns. The loudest failure is the *quiet* one.
2. It is the **dual** solver (`K = XXᵀ`, size rows × rows), which sklearn selects when the training matrix has fewer rows than columns — precisely the 5-of-15 inner folds with 488–491 rows against 512 features.
3. It appears at **small α**. Raising α removes it, and so does PCA: `docs/01_PIPELINE_DESIGN.md:67` records PCA ≈ 16–32 eliminating the warning on synthetic data.

> **Say it in your own words:** With 512 features and only about 764 rows, the matrix we have to invert is barely determined and hugely lopsided, so least squares assigns confident weights to directions the data cannot pin down — that is why plain linear regression scored about −3.

**Drill**
- *Q: How many parameters, on how many rows?* 513 on ~764, so 1.49 rows per parameter. In the inner tuning folds it drops to ~510 rows for 512 features, and 5 of 15 inner folds are underdetermined.
- *Q: Define condition number without equations.* How much the answer can move when the data moves a little. Four million means a one-in-a-million wobble can move the weights by a factor of four.
- *Q: Is `XᵀX` singular for us?* No. Full rank 512, smallest eigenvalue about 0.025. It is ill-conditioned, not singular — which is why nothing crashed and the failure was silent.
- *Q: Why is R² = −3 worse than R² = −0.09, in real terms?* R² = −3 means squared error 4× the baseline, i.e. about 2× the RMSE. R² = −0.09 means about 4% worse RMSE than the baseline. One is a breakdown; the other is a tie.

---

## 4. Ridge regression (L2 regularization)

### 4.1 The objective

Ridge changes one thing: it charges rent on large coefficients.

```
   minimise    Σ ( yᵢ − w·xᵢ − b )²   +   α · Σ wⱼ²
               └──── fit the data ───┘       └─ stay small ─┘
```

`α` (alpha) is a single positive number you choose. `Σ wⱼ²` is the squared length of the weight vector — the **L2 penalty**. The intercept `b` is deliberately **not** penalised: shrinking it would drag every prediction toward zero, meaningless for a score whose mean is 41.6 (Prices).

**In plain words:** the model can no longer buy a better fit at any price. Every unit of coefficient magnitude costs something, so it only takes a big coefficient when the data insists.

Our grid is `α ∈ {0.1, 1, 10, 100, 1000}` (`regression.py:43`) — five values over four orders of magnitude, spaced logarithmically because what matters is order of magnitude.

### 4.2 The two extremes

- **α → 0.** The penalty vanishes and Ridge becomes plain least squares. Measured cost of this end: R² ≈ −3.
- **α → ∞.** The penalty dominates, every `wⱼ` is driven arbitrarily near 0, and the model reduces to `ŷ = b = ȳ`. **It becomes the mean-predictor baseline.** So a badly over-regularized Ridge degrades gracefully into "guess the average", never into nonsense.

```
   α:  0 ───────────────────────────────────────────────▶  ∞
       │                                                   │
   plain least squares                            predicts the mean
   fits training noise                            ignores the features
   R² ≈ −3 on our data                            R² ≈ −0.08 / −0.05 / −0.004
       │                                                   │
       └──────────── somewhere in here ────────────────────┘
                     is the best available compromise
```

Worth stating clearly: on our data the α = ∞ end is *better* than the α = 0 end. That tells you how little signal there is.

### 4.3 The ridge closed form, and why it cures the inversion

```
   β̂_ridge  =  ( XᵀX  +  α I )⁻¹ Xᵀy
```

`I` is the identity matrix, so the only change from the normal equation is **adding α to every diagonal entry**.

**In plain words:** we pretend every feature has a bit more independent variance than it really does. That props up the flat directions so they are no longer nearly zero, and the inverse stops amplifying noise.

The effect on eigenvalues is exact — every `λ` becomes `λ + α` — so `κ_ridge = (λ_max + α)/(λ_min + α)`. Our measured λ_min ≈ 0.025 and λ_max ≈ 1.05 × 10⁵ give **[M]**:

| α | mean κ(XᵀX + αI) across the 5 folds | improvement vs α = 0 |
|--:|--:|--:|
| 0 | 4.01 × 10⁶ | — |
| 0.1 | 8.32 × 10⁵ | 4.8× better |
| 1 | 1.02 × 10⁵ | 39× |
| 10 | 1.05 × 10⁴ | 383× |
| 100 | 1.05 × 10³ | 3,800× |
| 1000 | **106** | **38,000×** |

α = 1000 turns four million into 106. And because α > 0 guarantees `λ_min + α > 0`, the matrix is **provably invertible** for any positive α, even with fewer rows than features. That is a structural guarantee, not just an improvement.

### 4.4 Which directions get shrunk — the part people miss

Ridge does not shrink all coefficients equally. In the coordinate system of `XᵀX`'s eigenvectors, the direction with eigenvalue `λ` is multiplied by **`λ / (λ + α)`**. Directions the data determines well (large λ) are almost untouched; directions it barely sees (small λ) are crushed. Measured on our real fold 1, where λ_max = 1.064 × 10⁵ and λ_min = 0.0275 **[M]**:

| α | factor on the *strongest* direction | factor on the *weakest* direction |
|--:|--:|--:|
| 1 | 0.999991 | 0.0268 |
| 10 | 0.999906 | **0.00275** |
| 100 | 0.999061 | 0.000275 |
| 1000 | 0.990687 | 0.0000275 |

At α = 10 the best-determined direction keeps 99.99% of its least-squares value while the worst keeps 0.27%. **Ridge is not a blunt instrument — it surgically removes exactly the directions responsible for the R² = −3 blow-up and leaves the rest alone.** If you say one thing about Ridge in a defence, say this.

### 4.5 Worked numeric example: coefficients shrinking

Reuse the 4-session example from §1.5. Features are centred, so the intercept stays `b = ȳ = 3` at every α. `XᵀX` for the two slopes is `[[4,0],[0,4]]` and `Xᵀy` is `(6, 4)`. Ridge adds α to each diagonal entry, so each slope is just `w₁ = 6/(4+α)` and `w₂ = 4/(4+α)`.

| α | w₁ | w₂ | SSE (fit) | penalty α·Σw² | total objective |
|--:|--:|--:|--:|--:|--:|
| 0 | **1.5000** | **1.0000** | 1.000 | 0.000 | 1.000 |
| 1 | 6/5 = 1.2000 | 4/5 = 0.8000 | 1.520 | 2.080 | 3.600 |
| 4 | 6/8 = **0.7500** | 4/8 = **0.5000** | 4.250 | 3.250 | 7.500 |
| 10 | 6/14 = 0.4286 | 4/14 = 0.2857 | 7.633 | 2.653 | 10.286 |
| 100 | 6/104 = 0.0577 | 4/104 = 0.0385 | 13.019 | 0.481 | 13.500 |
| 1000 | 6/1004 = 0.0060 | 4/1004 = 0.0040 | 13.897 | 0.052 | 13.949 |
| ∞ | 0 | 0 | **14.000** | 0 | 14.000 |

Every column verified numerically. Four things to point at:

- At **α = 4** (equal to the diagonal entry) both coefficients are **exactly halved**. Mnemonic: α equal to `XᵀX`'s diagonal halves the weights.
- SSE rises monotonically from 1.000 to 14.000. **Ridge always fits the training data worse than least squares.** That is the price; the hope is it fits *new participants* better — and on our data it does, by about 4× in squared error.
- At α = ∞, SSE = 14.000 = exactly the mean-predictor's error, confirming §4.2 numerically.
- Both coefficients shrink by the *same* factor here only because the features are uncorrelated and equally scaled. In the real 512-d case the factors differ enormously per direction (§4.4).

```
  |w|
  1.5 ●
      │  ╲
      │    ●                 the shrinkage path: coefficients slide toward 0
  1.0 │ ●    ╲               as α grows, but never quite arrive
      │   ╲    ●
      │     ●    ╲
  0.5 │       ╲    ●───────●─────────────●
      │         ●───────────────●
  0.0 ┼─────────────────────────────────────────  α
      0    1    4    10       100      1000
```

### 4.6 Ridge shrinks but never zeroes

Look at α = 1000: `w₁ = 6/1004 = 0.0060`. Small, but not zero. In general `λ/(λ+α) > 0` for any finite α, so **no Ridge coefficient is ever exactly 0**.

Consequence for us: **all 512 Chronos features stay in the model at every α we tried.** Ridge cannot tell us "these 40 dimensions matter and the other 472 do not". If you want that, you need Lasso.

> **Say it in your own words:** Ridge adds a fine for large weights, which mathematically means adding alpha to the diagonal so the matrix is always invertible, and it flattens exactly the directions the data cannot pin down.

**Drill**
- *Q: State the ridge objective and the closed form.* Minimise `Σ(y − w·x − b)² + α·Σw²`; solution `β̂ = (XᵀX + αI)⁻¹Xᵀy`.
- *Q: Why does adding αI guarantee invertibility?* Every eigenvalue becomes `λ + α`. With α > 0 and `λ ≥ 0`, all eigenvalues are strictly positive, so the determinant is non-zero. It holds even with more features than rows.
- *Q: What does Ridge do at α = 1000 on our data?* Condition number falls from 4 × 10⁶ to 106, and the weakest direction retains 0.0000275 of its least-squares value — effectively the mean-predictor.
- *Q: Is the intercept penalised?* No, in sklearn's Ridge it is not. Penalising it would drag every prediction toward zero, which would be wrong for Prices, whose mean is 41.6.
- *Q: Does Ridge ever set a coefficient to exactly zero?* No. It shrinks toward zero asymptotically. All 512 features remain in the model.

---

## 5. Lasso (L1), briefly

Swap the squared penalty for an absolute-value penalty and you get **Lasso**:

```
   Ridge:  minimise  Σ(y − ŷ)²  +  α · Σ wⱼ²        (L2 — squares)
   Lasso:  minimise  Σ(y − ŷ)²  +  α · Σ |wⱼ|       (L1 — absolute values)
```

The difference looks cosmetic and is not. **Lasso sets coefficients to exactly zero**, so it performs **feature selection** — it hands you a shortlist.

### Why L1 gives sparsity — the geometric intuition

Both penalties say "stay inside a budget of total coefficient size". The shape of that budget region differs.

```
      L2 budget (a circle)                 L1 budget (a diamond)

  w₂                                   w₂
   │        ,--''--,                    │          /\
   │      ,'        `.                  │        /    \
   │     /            \                 │      /        \
───┼────(------o-------)──── w₁     ────┼────<            >──── w₁
   │     \            /                 │      \        /
   │      `.        ,'                  │        \    /
   │        `--..--'                    │          \/

  least-squares contours (ellipses) grow outward from the unconstrained
  optimum until they first touch the budget region.

  On a circle, first contact is almost surely on a smooth arc
      → both w₁ and w₂ are non-zero, just smaller.
  On a diamond, the corners stick out and sit ON the axes,
      so first contact is very often exactly at a corner
      → one coefficient is exactly 0.
```

That is the whole intuition: **circles have no corners; diamonds do, and their corners lie on the axes, where coefficients are zero.** In 512 dimensions the L1 ball has an enormous number of such corners, edges and faces, and hitting one zeroes out many coefficients at once.

### Why we did not use Lasso

Four honest reasons, strongest last:

1. **It is not in our grid.** `regression.py:38-47` defines exactly `Ridge`, `SVR`, `LinearRegression`. Lasso was never run — say that plainly.
2. **Its selling point does not apply.** Lasso's value is interpretability: a shortlist to reason about. Chronos dimension 137 has no meaning to anyone, so a sparse subset of an opaque embedding is still opaque.
3. **Correlated features are Lasso's known weak spot.** With features correlating up to 0.953 **[M]**, Lasso picks one arbitrarily from each correlated cluster and zeroes its neighbours; a slightly different fold picks a different one. That is the §3.3 instability in a new costume.
4. **Ridge already landed at the ceiling.** It sits within 0.011–0.068 R² of the mean-predictor baseline **[M]**, and everything else we tried — bigger encoders, shorter windows, PCA, glucose-regime subgroups **[F]** — moved it by hundredths. No headroom for a different penalty to recover.

**If the advisor pushes:** *"Elastic Net — an α-weighted mix of L1 and L2 — is the standard next thing and is designed for exactly our correlated-feature case. It is fair criticism that we did not run it. I expect it lands where Ridge landed, because the shuffle test puts our real result at the bad end of the scrambled-score distribution (p = 1.000 / 1.000 / 0.995) [F], so there is no signal for a smarter penalty to find. But that is a prediction, not a measurement."*

> **Say it in your own words:** Lasso uses absolute values instead of squares, which pins some coefficients at exactly zero and gives you a shortlist of features — useless here, because nobody can interpret Chronos dimension 137, and we never ran it.

**Drill**
- *Q: One-line difference between Ridge and Lasso?* Ridge penalises squared coefficients and shrinks all of them; Lasso penalises absolute coefficients and zeroes some of them.
- *Q: Why do corners cause sparsity?* The L1 constraint region is a diamond whose corners lie on the coordinate axes. The expanding error contours usually touch a corner first, and being on an axis means the other coefficient is exactly 0.
- *Q: Did you run Lasso?* No. It is not in `regression.py:38-47`. Ridge, SVR, LinearRegression and the mean baseline are the whole set, and XGBoost was in the code but never executed because the package is not installed.

---

## 6. Bias, variance, and what α is actually trading

Two ways a model can be wrong.

- **Bias** — the model is too rigid to represent the truth. It is wrong the same way every time. The mean-predictor has maximum bias: it ignores glucose completely.
- **Variance** — the model is so flexible that it chases the particular noise in whichever rows it happened to see. Refit it on a different sample and it says something different. Plain least squares on 512 features has huge variance: §3.3 showed coefficients flipping sign from a 1% data change.

Expected error on unseen data splits into three pieces:

```
   expected test error  =  bias²  +  variance  +  irreducible noise
                              ↑         ↑            ↑
                       α pushes    α pushes      fixed by the data;
                        this up    this down      no model touches it
```

**α is the dial between the first two terms.**

```
  test
  error │                                         ╱  ← high variance:
        │╲                                      ╱      plain least squares,
        │ ╲                                   ╱        R² ≈ −3
        │  ╲                               ╱
        │   ╲__                        ╱
        │      ╲___             ____╱
        │          ╲___________╱          ← high bias: predicts the mean
        └──────────────────────────────────────  α
        ∞          the sweet spot          0
```

Two honest observations about our version of this curve:

1. **Our curve is almost flat.** Ridge (−0.089) and the mean-predictor (−0.078) differ by 0.011 R² on Grids **[M]**. The valley in the picture is barely a valley. This is what "no better than guessing the average" looks like in bias–variance language: the irreducible-noise term dominates both others.
2. **The irreducible term is large for a measurable reason.** Before any model, the strongest correlation anywhere between a glucose statistic and a score is r = −0.092, explaining **0.85% of the variance** **[F]**. And 62–92% of score variance is *within*-participant session-to-session variation **[F]**. Most of what we are asked to predict is not a function of glucose at all.

**How this connects to the inner loop.** Because the right α depends on how much noise this particular dataset has — which nobody can know in advance — α is not set by hand. It is selected by the inner cross-validation described next.

> **Say it in your own words:** Alpha trades "too rigid" against "too jumpy", and on our data the curve between those two failure modes is nearly flat because most of the score variation is not explainable from glucose at all.

**Drill**
- *Q: Which end of the α range is high variance?* Small α. No penalty, so the model chases noise. That is the R² = −3 end.
- *Q: If our best α is large, what does that say scientifically?* That the features carry little generalizable signal, so the honest thing to do is shrink toward the mean. A large selected α is itself a finding.
- *Q: Can regularization ever reduce the irreducible noise term?* No. It only reduces variance, at the cost of bias. Reducing irreducible noise needs better data — giving every session the same span of glucose, more participants, or a more reliable score.

---

## 7. Why α must be chosen on validation data, never on test data

### 7.1 The failure this prevents

Suppose we fit Ridge at all five α values, evaluate each on the held-out test fold, and report the best. We would be reporting a number that is optimistically biased, because we used the test scores to make a modelling choice. The test set has stopped being unseen. With 5 α values you get 5 chances to be lucky; the maximum of 5 noisy numbers is systematically above their average.

Concretely: our Grids per-fold Ridge R² values are −0.308, −0.136, −0.027, −0.010, **+0.038** **[M]**. One of the five folds is positive. Cherry-picking across α, or across folds, or across the three targets, could easily manufacture a positive-looking headline from data whose overall answer is "no better than guessing the average". **The protocol exists so we cannot do that even by accident.**

Terminology, defined once:
- **Training set** — used to fit coefficients.
- **Validation set** — used to compare hyperparameters (α, C, γ). Never used to fit coefficients.
- **Test set** — used exactly once, to report. Never used for any decision.

α is a **hyperparameter**: a setting of the learning procedure, not something least squares can solve for. Fitting α on the training data would always choose α = 0, since α = 0 fits the training data best by construction.

### 7.2 How the code enforces it

`regression.py:107-131` implements **nested cross-validation**.

```
  956 sessions, 20 participants
     │
     └── OUTER: GroupKFold(n_splits=5) on participant id        ← estimates accuracy
            4 participants held out per fold, ~191 test sessions
            │
            ├── assert set(train participants) ∩ set(test participants) == ∅
            │      (regression.py:113 — hard leakage guard, every fold)
            │
            └── INNER: GroupKFold(n_splits=3) inside GridSearchCV  ← chooses α
                   on the ~764 training sessions only
                   scoring = "neg_mean_squared_error"   (regression.py:124)
                   inner-train sizes measured: 488–523 rows  [M]
                   │
                   ├─ α = 0.1   → mean inner score
                   ├─ α = 1     → mean inner score
                   ├─ α = 10    → mean inner score
                   ├─ α = 100   → mean inner score
                   └─ α = 1000  → mean inner score
                        │
                        └── take the argmax, refit on all ~764 rows,
                            THEN predict the untouched ~191 test sessions
```

Three properties to be able to recite:

1. **The test fold is touched exactly once per fold**, at `best.predict(X[test_idx])` (`regression.py:133`). No hyperparameter ever sees it.
2. **α is re-chosen independently in every outer fold.** There is no single global α. The reported number is "the accuracy of the *procedure* including tuning", which is what you would actually get on a new participant.
3. **Grouping propagates.** `GridSearchCV` is called with `groups=groups[train_idx]` (`regression.py:127-128`), so the inner splits are also participant-disjoint. Without that, a participant's sessions would be split across inner train and inner validation, and α would be tuned to a task easier than the real one.

Fit counting, so you can do it live **[F]**: Ridge = 5 α × 3 inner folds = 15 scoring fits, + 1 refit = 16 per outer fold, × 5 outer folds = **80 fits**. SVR = 6 settings × 3 = 18, + 1 = 19, × 5 = **95**. Plus baseline 5 and Linear 5 → **185 fits per target**, **555 across the three targets**.

### 7.3 Why the split is by participant

`GroupKFold` guarantees no participant appears in both training and test. Random row-splitting would put ~47 sessions from the same child on both sides, and the model could learn "this child scores 0.58 on Grids" and score well without learning anything about glucose. The question the grant asks is *"does glucose predict cognition in a child we have never seen?"* — only a grouped split asks that question. Full treatment in `05_EVALUATION.md`.

### 7.4 The baseline is not exactly zero — know this

A subtle point that looks like an inconsistency if you cannot explain it. `DummyRegressor(strategy="mean")` measures **[M]** −0.078 (Grids), −0.050 (Symbols), −0.004 (Prices) — not 0.000.

**Why:** the Dummy predicts the *training* fold's mean, but R² is computed against the *test* fold's own mean. Because the 4 held-out participants have their own average scores (Cohort-2 children score worse on all three tests **[F]**), the training mean is the wrong constant for them, so even the trivial model scores slightly below zero.

**What this changes.** Ridge at −0.089 versus a baseline at −0.078 is a **tie**, not a defeat. The ceiling achievable by any model that cannot see which child it is looking at is itself slightly negative. State it that way — it is both more accurate and more defensible than "our model was worse than a trivial baseline".

> **Say it in your own words:** The outer split reports the score and the inner split picks alpha, so no number we report was ever chosen using the data it is scored on.

**Drill**
- *Q: Why not just pick the α with the best test R²?* Because then the test set was used to make a decision and the reported accuracy is optimistic. With 5 α values, 5 folds and 3 targets there are dozens of chances to be lucky.
- *Q: Exactly how many folds, and how many participants held out?* Outer 5-fold GroupKFold, 4 of 20 participants held out per fold, ~191 test sessions; inner 3-fold GroupKFold on the ~764 training sessions.
- *Q: Is there one α for the whole project?* No. It is re-selected inside each outer fold. The reported R² is the accuracy of the whole tune-then-fit procedure.
- *Q: What guards against leakage?* An `assert` at `regression.py:113` checks participant sets are disjoint on every outer fold, and `groups=` is passed into `GridSearchCV` so inner folds are grouped too. `StandardScaler` and PCA live inside the `Pipeline`, so they are fit on training rows only.

---

## 8. Support Vector Regression from absolute zero

Ridge asks "which weighted sum is closest to the scores?" SVR asks a different question: **"which function stays within ε of almost every point, and is as flat as possible?"**

### 8.1 The epsilon-insensitive tube

Pick a tolerance `ε`. Draw a band of half-width `ε` around your prediction function. **Any point inside the band costs nothing at all.** Only points outside are penalised, and only by how far outside they are.

```
   y
   │                            ● ← outside the tube: costs (distance − ε)
   │                          ⋰
   │        ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   upper edge  (ŷ + ε)
   │   ●    │       ●        ●          │
   │ ───────┼───────────────────────────┼──  the fitted function ŷ
   │   ●    │   ●        ●       ●      │
   │        └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   lower edge  (ŷ − ε)
   │    ●  ← outside below: also costs
   │
   └──────────────────────────────────────── x

   Points strictly inside the tube  →  zero loss, zero influence on the fit.
   Points on the edge or outside    →  these are the SUPPORT VECTORS.
                                       They alone determine the function.
```

This is the **ε-insensitive loss**. Contrast squared error, where *every* point pulls on the fit and one wild point drags the whole line. SVR ignores the comfortable majority and is determined by the awkward minority, which makes it robust to small residual noise.

**Support vectors** are exactly the training points on or outside the tube. Delete every interior point, refit, and you get the identical function. The name is literal: those points hold the tube up.

**Our ε, precisely.** `regression.py:44` writes `SVR(kernel="rbf")` and tunes only `C` and `gamma`, so **ε keeps its sklearn default of 0.1 and was never tuned** — verified. Because ε applies to the *raw target* (we scale features, not targets), the same 0.1 means very different things per score **[M]**:

| Target | target SD | ε = 0.1 as a fraction of SD |
|---|--:|--:|
| Grids | 0.500 | 0.200 SD |
| Symbols | 0.561 | 0.178 SD |
| Prices | 17.179 | **0.006 SD** |

So for Grids and Symbols the tube is a meaningful ±0.2 SD, while for Prices it is effectively zero-width and SVR degenerates toward ordinary robust regression. Nobody has raised this. It is a legitimate small criticism of our setup and you should volunteer it rather than be caught by it.

### 8.2 C — the price of leaving the tube

`C` multiplies the penalty for points outside the tube:

```
   minimise    ½‖w‖²   +   C · Σ (how far each point falls outside the tube)
               └ flatness ┘      └──────── fit the awkward points ────────┘
```

- **Small C** (we tried 0.1): leaving the tube is cheap, so the model stays flat and ignores outliers. *Strong* regularization.
- **Large C** (we tried 10): leaving the tube is expensive, so the model contorts to pull points inside. *Weak* regularization; risks overfitting.

Mnemonic: **C is the inverse of α.** Large C = small α = trust the data. Small C = large α = distrust it. Our grid is `C ∈ {0.1, 1, 10}` (`regression.py:45`).

### 8.3 The kernel trick and the RBF kernel

To let SVR bend you could invent new features (squares, products) and fit a line in that bigger space. Expensive: for 512 features, the pairwise products alone are over 131,000 new columns.

**The kernel trick** avoids building them. The SVR solution only ever needs *dot products between pairs of training points*, never the points themselves. So replace the dot product with a **kernel function** `K(x, x′)` that returns what the dot product *would have been* in some richer space. You get that space's flexibility for the cost of one number per pair.

The **RBF (Radial Basis Function) kernel**, also called the Gaussian kernel:

```
   K(x, x′)  =  exp( −γ · ‖x − x′‖² )
```

**In plain words:** a **similarity score between two sessions that fades as they get further apart.** Two identical embeddings give `K = exp(0) = 1`; far-apart ones give near 0. The prediction for a new session is then a weighted average of the support-vector sessions, weighted by similarity to each. It is "find the sessions that resemble this one, blend their answers." Concrete values at γ = 0.5 (verified):

```
   distance ‖x − x′‖ :   0      0.5     1.0     2.0     3.0     4.0
   K(x, x′)          :  1.000  0.882   0.607   0.135   0.011   0.0003
```

### 8.4 Gamma, and a sharp finding about our grid

`γ` sets how quickly similarity decays, i.e. how wide each session's zone of influence is.

- **Small γ** → slow decay → every session influences every other → a smooth, nearly linear function. Under-fits.
- **Large γ** → fast decay → each session influences only its immediate neighbours → a spiky function with a bump at each training point. Over-fits.

sklearn offers two automatic settings:

```
   gamma = "auto"   →  1 / n_features
   gamma = "scale"  →  1 / (n_features × X.var())        X.var() over the whole matrix
```

**Here is the finding.** Our pipeline is `StandardScaler → SVR`, no PCA, in the headline run. After standardization every column has mean 0 and variance exactly 1, so the variance of the whole matrix is 1. Therefore:

```
   "scale"  =  1 / (512 × 1)  =  0.001953125002778425     ← measured [M]
   "auto"   =  1 / 512        =  0.001953125              ← measured [M]
```

They agree to eleven decimal places — a floating-point difference, nothing more. **So in the no-PCA configuration the γ grid is effectively one value, and SVR's "6 settings" are really 3 distinct settings (the three C values), each fitted twice.** The 95 SVR fits per target **[F]** are honest as a count of work done, but the *search* explored 3 configurations, not 6. This stops being true once PCA is inserted, because PCA's output columns have unequal variances: with PCA(32), `scale = 0.002134` versus `auto = 0.031250` **[M]**, about 15× apart.

**Is γ = 1/512 sensible?** On the real standardized features the median squared distance between two sessions is 1012.6, so the typical exponent is `γ‖x−x′‖² = 1012.6/512 = 1.978`, giving a typical kernel value of **0.138**; the 5th percentile of distances gives K = 0.461, the 95th gives K = 0.026 **[M]**. The kernel sits in a healthy range — not saturated at 1 (every session looking identical) nor collapsed to 0 (every session an island). A reasonable choice, but a *default*, not a tuned one.

### 8.5 Why SVR needs standardized features

The kernel depends only on `‖x − x′‖²`, a **distance**, and distances are dominated by whichever features have the largest numeric range. If one Chronos dimension varied over ±50 and another over ±0.01, the first would supply essentially all of the distance and the second would be invisible — regardless of which carried information.

`StandardScaler` puts every feature on mean 0, SD 1, so each contributes comparably. In `regression.py:116-120` it is the first step of a `Pipeline`, so it is **fit inside each fold on training rows only** and the fitted centre and scale are applied to the test rows. Fitting on all 956 sessions first would leak test-fold information into training — see `03_PREPROCESSING.md`. (Ridge also benefits, since a single α is only fair if features share a scale. But SVR *requires* it: an unscaled kernel is meaningless.)

### 8.6 What SVR actually got us: essentially a tie

Be straight about this. From `results/headtohead_real.md`, the best model per target under grouped CV **[F]**:

| Score | best model | R² | Ridge alone, same features **[M]** |
|---|---|--:|--:|
| Grids | **Ridge** | −0.089 ± 0.123 | −0.089 |
| Symbols | **SVR** | −0.056 ± 0.060 | −0.092 |
| Prices | **SVR** | −0.012 ± 0.010 | −0.072 |

SVR won two of three targets. But look at the fold-to-fold standard deviations: ±0.060 on Symbols against a 0.036 gap, ±0.010 on Prices against a 0.060 gap. And every value is below the mean-predictor's −0.050 / −0.004 **[M]**. **Both models are tied with each other and tied with guessing the average.** Do not present SVR as the better method; present it as a second, non-linear method that reached the same conclusion, which strengthens the finding rather than the model.

> **Say it in your own words:** SVR draws a tolerance band and only pays for points outside it, and the RBF kernel lets it predict a new session by blending the training sessions that look most similar — it landed in the same place as Ridge.

**Drill**
- *Q: What is ε, and what is ours?* The half-width of the no-cost tube. Ours is sklearn's default 0.1 on the raw target, never tuned — 0.20 SD for Grids but only 0.006 SD for Prices.
- *Q: What is a support vector?* A training session on or outside the tube. Only those determine the fitted function; interior sessions could be deleted with no change.
- *Q: What does the RBF kernel compute, and what does γ control?* `exp(−γ‖x−x′‖²)`, a similarity between two sessions that decays with distance — 1 when identical, near 0 when far apart. γ sets how fast it decays, i.e. each session's radius of influence. Ours was effectively 1/512 = 0.00195, since after StandardScaler "scale" and "auto" coincide to eleven decimals.
- *Q: Why must features be standardized for SVR, and did SVR beat Ridge?* The kernel is a function of Euclidean distance, so any feature with a larger numeric range would dominate regardless of relevance. And no — SVR was nominally best on Symbols and Prices, but the gaps are inside the fold-to-fold standard deviations and both models are level with the mean-predictor. Call it a tie.

---

## 9. PCA, properly — and its trap

### 9.1 What PCA does

**Principal Component Analysis** rewrites the data in a new coordinate system whose axes are ranked by how much the data spreads along them. Procedure: centre the features; find the direction in 512-d space along which the sessions are most spread out (**PC1**); find the direction of greatest remaining spread that is **orthogonal** — at right angles — to PC1 (**PC2**); continue. Keep the first `k`, discard the rest.

```
   original axes: two correlated Chronos dimensions

   x₂ │            ●  ●                     PC1: the long axis of the cloud
      │        ●  ● ● ●   ●                      (most spread)
      │     ●  ● ●  ●  ●                    PC2: perpendicular to PC1,
      │  ●  ● ●  ●  ●                            the short axis
      │ ● ● ● ●
      └────────────────────── x₁

   after PCA: rotate onto the cloud's own axes

              PC2 (little spread — drop it)
               ↑
               │  ·  ·  ·  ·  ·  ·  ·
     ──────────┼───────────────────────▶ PC1 (lots of spread — keep it)
               │  ·  ·  ·  ·  ·  ·  ·

   The cloud is unchanged. Only the description changed — and now the
   information is concentrated in fewer coordinates.
```

**Explained-variance ratio** is PCA's report card: the fraction of total variance each component accounts for. Measured on our real embeddings, averaged across the 5 outer training folds **[M]**:

| components kept | cumulative variance explained | κ after PCA **[M]** |
|--:|--:|--:|
| 1 | 26.8% | — |
| 2 | 41.8% | — |
| 4 | 57.6% | — |
| 8 | **72.4%** | **13.1** |
| 16 | 83.1% | 29.7 |
| 32 | **91.5%** | **85.3** |
| 64 | 96.7% | 326.8 |
| 128 | 99.2% | 1,661 |
| 512 | 100.0% | 4.01 × 10⁶ |

Read the first row again: **a single direction accounts for 26.8% of all variation in the 512 Chronos numbers**, and eight directions account for 72.4%. The embedding is nominally 512-dimensional but behaves as though it has a few dozen dimensions of real content. Chronos was trained on general time-series data; our inputs are 956 glucose windows from 20 children — a narrow slice of what it can represent.

### 9.2 Why PCA helps conditioning

Compare the right-hand column above to §3.2. Reducing 512 → 32 takes the condition number from **4.01 × 10⁶ to 85.3** — a 47,000× improvement — while retaining 91.5% of the variance **[M]**. The discarded 480 directions are exactly the flat ones whose tiny eigenvalues were being inverted into enormous noisy weights. It also fixes the rows-per-unknown arithmetic: `764/513 = 1.49` becomes `764/33 = 23.2`, from a starved fit to a comfortable one. And it removes the `LinAlgWarning` (`docs/01_PIPELINE_DESIGN.md:67`).

**How the code does it safely.** `regression.py:116-120` inserts PCA as the middle `Pipeline` step: `StandardScaler → PCA(k) → model`. Because it lives in the pipeline, it is refit **inside every fold on training rows only** — the comment at `regression.py:104-106` says so explicitly. Fitting PCA on all 956 sessions first would let test-fold sessions influence the axes, a real and common form of leakage.

### 9.3 What PCA bought us: almost nothing

Measured PCA sweep, mean grouped-CV R² across the three targets **[F]**:

| PCA | mean R² |
|---|--:|
| none (512) | −0.052 |
| 8 | **−0.045** |
| 16 | −0.046 |
| 32 | **−0.045** |
| 64 | −0.053 |
| 128 | −0.051 |

The improvement from 512 → 32 is **0.007 R²**. A 47,000× improvement in conditioning bought seven thousandths of R², and the result stayed below zero. The correct reading: **ill-conditioning was a real defect and PCA fixed it, but ill-conditioning was not what stood between us and a signal.** There was no signal to uncover. Note too that 64 and 128 components are *worse* than no PCA — the bias–variance curve again: too many components reintroduce the flat noisy directions while still discarding a little real information.

### 9.4 The trap: PCA is unsupervised

**PCA never looks at the score.** It sees only `X`, and it ranks directions by variance — but variance is not usefulness. If the one direction that predicts cognition happens to be low-variance, PCA discards it, and it does so silently, because from PCA's point of view nothing went wrong. The measured demonstration:

```
  Experiment 1 — no PCA
    features : [mean glucose, glucose SD]        (2 columns)
    target   : mean glucose
    result   : R² = 1.0000                       ← perfect, as it must be:
                                                    the answer IS a column

  Experiment 2 — same target, MORE information, plus PCA(32)
    features : [512 Chronos numbers] + [mean glucose, glucose SD]   (514 columns)
    target   : mean glucose
    result   : R² = 0.0853                       ← collapsed
```

**We added information and accuracy fell from 1.0000 to 0.0853.** No bug. PCA(32) built its 32 axes from the highest-variance directions of the 514-column matrix; those are dominated by the 512 Chronos dimensions; and the two hand-crafted columns that contained the answer outright were smeared across the 482 discarded directions.

**The one-sentence rule:** *PCA keeps the directions where the data varies most, not the directions that predict the target — and it cannot tell the difference, because it never sees the target.*

Three consequences, worth saying unprompted:

1. Our PCA results mean *"reducing dimensionality did not help"* — **not** *"the information is definitely not in the discarded 480 directions."*
2. The supervised alternative is **PLS (Partial Least Squares)**, which maximises *covariance with the target* rather than variance. It is the natural next experiment. We did not run it — say so plainly.
3. Our headline results (`results/headtohead_real.md`, PCA `none`) use **all 512 features with no PCA**, so the headline conclusion is not exposed to this trap. Only the PCA sweep is.

> **Say it in your own words:** PCA finds the directions the data spreads out along and drops the rest, which fixes the numerical instability — but it never looks at the score, so it can throw away the exact column you needed, and we measured it doing that.

**Drill**
- *Q: What does PCA optimise?* Variance. It finds orthogonal directions of maximum spread in the features, ranked. It does not use the target at all.
- *Q: What is explained-variance ratio, with our numbers?* The fraction of total feature variance a component carries. Our PC1 carries 26.8%; the top 8 carry 72.4%; the top 32 carry 91.5%.
- *Q: Why does PCA help a linear model here?* It removes the near-flat directions that make `XᵀX` ill-conditioned: condition number 4.0 × 10⁶ → 85.3 at k = 32, and rows-per-unknown 1.49 → 23.2.
- *Q: Give an example where PCA hurts.* Predicting mean glucose from `[mean, sd]` gives R² = 1.0000. Append those two columns to the 512 Chronos features and apply PCA(32) and it falls to 0.0853, because the informative columns were low-variance relative to the embedding and got discarded.
- *Q: How do you avoid leaking through PCA?* Put it inside the `Pipeline` so it is refit on training rows only in every fold. We do: `regression.py:118`.

---

## 10. Why we deliberately used simple models

This will be challenged as "you didn't try hard enough". It is a design decision with an arithmetic justification.

### 10.1 The arithmetic of what the data can support

```
   participants ............................ 20        [F]
   sessions ................................ 956       [F]
   participants held out per outer fold ...  4          [F]
   training sessions per fold .............. ~764       [F]
   effective independent units ............. closer to 20 than to 956
```

That last line is the one that matters. Sessions from the same child are not independent — 15–38% of score variance is between-participant **[F]** — so the number of independent examples for a "predict an unseen child" question is nearer **16 training participants per fold** than 764 rows. A model with more freedom than that can support will fit whichever 16 children it saw and generalise nothing.

### 10.2 The measured evidence that complexity hurt

We do not have to argue this. It was tested three ways, and complexity lost every time.

**(a) The neural head did worse.** Arm B is `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`, **132,609 trainable parameters** on **~612** training rows — **217× more parameters than examples** **[F]**. Result **[F]**:

| Score | Chronos + Ridge/SVR | Chronos + MLP head |
|---|--:|--:|
| Grids | −0.089 | **−0.218** |
| Symbols | −0.056 | **−0.281** |
| Prices | −0.012 | **−0.168** |

The more flexible model was 2.4× to 14× worse on R². Dropout, weight decay and early stopping (patience 8) were all in place and were not enough.

**(b) A bigger encoder did worse.** Mean R² by checkpoint **[F]**: bolt-tiny (d=256) −0.049, bolt-mini (384) −0.052, bolt-small (512) −0.052, **bolt-base (768) −0.063 — the worst**, t5-small (512) −0.057. Scaling up the representation made it worse, not better.

**(c) One feature matched all 512.** Ridge under grouped CV **[F]**:

| Score | mean glucose only (1 feature) | Chronos (512 features) |
|---|--:|--:|
| Grids | **−0.075** | −0.114 |
| Symbols | **−0.050** | −0.093 |
| Prices | **−0.004** | −0.069 |

One number per session did **better** than 512. And note where those single-feature values land: −0.075 / −0.050 / −0.004 sit essentially on top of the measured mean-predictor baseline of −0.078 / −0.050 / −0.004 **[M]**. The honest reading is a chain of ties: *one feature ≈ 512 features ≈ predicting the average.* Adding capacity added variance and no signal.

*(Caveat to state yourself: that paired comparison's Chronos column reads −0.114/−0.093/−0.069, while the headline table reads −0.089/−0.056/−0.012. The headline takes the better of Ridge and SVR per target and comes from a separate run; the paired run is Ridge-only. Compare within a run, never across. My own Ridge-only measurement gives −0.089/−0.092/−0.072 **[M]**, close to the paired run on Symbols and Prices. Flag the discrepancy rather than average it away.)*

### 10.3 The scientific argument

Two more reasons beyond "it worked better".

**Comparability.** Arm A uses the *identical* protocol as the advisor's hand-crafted-feature pipeline — same 5-fold GroupKFold, same inner 3-fold GridSearchCV, same scaler placement, same metrics (`regression.py:1-17`). Only the feature matrix changes: (956, 512) Chronos versus (956, 43) hand-crafted **[F]**. That is what makes "raw-data versus feature-engineering" a fair comparison instead of two unrelated experiments. Swapping in an exotic model would have destroyed the comparison.

**Interpretability of the conclusion.** With a simple regularized model, "no better than guessing the average" is a statement about the *information in the features*. With a 132,609-parameter network on 612 rows, the same outcome is ambiguous — it could be the features, the architecture, the learning rate, the initialisation, or the early-stopping patience. Simple models make the low accuracy *attributable*.

**Say the quiet part out loud:** "The result is that nothing beat guessing the average. If I had produced that with a complicated model, the first question would be whether I had simply mis-tuned it. Because I produced it with Ridge — 5 hyperparameter values, a closed-form solution, and no randomness beyond the fold assignment — the result is reproducible and attributable to the data."

### 10.4 One honest gap

**XGBoost never ran.** `regression.py:171-179` imports it inside a `try/except` and **xgboost is not installed** in this environment **[F]**, so it was silently skipped in every result. If asked "did you try gradient boosting?", the answer is **no**. Do not say "we tried tree ensembles". The code would have; the package was absent. This is the single most likely place for a code-reader to catch an overstatement.

> **Say it in your own words:** With 20 participants, a regularized linear model is the most complex thing the data can honestly support — and we can show the more complex options we did run were measurably worse.

**Drill**
- *Q: Why not a deep network?* We ran one. 132,609 parameters on ~612 rows, and it scored −0.218 / −0.281 / −0.168 against Ridge's −0.089 / −0.056 / −0.012. More capacity, less accuracy.
- *Q: Isn't a simple model just a weaker attempt?* No. A simple model makes the conclusion attributable to the features rather than to tuning choices, and it preserves comparability with the hand-crafted-feature pipeline, which used the same protocol.
- *Q: Did you try gradient boosting?* No. XGBoost is in the code but wrapped in try/except and the package is not installed, so it never executed.
- *Q: What is the strongest single sentence for the simple-model choice?* One feature — mean glucose — did as well as all 512, and both did as well as predicting the average. There was nothing for extra capacity to find.

---

## 11. Which model would you use, and why

| Situation | Model | Why |
|---|---|---|
| Many features, few rows, features correlated | **Ridge** | Guarantees invertibility, shrinks only the poorly-determined directions, one hyperparameter, closed form, deterministic. Our default. |
| Same, but you need a shortlist of which features matter | **Lasso** / **Elastic Net** | Zeroes coefficients. Only worth it if features have meaning — Chronos dimensions do not. |
| You suspect a smooth non-linear relationship | **SVR (RBF)** | Non-linear via the kernel, no explicit feature expansion, robust to small residuals via the ε tube. We ran it; it tied with Ridge. |
| Features are meaningful and interactions are expected | **Gradient boosting** | Handles interactions and monotone thresholds naturally. **We did not run it** — not installed. |
| You must know whether information is *linearly decodable* from a frozen representation | **Ridge (a linear probe)** | This is the standard evaluation for frozen foundation-model embeddings, and it is our actual research question. |
| Conditioning is the problem and features are anonymous | **PCA → Ridge** | κ 4.0 × 10⁶ → 85.3 at k = 32. But PCA is unsupervised and can discard the useful direction. |
| You want dimension reduction that respects the target | **PLS** | Maximises covariance with the target, not variance. Not run. The most defensible "what next". |
| Thousands of rows per participant and many participants | **A trainable head or fine-tuning** | Only then. We have 20 participants; Arm B already showed what happens with 217 parameters per row. |
| Establishing whether anything works at all | **`DummyRegressor(mean)`** | Always report it. Ours is not 0.000 — it is −0.078 / −0.050 / −0.004, and that is the number to beat. |

**Our answer, in one line:** *Ridge, with α selected by an inner 3-fold participant-grouped GridSearchCV, on standardized 512-dimensional frozen Chronos embeddings — with SVR as a non-linear cross-check and a mean-predictor as the reference.*

---

## 12. The four hardest questions

**Q1. "Plain linear regression got R² = −3 and Ridge got −0.089 on the same features. Explain the difference without saying the word 'regularization'."**

There are 512 features and about 764 training rows, so 513 unknowns on 1.49 rows each. When you build the 513 × 513 matrix the formula has to invert, its condition number is about 4 million: it is stretched a hundred thousand times more along its strongest direction than its weakest, and 148 of the 512 directions have less than one unit of information in them. Plain least squares still assigns confident weights along those near-empty directions, and on four unseen children those weights are wrong — the RMSE on Grids is 0.951 against a baseline of 0.503, roughly double. Ridge adds a small constant to every diagonal entry of that matrix. Mathematically each direction's coefficient gets multiplied by λ/(λ+α), so at α = 10 the strongest direction keeps 99.99% of its value while the weakest keeps 0.27%. It deletes the unreliable directions and keeps the reliable ones. That is why the same features go from −3 to −0.089.

**Q2. "Your reported R² values are all negative, which means your model is worse than a trivial baseline. Why should I believe the pipeline works?"**

Two things. First, the trivial baseline is itself negative here: `DummyRegressor(mean)` measures −0.078, −0.050 and −0.004 on Grids, Symbols and Prices, because it predicts the training fold's mean while R² is scored against the held-out fold's own mean, and the four held-out children have their own average scores. Ridge at −0.089 is therefore a tie with the baseline, not a defeat by it. Second, the same pipeline, unchanged, predicts glucose *variability* from the same embeddings at R² = 0.462. So the machinery decodes real information when real information is there. The honest weak spot is that mean glucose only reaches 0.070, because Chronos internally rescales each series and largely discards absolute level — which means our Chronos result speaks to glucose *shape*, not *level*. The 43 hand-crafted features do encode absolute level, and they were flat too, so the conclusion holds for level-aware features as well.

**Q3. "You reduced 512 dimensions to 32 with PCA. How do you know the signal wasn't in the 480 you threw away?"**

I do not, and PCA cannot tell me, because PCA never looks at the score — it ranks directions purely by how much the features spread along them. I can show it failing on purpose: predicting mean glucose from `[mean, sd]` gives R² = 1.0000, because the answer is literally a column; append those two columns to the 512 Chronos features and apply PCA(32) and it falls to 0.0853, since the informative low-variance columns get discarded. So I do not lean on the PCA result. Our headline numbers use all 512 features with no PCA at all, which is not exposed to this failure mode. PCA was run as a conditioning experiment: it improved the condition number from 4.0 × 10⁶ to 85.3 while keeping 91.5% of the variance, and improved mean R² by 0.007, from −0.052 to −0.045. The correct supervised alternative is PLS, which maximises covariance with the target instead of variance. We have not run it, and it is the most defensible next step.

**Q4. "SVR with an RBF kernel is a much more powerful model than Ridge. Why didn't it help?"**

It is more expressive, and it did not help, which is informative. It was nominally best on two of three targets — Symbols −0.056 and Prices −0.012 — but the fold-to-fold standard deviations are ±0.060 and ±0.010, larger than or comparable to the gaps, so it is a tie with Ridge and with the mean-predictor. Three specifics I would add. Our γ grid was effectively one value, not two: after StandardScaler the whole matrix has variance 1, so "scale" = 1/(512 × 1) and "auto" = 1/512 agree to eleven decimal places. Our ε stayed at sklearn's default 0.1 and was never tuned, which is 0.20 SD for Grids but only 0.006 SD for Prices, so the tube is meaningful for one target and near-zero for another. And the reason extra flexibility gains nothing is upstream of the model: the strongest correlation anywhere between a glucose statistic and a score explains 0.85% of the variance, and 62–92% of score variance is within-participant session-to-session variation. A more flexible function class cannot manufacture a relationship that is not in the data — it can only fit more noise, which is exactly what Arm B's 132,609-parameter head did when it scored −0.218 / −0.281 / −0.168.

---

## 13. Ten-sentence summary

1. A linear model predicts a score as a weighted sum of the 512 Chronos numbers plus an intercept, and fitting means choosing those 513 values to minimise squared error.
2. The normal equation `β̂ = (XᵀX)⁻¹Xᵀy` solves that in one step, which requires inverting a 513 × 513 matrix.
3. "Linear" constrains how the weights combine, not the shape of the relationship — Chronos supplies the non-linearity, so the composition is highly non-linear in glucose.
4. With 513 unknowns on about 764 rows (1.49 rows each), and only 488–523 rows inside the tuning folds, that matrix has a measured condition number near 4 × 10⁶, so the fitted weights are unstable — plain `LinearRegression` scored R² = −3.03 / −2.66 / −3.20, roughly double the baseline RMSE.
5. Ridge minimises squared error plus `α·Σw²`, which mathematically adds α to every diagonal entry, guaranteeing invertibility and cutting the condition number to 106 at α = 1000.
6. Ridge shrinks each direction by `λ/(λ+α)`, so at α = 10 the best-determined direction keeps 99.99% of its value and the worst keeps 0.27% — it deletes precisely the directions responsible for the blow-up, taking R² from −3 to −0.089 with no other change.
7. α is a bias–variance dial that must be chosen on validation data, which the inner 3-fold participant-grouped `GridSearchCV` does, so the outer test fold is touched exactly once and no reported number was chosen using the data it is scored on.
8. SVR ignores errors inside an ε-wide tube, penalises the rest at rate C, and uses the RBF kernel `exp(−γ‖x−x′‖²)` as a distance-decaying similarity between sessions — it needs standardized features and, on our data, tied with Ridge.
9. PCA rotates onto orthogonal directions of maximum variance, which fixed the conditioning (4 × 10⁶ → 85.3 at 32 components, 91.5% of variance retained) but bought only 0.007 R², and because it is unsupervised it can discard the useful direction — measured, a perfect R² = 1.0000 collapsed to 0.0853.
10. With 20 participants, simple regularized models are the most complex choice the data can honestly support, and we can prove it: one feature matched all 512, both matched guessing the average, a bigger encoder was worse, and the 132,609-parameter head was worst of all.

---

## Appendix — provenance of the **[M]** numbers

Everything marked **[M]** was computed on 2026-07-30 with the project environment (`/data_1_8TB_ssd/brian_workspace/envs/cgm`) from:

- `.cache/embeddings/chronos__amazon_chronos-bolt-small__mean__4b2c75e042223011.npy` — shape `(956, 512)`, the cached frozen `chronos-bolt-small` mean-pooled embeddings used for the headline results.
- `cgm_tsfm.data.load_real_data()` for targets and participant ids.
- `sklearn.model_selection.GroupKFold(n_splits=5)` on participant id for outer folds and `GroupKFold(n_splits=3)` for inner folds — the same objects the pipeline uses.
- `StandardScaler` fit on the training rows of each fold, then eigenvalues of `XᵀX` via `numpy.linalg.eigvalsh`.
- `nested_group_cv` from `cgm_tsfm/regression.py` for the `LinearRegression`, `Ridge` and `DummyRegressor` R² and RMSE figures, with `warnings.catch_warnings(record=True)` to capture the `LinAlgWarning`.

Reproduce any of them by loading that `.npy`, splitting with `GroupKFold(5)` on `dataset.subjects`, and repeating the step above. Numbers marked **[F]** are from `01_FACT_SHEET.md`, which is the authority when documents disagree.

---

*Next: `05_EVALUATION.md` — R², RMSE, MAE, why grouped cross-validation is the only honest split here, the shuffle test, and how to present accuracy that is level with guessing the average.*
