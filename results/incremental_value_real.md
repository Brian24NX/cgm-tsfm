# Giving the model the child's own history — the first positive R²

**Date:** 2026-09-18 · **Data:** real, 20 participants, uniform 2 h window
**Model:** `StandardScaler → Ridge` (alpha chosen by inner `GroupKFold(3)` GridSearchCV on
neg-MSE), outer `GroupKFold(5)` on participant id — the same protocol as every other results file.

---

## 1. Why we went looking

Every number in this project so far has been at or below a mean-predictor. But the metric we were
using asks a harder question than the one we care about. **R² under leave-participants-out CV
asks: can you predict a new child's score better than the average child's score?** Meanwhile:

| Score | Variance between children | Variance **within** the same child |
|---|--:|--:|
| Grids | 15.4% | **84.6%** |
| Symbols | 37.7% | **62.3%** |
| Prices | 7.7% | **92.3%** |

Most of what we are trying to predict is session-to-session variation inside one child. A model
that only sees a 2-hour glucose window has nothing to anchor a child's own level with, so it is
forced to predict near the global average for everyone — which is exactly the flat behaviour we
have been reporting.

**The fix is to give it an anchor that is legitimately available at prediction time:** the child's
own earlier scores. For session *t* we use only sessions *< t* for that child:

- `prev` — the score at their immediately preceding session
- `prior_mean` — the running average of all their earlier sessions
- `prior_n` — how many earlier sessions there were

This is strictly causal (no future information, no other child's information) and it is what a
deployed system would actually have. Sessions with no earlier session are dropped, so every row
below is scored on the same reduced set: **891 / 862 / 877 sessions** (grids / symbols / prices).

---

## 2. The result

| Features | Grids | Symbols | Prices |
|---|--:|--:|--:|
| guess the average (baseline) | −0.076 | −0.299 | −0.013 |
| glucose curve, Chronos 512-d *(what we had)* | −0.097 | −0.326 | −0.020 |
| 9 plain glucose numbers (mean, SD, min, max, last, slope, …) | −0.102 | −0.299 | −0.014 |
| glucose curve + the 9 numbers | −0.099 | −0.327 | −0.020 |
| **the child's own history (3 columns)** | **+0.039** | **+0.413** | **+0.037** |
| history + the 9 glucose numbers | +0.022 | +0.406 | +0.033 |
| history + the glucose curve | −0.047 | +0.323 | −0.012 |
| history + curve + numbers | −0.067 | +0.319 | −0.012 |
| glucose curve minus that child's running average | −0.082 | −0.319 | −0.028 |
| time of day only | −0.106 | −0.296 | −0.011 |

**Symbols went from −0.326 to +0.413.** It is the first positive R² anywhere in this project, and
the rank correlation between predicted and actual is **ρ = +0.70** in held-out children.

---

## 3. But the gain is not glucose — and here is the clean test

Putting 3 useful columns and 512 noisy ones into one Ridge with one global alpha dilutes the
useful ones, which is why "history + curve" looks *worse* than history alone. That is a modelling
artefact, not an answer. The honest way to ask "does glucose add anything?" is to **fit history
first, then ask whether glucose explains what history left over** — per fold, inside the CV:

| | Grids | Symbols | Prices |
|---|--:|--:|--:|
| history alone | +0.039 | +0.413 | +0.037 |
| history, then glucose on the residual | +0.028 | +0.402 | +0.027 |
| history + glucose compressed to 8 dimensions first | +0.028 | +0.402 | +0.023 |
| **what glucose contributed** | **−0.011** | **−0.011** | **−0.010** |

**Glucose explains none of what the child's own history leaves unexplained**, on all three scores,
by both routes. The slight negative is the cost of fitting 512 parameters that carry no signal.

---

## 4. What the history model is actually using

Decomposed, so nobody has to guess:

| | Grids | Symbols | Prices |
|---|--:|--:|--:|
| that child's own true mean *(oracle reference)* | +0.116 | +0.307 | +0.070 |
| (a) child's level — running average of earlier sessions | +0.026 | +0.371 | +0.029 |
| (b) practice trend — session number only | −0.066 | −0.141 | −0.001 |
| (c) level + how far the last session deviated from it | +0.033 | **+0.415** | +0.027 |
| within-child: last deviation vs this deviation, *r* | +0.088 | **+0.247** | +0.040 |

So for Symbols: most of the gain is the child's own **level** (+0.371), tracked by the running
average — which beats the static oracle child mean (+0.307) because the running average also
follows drift over the 10 days. On top of that there is a real but small **session-to-session**
component: consecutive deviations correlate at *r* = +0.247, worth about +0.044 more R².
It is **not** a simple practice trend — session number alone predicts nothing (−0.141).

Grids and Prices have almost no within-child tracking (*r* = +0.09, +0.04), which is why their
gains are small.

---

## 5. Grids, asked with the right question

Grids has a hard floor: **31% of sessions score exactly 0** (all six items placed perfectly).
Least-squares on a zero-inflated bounded target is the wrong likelihood, so R² understates what
is findable. Asked as a classification instead — *was this round perfect or not?* — with AUROC
(0.500 = a coin flip), 5 grouped folds, `LogisticRegression`, class-balanced:

| Features | AUROC |
|---|--:|
| the child's own earlier results | **0.579 ± 0.069** |
| the 9 glucose numbers | 0.479 ± 0.037 |
| the glucose curve (Chronos 512-d) | 0.447 ± 0.027 |
| history + curve | 0.507 ± 0.030 |

Glucose is at or slightly below a coin flip even with the right question and the right metric.
**This closes the "you used the wrong likelihood for Grids" objection.**

---

## 6. What this does and does not change

**It does change:**
- We now have a model with a genuinely positive, defensible R² on this data, in these folds. The
  objection *"maybe your pipeline just cannot produce a positive number"* is now closed by
  demonstration, not by argument.
- The statement about glucose is **sharper**, not weaker. Before: "nothing beats guessing the
  average." Now: "we built a model that beats guessing the average by a wide margin on Symbols,
  and glucose contributes −0.01 of it."
- The Grids floor objection is answered (§5).

**It does not change:**
- Glucose still carries no detectable information about these scores.
- The new model **needs at least one earlier session from that child**. It cannot score a
  brand-new child cold, so it is not a screening tool — it is a within-child tracking model.
- The Grids and Prices gains are small (+0.04) and **do not survive leave-one-child-out CV**
  (−0.111 and −0.042). Only Symbols is robust to protocol (+0.141 under leave-one-child-out).

**One expectation that failed.** I switched to leave-one-child-out (20 folds instead of 5)
expecting the fold-to-fold spread to shrink. It got much worse — with one child held out, R² is
measured against that single child's own mean and the between-child offset dominates completely
(baseline: −0.358 / −0.881 / −0.100, SD up to ±1.03). **Leave-one-child-out is the wrong
protocol for this metric.** `GroupKFold(5)` stays.

---

## 7. Reproduce

The four experiment scripts are in the session scratchpad; the feature definitions are in §1 and
§3 above and depend only on `load_real_data()` plus `extract_embeddings()`. Nothing in
`cgm_tsfm/` was modified to produce this file — it is an analysis on top of the existing pipeline.

**Next step if this is pursued:** the incremental-value framing (§3) is the right way to state the
project's question from here on. "Does glucose add predictive value over what we already know
about the child?" is answerable with the data we have; "can glucose predict cognition from
scratch?" is not, at N=20.
