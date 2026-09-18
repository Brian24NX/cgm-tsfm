# Same-length glucose windows — the result

**Date:** 2026-08 · **Data:** real, 20 participants · **Model:** chronos-bolt-small (frozen), Ridge/SVR, 5-fold split by participant.

This answers the first item on the "what I would do next" slide:
*"Give every test the same amount of glucose data — right now some have 15 minutes of readings before them and others have 24 hours."*

---

## 1. What we thought was blocking us — and wasn't

Earlier notes said this needed the original glucose recording from Phil. **That was wrong.** Every session's stored list already *ends* at the test, so we can cut a same-length window from what we have. We only have to **drop sessions that are too short** rather than trim long ones.

The earlier "2 h window" row in `sweep_real.md` did **not** do this. `max_readings` only **capped** long sessions; short ones were left alone, so lengths still ran 3–24 readings. It compared *"everything"* vs *"at most 2 h"*, never *"exactly 2 h"*. Fixed by `WindowConfig(require_full=True)`.

## 2. What a same-length window costs

| Window | Readings | Sessions kept | % | Participants |
|---|--:|--:|--:|--:|
| 1.0 h | 12 | 942 | 98.5% | 20 |
| 1.5 h | 18 | 931 | 97.4% | 20 |
| **2.0 h** | **24** | **916** | **95.8%** | **20** |
| 2.5 h | 30 | 643 | 67.3% | 20 |
| 3.0 h | 36 | 490 | 51.3% | 20 |

**2 hours is the sweet spot** — 4% of sessions lost, all 20 participants kept. Beyond 2 h the cost rises steeply.

**And it is genuinely uniform in time, not just in count:** only **1 session out of 956** has a missing reading inside its list, so 24 readings = 2 hours almost everywhere.

## 3. The result — it does not change the answer

R² against each configuration's **own** "always guess the average" predictor, in the same folds. The gap is the honest measure, because the baseline itself shifts when the session set changes.

| Configuration | Sessions | grids | symbols | prices |
|---|--:|--:|--:|--:|
| A. everything, variable length *(current)* | 956 | −0.036 | −0.002 | −0.007 |
| **B. same 916 sessions, still variable** | 916 | −0.062 | +0.005 | −0.006 |
| C. uniform 1.0 h | 942 | −0.023 | −0.029 | −0.004 |
| C. uniform 1.5 h | 931 | −0.049 | −0.029 | −0.009 |
| **C. uniform 2.0 h** | **916** | **−0.064** | **−0.036** | **−0.004** |
| D. uniform 2 h + how high/low it was | 916 | −0.064 | −0.036 | −0.004 |
| E. just those 2 level numbers, no AI model | 916 | −0.016 | −0.003 | −0.002 |

*(gap = model R² − baseline R²; negative means the model did worse than guessing)*

**Row B is the control and the important one.** It is the *same 916 sessions* left at variable length. B and C are the same to within noise, so **making the windows uniform changed nothing** — the difference between A and C is the 40 dropped sessions, not the uniformity.

**Row D** answers the second item on that slide: adding back how high or low the glucose was makes no difference either (−0.064 vs −0.064).

### Why the *headline* R² looks much worse even though the gap barely moved

The headline numbers in `headtohead_real.md` are raw R², not gaps, and raw R² moved a lot. Symbols
is the extreme case:

| Symbols | model | its own mean-predictor | **gap** |
|---|--:|--:|--:|
| variable window, 956 sessions | −0.052 | −0.050 | **−0.002** |
| uniform 2 h, 916 sessions | −0.290 | −0.253 | **−0.036** |
| | *moved 0.238* | *moved 0.203* | ***moved 0.034*** |

**The baseline moved almost as far as the model did.** `r2_score` measures each test fold against
*that fold's own* mean while the model was trained on the training fold's mean; with 4
participants held out those differ, and how much they differ depends on which sessions are in the
set. So changing the session set moves the baseline. Read the gap, not the raw value — a raw R²
from one configuration is not comparable to a raw R² from another.

## 4. Two things we can now state as fact

**The window is the stretch since the child's previous test.** Confirmed against the timestamp columns: the time since the previous test and the span of the stored window correlate at **r = +0.977**, and agree to within 10 minutes in **87%** of sessions.

**Long window = morning test.** Time of day and window length correlate at **r = −0.506**. So the old variable window really did carry time-of-day information — which is exactly what the uniform window removes. Worth doing for that reason alone, even though accuracy did not improve.

**Time of day is not a hidden predictor either:** its correlation with the scores is +0.011 (grids), −0.091 (symbols), −0.005 (prices) — about the same as the best glucose correlation anywhere in the data (−0.092).

## 5. Reproduce

```python
from cgm_tsfm.config import WindowConfig
from cgm_tsfm.data import load_real_data
ds = load_real_data(window=WindowConfig(max_readings=24, require_full=True))
# -> 916 sessions, 20 participants, every window exactly 24 readings
```

---

## 6. The full window sweep Liuyi asked for — 15 min to 3 h

Gap = model R² − its own "guess the average" R², same folds. Negative = worse than guessing.

**Sweep A — each window keeps as many sessions as it can (what you would actually use):**

| Window | Readings | Sessions | Children | grids | symbols | prices |
|---|--:|--:|--:|--:|--:|--:|
| 15 min | 3 | 956 | 20 | −0.015 | −0.006 | −0.004 |
| 30 min | 6 | 954 | 20 | −0.011 | −0.021 | −0.005 |
| 45 min | 9 | 950 | 20 | −0.027 | −0.032 | −0.002 |
| 60 min | 12 | 942 | 20 | −0.023 | −0.029 | −0.004 |
| 90 min | 18 | 931 | 20 | −0.049 | −0.029 | −0.009 |
| 120 min | 24 | 916 | 20 | −0.064 | −0.036 | −0.004 |
| 150 min | 30 | 643 | 20 | −0.053 | −0.031 | −0.003 |
| 180 min | 36 | 490 | 20 | −0.028 | −0.034 | −0.009 |

**Sweep B — controlled: the same 490 sessions in every row, only the span changes:**

| Window | grids | symbols | prices |
|---|--:|--:|--:|
| 15 min | −0.004 | −0.016 | **+0.003** |
| 30 min | −0.014 | **+0.005** | −0.002 |
| 45 min | −0.023 | −0.026 | −0.005 |
| 60 min | **+0.000** | −0.037 | −0.006 |
| 90 min | −0.037 | −0.066 | −0.019 |
| 120 min | −0.036 | −0.048 | −0.019 |
| 150 min | −0.043 | −0.019 | −0.014 |
| 180 min | −0.028 | −0.034 | −0.009 |

## 7. ⚠️ Do NOT read "shorter is better" off those tables

Short windows *look* better, and three cells are even slightly positive. Both are artifacts. Two checks show why.

**Check 1 — the model is giving up, not getting smarter.** "Spread ratio" is how much the model's predictions vary as a fraction of how much the real scores vary. 1.00 = confident guesses; 0.00 = it is just saying the average every time.

| Window | Fold-to-fold SD of R² | Spread ratio |
|---|--:|--:|
| 15 min | 0.439 | **0.10** |
| 30 min | 0.436 | 0.25 |
| 60 min | 0.420 | 0.28 |
| 120 min | 0.410 | 0.31 |
| 180 min | 0.342 | 0.30 |

At 15 minutes the model's predictions are almost flat — it has 3 readings to work with, so it falls back on the average. A model that predicts the average necessarily *ties* the average. The gap goes to zero because the model stopped trying, not because it found something.

**Check 2 — the differences are far smaller than the noise.** The fold-to-fold spread is **±0.34 to ±0.44**. The differences between windows are 0.005 to 0.06 — roughly **80× smaller than the run-to-run variation**. They cannot be told apart.

We also compared 8 windows × 3 scores × 2 sweeps = **48 cells**. With that many looks, a few small positives are expected by luck alone.

**Conclusion: the data cannot tell us which window is best. Every window gives the same answer — no better than guessing.**

## 8. So which window should we use?

**Recommendation: 2 hours (24 readings), chosen on physiological grounds and stated in advance — not chosen by picking the best number.**

1. The K01 hypothesis is about roughly the 2 hours before a test, so it is the pre-committed choice.
2. It keeps 95.8% of sessions and all 20 participants.
3. It is uniform in real time, not just in count (only 1 session of 956 has an internal gap).
4. Picking the best-scoring window out of 48 cells is exactly how you fool yourself. Choosing on physiology and reporting the rest is the defensible move.

*(15 minutes is only 3 readings — too little for a time-series model to describe a curve at all, so it is not a meaningful test of the approach even though it scores "best".)*

## 9. ✅ Switched over (2026-09)

The pipeline default is now **uniform 2 h**: `WindowConfig(max_readings=24, require_full=True)`, set via `config.DEFAULT_MAX_READINGS` / `DEFAULT_REQUIRE_FULL`. Every runner picks it up; `--variable-window` reproduces the old behaviour. All result files regenerated; pre-change copies in `archive_variablewindow_2026-08/`.

**Two real costs of the change, worth stating:**

1. **Fewer excursions to look at.** A 2 h window contains far fewer low/high episodes than a 24 h one, so the glucose-regime subgroups shrank sharply: sessions with a reading below 70 went from **201 → 111**, and above 250 from **449 → 314**. That is a genuine loss of the cases where an effect is most plausible.
2. **The known-answer check got weaker on level.** From the same 2 h embeddings, glucose variability is still recovered (R² = **0.416**), but mean glucose is now **−0.114** and % time above 180 is **−0.079** (they were 0.042 and 0.068 on full windows). With only 2 h the embedding carries shape but essentially nothing about absolute level. The check still passes on the part that matters — the machinery demonstrably extracts real information — but it is weaker than before.
