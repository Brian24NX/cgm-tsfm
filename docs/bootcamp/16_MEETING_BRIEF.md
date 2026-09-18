# 16 · Meeting brief — 2026-09-18 (the handover)

> One page. Everything else in this folder is reference; this is what you actually hold.
> Audience: Liuyi (leaving) + **Ben** (taking over, knows nothing about this project).
> Deck: `Handover_CGM_Cognition.pptx`, 16 slides, speaker notes on every one.

---

## 1. The shape of it — four beats

| | Beat | Minutes | The one sentence |
|---|---|--:|---|
| 1 | **Your assignment is closed** | 3 | "Every session now gets exactly the same 2 hours of glucose. 916 of 956 sessions, all 20 participants." |
| 2 | **The number that moved** | 4 | *volunteer it before anyone finds it — see §3* |
| 3 | **The bug I found in my own code** | 3 | "I found it, fixed it, wrote a test for it, re-ran everything. The conclusion didn't change." |
| 4 | **Handover to Ben** | rest | walk the deck, hand him `docs/HANDOVER.md` |

Close by putting the two open decisions (§5) to Liuyi **while he is still here.**

---

## 2. The three numbers to say out loud

- **916 of 956 sessions (95.8%), all 20 participants** — the cost of uniform windows.
- **−0.135 / −0.290 / −0.015** — Arm A, grids / symbols / prices.
- **−0.071 / −0.253 / −0.011** — *the mean-predictor's own score under the same folds.*
  Always say this second number immediately after the first. Without it the result sounds broken;
  with it, the result is "we tie the trivial baseline."

**Why the baseline is negative at all:** `r2_score` measures each test fold against *that fold's
own* mean, but the model was trained on the *training* fold's mean. With only 4 participants held
out, those differ. So under leave-participants-out CV, slightly negative **is** the neutral point.

---

## 3. Prepared answer — "so the numbers got *worse* after your change?"

**This is the most likely hard question. Have this ready verbatim.**

Symbols went from −0.052 to −0.290. That looks like a collapse. It isn't — **the baseline moved
with it:**

| Symbols | model | mean-predictor | **gap** |
|---|--:|--:|--:|
| Variable window (956 sessions) | −0.052 | −0.050 | **−0.002** |
| Uniform 2 h (916 sessions) | −0.290 | −0.253 | **−0.036** |
| | *moved 0.238* | *moved 0.203* | ***moved 0.034*** |

> *"The raw R² dropped by 0.24, but the mean-predictor dropped by 0.20 at the same time, because
> R² is measured against each test fold's own average and the session set changed. Measured as the
> gap to that configuration's own baseline — which is the honest comparison — the change is 0.03,
> not 0.24. That's why the window write-up reports gaps."*

Source: `results/uniform_window_real.md` §3, rows A and C (−0.002 and −0.036, computed in one run).

**And if pushed on whether the change was worth making anyway — yes, for a reason that isn't
accuracy:** the old window was "everything since that child's previous test" (confirmed, r = 0.977
against the timestamps), so its *length* carried time of day (r = −0.506). A model could lean on
that. Now it can't.

**Second prepared answer — "your model predicts glucose variability at 0.416 but the scores at
−0.29. Isn't that contradictory?"** No — that's the point of the check. Same 512 features, same
folds, same model; only the question changes. It says the features genuinely carry information
about the glucose curve, so the flat cognitive result is about the *relationship*, not the
plumbing. Be honest that it's **partial**: absolute glucose level is recovered poorly (−0.114),
because Chronos normalises each series before the encoder sees it and we throw away the `loc`/`scale`.

---

## 4. Words to not use

He has corrected these. From `00_SAY_IT_SIMPLY.md` §2:

| Don't say | Say |
|---|---|
| null result | low prediction accuracy |
| positive control | known-answer check |
| permutation test | shuffle test |
| power / sensitivity | *(just give the fold-to-fold spread)* |
| confound | *(describe the specific thing: "window length also tells you the time of day")* |
| patients | **participants** (or **sessions** — never blur the two) |

---

## 5. Ask Liuyi before he goes — only he can answer these

1. **Is a rigorous "low prediction accuracy + characterisation" the deliverable, or do we keep
   hunting for signal?** The cheap options are exhausted. Going further means changing the
   problem, not the model.
2. **Is N=20 the sample for this phase, or is more coming toward the K01's 92?** Decides whether
   this is interim or a stopping point.
3. **Does the 40-min presentation / journal draft deliverable still stand under new leadership?**

Both #1 and #2 were raised months ago (`docs/04_OPEN_QUESTIONS.md` A1, A2) and never answered.

---

## 6. If someone opens the code

- **Gradient boosting was never tried.** `regression.py` wraps the xgboost import in `try/except`
  and xgboost isn't installed, so it silently skipped every run. Say "no", don't get caught.
- **`RANDOM_STATE = 42` does not control the outer folds.** `GroupKFold` has no shuffle, so the
  outer split is deterministic regardless of seed.
- **Arm B is the worst arm and that's expected** — 132,609 parameters on ~612 training rows per
  fold is 217 parameters per example.
- **Every number in this project has a derivation** in `01_FACT_SHEET.md`. If asked where a figure
  comes from and you don't remember, say you'll pull the derivation rather than guessing.
