# 11 · What the three scores actually are

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **Why this chapter exists.** You asked Liuyi what Grids, Symbols and Prices measure and in what units. He answered only the direction — *"For all three scores, the lower the better, the higher the worse"* — and linked the ARC app page. He did not answer the units question or whether the scores are age-adjusted.
>
> You cannot model a number well without knowing what it is. So I worked it out from the data itself. One of the three is now pinned down exactly, and the others are strongly constrained. This chapter also surfaces a property of the Grids score that **materially affects the modelling** and is not mentioned anywhere in the repo.

---

## 1. The short version

| | Grids | Symbols | Prices |
|---|---|---|---|
| Sessions with a score | 951 | 920 | 936 |
| Mean ± SD | 0.478 ± 0.500 | 1.846 ± 0.560 | 41.59 ± 17.17 |
| Range | 0.000 – 2.390 | 0.150 – 4.190 | 0 – 90 |
| Distinct values | 146 | 838 | **10** |
| **What it is** | **mean placement error, in grid cells, over 6 items on a 5×5 grid** — *proven below* | a **duration in seconds** (a response time) — strongly indicated | a **percentage in steps of 10**, i.e. a count out of 10 items — certain |
| Lower is better because… | 0 means every item placed exactly right | faster responding | fewer errors |
| Notable | **31.0% of sessions score exactly 0** | smooth, continuous | only 10 possible values |

---

## 2. Grids — solved

### The clue

The distinct Grids values are not arbitrary decimals. Sorted, the smallest ones are:

```
0.00000000
0.16666667   = 1/6
0.23570226   = √2 / 6
0.33333333   = 2/6
0.37267800   = √5 / 6
0.40236893   = ?
0.47140452   = 2√2 / 6
0.50000000   = 3/6
```

Most are `√(integer) / 6`. But not all — `0.40236893` isn't. Multiply it by 6 and square it:

```
0.40236893 × 6 = 2.41421356
2.41421356² = 5.82842712 = 3 + 2√2
```

And `3 + 2√2 = (1 + √2)²`. So `0.40236893 × 6 = 1 + √2`.

That is the tell. **Grids × 6 is a *sum* of square roots of integers**, not a single square root. Checking others:

| Value | × 6 | equals |
|---|--:|---|
| 0.53934466 | 3.23606798 | 1 + √5 |
| 0.56903559 | 3.41421356 | 2 + √2 |
| 0.63807119 | 3.82842712 | 1 + 2√2 |
| 0.69371294 | 4.16227766 | 1 + √10 |

A sum of terms of the form `√(a² + b²)`, divided by 6, is exactly what you get from **averaging six Euclidean distances on an integer grid**.

### The test

I enumerated every value reachable as `(sum of n Euclidean distances on a k×k integer grid) / n` and checked how many of the 94 distinct Cohort 1 Grids values it explains:

| Grid | Items | Reachable values | Explained | % |
|---|--:|--:|--:|--:|
| 4×4 | 4 | 518 | 6 | 6.4% |
| 4×4 | 5 | 1,260 | 2 | 2.1% |
| 4×4 | **6** | 2,730 | 87 | 92.6% |
| 4×4 | 7 | 5,412 | 2 | 2.1% |
| 5×5 | 4 | 1,830 | 6 | 6.4% |
| 5×5 | 5 | 5,470 | 2 | 2.1% |
| **5×5** | **6** | 14,178 | **94** | **100.0%** |
| 5×5 | 7 | 32,904 | 2 | 2.1% |
| 6×6 | 6 | 85,293 | 94 | 100.0% |

**All 94 distinct values are explained exactly by 6 items on a 5×5 grid**, with zero unexplained. The item count is pinned down hard — 4, 5 and 7 items explain only 2–6%. A 4×4 grid gets to 92.6% but misses 7 values, so the grid must be at least 5×5; 6×6 also works but 5×5 is the smallest that does, and it is the standard ARC configuration.

### The conclusion

> **`grids_cognitive_score` = the average straight-line distance, measured in grid cells, between where the child placed each of 6 objects and where it actually belonged, on a 5×5 grid.**

- **0.000** = all six placed perfectly.
- **Units: grid cells.** So a score of 0.478 means "off by about half a cell per item on average."
- Theoretical maximum on a 5×5 grid is `√32 = 5.657` (everything to the opposite corner). Our observed maximum is **2.390**, so nobody is close to random.
- Lower is better, exactly as Liuyi said — and now you know *why*.

This is the ARC **Grid Memory** task: objects are shown in positions, then the child places them from memory, and the score is the mean placement error.

**Say it in your own words:** *"Grids is a memory task. The child sees six objects on a 5-by-5 grid, then has to put them back where they were. The score is the average distance, in grid squares, between where they put each one and where it really was. Zero means perfect. I confirmed this from the data — every distinct value in the file is exactly the mean of six grid distances, and no other item count fits."*

---

## 3. The property that matters for modelling: 31% of sessions are exactly zero

This is the part with real consequences, and it is not mentioned anywhere in the repo.

```
Grids score distribution (951 sessions)

 0.000  ████████████████████████████████████████████████  295  (31.0%)
 0.167  ██████████████                                     86
 0.333  ███████████                                        68
 0.500  ████████                                    (long tail continues)
   ...
 2.390  ▏                                                    1
```

**295 of 951 sessions (31.0%) have a Grids score of exactly 0.000** — perfect recall of all six positions. The three most common values account for 449 sessions, nearly half the data.

### Why this is a problem for what we're doing

1. **A hard floor.** The score cannot go below 0, and almost a third of the data sits on that floor. If a child already scores 0, no amount of good glucose can improve their score. Any real effect on those sessions is invisible — it's censored, in the same way our glucose readings are censored at 400 mg/dL.
2. **Ridge and MSE assume otherwise.** Least-squares regression assumes the target is a smooth continuous quantity with roughly symmetric errors around the prediction. A spike of 31% at a boundary plus a long right tail (skewness **+1.06**) is not that shape. The model is being asked to fit something it is structurally unsuited to.
3. **It caps the achievable R².** A large chunk of the variance is the discrete difference between "perfect" and "not perfect," which is closer to a yes/no outcome than a continuous measurement.
4. **The direction of the fix.** The statistically appropriate treatment would be either a two-part model (first predict perfect vs not, then predict the magnitude of the error given not-perfect), or a model for bounded/zero-heavy outcomes. That is worth mentioning as a limitation — though note it is Phil's lane, not yours: Liuyi assigned mixed-effects and association analysis to Phil and told you to stay on prediction.

**Be careful how you raise this.** Don't present it as an excuse for the low accuracy — the accuracy is low for all three scores, including Prices, which has no floor problem. Present it as a limitation you noticed.

**Say it in your own words:** *"One thing I found looking at the scores: 31% of Grids sessions are exactly zero, meaning the child got all six positions perfect. That's a hard floor. Ridge regression assumes a smooth continuous target, and a third of the data piled up at one boundary isn't that. It doesn't explain our result — Prices has no floor and is also flat — but it's a real limitation of treating Grids as an ordinary continuous number."*

---

## 4. Prices — a count out of 10

**Certain, and easy.** The Prices score takes exactly **10 distinct values: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90** — in both cohorts independently. It is a percentage in steps of 10, which means **10 items** with one item worth 10 percentage points.

Distribution (Cohort 1, n = 738):

| Value | Count | % |
|--:|--:|--:|
| 0 | 7 | 0.9% |
| 10 | 41 | 5.6% |
| 20 | 99 | 13.4% |
| 30 | 139 | 18.8% |
| **40** | **157** | **21.3%** |
| 50 | 148 | 20.1% |
| 60 | 84 | 11.4% |
| 70 | 50 | 6.8% |
| 80 | 12 | 1.6% |
| 90 | 1 | 0.1% |

Nicely bell-shaped, centred on 40–50, essentially no floor or ceiling pile-up (0.9% and 0.1%). Since lower is better, this reads as a **percentage of items answered incorrectly** — the typical child gets about 4 of 10 wrong.

> **An honest caveat.** Interpreting it as *incorrect* follows from Liuyi's "lower is better" plus the 10-step granularity. I have not seen documentation confirming it, and the alternative — that it is percent correct and the direction statement was about a different quantity — cannot be ruled out from the data alone. **This is worth one clarifying question**, and it is a good question to ask because it is specific and answerable: *"Is the Prices score the percentage of items answered incorrectly?"*

**Note the modelling consequence:** with only 10 possible values, a single-item mistake moves the score by 10 points, and 10² = 100 squared error against a total variance of 295. So squared-error regression on Prices is coarse by construction. This is likely why Prices always shows the R² closest to zero (−0.012) — there is less fine-grained variance to get wrong.

**Say it in your own words:** *"Prices only ever takes 10 values — 0, 10, up to 90 — so it's a count out of 10 items expressed as a percentage. Since lower is better it's presumably the percentage wrong, and a typical session is around 40. I'd like that confirmed."*

---

## 5. Symbols — a response time in seconds

**Strongly indicated, not proven.**

- 729 distinct values out of 729 Cohort 1 sessions — **completely continuous**, no repeats, no discrete grid.
- Range **0.150 to 4.190**, mean **1.846**, SD 0.560, mild right skew (+0.50).
- Lower is better.

A continuous positive quantity around 1.8 with a mild right skew, where smaller is better, is the classic signature of a **response time in seconds**. The ARC **Symbols** task is a symbol-matching test of processing speed, and its standard score is a median or mean response time. A median response time of about 1.8 seconds for symbol matching is exactly the right order of magnitude.

The right skew is also characteristic — reaction-time distributions have a floor (you cannot respond faster than perception allows) and a long slow tail.

Unlike Grids, there is no arithmetic fingerprint to exploit here: an average of continuous times leaves no algebraic trace. So this stays an inference from the units, range, shape and the known task.

**Say it in your own words:** *"Symbols is a processing-speed task — match a symbol as fast as you can. The score is continuous, ranges from 0.15 to 4.19 with a mean of 1.85, and lower is better, so it's almost certainly a response time in seconds. I can't prove it from the numbers the way I could for Grids, but the range and shape fit nothing else."*

---

## 6. Cohort differences — know these before someone else finds them

Children in Cohort 2 score **worse on all three tests** (remembering lower = better):

| Score | Cohort 1 (14 children) | Cohort 2 (6 children) | Difference |
|---|--:|--:|--:|
| Grids | 0.447 | 0.580 | +30% worse |
| Symbols | 1.831 | 1.919 | +5% worse |
| Prices | 40.42 | 45.41 | +12% worse |

With 6 children in Cohort 2 this could easily be chance, or an age difference, or a protocol difference. **Don't over-interpret it** — but be able to say the numbers if asked, and note that our grouped cross-validation means a test fold can be cohort-imbalanced, which adds to the fold-to-fold spread.

There is also a **precision difference**: Cohort 1 stores full precision (`0.33333333`) while Cohort 2 is partly rounded to two decimals (`0.33`). Harmless for the modelling, but it shows the two files were produced by different processing, which is worth knowing when someone asks how comparable they are.

---

## 7. A correction to our own documents: within vs between participants

Several documents in this repo state that between-participant differences **dominate** the score variance, and that this is why grouped cross-validation is hard. **I measured it, and that framing is wrong.**

Share of each score's total variance attributable to differences *between* participants:

| Score | Between participants | **Within participant** |
|---|--:|--:|
| Grids | 0.153 | **0.847** |
| Symbols | 0.376 | **0.624** |
| Prices | 0.077 | **0.923** |

**62% to 92% of the variation is within a participant**, session to session. Who the child is explains only a modest share — most of the spread is one child varying across their own sessions.

### Why this correction matters

It changes the story about *why* the accuracy is low.

- **The old story:** "the signal is drowned out by big differences between children; remove each child's baseline and it might emerge."
- **What the data says:** there isn't much of a between-child effect to remove. Most of the variance is already session-to-session — which is exactly the variance a glucose→cognition effect would live in. And we still cannot predict it.

That actually makes the result *stronger*, not weaker. It is also consistent with what removing each child's own average did: R² moved from −0.089/−0.056/−0.012 to −0.018/−0.022/−0.002, i.e. towards zero but never above it. Nothing was hiding underneath.

Session-to-session variation in a short cognitive test is mostly ordinary noise — attention, mood, distraction, whether they were on a bus. Glucose is competing with all of that.

**Say it in your own words:** *"I checked something our notes had backwards. We'd written that differences between children dominate the score variance. Actually only 8% to 38% is between children — most of it, 62% to 92%, is one child varying from session to session. That matters, because session-to-session is exactly where a glucose effect would show up, and we still can't predict it. So the result is a bit stronger than we'd framed it, not weaker."*

---

## 8. Drill

<details>
<summary><b>What does a Grids score of 0.478 mean physically?</b></summary>

The child placed six objects on a 5×5 grid from memory, and on average each object was about 0.48 grid cells away from its true position. Since 31% of sessions score exactly 0, a score of 0.478 is a bit worse than the median session (median 0.333).
</details>

<details>
<summary><b>How do you know Grids uses 6 items and not 5 or 7?</b></summary>

I enumerated every value reachable as the mean of n Euclidean distances on a k×k integer grid and matched against the 94 distinct Cohort 1 values. With 6 items on a 5×5 grid, 100% match exactly. With 4, 5 or 7 items, only 2–6% match. The item count is uniquely determined.
</details>

<details>
<summary><b>Why does the Prices score always give the R² closest to zero?</b></summary>

Two reasons. It has only 10 possible values, so a one-item error moves it by 10 points against a total variance of 295 — the target is coarse, and there is less fine structure available to predict or mispredict. And it has the lowest between-participant share (0.077), so there is very little participant-level structure for a model to latch onto and then fail to transfer. The result is a model that stays close to just predicting the mean, which is R² ≈ 0.
</details>

<details>
<summary><b>Does the 31% floor in Grids explain the low accuracy?</b></summary>

No, and don't claim it does. It is a genuine limitation of treating Grids as a smooth continuous target, and it caps how much variance is available. But Prices has no floor problem — a clean bell shape from 0 to 90 — and its accuracy is equally flat. So the floor is a limitation to disclose, not the cause.
</details>

<details>
<summary><b>All three scores are "lower = better". Does that affect any of your reported numbers?</b></summary>

No. R², RMSE and MAE are all unchanged if you negate the target — negating y flips both the residuals and the deviations from the mean, and both get squared. That is why `config.PRICES_IS_INVERTED = False` and we keep the raw orientation. The direction only affects *interpretation*: a model predicting a lower score is predicting better performance.
</details>

<details>
<summary><b>Why do you keep saying "sessions" and not "patients"?</b></summary>

Because they are different units and mixing them up overstates the evidence. We have 956 sessions but only 20 participants, and sessions from one child are not independent of each other. Liuyi corrected this exact slip in the subgroup table — the 201 hypoglycemia rows are *sessions*, drawn from 19 *participants*.
</details>

---

## 9. Open questions worth asking Liuyi

Short, specific, answerable — the kind of question that shows you've done the work:

1. **"Is the Prices score the percentage of items answered incorrectly?"** (I can see it's a count out of 10; I want the direction confirmed.)
2. **"Is the Symbols score a response time in seconds?"**
3. **"Are these raw scores, or adjusted for age or practice effects?"** — asked before, not answered, and it matters: children were tested ~5×/day for 10 days, so practice effects are plausible, and an unadjusted practice trend would be a session-order effect competing with glucose.
4. **"Confirming Grids is mean placement error over 6 items on a 5×5 grid?"** — you've proven it from the data; confirmation just closes it.

---

## 10. Summary

1. Grids is **proven**: mean placement error in grid cells, 6 items, 5×5 grid, 0 = perfect. 100% of distinct values explained exactly; 6 items uniquely determined.
2. **31.0% of Grids sessions score exactly 0** — a hard floor that violates the smooth-continuous assumption behind least-squares regression. A real limitation, not the cause of the result.
3. Prices is **certain** to be a count out of 10 rendered as a percentage (0–90 in steps of 10); the "percent incorrect" reading follows from lower-is-better and is worth confirming.
4. Symbols is **almost certainly a response time in seconds** (continuous, 0.15–4.19, mean 1.85, right-skewed) — inferred from shape and task, not proven algebraically.
5. Cohort 2 children score worse on all three, and their scores are partly rounded to two decimals. Know the numbers; don't over-read them with n = 6.
6. **Our documents had within- vs between-participant variance backwards.** 62–92% of the variance is within participant. This strengthens the finding rather than weakening it.

→ Next: `10_RIGOR_AND_STATS.md` for why the low accuracy is credible, or `15_FINDINGS_TO_REPORT.md` for the pipeline bug found this week.
