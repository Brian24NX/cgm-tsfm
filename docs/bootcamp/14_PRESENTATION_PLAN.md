# 14 · The presentation — slide by slide

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **The brief, from Liuyi's own words.** Length: *"Short term (e.g., 2-3 weeks), a 40-min presentation justifying what we have done, the results we obtained is convincible and rigorously verified, no matter what the results are."* Emphasis, when asked directly whether to weight method or results: **"Result + rigor."**
>
> So: not a methods tour. The centre of gravity is *what we found* and *why you should believe it*. Method appears in service of credibility, and you must be able to go deeper on any of it on demand — because he separately demands to understand the pipeline and asked twice about input shapes.

---

## 1. The design rules

**Rule 1 — one idea per slide, and the idea is the title.** Write titles as sentences, not labels. "Results" tells him nothing. "Nothing predicted better than guessing the average" tells him everything, and the slide just supports it.

**Rule 2 — every technical word gets defined on the slide where it first appears.** Not in the appendix, not verbally. On the slide, in a small line under the term. He explicitly asked to stop introducing new vocabulary.

**Rule 3 — no banned words.** No "null," no "signal beyond chance," no "positive control," no "power," no "sensitivity," no "pre-test window," no "lever." See `00_SAY_IT_SIMPLY.md` §2 for the replacements. Go through your finished deck with that table open and grep for each one.

**Rule 4 — numbers over adjectives.** Not "the results were poor." Instead: "R² = −0.089, where 0 means as good as guessing the average."

**Rule 5 — say "sessions" or "participants," never "patients," and never blur them.** He corrected this exact slip.

**Rule 6 — build one diagram and reuse it.** Draw the pipeline once, then re-show that same picture with the relevant box highlighted whenever you talk about a stage. Repetition of one visual beats five new ones.

**Rule 7 — put the honest limitation on the same slide as the result.** Don't quarantine caveats at the end. A caveat next to its claim reads as rigor; a caveat at the end reads as an afterthought.

**Rule 8 — plain fonts, big type, no animation.** 24pt minimum body text. If it needs to be smaller, it belongs in the appendix.

---

## 2. The shape of 40 minutes

| Part | Slides | Minutes | Purpose |
|---|--:|--:|---|
| 0. Opening | 1–2 | 2 | The question and the answer, immediately |
| 1. What we're predicting | 3–7 | 6 | The data, honestly, including its problems |
| 2. How the pipeline works | 8–13 | 8 | Enough method to make the result meaningful |
| 3. The result | 14–17 | 6 | The headline, four ways |
| 4. **Why you should believe it** | 18–23 | **10** | The heart of the talk |
| 5. What I found wrong | 24–26 | 5 | Your corrections — the credibility anchor |
| 6. Where this leaves us | 27–28 | 3 | Options, not a verdict |
| — Appendix | A1–A12 | — | For questions |

**Note the shape:** Part 4 is the longest single block, and Part 2 is deliberately shorter than you'd instinctively make it. That's what "result + rigor" means. Resist the urge to spend fifteen minutes on Chronos.

---

## 3. Slide by slide

### Part 0 — Opening (2 min)

**Slide 1 — Title**
> **Can we predict a child's thinking-test score from their glucose beforehand?**
> Brian Zhou · [date]
> 20 participants · 956 test sessions · Type 1 Diabetes, ages 9–16

**Slide 2 — The answer, up front**
> **Short answer: no — not better than guessing the average score.**
>
> - True for all three tests
> - True for five model sizes, every window length, and every setting we tried
> - True for the older hand-built-features approach too
> - **And I've checked it three ways to make sure that's the data and not a mistake**
>
> *Small line at the bottom:* "Guessing the average" = a comparison predictor that ignores glucose entirely and always predicts the average score. Explained on slide 15.

*Why lead with the answer:* he asked for result + rigor. Making him wait 25 minutes for the finding, and then hearing it's flat, feels like a reveal of failure. Stating it in minute two turns the rest of the talk into *evidence*, which is what he actually asked for.

---

### Part 1 — What we're predicting (6 min)

**Slide 3 — The study, in one picture**
```
   A child wears a glucose sensor            A phone app gives 3 short
   for 10 days. It records a reading         thinking tests, about 5 times
   every 5 minutes.                          a day.

   ──────────────────────────────▶  time
      glucose readings before the test   │  TEST
                                          └─▶ 3 scores

   One row of our data = one test session.
   20 children · 956 sessions · 68,206 glucose readings
```

**Slide 4 — The three tests, and what they measure**

| Test | What the child does | The score is | Lower is better because |
|---|---|---|---|
| **Grids** | Remember where 6 objects sat on a 5×5 grid | Average distance, in grid squares, from the right spot | 0 = all six perfect |
| **Symbols** | Match symbols as fast as possible | A response time, in seconds | Faster |
| **Prices** | Judge item prices | A percentage, in steps of 10 | Fewer errors |

> Mean scores: Grids 0.478 · Symbols 1.846 · Prices 41.6

*Speaking note:* mention that you worked Grids out from the data — every distinct value is exactly the mean of six grid distances, which pins down both the item count and the grid size. It's a small thing that signals you looked at your data rather than accepting a column name.

**Slide 5 — The glucose, honestly**
> 68,206 readings · mean **177 mg/dL** · range **40–400**
>
> | | share of readings |
> |---|--:|
> | below 70 (low) | **2.3%** |
> | 70–180 (in range) | 58.1% |
> | above 180 (high) | 39.5% |
>
> **Two things worth flagging:**
> - The 40 and 400 are the **sensor's reporting limits**, not real values. 2.2% of readings sit at exactly 400, meaning "400 or above."
> - **Low readings are rare (2.3%)** — and low glucose is where a thinking effect is most expected.

**Slide 6 — The problem with our input**
> **Every session has a different amount of glucose data.**
>
> ```
>   shortest session:    3 readings   (15 minutes)
>   median  session:    36 readings   (3 hours)
>   longest session:   288 readings   (24 hours)
> ```
>
> It looks like the stored list holds everything since that child's previous test. So an overnight gap gives ~24 hours; two tests an hour apart give ~12 readings.
>
> **Why that's a problem:** the length itself tells you about time of day, not glucose. The strongest single relationship I found with the Grids score isn't a glucose statistic at all — it's the **number of readings** (r = −0.077).
>
> *This is the one thing I'd most want to fix.*

*Speaking note:* this is where he previously said *"This is a very good point, actually. I did not consider this point before."* Give it room. Do **not** call it a confound — see `00_SAY_IT_SIMPLY.md` §4.

**Slide 7 — How much relationship is there before any model?**
> Correlation between a simple glucose number and a test score, across all 956 sessions:
>
> | | strongest correlation | explains |
> |---|--:|--:|
> | Grids | −0.077 | 0.6% of the variation |
> | Symbols | **−0.092** | **0.85%** |
> | Prices | +0.056 | 0.3% |
>
> **The strongest relationship anywhere in this dataset explains under 1% of the variation.**
>
> No model can create a pattern that isn't there. This slide is why the later result is unsurprising rather than suspicious.

---

### Part 2 — How the pipeline works (8 min)

**Slide 8 — The pipeline (THE diagram — reuse this)**
```
  ┌──────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐
  │ 1. DATA  │──▶│ 2. DESCRIBE   │──▶│ 3. 512       │──▶│ 4. PREDICT│──▶│ 5. SCORE   │
  │ glucose  │   │ the curve     │   │ numbers per  │   │ the test  │   │ it fairly  │
  │ + score  │   │ (Chronos)     │   │ session      │   │ score     │   │            │
  └──────────┘   └───────────────┘   └──────────────┘   └──────────┘   └────────────┘
     956 rows      a pretrained          (956, 512)       Ridge / SVR    hold out whole
                   time-series model                      or a small     children
                   we do not modify                       network
```

**Slide 9 — What Chronos is, and what it isn't**
> **Chronos** is a large model Amazon trained to *forecast* time series, using far more data than we have.
>
> - We **don't use its forecasts.** We run a glucose curve through it and take the internal description it builds.
> - We **never change it.** Its 47.7 million numbers stay fixed. Only a small piece on top learns anything.
> - **Why use it at all:** with 20 children we cannot train a model to understand time series from scratch. We borrow one that already does.
> - **The honest catch:** its description was built for predicting the next glucose values, not for predicting a test score. Nothing guarantees it keeps what we need.

**Slide 10 — The shape walkthrough** *(he asked for this twice — make it a full slide)*
```
  ONE SESSION: 45 glucose readings

  [142, 145, 139, ... , 151]        45 numbers. That is the entire input.
        │                            No timestamps. No second measurement.
        │                            No participant id.
        ▼
   standardize, then cut into blocks of 16 readings
        │                            one block = 80 minutes of glucose
        ▼
   3 blocks  +  1 summary slot  =  4 "tokens"
        │
        ▼
   each token → 512 numbers        →   a 4 × 512 block
        │
        │  average down the token direction
        ▼
   512 numbers for this session

   All 956 sessions stacked:   a 956 × 512 table   →   this is what we predict from
```
> **block of 16 readings = "patch"** · **a patch turned into 512 numbers = "token"**

**Slide 11 — Something important Chronos does to our data**
> **Before Chronos looks at a glucose curve, it subtracts that curve's own average and divides by its own spread.**
>
> ```
>   glucose sitting at 250, swinging ±30   ─┐
>                                            ├─▶  look nearly IDENTICAL to Chronos
>   glucose sitting at 100, swinging ±30   ─┘
> ```
>
> **So our description captures the *shape*, not how high or low the glucose actually was.**
>
> That matters here, because "low" and "high" are defined by absolute numbers — below 70, above 180.
>
> Chronos actually hands those two removed numbers back, and our code was discarding them. I've tested adding them in — more on slide 25.

*Speaking note:* this slide does double duty. It's a genuine scientific limitation, and it pre-empts the sharp question "if your description is any good, why can't it recover the average glucose?" — which comes up on slide 21.

**Slide 12 — Two ways to turn 512 numbers into a score**

| | **Arm A** | **Arm B** |
|---|---|---|
| What | Classical regression (Ridge, SVR) | A small neural network on top |
| Adjustable numbers | 513 | **132,609** |
| How it learns | solved in one step by algebra | many small steps, up to 100 passes over the data |
| Why it exists | the simple, appropriate choice | **you asked me to adapt Ben's code to single-channel glucose** |

> Both use the identical 512 numbers and the identical folds, so the comparison is fair.

**Slide 13 — How we test fairly: hold out whole children**
```
  20 participants  →  5 groups of 4

  Round 1:  train on 16 children  │  test on 4 children they've never seen
  Round 2:  a different 4 held out
  ...  5 rounds, every child tested exactly once

  ✗ WRONG WAY: split by session
      the same child appears in training AND testing
      → the model learns "this child usually scores 1.8"
      → looks good, has learned nothing about glucose
```
> Each child has 32–67 sessions, which is exactly why splitting by session would let the model cheat.
> We check on every round that no child appears on both sides.

---

### Part 3 — The result (6 min)

**Slide 14 — The headline**
> **R², under "predict a child we've never seen"**
> *0 = as good as guessing the average · higher is better · negative = worse*
>
> | | Arm A (Chronos) | Arm B (network) | hand-built features |
> |---|--:|--:|--:|
> | Grids | **−0.089** | −0.218 | −0.065 |
> | Symbols | **−0.056** | −0.281 | −0.038 |
> | Prices | **−0.012** | −0.168 | −0.009 |
>
> **Nothing is above 0.**

**Slide 15 — What "guessing the average" means, and why 0 isn't the neutral point**

*This is the most important explanatory slide in the deck. He told you he doesn't know what this baseline is.*

> **The comparison predictor:** ignore the glucose completely. Take the average score of the training children. Predict that same number every time. For Grids it always says **0.478**.
>
> **Why we need it:** an error on its own means nothing. "My error was 0.50" — good or bad? Only the comparison tells you.
>
> **Now the subtle part.** That comparison predictor, measured under our own folds, **also scores negative**:
>
> | | the comparison predictor | our model |
> |---|--:|--:|
> | Grids | **−0.078** | −0.089 |
> | Symbols | **−0.050** | −0.056 |
> | Prices | **−0.004** | −0.012 |
>
> **So our model ties the trivial predictor — within 0.01. It has not collapsed.**
>
> *Why the comparison is negative:* R² is measured against each test group's own average, but the model was trained on the training group's average. With only 4 children held out, those two averages differ. Under this kind of test, slightly-below-zero **is** the neutral point.

*Speaking note:* if he takes away one methodological point, this should be it. It's the difference between "your model failed catastrophically" and "your model matched the trivial baseline, which is what a flat result looks like."

**Slide 16 — Where we looked for a relationship**
> | We varied | Result |
> |---|---|
> | Model size (4 sizes, 24× parameter range) | flat; the **biggest was worst** (−0.063) |
> | Window length (2 h / 2.5 h / 3 h / all) | shortening it made things **worse** |
> | How we summarize the tokens | no meaningful difference |
> | Reducing 512 numbers to 8–128 | tiny improvement, still negative |
> | **Only low-glucose sessions** (201 sessions, 19 participants) | −0.09 to −0.19 |
> | **Only high-glucose sessions** (449 sessions, 20 participants) | −0.08 to −0.12 |
> | Removing each child's own average score | −0.002 to −0.022 |

**Slide 17 — Reading that honestly**
> - **A bigger model won't fix this.** We spanned a 24× range and the largest did worst — the pattern of something limited by the data, not the model.
> - **The low- and high-glucose sessions are where an effect is most expected physiologically**, and they're still flat. These groups are a decent size, so that's a real finding rather than a small-sample artefact.
> - **Removing each child's personal average** — which asks "can we predict this child's good sessions from their bad ones?" — moves the numbers towards 0 but never above it. Nothing was hiding underneath.
> - Note this uses each child's own sessions to compute their average, so it describes the data rather than predicting for a new child.

---

### Part 4 — Why you should believe it (10 min) ← the heart

**Slide 18 — The question this section answers**
> **A flat result is only worth anything if you've ruled out a broken pipeline.**
>
> Three checks:
> 1. Scramble the scores and see whether we do any worse
> 2. Ask the same machinery to predict something we know is in the data
> 3. Find and fix the mistakes in my own code

**Slide 19 — Check 1: scramble the scores**
> Randomly reassign the scores to the wrong sessions, 200 times, destroying any real relationship. Re-run everything each time.
>
> ```
>   How often does scrambled data score this well?
>
>   −0.15   −0.10   −0.05    0.00
>     │       │       │       │
>     │    ▁▃▅█▇▅▃▁          ← 200 scrambled runs
>     │    ↑                    (average −0.049)
>     │    │
>     └────┴── our real result, −0.106
> ```
>
> | | real | scrambled average | how often scrambled beat us |
> |---|--:|--:|--:|
> | Grids | −0.106 | −0.049 | **100%** |
> | Symbols | −0.127 | −0.052 | **100%** |
> | Prices | −0.085 | −0.051 | 99.5% |
>
> **Our real result is not just inside the scrambled range — it's at the worse end of it.** Scrambled scores do as well as real ones.

*Speaking note:* he flagged the phrase "no signal beyond chance." Never say it. Say: *"scrambled scores do as well as the real ones."*

**Slide 20 — Why the scrambled runs average −0.05, not 0**
> Same reason the comparison predictor is negative (slide 15): R² is measured against each test group's own average. With 4 children per group, a useless model lands slightly below zero by construction.
>
> **So the honest comparison point isn't 0 — it's about −0.05.** And that's exactly where our real result sits.

**Slide 21 — Check 2: can it predict something we know is there?**
> Same 512 numbers, same folds. Instead of a test score, predict a property **of the glucose itself**:
>
> | Predicting… | R² |
> |---|--:|
> | **how variable the glucose was** | **0.462** |
> | the average glucose | 0.070 |
> | share of time above 180 | 0.075 |
>
> **The machinery does extract real information when there is some.** Variability comes back clearly.
>
> **And the low number is expected, not alarming** — because Chronos subtracts each curve's own average before looking at it (slide 11). The average glucose was removed on purpose.
>
> **The reassurance:** the older hand-built features *do* keep absolute level — average, time-in-range — and they were flat too. So "no relationship" holds for level-aware descriptions as well.

*Speaking note:* do **not** say this check "passed." The code only declares success if all three exceed 0.5, and only one comes close. Say: *"variability is recovered well, absolute level is not, and we know why."*

**Slide 22 — What this study could and couldn't have detected**
> With 20 children and 4 held out per round, the **round-to-round variation alone** is ±0.04 to ±0.12 in R².
>
> **So any real relationship smaller than roughly that is invisible to us.**
>
> - We **can** say: a large relationship between glucose and these scores would have shown up, and didn't.
> - We **cannot** say: no relationship exists. A small one could be there.
>
> Also worth being precise about: we have 956 sessions but only **20 children**. Sessions from one child aren't independent, so for a statement about children the effective sample is much closer to 20 than 956.

**Slide 23 — Where most of the variation actually is**
> How much of each score's variation is *between* children versus *within* one child across sessions?
>
> | | between children | **within one child** |
> |---|--:|--:|
> | Grids | 15% | **85%** |
> | Symbols | 38% | **62%** |
> | Prices | 8% | **92%** |
>
> **This corrects something in my own earlier notes.** I had written that differences between children dominate. They don't — most of the variation is one child varying session to session.
>
> **Why that matters:** session-to-session is exactly where a glucose effect would live. So this makes the flat result *stronger*, not weaker — we're looking in the right place and still finding nothing.

---

### Part 5 — What I found wrong (5 min)

*This section is why he will believe you understand the pipeline. Don't skip it and don't rush it.*

**Slide 24 — A real bug in our own code**
> **Chronos pads short sessions to match the longest one in the batch. Our code then averaged over the padding.**
>
> ```
>   a 13-reading session, on its own          →   2 tokens, both real
>   the same session, batched with a 288-reading one → 19 tokens, ~17 of them padding
> ```
>
> **Measured:** those two descriptions of the *same session* have a similarity of **0.22** (1.0 would be identical).
>
> Across all 956 sessions: **861 (90%) were substantially wrong.** Median similarity 0.37.
>
> So a session's description depended on which *other* sessions happened to share its batch — which is not something a description is allowed to depend on.

**Slide 25 — And the result didn't change**
> I recomputed everything with corrected descriptions:
>
> | | with the bug | corrected | using the summary token instead |
> |---|--:|--:|--:|
> | Grids | −0.089 | −0.114 | −0.114 |
> | Symbols | −0.056 | −0.052 | −0.053 |
> | Prices | −0.012 | −0.012 | −0.013 |
>
> **Still nothing above 0.** If anything the bug was flattering the results slightly.
>
> I also tested adding back the two numbers Chronos discards — the average and spread of each curve (slide 11). Also no change.

**Slide 26 — And one uncomfortable comparison**
> How much is the 512-number description actually buying us?
>
> | Description | Numbers | Grids | Symbols | Prices |
> |---|--:|--:|--:|--:|
> | Chronos | 512 | −0.114 | −0.093 | −0.069 |
> | average + spread of glucose | 2 | −0.081 | −0.051 | −0.003 |
> | **just the average glucose** | **1** | **−0.075** | **−0.050** | **−0.004** |
>
> **One number does as well as all 512.**
>
> Nothing beats guessing the average either way — but this tells us where the ceiling is. And it fits slide 7: the raw relationship is under 1% of the variation, so there isn't much for a richer description to find.

---

### Part 6 — Where this leaves us (3 min)

**Slide 27 — What's actually left to try**
> **Inside the current scope (glucose only, this model):**
> 1. Fix the padding, regenerate the stored descriptions, refresh the tables *(measured: won't change the answer, but the code should be right)*
> 2. Keep the two numbers Chronos discards — free, and restores the absolute-level information
> 3. **Make every session use the same span of glucose** — the one structural problem left. Needs the uncut recording; Phil's script would tell us if it's possible.
>
> **What I'd stop doing:** trying bigger models. Four sizes across a 24× range, biggest was worst.

**Slide 28 — What I'd like to decide together**
> - Is the equal-length-window question worth chasing, or do we accept the finding as it stands?
> - What's the bar for saying we've exhausted the glucose-only approach?
> - Three things I still don't know and would like to close:
>   - how the Cohort 2 glucose was filled in *(Phil's code)*
>   - whether the scores are adjusted for age or practice
>   - whether the Prices score is percent incorrect

*Don't put a recommendation slide after this. Ask, then stop talking.*

---

## 4. The appendix — build these, don't present them

He will go deeper. Have these ready to jump to:

| | Contents |
|---|---|
| A1 | Row accounting: 980 → 956, with the 24 dropped rows |
| A2 | Full results tables, all five sweep axes |
| A3 | The exact fold composition: which participants, how many sessions |
| A4 | Chronos architecture: 6 layers, 8 heads, 47,718,016 parameters, of which we use 41.9% |
| A5 | The token-count formula, with the measured table for T = 3 … 2048 |
| A6 | Arm B's parameter breakdown to 132,609, and the epoch arithmetic |
| A7 | Every hyperparameter and where it's set in the code |
| A8 | The three normalizations, and what our docs got wrong about Chronos-Bolt |
| A9 | The four glucose-regime subgroup tables |
| A10 | What Grids is, and the proof (6 items, 5×5 grid, 100% of values explained) |
| A11 | The Grids floor: 31% of sessions score exactly 0 |
| A12 | Known limitations, in one list |

---

## 5. Making the file

If you build it in PowerPoint or Google Slides, that's fine — these are plain layouts. A few practical notes:

- **The ASCII diagrams here are layout sketches, not final art.** Redraw them as boxes and arrows. Keep the same structure.
- **One accent colour**, used only for the number you want looked at. Everything else black on white.
- **Do not use red for the results.** Negative R² isn't a failure state, and colouring it red frames it as one.
- **Slide numbers on every slide.** You will be asked to go back to one.
- **Print the appendix.** Flipping to a printed page is faster than hunting in a file.

There's an existing `CGM_TSFM_Pipeline_Overview.pptx` in the repo root. It predates the real results and the corrections in this bootcamp — treat it as raw material, not a starting point.

---

## 6. The rehearsal that matters

Give the talk out loud, timed, to an empty room. Then:

1. **Did you exceed 40 minutes?** If yes, cut from Part 2, not Part 4.
2. **Did you say any banned word?** Check against `00_SAY_IT_SIMPLY.md` §2.
3. **Did you say "patients" when you meant "sessions"?**
4. **Could you have answered every question in `12_EXAM_DRILL.md` Tier 1?** Those are the ones he's already asked.
5. **Did you lead with the answer, or bury it?**
6. **Did you present the bug yourself, or would he have had to find it?**

One more thing worth saying plainly at some point in the talk, in your own words: *"I want to be clear about what I did and didn't verify myself. Every number in this deck I recomputed from the CSVs rather than copying from our older documents, and that's how I found the three errors I showed you."*

That sentence answers the criticism he actually made, without either defending yourself or apologising.
