# 10 · Rigor and Statistics — how to defend a low-accuracy finding

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you (5 lines).**
> 1. You will be able to explain both rigor checks **without using a single borrowed phrase** — no "no signal beyond chance", no "positive control", no "null".
> 2. You will know *why* the scrambled runs average **−0.05 instead of 0.00**, which is the single best question an examiner can ask you here.
> 3. You will be able to say, with numbers, **how big an effect this study was ever capable of finding** — and admit what it could not.
> 4. You will never again be caught claiming the known-answer check "passed" (it did not fully pass, and the honest version is stronger anyway).
> 5. You will finish with a five-sentence spoken script that makes "we found low accuracy" a *result*, not an excuse.

**Prerequisites:** [`01_FACT_SHEET.md`](01_FACT_SHEET.md) for the raw numbers, [`05_EVALUATION.md`](05_EVALUATION.md) for what R² and grouped folds are. Practice questions continue in [`12_EXAM_DRILL.md`](12_EXAM_DRILL.md).

---

## 0. The vocabulary swap table — read this first, memorize it

Liuyi told you directly, about the rigor section:

> *"I cannot sense the details of these two steps... the words/phrases in this part are not commonly used in my daily life in this project, e.g., 'there is no signal beyong chance', 'positive control'. I feel your reliance on AI is beyond the extent I was expecting."*

and

> *"What does 'Power note' mean? What does 'sensitivity' mean?"*

and

> *"let's always say 'lower accuracy' instead of 'null'."*

He is not objecting to the statistics. He is objecting to **imported vocabulary you cannot unpack on demand**. The fix is not to know fewer words; it is to have a plain phrase ready for every one of them.

| ❌ Don't say | ✅ Say instead |
|---|---|
| "a null result" | "**lower accuracy**" / "**no better than guessing the average**" |
| "no signal beyond chance" | "**the result we got is the same as what we get from scrambled data**" |
| "the null hypothesis" | "**the starting assumption that glucose tells us nothing about the score**" |
| "positive control" | "**a check that asks the pipeline to predict something we know is in the data**" (short handle: *the known-answer check*) |
| "permutation test" | "**the shuffle test**" |
| "the null distribution" | "**the pile of results from scrambled data**" |
| "statistical power" | "**how big an effect this study is even capable of detecting**" |
| "sensitivity" | "**the smallest real effect we could have noticed**" |
| "power note" | (delete the heading; write "**how big an effect we could have detected**") |
| "p-value" | "**how often scrambled data did at least as well as the real data**" (say "p" only after you've said that) |
| "effect size" | "**how strong the relationship is**" |
| "multiple comparisons" | "**we looked at the data many times, so something could look good by luck**" |
| "sample size n = 956" | "**956 test sessions from 20 children**" — always name both numbers |
| "patients" | "**sessions**" or "**participants**" — whichever you actually mean (Liuyi corrected you on exactly this) |
| "confound" (for the variable window) | "**one variable taking different values across sessions**" — Liuyi's rule: a confound needs *two* variables |

**Rule of thumb for the meeting:** say the plain phrase first, then, *once*, "the textbook name for this is X." Never lead with X.

**Say it in your own words:** "I'll use everyday words for these checks, and I'll give the technical name once so you know what I looked up."

**Drill**
1. *Q: What do you mean by "null result"?* → "I don't use that phrase any more. I mean the model's accuracy was low — specifically, it was no better than just guessing the average score."
2. *Q: You wrote "positive control." What is that?* → "It's a check where I ask the exact same pipeline to predict something I already know is in the data. If it can do that, the pipeline works, so the low accuracy on cognition isn't a bug."
3. *Q: Why did you use that jargon in the first place?* → "Because I copied the vocabulary from the method instead of translating it. That was my mistake; here is the plain version." (Say this. It is the answer that ends the objection.)

---

## 1. What "evidence" means when your finding is *low accuracy*

### The asymmetry

Showing that something **exists** is easy in principle: point at it once, convincingly. Showing that something **does not exist** is structurally harder — you can only ever say "I looked in these places, with this much care, and did not find it."

Two sentences that sound the same but are not:

| Sentence | What it actually claims | Can we support it? |
|---|---|---|
| "**We did not find** a relationship between glucose and cognitive score." | A fact about *our study*. | **Yes.** This is our finding. |
| "**There is no** relationship between glucose and cognitive score." | A fact about *the world*. | **No.** 20 children cannot establish that. |

Everything in this chapter exists to make the first sentence airtight and to keep you from ever accidentally saying the second one.

Liuyi's demand was:

> *"if it turns out no signal, we want very concrete evidence to make this 'no signal' claim convincible."*

So "concrete evidence" for low accuracy means answering four suspicions an examiner will have, in this order:

| Suspicion | Our answer | Where |
|---|---|---|
| "Your code is broken." | The same pipeline, same folds, same embeddings recovers glucose **variability** at R² = **0.462**. | §4 |
| "Your result is just noise — maybe you got unlucky." | Scrambled data does **better** than the real data, in 200 out of 200 tries for two of the three scores. | §2 |
| "Maybe there's a signal and your model is too weak." | The strongest raw correlation between *any* simple glucose number and *any* score across all 956 sessions is **r = −0.092** (0.85% of the variation). There is very little there for any model to find. | §5, and `01_FACT_SHEET.md` §E |
| "Maybe you just didn't try hard enough." | 90 reported R² cells across 5 model sizes, 4 window lengths, 2 pooling choices, 6 PCA settings, 3 score-normalizations and 5 glucose-regime subgroups. **Every single one is ≤ 0.** | §7 |

Notice what is *not* on that list: "our result proves glucose doesn't affect cognition." That claim is not available to us and we do not make it.

**Say it in your own words:** "It's easy to prove something is there and hard to prove something isn't. So instead of claiming there's no relationship, I claim we didn't find one — and then I show that the pipeline works, that scrambled data does just as well, and that the raw correlations were nearly flat to begin with."

**Drill**
1. *Q: So glucose doesn't affect thinking?* → "I can't say that. What I can say is that in these 20 children, glucose in the hours before a test did not predict that test's score better than guessing the average. A real effect smaller than what this study can see would look exactly like this."
2. *Q: Why should I believe a negative finding at all?* → "Because I can show three things: the pipeline predicts glucose variability well, the real result is no better than scrambled data, and the raw correlations were under 1% of the variance before any model touched them."
3. *Q: Which of the four suspicions is your weakest answer?* → "The third. I've shown the raw *linear* correlations are flat, but that doesn't rule out a complicated non-linear pattern. My defense there is that a foundation model plus a non-linear regressor is exactly the tool that would find one, and it didn't."

---

## 2. The shuffle test, from zero

*(Textbook name, said once: a **permutation test**. After this paragraph I call it the shuffle test.)*

### 2.1 The logic, in one paragraph

We have 956 rows. Each row pairs one glucose window with one cognitive score. If glucose really predicts the score, the *pairing* is what carries the information. So: **rip the pairing apart.** Randomly reassign which score goes with which glucose window, keep everything else identical, and re-run the entire evaluation. Whatever R² comes out of that run is what our pipeline produces from data where the link is *guaranteed to be gone*. Do it 200 times and you have 200 numbers that describe "what this exact pipeline scores on data with no relationship in it." Then look at where the real number falls in that pile.

That is the whole idea. There is no formula to trust — you *manufacture* the comparison by brute force.

### 2.2 Exactly what we ran

From `cgm_tsfm/run_rigor.py`:

| Setting | Value | Line |
|---|---|---|
| Shuffles per score | **200** | `--n-perm 200` |
| Model, **fixed, no tuning** | `StandardScaler → PCA(32) → Ridge(alpha=10)` | `run_rigor.py:42-54, 75` |
| Folds | 5-fold `GroupKFold` by participant (4 participants held out per fold) | `run_rigor.py:45` |
| Seed | 42 | `config.py:39` |
| What gets shuffled | the score vector, `rng.permutation(yt)` | `run_rigor.py:101` |
| p | `(count of shuffles ≥ real + 1) / (n_shuffles + 1)` | `run_rigor.py:102` |

Total model fits for this one check: 3 scores × (1 real + 200 shuffles) × 5 folds = **3,015 fits.**

### 2.3 The results (`results/rigor_real.md`)

| Score | Real R² | Scrambled: mean ± SD | Scrambled 95th percentile | p |
|---|--:|--:|--:|--:|
| Grids | **−0.106** | −0.049 ± 0.015 | −0.027 | **1.000** |
| Symbols | **−0.127** | −0.052 ± 0.016 | −0.028 | **1.000** |
| Prices | **−0.085** | −0.051 ± 0.014 | −0.025 | **0.995** |

### 2.4 The picture — draw this on a whiteboard

Grids. Each `#` is roughly 2 of the 200 scrambled runs.

```
   how often the scrambled runs landed here
   (200 shuffles, grids)

  50 |                              ####
     |                        ####  ####
  40 |                        ####  ####
     |                        ####  ####
  30 |                  ####  ####  ####
     |                  ####  ####  ####  ####
  20 |                  ####  ####  ####  ####
     |            ####  ####  ####  ####  ####
  10 |            ####  ####  ####  ####  ####  ####
     |      ####  ####  ####  ####  ####  ####  ####
   0 +--*---####--####--####--####--####--####--####-------->  R²
       -0.11 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02  -0.01   0.00
        ^                              ^                ^                ^
        |                              |                |                |
     REAL = -0.106            scrambled mean         95th %ile      "as good as
     (worse than almost        = -0.049              = -0.027       guessing the
      every shuffle)                                                 average"
```

> ⚠️ **Honesty note about this drawing.** The bar heights are a *sketch* built from the reported mean (−0.049) and SD (0.015). The 200 individual shuffle values are not written to disk — only the mean, SD, 95th percentile and p are. So present the three reported numbers as facts and the shape as an illustration. (A small, fair improvement to the code would be to save the raw 200 values.)

The three facts to read off that picture:
1. The scrambled pile sits around **−0.05**, not 0. (§3 explains why. This is the question you will get.)
2. The real value **−0.106** is not in the middle of the pile — it is off the **left** (worse) edge.
3. Distance, in units of the scrambled spread: (−0.106 − (−0.049)) / 0.015 = **−3.8 SD**. Symbols: **−4.7 SD**. Prices: **−2.4 SD**. The real run is *worse* than the scrambled runs by several times their own wobble.

### 2.5 What a p-value is, in plain words

**p = the fraction of scrambled runs that did at least as well as the real run.**

That's it. No theory. It is a counting exercise on the histogram above.

- Grids, p = 1.000 → **every one of the 200** scrambled runs scored at least as high as the real run.
- Symbols, p = 1.000 → same.
- Prices, p = 0.995 → **199 of the 200** scrambled runs did at least as well; exactly **one** did worse.

*(Check the arithmetic on prices: p = (199 + 1) / (200 + 1) = 200/201 = 0.995. Yes.)*

A **small** p (say 0.01) would mean "hardly any scrambled run matched the real one" → the real result stands out → evidence of a relationship. A **large** p means the opposite. **p = 1.000 is the most extreme possible version of "the real result does not stand out."** You literally cannot get a weaker signal than "scrambled data beat us 200 times out of 200."

That is why the phrasing matters so much. "p > 0.05, no signal beyond chance" is technically correct and completely opaque. What you should actually say is:

> *"I scrambled which score belongs to which glucose window, 200 times, and re-ran the whole thing. All 200 of the scrambled runs did at least as well as the real data. So the real pairing is carrying no useful information — the model does not do better with the true pairing than with a random one."*

### 2.6 The "+1" — why it's there

The formula is `(count of shuffles ≥ real + 1) / (n_shuffles + 1)`, not `count / n`.

Plain reason: **the true, unscrambled arrangement is itself one of the arrangements you're comparing against.** You add it to both the top and the bottom of the fraction so you never report "0 out of 200 → p = 0." A p of exactly zero would claim "it is impossible for chance to produce this," and 200 random tries can never justify that word.

Consequence: the **smallest p this test can ever report is 1/201 = 0.00498.** That is the resolution of the instrument. It is not a limitation for us — we needed to detect "p is large" and we got 1.000 — but it means if you ever *do* find something, you cannot report a p below 0.005 without running more shuffles.

### 2.7 Why 200 shuffles is enough here

Two separate arguments, both concrete.

1. **Resolution.** 200 shuffles gives a grid of p-values in steps of 1/201 ≈ 0.005. The threshold everyone uses is 0.05, which is 10 steps up from the floor. So 200 is ample to distinguish "clearly below 0.05" from "clearly above." (If we needed to report p < 0.001 we would need at least 1,000.)
2. **Stability of the reference numbers.** The uncertainty in the scrambled *mean* is SD/√200 = 0.015/14.1 = **±0.001**. So "the scrambled pile centres on −0.049" is pinned to the third decimal. Running 2,000 shuffles would move that by ~0.0005 and change no conclusion, at 10× the compute (3,015 fits → 30,015 fits).

### 2.8 The nuance that makes this result *stronger*: we are at the bad end

Most write-ups of a shuffle test say "the real value fell inside the random range." Ours is more specific and you should lead with the specific version:

> **The real result is not merely inside the scrambled range. It is worse than nearly all of it.**

What does "worse than scrambled" mean concretely? It means the model is **not just uninformative — it is mildly harmful.** Fitting it on real data and applying it to a new child does worse than fitting it on nonsense.

Why would that happen? Because with the *real* pairing there is one thing the model *can* learn: **who the child is.** A given child's glucose curves have a recognizable character (their typical variability, their typical shape), so the 512-number embeddings cluster by child. The real scores also cluster by child (each child has their own average score — the between-participant share of score variation is 0.153 for grids, 0.376 for symbols, 0.077 for prices; see `01_FACT_SHEET.md` §D). So the model learns a shortcut: *"this curve looks like child 7, and child 7 scores around 0.30 — predict 0.30."*

That shortcut is worth something **inside** the training children and worth **less than nothing** on a child it has never seen: it confidently outputs the wrong child's typical level. When you scramble the scores, that shortcut disappears — the shuffled scores have no per-child clustering left — so the shuffled runs are *spared* the harm and score better.

This is why the shuffle test is worth more than a comparison to 0. Comparing to 0 tells you "we didn't beat the average." Comparing to the scrambled pile tells you **"the only thing the model found in the real pairing was participant identity, and that transfers negatively."** That is a mechanism, not just a number, and mechanisms are what survive cross-examination.

### 2.9 The trap: **−0.106 here vs −0.089 in the main table**

`01_FACT_SHEET.md` §I reports grids at **−0.089**. This chapter reports **−0.106**. Both are correct. If you cannot explain the gap in one breath, it looks like you don't know which of your own numbers is real.

| | Main results table | Shuffle test |
|---|---|---|
| Model | Ridge **and** SVR **and** LinearRegression, with the best `alpha`/`C` chosen **per fold** by an inner 3-fold search | **One** fixed model: `PCA(32) → Ridge(alpha=10)`, no tuning at all |
| Grids R² | **−0.089** | **−0.106** |
| Why | tuning per fold gives it the best available chance | fixed settings, so the 201 runs are strictly comparable |

**Why the shuffle test deliberately does *not* tune.** Every one of the 201 runs (1 real + 200 scrambled) must be the *same procedure*, otherwise the comparison is unfair. Tuning inside 201 runs would cost 201 × 5 × (inner search) fits, and — more importantly — a tuner behaves differently on scrambled data than on real data (on noise it just picks the strongest shrinkage and collapses toward the mean), which would bias the comparison in our favour. Fixing the model removes that whole argument.

**Say it out loud like this:** *"The main table lets the model pick its own settings per fold, which gives −0.089. The shuffle test freezes one model so that all 201 runs are identical procedures, and that fixed model gives −0.106. The frozen model is slightly worse, which is expected, and it's the one I compare against the shuffles."*

**Say it in your own words (whole section):** "I randomly reassigned which cognitive score went with which glucose window, 200 times, and re-ran everything. All 200 of those nonsense runs did at least as well as the real data — for two of the three tests, every single one. So the true pairing carries no useful information, and in fact the real run is a bit worse, because the only pattern the model could latch onto was which child it was looking at, and that doesn't help on a new child."

**Drill**
1. *Q: What exactly did you shuffle?* → "The cognitive scores. The glucose windows, the participant labels and the fold split stayed exactly where they were. Only the assignment of score-to-window was randomized."
2. *Q: Does the shuffle keep the participant structure?* → "No, and that's worth flagging. It's a global shuffle, so scores get moved between children too. A stricter version would shuffle only *within* each child, which would test the session-to-session link while keeping each child's own average intact. We haven't run that version; it would be a fair thing to add."
3. *Q: p = 1.000. Isn't that suspiciously round?* → "It's exact, not rounded. It means all 200 shuffles scored at least as high as the real run, so the count is 201/201. The formula includes the real arrangement in both the numerator and the denominator, which is why 1.000 is reachable and 0.000 is not."
4. *Q: Why not 10,000 shuffles?* → "The resolution I need is around 0.05 and 200 shuffles resolves 0.005. The scrambled mean is already pinned to ±0.001. More shuffles would cost 10× the compute and move nothing."

---

## 3. Why the scrambled runs average **−0.05** and not **0.00**

This is the best exam question in the chapter, and it is the one that separates "I ran a script" from "I understand my evaluation."

### 3.1 The intuition first

R² is defined against **the held-out fold's own average**:

```
R² = 1 − (how wrong our predictions were) / (how wrong "always guess this fold's average" would have been)
```

The catch: our model **does not know the test fold's average.** It only knows the training children's average. So even a model that has learned literally nothing is being graded against a benchmark it cannot reach. Two separate things then push it below zero.

### 3.2 Cause A — the model has 32 knobs and fits them to noise (the big one)

The fixed pipeline reduces 512 numbers to **32** components and fits a Ridge on them, using ~765 training sessions. Those 32 weights get tuned to whatever accidental wiggles happen to exist in the training rows. On the training rows this *looks* like an improvement. On new rows those wiggles don't repeat, so each fitted knob adds error instead of removing it.

The standard rule for a linear model fit to pure noise is that the out-of-sample penalty is about **(number of fitted knobs) / (number of training rows)**:

```
32 knobs / 765 training sessions ≈ 0.042
                                   ↓
                    expected R² ≈ −0.042
```

### 3.3 Cause B — the held-out fold's average isn't the training average (the small one)

Suppose the model does the smartest useless thing possible: always predict the training average. Then

```
R² ≈ − (test-fold average − training average)² / (variance within the test fold)
```

which is **never positive**. With scores scrambled, the per-child clustering is gone, so this is just ordinary sampling wobble between ~191 test rows and ~765 train rows:

```
1/191 + 1/765 ≈ 0.0052 + 0.0013 ≈ 0.0065  →  about −0.007
```

### 3.4 Add them up

```
  −0.042   (32 knobs fitted to noise)
  −0.007   (test fold's average ≠ training average)
  ───────
  −0.049   predicted
  −0.049   measured for grids  ✅   (symbols −0.052, prices −0.051)
```

That is a satisfying match. And it explains something else that would otherwise look odd: **all three scores have almost the same scrambled mean (−0.049, −0.052, −0.051)** even though the three scores are wildly different in scale (0.478, 1.846, 41.59) and in how much of their variation is between-child (0.153, 0.376, 0.077). Of course they do — once you scramble, the number depends only on *the shape of the pipeline and the fold sizes*, not on the score at all.

### 3.5 The same arithmetic on the *real* run (and where it breaks)

Now put the participant clustering back. For grids the between-child share is 0.153, and each fold holds out 4 children while training on 16:

```
Cause B, real data:  0.153/4 + 0.153/16 + sampling ≈ 0.038 + 0.010 + 0.006 ≈ 0.053
Cause A, unchanged:                                                        ≈ 0.042
                                                                            ─────
                                                            predicted R² ≈ −0.095
                                                            measured      = −0.106
```

Close. **But be honest about the limits of this arithmetic:** applied to symbols (between-child share 0.376) the same calculation predicts about **−0.17** where we actually measured **−0.127**. So this is *intuition that lands in the right neighbourhood*, not a derivation. Present it that way and you cannot be trapped:

> *"Roughly, about 0.04 of the negative R² comes from fitting 32 components to 765 noisy rows, and the rest comes from the held-out children having a different average score than the training children. That accounting matches grids almost exactly and over-shoots symbols, so I treat it as intuition rather than a proof. The number I actually rely on is the measured one from the shuffles."*

### 3.6 Why this matters more than it looks

If you had compared the real R² only to **0**, you would have concluded "we're 0.106 below where a useless model would be — something is badly wrong." Wrong conclusion. **The floor for a useless model in this design is not 0, it is about −0.05**, and the only reliable way to find that floor is to measure it — which is precisely what the shuffle test does. That is the real argument for running it.

**Say it in your own words:** "R² is scored against the held-out group's own average, but the model only ever knows the training group's average — and it also has 32 free weights it fits to noise in 765 rows. Both of those cost a little accuracy, which is why even a completely useless model lands around minus 0.05 in this setup rather than at zero. The shuffle test measures that floor instead of assuming it's zero."

**Drill**
1. *Q: Shouldn't a useless model get R² = 0?* → "Only if it predicted the test fold's own average, which it can't know. It predicts the training average, and it also fits 32 weights to noise. Those two together cost about 0.05 in this design."
2. *Q: Which of your two causes is bigger?* → "The 32 fitted components: about 0.042 of the 0.049. The average-shift is only about 0.007 once the scores are scrambled."
3. *Q: What if you used no PCA — 512 features instead of 32?* → "The knobs-over-rows term would get much worse: 512/765 is 0.67. In practice Ridge shrinkage prevents anything that extreme, and empirically the full-dimension setting scores −0.052 mean vs −0.045 for PCA(32) — a bit worse, consistent with the direction." *(from `results/sweep_real.md`)*
4. *Q: Then why is the real run worse than the scrambled one?* → "Because with the real pairing, the held-out children's average score genuinely differs from the training children's, and the model has learned the training children's levels. Scrambling erases per-child levels, so the scrambled runs don't pay that cost."

---

## 4. The check that asks the pipeline to predict something we already know is there

*(Textbook name, said once: a **positive control**. Then: the known-answer check.)*

### 4.1 The logic

The worry this addresses is simple and completely reasonable: **maybe the low accuracy is a bug.** Maybe the embeddings are garbage, maybe the folds are wired wrong, maybe the scores got misaligned.

So we take the **identical** 512-number embeddings, the **identical** 5 grouped folds and the **identical** model, and change only the thing being predicted. Instead of a cognitive score, we ask it to predict a property of the glucose curve *itself*. Those properties are computed directly from the very window the embedding came from, so they must be in there. If the pipeline can recover them, it works.

Three targets (`run_rigor.py:79-83`), all computed from the same window `w`:

| What we asked it to predict | How it's computed | Best R² |
|---|---|--:|
| **How much the glucose swung** (standard deviation) | `np.std(w)` | **0.462** ✅ |
| **How high the glucose was on average** | `np.mean(w)` | **0.070** ❌ |
| **What fraction of readings were above 180** | `np.mean(w > 180)` | **0.075** ❌ |

*(Best of Ridge and SVR, from `results/rigor_real.md`. SVR on "% time > 180" actually scored −0.245.)*

### 4.2 Never say this "passed"

`run_rigor.py:109` reads:

```python
if all(b > 0.5 for _, _, _, b in ctrl_rows):
    verdicts.append("✅ Positive control passes: ...")
```

It requires **all three** above 0.5. Only one is (0.462 is also just under 0.5). So **that message never printed, and it is not in `results/rigor_real.md`.** Go look — the verdict section of that file has exactly one bullet, about the shuffle test. If you claim this check "passed," a supervisor who opens the file catches you in ten seconds.

### 4.3 Why "how high" fails and "how much it swung" succeeds — the verified reason

This is not a mystery and it is not a bug. It is Chronos doing exactly what it was designed to do.

Chronos-Bolt applies **instance normalization** to every series before the encoder sees it. From the installed library, `chronos/chronos_bolt.py:105-122`:

```python
loc   = nanmean(x)                               # the series' own average
scale = sqrt(mean((x - loc)²))                   # the series' own standard deviation
scaled_x = (x - loc) / scale                     # ← what the encoder actually receives
```

and `chronos_bolt.py:288` shows it is applied inside `encode()`, on the path our `embed()` call takes.

So for a window averaging 250 mg/dL and one averaging 120 mg/dL, **the encoder sees nearly the same input** — both have been recentered to average 0 and rescaled to swing about ±1. The absolute height and the absolute swing size have been *divided out before the model looks at anything*. What survives is the **shape**: how jagged, how many turns, whether it rises then plateaus.

Those two removed numbers are not thrown away by Chronos — they are handed back. `chronos_bolt.py:461-498`:

```python
embeddings, loc_scale = pipeline.embed(context)   # loc = the mean, scale = the SD
```

And our code, `cgm_tsfm/encoders.py:104`:

```python
emb, _ = self.pipeline.embed(batch)   # ← the underscore IS loc and scale. We discard them.
```

**We throw away the mean and the SD of every window.** That one underscore is the whole explanation for the 0.070.

*(Then why does SD score 0.462 rather than 0? Because after dividing by the SD, the *relative* character of a jagged series still differs from a smooth one — a high-variability window looks different in shape, not just in size — and the model can partly infer the swing size from that. It's a leftover, not a direct read. Say "partly recoverable from the shape," not "fully encoded.")*

### 4.4 The honest sentence to say

Do not say "the check passed." Say this:

> *"I asked the same pipeline to predict three things about the glucose curve itself. It recovered how much the glucose swung — R² of 0.46 — so the embeddings do carry real information and the folds and the plumbing are working. It did **not** recover the average level, only 0.07, and I know exactly why: Chronos subtracts each series' own mean and divides by its own standard deviation before the encoder sees it, and our code discards those two numbers. So my Chronos result is a statement about glucose **shape**, not glucose **level**."*

And then immediately close the hole, because this is the obvious follow-up:

> *"That would be a serious gap if level were untested. It isn't: the 43 hand-crafted features include the mean, the min, the max and the time-in-range fractions — level information, explicitly — and those were flat too (grids −0.065, symbols −0.038, prices −0.009). So the low accuracy holds for level-aware features as well as shape-only ones."*

**The one-line fix worth proposing.** Concatenating `loc` and `scale` back on would give a 514-number feature per session and would let the level-based question be asked properly within the Chronos arm. It is a two-line change in `encoders.py`. Offering this unprompted is the single best move available to you in this section — it shows you understand the limitation well enough to repair it.

**Say it in your own words:** "To prove the pipeline isn't broken, I asked it to predict facts about the glucose curve itself. It got the size of the swings right, which proves the machinery works. It got the average level wrong, because Chronos deliberately removes each curve's average and scale before reading it — so my result is about the shape of the curve, not its height. The hand-crafted features cover height, and they were flat too."

**Drill**
1. *Q: Did the check pass?* → "Partly, and I won't overstate it. Variability came back at 0.46, which is the part I need — it proves the embeddings and folds work. Level came back at only 0.07, and I can tell you the exact line of library code that causes it."
2. *Q: 0.462 isn't very high either. Doesn't that worry you?* → "It's the right order of magnitude for something only *partly* preserved. Chronos divides each window by its own standard deviation, so the swing size is not directly available — the model has to infer it from shape. Getting 46% of the variation of a quantity that was explicitly divided out is evidence the embeddings are rich, not evidence they're weak."
3. *Q: So your whole Chronos analysis is blind to hyperglycemia level?* → "Largely yes, and that's an honest limitation. It sees the shape of an excursion but not its absolute height. The hand-crafted-feature arm does see height, and it also found nothing. The fix is two lines — keep the `loc` and `scale` that `embed()` already returns — and I'd like to run it."
4. *Q: Show me the code.* → `chronos_bolt.py:105-122` for the normalization, `chronos_bolt.py:461-498` for `embed()` returning `loc_scale`, `cgm_tsfm/encoders.py:104` for the `_` that discards it.

---

## 5. How big an effect could this study even detect?

*(This is what the phrase "power note" was trying to say. Delete the phrase. Ask the question.)*

### 5.1 "How strong is a relationship?" — two numbers, plainly

| Number | Plain meaning | Range |
|---|---|---|
| **r** (correlation) | How reliably two things move together. 0 = unrelated. 1 = one is a perfect straight-line copy of the other. Negative = one goes up as the other goes down. | −1 to +1 |
| **R²** | The share of the ups and downs in the score that we can account for. 0.10 = "we explain 10% of why scores vary; 90% is still unexplained." | ≤ 1 (can be negative out-of-sample) |

For a simple straight-line relationship, **R² = r².** That squaring is brutal and worth internalizing:

| r | r² | In words |
|---|---|---|
| 0.09 | 0.008 | 0.8% — **this is the biggest thing in our dataset** |
| 0.1 | 0.01 | 1% |
| 0.3 | 0.09 | 9% — conventionally a "medium" effect |
| 0.5 | 0.25 | 25% — a big effect in behavioural data |
| 0.7 | 0.49 | half of everything |

Our verified ceiling (`01_FACT_SHEET.md` §E): the strongest correlation between any simple glucose statistic and any score across all 956 sessions is **r = −0.092** (fraction of readings below 70, vs the symbols score), i.e. **r² = 0.0085 → 0.85% of the variation.** Every other pairing is under 0.6%.

### 5.2 How much wobble is in our headline number?

You cannot detect an effect smaller than your own measurement noise. Ours, from `results/headtohead_real.md` (spread of R² across the 5 held-out groups):

| Score | R² | Fold-to-fold spread |
|---|--:|--:|
| Grids | −0.089 | **± 0.123** |
| Symbols | −0.056 | **± 0.060** |
| Prices | −0.012 | **± 0.010** |

And separately, the scrambled-run spread is **± 0.014 to ± 0.016** for all three (§2.3) — that is the pure run-to-run noise of the procedure itself.

Put those against the effect sizes above. A true relationship worth R² = 0.01 (r = 0.1) would move grids from −0.089 to −0.079 — **one twelfth of the fold-to-fold spread.** It would be completely invisible. Even R² = 0.09 (r = 0.3, a "medium" effect) is only about *three quarters* of the grids spread — detectable, but not comfortably.

### 5.3 A back-of-envelope on how many participants you'd need

> ⚠️ **Label this out loud as a rough argument, not a formal calculation.** If you present it as a real power analysis and someone asks which test and which assumptions, you're stuck. Say "back-of-envelope."

The standard rule of thumb for detecting a correlation r, with the conventional 80% chance of finding it at the usual 5% threshold, is

```
independent observations needed  ≈  8 / r²
```

| To detect r = | you need about | we have |
|---|--:|---|
| 0.1 (1% of variation) | **800** independent units | — |
| 0.2 | 200 | — |
| 0.3 | 89 | — |
| 0.5 | 32 | — |

Run it backwards for what we actually have. **20 participants**: the smallest correlation we could reliably catch is √(8/20) ≈ **r = 0.63**, i.e. R² ≈ 0.40 — a relationship so strong that glucose would explain 40% of cognitive performance. Nobody believes glucose does that. So *at the participant level, this study is only able to see effects far larger than any plausible truth.*

### 5.4 But we do have 956 sessions — doesn't that help?

Yes, for the *within-child* question, and this is the nuance that makes the argument credible rather than defeatist.

There are two different questions and they have two different sample sizes:

| Question | Effective sample size | What we can say |
|---|---|---|
| "Do children with worse glucose control have worse cognition?" (between-child) | ≈ **20** | Almost nothing. Only a giant effect would show. |
| "Within one child, are their bad sessions the ones after glucose swings?" (within-child) | Much more than 20, but well under 956 (§6) | Something — and this is where our −0.018 / −0.022 / −0.002 centered numbers live. |

Using the standard rough correction for clustered data (see §6 for the formula), the 956 sessions behave like roughly **117** independent observations for grids, **51** for symbols, **207** for prices. Feed those into the same rule of thumb:

| Score | Effective observations | Smallest correlation reliably detectable |
|---|--:|--:|
| Grids | ~117 | r ≈ **0.26** |
| Symbols | ~51 | r ≈ **0.40** |
| Prices | ~207 | r ≈ **0.20** |

And the largest correlation anywhere in the data is **0.092**.

**That is the whole story in one comparison.** We could reliably catch a correlation of 0.2–0.4. What is actually present is at most about 0.09. So an effect of the size that appears to be there is *exactly* the size this study cannot resolve.

### 5.5 The conclusion, in the exact words to use

> **"We can rule out a large effect. We cannot rule out a small one."**

Expanded:

> *"With 20 children, this study could only reliably detect a correlation around 0.2 to 0.4 — glucose explaining roughly 4% to 15% of cognitive performance. Nothing that big is here: the biggest correlation in the whole dataset is 0.09, under 1% of the variation. So the honest reading is that a large effect is ruled out and a small one is not. A small real effect — say 1% of the variation — would be genuinely invisible to a study this size, and I would not be able to tell it apart from what I'm seeing."*

**Say it in your own words:** "With 20 children we could only have spotted a big relationship. The biggest thing in this dataset explains under 1% of the score, and we'd need several hundred children to nail down something that small. So I can say confidently there's no large effect, and I can't say there's no small one."

**Drill**
1. *Q: What does "sensitivity" mean?* → "The smallest real effect we could have noticed. For us that's a correlation of roughly 0.2 to 0.4 depending on which test — anything weaker gets buried in the fold-to-fold wobble."
2. *Q: How many participants would you need?* → "Very roughly, 8 divided by the squared correlation. To pin down a correlation of 0.1 you'd want on the order of 800 independent observations. The grant targeted 92 children; we have 20. I want to be clear this is a back-of-envelope figure, not a formal power calculation."
3. *Q: Is r = 0.09 a real effect or noise?* → "I can't tell, and that's the point. It's a pooled correlation across 956 sessions from only 20 children, so its own uncertainty is larger than the standard formula suggests. I'd call it consistent with zero and also consistent with a small real effect."
4. *Q: Then was this study worth running?* → "For the between-child question, it was always going to be underpowered — that's why the grant plans 92. What it *did* deliver is a properly evaluated pipeline, a measured ceiling on how strong any effect can be, and the finding that a foundation model does no better than 43 hand-crafted features. All three are useful before more data arrives."

---

## 6. The unit-of-analysis trap: 956 sessions, but only 20 children

### 6.1 The picture

```
    20 PARTICIPANTS  ← this is the number that limits us
    ────────────────────────────────────────────────────────
    P01   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●   (32–67 sessions each,
    P02   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●          median 47.5)
    P03   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
     ⋮
    P20   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
    ────────────────────────────────────────────────────────
    956 SESSIONS total   ← this is NOT 956 independent facts

    Each ● is one cognitive test session (one row of the CSV).
    All the ● in a row come from the SAME CHILD, on 10 consecutive days:
      same brain, same age, same insulin regimen, same school-day rhythm,
      same phone, same practice effect from having taken the test before.

    So:  n = 956  for "how many rows do I have"
         n ≈ 20   for "how many independent children do I have"
```

### 6.2 Why this is a trap and not a technicality

If you write "n = 956," a reader assumes 956 independent pieces of evidence and mentally grants you the precision that comes with 956. You would be borrowing credibility you don't have. Two children's sessions are independent; two sessions from the same child are not — they share everything except the day.

Concretely, the standard rough correction for clustered data (the **design effect**) is

```
effective n ≈ n / (1 + (sessions per child − 1) × between-child share of variance)
```

| Score | Sessions | Sessions per child | Between-child share | Effective n |
|---|--:|--:|--:|--:|
| Grids | 951 | 47.6 | 0.153 | ≈ **117** |
| Symbols | 920 | 46.0 | 0.376 | ≈ **51** |
| Prices | 936 | 46.8 | 0.077 | ≈ **207** |

*(Verified inputs from `01_FACT_SHEET.md` §D. The formula is a standard order-of-magnitude correction for averages — use it to argue "roughly 50–200, not 956," not to quote a precise number.)*

Look at symbols: 920 rows behave like about **51**. That is the price of clustering, and symbols pays the most because it has the most between-child structure (0.376).

### 6.3 The two things this justifies

1. **Why we split by participant, never by row.** `GroupKFold` on `subid` holds out 4 whole children per fold. If we split rows randomly, the same child would appear in training and testing, the model would recognize them, and R² would look better while measuring nothing useful. (`regression.py:113` asserts train/test participants are disjoint on every fold — the leakage guard.)
2. **Why R² is negative here and would be positive under row-splitting.** Grouped folds are a genuinely harder test. Do not describe the negative R² as a failure without saying which test produced it.

### 6.4 Precision of language — Liuyi already corrected you on this

His Word comment (C0) on the subgroup table: say **sessions**, not "patients." In that table 201 and 449 are *sessions*; 19 and 20 are *participants*.

Build the habit of never using a bare number:

| ❌ | ✅ |
|---|---|
| "201 hypoglycemic patients" | "201 **sessions** in which at least one reading was below 70, from **19 participants**" |
| "we have 956 samples" | "**956 test sessions** from **20 participants**" |
| "n = 951 for grids" | "**951 sessions have a grids score** — 5 are missing — across the same 20 participants" |

**Say it in your own words:** "We have 956 test sessions but only 20 children, and 40-odd sessions from one child aren't 40 independent facts — same brain, same ten days. So for any claim about children, our effective sample is closer to 20 than 956. That's why every fold holds out whole children instead of random rows."

**Drill**
1. *Q: You have almost a thousand data points. Why is 20 the number you keep quoting?* → "Because the question is about children. Forty sessions from one child tell me a lot about that child and very little about the next one. For a claim that generalizes to new children, the sample is 20."
2. *Q: What would happen if you split rows randomly instead?* → "R² would go up and it would be meaningless — the model would be recognizing children it had already seen. The code asserts on every fold that no participant appears in both the training and test sides."
3. *Q: So the 956 sessions are useless?* → "Not at all — they're what lets us ask the within-child question: is this child worse on sessions that follow a glucose swing? That's the centered analysis, where the numbers are −0.018, −0.022 and −0.002. Still ≤ 0, but it's a different and better-powered question."
4. *Q: How many sessions per child?* → "Minimum 32, median 47.5, maximum 67, mean 47.8. No child dominates the dataset."

---

## 7. We looked at this data many times

### 7.1 The idea, with coins

Flip a fair coin 5 times. Getting 5 heads has about a 1-in-32 chance — surprising. Now let 90 people each flip 5 times. Somebody getting 5 heads is now *expected*, and if you report only that person and call them gifted, you have fooled yourself and your reader.

The arithmetic version: if each of 20 independent settings has a 1-in-20 chance of looking good by luck, the chance that **at least one** looks good is 1 − 0.95²⁰ = **64%**. With 90 settings it's 1 − 0.95⁹⁰ = **99%**. Practically guaranteed.

### 7.2 How many looks we actually took

The results files in this repo report **90 R² cells on real data** (some are the same configuration repeated as a reference row in more than one table):

| File | Axis explored | Settings | Cells (× 3 scores) |
|---|---|--:|--:|
| `sweep_real.md` | model checkpoint | 5 (tiny/mini/small/base/t5-small) | 15 |
| `sweep_real.md` | window length | 4 (2.0 h / 2.5 h / 3.0 h / full) | 12 |
| `sweep_real.md` | pooling | 2 (mean / last) | 6 |
| `sweep_real.md` | PCA components | 6 (none/8/16/32/64/128) | 18 |
| `sweep_real.md` | score normalization | 3 (raw / centered / z-scored) | 9 |
| `subgroups_real.md` | glucose regime | 5 (all / <70 / >250 / any excursion / in-range only) | 15 |
| `headtohead_real.md` | representation | 3 (Chronos+Ridge/SVR, Chronos+MLP, 43 features) | 9 |
| `headtohead_real_centered.md` | representation, centered | 2 | 6 |
| | | | **90** |

Plus, inside every one of those cells, the inner search picked the best of ~11 model settings (5 Ridge alphas, 6 SVR combinations) per fold. So the true number of models fitted is in the thousands — `01_FACT_SHEET.md` §G counts **555 fits per Arm A run**.

*(One honest detail: the sweeps vary **one axis at a time**, holding the others at bolt-small / full window / mean pooling / no PCA / raw scores. It is **not** a full grid — a full grid would be 5 × 4 × 2 × 6 × 3 × 5 = 3,600 configurations, i.e. 10,800 cells. We report 90. Say "one axis at a time" if asked, because claiming a full grid would be false.)*

### 7.3 The standard fixes

| Fix | What it does | Plain version |
|---|---|---|
| Raise the bar | Divide the 0.05 threshold by the number of looks: 0.05/90 = **0.00056** (Bonferroni) | "If I take 90 looks, one look has to be 90 times more impressive to count." |
| Pre-commit | Decide the single analysis **before** seeing results; report everything else as exploration | "Pick your shot before you fire." |
| Hold out a second cohort | Confirm on data you never touched | "Find it in one group, then check it in another." |

### 7.4 The honest position — and it is a comfortable one

**Every one of the 90 cells came out ≤ 0.** The many-looks problem is a problem when you report *the best of many* and treat it as if it were the only one. We are reporting *the worst case across all of them*, which is the opposite failure mode: taking many looks and finding nothing makes the "we found nothing" claim **stronger**, not weaker.

But say the second half too, because it is where you would actually be vulnerable:

> *"Because everything came out at or below zero, I'm not at risk of reporting a lucky false positive here. The risk shows up the moment we start hunting through subgroups. `subgroups_real.md` already carries an 'exploratory' warning for exactly that reason — the hypoglycemia subgroup drops to 201 sessions from 19 participants, so its numbers are noisy. If we ever go looking for a signal inside a subgroup, that analysis needs to be decided in advance, or the threshold corrected for how many subgroups we tried."*

**Say it in your own words:** "If you try enough settings, one will look good by accident — with 90 tries it's basically guaranteed. We tried 90 and every one came out at or below zero, so we don't have a lucky winner to worry about. But if we ever go hunting for a signal inside a subgroup, we have to decide which subgroup in advance, or raise the bar for how impressive it has to be."

**Drill**
1. *Q: You tried 90 combinations. Isn't that fishing?* → "It would be, if I were reporting the best one as the headline. I'm reporting that all 90 are at or below zero. Trying hard and finding nothing supports the finding; trying hard and reporting the single best would undermine it."
2. *Q: What's the corrected threshold for 90 looks?* → "0.05 divided by 90, about 0.0006. Our shuffle-test p-values are 1.000, 1.000 and 0.995, so no correction changes anything."
3. *Q: The in-range-only subgroup is the worst at −0.175 / −0.296 / −0.111. Is that meaningful?* → "I'd treat it as noise. That subgroup is 143 sessions from 19 participants — the smallest slice we have — and small grouped folds make R² unstable in the negative direction, which is the same mechanism from section 3. The file labels it exploratory."
4. *Q: What would pre-registration look like here?* → "Write down before the run: one score, one model, one window, one subgroup, and the criteria in section 8. Then report that, and label everything else as exploration."

---

## 8. What we would need to see to believe there IS a relationship

Writing these down **before** looking is what would make a later positive finding believable. Committing to them now costs nothing and buys enormous credibility, because it proves the criteria weren't reverse-engineered from a result someone liked.

**Pre-committed criteria. A positive finding must clear all four.**

| # | Criterion | For us, concretely |
|---|---|---|
| 1 | **R² above zero on held-out children.** Not on the training children, not with the folds split by row. | Anything ≤ 0 does not count. Our best is −0.012. |
| 2 | **Same direction in most folds.** At least 4 of the 5 held-out groups positive. | A single strong fold and four bad ones is one lucky group of 4 children, not a finding. |
| 3 | **Clearly outside the scrambled pile.** Above the 95th percentile of the 200 shuffles — i.e. p < 0.05. | The bar to beat is **−0.027** (grids), **−0.028** (symbols), **−0.025** (prices). Note these bars are *negative* — that's the floor from §3. |
| 4 | **Bigger than the fold-to-fold spread.** | Must exceed about **0.12** for grids, **0.06** for symbols, **0.01** for prices. |

Two more that would move it from "believable" to "solid":

| # | Criterion |
|---|---|
| 5 | **Survives the number of looks** — either pre-committed to one setting, or clears a corrected threshold. |
| 6 | **Replicates in a second cohort** — found in Cohort 1, confirmed in Cohort 2 (or in the additional participants the grant plans). |

**How to deploy this in the meeting.** When Liuyi asks "what would change your mind?", do not improvise. Read the list. Then add:

> *"I wrote these down before running anything further, so if a future version does show a signal, you'll know I didn't invent the standard afterwards to fit the answer."*

**Say it in your own words:** "For me to believe there's a real relationship, I'd need R² above zero on children the model never saw, pointing the same way in at least four of five folds, clearly better than the scrambled runs, and bigger than the fold-to-fold wobble. I wrote those down in advance on purpose."

**Drill**
1. *Q: Why does criterion 3 use a negative bar?* → "Because the floor for a useless model in this design is about −0.05, not 0 — that's section 3. The 95th percentile of the scrambled runs is where a useless model's *lucky* results reach, and that's −0.027 for grids."
2. *Q: Isn't criterion 4 very strict?* → "It is, and it's strict because our fold-to-fold spread is large — ±0.123 for grids. An effect smaller than the spread means we can't tell which folds it's in. The way to relax that criterion is more participants, not a lower bar."
3. *Q: Could all four be met and the finding still be wrong?* → "Yes — one dataset can be idiosyncratic. That's what criterion 6 is for."

---

## 9. "Mean ± SD across 5 folds" is **not** a confidence interval

### 9.1 What we report, and what it is

Every R² in our tables looks like `−0.089±0.123`. That is:

> the **average** R² across the 5 held-out groups, and the **spread** of those 5 numbers around their own average.

It is a description of **how much the folds disagreed with each other**. Nothing more.

### 9.2 What a confidence interval would be, and why we can't just make one

A confidence interval is a range built to contain the true value a stated fraction of the time under repeated sampling. To turn our 5 numbers into one you'd divide the spread by √5 and add a distributional assumption. That would be wrong here, for three reasons:

1. **The 5 folds are not independent.** With 5 grouped folds over 20 participants, any two training sets share 12 of their 16 participants. Five heavily overlapping estimates do not carry five estimates' worth of information, so √5 flatters us.
2. **The spread mixes two different things.** Part of it is estimation noise; part of it is that *fold 3's four children are genuinely different children* from fold 1's. Averaging them into one "uncertainty" hides that.
3. **Honest interval estimation for nested cross-validation is a known hard problem.** There is no one-line fix, which is exactly why we lean on the shuffle test instead: it gives a properly calibrated reference distribution *for this specific pipeline and this specific fold structure*, without needing a formula.

### 9.3 How to phrase uncertainty honestly

| ❌ Don't write | ✅ Write |
|---|---|
| "R² = −0.089 ± 0.123 (95% CI)" | "R² = −0.089; across the five held-out groups the values spread by ±0.123. That is fold-to-fold disagreement, not a confidence interval." |
| "R² was significantly below zero" | "R² was below zero in every configuration we tried; the shuffle test puts it below the 5th percentile of what scrambled data produces." |
| "the model explains −8.9% of variance" | "the model did **worse** than simply guessing the average score, by about 9% of the score's variation." |
| "results were stable across folds" | "the five folds ranged over about a quarter of an R² point for grids, which is large relative to the effect we're looking for." |

Say the phrase out loud once so it's ready: **"That's the spread across folds, not a confidence interval."** It is the single most credibility-buying sentence in this chapter, because almost nobody volunteers it.

### 9.4 A small gap worth admitting

The per-fold R² values *are* computed (`regression.py:60, 135` build a list of `FoldResult`), but only the mean and SD are written to the results files — the five individual numbers are not saved. So "was it 4 of 5 folds or 1 of 5?" cannot currently be answered from the files. Saving the per-fold numbers, and the 200 raw shuffle values, is a small honest improvement to propose.

**Say it in your own words:** "The ± in our tables is just how much the five folds disagreed with each other. It isn't a confidence interval, and I don't want it read as one — the five folds share most of their training children, so they're not independent measurements."

**Drill**
1. *Q: Is −0.089 ± 0.123 significantly different from zero?* → "I wouldn't put it that way, because that ± isn't a confidence interval — the five folds share most of their training data. What I can say is the shuffle test compared the real run against 200 scrambled ones, and it landed at the bad end of them."
2. *Q: Why not just divide by √5?* → "That assumes five independent estimates. With 5 grouped folds over 20 children, any two training sets overlap in 12 of 16 children, so √5 would make the interval look tighter than it is."
3. *Q: Then what is your honest uncertainty?* → "Two anchors. The folds disagree by ±0.123 for grids, and the same pipeline on scrambled data varies by ±0.015. The real number sits about 3.8 of those scrambled spreads below the middle of the scrambled pile."

---

## 10. The five sentences that make our low-accuracy finding credible

Memorize these. Deliver them in this order. No jargon anywhere in them.

> **1.** *"Across 956 test sessions from 20 children, the model never beat simply guessing the average score — the best result was minus 0.012, and every one of the 90 settings we tried came out at or below zero."*
>
> **2.** *"That isn't a broken pipeline: when I asked the exact same embeddings and the exact same folds to predict how much the glucose swung, it got an R² of 0.46. The machinery works — it just can't find the cognitive score."*
>
> **3.** *"I then scrambled which score went with which glucose window, 200 times, and re-ran everything. All 200 scrambled runs did at least as well as the real data. So the true pairing carries no useful information."*
>
> **4.** *"The reason is visible before any model runs: the strongest correlation between any simple glucose number and any score, across all 956 sessions, is 0.09 — under 1% of the variation in the scores."*
>
> **5.** *"So my claim is narrow and I'll keep it narrow: with 20 children we can rule out a large relationship, and we cannot rule out a small one. To pin down something explaining 1% of the variation you'd need hundreds of participants, not twenty."*

**And the follow-up you volunteer without being asked** — this is what makes it read as science rather than a defense:

> *"Two things I'd fix. Chronos removes each window's average and scale before it reads it, and our code throws those two numbers away, so the Chronos arm is blind to absolute glucose level — that's a two-line fix in `encoders.py`. And I'd like to run the shuffle test in a stricter form that shuffles only within each child, so it tests the session-to-session link while keeping each child's own average intact."*

---

## The five hardest questions, with model answers

**Q1. "You keep saying 'no signal beyond chance.' I don't know what that means. Say it in normal words."**

> "You're right, and I'll drop that phrase. Here's what I did. I took the 956 rows and randomly reshuffled which cognitive score belonged to which glucose window — so any real link is destroyed by construction. I re-ran the entire evaluation on the scrambled version, 200 times. Then I compared: does the real, correctly-paired data do better than the scrambled data? It does not. For grids and symbols, all 200 scrambled runs scored at least as well as the real run. For prices, 199 of 200 did. So the correct pairing gives the model no advantage over a random one. That's the whole claim."

**Q2. "Your 'positive control' — what is it, and did it pass?"**

> "I'll call it what it is: a check where I ask the pipeline to predict something I already know is in the data. Same 512-number embeddings, same five folds, same model — I just swap the thing being predicted, from a cognitive score to a property of the glucose curve itself. It recovered how much the glucose swung at R² 0.46, which tells me the embeddings carry real information and the folds and plumbing are correct. It did **not** recover the average glucose level — only 0.07. So no, I won't say it passed; it half-passed, and the half that failed has a specific known cause. Chronos subtracts each series' own average and divides by its own standard deviation before the encoder sees anything, and it hands those two numbers back separately — and our code discards them, at `encoders.py` line 104. So this arm sees the *shape* of a glucose curve, not its *height*. The 43 hand-crafted features do encode height and they were flat too, which is why I still think the finding holds."

**Q3. "Why do the scrambled runs average minus 0.05? Shouldn't random data score zero?"**

> "It would, if the model could predict the held-out group's own average — but R² is graded against that average and the model only knows the *training* group's average. Two things cost it accuracy. First and bigger: the model fits 32 components to about 765 training rows, and those 32 weights get tuned to noise that doesn't repeat on new rows. The rule of thumb is knobs over rows: 32 over 765 is about 0.042. Second and smaller: the held-out group's average differs a little from the training group's, which costs about 0.007 once the scores are scrambled. Together, minus 0.049 — which is exactly what we measured, and it's nearly identical for all three tests, as it should be, because after scrambling the number depends only on the pipeline and the fold sizes, not on the score. That's the reason the shuffle test is worth running: it *measures* the floor instead of assuming the floor is zero."

**Q4. "You said the real result is worse than the scrambled result. Doesn't that mean something is broken?"**

> "It means something specific and I think it's informative. With the real pairing there is exactly one pattern available to the model: which child it's looking at. A child's glucose curves have a recognizable character, so the embeddings cluster by child, and the real scores also cluster by child — between 8% and 38% of score variation is 'who you are.' So the model learns 'this looks like child 7, predict child 7's usual score.' Inside the training children that helps a little. On a child it has never seen, it confidently predicts the wrong child's level, which is worse than useless. Scrambling the scores erases the per-child clustering, so the scrambled runs never pay that cost — which is why they score better. The model isn't broken; it found the only learnable pattern in the data and that pattern doesn't transfer."

**Q5. "So how confident are you that glucose doesn't affect cognition in these kids?"**

> "I'm not making that claim, and I want to be precise about the difference. What I can defend is: in this dataset, glucose in the hours before a test did not predict that test's score better than guessing the average, and the evidence for that is strong — the pipeline demonstrably works, scrambled data does just as well, and the raw correlations were under 1% of the variation before any model ran. What I cannot defend is that no relationship exists. With 20 children, this study could only reliably detect a correlation somewhere around 0.2 to 0.4. The biggest correlation anywhere in the data is 0.09. So a real effect of the size that might actually be there is exactly the size we're unable to resolve. Large effect: ruled out. Small effect: still open. That's the honest boundary, and I'd rather state it than blur it."

---

## Ten-sentence summary

1. Showing that something exists is easier than showing it doesn't, so a low-accuracy finding needs *manufactured* evidence rather than an absence of evidence.
2. Never say "null result" — say "lower accuracy" or "no better than guessing the average"; never say "no signal beyond chance" — say "scrambled data did just as well."
3. **The shuffle test:** we randomly reassigned which cognitive score went with which glucose window 200 times and re-ran the full grouped evaluation; all 200 scrambled runs matched or beat the real run for grids and symbols, and 199 of 200 for prices.
4. p is just the fraction of scrambled runs that did at least as well as the real one; p = 1.000 is the strongest possible version of "our result does not stand out," and the +1 in the formula puts a floor of 1/201 ≈ 0.005 on what we can ever report.
5. The scrambled runs average −0.05 rather than 0.00 because the model fits 32 components to ~765 noisy rows (costing ≈ 0.042) and is graded against a held-out average it cannot know (costing ≈ 0.007) — so the floor for a useless model in this design is measured, not assumed.
6. Our real result is not merely inside the scrambled pile but 2.4 to 4.7 spreads below its centre, because the only pattern the real pairing offers is participant identity, which transfers negatively to a new child.
7. **The known-answer check** asked the identical pipeline to predict properties of the glucose curve itself: variability came back at R² 0.462 (so the machinery works), but average level only at 0.070 — because Chronos subtracts each window's mean and divides by its standard deviation, and `encoders.py:104` discards those two numbers; we must therefore never say this check simply "passed."
8. With 20 participants, the smallest relationship we could reliably detect is a correlation of roughly 0.2 to 0.4, while the strongest correlation anywhere in the data is 0.092 — so we can rule out a large effect and cannot rule out a small one.
9. We have 956 sessions but only 20 children, and the clustering makes them behave like roughly 51 to 207 independent observations — which is why folds hold out whole participants and why saying "sessions" versus "participants" precisely is non-negotiable.
10. We tried 90 settings and all came out ≤ 0, so we are not reporting a lucky winner — but any future subgroup hunt must be decided in advance or held to a corrected threshold, and the four pre-committed criteria in §8 are what a future positive finding would have to clear.

---

*Every number in this chapter traces to `results/rigor_real.md`, `results/sweep_real.md`, `results/subgroups_real.md`, `results/headtohead_real.md`, `results/headtohead_real_centered.md`, `cgm_tsfm/run_rigor.py`, `cgm_tsfm/encoders.py`, `cgm_tsfm/config.py`, or the installed `chronos/chronos_bolt.py`. The arithmetic in §3.4, §3.5, §5.3 and §6.2 is derived from those numbers and is labelled as approximate wherever it is. Companion chapters: [`01_FACT_SHEET.md`](01_FACT_SHEET.md) (every number with proof), [`05_EVALUATION.md`](05_EVALUATION.md) (R², grouped folds, leakage), [`12_EXAM_DRILL.md`](12_EXAM_DRILL.md) (question bank).*
