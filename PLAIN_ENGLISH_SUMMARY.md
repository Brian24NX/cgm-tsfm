# Plain-English Summary — What We Built, What Everything Means, & What's Next

*A no-jargon catch-up + a meticulous deep-dive on every confusing concept. If you read one file in this project, read this one. It defines every technical word and draws pictures.*

**This file has two parts:**
- **Part I — the quick catch-up** (what the project is, what we built, the plan). Skim this first.
- **Part II — Deep-Dive Clarifications**: your 14 questions, each answered slowly, with diagrams and examples. Jump to any of them from the table below.

> ### 🆕 What changed at last week's meeting (2026-07) — read this first
> Three updates that override older wording elsewhere in the repo:
> 1. **Forget the comparison with Mack's approach.** Liuyi confirmed: **focus only on our own AI (TSFM) approach.** We do *not* need to frame the project as "beat Mack's hand-crafted features." (We keep his features in the code only as an optional internal sanity-check, not as the headline. See [Q12](#q12).)
> 2. **The lab-server environment is set up and GPU-validated** on `cpsl-mds`. That effectively completes **Task 1**. (See [Q14](#q14) + `docs/05_SERVER_SETUP.md`.)
> 3. **The real data is on the way.** Another student, **Phil**, is finishing the statistics/correlation analysis for the 20 patients; once he's done, Liuyi sends us the merged glucose file. So the "get the data" ask is no longer really an open question — it's just in progress. (See [Q8](#q8), [Q14](#q14).)

---
---

# PART I — THE QUICK CATCH-UP

## 0. The 30-second version

We're studying: **when a kid with Type 1 Diabetes has their blood sugar go up or down, does it change how well they do on a quick thinking test a few minutes later?**

Your job: build and run a **modern AI approach** to answer this. Over these sessions we **built the entire software pipeline**, tested it end-to-end on stand-in data, ran the real Amazon Chronos AI on a GPU, and got it ready so that the day the real data arrives, it's basically one command. We can't get "real answers" yet because the real data file hasn't been sent — but everything is built and waiting.

## 1. What the project is actually about (no jargon)

- Kids with **Type 1 Diabetes (T1D)** have blood sugar (**glucose**) that swings up and down a lot.
- They wear a **CGM** = *Continuous Glucose Monitor* — a small sensor that measures glucose automatically every 5 minutes. So we get a long list of glucose numbers over time.
- They also play short thinking games on a phone app several times a day. Three games/scores: **Grids, Symbols, Prices** (they test memory and mental speed — see [Q2](#q2)).
- **The question:** using the glucose readings from the ~couple hours *before* a game, can we predict the game score? If yes, that means short-term blood sugar affects thinking in the moment.

That's it. **Glucose readings in → predicted thinking-test score out.** (The full pipeline picture is in [Q9](#q9).)

## 2. Mini-dictionary (quick defs — each links to its full explanation in Part II)

- **Model** — a computer program that makes predictions after learning from examples.
- **Regression / target** — "regression" = *predicting a number* (the test score). The number we predict is the **target**.
- **Feature engineering** vs **raw-data learning** — the old way (a human hand-picks summary numbers) vs the new way (feed the raw curve to an AI). → **[Q3](#q3)**
- **Foundation model** — a big AI pre-trained on huge data, reused instead of trained from scratch. (ChatGPT is one, for language.)
- **TSFM** = *Time-Series Foundation Model* — the same idea, but for numbers-over-time (like glucose). → **[Q1](#q1)**
- **Chronos** — the specific TSFM we use (made by Amazon, free). "ChatGPT, but for sequences of numbers over time." → **[Q1](#q1)**
- **Embedding / fingerprint** — the AI's output when you feed it a glucose window: a list of 512 numbers that captures the curve's shape. We predict the score from this. → **[Q4](#q4), [Q10](#q10)**
- **R² ("R-squared")** — the report card. **1.0 = perfect, 0 = no better than guessing the average, below 0 = worse than guessing.** → **[Q5](#q5)**
- **Baseline** — the "just guess the average every time" strategy. A real model must beat it. → **[Q6](#q6)**
- **Cross-validation** — a fair test: train on some data, test on *different* data, repeat. → **[Q7](#q7)**
- **Grouped by subject / leakage** — never test on a kid seen in training, or the model "cheats." → **[Q7](#q7)**
- **Overfitting** — a model *memorizes* the practice examples instead of learning the real pattern → looks great in practice, fails on new data.
- **Synthetic data** — fake data with the exact same *shape/format* as the real data, so we can build & test before the real file arrives. → **[Q8](#q8)**
- **Arm A / Arm B** — two versions of our approach (simple predictor vs small trainable network). → **[Q11](#q11)**

## 3. The one thing to keep honest (expectations)

An earlier careful attempt on this exact problem found the glucose→thinking signal is **weak**, and that most of the difference between test scores is just *"some kids score higher than others in general"* (their personal baseline) — **not** *"their blood sugar in the last hour changed their score."*

Why this matters even though we're not comparing to that attempt anymore:
- **A fancy AI cannot invent a signal that isn't there.** So we shouldn't *promise* Liuyi our approach will suddenly work great.
- What our approach *can* honestly do: **test the modern method fairly**, and **directly check the "it's just baseline" theory** (via the within-subject knob — see [Q13](#q13)). That's a legitimate, publishable result *even if the honest answer is "the signal is weak."*

Saying this out loud shows maturity — you understand the science, not just the code.

## 4. What we actually built (plain words)

An assembly line for "glucose window → predicted score." We built the whole line:

1. **A data reader** that understands the real data's format (one row per game session, with the glucose readings before it and the three scores), plus a **synthetic-data generator** that fakes this exact format so the line runs *today*. → [Q8](#q8)
2. **The AI fingerprint step** — feeds each glucose window into Chronos → a 512-number fingerprint. We downloaded and ran the real Amazon Chronos model (on the GPU) to confirm it works. → [Q10](#q10)
3. **The prediction + report-card step** — predicts each score and measures R² *fairly* (grouped by kid, no cheating). → [Q5](#q5), [Q7](#q7)
4. **Two versions ("arms")** — Arm A (fingerprint + simple predictor) and Arm B (fingerprint + small trainable network, the "adapt Ben's code" task). → [Q11](#q11)
5. **"Knobs"** to test settings that each answer a real question (model size, history length, PCA, within-subject test). → [Q13](#q13)
6. **Auto-generated result tables** in `results/`, with a plain-English `results/README.md`.

**Bottom line:** the machine is fully built, tested, and GPU-validated. It's parked, waiting for the real fuel (the data from Phil/Liuyi).

## 5. How this maps to the 4 tasks Liuyi gave you

| Task | Plain meaning | Status |
|---|---|---|
| 1. compute/server environment | Set up the lab computer (GPU) | ✅ **Done** — built & GPU-validated on `cpsl-mds` (`docs/05`). *(If "compute2" is a different specific machine, that's a quick follow-up — see [Q14](#q14).)* |
| 2. Understand the code + how the AI takes input | Learn the tools | ✅ Done |
| 3. Understand the AI's input format | Exactly how to feed data to Chronos | ✅ Done — studied the ICML paper's code Liuyi pointed to **and implemented it** |
| 4. Adapt Ben's code for our case | 1 glucose input, predict a number, 3 scores | ✅ Done (and tested) |

Plus extra: the full pipeline, the analysis knobs, and the auto-reports.

## 6. What to TELL Liuyi (almost word-for-word)

1. "I re-scoped to your answers: **one input channel (just glucose), one score at a time (three separate models), the raw-data/AI approach only** — and per last meeting I'm **not** framing it against Mack's features anymore."
2. "I learned how Chronos takes input from the ICML paper's code — and **implemented that recipe** (Task 3, done in practice)."
3. "I **adapted Ben's code** to our single-glucose-channel, predict-a-number case (Task 4)."
4. "I built the **whole pipeline end-to-end**, and **set up + GPU-validated the lab server** (`cpsl-mds`) — real Amazon Chronos runs on the GPU. So **Task 1 is effectively done** too. It's on **synthetic stand-in data** until the real file arrives."
5. "I added knobs to test model size, history-window length, a dimensionality trick (PCA), and — importantly — a **within-subject test** that directly checks whether the signal is just per-kid baseline."
6. "Everything auto-writes result tables. **The only thing left for real results is the data** — I understand Phil is finishing the correlation analysis first."

## 7. The honest caveat (worth saying)

"All my current numbers are from **synthetic stand-in data**, so they're a **plumbing test, not a scientific result** — they prove the machine works, not that glucose predicts thinking. Given the signal looks weak, I'm not expecting a miracle; the value is an **honest, fair evaluation** of the modern approach and a **direct test of the baseline theory**. Real numbers come the moment I get the data."

*(Supervisors trust students who don't oversell.)*

---
---

# PART II — DEEP-DIVE CLARIFICATIONS (your 14 questions)

Each section is self-contained. Jump to whichever confuses you.

| # | Your question | Section |
|---|---|---|
| 1 | What is Amazon Chronos? How good? Alternatives? Why this one? | [Q1](#q1) |
| 2 | The three games: Grids, Symbols, Prices — what/why | [Q2](#q2) |
| 3 | Feature engineering vs raw-data learning (+diagram) | [Q3](#q3) |
| 4 | Embedding — what is it, what does it look like, how do we predict from it | [Q4](#q4) |
| 5 | R² — what/why/what 1, 0, <0 mean (+example) | [Q5](#q5) |
| 6 | Baseline | [Q6](#q6) |
| 7 | Grouped-by-subject / leakage (+diagram) | [Q7](#q7) |
| 8 | Synthetic data — what/why (mock runs before real data?) | [Q8](#q8) |
| 9 | The pipeline: glucose window → predicted score (+diagram) | [Q9](#q9) |
| 10 | The "AI fingerprint" step / the 512-number fingerprint | [Q10](#q10) |
| 11 | Arm A vs Arm B (+diagram) | [Q11](#q11) |
| 12 | Mack's old approach + why we now ignore it | [Q12](#q12) |
| 13 | The "knobs" — the real question each one answers | [Q13](#q13) |
| 14 | What to ask Liuyi — updated with your answers + my decisions | [Q14](#q14) |

---

<a name="q1"></a>
## Q1 — What is Amazon Chronos? How good is it? What are the alternatives, and why did we pick it?

**What it is.** Chronos is a **Time-Series Foundation Model (TSFM)** — a big neural network that Amazon pre-trained on a huge pile of time-series data (millions of real + synthetic sequences). The clever trick: Chronos treats a time series *like a sentence*. It **scales** the numbers, **chops the value-range into "bins"** and turns each reading into a "word" (token), then trains a **language model** (the T5 architecture — same family idea as the models behind chatbots) to understand those sequences. So literally: **Chronos is "a language model, but its language is sequences of numbers over time."**

Because it saw so many time series in training, it already "knows" the general shapes time series take (trends, spikes, daily cycles, noise) — so you can point it at *your* series (glucose) with **zero extra training** and it produces something useful.

**Chronos vs Chronos-Bolt** (we use **Bolt**): the original Chronos generates forecasts one token at a time (slow). **Chronos-Bolt** is a newer variant that reads the series in **patches** (chunks) and predicts directly — it's **~an order of magnitude faster** and runs comfortably on CPU *or* GPU, while being at least as accurate. That speed matters because we extract fingerprints for ~900+ sessions and iterate a lot.

**How we actually use it (this is the key twist).** We do **not** ask Chronos to *forecast* future glucose. We feed it the glucose window and grab its **internal representation** — the hidden "understanding" it forms of the curve — and use that as a **fingerprint** (embedding) of the curve's shape. See [Q4](#q4)/[Q10](#q10).

**How good is it?** In the TSFM world, the Chronos family is one of the **strongest and most widely used** zero-shot forecasters, and Chronos-Bolt is notable for being *both faster and more accurate* than the original. It's well-maintained by Amazon, free, and a one-line `pip install`. **BUT** — an important honesty note: Chronos was built for *forecasting*, and we're repurposing its embeddings for *prediction of a separate label (test score)*. Good forecasting skill doesn't guarantee its embeddings carry a glucose→cognition signal. If our glucose→thinking signal is genuinely weak (it may be — see [Q3](#q3)/[Q12](#q12)), no TSFM will manufacture it.

**Alternatives, and why Chronos for us:**

| Model | Who | Style (1 line) | Why we didn't start here |
|---|---|---|---|
| **Chronos / Chronos-Bolt** ✅ | Amazon | "language model for time series"; Bolt = fast patch-based | **This is our pick** (reasons below) |
| **MOMENT** | CMU | pre-trained to *produce embeddings* for many tasks (not just forecasting) | The most natural **Plan B** if Chronos embeddings underwhelm — it's *designed* for representations. The ICML paper Liuyi pointed us to already covers it, so it's a clean next experiment. |
| **TimesFM** | Google | decoder-only, patch-based forecaster | Solid; Ben used it too. Reasonable alternative, but Chronos had the cleaner embedding recipe for us. |
| **Moirai** | Salesforce | handles *many channels at once* (multivariate) | Overkill — we have **one** channel (glucose only). |
| **Lag-Llama** | open-source | forecaster using "lags" | Smaller ecosystem / less convenient embeddings. |
| **TinyTimeMixer (TTM)** | IBM | tiny, ultra-fast MLP model | An option if we ever need something featherweight. |

**Why Chronos specifically:**
1. **Continuity + a validated recipe.** The research paper Liuyi told us to study (the ICML `representations-in-tsfms` repo) uses Chronos, and labmate Ben used Chronos-Bolt. We're standing on a recipe that's already been shown to work, not inventing one.
2. **It fits raw CGM perfectly.** Real CGM data is **gappy** (missing readings) and **variable-length**. Chronos is **NaN-tolerant** (missing readings become a "mask" token — no gap-filling needed) and takes **any length** natively, and it **scales each series internally** (so we don't have to normalize). That's a lot of messy preprocessing we simply don't have to do.
3. **Fast + free + easy.** Bolt runs on CPU and flies on our GPU; weights are free on HuggingFace; the API is a couple of lines.

**My recommendation on version** (you asked me to decide — see [Q14](#q14) item 5): **standardize on `amazon/chronos-bolt-small` as the primary, and also run `amazon/chronos-bolt-base` as a sensitivity check** now that we have the GPU. Keep it **frozen** (no fine-tuning yet). *Not* "latest/biggest = best" — bigger embeddings can actually hurt on a small dataset (they overfit; see PCA in [Q13](#q13)), and Bolt has no "large" size anyway. Our code is size-agnostic, so swapping is a one-word change.

---

<a name="q2"></a>
## Q2 — The three games/scores: Grids, Symbols, Prices. What are they? Why these? What do results look like?

These come from the **ARC/PARC app** (a mobile cognitive-testing app built by our co-mentor **Dr. Hassenstab**). Kids play these short games **~5×/day for ~10 days**, which is exactly why they're perfect for this study: you get **many repeated score measurements per kid**, each with a glucose window right before it — lots of chances to link momentary glucose to momentary thinking.

Each game targets a different **cognitive domain** (a type of mental skill). Best current understanding:

| Game | What the kid does | Mental skill it measures | Score in our data |
|---|---|---|---|
| **Grids** | Remember where items were placed on a grid, then reproduce it | **Visuospatial working memory** (holding a picture "in mind") | small numeric score |
| **Symbols** | Quickly match symbols against a key | **Processing speed** (how fast you think) | small numeric score |
| **Prices** | A grocery-prices task | a memory/speed measure — **in our data it's stored inverted (higher = worse) and on a bigger scale**, which is consistent with a *latency/error-type* score (like reaction time) rather than an accuracy score | larger numeric score |

⚠️ **Honesty flag:** the code (and Mack's original) never document *exactly* what each score is (accuracy? reaction time? a composite?) or its units. Grids≈working memory and Symbols≈processing speed are safe (they match the grant's stated focus). **Prices is the uncertain one** — its inverted, larger-scale nature is my inference, not a confirmed fact. This is worth one question to Liuyi / Dr. Hassenstab ([Q14](#q14) item 3).

**"Prices is backwards."** For Grids and Symbols, higher score = better. For Prices, higher = *worse*. To keep life simple, our code **flips the sign of Prices** (negates it) so that for all three targets **"higher = better."** (Toggle: `config.PRICES_IS_INVERTED`.)

**Why "three separate models"?** Liuyi was explicit: predict **one score at a time** — a Grids model, a Symbols model, a Prices model — not one model juggling all three. They measure different skills and may relate to glucose differently, so we keep them separate.

**What do the results look like?** One number per game per session. As a *scale illustration* (from our synthetic stand-in data — real numbers will differ): Grids ≈ 0.3 ± 0.35, Symbols ≈ 1.8 ± 0.6, Prices (raw, pre-flip) ≈ 40 ± 16. The point isn't the exact values — it's that they're **continuous numbers** (hence "regression"), on different scales (which is why we standardize each one before modeling).

---

<a name="q3"></a>
## Q3 — Feature engineering vs raw-data learning: what's the difference?

Same input (a glucose curve), two philosophies for turning it into something a predictor can use.

```
                         THE SAME GLUCOSE CURVE
                 (readings every 5 min for ~2 hours)
      mg/dL
        180 |        .-.
        150 |    .-.  |  .--.            <-- this is the raw input, either way
        120 |  .-  '-'    '- .
         90 | -             '---
            +--------------------------> time (each step = 5 min)


   OLD WAY: FEATURE ENGINEERING              NEW WAY: RAW-DATA LEARNING  (ours)
   (a HUMAN decides what to measure)         (the AI decides what matters)

   curve                                     curve
     |                                         |
     v                                         v
   [human writes formulas]                   [ Chronos AI reads the whole curve ]
     mean glucose      = 132                   |
     variability (std) = 24                     v
     % time below 70   = 0.05                 [ 512-number fingerprint ]
     % time above 180  = 0.10                  (the AI's own summary — it
     recent slope      = +3.2                   captures shape, timing, trends
     ... (43 numbers total) ...                 that no one hand-coded)
     |                                          |
     v                                          v
   [ predictor ] --> score                    [ predictor ] --> score

   + Simple, interpretable.                   + Can capture subtle temporal
   - Only measures what a human                 patterns humans didn't think
     thought to measure; throws away            to measure (order, timing, shape).
     the ordering/shape of the curve.         - Fingerprint isn't human-readable.
```

- **Feature engineering** = a person hand-writes formulas that squeeze the curve down to a handful of summary numbers ("features"), e.g. average glucose, how bumpy it was, how long it was too low. The predictor only ever sees those human-chosen summaries. **Risk:** if the real signal lives in something nobody thought to measure (e.g. *the exact timing* of a dip), it's gone.
- **Raw-data learning (ours)** = don't hand-pick anything. Feed the raw sequence to a pretrained AI (Chronos) and let *it* produce the summary (the 512-number fingerprint). The AI has seen millions of time series, so it can encode shape/timing/trend patterns automatically. **Risk:** the fingerprint isn't human-interpretable, and it's higher-dimensional (needs care — see PCA in [Q13](#q13)).

That contrast — "hand-crafted features vs learned representations" — is the whole methodological point of the project. (We *used* to frame it as a head-to-head against Mack's features; per last meeting we now just develop **our** raw-data approach — see [Q12](#q12).)

---

<a name="q4"></a>
## Q4 — Embedding: what is it? What does it look like? How do we predict the score from it?

An **embedding** is **a list of numbers that represents something complex as a point in "meaning space."** When you feed the glucose window into Chronos, its output is a list of **512 numbers** — that's the embedding. Think of it as a **fingerprint of the curve's shape**: two glucose curves that "look/behave similarly" get similar fingerprints; very different curves get very different fingerprints.

**What it literally looks like** — just a vector of 512 floating-point numbers:

```
 glucose window (e.g. 45 readings)         embedding / "fingerprint" (always 512 numbers)
   [140, 138, 135, 150, 161, ... ]  --->   [ 0.12, -1.84, 0.55, 2.01, -0.09, ...,  0.77 ]
        (length varies per session)                  (exactly 512, every time)
```

Notice two things:
- The input length **varies** (some sessions have 20 readings, some 70), but the fingerprint is **always exactly 512 numbers**. That fixed size is *why* embeddings are so convenient: every session becomes a row of the same width, so a standard predictor can eat them.
- The individual numbers **have no human meaning** ("number #237 = 0.55" isn't "average glucose"). They're the AI's own internal code. We don't interpret them one by one; we let a predictor find which combinations relate to the score.

**A useful analogy:** a face-recognition system turns a photo (millions of pixels, any size) into ~128 numbers that capture "what this face is like." Same idea here: Chronos turns a glucose curve (any length) into 512 numbers that capture "what this glucose trajectory is like."

**How we predict the score from the fingerprint.** The fingerprint is just a row of 512 numbers → a **regressor** (a predictor that outputs a number) learns the mapping *"these 512 numbers → this Symbols score."* We try:
- **Arm A**: classic predictors (Ridge / SVR) that fit a math formula from the 512 numbers to the score.
- **Arm B**: a small trainable neural network ("head") that learns the same mapping more flexibly.

(See [Q11](#q11) for A vs B, and [Q10](#q10) for the mechanics of *how* Chronos produces the 512 numbers.)

---

<a name="q5"></a>
## Q5 — R² ("R-squared"): what is it, why here, what do 1 / 0 / below-0 mean? (with an example)

R² is a **report card for a number-predicting model**. It answers one question: **"Is my model better than just guessing the average every time — and by how much?"**

```
 R²  =  1  -   (total squared error of MY model)
                ---------------------------------------
                (total squared error of "always guess the average")


   1.0  | perfect — predictions exactly match reality
        |
   0.7  | strong
   0.5  | explains half the variation (already good for messy human data)
   0.2  | weak-but-real signal
        |
   0.0  |======= no better than always guessing the average =======
        |
  -0.5  | WORSE than guessing the average
  <0    | (model is actively unhelpful / overfit)
```

- **R² = 1.0** → your predictions are perfect.
- **R² = 0.0** → your model is exactly as good as ignoring the glucose entirely and always guessing the average score. **Useless but not harmful.**
- **R² < 0** → your model is **worse than guessing the average** — it's actively misleading (usually a sign of overfitting; see [Q4](#q4)/[Q13](#q13)). Yes, R² can go negative; it's not literally "r squared" here, it's "fraction of variance explained," which can be negative on held-out data.

**Why we use it here:** it's the same metric the earlier work used (so we're consistent), and it has a built-in fairness bar — **you have to beat "guess the average"** ([Q6](#q6)) just to reach 0. It makes "the model found real signal" (R² clearly > 0) unambiguous.

**Tiny worked example.** Suppose 3 test sessions had real Symbols scores of **2, 4, 9** (their average = 5).

- The **baseline** always guesses 5. Its errors: (2−5), (4−5), (9−5) = −3, −1, +4 → squared = 9, 1, 16 → **sum = 26**.
- **Our model** guesses 3, 5, 8. Its errors: −1, −1, +1 → squared = 1, 1, 1 → **sum = 3**.
- **R² = 1 − 3/26 = 0.88** → excellent, we explained most of the variation.
- If our model had guessed 5, 5, 5 (i.e. copied the baseline) → error sum 26 → R² = 1 − 26/26 = **0**.
- If our model had guessed 0, 0, 0 → errors −2,−4,−9 → squared 4,16,81 = 101 → R² = 1 − 101/26 = **−2.88** (much worse than the baseline).

We report R² as **mean ± spread across the CV folds** (e.g. `-0.21 ± 0.09`) so you see both the level and how much it wobbles between folds.

---

<a name="q6"></a>
## Q6 — Baseline: what is it?

The **baseline** is the dumbest reasonable strategy, used as the bar to clear: **"always predict the average score, ignoring the glucose entirely."** In code it's scikit-learn's `DummyRegressor(strategy="mean")`.

Why it exists: **a model is only worth anything if it beats "just guess the average."** By measuring R² *relative to* this baseline ([Q5](#q5)), we make the bar explicit — the baseline sits exactly at **R² = 0**, so:
- model R² **> 0** → it learned something real from the glucose,
- model R² **≈ 0** → it's no better than the baseline,
- model R² **< 0** → it's *worse* than the baseline (overfit / unhelpful).

Every results table lists `Baseline(mean)` as its own row so you can eyeball whether the real models actually beat it. (On our synthetic data they mostly *don't* clearly beat it — which is expected, because synthetic data has little real signal; see [Q8](#q8).)

---

<a name="q7"></a>
## Q7 — Grouped-by-subject / leakage: explain in detail

This is the single most important *fairness* rule in the project. **Leakage** = your test accidentally lets the model "cheat," so it looks smarter than it really is. The specific cheat we must prevent: **testing the model on a kid it already saw during training.**

**Why that's cheating.** Each kid has a personal baseline (some kids just score higher). If a kid's sessions appear in *both* training and testing, the model can quietly learn *"oh, this is kid #7, and kid #7 usually scores ~1.9"* and predict well — **without learning anything about glucose→thinking at all.** It would look great in testing and then **fail completely on a brand-new kid.** That's a fake result.

**The fix: group by subject** (`subid`). We split the *kids* into folds, and guarantee **a kid is never in both train and test.**

```
 20 kids, split into 5 folds. Each round trains on 16 kids, tests on 4 UNSEEN kids:

   Round 1:  TEST [k1  k2  k3  k4]   TRAIN [k5 .......................... k20]
   Round 2:  TRAIN[k1..k4]  TEST[k5 k6 k7 k8]  TRAIN[k9 ............... k20]
   Round 3:  TRAIN[k1 ..... k8]  TEST[k9 k10 k11 k12]  TRAIN[k13 ...... k20]
   Round 4:  TRAIN[k1 ........... k12]  TEST[k13 k14 k15 k16]  TRAIN[k17..k20]
   Round 5:  TRAIN[k1 ....................... k16]   TEST [k17 k18 k19 k20]

   ✅ RULE (enforced by an assert in the code): no kid appears in both TRAIN and TEST.
   We average the 5 test R²s → an honest estimate of "how well would this work
   on a NEW kid we've never seen?"  (This is called GroupKFold cross-validation.)
```

Contrast with the **wrong** way, which we keep *only* as a diagnostic:

```
 SESSION-CV (the LEAKY way): shuffle ALL sessions and split ignoring kids.
   -> the same kid's sessions land in BOTH train and test
   -> the model can "recognize the kid" -> inflated, dishonest score.
```

**The clever diagnostic use of the leaky way.** We *deliberately* run both:
- **Grouped CV** (honest, no leakage) — the real answer.
- **Session CV** (leaky, kid can appear in both) — intentionally lets baseline "cheating" through.

If **session-CV R² is clearly higher than grouped-CV R²**, that's a fingerprint of *"most of the apparent signal is just per-kid baseline, not a real glucose→thinking effect."* That exact gap is what told the earlier team the signal was baseline-dominated — and our pipeline **reproduces this diagnostic**, so we can tell "no real signal" apart from "a bug." The **within-subject knob** ([Q13](#q13)) is the follow-up that strips the baseline out to hunt for a *within-kid* effect.

(There's also **nested** CV: inside each training set we do a smaller inner CV to tune the model's settings — so tuning never peeks at the test kids either. Detail, not essential.)

---

<a name="q8"></a>
## Q8 — Synthetic data: what is it? Why use it? Is it "mock runs on fake glucose data before the real data"?

**Yes — you've got it exactly.** Synthetic data is **fake data we generate that has the *identical shape and format* as the real merged glucose file**, so we can build, run, and debug the entire pipeline **today**, before Phil/Liuyi send the real CSV. It's a "mock fuel" that lets us test-drive the whole machine.

**What "same format" means concretely.** Our generator (`data.generate_synthetic_data`) produces the *same columns and structure* the real file will have:
- ~20 fake "subjects," each with ~32–67 fake "sessions,"
- each session gets a **realistic variable-length glucose curve** (a mean-reverting wiggle around that subject's typical level, in mg/dL, 5-min spacing, with occasional missing readings — just like a real CGM),
- and three fake cognitive scores (Grids/Symbols/Prices) on the right scales, including the Prices inversion and a little random missingness.

So the pipeline can't tell it's fake — which is the point. **The day the real CSV lands, we swap `generate_synthetic_data()` for `load_real_data()` and everything else runs unchanged** (`load_real_data` already mirrors the real file's exact quirks, down to how Cohort 2 writes `nan` inside the glucose text).

**Why it's genuinely useful (not just busywork):**
1. **Build & debug now.** We caught and fixed real bugs (e.g. plain Linear regression *catastrophically overfits* the 512-d fingerprint → we learned we must use PCA/regularization) — all before touching patient data.
2. **Prove the plumbing.** The synthetic run + GPU Chronos run exit cleanly → the machine works end-to-end.
3. **It reproduces the real problem's *character*.** We built the generator so that **most of the score variance is between-subjects (baseline)**, with only a *weak* glucose effect — mimicking the real finding. And it has **two tunable signal knobs**:
   - a **between-subject** signal (raises grouped-CV R², but disappears when you center per subject),
   - a **within-subject** signal (survives centering).
   
   Setting the within-subject knob > 0 and watching centered R² jump to ~0.93 **proves our within-subject test actually detects a real within-kid signal when one exists** — so if the real data shows nothing, we know it's the data, not a broken detector.

**The one thing to never forget:** synthetic numbers are a **plumbing test, not a scientific result.** The fake "signal" is defined by simple formulas, so it can't tell us anything about real biology. Every synthetic results file is stamped with a ⚠️ warning saying exactly this.

---

<a name="q9"></a>
## Q9 — The pipeline: how does "glucose window → predicted score" actually work?

Yes — **glucose data in, cognitive score out.** Here's the whole assembly line for **one session** (one time a kid played a game):

```
 STEP 0.  ONE SESSION = the glucose readings taken BEFORE one game, + the real score.

    glucose window (raw CGM, variable length, e.g. 45 readings, some missing)
        [ 140, 138, 135, 150, 161, 158, ...,  132 ]
                       |
                       |   STEP 1.  THE FINGERPRINT STEP  (Chronos, frozen)   [Q10]
                       v
    ┌───────────────────────────────────────────────┐
    │   Amazon Chronos  (pretrained, never trained    │
    │   by us)  →  reads the curve  →  mean-pools      │
    └───────────────────────────────────────────────┘
                       |
                       v
        embedding / fingerprint:  [ 0.12, -1.84, ..., 0.77 ]   (exactly 512 numbers) [Q4]
                       |
                       |   STEP 2.  THE PREDICTOR
                       v
    ┌───────────────────────────────────────────────┐
    │   Arm A: Ridge / SVR      OR      Arm B: MLP     │   [Q11]
    │   (learns: 512 numbers  →  a score)             │
    └───────────────────────────────────────────────┘
                       |
                       v
        predicted score, e.g. Symbols = 1.83
                       |
                       |   STEP 3.  THE REPORT CARD
                       v
        compare predicted vs the REAL score, across many sessions,
        under grouped cross-validation  →  R²    [Q5, Q7]
```

Repeat for **every session** (~900+), for **each of the three games** (three separate models — [Q2](#q2)). The expensive part (Step 1, running Chronos) is done **once and cached** to disk, then reused for all three targets and every predictor variation, so experiments are fast.

**Two shortcuts baked in:**
- The **frozen encoder is cached** — we never re-run Chronos for the same windows.
- The **mock encoder** — for pure offline testing there's a fake stand-in "fingerprint" (16 simple summary numbers) so the line runs with no model download at all.

---

<a name="q10"></a>
## Q10 — The "AI fingerprint step": explain it more. What is the 512-number fingerprint?

This zooms into **Step 1** of [Q9](#q9). "Fingerprint" is just our friendly word for **embedding** ([Q4](#q4)) — a fixed list of 512 numbers that stands in for the whole glucose curve.

**Mechanically, what Chronos does to make it:**

```
  raw window:  [140, 138, 135, 150, 161, ...]   (1 series, any length, NaNs OK)
      |
      | 1) Chronos internally SCALES the series (so 100 mg/dL and 300 mg/dL
      |    series are treated on a common footing — we don't normalize ourselves)
      | 2) it splits the series into PATCHES (little chunks) and runs its
      |    pretrained transformer over them
      v
  hidden states:  a grid of shape (number_of_patches + 1, 512)
      |            e.g. (13, 512) — one 512-length row per patch, +1 summary token
      |
      | 3) we MEAN-POOL: average down the patch axis (collapse the grid to one row)
      |    (this is the exact recipe from the ICML paper Liuyi pointed us to)
      v
  fingerprint:  a single row of 512 numbers        <-- one per session
```

- **Why exactly 512?** That's the internal "width" (hidden size, `d_model`) of `chronos-bolt-small`. Different sizes give different widths — e.g. `bolt-tiny`/`mini` give 256, `bolt-base` gives more. The number is just "how many dials the model uses internally to describe a series." Bigger *can* mean richer, but also harder to fit on a small dataset (why we sometimes shrink it with PCA — [Q13](#q13)).
- **Why "mean-pool"?** Chronos gives one 512-vector *per patch* (so a grid, not a single row). To get **one** fingerprint per session, we **average across the patches**. (We can also take the last token instead — that's the "pooling" knob in [Q13](#q13).)
- **"Frozen" = we never train Chronos.** We only ever run it forward to read out the fingerprint; all *learning* happens in the small predictor after it. This is the fast, standard "zero-shot embeddings" approach (and it's what the reference paper does).
- **What the 512 numbers are:** the model's own compressed description of the curve's shape/behavior. Individually meaningless to humans; collectively a rich, fixed-size summary a predictor can learn from. (Analogy again: like the ~128 numbers a face-ID system uses to describe a face.)

We **verified this really runs**: on the GPU, `chronos-bolt-small` turned 938 synthetic windows into a `(938, 512)` fingerprint matrix, and it's cached to `.npy` so we never recompute it.

---

<a name="q11"></a>
## Q11 — What are Arm A and Arm B?

They're **two versions of the "predict the score from the fingerprint" step** ([Q9](#q9) Step 2). **Both share the exact same frozen Chronos encoder and the same fingerprints** — the *only* difference is what predicts the score on top. Think **two different heads on one shared body.**

```
                          [ glucose window ]
                                 |
                                 v
                    ┌────────────────────────────┐
                    │   Chronos  (FROZEN)          │   <- identical in both arms;
                    │   → 512-number fingerprint   │      never trained by us
                    └────────────────────────────┘
                                 |
                 ┌───────────────┴────────────────┐
                 v                                 v
        ═══ ARM A ═══                       ═══ ARM B ═══
   ┌────────────────────┐            ┌──────────────────────────┐
   │ Ridge / SVR /       │            │ a small TRAINABLE neural  │
   │ Linear regressor    │            │ network "head" (an MLP),  │
   │ (scikit-learn)      │            │ trained with early-stopping│
   │ = fits a formula    │            │ = learns from OUR data     │
   └────────────────────┘            └──────────────────────────┘
                 |                                 |
                 v                                 v
              score                             score

   • Fast; almost nothing to train.        • More flexible (can learn nonlinear
   • The quickest honest result.             combinations of the 512 numbers).
   • Uses grid-search to tune settings.    • This IS the "adapt Ben's code" task (Task 4).
```

**Arm A — frozen fingerprint + a classic predictor.**
- Predictors: **Ridge** (a line-fit with a "don't overfit" penalty), **SVR** (a flexible curve-fit), plain **Linear**, (+ XGBoost if available). Always includes the **baseline** ([Q6](#q6)).
- It's the **fastest path to an honest number** and mirrors the reference paper's approach exactly. Tunes its own knobs via inner cross-validation ([Q7](#q7)).

**Arm B — frozen fingerprint + a small trainable neural network.**
- Instead of a fixed formula, a little neural-network "head" (an MLP — a few layers) **learns** the fingerprint→score mapping on our data, using early-stopping so it doesn't overfit.
- This is **the literal task Liuyi gave you as Task 4: "adapt Ben's code."** Ben (another student) had code that fed a foundation model into a trainable head for *his* project (2 sensors, predict a category). We **rewired it** to: **1 channel (glucose), predict a number, and stripped out the parts specific to his multi-day/2-sensor setup.** It's trained fold-by-fold under the *same* grouped CV as Arm A, so its R² is directly comparable.

**Why keep both?** Arm A is the quick, robust, directly-interpretable baseline of *our* approach. Arm B is the flexible version and the explicit Ben-adaptation deliverable — and it's the door to fancier future ideas (e.g. actually fine-tuning Chronos with LoRA, or feeding multiple windows). Same backbone, two heads.

---

<a name="q12"></a>
## Q12 — What was Mack's old approach, and how do we handle it now?

**Mack** is a labmate who already attacked this exact problem the **old way** — **feature engineering** ([Q3](#q3)). He hand-wrote **43 glycemic summary features** (average glucose, variability, % time low/high, slopes, etc.) and fed them to classical models under careful grouped cross-validation.

**What he found:** essentially **no predictive signal**. Across all 43 features, several model families, and multiple CV schemes, **R² was negative for all three scores** (worse than guessing the average), classification was at chance, and the diagnostic ([Q7](#q7)) said most apparent signal was **per-kid baseline**, not a real glucose→thinking effect.

**How we *used to* relate to it:** the earlier plan framed our project as a **fair head-to-head — "do learned Chronos representations beat Mack's hand-crafted features under identical rules?"** To make that fair, we even copied his exact 43 features into our code (verified numerically identical) so both arms ran through the *same* evaluation.

**How we relate to it NOW (last meeting's decision):** 🚫 **We ignore Mack's approach.** Liuyi was clear: **don't worry about comparing to his feature approach — focus only on our own AI/TSFM approach.** So:
- The **headline deliverable is our raw-data approach on its own**: does feeding raw CGM through Chronos (Arm A / Arm B) find any honest glucose→cognition signal, and does the within-subject test reveal a within-kid effect?
- The 43 hand-crafted features stay in the code **only as an optional internal sanity-check** (a familiar reference point to confirm our numbers are in a sane range) — **not** as the point of the project, and **not** something to write up as "we beat Mack."
- We **keep the honest expectation** from his result (the signal is likely weak — [Q3](#q3) intro), because that's just true and keeps us from overselling. We're dropping the *comparison*, not the *scientific humility*.

**Practical note:** the code still *has* a `run_headtohead` command that shows both side by side — that's fine to keep as a tool, but per the new direction, when we report results we lead with the TSFM arm alone.

---

<a name="q13"></a>
## Q13 — The "knobs" to test different settings — what real question does each answer?

A **knob** = a setting we can sweep (try several values) to answer a specific scientific/engineering question. We test **one knob at a time** (hold everything else fixed) and read the grouped-CV R². Command: `python -m cgm_tsfm.run_sweep --kind all`. Here are the meaningful ones and *the question each answers*:

| Knob | Values we try | The real question it answers |
|---|---|---|
| **1. Chronos size** (`checkpoint`) | bolt-tiny, mini, small, base, (t5-small) | *"Does a bigger AI give better fingerprints — or is small enough?"* (Bigger isn't always better; more numbers can overfit a small dataset.) |
| **2. History window** (`window`) | 24, 30, 36 readings, or full (≈2h / 2.5h / 3h) | *"How much glucose history before the test actually matters?"* The grant hypothesizes ~2h — this checks it empirically. |
| **3. PCA** (`pca`) | none, 8, 16, 32, 64, 128 | *"Should we shrink the 512-number fingerprint down to its most useful few?"* **PCA** is a math trick that compresses 512 → e.g. 32 numbers, keeping the informative directions. Needed because 512 numbers on ~900 samples invites overfitting — in our tests, PCA≈16–32 **helped** and killed the numerical warnings. |
| **4. Within-subject normalization** (`targetnorm`) | none, center, zscore | ⭐ **The important one.** *"Is the signal just per-kid baseline, or is there a real within-kid effect?"* Instead of predicting the raw score, predict **how much this kid did better/worse than their own usual score.** This **strips out the baseline** ([Q7](#q7)) so any leftover predictability is a genuine glucose→thinking effect. This is our **direct test of the "it's just baseline" theory.** |
| (minor) **Pooling** | mean vs last | *"Average all patches, or just take the final summary token?"* A small technical choice for how we collapse Chronos's output to one fingerprint ([Q10](#q10)). |

You asked about "the 4 questions" — those are knobs **1–4** above (the summary listed those four; pooling is a minor fifth). ⭐ **Knob 4 is the scientifically central one**, because it directly attacks the main confound.

⚠️ One honesty caveat on knob 4: centering uses *each kid's own sessions* to compute their personal average ("oracle" centering). So it answers *"is there a within-kid signal at all?"* — **not** *"can we predict a brand-new kid,"* since for a new kid we wouldn't know their personal average yet. Same framing the earlier team used; just be precise about the claim when writing it up.

---

<a name="q14"></a>
## Q14 — What to ask Liuyi (the decisions) — updated with your answers + the calls I'm making

Here's each original open question, its **current status** given what you told me, and — where you delegated it — **the decision I'm making and why.** (The detailed versions live in `docs/04_OPEN_QUESTIONS.md`.)

**1. The real data file** — ✅ *Resolved / in progress.* Not really an open question anymore: **Phil is finishing the statistics + correlation analysis for the 20 patients, and Liuyi will send the merged glucose file after that.** Nothing for us to do but wait and be ready (we are). *Only thing worth doing:* a light check-in on timing so we can plan. **This is the sole remaining blocker to real scientific numbers.**

**2. The pre-test window (how much glucose history counts, and do we have the raw stream?)** — 🟡 *You told me to use my judgment. My decision:* treat the **full pre-clipped `Glucose_Before_Test` window as the primary input**, and also run a **2.5-hour cap (30 readings) as a sensitivity check**, since the grant's hypothesis is about ~2h before the test. I'll document whatever lookback the real arrays turn out to have. *(If, once the data arrives, we discover we need a specific/uniform lookback, the only thing we'd have to ask for is the upstream raw Dexcom stream or the merge script — I'll flag it then, not now.)*

**3. What the three scores measure + the Prices inversion** — 🟠 *You told me to decide.* My decision: **keep negating Prices** so all three read "higher = better" (it's reversible with one flag), and proceed treating **Grids ≈ working memory, Symbols ≈ processing speed, Prices ≈ a latency/error-type score** ([Q2](#q2)). *These don't block anything.* The **one low-cost question worth asking Dr. Hassenstab/Liuyi eventually** (not urgent): the exact definition + units of each score, especially **Prices** — purely to sanity-check predictions and pick sensible scaling. I'll proceed on my assumptions until told otherwise.

**4. What does "success" look like?** — ✅ *Resolved by Liuyi.* **Ignore Mack's feature approach; focus on our own TSFM approach** ([Q12](#q12)). So success = *an honest, leakage-free evaluation of the raw-CGM→Chronos approach for all three scores, plus the within-subject test of the baseline theory* — reported on its own merits, whatever the answer. We are **not** promising a big accuracy win.

**5. Which Chronos version?** — 🟡 *You told me to decide (and no, "latest = best" isn't automatic).* **My decision: standardize on `amazon/chronos-bolt-small` as the primary model, frozen, and also run `amazon/chronos-bolt-base` as a sensitivity check on the GPU.** Reasoning: Bolt is the fast, accurate, CPU/GPU-friendly line; **small** is plenty for ~900 sessions with a baseline-dominated signal (bigger embeddings mainly add overfitting risk — see PCA, [Q13](#q13)); Bolt has no "large," and T5-large would be slower for little expected gain. Our code is version-agnostic, so this is trivial to revisit. The checkpoint **sweep** ([Q13](#q13) knob 1) will confirm empirically that bigger doesn't help here.

**6. The compute/server environment (Task 1)** — ✅ *Effectively done.* The env is **built and GPU-validated on `cpsl-mds`** (Miniforge + prefix env on the SSD, torch+CUDA on an RTX 6000 Ada, real Chronos runs on GPU — see `docs/05_SERVER_SETUP.md`). **So yes, check Task 1 off**, with one small confirmation for Liuyi: *the task originally said "compute2" — if that's a specific different machine than `cpsl-mds`, tell me and I'll replicate the same setup there (it's ~20 min, fully scripted now).* Otherwise Task 1 is complete.

**7. (Later) Frozen vs fine-tuned encoder** — 🟢 *No action.* We stay **frozen** for now; only worth trying LoRA fine-tuning if the frozen approach shows promise on real data.

### 🧭 The decisions I made for you, in one place
| You delegated | My call | One-line why |
|---|---|---|
| Chronos version (item 5) | `chronos-bolt-small` primary + `bolt-base` sensitivity, **frozen** | fast, enough capacity for a small baseline-dominated dataset; bigger risks overfitting |
| Pre-test window (item 2) | full window primary + 2.5h sensitivity | matches the code default and the grant's ~2h hypothesis |
| Score definitions / Prices (item 3) | keep Prices flipped; proceed on WM/speed/latency reading | reversible, unblocks work; confirm units with Hassenstab later |
| Task 1 status (item 6) | **checked off** (GPU-validated on `cpsl-mds`) | confirm only whether "compute2" is a different box |

---

### If you remember only 7 things
1. **Goal:** predict a kid's thinking-test score from their recent blood sugar. Glucose in → score out. ([Q9](#q9))
2. **Method:** feed the *raw* glucose to an AI (Chronos) → a 512-number **fingerprint** → a small predictor. No hand-picked features. ([Q3](#q3), [Q4](#q4), [Q10](#q10))
3. **New direction (last meeting):** develop **our TSFM approach on its own** — **ignore the comparison with Mack's features.** ([Q12](#q12))
4. **Honesty:** the signal is probably weak (baseline-dominated). We deliver a **fair, leakage-free evaluation + a within-subject test**, not a promised miracle. ([Q5](#q5), [Q7](#q7), [Q13](#q13))
5. **Two arms:** Arm A (fingerprint + Ridge/SVR) and Arm B (fingerprint + small trainable net = the "adapt Ben's code" task). ([Q11](#q11))
6. **Status:** whole pipeline built, **server set up + GPU-validated** (Task 1 done), running on **synthetic stand-in data**. ([Q8](#q8))
7. **Only blocker for real numbers:** the data file — **coming after Phil finishes his analysis.** ([Q14](#q14))

---

## Where things live (so you can find stuff)
- **`PLAIN_ENGLISH_SUMMARY.md`** ← you are here (plain-English + the 14-question deep-dive).
- **`README.md`** — technical front page + all the commands.
- **`docs/`** — detailed writeups: `00_PROJECT_OVERVIEW.md` (story), `01_PIPELINE_DESIGN.md` (technical), `02_ROADMAP.md` (checklist), `04_OPEN_QUESTIONS.md` (questions), **`05_SERVER_SETUP.md` (the lab-server setup — Task 1)**, **`06_CHRONOS_INPUT_FORMAT.md` (Chronos dimensions + sample→score — Tasks 3 & 4)**.
- **`cgm_tsfm/`** — the actual code (the assembly line).
- **`results/`** — output tables + a `results/README.md` that explains them.
