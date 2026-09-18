# 17 · Catch-up — the project and the new slides, in plain words

> Written 2026-09-18 because you had been away from the project for a while. No jargon
> that isn't defined on the spot. Read §1 and §2 and you can present the deck.
> The deck this explains is `Progress_Incremental_Value.pptx` (6 slides).

---

## 1. The whole project, in six sentences

1. Kids with Type 1 Diabetes wear a sensor that records their **glucose every 5 minutes**.
2. A few times a day they play **three quick games on a phone** and get three scores.
3. **The question:** using only the glucose from the 2 hours before a game, can we guess the score?
4. For months the answer was **no** — we could never beat a *simple guess* (see §2.1).
5. **This week** I found something that *can* beat the simple guess: **the kid's own recent scores**.
6. And I then showed, carefully, that **glucose contributed none of that improvement.**

That last sentence is the whole point of the new deck. The number went up, and I can prove
glucose is not the reason.

---

## 2. The five ideas you need

### 2.1 — The "simple guess", and what R² actually is

The **simple guess** is a predictor that ignores everything and always says the average score.
It's the bar we have to beat. If we can't beat it, our fancy model is worthless.

**R² is just: how much of the simple guess's error did we remove?**

Tiny example. Five rounds of the speed game, real times in seconds:

```
real scores:     1.5   2.0   1.8   2.5   1.2      average = 1.8
```

*The simple guess says 1.8 every time.* Its misses, squared and added up:

```
(1.5−1.8)² + (2.0−1.8)² + (1.8−1.8)² + (2.5−1.8)² + (1.2−1.8)²
   0.09    +    0.04    +    0.00    +    0.49    +    0.36      =  0.98
```

*Our model guesses* 1.6, 1.9, 1.8, 2.2, 1.4. Its misses, squared and added up:

```
   0.01    +    0.01    +    0.00    +    0.09    +    0.04      =  0.15
```

```
R² = 1 −  (our error / simple-guess error)  =  1 − (0.15 / 0.98)  =  +0.85
```

So **R² = +0.85 means we removed 85% of the simple guess's error.**

| R² | What it means |
|---|---|
| **1.0** | perfect — we removed all the error |
| **+0.41** | we removed 41% of it ← *our new speed-game result* |
| **0** | we removed nothing; exactly as good as the simple guess |
| **negative** | **we made it worse than the simple guess** |

That last row is the one to remember. Our old numbers were −0.33, −0.10, −0.02. Negative isn't
mysterious or broken — it just means *worse than always saying the average.*

### 2.2 — Why our old answer was always "no" (slide 2)

Picture two kids' speed-game times across 10 days:

```
kid A:  ●    ●  ●   ●●   ●    ●  ●●   ●        (roughly 1.5 – 2.5 seconds)
kid B:    ●  ●●   ●    ●   ● ●    ●  ●         (roughly 1.4 – 2.4 seconds)
         └──────────── they overlap almost completely ────────────┘
```

Two things are true here:

- **Kid A and kid B are not very different from each other.**
- **Each kid bounces around a lot from round to round.**

We measured our model with a rule that says: *predict a kid you have never met.* But if kids
barely differ from each other, there's almost nothing to win by telling them apart. The real
variation — the good rounds and bad rounds — is **inside** each kid.

The bars on slide 2 measure exactly this. For the speed game, **62%** of all the variation is one
kid changing from round to round, and only 38% is one kid being different from another. For the
other two games it's even more lopsided: **85%** and **92%**.

**And 24 glucose readings can't tell the model which kid it's looking at, or how that kid has been
doing lately.** So the model had no choice but to guess near the overall average for everybody —
which is the simple guess. That's why we kept landing at or below zero.

### 2.3 — What I added (slide 3)

Three extra numbers per round, all about **that kid's own earlier rounds**:

1. their score on their **previous** round
2. the **average** of all their earlier rounds
3. **how many** earlier rounds there were (so the model knows how trustworthy #2 is)

**Why this is allowed, and not cheating.** For a round on Tuesday at 3pm we use *only* that kid's
rounds from before Tuesday 3pm. Never a future round. Never another kid's scores. It's information
a real app would already have sitting in its database.

*Analogy:* guessing how many points a basketball player scores tonight. Knowing "they're an NBA
player" barely helps. Knowing "they scored 30, 28 and 31 in their last three games" helps a lot.
We were only ever giving the model the first kind of information.

### 2.4 — The result (slide 4), and what ρ = 0.70 means

| Game | before (glucose only) | after (kid's own recent scores) |
|---|--:|--:|
| Speed (Symbols) | −0.326 | **+0.413** |
| Memory (Grids) | −0.097 | +0.039 |
| Prices | −0.020 | +0.037 |

The speed game went from *worse than the simple guess* to **removing 41% of its error.** First
positive number this project has ever produced.

**ρ = +0.70** ("rho") is a second, easier-to-picture measure. Line up a kid's rounds from slowest
to fastest **by our prediction**, then line them up again **by their real score**, and compare the
two orderings. 0 means the orderings are unrelated; 1 means they match perfectly. **0.70 means
they largely agree.** So we're not just close on average — we're getting the order of a kid's good
and bad rounds mostly right.

### 2.5 — The most important idea: "does glucose add anything?" (slide 5)

Now the honest question. The number went up — but **was it glucose?**

You can't answer this by dumping everything into one model, because the 3 useful numbers get
drowned out by the 512 numbers describing the glucose curve. *(Chronos turns the 24 glucose
readings into 512 numbers that describe the shape of the curve — that's what "the glucose curve"
means in the deck.)* It's like hiding 3 real clues in a stack of 515 pages: the reader gets lost
in the 512.

So I did it in **two steps** instead — this is the "leftover" or *residual* test:

```
STEP 1   Use only the kid's own history.        real score 2.4  →  we guess 2.2
                                                 LEFTOVER = 0.2   ← history couldn't explain this

STEP 2   Now hand that leftover to glucose.     "Can you predict the 0.2?"
                                                 If glucose knows something history doesn't,
                                                 it should be able to. 
```

Do that for every round, inside the test properly. **Result: glucose predicted the leftovers no
better than guessing zero.** Its contribution was **−0.011** on the speed game, −0.011 on memory,
−0.010 on prices — slightly *negative*, which is simply the cost of fitting 512 numbers that
carry no information.

**Say it like this:** *"I first let the kid's own history explain what it can. Then I asked glucose
to explain what was left over. It explained none of it."*

I also checked it a second way — squeezing the 512 numbers down to 8 first, so they couldn't be
drowned out — and got the same answer. That's the "cross-checked" line at the bottom of slide 5.

---

## 3. Slide 6 — two extra checks

### 3.1 Why the memory game needed a different question

**31% of memory-game rounds are perfect** (the kid places all six items exactly right, score = 0).
When a third of your answers are the same number sitting on a hard floor, asking "guess the exact
score" is the wrong question, and R² undersells what's findable.

So I asked a yes/no question instead: **was this round perfect or not?** The score for a yes/no
question is **AUROC**:

> Pick one perfect round and one imperfect round at random. How often does the model give the
> perfect one a higher chance of being perfect? **0.5 = a coin flip. 1.0 = always right.**

| Using… | AUROC |
|---|--:|
| the kid's own earlier results | 0.579 |
| plain glucose numbers | 0.479 |
| the glucose curve | 0.447 |

Glucose is **at or just below a coin flip**. This matters because it closes an objection someone
could raise: *"your maths was wrong for the memory game."* We asked it the right way, with the
right score, and glucose still found nothing.

### 3.2 The three honest limits — say these before you're asked

1. **The new model needs at least one earlier round from that kid.** It can't score a brand-new
   kid cold. So it's a *tracking* tool, not a *screening* tool.
2. **Only the speed game is solid.** The memory and prices gains are small (+0.04) and they
   disappear under a stricter test. Don't oversell them.
3. **A test I expected to help made things worse.** Normally we split the 20 kids into 5 groups of
   4: train on 16 kids, test on 4, repeat. I tried training on 19 and testing on **1**, twenty
   times over, expecting the numbers to steady. They got much worse — because with only one kid in
   the test set, everything is judged against *that single kid's* own average, and if that kid is
   unusual the number swings wildly. So I went back to groups of 4 and reported that this didn't
   work.

**Volunteering #3 is worth more than any of the results.** It's the difference between checking
your work and hunting for the best-looking number.

---

## 4. Words that appear on the slides, translated

| On the slide | Plain meaning |
|---|---|
| **R²** | how much of the simple guess's error we removed (§2.1) |
| **baseline / mean-predictor / "the simple guess"** | always predict the average score, ignore everything |
| **folds** | we split the 20 kids into 5 groups and always test on kids the model never trained on |
| **512 columns / the glucose curve** | Chronos turns 24 glucose readings into 512 numbers describing the curve's shape |
| **Chronos** | Amazon's pre-trained time-series model. We never train it — we just feed glucose in and take the 512 numbers out |
| **Ridge** | the simple statistical model that turns those numbers into one predicted score |
| **residual / leftover** | the part of the real score the first model couldn't explain |
| **ρ (rho)** | do our predictions put a kid's rounds in the right order? 0 = no, 1 = perfectly |
| **AUROC** | score for a yes/no question. 0.5 = coin flip |
| **leave-one-child-out** | train on 19 kids, test on 1, repeat 20 times |
| **sessions** | one round of the three games by one kid. We have 916 |

---

## 5. The three sentences to have memorised

> **1.** *"We were measuring whether glucose can tell one child from another — but children barely
> differ from each other. Most of the variation is one child having a good round or a bad round."*
>
> **2.** *"So I gave the model that child's own earlier scores. The speed game went from worse than
> guessing to explaining 41% of the variation — the first positive result in this project."*
>
> **3.** *"Then I tested whether glucose added anything on top of that. It added −0.01. So the
> finding hasn't changed — glucose still tells us nothing — but now I can prove the pipeline is
> capable of producing a positive number, which is a much stronger position than before."*

Full numbers and how each was computed: [`../../results/incremental_value_real.md`](../../results/incremental_value_real.md).
