# The Bootcamp — ML/DL from scratch, welded to this project

*Built 2026-07-30 for Brian Zhou. Everything here was recomputed from the actual CSVs, the actual code, and the actual installed `chronos` package — not remembered, not copied from the older documents in this repo. Where a number here disagrees with an older file, this set is the newer one.*

---

## Why this exists

Liuyi's feedback was that you couldn't explain early stopping, epochs, normalization, why a transformer, or how the MLP head works — and that he felt he was being asked to check AI output you hadn't checked yourself. Those are two different problems and they need two different fixes:

- **The knowledge gap** → the chapters below, from zero, with this project's real numbers in every example.
- **The trust problem** → `00_SAY_IT_SIMPLY.md`. Read it first and last. Learning the material is not enough if you then recite polished sentences at him; that is what triggered the reaction. The goal is to know it well enough to say it roughly, in your own voice.

**The single most useful thing in here** is `15_FINDINGS_TO_REPORT.md`. This week's work turned up a real bug in the embedding code, two factual errors in our own documentation, and a measurement showing the 512-number representation adds nothing over a single number. Leading a meeting with your own corrections is worth more than any amount of fluency.

---

## The files

**Read these first — they frame everything else**

| File | What it's for |
|---|---|
| **`00_SAY_IT_SIMPLY.md`** | How to *talk* about the work. The banned-words table, the five explanations he has already asked for, and what to say when you don't know. |
| **`01_FACT_SHEET.md`** | Every number in the project with its derivation. The thing to memorize. |
| **`15_FINDINGS_TO_REPORT.md`** | The bug, the two doc errors, and the one-number-vs-512 result. Your opening material. |

**The curriculum, in order**

| File | Covers |
|---|---|
| `02_ML_FROM_ZERO.md` | Supervised learning, X and y, train/validation/test, overfitting and underfitting, bias–variance, parameters vs hyperparameters, regularization as one family, loss vs metric |
| `03_PREPROCESSING.md` | **Normalization — why data must be processed before training.** All four normalizations in our pipeline, z-scoring by hand, leakage, LayerNorm vs BatchNorm vs StandardScaler |
| `04_LINEAR_MODELS.md` | Least squares, why 512 features on 764 rows breaks, Ridge and alpha, Lasso, SVR and the kernel trick, PCA and how it can silently delete a feature you need |
| `05_EVALUATION.md` | **The "predict the average" baseline**, MSE/RMSE/MAE/R² by hand, why R² goes negative, grouped cross-validation, nested cross-validation, many-looks |
| `06_DEEP_LEARNING_CORE.md` | **Epochs, batches, steps, early stopping** — plus neurons, activations, gradients, backpropagation, gradient descent, AdamW, dropout, LayerNorm, weight decay. The biggest chapter, because it was the biggest gap. |
| `07_TRANSFORMERS.md` | **Why a transformer and not an LSTM**, attention worked out by hand with real arithmetic, multi-head, residuals, how the model knows the order |
| `08_CHRONOS.md` | **How Chronos works under the hood** — patches, tokens, instance normalization, the `[REG]` token, the exact shape formula, all five checkpoints, which 41.9% of the model we actually use |
| `09_THE_TWO_ARMS.md` | **How the MLP head works.** Arm A vs Arm B line by line, every parameter counted, why Arm B came last, and four honest problems to disclose |
| `10_RIGOR_AND_STATS.md` | The shuffle test, p-values, the known-answer check, what this study could have detected, sessions vs participants — all in plain words, since he flagged this vocabulary |
| `11_WHAT_THE_SCORES_ARE.md` | What Grids, Symbols and Prices actually measure (Grids is *proven* from the data), the 31% floor, and a correction to our within/between-participant claim |

**Practice**

| File | What it's for |
|---|---|
| `12_EXAM_DRILL.md` | ~90 questions with answers, marked **[HE ASKED THIS]**, **[core]**, **[hard]**, **[trap]** |
| `13_PROBLEM_SETS.md` | 9 pen-and-paper problem sets with full worked solutions, plus a 15-question final exam |
| `html/flashcards.html` | ~130 flashcards in 10 decks, with progress saved locally. Open by double-clicking. |
| `html/quiz.html` | Self-test with explanations and a weak-topic report pointing at the right chapter |
| `14_PRESENTATION_PLAN.md` | The 40-minute talk, slide by slide, weighted to "result + rigor" the way he asked |

---

## A seven-day plan

Roughly 3–4 hours a day. Adjust, but keep the order — later chapters assume earlier ones.

**Day 1 — Orientation and numbers**
`00_SAY_IT_SIMPLY.md` → `01_FACT_SHEET.md` → `15_FINDINGS_TO_REPORT.md`.
*Goal:* the ten numbers in §K of the fact sheet, cold. Do `13_PROBLEM_SETS.md` PS1.

**Day 2 — Machine learning basics**
`02_ML_FROM_ZERO.md`. Then PS3 (metrics by hand).
*Goal:* explain overfitting using our own R² ≈ −3 example without notes.

**Day 3 — Preprocessing and linear models**
`03_PREPROCESSING.md` → `04_LINEAR_MODELS.md`. Then PS2 and PS5.
*Goal:* answer "what is normalization and why process data before training" in three concrete reasons.

**Day 4 — Evaluation**
`05_EVALUATION.md`. Then PS4.
*Goal:* explain the mean baseline in plain words, and explain why **both** it and our model score negative. This is the single most important explanation in the whole talk.

**Day 5 — Deep learning**
`06_DEEP_LEARNING_CORE.md` — the long one. Then PS6.
*Goal:* epoch, batch, step, early stopping, patience, dropout — all without notes, with our real arithmetic (612 rows, batch 64, 10 steps per epoch).

**Day 6 — Transformers and Chronos**
`07_TRANSFORMERS.md` → `08_CHRONOS.md` → `09_THE_TWO_ARMS.md`. Then PS7.
*Goal:* draw the shape walkthrough from memory (he asked twice), and count the head's 132,609 parameters layer by layer.

**Day 7 — Rigor, drill, rehearse**
`10_RIGOR_AND_STATS.md` → `11_WHAT_THE_SCORES_ARE.md` → all of `12_EXAM_DRILL.md` → `14_PRESENTATION_PLAN.md`.
*Goal:* every **[HE ASKED THIS]** and every **[trap]** answered cold. Then give the talk out loud, timed.

**Every day, ten minutes:** `html/flashcards.html`. Spaced repetition is what makes numbers stick.

---

## What "ready" looks like

- [ ] The ten numbers in `01_FACT_SHEET.md` §K, from memory.
- [ ] Draw the pipeline with shapes at every stage on a blank whiteboard.
- [ ] Define, unprompted: epoch, batch, step, patch, token, early stopping, normalization, the mean baseline.
- [ ] Explain why R² is negative **and** why the mean baseline is also negative.
- [ ] Explain the padding bug, what you measured, and why the conclusion held.
- [ ] Answer every **[HE ASKED THIS]** question in `12_EXAM_DRILL.md`.
- [ ] Score above 80% on `html/quiz.html`.
- [ ] Say the three summary sentences in `00_SAY_IT_SIMPLY.md` §6 in your own words, not the ones written there.
- [ ] Name the five things you genuinely don't know (`12_EXAM_DRILL.md`, the "I don't know" list).

---

## The three corrections these files make to the existing repo

Worth knowing, because our older documents contradict each other and someone reading them will notice:

1. **Chronos-Bolt does instance normalization, not mean-scaling.** It subtracts each series' mean *and* divides by its standard deviation. Several files describe Chronos-**T5**'s scheme instead. The consequence is real: our features are largely blind to absolute glucose level, which is what "hypo" and "hyper" are defined by.
2. **Within-participant variance dominates, not between.** 62–92% of each score's variation is one child varying session to session. Our older notes said the opposite. This makes the flat result *stronger*, since session-to-session is where a glucose effect would live.
3. **`pooling="mean"` averaged over padding, so embeddings depended on batch composition.** 90% of sessions were affected. The `[REG]` token is immune. **This has since been FIXED in the code (2026-08-01) and every result regenerated** — the conclusion did not change.

> ### 🔧 Pipeline fixed and results regenerated — 2026-08-01
>
> `cgm_tsfm/encoders.py` and `ben_adapter/model.py` now bucket windows by token count before encoding, so no window is padded to match a longer batch-mate. Verified **bit-identical** to encoding each window alone (max difference 2.4e-7) and invariant to batch size and input order. The embedding cache key is versioned (`v2`) so stale files cannot be silently reused, and a runtime assertion fails loudly on any geometry surprise.
>
> **Current headline numbers** (grids / symbols / prices, grouped CV):
>
> | | before | **after** |
> |---|---|---|
> | Arm A | −0.089 / −0.056 / −0.012 | **−0.114 / −0.052 / −0.012** |
> | Arm B | −0.218 / −0.281 / −0.168 | **−0.313 / −0.412 / −0.360** |
> | Centered | −0.018 / −0.022 / −0.002 | **−0.027 / −0.028 / −0.001** |
>
> Everything is still at or below zero. Arm B's larger drop was checked across 3 seeds and is real, not noise — the buggy pooling had been accidentally smoothing the features, which flattered a 132,609-parameter head. Pre-fix outputs are archived in `results/archive_prebugfix_2026-07-07/`.
>
> Chapters 02–14 carry a banner where their worked examples still quote the pre-fix figures. **`01_FACT_SHEET.md` §I and `results/` are authoritative.**

Also worth knowing: **XGBoost is in the code but never ran** (not installed), and **`RANDOM_STATE = 42` does not control the outer folds** (`GroupKFold` doesn't shuffle).

---

## How to read the chapters

Each one has the same shape:

- a short "what this chapter buys you" box at the top
- plain-words definitions, then a worked numeric example on **our** data
- ASCII diagrams you can redraw on a whiteboard
- **"Say it in your own words"** — a spoken sentence per section
- **"Drill"** blocks in collapsible `<details>` so you can self-test before looking
- a numbered summary at the end

The "Say it in your own words" lines are the ones that matter most. They are written to be *spoken*, not read. Use them as a starting point and then say it your own way — the point is not to memorize my sentence, it's to have one of your own.

---

## Related files elsewhere in this repo

- `docs/understanding/` — the earlier from-scratch course, focused on walking through the code file by file. Still useful; predates the corrections above.
- `results/README.md` — the narrative of the real results.
- `docs/06_CHRONOS_INPUT_FORMAT.md` — the earlier shapes-and-dimensions writeup. Good, but see `08_CHRONOS.md` for the verified version and the `[REG]`-token correction.
- `PLAIN_ENGLISH_SUMMARY.md` — **stale.** It says the pipeline is still on synthetic data and cites a `(938, 512)` matrix. The real results are in and the matrix is `(956, 512)`.

---

## One last thing

You are not walking in empty-handed. You found a real bug in your own pipeline this week, measured whether it changed the answer, and it didn't. You worked out what the Grids score actually is from first principles. You corrected two errors in your own documentation, and you established that a single number does as well as all 512.

That is what understanding looks like from the outside — not fluency, but finding the thing that was wrong and knowing what it did and didn't affect. Lead with it.
