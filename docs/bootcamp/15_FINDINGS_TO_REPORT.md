# 15 · Three things I found this week — and what they change

> **This is your opening material.** Nobody who is faking understanding walks into a meeting and says *"I found a bug in my own pipeline, here is what I measured, and here is why the conclusion still holds."* That sentence is worth more than any amount of polished explanation.
>
> Everything below was measured on 2026-07-30 against the real data. The commands are recorded so any of it can be reproduced.

> ## ✅ STATUS: the bug is FIXED and every result has been regenerated (2026-08-01)
>
> This page originally described a bug awaiting a fix. It has since been repaired in the code and the entire results set was rebuilt from corrected embeddings.
>
> **What changed in the code**
> - `cgm_tsfm/encoders.py` — windows are now **grouped by token count** before being handed to `embed()`, so no window is ever padded out to match a longer one in its batch. Verified **bit-identical** to encoding each window alone (max difference 2.4e-7, i.e. float32 rounding), and invariant to batch size (1 / 8 / 64 / 256) and to input order.
> - `cgm_tsfm/ben_adapter/model.py` — Arm B's `ChronosTorchEmbedder` had the **same** bug; fixed via the shared helpers, so the logic lives in one place.
> - `cgm_tsfm/config.py` — pooling options documented: `"mean"`, `"mean_patches"`, `"reg"`/`"last"` (the last two select the summary token, *not* the most recent patch).
> - The embedding cache key now carries a recipe version (**`v2`**). Without this, the old buggy `.npy` files would have been silently reused and "regenerating" would have changed nothing.
> - A runtime `assert` checks the returned token count against the expected bucket, so any future geometry surprise fails loudly instead of quietly corrupting features.
>
> **The headline outcome is unchanged: every result is still at or below zero.** Pre-fix files are archived in `results/archive_prebugfix_2026-07-07/`.
>
> Current numbers live in `01_FACT_SHEET.md` §I and in `results/`. Section 1 below is now the *history* of the bug — still exactly what you should present, because the finding and the measurement are the point.

---

## Finding 1 — a real bug: the embeddings depended on which sessions shared a batch

### What the code does

`cgm_tsfm/encoders.py:97-112` processes sessions in batches of 32 and pools like this:

```python
emb, _ = self.pipeline.embed(batch)   # (B, tokens, 512)
pooled = emb.mean(dim=1)              # average over the token axis
```

Sessions in a batch have very different lengths — ours run from 3 readings to 288. Chronos handles that by **left-padding every series in the batch with NaN out to the longest one**. So in a batch containing a 288-reading session, a 13-reading session gets padded out to 288, which becomes 18 patches plus one extra token = 19 token positions, of which only about one holds real glucose.

`emb.mean(dim=1)` then averages **all 19 positions**, including the ~17 that correspond to nothing but padding.

### The measurement

I embedded the same session two ways — alone, and in a batch alongside longer sessions — and compared the results:

| Session length | max abs difference | relative size of change | cosine similarity |
|---|--:|--:|--:|
| 13 readings | 1.190 | 188% | **0.219** |
| 45 readings | 0.633 | 141% | **0.396** |
| 288 readings (the longest) | 0.000 | 0% | 1.000 |

A cosine similarity of 0.22 means the two vectors point in almost unrelated directions. **These are not the same features.**

Across all 956 real sessions, comparing the cached embeddings against correctly-pooled ones:

| Window length | Sessions | Median cosine similarity |
|---|--:|--:|
| under 16 readings | 22 | 0.211 |
| 16–32 | 359 | 0.289 |
| 32–64 | 274 | 0.370 |
| 64–128 | 73 | 0.491 |
| 128–288 | 227 | 0.868 |
| 288 (longest) | 1 | 1.000 |

**861 of 956 sessions (90%) had a cosine similarity below 0.9.** The median was **0.37**. I confirmed the cached file `.cache/embeddings/chronos__amazon_chronos-bolt-small__mean__4b2c75e042223011.npy` matches the buggy recipe exactly (difference 0.00e+00), so this is what produced every headline number.

### Why it happens, mechanically

Chronos does two things correctly that hid the problem:

1. The padding is **NaN**, and the internal normalization uses NaN-aware averages, so `loc` and `scale` are computed correctly from the real readings only. I verified this: a batch of lengths [20, 71, 288] returns `loc = [9.5, 35.0, 143.5]`, which are the true per-series means.
2. Padded patches are **excluded from attention** (`chronos_bolt.py:303`: a patch is attended to only if at least one point in it was observed).

So the model itself behaves properly. But being excluded from attention does not mean those positions produce *no output* — the encoder still emits a hidden state at every position. Our unmasked average then folds those meaningless positions into the feature vector, and the more padding a session got, the more diluted its features became.

### The one-line diagnosis

> *"Chronos pads short sessions to match the longest one in the batch. Chronos itself ignores the padding, but our pooling step averaged over it anyway. So a session's features depended on which other sessions happened to be in its batch — which is not a property features are allowed to have."*

### The fix

Two options, both verified:

**Option A — the `[REG]` token.** Chronos appends one extra token at the end of every sequence, a summary slot in the style of BERT's `[CLS]`. It is always at index `-1`, it is always attended to, and I verified it is **bit-identical** whether a session is embedded alone or in a mixed batch (difference exactly 0.000 for all three test lengths). Setting `pooling="last"` already selects it.

> Worth knowing: `pooling="last"` in our config does **not** mean "the last time step." It means the `[REG]` token. Our docs describe it as last-token pooling, which reads as the final patch. It isn't.

**Option B — masked mean.** Average only over the token positions that contain real data, ignoring padded ones.

**Option C — batch size 1.** Correct but slow, and unnecessary.

### Does it change the conclusion? No.

This is the part that matters. I recomputed everything three ways under the identical protocol (5-fold grouped cross-validation by participant, best of Ridge/SVR with per-fold tuning):

| Representation | grids | symbols | prices |
|---|--:|--:|--:|
| A. current, batched mean **(buggy)** | −0.089 ± 0.123 | −0.056 ± 0.060 | −0.012 ± 0.010 |
| B. corrected mean, per session | −0.114 ± 0.119 | −0.052 ± 0.049 | −0.012 ± 0.010 |
| C. `[REG]` token (batch-safe) | −0.114 ± 0.123 | −0.053 ± 0.053 | −0.013 ± 0.008 |

**Every value is still at or below zero.** Fixing the bug makes grids slightly *worse* and leaves symbols and prices unchanged. So the bug was, if anything, flattering the results a little.

### Arm B moved a lot — and it is the fix, not luck

The full regeneration showed Arm B dropping much further than Arm A: −0.218 → −0.313 on grids, −0.168 → −0.360 on prices. Before attributing that to the fix, I checked whether Arm B is just noisy between runs (it retrains 15 networks with random initialization and shuffling). Three seeds on each embedding set:

| | grids | symbols | prices |
|---|--:|--:|--:|
| Old (buggy) embeddings | −0.218 ± 0.018 | −0.257 ± 0.054 | −0.144 ± 0.005 |
| **Corrected embeddings** | **−0.323 ± 0.011** | **−0.369 ± 0.054** | **−0.341 ± 0.041** |

The grids gap (≈0.105) is about **6× the seed spread**, and prices (≈0.197) is far larger still. So the change is genuine.

**Why does fixing a bug make a model worse? Have this answer ready — it sounds damning and isn't.** The buggy pooling averaged each session's real tokens together with padding positions, which blurred the sessions toward one another. That blurring acted as an accidental smoother. A head with 132,609 parameters trained on ~612 rows *benefits* from blurred inputs, because there is less genuine per-session detail available to memorize. Hand it sharper, correctly-computed features and it overfits harder.

So this is not a regression — it is the capacity problem becoming more visible now that the features are right. It strengthens rather than weakens the Arm A vs Arm B story: **the more flexible model is punished more, and correcting the input made that clearer.**

**How to say it:**

> *"I found a bug in how we pooled the features, and I want to lead with it. It was real — 90% of sessions had features that changed substantially once I fixed it. But I re-ran the whole evaluation with the fix and the answer didn't change: still no better than guessing the average, on all three tests. So the finding stands, and now it stands on correct features."*

---

## Finding 2 — the model is blind to absolute glucose level, and we were throwing away the two numbers that carry it

### The correction

Several documents in this repo say Chronos "mean-scales each series internally (divides by its own average magnitude)." **That is Chronos-T5's scheme, not the model we use.** Chronos-**Bolt** uses **instance normalization** (`chronos_bolt.py:95-134`):

```
loc   = NaN-aware arithmetic mean over time
scale = population standard deviation (ddof = 0)
output = (x − loc) / scale
```

It **subtracts the mean and divides by the standard deviation**. I verified the statistic numerically: for `x = [100, 120, 90, 150, 80]`, `loc = 108.0` and `scale = 24.819347`, which equals `x.std(ddof=0)` exactly and is *not* `std(ddof=1) = 27.748875`.

### Why this matters scientifically

Because both the level and the spread are divided out, the embedding is largely **blind to how high or low the glucose actually was, and to how much it actually swung**. It sees the *shape*, in units of that session's own standard deviation.

That is a problem for this project specifically, because the clinical hypothesis is about hypoglycemia and hyperglycemia — which are defined by **absolute** thresholds (below 70, above 180, above 250 mg/dL). Our representation has had exactly that information normalized away.

This explains a result that otherwise looks suspicious. Asking the same pipeline to predict properties of the glucose itself:

| Target | R² |
|---|--:|
| Glucose variability (standard deviation) | **0.462** |
| Mean glucose | **0.070** |
| Fraction of time above 180 | 0.075 |

If someone challenges *"if your embedding is any good, why can't it recover the mean glucose?"* — the answer is that the mean is subtracted off before the encoder ever sees the data. It is not a defect; it is the design.

### The two numbers we discard

`embed()` returns **two** things:

```python
emb, _ = self.pipeline.embed(batch)   # encoders.py:104 — the second value is thrown away
```

That discarded second value is `(loc, scale)` — precisely the mean and standard deviation that were removed. We had the level information in hand and dropped it.

### I tested adding them back. It does not rescue the result.

| Representation | grids | symbols | prices |
|---|--:|--:|--:|
| corrected mean, 512 numbers | −0.114 | −0.052 | −0.012 |
| corrected mean + `loc`, `scale` (514) | −0.106 | −0.052 | −0.012 |
| `[REG]` + `loc`, `scale` (514) | −0.109 | −0.052 | −0.013 |

Still at or below zero. And the reassuring cross-check: the 43 hand-built features **do** encode absolute level (mean, time-in-range, min, max) and they were flat too (−0.065 / −0.038 / −0.009). So "no relationship" holds for level-aware representations as well — the blindness is a genuine limitation of our representation, but it is not the reason for the result.

**How to say it:**

> *"I had a factual error in my own notes. Chronos-Bolt doesn't just rescale each series, it subtracts the mean and divides by the standard deviation. That means our features are mostly blind to how high or low the glucose actually was — which is awkward, because hypo and hyper are defined by absolute numbers. Chronos actually hands those two numbers back and we were discarding them. I added them back in and it made no difference. And the older hand-built features, which do keep absolute level, were also flat. So the limitation is real but it isn't what's driving the answer."*

---

## Finding 3 — the 512 numbers add nothing over a single number

This is the most uncomfortable finding, and the most useful.

Same protocol throughout (grouped 5-fold, Ridge with per-fold alpha tuning):

| Representation | Features | grids | symbols | prices |
|---|--:|--:|--:|--:|
| Chronos, corrected | 512 | −0.114 | −0.093 | −0.069 |
| Mean + standard deviation of glucose | **2** | −0.081 | −0.051 | −0.003 |
| **Mean glucose only** | **1** | **−0.075** | **−0.050** | **−0.004** |
| Hand-built glycemic features | 43 | −0.065 | −0.038 | −0.009 |

Reading it honestly: **one number — the average glucose over the window — does as well as, or slightly better than, all 512 Chronos numbers.** Everything is still at or below zero, so nobody "wins"; the point is that the elaborate representation is not buying anything.

That is not a failure of the work. It is a *measurement*, and it is the kind of measurement that tells you where the ceiling is. The reason the simpler representations look marginally better is that with ~764 training rows, 512 columns give a model far more room to fit noise than 1 column does. Fewer features, less overfitting, R² closer to zero.

**How to say it:**

> *"I ran a check I hadn't run before: how well does a single number do — just the average glucose — compared to all 512 from Chronos? The answer is the same or slightly better. Nothing beats guessing the average either way, but it tells me the 512-number representation isn't adding anything here. Which is consistent with everything else: the raw relationship between glucose and score in this data is about 0.85% of the variance at best, so there isn't much for a richer representation to find."*

---

## A fourth, smaller thing: PCA was silently discarding the features I added

While testing Finding 2 I hit something worth knowing, because it is a classic trap.

The check on glucose properties uses `StandardScaler → PCA(32) → Ridge(alpha=10)` (`run_rigor.py:125`). When I appended `loc` and `scale` to the 512 features and asked it to predict mean glucose — which is *literally one of the appended columns* — I got R² = 0.085. It should have been 1.0.

The cause: **PCA is unsupervised.** It keeps the 32 directions with the most variance and never looks at what you are predicting. Two appended columns among 512 barely move the variance, so PCA threw them out.

| Setup | R² predicting mean glucose |
|---|--:|
| `[loc, scale]` only, no PCA | **1.0000** |
| `[loc, scale]` only, PCA(2) | 1.0000 |
| 512 + `[loc, scale]`, no PCA | **1.0000** |
| 512 + `[loc, scale]`, **PCA(32)** | **0.0853** |

A good one-sentence lesson: *"PCA keeps the directions with the biggest spread, not the ones that are useful. If you add an important feature to a large pile of unimportant ones and then run PCA, you can lose it."*

---

## What to do about it — proposed next steps

Ordered by cost. Notice that these all sit inside the current scope (glucose only, Chronos only), which is where Liuyi wants to stay until the pipeline is fully understood.

1. ~~**Fix the pooling.**~~ ✅ **Done 2026-08-01.** Windows are now bucketed by token count so batches are padding-homogeneous; verified bit-identical to per-window encoding; the same bug in Arm B's embedder is fixed; the cache key is versioned so stale embeddings cannot be reused; all five result files regenerated and the pre-fix set archived.
2. **Keep `loc` and `scale`.** Two extra columns, free, and they restore the absolute-level information the clinical hypothesis is actually about. Measured as not sufficient on its own, but it removes an obvious criticism. **Still open.**
3. **Report the single-number comparison.** "One number does as well as 512" belongs in the write-up. It is the clearest statement of where the ceiling is.
4. **Drop PCA, or check what it kept.** For any feature we deliberately add, verify it survives the reduction.
5. **Fix the equal-length input question.** Still the biggest open item, and it needs Phil's script — see `00_SAY_IT_SIMPLY.md` §3.3 for how to raise it without contradicting Liuyi's position that we already have the raw stream.
6. **Restore best weights in Arm B.** Early stopping currently keeps the weights from the moment training halted, up to 8 epochs past the best validation loss, because checkpointing is disabled. That makes Arm B look slightly worse than it is. Cheap to fix.

---

## Reproducing all of this

Scripts used are in the session scratchpad; the essential recipe is short:

```python
# The bug, in six lines
from chronos import BaseChronosPipeline
import torch, numpy as np
p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                        device_map="cpu", torch_dtype=torch.float32)
short = torch.tensor(np.random.default_rng(0).normal(150, 30, 13).astype(np.float32))
long_ = torch.tensor(np.random.default_rng(1).normal(150, 30, 288).astype(np.float32))

alone = p.embed([short])[0].mean(dim=1)[0]          # 2 tokens
mixed = p.embed([short, long_])[0].mean(dim=1)[0]   # 19 tokens, 17 of them padding
print(torch.nn.functional.cosine_similarity(alone, mixed, dim=0))   # ≈ 0.22

# The [REG] token is stable:
print(torch.allclose(p.embed([short])[0][:, -1, :],
                     p.embed([short, long_])[0][:1, -1, :]))         # True
```

Everything else used `cgm_tsfm.data.load_real_data()` plus the same `GroupKFold(5)` / inner `GroupKFold(3)` `GridSearchCV` protocol as `cgm_tsfm/regression.py`, so the numbers are directly comparable to `results/headtohead_real.md`.

---

## The 60-second version

1. **Our pooling step averaged over padding.** 90% of sessions had materially wrong features. The `[REG]` token is immune; I verified it is bit-identical across batchings.
2. **I re-ran everything with correct features. The conclusion did not change** — still no better than guessing the average, on all three tests.
3. **Chronos-Bolt subtracts the mean and divides by the standard deviation**, so our features are largely blind to absolute glucose level. It hands those two numbers back and we were discarding them. Adding them back doesn't change the answer either.
4. **One number — average glucose — does as well as all 512.** Which fits: the strongest raw relationship in this data is 0.85% of the variance.
5. **PCA was quietly deleting features I had deliberately added.** Worth remembering.

→ Related: `08_CHRONOS.md` for the verified internals, `10_RIGOR_AND_STATS.md` for the shuffle test, `14_PRESENTATION_PLAN.md` for where this goes in the talk.
