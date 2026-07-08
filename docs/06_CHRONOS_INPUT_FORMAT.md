# Chronos Input/Output **Dimensions** + How a CGM Sample Becomes a Score

*Meeting-prep deep-dive for Brian. Covers two of the four tasks:*
- **Task 3 — the input dimension format of Chronos** (Part A). This is the one Liuyi will quiz you on.
- **Task 4 — how a CGM sample input becomes a predicted cognitive score** (Part B).

*Every shape below is **real** — measured on `cpsl-mds` with `chronos-bolt-small` and `chronos-t5-small` (the exact script is in the Appendix; you can re-run it live in the meeting). Plain-English versions of the surrounding concepts are in [`../PLAIN_ENGLISH_SUMMARY.md`](../PLAIN_ENGLISH_SUMMARY.md) (Q4, Q9, Q10).*

## Contents
- [Part A — Chronos input dimension format](#part-a) ← the meeting star
  - [A0. The 30-second answer](#a0)
  - [A1. The vocabulary of dimensions](#a1)
  - [A2. What Chronos takes as INPUT](#a2)
  - [A3. What Chronos gives back (OUTPUT)](#a3)
  - [A4. Bolt vs T5 — patches vs one-token-per-reading (the key picture)](#a4)
  - [A5. Five details people get wrong](#a5)
  - [A6. Anticipated meeting Q&A](#a6)
- [Part B — CGM sample → predicted score, end to end](#part-b)

---
<a name="part-a"></a>
# PART A — The input dimension format of Chronos

<a name="a0"></a>
## A0. The 30-second answer (say this first)

> "Chronos is a **univariate** time-series model. Its input is **one sequence of numbers over time** — conceptually shape **(batch, length)**: `batch` = how many series you feed at once, `length` = how many readings each has. There's **no channel dimension** because it models one variable at a time (for us that's glucose — which is exactly why the project is single-channel). Length can vary between series; Chronos left-pads a batch to the longest. Internally Chronos turns each series into a short sequence of **tokens** — **Chronos-T5 makes one token per reading**, while **Chronos-Bolt makes one token per 16-reading 'patch'** (faster). The model then outputs a hidden vector per token, shape **(batch, tokens, 512)**, and we **average over the tokens** to get **one 512-number embedding per series: (batch, 512)**."

Everything below is the detail behind that paragraph.

<a name="a1"></a>
## A1. The vocabulary of dimensions

| Symbol | Name | What it is | Our CGM value |
|---|---|---|---|
| **B** | batch | how many windows we feed at once | e.g. 32, or all 956 sessions |
| **L** | length | number of readings in one window | 3–288 (median ~36); 1 reading = 5 min |
| **C** | channels / variates | how many signals per timestep | **1** (glucose only) — Chronos is univariate |
| **P** | patch size | readings per patch (**Bolt only**) | **16** (= 16×5 min = 80 min of glucose per patch) |
| **T** | tokens | length of the sequence the transformer sees | Bolt: ≈L/16 (+1); T5: L (+1) |
| **d_model** | model width | size of each hidden vector = **the embedding length** | **512** (for the `-small` models) |

Keep these six letters in your head; every shape below is built from them.

<a name="a2"></a>
## A2. What Chronos takes as INPUT

The call we use is `BaseChronosPipeline.embed(context)`. `context` can be:

```
 (a) a single series      → a 1-D tensor of shape (L,)          e.g. (45,)
 (b) a batch, same length → a 2-D tensor of shape (B, L)         e.g. (32, 45)
 (c) a batch, ragged      → a LIST of B 1-D tensors, lengths vary e.g. [ (13,), (45,), (288,) ]
```

We use **(c)** — a list of variable-length windows — because real CGM windows differ in length (see `cgm_tsfm/encoders.py`). Notice what is **absent**: there is **no channel axis**. The input is `(B, L)`, not `(B, C, L)`, because Chronos is **univariate** — one variable over time. Our one variable is glucose in mg/dL. (If we ever had 2 signals, we'd run Chronos once per signal, or switch to a multivariate TSFM like Moirai.)

**Our CGM window → Chronos input:**
```
 one cognitive-test session's pre-test glucose:
   [140, 138, 135, 150, 161, ... , 132]      raw mg/dL, NaNs already dropped
   └───────────── L readings ─────────────┘   L varies per session (3 … 288)

   → wrapped as a 1-D float tensor of shape (L,)  → put in a list with the other windows in the batch
```
No normalization, no gap-filling, no fixed length required — Chronos handles all of that internally (see [A5](#a5)).

<a name="a3"></a>
## A3. What Chronos gives back (OUTPUT) — real measured shapes

`embed()` returns `(embeddings, scale)`. We keep `embeddings`:

```
 embeddings shape = (B, T, d_model)          # one 512-vector per token, per series
```

Then we **mean-pool over the token axis (axis 1)** to collapse each series to a single vector:

```
 (B, T, 512)   ──mean over axis 1──▶   (B, 512)      # ONE fingerprint per window
```

**Measured on the server** (`chronos-bolt-small`, `d_model = 512`, patch `P = 16`):

| input | `embed()` output | after mean-pool |
|---|---|---|
| one window, L=45 | **(1, 4, 512)** — 3 patches + 1 token | **(1, 512)** |
| batch of L=[13, 45, 288] | **(3, 19, 512)** — padded to 288 → 18 patches + 1 | **(3, 512)** |

The `T` numbers: `45 → pad to 48 → 48/16 = 3 patches → +1 register token = 4`. And `288/16 = 18 → +1 = 19`. The **+1** is one extra "summary" token the model adds; the mean-pool averages it in with the rest.

<a name="a4"></a>
## A4. Bolt vs T5 — the key picture (this is the memorable one)

Both models are univariate and both end at a 512-vector. They differ in **how they chop the series into tokens** — and that sets the middle dimension `T`. Same 45-reading window, measured on the server:

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CHRONOS-T5   → ONE TOKEN PER READING                                      │
  │                                                                           │
  │   r1  r2  r3  r4  ............................................  r45  [EOS] │
  │   │   │   │   │                                                 │     │    │
  │   45 value-tokens  +  1 end token          →   T = 46                     │
  │                                                                           │
  │   embed(45-window) = (1, 46, 512)      ← MEASURED                          │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CHRONOS-BOLT  → ONE TOKEN PER 16-READING "PATCH"   (what we use)          │
  │                                                                           │
  │   [ r1 … r16 ] [ r17 … r32 ] [ r33 … r45 +pad ]   (+1 register token)      │
  │    └ patch 1 ┘  └ patch 2 ┘   └── patch 3 ──┘                              │
  │   3 patch-tokens  +  1 token               →   T = 4                       │
  │                                                                           │
  │   embed(45-window) = (1, 4, 512)       ← MEASURED                          │
  └─────────────────────────────────────────────────────────────────────────┘

  Same input, same 512-d output after pooling — but Bolt's sequence is ~16× shorter.
  That's why Bolt is much faster and why we default to it. (Config: patch size 16,
  context_length 2048, prediction_length 64.)
```

For the longest real window (288 readings): **T5 → 289 tokens, Bolt → 19 patch-tokens.** One line to remember: **T5 reads glucose value-by-value; Bolt reads it in 80-minute chunks.**

> Why this matters for the meeting: the ICML paper Liuyi cited studied Chronos-**T5**; Ben and we use Chronos-**Bolt**. You can now speak to both: *"same univariate `(B, L)` input, same `(B, 512)` embedding after pooling — Bolt just patches the series so the transformer sees ~16× fewer tokens."*

<a name="a5"></a>
## A5. Five details people get wrong (know these cold)

1. **Univariate, so no channel dimension.** Input is `(B, L)`, not `(B, C, L)`. Our project being "1 channel (glucose)" is a *perfect* fit — it's literally what Chronos expects. Multi-signal would need per-signal passes or a different model.
2. **Variable length is fine.** Different windows have different `L`. In a batch, Chronos **left-pads** to the longest (older readings on the left get padding) and masks the padding — so padding never corrupts the embedding.
3. **No manual normalization.** Chronos **mean-scales each series internally** (divides by its own average magnitude), so a kid running at 250 mg/dL and one at 100 mg/dL are put on a common footing. We feed **raw mg/dL**. (The `scale` returned by `embed()` is that internal scaling factor.)
4. **NaNs are allowed.** Missing readings map to a mask token — no interpolation needed. Great for gappy CGM. (We still drop fully-empty windows.)
5. **There's a max length (`context_length = 2048`).** Way above our longest window (288), so we never hit it. If we ever fed days of 5-min data (>2048 readings ≈ 7 days) we'd need to window it.

<a name="a6"></a>
## A6. Anticipated meeting Q&A (rehearse these)

- **"What's the input shape?"** → "`(batch, length)` — univariate, no channel axis. We pass a list of variable-length windows; Chronos left-pads the batch."
- **"What's the output shape?"** → "`(batch, tokens, 512)`; we mean-pool the tokens to `(batch, 512)` — one 512-d embedding per window."
- **"Why 512?"** → "It's the model width `d_model` of the `-small` checkpoints. Bigger checkpoints are wider; the sweep tests whether that helps (it usually doesn't on our small N)."
- **"What's a patch?"** → "Bolt groups 16 consecutive readings (80 min) into one token, so a 288-reading window becomes ~18 tokens instead of 288. That's the speed trick."
- **"Do you normalize the glucose?"** → "No — Chronos mean-scales each series internally. We feed raw mg/dL."
- **"How do you handle missing readings / different lengths?"** → "NaNs become mask tokens; variable lengths are left-padded and masked. No gap-filling."
- **"Why mean-pool — why not the last token?"** → "Mean over tokens is the recipe from the ICML paper; it summarizes the whole window. `last` is available as a knob (`pooling`), and the sweep can compare them."
- **"Is this the same as forecasting?"** → "No — we don't predict future glucose. We take the encoder's hidden representation as a feature vector and predict the cognitive score from it."

---
<a name="part-b"></a>
# PART B — How a CGM sample becomes a predicted score (end to end, with real shapes)

This is the whole journey for **one session**, with the actual dimensions at each step. (The plain-English version is [`../PLAIN_ENGLISH_SUMMARY.md`](../PLAIN_ENGLISH_SUMMARY.md) Q9; here it's the dimension-level trace.)

```
 STEP 1 — RAW SAMPLE (one cognitive-test session)
   pre-test glucose window:  [140, 138, 135, ... , 132]     shape (L,)   e.g. L=45
   the true score we want to predict, e.g. symbols = 1.83    one number
        │
        ▼
 STEP 2 — ENCODE with frozen Chronos-Bolt          (cgm_tsfm/encoders.py)
   embed([window])            →  (1, 4, 512)        # 3 patches + 1 token, each 512-wide
   mean-pool over axis 1       →  (1, 512)           # the "fingerprint" of this window
        │                          └── this 512-vector is the ONLY thing that leaves Chronos
        ▼
 STEP 3 — STACK the whole dataset
   956 sessions each → a 512-vector  ⇒  feature matrix  X of shape (956, 512)
   the three score columns           ⇒  targets y_grids, y_symbols, y_prices  (each length 956)
        │
        ▼
 STEP 4 — PREDICT the score from the 512 numbers    (one model per target)
   Arm A:  Ridge/SVR    learns   f(512 numbers) → score
   Arm B:  small MLP    learns   f(512 numbers) → score
        │
        ▼
 STEP 5 — SCORE OURSELVES (honestly)                (cgm_tsfm/regression.py)
   subject-grouped cross-validation (never test on a kid seen in training)
   report R² vs a "guess-the-average" baseline, per target
```

### B1. The one-sample math in Step 4 (a tiny worked example)

The regressor is just a function of the 512 numbers. For **Ridge** (Arm A) it's literally a weighted sum:

```
  predicted_score  =  w1·e1  +  w2·e2  +  …  +  w512·e512  +  b
                          ▲                        ▲          ▲
                   the 512 embedding numbers   learned weights  learned offset
```

Training finds the weights `w` (and offset `b`) that make `predicted_score` closest to the true scores **on the training kids**, with a penalty that keeps the weights small (that's the "Ridge" regularization — it stops the 512 weights from overfitting 956 samples). SVR (Arm A) and the MLP (Arm B) are fancier functions of the same 512 numbers, but the input/output are identical: **512 in → 1 number out.**

> **Why we must reduce/regularize:** 512 weights fit on ~950 samples from only 20 kids overfits easily. Plain unregularized `LinearRegression` gave **R² ≈ −4** on real Chronos embeddings (worse than guessing!). Ridge/SVR + optional **PCA** (512 → ~32) fix this. (See `PLAIN_ENGLISH_SUMMARY.md` Q13.)

### B2. Why grouped CV closes the loop (Step 5, one line)

We split the **20 kids** into 5 groups; each round trains on 16 kids and tests on 4 **unseen** kids, then average the test R². This answers the only honest question — *"does the glucose fingerprint predict the score for a kid we've never seen?"* — and blocks the model from "recognizing the kid." Full explanation + diagram: `PLAIN_ENGLISH_SUMMARY.md` Q5/Q7.

**One sentence for the meeting:** *"A CGM window of L readings → Chronos-Bolt → a 512-number fingerprint → a small regressor → one predicted score; we stack all 956 sessions into a 956×512 matrix and evaluate per target under subject-grouped cross-validation."*

---
## Appendix — reproduce the shapes live (copy-paste on the server)

```python
# env: /data_1_8TB_ssd/brian_workspace/envs/cgm/bin/python   (HF_HOME on the SSD)
import torch, numpy as np
from chronos import BaseChronosPipeline
dev = "cuda" if torch.cuda.is_available() else "cpu"
w = torch.tensor(np.clip(140 + np.cumsum(np.random.default_rng(2).normal(0, 8, 45)), 40, 400),
                 dtype=torch.float32)                       # a fake 45-reading window

for name in ["amazon/chronos-bolt-small", "amazon/chronos-t5-small"]:
    pipe = BaseChronosPipeline.from_pretrained(name, device_map=dev, torch_dtype=torch.float32)
    emb, scale = pipe.embed([w])                            # input: list of 1-D tensors, shape (45,)
    print(name, "embed →", tuple(emb.shape), "→ pool →", tuple(emb.mean(dim=1).shape))
# amazon/chronos-bolt-small embed → (1, 4, 512)  → pool → (1, 512)
# amazon/chronos-t5-small   embed → (1, 46, 512) → pool → (1, 512)
```
