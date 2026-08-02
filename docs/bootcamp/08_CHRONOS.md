# 08 · Chronos under the hood

> **What this chapter buys you.** Liuyi asked *"how chronos works under the hood, you don't know"* — and separately asked **twice** whether we understand *"the dimension/shape format of data input to the TSFM."* This chapter answers both, from the actual installed source code rather than from the paper or from memory. By the end you can state the exact shape at every step, explain what a patch and a token are, explain how the model normalizes its input, and name which parts of the model we use and which we don't.
>
> Every fact here was read off `chronos` version **2.3.1** as installed at
> `/data_1_8TB_ssd/brian_workspace/envs/cgm/lib/python3.11/site-packages/chronos/`, with line references. Anything I could not verify is flagged.
>
> `07_TRANSFORMERS.md` covers the general transformer mechanism (attention, heads, residuals). This chapter is Chronos-specific.

---

## 1. What "foundation model" means here

A **foundation model** is a large network trained once, at expense, on a huge amount of general data, and then reused for many specific tasks without retraining. The idea arrived in language (GPT, BERT) and has since been copied for time series.

**Chronos** is Amazon's time-series foundation model. It was trained to **forecast**: given the first part of a series, predict what comes next. It saw enormous quantities of real and synthetically generated time series during training — energy demand, traffic, weather, sales, and so on.

**We are not forecasting.** We take the internal representation Chronos builds while reading a series, and use it as a description of that series. This is called using it as a **frozen encoder**: we run the model forward only, never update its weights, and treat its hidden state as a set of features.

**Why this is reasonable:** to forecast well, a model must build a useful internal description of what it has read. We borrow the description and throw away the forecast.

**Why this is also a genuine weakness — say this before someone says it to you:** the description was optimized for predicting the *next values of the glucose curve*, not for predicting *a child's test score*. Nothing guarantees the features it keeps are the ones our question needs. This is the honest caveat on the whole approach.

**Say it in your own words:** *"Chronos is a big model Amazon trained to forecast time series, using a huge amount of data we don't have. We don't use its forecasts. We run a glucose series through it and take the internal summary it builds — 512 numbers — and use that as our description of the curve. The catch is that summary was built for forecasting, not for our question."*

---

## 2. Two different Chronos families — and we must not mix them up

This is where our own documents got it wrong, so be careful.

|  | **Chronos-Bolt** ← *what we use* | **Chronos-T5** |
|---|---|---|
| Turns a reading into… | groups of **16** readings ("patches") | **one token per single reading** |
| Value representation | **continuous** numbers, kept as numbers | **discretized** into one of **4096** bins |
| `vocab_size` | **2** (no value vocabulary at all) | **4096** |
| Normalization | **instance norm**: (x − mean) / std | **mean absolute scaling**: x / mean(\|x\|), no centering |
| Missing values | explicit 0/1 mask channel | mapped to the pad token — **lossy** |
| Output | all 9 quantiles × 64 horizons in **one** pass | **autoregressive** sampling, 20 sampled paths |
| Sequence length for T=288 | **19 tokens** | **289 tokens** |
| `context_length` | 2048 | 512 |

> ⚠️ **A correction to our own documents.** `PLAIN_ENGLISH_SUMMARY.md` and `docs/understanding/01_CONCEPTS.md` describe Chronos as *"chopping the value-range into bins and turning each reading into a word."* **That is Chronos-T5's mechanism, not Bolt's.** Bolt does not bin values at all — its `vocab_size` is literally 2, holding only a decoder-start marker and one special token. If someone who knows the Bolt paper reads our docs, this is the first thing they will catch. Fix it in any slide you make.

**Say it in your own words:** *"There are two versions. The T5 one chops the value range into 4096 bins and treats each reading like a word in a sentence. The Bolt one, which we use, doesn't do that — it keeps the numbers as numbers and groups them into blocks of 16. Our older notes described the T5 version by mistake."*

---

## 3. Patches and tokens — the two words Liuyi told you to master

He wrote: *"This could be promising after you fully understand your code python code [sic] and all technical terms, e.g., token, patch, epoch."* So these definitions have to be exact.

### Patch

A **patch** is a block of **16 consecutive glucose readings**. Config: `input_patch_size = 16`, `input_patch_stride = 16`. Stride equals size, so patches are **non-overlapping and contiguous** — reading 1–16, then 17–32, and so on.

Since a reading is 5 minutes apart, **one patch = 80 minutes of glucose**.

### Token

A **token** is one patch after it has been converted into a vector of 512 numbers. It is the unit the transformer actually processes. Where a language model's tokens are pieces of words, Chronos-Bolt's tokens are 80-minute stretches of glucose.

### How a patch becomes a token

```
16 raw readings          [142, 145, 139, 141, ... ]      (16 numbers)
      │
      │  instance-normalize the whole series first (see §4)
      ▼
16 standardized values   [-0.31, -0.18, -0.44, ...]      (16 numbers)
      +
16 mask flags            [ 1, 1, 1, 1, ... ]  1 = this reading was real
      │
      │  concatenate → 32 numbers per patch
      ▼
   ResidualBlock:  Linear(32 → 2048) → ReLU → Linear(2048 → 512)  (+ a linear skip)
      ▼
one token: 512 numbers
```

The 32 is verifiable: `chronos_bolt.py:201` sets the patch-embedding input width to `input_patch_size * 2`, and I confirmed `in_features == 32` at runtime. The model always sees *both* the values and which of them were genuinely observed.

**Say it in your own words:** *"A patch is 16 consecutive readings — 80 minutes of glucose, since readings are 5 minutes apart. Each patch gets turned into 512 numbers by a small network, and that's a token. So a 3-hour session becomes about 3 patches, which is 3 tokens plus one extra."*

---

## 4. How Chronos normalizes the input — and why it matters for our science

**This is the most scientifically important fact in the chapter.**

Chronos-Bolt applies **instance normalization** to every series before doing anything else. From `chronos_bolt.py:108-117`:

```python
loc   = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
scale = torch.where(scale == 0, self.eps, scale)
scaled_x = (x - loc) / scale
```

In words: for each series independently, compute its **mean** and its **standard deviation**, subtract the mean, divide by the standard deviation.

Verified precisely:
- `loc` is the **NaN-aware arithmetic mean** over the time axis.
- `scale` is the **population standard deviation (ddof = 0)**, *not* the sample standard deviation. Checked numerically: for `x = [100, 120, 90, 150, 80]`, `loc = 108.0` and `scale = 24.819347`, which equals `std(ddof=0)` exactly and not `std(ddof=1) = 27.748875`.
- Guards: an all-NaN series gives `loc = 0, scale = 1`; a constant series gives `scale = eps = 1e-5`.
- Truncation to the last 2048 points happens **before** normalization (`chronos_bolt.py:283-288`), so the statistics describe the kept window.

### The consequence for our project

Both the level and the spread are divided out. So a glucose series sitting at 250 with swings of ±30 and one sitting at 100 with swings of ±30 **look nearly identical to the model.** The embedding describes the *shape*, in units of that session's own standard deviation.

That is awkward for this study specifically, because the clinical hypothesis is about hypoglycemia and hyperglycemia, and those are defined by **absolute** thresholds — below 70, above 180, above 250 mg/dL. Our representation normalizes away exactly the quantity the hypothesis is about.

It explains a measured result that otherwise looks like a broken pipeline:

| Ask the same embeddings to predict… | R² |
|---|--:|
| Glucose variability (standard deviation) | **0.462** |
| Mean glucose | **0.070** |
| Fraction of time above 180 | 0.075 |

If challenged — *"if your features are any good, why can't they recover the mean glucose?"* — the answer is: **the mean is subtracted off before the encoder sees the data.** That is by design, not a defect.

### And we were throwing away the fix

`embed()` returns **two** values — the embeddings *and* `(loc, scale)`. Our code discards the second (`cgm_tsfm/encoders.py:104`: `emb, _ = self.pipeline.embed(batch)`). Those two numbers are precisely the level and amplitude that were removed. See `15_FINDINGS_TO_REPORT.md` — I tested adding them back, and it does not change the conclusion, but it removes an obvious criticism.

> **Correct the record:** our docs say Chronos "mean-scales each series (divides by its own average magnitude)". That is the **T5** scheme. Bolt centres *and* scales.

**Say it in your own words:** *"Before Chronos looks at a series, it subtracts that series' own average and divides by its own standard deviation. So it sees the shape of the curve but not how high or low it actually was. That's a problem for us, because hypo and hyper are defined by absolute numbers. It also explains why the same features predict how *variable* the glucose was quite well, R² 0.46, but the average level badly, 0.07 — the average was removed on purpose."*

---

## 5. The shape walkthrough — the answer he asked for twice

### The formula, verified

```
embed(context) → (B,  ceil( min(T, 2048) / 16 )  +  1,  512)
                  │              │                  │     │
                  │              │                  │     └── d_model
                  │              │                  └──────── the [REG] token
                  │              └───────────────────────────  number of patches
                  └──────────────────────────────────────────  batch size
```

### Measured, not assumed

| Input length T | T % 16 | NaN left-padding added | Patches | **`embed()` shape** |
|---:|---:|---:|---:|:---|
| 3 | 3 | 13 | 1 | **(1, 2, 512)** |
| 16 | 0 | 0 | 1 | (1, 2, 512) |
| 20 | 4 | 12 | 2 | (1, 3, 512) |
| **36** *(our median session)* | 4 | 12 | 3 | **(1, 4, 512)** |
| 45 | 13 | 3 | 3 | (1, 4, 512) |
| 64 | 0 | 0 | 4 | (1, 5, 512) |
| 71 | 7 | 9 | 5 | (1, 6, 512) |
| **288** *(our longest, 24 h)* | 0 | 0 | 18 | **(1, 19, 512)** |
| 2048 | 0 | 0 | 128 | (1, 129, 512) |
| 4096 | — | — | 128 | (1, 129, 512) — **truncated** |

Note it is `ceil`, not `floor`: the exact-multiple rows (16, 32, 64, 288, 2048) confirm this.

### Padding is on the LEFT, and that is deliberate

From `chronos_bolt.py:83-89`, when `T % 16 != 0` exactly `16 − (T % 16)` NaN values are **prepended**:

```python
padding = torch.full(size=padding_size, fill_value=torch.nan, ...)
x = torch.concat((padding, x), dim=-1)      # padding FIRST
```

Why left and not right: it keeps the **most recent reading at the right-hand edge of the final patch**. For a forecasting model, "what happens next" follows on from the end of the series, so the end must stay aligned. The ragged edge is pushed to the oldest data instead.

Because the padding is always between 1 and 15 values — strictly less than a patch — the first patch always contains at least one real reading. Even T = 1 gives 15 NaN plus 1 real = one valid patch.

### The full pipeline, end to end

```
ONE SESSION
  one child, one test session, 45 readings before the test

  [142, 145, 139, ..., 151]                       45 numbers, mg/dL
        │
        │  instance-normalize:  (x − mean) / std          §4
        ▼
  [-0.31, -0.18, -0.44, ..., 0.22]                45 standardized numbers
        │
        │  left-pad with 3 NaN → 48 → cut into 3 patches of 16
        ▼
  patch 0: readings  1-13 (+3 NaN)   ┐
  patch 1: readings 14-29            │  each: 16 values + 16 mask flags = 32
  patch 2: readings 30-45            ┘
        │
        │  ResidualBlock(32 → 2048 → 512)
        ▼
  3 tokens of 512     +    1 [REG] token of 512   = 4 tokens
        │
        │  6 transformer encoder layers, 8 heads each
        ▼
  (1, 4, 512)          ← this is what embed() returns
        │
        │  mean over the token axis  (our pooling choice)
        ▼
  (1, 512)             ← 512 numbers describing this session

  Repeat for all 956 sessions and stack:

  X = (956, 512)       ← the input to Ridge / SVR / the MLP head
  y = (956,)           ← one cognitive score column
```

**Say it in your own words:** *"One session goes in as a single list of glucose numbers — nothing else, no timestamps, no participant id. Chronos standardizes it, pads it to a multiple of 16, cuts it into blocks of 16, and turns each block into 512 numbers. A 45-reading session becomes 3 blocks plus one summary token, so a 4-by-512 block. We average down to 512 numbers for the session. Stack all 956 sessions and you get a 956-by-512 table."*

---

## 6. The `[REG]` token — the mysterious "+1"

Our documents call this "one extra summary token" in three places and "a register token" in another, and never explain it. Here it is, from `chronos_bolt.py:307-322`:

```python
if self.chronos_config.use_reg_token:
    reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, ...)
    reg_embeds = self.shared(reg_input_ids)
    input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)   # APPENDED LAST
    attention_mask = torch.cat([attention_mask, torch.ones_like(reg_input_ids)], dim=-1)
```

Facts:

- It is a **single learned 512-number vector**, `shared.weight[1]` — the *same* vector prepended to every series regardless of content. The whole embedding table is `nn.Embedding(2, 512)`: id 0 is the decoder-start marker, id 1 is `[REG]`.
- It is **appended at the end** (`dim=-2` after the patches), so the token axis reads `[patch_0, patch_1, …, patch_{N−1}, REG]` and `[REG]` is always at index `-1`.
- Its attention mask entry is forced to 1, so it is **always attended to**.
- Its *input* is content-free, but its *output* position aggregates the whole sequence through self-attention — the same trick as BERT's `[CLS]`. It's a scratchpad the model can use to accumulate a summary.
- `use_reg_token = true` for all four Bolt checkpoints, so the +1 is always there.

> ⚠️ **Two consequences our docs get wrong.**
> 1. **`pooling="last"` in our config is the `[REG]` token, not the last time step.** `emb[:, -1, :]` selects index −1, which is `[REG]`. Anyone reading "last-token pooling" will assume "most recent patch." It isn't.
> 2. **`pooling="mean"` averages the `[REG]` output together with the patch outputs.** That's defensible, but it is worth knowing that the average mixes a summary vector with the per-patch vectors.
>
> **Flagged as unverified:** the `use_reg_token = False` branch would return `(B, num_patches, 512)` with no extra token. No cached checkpoint has that setting, so this is read from source, not executed.

**Say it in your own words:** *"Chronos adds one extra slot at the end of the sequence, called the REG token. Its input is the same learned vector every time, but because attention lets it look at all the patches, its output becomes a summary of the whole series — like the CLS token in BERT. It sits at the last index, which means our 'last' pooling option is actually grabbing that summary token, not the most recent glucose."*

---

## 7. The architecture, exactly

`amazon/chronos-bolt-small`, read from the loaded config:

| Field | Value |
|---|---|
| `d_model` | **512** |
| encoder layers | **6** |
| decoder layers | **6** |
| attention heads | **8** |
| `d_kv` (per head) | **64** (8 × 64 = 512) |
| `d_ff` | **2048** |
| dropout | 0.1 |
| feed-forward activation | `relu`, not gated |
| position information | **relative position bias**, 32 buckets, max distance 128 |
| `context_length` | **2048** (≈ 7 days of 5-min readings) |
| `prediction_length` | **64** |
| quantiles | **9**: 0.1, 0.2, … 0.9 |
| `vocab_size` | **2** |
| **total parameters** | **47,718,016** |

### Parameters by submodule

| Submodule | Parameters | Used by `.embed()`? |
|---|--:|:--:|
| `shared` (2×512 embedding) | 1,024 | yes |
| `input_patch_embedding` | 1,133,568 | **yes** |
| `encoder` | 18,882,304 | **yes** |
| `decoder` | 25,176,832 | no |
| `output_patch_embedding` | 2,526,336 | no |

**`.embed()` uses only 20,015,872 parameters — 41.9% of the model.** The decoder side (27,703,168 parameters, 58.1%) is loaded into memory and never executed on our path. That's a good, specific fact to have: *we load a 47.7M-parameter model and use 20M of it.*

*(A pedantic detail worth knowing: summing the submodules gives 47,720,064, which is 2,048 more than the total. That's because `shared.weight` is weight-tied into both the encoder and decoder embedding slots — the same tensor counted twice. `model.parameters()` de-duplicates.)*

### All five checkpoints

| Checkpoint | `d_model` | Total parameters | Layers | Heads | Tokens for T=288 |
|---|--:|--:|--:|--:|:--|
| `chronos-bolt-tiny` | 256 | 8,652,672 | 4+4 | 4 | 19 |
| `chronos-bolt-mini` | 384 | 21,236,096 | 4+4 | 8 | 19 |
| **`chronos-bolt-small`** | **512** | **47,718,016** | **6+6** | **8** | **19** |
| `chronos-bolt-base` | 768 | 205,292,928 | 12+12 | 12 | 19 |
| `chronos-t5-small` | 512 | 46,154,240 | 6+6 | 8 | **289** |

All four Bolt sizes share patch size 16, `context_length` 2048, `prediction_length` 64, dropout 0.1 and the same 9 quantiles. **Only the width and depth change** — so the token count is identical across sizes.

Note that `chronos-t5-small` has almost the same parameter count as `bolt-small` but produces a **15× longer sequence** for the same input (289 tokens vs 19). That's the main practical cost difference between the families.

**The result that matters:** bigger did not help. Mean R² across the three scores was tiny −0.049, mini −0.052, small −0.052, **base −0.063 (the worst)**, t5-small −0.057. When the limit is the data rather than the model, a bigger model just has more room to overfit. Being able to say *"I swept four sizes across a 24× parameter range and the biggest was worst"* is a strong answer to "did you try a bigger model?"

---

## 8. Is it encoder-only or encoder-decoder?

Verified from the object, not assumed: **encoder-decoder — but the decoder runs exactly one step, and we never run it.**

- Both stacks exist. Encoder blocks are `[SelfAttention, FeedForward]`; decoder blocks are `[SelfAttention, CrossAttention, FeedForward]`. `is_encoder_decoder = True`.
- The decoder is **degenerate**: `decode()` feeds a single start token and returns `(batch, 1, d_model)`. There is no token-by-token generation. It is best understood as a learned single-query pooler over the encoder's outputs.
- `.embed()` calls `self.model.encode(...)` and returns the encoder's `last_hidden_state`, stopping before the decoder entirely.

So: *"it is built as an encoder-decoder in the T5 style, but nothing autoregressive happens — the decoder is one step — and our embedding path uses only the encoder."*

### How it forecasts, in one paragraph (so you can answer if asked)

The single decoder output vector goes through `output_patch_embedding`, a `512 → 2048 → 576` block, and `576 = 9 quantiles × 64 horizon steps`. So **one forward pass emits all 9 quantiles for all 64 future steps at once**, which are then un-standardized using the input series' own `loc` and `scale`. Training used the **quantile (pinball) loss**. There is no sampling and no autoregression within those 64 steps.

**A good question you should have an answer to:** *"If you never forecast, why does `prediction_length = 64` exist, and does it affect your embedding?"* → *"It's part of the pretrained checkpoint — it sets the width of the output head, which is 9 × 64 = 576. It has no effect on our embeddings, because we stop at the encoder and never build the output. It's just inert configuration on our path."*

---

## 9. Missing readings — the mechanism, and why we never use it

Chronos-Bolt handles gaps properly (`chronos_bolt.py:296-303`):

```python
patched_context = self.patch(context)
patched_mask    = torch.nan_to_num(self.patch(mask), nan=0.0)
patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
patched_context = torch.cat([patched_context, patched_mask], dim=-1)   # 16 + 16 = 32
attention_mask  = patched_mask.sum(dim=-1) > 0    # keep a patch if ≥1 real reading
```

So: NaN marks missing; values and a 0/1 observed-mask are patched together giving 32 inputs per patch; missing values are set to **0.0 after standardization**, meaning "at this series' average"; and a patch with no observed readings at all is dropped from attention.

> ⚠️ **But we never exercise any of this.** `cgm_tsfm/data.py:47` does `arr = arr[~np.isnan(arr)]` — **NaNs are deleted before Chronos ever sees them.** So the claim in our docs that "Chronos is NaN-tolerant, so we need no gap-filling" is true of Chronos and irrelevant to our pipeline.
>
> **And there's a real cost.** Deleting a reading silently **breaks the even 5-minute spacing**. Chronos receives no timestamps and assumes evenly spaced points. So a session with a 30-minute sensor gap becomes a series where two adjacent numbers are half an hour apart, and the model has no way to know. Our "one patch = 80 minutes" story is only true when no readings were dropped.
>
> This is a genuine, unaddressed limitation and a fair thing to be asked about. The better handling would be to keep the NaNs and let Chronos's mask mechanism do its job.

**Say it in your own words:** *"Chronos has a proper way to handle missing readings — it passes a flag alongside each value saying whether it was real. But we delete missing readings before it ever sees them, so we never use that. The cost is that deleting a reading breaks the even 5-minute spacing, and Chronos has no timestamps, so it assumes the points are evenly spaced when they aren't."*

---

## 10. Our exact usage, line by line

```python
# cgm_tsfm/encoders.py:97-112
def encode(self, windows):
    for i in range(0, len(windows), 32):                       # batches of 32
        batch = [torch.tensor(w) for w in windows[i:i+32]]
        with torch.no_grad():                                  # ← no gradients: frozen
            emb, _ = self.pipeline.embed(batch)                # (B, tokens, 512)
            #      ↑ the discarded (loc, scale) — see §4
            if self.cfg.pooling == "last":
                pooled = emb[:, -1, :]                         # ← this is [REG], not "last time"
            else:
                pooled = emb.mean(dim=1)                       # ← averages padding too — see §11
        out.append(pooled.cpu().numpy())
```

**How "frozen" is actually enforced:** two ways. `torch.no_grad()` means no gradients are computed. And `from_pretrained` returns the model in eval mode, so dropout is inactive and the output is deterministic. Worth knowing: **nothing in the checkpoint is marked frozen** — all 47.7M parameters load with `requires_grad=True`. Freezing here is behavioural, not a property of the weights. In Arm B it's also enforced explicitly by setting `requires_grad = False` on every encoder parameter (`ben_adapter/model.py:141-142`).

A verified corollary: `embed()` is bit-deterministic in eval mode, but **would become non-deterministic if the model were ever switched to `.train()`**, because `dropout_rate = 0.1` is live in the patch embedding and every encoder block.

**Caching:** the forward pass is the expensive step, so embeddings are computed once and cached to `.npy`, keyed by a hash of (model name, pooling, the actual window data) — `encoders.py:136`. That's why the sweeps are fast: three targets and every regressor reuse one encoder pass.

---

## 11. The bug in our pooling — know this cold

Because `emb.mean(dim=1)` averages over **all** token positions, and Chronos left-pads every series in a batch out to the longest one, a short session in a batch with a long one gets its features diluted by padding positions.

Measured: a 13-reading session embedded alone versus in a batch with a 288-reading session gives vectors with **cosine similarity 0.219**. Across our 956 real sessions, **861 (90%) had cosine similarity below 0.9** against correctly-pooled features, median **0.37**.

The `[REG]` token is **immune** — bit-identical across batchings, difference exactly 0.000.

And critically: **I re-ran the full evaluation with corrected features and the conclusion did not change.** Full detail, tables and the fix are in **`15_FINDINGS_TO_REPORT.md`** — read that before the meeting.

---

## 12. Drill

<details>
<summary><b>What shape does <code>embed()</code> return for a 36-reading session, and why?</b></summary>

`(1, 4, 512)`. 36 readings, patch size 16: 36 % 16 = 4, so 12 NaN values are prepended to reach 48, which is 3 patches. Plus the `[REG]` token = 4 token positions. Each is 512 numbers wide because `d_model = 512`.
</details>

<details>
<summary><b>What is the "+1" in (B, num_patches+1, d_model)?</b></summary>

The `[REG]` token — a single learned 512-number vector appended at the end of the sequence, always attended to. Its input is identical for every series; its output becomes a summary of the whole series through self-attention, like BERT's `[CLS]`. It lives at index −1.
</details>

<details>
<summary><b>Does Chronos-Bolt turn glucose values into words from a vocabulary?</b></summary>

No — that's Chronos-T5, which bins values into 4096 discrete tokens. Bolt keeps values continuous, groups them into patches of 16, and passes each patch through a small network. Bolt's `vocab_size` is 2, holding only a decoder-start marker and the `[REG]` token.
</details>

<details>
<summary><b>Do you normalize the glucose before feeding it in?</b></summary>

We don't do anything ourselves — we pass raw mg/dL. Chronos-Bolt normalizes internally: it subtracts each series' own mean and divides by its own population standard deviation. The consequence is that our features are largely blind to absolute glucose level, which matters because hypo and hyper are defined by absolute thresholds.
</details>

<details>
<summary><b>Why is the padding on the left rather than the right?</b></summary>

So the most recent reading stays at the right-hand edge of the final patch. Chronos was built to forecast, and the forecast continues from the end of the series, so the end must stay aligned to a patch boundary. The ragged edge goes to the oldest data instead.
</details>

<details>
<summary><b>How many of Chronos's parameters do you actually use?</b></summary>

20,015,872 of 47,718,016 — 41.9%. That's the input patch embedding plus the 6 encoder layers. `embed()` stops at the encoder's last hidden state, so the 6 decoder layers and the output head (27,703,168 parameters) are loaded but never executed.
</details>

<details>
<summary><b>You said Chronos is univariate. What would it take to add insulin or activity?</b></summary>

Chronos-Bolt takes one series and has no channel axis at all. So you cannot simply add a second input. The options would be: encode each channel separately and concatenate the embeddings; or switch to a genuinely multivariate model like Moirai. Note Liuyi has gated this — no extra channels until the glucose-only pipeline is fully understood and exhausted.
</details>

<details>
<summary><b>Why did the biggest model do worst?</b></summary>

Because the limit is the data, not the model. bolt-base has 205M parameters against 4× more embedding width (768 vs 512 for small), which means more columns for the downstream regressor to overfit on ~764 training rows — while the underlying relationship between glucose and score is essentially absent (strongest raw correlation r = −0.092). More capacity with no signal to find means more room to fit noise. Mean R²: tiny −0.049, mini −0.052, small −0.052, base −0.063.
</details>

---

## 13. Twelve sentences that cover this chapter

1. Chronos is a transformer Amazon pretrained to forecast time series; we use it frozen, as a feature extractor, and never update its weights.
2. We use **Chronos-Bolt**, which keeps values continuous and groups them into patches — *not* Chronos-T5, which bins values into 4096 discrete tokens.
3. A **patch** is 16 consecutive readings = 80 minutes of glucose; stride 16 means patches don't overlap.
4. A **token** is one patch turned into 512 numbers by a small network whose input is 32 wide: 16 values plus 16 flags saying which were real.
5. Before anything else, Bolt **subtracts each series' mean and divides by its standard deviation** (population, ddof=0) — so our features see shape, not absolute level.
6. That is why the same features predict glucose *variability* at R² = 0.462 but *mean glucose* at only 0.070 — and `embed()` hands back the two removed numbers, which our code discards.
7. `embed()` returns `(B, ceil(min(T, 2048)/16) + 1, 512)`. Our median session, 36 readings, gives `(1, 4, 512)`; the longest, 288 readings, gives `(1, 19, 512)`.
8. Padding is **NaN on the left**, so the most recent reading stays at the right edge of the last patch.
9. The **+1** is the `[REG]` token — one learned vector appended last, always attended, acting like BERT's `[CLS]`. Our `pooling="last"` selects it, not the final time step.
10. `chronos-bolt-small` is 47,718,016 parameters: 6 encoder + 6 decoder layers, 8 heads, `d_model` 512, `d_ff` 2048. `embed()` uses only 41.9% of them.
11. Bigger did not help — four sizes across a 24× parameter range, and the largest was the **worst** (−0.063 vs −0.049).
12. Two things to disclose: our mean pooling averaged over padding (fixed, conclusion unchanged), and we delete missing readings ourselves, which breaks the even 5-minute spacing Chronos assumes.

→ Next: `09_THE_TWO_ARMS.md` for what happens to those 512 numbers, or `15_FINDINGS_TO_REPORT.md` for the bug and its measurement.
