# 07 · Transformers — what they are, and why one is in this project

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you.**
> 1. You can answer "why a transformer?" with an honest reason (pretraining transfer) instead of a slogan, and name what the architecture actually contributes.
> 2. You can compute self-attention by hand on 3 tokens and show every intermediate matrix, so nobody can ask you to "just show me the calculation" and catch you.
> 3. You can say what "token" and "patch" mean in one plain sentence each, which is exactly what you were asked and couldn't answer.
> 4. You can explain how the model knows the *order* of the glucose readings — relative position bias — which none of our older documents even mentions.
> 5. You can explain why the biggest checkpoint scored the worst without sounding defensive.

The question this chapter answers, verbatim: *"Why do we need transformer for our this project and how chronos works under the hood, you don't know."*

That is two questions. **This file answers the first one** — what a transformer is, and why this project uses one. The second one (patching, instance normalization, the `[REG]` token, exact tensor shapes, checkpoint sizes) is **`08_CHRONOS.md`**. Do not mix them up in the meeting; answer whichever was asked.

Companion pages: numbers are cross-checked against **`01_FACT_SHEET.md`**. Neural-network basics — a layer, weights, ReLU, dropout, backpropagation — are **`06_DEEP_LEARNING_CORE.md`**; read that first if "linear layer" is not yet a solid idea. What we do with the 512 output numbers is **`09_THE_TWO_ARMS.md`**.

---

## The build order

Read top to bottom. Nothing below depends on anything further down.

```
 1. what a "sequence" is here, and why a plain network is a bad fit
                          |
                          v
 2. the four candidate approaches, compared honestly
                          |
                          v
 3. THE REAL REASON we use a transformer (pretraining, not elegance)
                          |
                          v
 4. self-attention from zero  ->  Q, K, V  ->  softmax  ->  worked example
                          |
                          v
 5. multi-head attention          6. the rest of an encoder block
                          |
                          v
 7. how it knows the ORDER (relative position bias)
                          |
                          v
 8. encoder vs decoder     9. cost (n squared)     10. what it does NOT give you
                          |
                          v
                11. bridge to 08_CHRONOS.md
```

---

## 1 · The problem a sequence model has to solve

### 1.1 What "a sequence" means in this project

One example in our dataset is **one cognitive test session**. Its input is a list of glucose numbers, one every 5 minutes, taken before that test:

```
session 412:  [ 143, 147, 152, 158, 161, 159, 154, ... , 188, 191 ]
                 |                                            |
              oldest reading                            most recent reading
```

Two properties of that list cause all the trouble:

| Property | The number | Why it hurts |
|---|---|---|
| **The order carries meaning** | a rise from 100 to 250 is not the same event as a fall from 250 to 100 | any method that treats the list as an unordered bag throws this away |
| **The length varies enormously** | **3 to 288** readings, median **36** | a fixed-size input layer cannot accept both |

The length spread is 96×, and it is not a rounding detail — see `01_FACT_SHEET.md` §C. It happens because the stored array is everything since that child's *previous* test, so an overnight gap gives ~288 readings and two tests an hour apart give ~12.

### 1.2 Why you cannot just feed the raw readings to an ordinary network

An ordinary feed-forward network (also called a fully-connected network, or a multilayer perceptron / MLP) takes a fixed number of inputs and multiplies them by a learned weight matrix. Three separate problems:

**Problem 1 — variable length has no natural fix.** A layer expecting 288 inputs cannot be handed 3. Your options are all bad: pad the short ones with zeros (the network then spends capacity learning to ignore padding, and zero is a *meaningful* glucose value away from real), truncate the long ones to 3 (throwing away 285 readings), or resample everything to a fixed length (which distorts the time axis differently for every session).

**Problem 2 — no built-in notion of order.** A fully-connected layer has one weight per input position. It *can* learn that position 5 and position 6 are adjacent, but nothing tells it so — it must learn that from data, separately for every pair of positions. With 956 examples it will not.

**Problem 3 — the parameter count scales with length.** A single layer from 288 inputs to 512 outputs is 288 × 512 = 147,456 weights, and it is *position-specific*: the weight it learned for "reading number 40" is useless for "reading number 41". Compare with our trainable head, which has 132,609 parameters total and is already 217× larger than the number of training rows (`01_FACT_SHEET.md` §H). We have no parameter budget for this.

**Say it in your own words:** *"Our input is a list of glucose readings whose length swings from 3 to 288, and the order matters. A plain network needs a fixed-size input and has to learn 'position 5 is next to position 6' from scratch, so it's the wrong tool."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. Why not just pad every session to 288 readings with zeros?**
A. Three reasons. The network has to learn to ignore the padding, which costs capacity we do not have. Zero is not a neutral value on a glucose axis — real readings run 40 to 400, so a zero is an extreme, not an absence. And the median session is 36 readings, so a typical row would be 87% padding.

**Q. Give one concrete pair of sequences a bag-of-numbers method cannot tell apart.**
A. `[100, 150, 200, 250]` and `[250, 200, 150, 100]`. Identical mean, identical standard deviation, identical minimum and maximum, identical time-in-range. One is a child's glucose rising into hyperglycemia, the other is it coming back down. Only an order-aware method can separate them.

**Q. How many weights would a first layer from raw readings cost?**
A. 288 × 512 = 147,456 for one layer, on 956 examples — and every weight is tied to one specific position in the sequence.

</details>

---

## 2 · The four candidate approaches, compared honestly

This section is the substance of "why a transformer". There are four families you could reasonably use. All four are real options; one of them is what this project's other arm already does.

### 2.1 The comparison table

| | **Summary statistics** | **RNN / LSTM / GRU** | **1-D CNN** | **Transformer** |
|---|---|---|---|---|
| What it does | you compute a fixed list of numbers describing the series | walks the series left to right carrying a memory | slides small learned filters over the series | every position looks directly at every other position |
| Handles variable length? | yes, naturally | yes, naturally | yes, with pooling | yes, naturally |
| Keeps the order? | mostly **no** | yes, strictly | yes, locally | yes, via a position signal added to the scores |
| Steps to link reading 1 to reading 36 | 1, but the link is whatever you built in by hand | **35 sequential updates** | **18 layers** (kernel 3, stride 1) | **1 attention step** |
| Parallel over time? | n/a | **no** — step t needs step t−1 | yes | yes |
| Who chooses which pattern matters? | **you do, in advance** | learned | learned, local first | learned, any distance |
| Data needed to train from scratch | tiny | thousands of sequences | thousands | tens of thousands+ |
| **Good pretrained checkpoints available?** | n/a | essentially none | few | **many** |
| Used in this project? | yes — the 43-feature arm | no | no | yes — Chronos |

### 2.2 The picture: how far can information travel?

The task: let the *first* reading of a 36-reading session influence the representation that eventually predicts the score, together with the *last* one.

```
(a) SUMMARY STATISTICS  -- 43 hand-built numbers
     r1  r2  r3  ...  r35  r36
      \   \   |        /   /
       +------+--------+--+
              |
              v
     mean, sd, min, max, %<70, %>180, slope, ...      (43 numbers)

     Reach: everything, in one step.
     Cost:  the ONLY relationships that survive are the ones you thought of.
            [100,150,200,250] and [250,200,150,100] collapse to the same row
            for most of the 43 (slope is the exception -- that is exactly why
            slope was hand-added).


(b) RNN / LSTM / GRU  -- one memory vector passed along
     r1 --> r2 --> r3 --> ... --> r34 --> r35 --> r36 --> (final memory)
     h1     h2     h3            h34     h35     h36

     Reach: r1 can affect h36, but only by surviving 35 rewrites of h.
     Cost:  35 sequential steps. Cannot be computed in parallel.
            Training signal must travel back through all 35 (vanishing gradient).


(c) 1-D CNN  -- learned filters of width 3, stacked
     layer 1  window = 3     [r1 r2 r3]
     layer 2  window = 5     [r1 .. r5]
     layer 3  window = 7     [r1 .. r7]
       ...
     layer 18 window = 37    <-- first layer where r1 and r36 are in one window

     Reach: grows by 2 per layer (1 + 2L). 18 layers to span 36 readings,
            or 5 layers with dilation 1,2,4,8,16 (window 63).
     Cost:  depth, or a hand-chosen dilation schedule.


(d) TRANSFORMER  -- on Chronos's 4 tokens for a 36-reading session
        p1      p2      p3      REG
         ^-------^-------^-------^
         |       |       |       |
         +-------+-------+-------+     every pair, in ONE step
         (each token computes a weighted blend of ALL tokens, itself included)

     Reach: any pair, distance 1 or distance 18, in a single attention step.
     Cost:  n-squared pairs -- irrelevant at n = 4 (see section 9).
```

### 2.3 Each option in prose

**Summary statistics.** This is not a straw man — it is the arm we actually compare against. `cgm_tsfm/handcrafted.py` builds **43** numbers per session: mean, standard deviation, minimum, maximum, percentiles, fraction of readings below 70 (time in hypoglycemia), fraction above 180 and above 250, mean and standard deviation of the 5-minute differences, the fitted slope, and so on. Strengths, honestly stated: every column has a name a clinician recognizes, it works on 956 rows without blinking, and it encodes the **absolute glucose level**, which matters because Chronos largely does not (`01_FACT_SHEET.md` §I). Weakness: you must decide *in advance* which patterns matter. If the thing that affects cognition is "a fast fall that ends 20 minutes before the test", no cell in that 43-wide table sees it unless you thought to add it.

**RNN / LSTM / GRU.** A recurrent network reads one value at a time and keeps a **hidden state** — a vector of numbers that is its running memory. At each reading it combines the new value with the memory to produce a new memory. LSTM and GRU are recurrent networks with learned gates that decide what to keep and what to erase; they exist specifically because plain RNNs forget too fast. Two problems for us. First, path length: to let reading 1 affect the final memory, the information has to survive **35** consecutive updates, each of which can overwrite it, and during training the correction signal has to travel back through all 35 multiplications — the classic **vanishing gradient** problem, where the signal shrinks toward zero the further back it goes. Second, the computation is inherently sequential: step 20 cannot start until step 19 is done, so you cannot use a GPU's parallelism across time. And the decisive practical problem: there is no well-known pretrained LSTM for general time series that we could download and use frozen.

**1-D CNN (temporal convolution).** A convolution slides a small learned filter along the sequence — a width-3 filter looks at three consecutive readings and outputs one number, then moves along by one. This is genuinely good at local shape: "three readings rising steeply" is exactly what one filter can learn, and the same filter is reused at every position, so it is parameter-cheap and order-aware. The limit is the **receptive field** — how much of the input one output number can see. With width-3 filters and stride 1 the field grows by 2 per layer: 3, 5, 7, … , 1 + 2L. To span 36 readings you need 18 layers. **Dilation** (skipping inputs, so filters see every 2nd, then every 4th, then every 8th reading) fixes this cheaply — dilations 1, 2, 4, 8, 16 give a field of 63 readings in 5 layers — but the schedule is a design choice you make, not something learned.

**Transformer.** Every position computes its output as a weighted blend of *all* positions, with the weights computed from the content itself. So the path length between any two positions is **1**, regardless of distance, and all positions are computed at once. Concretely for us: a 36-reading session becomes 4 tokens, and token 1 relates to token 3 in one hop. A 288-reading session becomes 19 tokens, and token 1 still relates to token 18 in one hop. The LSTM's cost would have grown to 287 sequential steps; the transformer's does not grow at all.

> **Two words you were asked about, defined once.**
> **Patch** — a short block of consecutive readings glued together and treated as one unit. Chronos-Bolt uses blocks of **16** readings (`input_patch_size: 16`, `input_patch_stride: 16` in the checkpoint config), so 16 readings = 80 minutes = one patch.
> **Token** — one position in the sequence the transformer actually processes; each token is a vector of 512 numbers. For us one token = one patch (plus one extra bookkeeping token). Verified by running the model: 3 readings → 2 tokens, 36 readings → **4** tokens, 288 readings → **19** tokens.

**Say it in your own words:** *"Four options. Hand-built statistics need me to guess in advance what matters. An LSTM has to carry information from reading one through 35 updates, and can't be parallelized. A CNN sees local shape but needs 18 layers to span three hours. A transformer lets any two positions talk in one step. But the real reason is the next section."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Why is "path length" a real problem and not just aesthetics?**
A. Two concrete consequences. Forward: the LSTM's memory is a fixed-size vector rewritten at every step, so early information competes with 35 later updates for room. Backward: the training correction reaches reading 1 only after being multiplied through 35 steps, and repeated multiplication by numbers below 1 drives it toward zero. A transformer's first-to-last connection is a single weighted sum, so neither problem applies.

**Q. How many layers does a width-3 CNN need to span our median session?**
A. Receptive field is 1 + 2L, so L = 18 gives 37 ≥ 36. With dilations 1, 2, 4, 8, 16 it takes 5 layers for a field of 63.

**Q. What does the hand-crafted arm do *better* than Chronos?**
A. It keeps the absolute glucose level. Chronos rescales each series internally, so its features describe shape more than height — visible in our own check, where the same pipeline predicts glucose **variability** at R² = 0.462 but mean glucose at only 0.070 (`01_FACT_SHEET.md` §I). The 43 features include mean and time-in-range directly.

**Q. If the transformer is better on every row of the table, why is the hand-crafted arm still in the project?**
A. Because it is the comparison that makes the result meaningful, and because it is level-aware where Chronos is not. Both came out no better than guessing the average, and the fact that the level-aware method *also* came out flat is what lets us say "no relationship" rather than "Chronos happened to discard the useful part."

</details>

---

## 3 · The honest primary reason: pretraining transfer

Everything in section 2 is true, and none of it is the decisive reason. Say the real one first, because if you lead with architecture elegance a picky examiner will ask "so would a CNN have worked?" and the honest answer is "probably about as well" — which then looks like a retreat.

**The decisive reason is that we cannot train any sequence model from scratch, and the only usable pretrained time-series models are transformers.**

The arithmetic:

```
  20 participants
 956 sessions          <- our entire supervised dataset
   1 input channel (CGM only)
   3 targets, modelled separately

 A small LSTM or CNN trained from scratch: tens of thousands of parameters,
 needing thousands of sequences to constrain them.
 Our own 132,609-parameter head on ~612 training rows is already 217x more
 parameters than examples, and it is the WORST performer we have
 (R^2 = -0.168 to -0.281).  See 01_FACT_SHEET.md sections H and I.
```

So training a sequence model on our data is off the table. That forces the approach: take a model that already learned what time series look like from a large amount of *other* data, freeze it, and use it only as a fixed translator from a glucose list into 512 numbers. This is **transfer learning** — reuse of representations learned on one task for a different task — and the frozen use of it is what `cgm_tsfm/encoders.py` does under `torch.no_grad()`.

Now the second half of the argument. Once you have decided "must be pretrained", the choice of architecture is made for you. The pretrained time-series models that are actually available, documented and used in our reference codebases — Amazon **Chronos** / Chronos-Bolt, Google **TimesFM**, **MOMENT**, **MOIRAI** — are all transformers. Ben's actigraphy repo, which is the architectural template for our Arm B, uses Chronos-Bolt and TimesFM; both are transformers. There is no comparable download-and-freeze LSTM for general time series.

**The sentence to say out loud:**

> *"The honest reason is not that transformers are elegant. With 20 participants and 956 sessions we cannot train any sequence model from scratch, so we need one that was already trained on a lot of other time-series data. Every strong pretrained time-series model you can download is a transformer, so choosing 'pretrained' is what chose 'transformer'. The architecture's own advantages — any two positions can talk in one step, and it parallelizes — are real, but they are the second reason, not the first."*

**Two follow-ups you should have ready.**

*"How much data was Chronos trained on?"* — A large collection of time series from many domains, gathered and generated by Amazon. If pushed for the exact corpus and totals, say you would check the Chronos papers rather than quote a number; nothing in our repo pins it down. (`08_CHRONOS.md` covers the checkpoint facts we *have* verified.)

*"Are there non-transformer pretrained time-series models?"* — Yes, a few exist, including mixer-style ones. The accurate claim is narrower and safer: the pretrained models that are mainstream, well documented, and already in use in our lab's codebases are transformers.

**Say it in your own words:** *"We have 956 sessions. That's nowhere near enough to train a sequence model, so we need a pretrained one, and the pretrained ones are all transformers. That's the reason — the architecture arguments come second."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. So you chose a transformer for its architecture?**
A. No. I chose it because it had to be pretrained. With 956 sessions from 20 children, training a sequence model from scratch is not possible — our own 132,609-parameter head on about 612 training rows is already the worst-performing thing in the project. The pretrained time-series models that are actually available are transformers, so that decision made the architecture decision.

**Q. Would a CNN have done as well?**
A. Trained from scratch on our data, probably about as poorly, for the same data-limited reason. A *pretrained* CNN on general time series is not something I can download. So the comparison never really arises, and I would rather say that than pretend the architecture was the deciding factor.

**Q. What does "frozen" mean, exactly?**
A. The Chronos weights are downloaded already trained and never change. `cgm_tsfm/encoders.py` runs it inside `torch.no_grad()`, which turns off gradient tracking, so no update can be computed and none is applied. It is used purely as a fixed function from a glucose list to 512 numbers. The only thing we fit is the regressor on top.

</details>

---

## 4 · Self-attention from absolute zero

This is the mechanism. Build it in five steps.

### 4.1 The intuition, in three sentences

Each token asks a question: *"what am I looking for?"* Every token advertises what it has: *"here is what I am about."* The answer each token receives is a **weighted blend of what all the tokens hold**, where the weight on each is how well its advertisement matched the question.

Nothing more mysterious than that. The whole apparatus below is bookkeeping for those three sentences.

### 4.2 Query, Key, Value — where they come from

Start with the input: `n` tokens, each a vector of `d_model` numbers. In Chronos-Bolt-small, `d_model = 512`.

Three learned matrices — the *same* three for every token, and the same three at every position — turn each token into three different vectors:

| Name | Symbol | Role in the metaphor | In Chronos-Bolt-small |
|---|---|---|---|
| **Query** | Q | the question this token is asking | `Linear(512 -> 512)`, no bias |
| **Key** | K | the advertisement this token displays | `Linear(512 -> 512)`, no bias |
| **Value** | V | the content this token will hand over | `Linear(512 -> 512)`, no bias |

"Self-attention" means Q, K and V are all computed from the *same* sequence. (In cross-attention, which the decoder uses, Q comes from one sequence and K, V from another.)

```
                       +--> x Wq --> Q   "what each token is asking for"
    X                  |
  (n tokens x 512) ----+--> x Wk --> K   "what each token advertises"
                       |
                       +--> x Wv --> V   "what each token will hand over"

  Wq, Wk, Wv are learned during pretraining. In our pipeline they are frozen.
```

### 4.3 The formula, piece by piece

```
                       ( Q K^T )
   Attention(Q,K,V) = softmax( ------- ) V
                       ( sqrt(d_k) )
```

Read it right to left inside out:

| Piece | What it is | Shape (n tokens, d_k per head) |
|---|---|---|
| `Q K^T` | every query dotted with every key — one **score** per (query, key) pair. A dot product is large when two vectors point the same way, so it measures "how well does this advertisement match this question". | (n, n) |
| `/ sqrt(d_k)` | shrink the scores before the softmax (why, below) | (n, n) |
| `softmax(...)` | turn each row of scores into positive weights summing to 1 | (n, n) |
| `... V` | each output row = the weighted sum of all value vectors | (n, d_k) |

**Why divide by `sqrt(d_k)`.** A dot product of two `d_k`-dimensional vectors is a sum of `d_k` products. If the entries are roughly independent with variance 1, the sum has variance `d_k` — so with `d_k = 64` the scores have standard deviation `sqrt(64) = 8`. Feed numbers of that size into a softmax and it saturates: almost all the weight lands on the single largest score, the output becomes a hard pick of one token instead of a blend, and — the part that matters for training — the gradient through a saturated softmax is nearly zero, so learning stalls. Dividing by `sqrt(d_k)` puts the scores back at standard deviation about 1.

Concretely, softmax of `[20, 10, 0]` gives weights `[0.99995, 0.0000454, 0.0000000021]` — effectively "token 1, ignore the rest". Softmax of the same numbers divided by 10, i.e. `[2, 1, 0]`, gives `[0.665, 0.245, 0.090]` — still a preference, but a blend.

> **A precise detail worth knowing, because it is in the code.** T5 — and therefore Chronos-Bolt — **does not perform that division explicitly.** In `transformers/models/t5/modeling_t5.py` the score line is literally `scores = torch.matmul(query_states, key_states.transpose(3, 2))`, with no scaling, and then the position bias is added and softmax applied. T5 folds the scaling into how the query projection is initialized instead. So: know the textbook formula with `sqrt(d_k)` and know *why* it is there, but if someone opens the T5 source in front of you, the honest statement is *"T5 absorbs that factor into the weight initialization rather than dividing at runtime."*

### 4.4 Softmax, defined and worked

**Softmax** takes a list of arbitrary real numbers and returns positive numbers that sum to 1, preserving the ranking, with bigger gaps in the input producing bigger gaps in the output.

```
                   exp(s_i)
   softmax(s)_i = -----------------
                   sum_j exp(s_j)
```

`exp` is the exponential function, `exp(x) = e^x`. It is always positive, which is what guarantees the weights are positive; dividing by the total is what makes them sum to 1.

Worked case on `s = [2, 1, 0]`:

```
   exp(2) = 7.38906
   exp(1) = 2.71828
   exp(0) = 1.00000
   sum    = 11.10734

   7.38906 / 11.10734 = 0.66524
   2.71828 / 11.10734 = 0.24473
   1.00000 / 11.10734 = 0.09003
                        -------
                        1.00000
```

Two properties to remember. **Shift invariance:** adding the same constant to every input leaves the output unchanged, so `softmax([2,1,0])` equals `softmax([1,0,-1])` equals `softmax([12,11,10])` — only the *differences* matter. **Scale sensitivity:** multiplying all inputs by a constant changes the answer a lot, which is exactly the `sqrt(d_k)` issue above.

### 4.5 The fully worked numeric example

Three tokens, two dimensions each, small integers. Every number below was recomputed; the arithmetic chain is checkable by hand.

**Setup.** The input, one row per token:

```
        X                    x1 = [1, 0]      "token 1"
  +---------+                x2 = [0, 1]      "token 2"
  | 1     0 |                x3 = [1, 1]      "token 3"
  | 0     1 |
  | 1     1 |
  +---------+
   shape (3, 2)
```

The three learned matrices (invented for this example; in Chronos they are 512×512 and frozen):

```
        Wq                 Wk                 Wv
  +-----------+      +-----------+      +-----------+
  | 1      1  |      | 1      0  |      | 1      2  |
  | 0      1  |      | 1      1  |      | 3      1  |
  +-----------+      +-----------+      +-----------+
```

**Step 1 — project.** Each row of X times each matrix. (`q1 = x1 Wq = [1*1 + 0*0, 1*1 + 0*1] = [1, 1]`, and so on.)

```
      Q = X Wq            K = X Wk            V = X Wv
  +-----------+      +-----------+      +-----------+
  | 1      1  | q1   | 1      0  | k1   | 1      2  | v1
  | 0      1  | q2   | 1      1  | k2   | 3      1  | v2
  | 1      2  | q3   | 2      1  | k3   | 4      3  | v3
  +-----------+      +-----------+      +-----------+
```

**Step 2 — raw scores, `S = Q K^T`.** Entry (i, j) is `q_i · k_j`:

```
  s11 = q1.k1 = 1*1 + 1*0 = 1        s12 = q1.k2 = 1*1 + 1*1 = 2
  s13 = q1.k3 = 1*2 + 1*1 = 3        s21 = q2.k1 = 0*1 + 1*0 = 0
  s22 = q2.k2 = 0*1 + 1*1 = 1        s23 = q2.k3 = 0*2 + 1*1 = 1
  s31 = q3.k1 = 1*1 + 2*0 = 1        s32 = q3.k2 = 1*1 + 2*1 = 3
  s33 = q3.k3 = 1*2 + 2*1 = 4

                        key 1     key 2     key 3
                     +---------+---------+---------+
      query 1 (t1)   |    1    |    2    |    3    |
                     +---------+---------+---------+
      query 2 (t2)   |    0    |    1    |    1    |
                     +---------+---------+---------+
      query 3 (t3)   |    1    |    3    |    4    |
                     +---------+---------+---------+
                              S, shape (3, 3)
```

Rows are queries, columns are keys. Row 3 has the biggest numbers on the right, meaning token 3 is most interested in token 3 and token 2.

**Step 3 — divide by `sqrt(d_k)`.** Here `d_k = 2`, so `sqrt(2) = 1.41421`:

```
                        key 1      key 2      key 3
                     +----------+----------+----------+
      query 1        |  0.70711 |  1.41421 |  2.12132 |
                     +----------+----------+----------+
      query 2        |  0.00000 |  0.70711 |  0.70711 |
                     +----------+----------+----------+
      query 3        |  0.70711 |  2.12132 |  2.82843 |
                     +----------+----------+----------+
```

**Step 4 — softmax each row.** The exponentials needed, to four decimals:

```
   exp(0.00000) =  1.0000
   exp(0.70711) =  2.0281
   exp(1.41421) =  4.1133
   exp(2.12132) =  8.3421
   exp(2.82843) = 16.9188
```

Row 1: exps `[2.0281, 4.1133, 8.3421]`, sum `14.4835`

```
    2.0281 / 14.4835 = 0.1400
    4.1133 / 14.4835 = 0.2840
    8.3421 / 14.4835 = 0.5760      sum = 1.0000
```

Row 2: exps `[1.0000, 2.0281, 2.0281]`, sum `5.0562`

```
    1.0000 / 5.0562 = 0.1978
    2.0281 / 5.0562 = 0.4011
    2.0281 / 5.0562 = 0.4011       sum = 1.0000
```

Row 3: exps `[2.0281, 8.3421, 16.9188]`, sum `27.2890`

```
    2.0281 / 27.2890 = 0.0743
    8.3421 / 27.2890 = 0.3057
   16.9188 / 27.2890 = 0.6200      sum = 1.0000
```

The attention weight matrix, which is the thing people mean by "what the model attended to":

```
                     to token 1  to token 2  to token 3     row sum
                    +-----------+-----------+-----------+
      token 1 asks  |  0.1400   |  0.2840   |  0.5760   |   1.0000
                    +-----------+-----------+-----------+
      token 2 asks  |  0.1978   |  0.4011   |  0.4011   |   1.0000
                    +-----------+-----------+-----------+
      token 3 asks  |  0.0743   |  0.3057   |  0.6200   |   1.0000
                    +-----------+-----------+-----------+

   Every row is positive and sums to 1. Columns need not sum to anything.
```

*Check the `sqrt(d_k)` claim on row 1.* Without the division, row 1's scores are `[1, 2, 3]`, whose softmax is `[0.0900, 0.2447, 0.6652]`. With the division it is `[0.1400, 0.2840, 0.5760]`. The division made the blend gentler — visibly less winner-take-all.

**Step 5 — weighted sums, `output = weights · V`.** With `v1 = [1, 2]`, `v2 = [3, 1]`, `v3 = [4, 3]`:

```
  z1 = 0.1400*[1,2] + 0.2840*[3,1] + 0.5760*[4,3]
     dim1: 0.1400*1 + 0.2840*3 + 0.5760*4 = 0.1400 + 0.8520 + 2.3040 = 3.2960
     dim2: 0.1400*2 + 0.2840*1 + 0.5760*3 = 0.2800 + 0.2840 + 1.7280 = 2.2920
  z1 = [3.2960, 2.2920]

  z2 = 0.1978*[1,2] + 0.4011*[3,1] + 0.4011*[4,3]
     dim1: 0.1978 + 1.2033 + 1.6044 = 3.0055
     dim2: 0.3956 + 0.4011 + 1.2033 = 2.0000
  z2 = [3.0055, 2.0000]

  z3 = 0.0743*[1,2] + 0.3057*[3,1] + 0.6200*[4,3]
     dim1: 0.0743 + 0.9171 + 2.4800 = 3.4714
     dim2: 0.1486 + 0.3057 + 1.8600 = 2.3143
  z3 = [3.4714, 2.3143]
```

Final output of the attention layer:

```
        Z
  +-------------------+
  | 3.2960    2.2920  |   <- new vector for token 1
  | 3.0055    2.0000  |   <- new vector for token 2
  | 3.4714    2.3143  |   <- new vector for token 3
  +-------------------+
     shape (3, 2) -- same shape as the input X
```

**Three sanity checks you can state.**
1. The output has the *same shape* as the input, `(3, 2)`. Attention is a transformation, not a reduction — `n` tokens in, `n` tokens out.
2. Every output is a weighted average of `v1, v2, v3` with positive weights summing to 1, so every coordinate must lie inside the range of that coordinate across the V rows. Dimension 1 of V is `{1, 3, 4}`, and all three outputs are between 1 and 4. Dimension 2 is `{2, 1, 3}`, and all three are between 1 and 3.
3. Token 1's output leans toward `v3 = [4, 3]` because its weight there was 0.5760, the largest in its row. Token 2's output is nearly the midpoint of `v2` and `v3` because its two largest weights are equal at 0.4011.

*(Verified numerically with full floating-point precision: `z1 = [3.29592, 2.29198]`, `z2 = [3.00556, 2.00000]`, `z3 = [3.47135, 2.31429]`. The hand chain above agrees to four decimals.)*

**Say it in your own words:** *"Every token makes three vectors from itself — a question, an advertisement, and its content. Dot the question against everyone's advertisement to get scores, shrink them, softmax them into weights that add to one, then take that weighted mix of everyone's content. That mix is the token's new vector."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. In the worked example, why did token 1 put 0.5760 of its weight on token 3?**
A. Because its raw score there was highest: `q1 · k3 = [1,1] · [2,1] = 3`, against 2 for token 2 and 1 for token 1. The query and that key pointed most nearly the same way. After dividing by `sqrt(2)` and softmaxing, 3 became 0.5760.

**Q. What exactly does the softmax denominator do?**
A. It normalizes. `exp` makes every score positive; dividing by the sum of all the exps makes the row add to 1, which is what makes the output a proper weighted average rather than an arbitrary scaled sum.

**Q. Why not skip softmax and use the raw scores as weights?**
A. Raw scores can be negative and do not sum to anything, so the output would not be a blend — it could grow without bound as the sequence gets longer, and its scale would drift between sessions of different length. Softmax fixes both.

**Q. What breaks without the `sqrt(d_k)` division?**
A. With `d_k = 64` the scores have standard deviation about 8, so the softmax nearly always puts all its weight on one token. That makes the layer a hard selection instead of a blend, and the gradient through a saturated softmax is near zero, so pretraining would stall. Worth adding: T5 does not divide at runtime — it absorbs the factor into the initialization of the query weights.

</details>

---

## 5 · Multi-head attention

### 5.1 Why more than one head

One attention operation produces one set of weights per token. That is one question per token. But a token might reasonably want to ask several unrelated questions at once: *where did the steep rise start?*, *what was the overall level?*, *is the most recent stretch flat?* One weight matrix must compromise between all of them.

**Multi-head attention** runs several independent attention operations in parallel on different slices of the vector, then joins the answers. Plain-words version: **several people read the same glucose series at the same time, each looking for something different, then pool their notes.**

### 5.2 The numbers for our model

From `chronos-bolt-small`'s config (verified in the downloaded checkpoint):

```
   d_model = 512      num_heads = 8      d_kv = 64
   8 heads x 64 dimensions each = 512
```

The split is exactly even and exactly fills `d_model`, which is the usual design. The consequence is that **multi-head attention costs the same as single-head attention** — the four projections are 512→512 either way; the only difference is that the 512 outputs are cut into 8 slices of 64 and each slice runs its own softmax.

```
   one token: 512 numbers
        |
        |  x Wq (512x512), x Wk, x Wv   -- one big matrix multiply
        v
   512 numbers, then SPLIT into 8 slices of 64
   +------+------+------+------+------+------+------+------+
   |head 1|head 2|head 3|head 4|head 5|head 6|head 7|head 8|
   |  64  |  64  |  64  |  64  |  64  |  64  |  64  |  64  |
   +------+------+------+------+------+------+------+------+
      |      |      |      |      |      |      |      |
      |      |      |   each head runs the ENTIRE story of section 4
      |      |      |   on its own 64 dimensions: its own (n x n) score
      |      |      |   matrix, its own softmax, its own weighted sum
      v      v      v      v      v      v      v      v
   +------+------+------+------+------+------+------+------+
   |  64  |  64  |  64  |  64  |  64  |  64  |  64  |  64  |
   +------+------+------+------+------+------+------+------+
        |
        |  CONCATENATE back to 512
        v
   512 numbers  --> x Wo (512x512, no bias) --> 512 numbers out
```

The final `Wo` projection matters: without it the 8 head outputs would just sit side by side in fixed slots, never mixed. `Wo` lets the model combine what different heads found.

**Per-head parameter arithmetic**, which is the kind of thing you can be asked to derive live:

```
   Wq, Wk, Wv, Wo : 4 x (512 x 512) = 4 x 262,144 = 1,048,576   (no biases in T5)
```

Note there are **8 separate score matrices per layer**, one per head. For a 19-token session that is 8 × 19 × 19 = 2,888 numbers per layer.

**Say it in your own words:** *"Instead of one question per token we let it ask eight, by cutting the 512 numbers into eight slices of 64 and running attention separately in each slice. Then we glue the eight answers back into 512 and mix them with one more matrix. Eight heads of 64 cost the same as one head of 512."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. Do 8 heads make the model 8 times bigger?**
A. No. `8 × 64 = 512 = d_model`, so the projections are the same size as they would be with one head. The heads split the existing width rather than adding to it.

**Q. What would you lose with only one head?**
A. The ability to attend to several things at once for the same token. With one set of weights per token per layer, a token that needs both "the start of the rise" and "the recent flat stretch" has to average the two into one weight pattern.

**Q. What is `Wo` for?**
A. It mixes the concatenated head outputs. Without it, dimensions 1-64 of the output would always come only from head 1, and no later computation could combine head 1's finding with head 5's until the feed-forward sub-layer.

</details>

---

## 6 · The rest of an encoder block

Attention is one of two sub-layers. Each of Chronos-Bolt-small's **6 encoder blocks** is: self-attention sub-layer, then feed-forward sub-layer, each wrapped in a normalization and a residual connection.

### 6.1 The position-wise feed-forward network

```
   Linear(512 -> 2048)  ->  ReLU  ->  Linear(2048 -> 512)
```

Confirmed from the config: `d_ff = 2048`, `dense_act_fn = "relu"`, `is_gated_act = false`, and both linear layers have no bias in T5.

**"Position-wise"** means the *same* small network is applied to each token independently — token 4's 512 numbers go in, 512 numbers come out, with no reference to any other token. All cross-token mixing happens in attention; all per-token nonlinear reshaping happens here. That division of labour is the whole design.

**What it is for.** Attention can only produce weighted *averages* of value vectors — a linear operation given the weights. Without a nonlinearity the stack would collapse into one big linear map. Widening to 2048, applying ReLU (which zeroes negatives and passes positives through unchanged; see `06_DEEP_LEARNING_CORE.md`), then narrowing back to 512 gives each token a chance to compute nonlinear functions of its own mixed content — "this token now represents a fast rise from a low base" rather than just a blend of numbers.

It is also where most of the parameters live:

```
   attention:     4 x 512 x 512 = 1,048,576
   feed-forward:  512 x 2048 + 2048 x 512 = 1,048,576 + 1,048,576 = 2,097,152
   2 layer norms: 512 + 512 = 1,024
                                 ---------
   per encoder block:            3,146,752
```

Two thirds of every block is the feed-forward network.

### 6.2 Residual connections

A **residual** (or **skip**) connection adds a sub-layer's input to its output: instead of `x -> F(x)`, the block computes `x -> x + F(x)`. Two reasons this matters:

- **Identity is free.** If a sub-layer has nothing useful to add, it can output near zero and the input passes through untouched. Without a residual, every layer must actively reconstruct everything it wants to keep, and a 6-layer stack has 6 chances to corrupt it.
- **Gradients get a short path.** During training the correction signal flows back through the `+` unchanged as well as through `F`, so it reaches early layers without being multiplied through every intermediate transformation. This is the fix for the same vanishing-gradient problem that limits LSTMs, and it is the single reason deep stacks are trainable at all.

### 6.3 Layer normalization

**Normalization** here means rescaling a token's 512 numbers so their overall size is controlled, which keeps activations from growing or shrinking as they pass through 6 blocks. It acts across the 512 features of one token, independently for each token — which is why it works fine on variable-length sequences.

T5's version, and therefore Chronos-Bolt's, is a **root-mean-square norm**: divide each token's vector by the root mean square of its entries, then multiply by 512 learned scale numbers. Verified in the installed source (`transformers/models/t5/modeling_t5.py`, `class T5LayerNorm`): *"No bias and no subtraction of mean."* So it is a pure rescaling — no centering, no shift — with `layer_norm_epsilon = 1e-6` added inside the square root for numerical safety.

**Where it sits.** T5 uses **pre-normalization**: the norm is applied to the sub-layer's *input*, and the residual is added around the whole thing. From the installed source:

```python
normed_hidden_states = self.layer_norm(hidden_states)
attention_output = self.SelfAttention(normed_hidden_states, ...)
hidden_states = hidden_states + self.dropout(attention_output[0])
```

That is `x + Dropout(SelfAttn(LayerNorm(x)))`. The original 2017 transformer paper did the opposite — `LayerNorm(x + SelfAttn(x))`, called post-normalization. Pre-norm is now standard because it trains more stably at depth. If you are asked "where does layer norm go?", the correct answer for *our* model is *before* each sub-layer.

### 6.4 The block, drawn

```
     input:  n tokens x 512
        |
        +--------------------------------------------+   (residual)
        v                                            |
   RMS LayerNorm (scale only, no mean subtraction)    |
        v                                            |
   Multi-head self-attention                         |
     - 8 heads x 64 dims                             |
     - Wq, Wk, Wv, Wo: 512x512, no bias              |
     - relative position bias added to the scores    |
        v                                            |
   Dropout (0.1; inactive at inference)              |
        v                                            |
       ( + ) <------------------------------------- -+
        |
        +--------------------------------------------+   (residual)
        v                                            |
   RMS LayerNorm                                     |
        v                                            |
   Linear 512 -> 2048                                |
        v                                            |
   ReLU                                              |
        v                                            |
   Linear 2048 -> 512                                |
        v                                            |
   Dropout (0.1)                                     |
        v                                            |
       ( + ) <------------------------------------- -+
        |
        v
     output: n tokens x 512    ->  into the next block

   6 identical blocks (3,146,752 parameters each), then one final RMS LayerNorm (512).
   Encoder total: 6 x 3,146,752 + 256 (position bias, block 0 only) + 512 = 18,881,280
```

**Dropout** (rate 0.1) randomly zeroes 10% of the activations during *training* only, as a regularizer. Our pipeline runs inference under `torch.no_grad()` with the model in eval mode, so dropout does nothing for us — worth knowing so you do not claim our embeddings are stochastic. They are deterministic, which is why `cgm_tsfm/encoders.py` can cache them to a `.npy` file keyed by a hash of the input.

**Say it in your own words:** *"Each block does two things. Attention mixes information across tokens. Then a small two-layer network reshapes each token on its own — widen to 2048, ReLU, back to 512. Each is wrapped so the input is added back on afterwards, and each gets its input rescaled first. Six of those stacked."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Why is a nonlinearity needed at all?**
A. Attention with fixed weights is a weighted average, which is linear. Stack linear operations and you get one linear operation. The ReLU in the feed-forward sub-layer is what makes six blocks more expressive than one.

**Q. What does "position-wise" mean?**
A. The same two-layer network is applied to each token separately, with no access to other tokens. Attention is the only place tokens interact.

**Q. Where does layer norm sit in our model, before or after the sub-layer?**
A. Before — T5 is pre-norm: `x + Dropout(SubLayer(LayerNorm(x)))`. The original 2017 paper was post-norm. And T5's norm is RMS-style: rescale only, no mean subtraction, no bias.

**Q. Does dropout affect our embeddings?**
A. No. Dropout is a training-time regularizer; we run the frozen encoder in eval mode under `torch.no_grad()`, so the 512 numbers are deterministic for a given glucose list. That is why they can be cached to disk.

</details>

---

## 7 · How the model knows the ORDER

This is the section our older documents never covered, and it is a natural thing to be asked.

### 7.1 Attention alone cannot tell order

Look again at the worked example. Nothing in `softmax(QK^T/sqrt(d_k))V` refers to *where* a token sits. Every token is projected by the same Wq, Wk, Wv; every score is a dot product between content vectors. So if you shuffle the input rows, the output rows come out shuffled in exactly the same way and are otherwise identical. The operation is **permutation-equivariant** — reorder the input, and you get the same set of outputs, reordered.

Verified on the worked example: swapping tokens 1 and 2 in `X` produces outputs `[3.00556, 2.00000]`, `[3.29592, 2.29198]`, `[3.47135, 2.31429]` — exactly `z2, z1, z3`. Same set, permuted.

**Why that is fatal without a fix.** `[100, 150, 200, 250]` and `[250, 200, 150, 100]` would produce the same set of token vectors, and after mean-pooling over tokens — which is what our pipeline does — the *same 512 numbers*. A rise and a fall would be indistinguishable. Position information has to be injected deliberately.

### 7.2 The original transformer's answer: sinusoidal absolute encodings

The 2017 paper added, to each token's input vector, a fixed pattern of sines and cosines at different frequencies determined by the token's absolute index. Token 5 got a different additive pattern than token 6. A common variant makes those patterns learned rather than fixed — a lookup table of one vector per position. Either way the token carries "I am at position 5" in its content. Chronos-Bolt does **not** do this.

### 7.3 What our model actually does: relative position bias

T5's mechanism — and Chronos-Bolt is a T5 — adds a learned number directly **to the attention score**, chosen by how far apart the query and key are:

```
   score(i, j) = q_i . k_j  +  b[ head, bucket(j - i) ]
```

So a token never learns "I am fifth". Instead, every score learns an adjustment for "the token I am looking at is 3 positions before me".

**The verified specifics for `chronos-bolt-small`:**

| Fact | Value | How verified |
|---|---|---|
| `relative_attention_num_buckets` | **32** | checkpoint `config.json` |
| `relative_attention_max_distance` | **128** | checkpoint `config.json` |
| The parameter | `Embedding(32, 8)` = **256 numbers** | inspected the loaded model |
| Where it lives | `encoder.block.0.layer.0.SelfAttention.relative_attention_bias` — **block 0 only** | only block 0 has the parameter |
| Shared across layers? | **yes** — block 0 computes the bias and it is passed to all 6 blocks | T5 threads `position_bias` through |
| Per-head? | **yes** — the 8 columns are one bias table per head | shape `(32, 8)` |

**Why buckets.** You do not want a separate learned number for every possible distance — that would not generalize to distances unseen in pretraining. So distances are grouped: near distances get their own bucket, far distances share coarse buckets on a roughly logarithmic scale, and everything beyond 128 lands in the last bucket. The exact mapping, computed by calling T5's own bucketing function:

```
   distance      bucket        (looking BACKWARD: key is before query)
   ---------------------
      0             0          <- itself
      1             1
      2             2
      3             3
      4             4
      5             5
      6             6
      7             7          exact, one bucket each
     8..11          8
    12..15          9
    16..23         10          coarse, shared
     ...          ...
    >=128          15          everything very far away

   Looking FORWARD (key is after query) uses a second set: buckets 17..31,
   with the same shape. Bucket 0 is distance zero. The encoder is
   bidirectional, so both directions are used.
```

So the *sign* is kept (before vs after) and the *magnitude* is kept exactly out to 7 and approximately beyond that.

**What this means for our short sessions.** A 288-reading session is 19 tokens, so the largest distance we ever see is 18. From the table, distances 0-18 use buckets 0-10 backward and 17-26 forward — about 21 of the 32 available. Nothing is ever clipped at 128; we operate entirely in the fine-grained end of the scheme. And distances above 7 (that is, more than 7 patches ≈ 9.3 hours apart) get lumped into shared buckets, so the model's sense of "very far apart" is coarse — which for us only affects the long overnight sessions.

### 7.4 The practical consequence, and the honest caveat

Two things follow, and both are worth volunteering before you are asked:

1. **The model knows spacing, not clock time.** It knows patch 3 comes two positions after patch 1. It does not know whether the test was at 8am or 11pm. Time of day is a genuinely separate variable, and it is not in the model's input at all.

2. **It assumes evenly spaced points, and our preprocessing breaks that.** Relative position bias is defined on *index* distance, so index distance is only proportional to elapsed time if the samples are evenly spaced. Dexcom G6 samples every 5 minutes, so they should be. But `cgm_tsfm/data.py` **drops** unparseable and missing readings rather than filling them (`01_FACT_SHEET.md` §B), which silently splices the remaining readings together — so after a gap, "3 positions back" no longer means 15 minutes back. This affects a small number of sessions, but it is a real flaw in the pipeline and stating it yourself is much better than having it found.

**Say it in your own words:** *"Attention on its own can't tell order — shuffle the input and you get the same outputs shuffled. T5 fixes it by adding a learned number to each attention score based on how far apart the two positions are, grouped into 32 distance buckets, one table per head, computed once in the first block and reused by all six. So the model knows spacing, not clock time — and it assumes even spacing, which our dropping of missing readings quietly breaks."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Prove attention alone is order-blind.**
A. Every token goes through the same Wq, Wk, Wv, and every score is a dot product of content vectors, with no index anywhere in the formula. Permute the input rows and the score matrix's rows and columns permute identically, so the outputs permute identically. I checked it on my 3-token example: swapping tokens 1 and 2 returns exactly `z2, z1, z3`.

**Q. Does Chronos-Bolt use sinusoidal positional encodings?**
A. No. That is the 2017 paper. Chronos-Bolt is a T5, which uses relative position bias: a learned number added to the attention score for each query-key pair, keyed by their distance bucket. There are 32 buckets, a maximum distance of 128, one table per head, `Embedding(32, 8)` = 256 numbers, held in encoder block 0 and shared with all 6 blocks.

**Q. Why bucket the distances instead of learning one value per distance?**
A. To generalize. Buckets are exact for distances 1 through 7 and increasingly coarse beyond, so a distance never seen during pretraining still maps into a bucket that was trained, and everything past 128 collapses into one. For us the longest sequence is 19 tokens, so we never leave the fine-grained end.

**Q. Does the model know what time of day the test happened?**
A. No. It only sees the glucose values and their spacing. Time of day is a separate variable that is not in the input — and one that plausibly affects both glucose and cognitive performance.

</details>

---

## 8 · Encoder vs decoder

Two plain definitions:

- An **encoder** reads a whole sequence at once and returns one vector per position. Every position may look at every other position, in both directions. Its job is to *represent*.
- A **decoder** produces a sequence one element at a time and is only allowed to look backwards. A **causal mask** sets the attention score to negative infinity for any future position, so its softmax weight becomes zero. Its job is to *generate*.

Sketched:

```
  ENCODER (bidirectional -- what we use)
      t1 <---> t2 <---> t3 <---> t4        every arrow allowed
      "token 2 may look at token 4"

  DECODER (causal -- masked)
      t1 ----> t2 ----> t3 ----> t4        forward arrows only
      "token 2 may NOT look at token 3 or 4"
```

**What Chronos-Bolt is, and what we run.** The checkpoint is a T5-style encoder-decoder: `is_encoder_decoder: true`, `num_layers: 6`, `num_decoder_layers: 6`. But two things keep it from being anything text-like. First, its **decoder runs exactly one step** — it is not producing a sequence one element at a time; it emits the whole forecast in a single pass. Second, and more to the point for us: **our pipeline never runs the decoder at all.**

`cgm_tsfm/encoders.py` calls `pipeline.embed(batch)`. In the installed `chronos` package, `ChronosBoltPipeline.embed` calls `self.model.encode(context=...)` and returns the encoder's last hidden state. The decoder is loaded into memory and never executed. Verified parameter split:

| Part | Parameters | Executed by `.embed()`? |
|---|--:|---|
| `shared` (the `[REG]` token embedding) | 1,024 | yes |
| `input_patch_embedding` | 1,133,568 | yes |
| `encoder` (6 blocks + final norm + position bias) | 18,881,280 | yes |
| **used by our pipeline** | **20,015,872** | |
| `decoder` (6 blocks) | 25,175,808 | **no** |
| `output_patch_embedding` (the forecast head) | 2,526,336 | **no** |
| **loaded but never run** | **27,702,144** | |
| **total** | **47,718,016** | |

So of the 47.7M parameters in the checkpoint, **42%** are what produce our 512 numbers and **58%** sit idle. This is a good fact to have ready, because it is exactly the sort of thing an examiner probes — and the answer "we run the encoder only" is much stronger with the split attached.

**Say it in your own words:** *"An encoder reads the whole sequence and gives one vector per position, with every position free to look at every other. A decoder generates and can only look backwards. Chronos-Bolt has both, but we only ever run the encoder — 20 million of its 47.7 million parameters — and even its decoder only runs one step, so nothing text-like is happening."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. Is Chronos generating text?**
A. No. It is a T5-*style* architecture — same block design, relative position bias, encoder-decoder shape — but its inputs are blocks of numbers, not words, its vocabulary size is 2 (it has no word vocabulary), and its decoder runs a single step to emit nine forecast quantiles. And we never run the decoder.

**Q. Why use the encoder rather than the forecast?**
A. We do not want a glucose forecast. We want a fixed-length summary of the session to hand to a regressor. The encoder's last hidden state is that summary — one 512-vector per token, which we mean-pool over tokens into 512 numbers per session (`09_THE_TWO_ARMS.md`).

**Q. What is a causal mask?**
A. Setting attention scores to negative infinity for future positions, so their softmax weight is exactly zero and a position cannot see anything after itself. Decoders need it; our encoder does not use it and is fully bidirectional.

</details>

---

## 9 · What attention costs

Attention computes a score for every (query, key) pair, so an `n`-token sequence needs an `n × n` matrix — **O(n²)** in time and memory, per head, per layer. This is the standard criticism of transformers and the reason a large literature exists on making it cheaper.

**For us it is a non-issue**, and the reason is patching:

```
   session        readings   tokens   n^2 pairs   x 8 heads x 6 layers
   -----------------------------------------------------------------
   shortest            3        2          4            192
   median             36        4         16            768
   longest           288       19        361         17,328
   -----------------------------------------------------------------
   if fed reading-by-reading (no patching):
   longest           288      288     82,944      3,981,312
```

Patching 16 readings into one token cuts the longest session's pair count by **82,944 / 361 ≈ 230×**. Our attention matrices are 2×2 up to 19×19 — the sort of thing that is free.

Where it *does* matter: a model reading a full month of 5-minute CGM at reading-level resolution would be 8,640 positions, i.e. 74.6 million pairs per head per layer. That is when you need patching, sparse attention, or a different architecture. `08_CHRONOS.md` covers the patching mechanics; here just note that patching is not only a cost trick — it also gives each token a local window of shape to work with, which is closer to the CNN's strength than to reading one number at a time.

**Say it in your own words:** *"Attention compares every position with every other, so cost grows with the square of the sequence length. Chronos avoids the problem by gluing 16 readings into one token, so our longest session is 19 tokens instead of 288 — about 230 times fewer pairs. At this size the cost is nothing."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q. Is attention's quadratic cost a problem in this project?**
A. No. Our sequences are 2 to 19 tokens because 16 readings become one patch. The largest attention matrix in the whole project is 19×19.

**Q. When would it become a problem?**
A. When `n` reaches thousands. A month of 5-minute CGM read one reading at a time is 8,640 positions, about 74.6 million pairs per head per layer.

**Q. Name a second benefit of patching besides cost.**
A. Each token starts out already describing a local stretch of shape — 16 readings, 80 minutes — instead of a single number, so attention operates over meaningful segments rather than individual points.

</details>

---

## 10 · What a transformer does NOT give you

Read this section twice. Overclaiming here is how the whole result gets doubted.

**1. It cannot find a relationship that is not in the data.** In our 956 sessions, the strongest correlation between *any* simple glucose statistic and *any* of the three scores is r = −0.092 — about 0.85% of the variance (`01_FACT_SHEET.md` §E). No architecture manufactures signal.

**2. It has no notion of physical units.** 142 is not "mg/dL" to the model; it is a number. Worse, Chronos rescales each series internally before the encoder sees it, so the absolute height of the curve is largely discarded. Our own check shows this precisely: the same embeddings and folds predict glucose **variability** at R² = 0.462 but mean glucose at only 0.070. Chronos speaks to **shape**, not **level**. (What saves the conclusion is that the 43 hand-crafted features *do* encode level, and they were flat too.)

**3. It was pretrained for the wrong task.** Chronos was trained to forecast the next stretch of a series. Nothing in that objective asks it to produce a summary useful for predicting a cognitive score. Its features encode whatever helps forecasting — recent trend, volatility, periodicity — and there is no guarantee that overlaps with what a cognitive test responds to. This is the standard caveat on frozen features, and it is a legitimate limitation of the design, not an excuse.

**4. It has no access to anything outside the glucose list.** No age, no time of day, no insulin, no sleep, no practice effect from repeated testing. One channel, as scoped.

**5. Bigger did not help — and that is the expected shape of the evidence.** From `results/sweep_real.md` (`01_FACT_SHEET.md` §I), mean R² across the three scores:

| Checkpoint | d_model | mean R² |
|---|--:|--:|
| bolt-tiny | 256 | −0.049 |
| bolt-mini | 384 | −0.052 |
| bolt-small | 512 | −0.052 |
| **bolt-base** | **768** | **−0.063**  ← worst |
| t5-small | 512 | −0.057 |

The **largest checkpoint was the worst**. That is not a paradox, and the explanation is not about transformers at all. A larger encoder produces *wider* features — 768 numbers instead of 256 — for the same 956 rows split into folds of about 765 training sessions. When the features carry no signal about the target, extra width only gives the downstream regressor more ways to fit fold-specific noise, which then fails to transfer to held-out children. The pattern "accuracy flat or slightly worse as the model grows" is the signature of a **data-limited** problem rather than a capacity-limited one. If the encoder had been the bottleneck, scaling it up would have helped; it did not, so it was not.

**Say it in your own words:** *"A transformer isn't magic. It can't find a relationship that isn't there, it doesn't know mg/dL, it rescales away the absolute level, and it was trained to forecast rather than to summarize for our regression. And it only sees glucose — no age, no time of day. Our sweep shows the biggest model doing the worst, which is what you see when the data is the limit, not the model."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q. Why would a bigger encoder ever be worse?**
A. It hands the regressor wider features — 768 instead of 256 — on the same ~765 training sessions per fold. If the features contain nothing about the score, extra width only supplies more ways to fit noise in the training children, which does not transfer to the held-out ones. Mean R² went from −0.049 at bolt-tiny to −0.063 at bolt-base.

**Q. Does that mean transformers are the wrong tool here?**
A. It means the encoder was not the bottleneck. Three independent things point the same way: the strongest raw glucose-to-score correlation in the data is r = −0.092; the level-aware hand-crafted features were also no better than guessing the average; and scrambling the scores 200 times produces results indistinguishable from the real ones (p = 1.000 / 1.000 / 0.995). Those are data facts, not architecture facts.

**Q. What is the strongest criticism of using frozen Chronos features?**
A. That the pretraining objective is forecasting, not "summarize this window in a way that predicts cognition", so there is no guarantee the 512 numbers encode what we care about. Two honest responses: the same embeddings recover glucose variability at R² = 0.462, so they do carry real information about the series; and the alternative of fine-tuning is closed off by having 956 sessions from 20 children.

**Q. What is the single biggest thing missing from the model's input?**
A. A uniform lookback window. The input is currently everything since the child's previous test, so it ranges from 3 to 288 readings. Time of day is a close second — it plausibly affects both glucose and test performance and is not in the input at all.

</details>

---

## 11 · Bridge to `08_CHRONOS.md`

This chapter stayed general on purpose. The next one is Chronos-specific and does not repeat any of the above.

`08_CHRONOS.md` covers: how a glucose list becomes tokens (**patching** — 16 readings per patch, stride 16, and how sessions that are not a multiple of 16 are handled); **instance normalization**, the per-series rescaling that removes the absolute glucose level and is the reason mean glucose is hard to recover from the embeddings; the **`[REG]` token**, the one extra bookkeeping token appended after the patches; the mask channel that lets missing readings pass through without being filled; the exact tensor shapes end to end, from a 36-value list to a 512-number row; and the checkpoint family — bolt-tiny through bolt-base plus chronos-t5-small — with what differs between them.

If you are asked "how does Chronos work under the hood", that is the file. If you are asked "why a transformer at all", it is this one.

---

## The 6 hardest questions, with model answers

**1. "Why not an LSTM?"**

> Two reasons, and the second is the real one. Mechanically, an LSTM reads one value at a time and carries a memory vector, so for a three-hour session the first reading has to survive 35 consecutive rewrites of that memory, and during training the correction signal has to travel back through all 35 steps, which drives it toward zero. It also cannot be parallelized across time. A transformer relates any two positions in one weighted sum, and for us that is one hop whether the session is 4 tokens or 19. But the decisive reason is data: we have 956 sessions from 20 children, which is not enough to train any sequence model from scratch, so the model has to be pretrained — and there is no pretrained general-purpose LSTM for time series that I can download and freeze, while there are several transformers. Choosing "pretrained" is what chose "transformer".

**2. "How does the model know the order of the readings?"**

> Not from attention — attention alone is order-blind, because every token goes through the same query, key and value projections and every score is a dot product of content, with no index anywhere. Shuffle the input and the outputs come out shuffled and otherwise identical; I verified that on a 3-token example. Chronos-Bolt is a T5, and T5 injects order as **relative position bias**: a learned number added to each attention score depending on how far apart the query and key are. Distances are grouped into 32 buckets up to a maximum of 128 — exact for distances 1 through 7, then increasingly coarse — and there is one bias table per head, stored as an `Embedding(32, 8)` in encoder block 0 and shared with all 6 blocks. Note this is *not* the sinusoidal absolute encoding from the 2017 paper. Two consequences for us: the model knows spacing but not clock time, and it assumes evenly spaced samples — which our preprocessing breaks, because we drop missing readings instead of filling them.

**3. "Why did the bigger model do worse?"**

> Because our limit is the data, not the model. The sweep gives mean R² of −0.049 for bolt-tiny at 256 dimensions and −0.063 for bolt-base at 768 — the largest checkpoint was the worst. A bigger encoder produces wider features for the same roughly 765 training sessions per fold, and if those features carry nothing about the cognitive score, extra width only gives the downstream regressor more ways to fit noise specific to the training children, which then fails on the held-out ones. Flat-to-slightly-worse as capacity grows is the signature of a data-limited problem. If the encoder had been the bottleneck, scaling it would have helped.

**4. "Show me self-attention. Compute it."**

> Take three tokens with two dimensions: `x1 = [1,0]`, `x2 = [0,1]`, `x3 = [1,1]`. Three learned matrices give a query, a key and a value per token; with `Wq = [[1,1],[0,1]]`, `Wk = [[1,0],[1,1]]`, `Wv = [[1,2],[3,1]]` I get `Q = [[1,1],[0,1],[1,2]]`, `K = [[1,0],[1,1],[2,1]]`, `V = [[1,2],[3,1],[4,3]]`. Dot every query against every key: the score matrix is `[[1,2,3],[0,1,1],[1,3,4]]`. Divide by `sqrt(d_k) = sqrt(2) = 1.41421`, giving rows `[0.707, 1.414, 2.121]`, `[0, 0.707, 0.707]`, `[0.707, 2.121, 2.828]`. Exponentiate — `exp(0.707) = 2.0281`, `exp(1.414) = 4.1133`, `exp(2.121) = 8.3421`, `exp(2.828) = 16.9188`, `exp(0) = 1` — and normalize each row: `[0.1400, 0.2840, 0.5760]`, `[0.1978, 0.4011, 0.4011]`, `[0.0743, 0.3057, 0.6200]`. Each row is positive and sums to one. Multiply by V: token 1 gets `[3.296, 2.292]`, token 2 gets `[3.006, 2.000]`, token 3 gets `[3.471, 2.314]`. Same shape in, same shape out, and every output sits inside the range of the value vectors because it is a weighted average of them.

**5. "What is a token, and what is a patch?"**

> A patch is a short block of consecutive readings glued together and treated as one unit — Chronos-Bolt uses blocks of 16, so 16 readings, 80 minutes, become one patch. A token is one position in the sequence the transformer actually processes, and each token is a vector of 512 numbers; for us one token is one patch, plus one extra bookkeeping token appended at the end. I checked the counts by running the model: a 3-reading session gives 2 tokens, our median 36-reading session gives 4, and the longest 288-reading session gives 19. So attention here operates over a handful of positions, not thousands.

**6. "Why a transformer, in one paragraph, and what does the architecture actually buy you?"**

> The reason is pretraining. With 20 children and 956 sessions we cannot train a sequence model from scratch — our own 132,609-parameter head on about 612 training rows is already the worst thing in the project — so we need a model that already learned what time series look like from a lot of other data, used frozen. Every strong pretrained time-series model available is a transformer, so that decision made the architecture decision. What the architecture then buys us, genuinely: any two positions relate in one step rather than through a chain of updates, it handles variable-length input natively with no padding or resampling, and it parallelizes. What it does not buy us is signal that is not in the data, which is what we found.

---

## Every term introduced, defined once

| Term | One-line definition |
|---|---|
| **Sequence** | An ordered list of values; here, one session's glucose readings at 5-minute spacing. |
| **Feed-forward / fully-connected network / MLP** | A stack of layers where every input is multiplied by a learned weight and summed; needs a fixed input size. |
| **Recurrent network (RNN)** | Reads one element at a time, carrying a memory vector that is updated at each step. |
| **Hidden state** | The memory vector a recurrent network carries; also used for a transformer's per-token vector. |
| **LSTM / GRU** | Recurrent networks with learned gates controlling what to keep and what to erase. |
| **Vanishing gradient** | The training correction shrinking toward zero as it is multiplied back through many steps. |
| **1-D CNN / temporal convolution** | Slides small learned filters along the sequence; parameter-cheap and order-aware, but local. |
| **Receptive field** | How much of the input a single output value can see; for width-3 filters, `1 + 2L` after `L` layers. |
| **Dilation** | Spacing a convolution filter's taps out (every 2nd, 4th, 8th value) to widen the receptive field cheaply. |
| **Transfer learning** | Reusing representations learned on one task for a different task. |
| **Pretrained / frozen** | Weights arrive already trained; frozen means they are never updated (we use `torch.no_grad()`). |
| **Token** | One position the transformer processes; for us one patch, carried as a 512-number vector. |
| **Patch** | A block of consecutive readings treated as one unit; Chronos-Bolt uses 16 readings per patch. |
| **Embedding** | A fixed-length vector of numbers standing in for something; here 512 numbers per token, mean-pooled to 512 per session. |
| **d_model** | The width of a token's vector inside the model; 512 for `chronos-bolt-small`. |
| **Self-attention** | Each position's output is a weighted blend of all positions' values, with weights computed from the content. |
| **Query (Q)** | The vector expressing what a token is looking for; a learned linear projection of the token. |
| **Key (K)** | The vector a token advertises, matched against other tokens' queries. |
| **Value (V)** | The content a token contributes to whoever attends to it. |
| **Attention score** | A dot product `q_i · k_j`, one per query-key pair, forming an `n × n` matrix. |
| **d_k** | The per-head width used in the `sqrt(d_k)` scaling; 64 for `chronos-bolt-small`. |
| **Softmax** | `exp(s_i) / sum_j exp(s_j)`; turns arbitrary numbers into positive weights summing to 1. |
| **Attention weights** | The post-softmax matrix; each row is positive and sums to 1. |
| **Multi-head attention** | Several attention operations run in parallel on slices of the vector, then concatenated and projected. |
| **Head** | One such slice; 8 heads of 64 dimensions each in our model. |
| **Position-wise feed-forward** | `Linear(512→2048) → ReLU → Linear(2048→512)`, applied to each token independently. |
| **ReLU** | Outputs zero for negative inputs and the input unchanged for positive ones. |
| **Residual / skip connection** | Adding a sub-layer's input to its output (`x + F(x)`), so identity is free and gradients have a short path. |
| **Layer normalization** | Rescaling a token's 512 numbers so their overall size is controlled; T5's version rescales only, with no mean subtraction and no bias. |
| **Pre-norm** | Normalizing a sub-layer's input rather than its output; what T5 and therefore Chronos-Bolt does. |
| **Dropout** | Randomly zeroing a fraction of activations during training only; 0.1 here, and inactive for us. |
| **Encoder** | Reads a whole sequence and returns one vector per position, with every position free to look at every other. |
| **Decoder** | Produces output while only allowed to look backwards. |
| **Causal mask** | Setting future positions' attention scores to negative infinity so their weight is exactly zero. |
| **Permutation-equivariant** | Reorder the input and the output reorders identically; true of attention before position information is added. |
| **Absolute positional encoding** | A per-position pattern added to the token's vector; the 2017 paper's sinusoidal scheme. Chronos-Bolt does *not* use this. |
| **Relative position bias** | A learned number added to each attention score based on the query-key distance; T5's scheme, 32 buckets, max distance 128. |
| **Bucket** | A group of distances sharing one learned bias value; exact for 1-7, coarser beyond. |
| **O(n²)** | Cost growing with the square of sequence length, because every pair of positions is scored. |

---

## Ten-sentence summary

1. Our input is one session's glucose readings, 3 to 288 of them at 5-minute spacing, where the order carries meaning and the length varies 96-fold — so a plain fixed-input network is the wrong shape for the problem.
2. Four families could handle it: hand-built summary statistics (which is our comparison arm, 43 numbers), recurrent networks, 1-D convolutions, and transformers; they differ mainly in how far information can travel and how much data they need.
3. A transformer relates any two positions in one weighted sum, where an LSTM would need 35 sequential memory updates across a three-hour session and a width-3 CNN would need 18 layers.
4. But the decisive reason we use one is pretraining, not architecture: 956 sessions from 20 children cannot train a sequence model from scratch, so the model has to arrive already trained, and every strong pretrained time-series model available is a transformer.
5. Self-attention gives each token three learned projections of itself — a query, a key and a value — scores every query against every key with a dot product, divides by `sqrt(d_k)` so the softmax does not saturate, softmaxes each row into positive weights summing to 1, and returns each token's weighted blend of all the values.
6. Multi-head attention runs 8 of those in parallel on 8 slices of 64 dimensions, concatenates the results back to 512, and mixes them with one more projection — costing the same as a single head because 8 × 64 = 512.
7. Each of the 6 encoder blocks pairs that attention with a position-wise `Linear(512→2048) → ReLU → Linear(2048→512)` network, wrapping both in residual connections that make identity free and give gradients a short path, with root-mean-square layer normalization applied to each sub-layer's input.
8. Attention alone is order-blind, so T5 — and therefore Chronos-Bolt — adds a learned relative position bias to each attention score, keyed by query-key distance in 32 buckets up to distance 128, one table per head, computed in block 0 and shared across all 6 blocks; the model therefore knows spacing but not clock time, and assumes even spacing that our dropping of missing readings breaks.
9. We run the encoder only: `pipeline.embed()` calls the encoder and returns its last hidden state, so 20,015,872 of the checkpoint's 47,718,016 parameters do the work and the 27,702,144 in the decoder and forecast head are loaded but never executed; the quadratic cost of attention is irrelevant because patching turns 288 readings into 19 tokens.
10. A transformer cannot create a relationship that is not in the data, it rescales away the absolute glucose level, and it was pretrained to forecast rather than to summarize for our regression — which is why the largest checkpoint scored the worst (bolt-base −0.063 against bolt-tiny −0.049) and why the honest conclusion is that the limit here is the data, not the model.

---

*Cross-references: `01_FACT_SHEET.md` (every number with its derivation), `06_DEEP_LEARNING_CORE.md` (layers, ReLU, dropout, backpropagation, the trainable head), `08_CHRONOS.md` (patching, instance normalization, the `[REG]` token, exact shapes, checkpoint family), `09_THE_TWO_ARMS.md` (what happens to the 512 numbers), `05_EVALUATION.md` (grouped cross-validation and why R² is negative).*

*Provenance. Architecture facts were read from the downloaded checkpoint at `hf_cache/hub/models--amazon--chronos-bolt-small/.../config.json` and from the loaded model itself (`d_model` 512, `num_heads` 8, `d_kv` 64, `d_ff` 2048, `num_layers` 6, `num_decoder_layers` 6, `dropout_rate` 0.1, `dense_act_fn` relu, `is_gated_act` false, `relative_attention_num_buckets` 32, `relative_attention_max_distance` 128). Parameter counts were summed from `model.named_parameters()`: total 47,718,016; encoder 18,881,280 + `input_patch_embedding` 1,133,568 + `shared` 1,024 = 20,015,872 used by `.embed()`; `decoder` 25,175,808 + `output_patch_embedding` 2,526,336 = 27,702,144 never executed. Token counts (2 / 4 / 19 for 3 / 36 / 288 readings) were obtained by calling `pipeline.embed()` and reading the output shape. The absence of a runtime `sqrt(d_k)` division, the pre-norm ordering, and the RMS-style layer norm were read from `transformers/models/t5/modeling_t5.py` in the installed environment; the relative-position bucket table was produced by calling `T5Attention._relative_position_bucket`. Every figure in the worked attention example was recomputed in floating point and agrees with the hand chain shown. Result numbers come from `results/sweep_real.md` and `01_FACT_SHEET.md`. Verified 2026-07-30.*
