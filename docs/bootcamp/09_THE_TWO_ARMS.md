# 09 · The two arms — Arm A and Arm B, line by line

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you.** Both arms take the same 512 numbers per session and produce a predicted score. Arm A uses classical statistics; Arm B trains a small neural network on top. Liuyi named this as a gap — *"How MLP head works for arm B and what is that, you all don't know."* By the end of this chapter you can draw both arms with shapes, count every parameter, explain why Arm B exists at all, and explain why it came **last**.
>
> `06_DEEP_LEARNING_CORE.md` teaches the mechanics (epochs, gradients, early stopping, dropout). This chapter is about the two specific designs in this repo and how they compare.

---

## 1. The one picture

```
                        ┌──────────────────────────────────────┐
   ONE SESSION          │  SHARED — identical for both arms    │
   glucose readings ───▶│                                      │
   (3 to 288 numbers)   │  Chronos-Bolt-small, FROZEN          │
                        │  → (tokens, 512) → pool → 512 numbers│
                        └──────────────────┬───────────────────┘
                                           │
                            X = (956, 512)  ·  y = (956,)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │  ARM A  —  sklearn    │                     │  ARM B  —  PyTorch    │
        │                       │                     │                       │
        │  StandardScaler       │                     │  LayerNorm(512)       │
        │        ↓              │                     │        ↓              │
        │  (optional PCA)       │                     │  Linear(512 → 256)    │
        │        ↓              │                     │        ↓              │
        │  Ridge  /  SVR        │                     │  GELU                 │
        │  /  LinearRegression  │                     │        ↓              │
        │                       │                     │  Dropout(0.1)         │
        │  solved in ONE shot   │                     │        ↓              │
        │  by linear algebra    │                     │  Linear(256 → 1)      │
        │                       │                     │                       │
        │  nothing to "train"   │                     │  trained by gradient  │
        │  beyond fitting       │                     │  descent, 100 epochs  │
        │                       │                     │  max, early-stopped   │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    ▼                                             ▼
             one predicted score                          one predicted score
             R² = −0.089 / −0.056 / −0.012                R² = −0.218 / −0.281 / −0.168
                     (grids/symbols/prices)
```

Both arms are evaluated with **the same 5 grouped folds** on the same participants, so the numbers are comparable.

---

## 2. Why two arms exist at all

Not a scientific decision — an **assignment**. Liuyi asked you to *"adapt/generalize Ben's code to accept one channel of CGM."* Ben is a labmate whose codebase applies time-series foundation models to actigraphy data for preterm-birth prediction, using a frozen encoder plus a trainable classification head. Arm B is that architecture, converted from classification to regression and from two input channels to one.

So Arm B answers a question Liuyi asked, and Arm A answers the scientific question in the simplest defensible way. Being able to say *why each exists* is part of the answer.

**Say it in your own words:** *"Arm A is the simple, appropriate approach — classical regression on the Chronos features. Arm B exists because you asked me to adapt Ben's code to single-channel CGM. It's the same frozen encoder plus a small trainable network on top, converted from his classifier to a regressor. Arm A is the scientific answer; Arm B is the requested comparison."*

---

## 3. Arm A — `cgm_tsfm/regression.py`

### The models

| Model | Settings searched | What it is |
|---|---|---|
| `DummyRegressor(strategy="mean")` | — | **The comparison baseline.** Ignores the input; always predicts the training set's average score. |
| `Ridge` | alpha ∈ {0.1, 1, 10, 100, 1000} | Linear regression with a penalty on large coefficients. |
| `SVR(kernel="rbf")` | C ∈ {0.1, 1, 10} × gamma ∈ {scale, auto} | Non-linear, distance-based. |
| `LinearRegression` | none | Unregularized least squares — included **to demonstrate failure**. |
| ~~`XGBoost`~~ | — | **Never ran.** See below. |

> ⚠️ **XGBoost is in the code but did not run.** `regression.py:172-179` wraps the import in `try/except`, and xgboost is **not installed** in this environment. So it was silently skipped in every result. If asked "did you try gradient boosting?" the true answer is **no**. Don't get caught by a code-reader on this.

### The structure of one fold

```python
steps = [("scaler", StandardScaler())]
if pca_components:
    steps.append(("pca", PCA(n_components=pca_components)))
steps.append(("model", estimator))
pipe = Pipeline(steps)
```

Everything lives inside a single sklearn `Pipeline`. That is the whole leakage defence: when `.fit()` is called on the training fold, the scaler and PCA learn their statistics **from the training fold only**. When `.predict()` runs on the test fold, they merely apply what they already learned. It is structurally impossible to leak, as opposed to merely remembering not to.

Plus an explicit guard on every fold (`regression.py:113`):

```python
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
    "subject leakage between train and test!"
```

### The fits, counted

Per target: 5 (baseline) + 80 (Ridge) + 95 (SVR) + 5 (Linear) = **185 model fits**; × 3 targets = **555 per run**.

*Ridge: 5 alpha values × 3 inner folds = 15 fits to choose alpha, +1 refit on the full training fold = 16 per outer fold, × 5 outer folds = 80. SVR: 6 settings × 3 = 18, +1 = 19, × 5 = 95.*

### The teaching result

Plain `LinearRegression` on 512 features gives **R² ≈ −3**, while `Ridge` on the identical features gives **−0.089**. Same data, same protocol; the only difference is the penalty on coefficient size.

Why: fitting 512 coefficients plus an intercept — **513 unknowns** — from about **764 training rows** is a ratio of 1.49 rows per unknown. There is barely enough data to pin the coefficients down, and the Chronos dimensions are correlated with each other, so the solution is unstable: tiny changes in the training data swing the coefficients wildly, and they don't transfer to new participants.

**This is the single best concrete demonstration of overfitting in the whole project.** Keep it ready.

**Say it in your own words:** *"Arm A scales the 512 features using only the training fold's statistics, then fits Ridge or SVR, picking the regularization strength with an inner loop on the training data. The whole thing is one sklearn Pipeline so test data can never influence the scaling. And plain least squares on the same features collapses to about −3, which shows why the penalty is not optional."*

---

## 4. Arm B — the MLP head

### What "MLP head" means

**MLP** = multi-layer perceptron, the simplest kind of neural network: alternating linear layers and non-linear activations. **Head** = the small trainable piece bolted onto a frozen pretrained model. The encoder does the hard work of describing the glucose curve; the head just learns the mapping from that description to a score.

### The architecture, exactly (`ben_adapter/model.py:146-152`)

```python
self.head = nn.Sequential(
    nn.LayerNorm(d),                     # d = 512
    nn.Linear(d, head_hidden_dim),       # 512 → 256
    nn.GELU(),
    nn.Dropout(head_dropout),            # 0.1
    nn.Linear(head_hidden_dim, num_targets),   # 256 → 1
)
```

### Forward pass with shapes

```
input:  the 512 Chronos numbers for a batch of 64 sessions
        (64, 512)
          │
          ├─ LayerNorm(512)     → (64, 512)   normalize each row across its 512 values
          │                                    (per-sample; no dataset statistics)
          ├─ Linear(512 → 256)  → (64, 256)   256 weighted sums of the 512 inputs
          ├─ GELU()             → (64, 256)   smooth non-linear bend, shape unchanged
          ├─ Dropout(0.1)       → (64, 256)   training only: zero ~10% of values at random
          ├─ Linear(256 → 1)    → (64, 1)     collapse to one number per session
          └─ .squeeze(-1)       → (64,)       one predicted score per session
```

### Every parameter, counted

| Layer | Shape | Parameters |
|---|---|--:|
| `LayerNorm` weight (γ) | (512,) | 512 |
| `LayerNorm` bias (β) | (512,) | 512 |
| `Linear1` weight | (256, 512) | 131,072 |
| `Linear1` bias | (256,) | 256 |
| `Linear2` weight | (1, 256) | 256 |
| `Linear2` bias | (1,) | 1 |
| | **Total trainable** | **132,609** |

General rule: `Linear(in → out)` has `in × out + out` parameters.

By encoder width, for reference: tiny (256) → 66,561; mini (384) → 99,585; **small (512) → 132,609**; base (768) → 198,657.

The frozen Chronos encoder contributes **47,718,016** parameters, none of them trainable. So **0.28%** of the total model trains.

### Why each piece is there

- **`LayerNorm` first.** The 512 Chronos numbers are not on a controlled scale. Normalizing each row before the first linear layer keeps the numbers entering the network well-behaved, which makes gradient descent better conditioned.
- **`Linear(512→256)`.** The learning happens here — 131,072 weights forming 256 different weighted combinations of the input.
- **`GELU`.** The non-linearity, and it is **mandatory**. Without an activation between two linear layers, they collapse: `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`, which is just one linear layer. Two stacked linear layers with no activation are strictly no more expressive than one. GELU is a smoothed ReLU that lets slightly negative values through instead of hard-clipping them to zero.
- **`Dropout(0.1)`.** During training, randomly zeroes about 10% of the 256 values each time, so the network can't rely on any single one. **Active in training, completely off at evaluation** — Lightning switches this automatically via `model.train()` / `model.eval()`. This is the classic exam point.
- **`Linear(256→1)`, no activation.** The output must be an unbounded real number because we're predicting a continuous score. Any activation here would restrict the range. **This is the substantive difference from Ben's classifier**, which ended in softmax over class probabilities with a cross-entropy loss. Ours ends in identity with a squared-error loss. Our own docs describe the adaptation as just changing the output *width* from `num_classes` to `num_targets=1` — that undersells it; the loss function changed too.

**Say it in your own words:** *"The head is a small network with one hidden layer. It normalizes the 512 numbers, maps them to 256 through a linear layer, bends them with GELU, randomly drops 10% during training so it can't over-rely on any one value, then maps the 256 down to a single predicted score. 132,609 numbers get adjusted during training. Chronos itself, 47.7 million parameters, never changes."*

---

## 5. How Arm B is trained

| Setting | Value | Where |
|---|---|---|
| Optimizer | AdamW | `lightning_module.py:90` |
| Learning rate | 1e-3 | `train.py:73` |
| Weight decay | 0.01 | `lightning_module.py:35` |
| Loss | MSE on the **z-scored** target | `lightning_module.py:44,68` |
| Batch size | 64 | `train.py:71` |
| Max epochs | 100 | `train.py:70` |
| Early stopping | `val/loss`, patience 8, mode min | `train.py:53` |
| Validation split | 20% of outer-train **participants** | `train.py:92` |
| Networks per run | 5 folds × 3 targets = **15** | |

### The epoch arithmetic — be able to do this live

```
956 sessions
  ├─ grouped 5-fold → test ≈ 191,  outer-train ≈ 765
  └─ GroupShuffleSplit(test_size=0.2) on the 16 outer-train participants
       → validation ≈ 153 sessions (≈ 3 participants)
       → training   ≈ 612 sessions (≈ 13 participants)

batch_size 64  →  ceil(612 / 64) = 10 batches
  ⇒ 1 epoch = 10 optimizer steps
  ⇒ 100 epochs = at most 1,000 steps
  ⇒ validation runs once per epoch, so "patience 8" = 8 epochs
```

### Target normalization, and why

The three scores live on wildly different scales — Grids around 0.5, Prices around 42. Rather than special-casing each, the module z-scores the target using **training-fold statistics only**, computes the loss on that scale, and converts predictions back to real units before computing metrics:

```python
labels_norm = (labels - self.target_mean) / self.target_std   # train-set stats
loss = self.loss_fn(logits, labels_norm)
preds = logits * self.target_std + self.target_mean            # back to real units
```

This replaced a hard-coded `/365.0` in Ben's original, which was specific to gestational age in days.

> **A naming wart worth knowing.** The variable is called `logits` even though this is regression. "Logits" properly means the pre-softmax scores of a classifier. It's a leftover from Ben's classifier. If asked, say: *"That's an inherited variable name from the classifier this was adapted from. It's a plain predicted value, not a logit."* Better to know it than to be surprised by it.

---

## 6. Four honest problems with Arm B

Raise these yourself. They're all real, and a careful code-reader will find them.

**1. Early stopping does not keep the best weights.** `train.py:54-55` sets `enable_checkpointing=False` and there is no `ModelCheckpoint` callback, so Lightning restores nothing when training halts. The reported test metrics come from the weights **at the moment training stopped** — which is up to 8 epochs *past* the best validation loss. So Arm B is being judged on slightly-degraded weights, and its numbers are a little worse than the design intends. The fix is one callback plus loading the best checkpoint before testing.

**2. The validation set is about 3 participants.** 20% of 16 outer-train participants. The entire early-stopping decision — when to stop training — rests on ~3 children. That is a very noisy signal, and it's part of why Arm B's fold-to-fold spread is large (±0.102 to ±0.141).

**3. "Arm A tunes, Arm B doesn't" is wrong.** Two ways. Arm B chooses its stopping epoch using validation data, which *is* tuning. And `weight_decay=0.01` in AdamW means Arm B is **also** L2-regularized — the same family of penalty as Ridge's alpha. The difference is that Ridge searches five alpha values explicitly while Arm B uses one fixed decay value nobody tuned.

**4. The two arms do not see identically preprocessed features.** Arm A applies `StandardScaler` — **per feature, across the training rows**. Arm B applies `LayerNorm` — **per sample, across the 512 features** — and additionally z-scores the target. Different transforms on different axes. So the claim in our docs that "only the predictor box differs" is an overstatement. Say instead: *"they share the encoder and the fold structure; the preprocessing differs slightly and I can tell you how."*

---

## 7. Why Arm B lost — the real answer

| | grids | symbols | prices |
|---|--:|--:|--:|
| Arm A (Chronos + Ridge/SVR) | −0.089 | −0.056 | −0.012 |
| **Arm B (Chronos + MLP head)** | **−0.218** | **−0.281** | **−0.168** |
| Hand-built 43 features | −0.065 | −0.038 | −0.009 |

Arm B is **two to five times further below the baseline** than Arm A. Four reasons, in order of importance:

**1. Capacity versus data.** 132,609 trainable parameters against ~612 training rows = **217 parameters per training example**. Ridge fits 513. Arm B has 258× more freedom than Ridge on the same problem, and the data cannot constrain it.

**2. There is almost nothing to learn.** The strongest raw relationship between any simple glucose statistic and any score in this dataset is r = −0.092, i.e. **0.85% of the variance**. When there is no signal, extra capacity can only fit noise. Flexibility is a liability here, not an asset.

**3. Weaker regularization, chosen worse.** Ridge searches five alpha values per fold with a proper inner loop and picks the best. Arm B has a single untuned weight decay of 0.01, dropout of 0.1, and an early-stopping decision made on ~3 children — and then doesn't even keep the best weights.

**4. No closed form.** Ridge has an exact solution. Arm B has to find its way there by gradient descent in at most 1,000 steps. Note the theory point, because it's a good one: **an MLP with GELU can represent Ridge's linear solution exactly** — it is strictly more expressive. So this isn't a limit on what Arm B *could* do; it's a failure to *find* the good solution and stay there.

**The honest framing:** this is the expected result, not a surprise or a bug. A more flexible model on 20 participants with essentially no signal should do worse, and it did. That's evidence the evaluation is working.

**Say it in your own words:** *"Arm B did worst, and that's what you'd predict. It has 132,609 adjustable numbers trained on about 612 sessions — 217 parameters per example — while Ridge has 513. With this little data and essentially no relationship in it to find, extra flexibility just means more room to fit noise. It's also regularized less carefully: Ridge searches five penalty strengths per fold, Arm B uses one fixed value and decides when to stop based on about three children."*

---

## 8. One quiet efficiency trick: `PassthroughEmbedder`

Arm B's design nominally wraps the frozen encoder inside the network. But since the encoder never changes, re-running Chronos every epoch would recompute identical numbers 100 times.

So `model.py:108-126` defines a no-op "encoder" that just returns the batch it was given. The real Chronos pass runs **once**, upstream, and is cached; Arm B trains on those cached 512-number vectors. Mathematically identical, dramatically faster.

Practical consequence to know: **Arm B never touches Chronos during training.** It trains a small network on a fixed table of numbers. That is worth being precise about, because "we fine-tuned a foundation model" would be false — nothing in Chronos was fine-tuned.

---

## 9. Drill

<details>
<summary><b>How many trainable parameters does Arm B have, and how do you get there?</b></summary>

132,609. LayerNorm contributes 512 + 512 = 1,024 (a scale and a shift per feature). `Linear(512→256)` contributes 512×256 = 131,072 weights plus 256 biases. `Linear(256→1)` contributes 256 weights plus 1 bias. Total 1,024 + 131,328 + 257 = 132,609. The frozen encoder adds 47,718,016 non-trainable parameters, so 0.28% of the model trains.
</details>

<details>
<summary><b>Why does the head need GELU? What breaks without it?</b></summary>

Without an activation, two consecutive linear layers collapse into one: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂). So the 512→256→1 stack would be mathematically identical to a single 512→1 linear layer, and the hidden layer would be pointless. GELU makes the composition genuinely non-linear.
</details>

<details>
<summary><b>Why is there no activation on the final layer?</b></summary>

Because we're predicting a continuous score that can take any value. An activation would restrict the output range — sigmoid would cap it to (0,1), ReLU would forbid negatives. This is also the substantive difference from Ben's classifier, which ended in softmax over classes with cross-entropy loss; ours ends in identity with squared-error loss.
</details>

<details>
<summary><b>What is one epoch in Arm B, in steps?</b></summary>

Ten. About 612 training sessions at batch size 64 gives ceil(612/64) = 10 batches, and one epoch is one pass through all of them, so 10 optimizer updates. With max_epochs 100 that is at most 1,000 updates, though early stopping usually ends it sooner.
</details>

<details>
<summary><b>Patience is 8 — eight what? And what happens at the stop?</b></summary>

Eight epochs, because validation runs once per epoch. If validation loss fails to improve for 8 consecutive epochs, training halts. And here's the catch worth volunteering: checkpointing is disabled, so the weights that get tested are the ones from the moment training stopped — up to 8 epochs past the best validation loss. Nothing is restored. That makes Arm B look slightly worse than intended, and it's a one-callback fix.
</details>

<details>
<summary><b>Is dropout active when you evaluate?</b></summary>

No. Dropout only applies during training. At evaluation the layer becomes an identity pass-through, so predictions are deterministic. PyTorch switches this via model.train() / model.eval(), and Lightning calls those automatically around the training and validation loops.
</details>

<details>
<summary><b>Arm A uses StandardScaler, Arm B uses LayerNorm. What's the difference?</b></summary>

Different axes and different statistics. StandardScaler works **per feature across the training rows** — for each of the 512 columns it computes a mean and standard deviation from the training fold and applies them. LayerNorm works **per sample across the features** — for each single session it normalizes its own 512 values against their own mean and standard deviation. StandardScaler uses dataset statistics and could leak if fit on the wrong rows; LayerNorm uses no dataset statistics at all and therefore cannot leak. It also means the arms don't see identically preprocessed inputs.
</details>

<details>
<summary><b>Did you fine-tune Chronos?</b></summary>

No. Chronos is frozen throughout — enforced by torch.no_grad() in Arm A and by setting requires_grad=False on every encoder parameter in Arm B, plus the optimizer only receiving parameters where requires_grad is True. In the actual Arm B runs the embeddings are precomputed once and cached, and a pass-through stand-in replaces the encoder, so Chronos isn't even executed during training. With 20 participants, fine-tuning 47.7 million parameters would be indefensible.
</details>

<details>
<summary><b>If the MLP is more expressive than Ridge, how can it do worse?</b></summary>

Expressiveness is about what a model *can* represent; generalization is about what it *does* find from limited data. The MLP can represent Ridge's solution exactly — it's a strict superset. But it has to reach it by gradient descent in at most 1,000 steps, with regularization that wasn't tuned, and a stopping point chosen from ~3 children. Ridge gets its answer in closed form with a penalty strength selected by a proper inner search. More capacity with no signal to find means more room to fit noise.
</details>

---

## 10. Ten sentences that cover this chapter

1. Both arms consume the identical `(956, 512)` Chronos features and the identical 5 grouped folds, so their numbers are comparable.
2. Arm A scales per feature on the training fold, optionally reduces dimensions, then fits Ridge, SVR or plain least squares — all inside one sklearn Pipeline so test data cannot leak in.
3. Arm A does **555 model fits** per run and includes a mean-predictor baseline to measure everything against.
4. Plain least squares on 512 features gives R² ≈ −3 versus Ridge's −0.089 — the clearest overfitting demonstration in the project, caused by fitting 513 unknowns from ~764 rows.
5. XGBoost appears in the code but is not installed, so it never ran.
6. Arm B exists because Liuyi asked for Ben's frozen-encoder-plus-head design adapted to one CGM channel and to regression.
7. The head is `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`, **132,609 trainable parameters** — 0.28% of the full model, since Chronos's 47.7M stay frozen.
8. One epoch is 10 optimizer steps (≈612 training rows at batch 64); max 100 epochs; early stopping on validation loss with patience 8 epochs.
9. Arm B came **last** (−0.218 / −0.281 / −0.168), and that is the expected outcome: 217 parameters per training row, a raw relationship worth 0.85% of the variance, untuned regularization, and a stopping decision resting on ~3 participants.
10. Four things to disclose without being asked: early stopping doesn't restore the best weights, the validation set is ~3 participants, Arm B *is* regularized (weight decay 0.01) so the "A regularized / B not" framing is wrong, and the two arms use slightly different preprocessing.

→ Next: `10_RIGOR_AND_STATS.md` for why the low accuracy is credible, `12_EXAM_DRILL.md` to test yourself, `15_FINDINGS_TO_REPORT.md` for this week's corrections.
