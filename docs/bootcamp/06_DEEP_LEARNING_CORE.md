# 06 · Deep Learning Core — the neural head, mechanism by mechanism

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **What this chapter buys you (5 lines).**
> 1. You can answer *"what is early stopping"* with the monitored quantity, the patience counter, a real 18-epoch table, and the one thing our code gets wrong about it.
> 2. You can answer *"how does the epoch work"* with arithmetic: our epoch is **9 optimizer steps**, and you can derive the 9 from 956 sessions in front of someone.
> 3. You can define every standard deep-learning term — neuron, activation, forward pass, backward pass, gradient, loss, batch, step, optimizer, dropout, LayerNorm, weight decay — in plain words, on demand.
> 4. You can hand-work a neuron, a two-layer chain-rule gradient, and three steps of gradient descent on paper, without a computer.
> 5. You can explain why Arm B is our **least accurate** arm using its own numbers — 132,609 parameters, ~550 training sessions, a validation signal from 4 children — instead of sounding apologetic.

Companion chapters: `01_FACT_SHEET.md` (every number with its derivation), `02_ML_FROM_ZERO.md` (what a model, a feature, and overfitting are in general), `03_PREPROCESSING.md` (how glucose becomes 512 numbers), `07_TRANSFORMERS.md` (what the frozen Chronos encoder is doing internally), `09_THE_TWO_ARMS.md` (Arm A vs Arm B side by side), `13_PROBLEM_SETS.md` (exercises on this material).

**Terminology guard-rail.** This chapter says **sessions** for rows of data (one cognitive test taken at one time) and **participants** for children. It never says "patients" for sessions. It never says "null result" — it says **lower accuracy** or **no better than guessing the average**.

---

## What we are actually talking about

The whole of "the deep learning part" of this project is these five lines (`cgm_tsfm/ben_adapter/model.py:146-152`):

```python
self.head = nn.Sequential(
    nn.LayerNorm(d),                        # d = 512
    nn.Linear(d, head_hidden_dim),          # 512 -> 256
    nn.GELU(),
    nn.Dropout(head_dropout),               # p = 0.1
    nn.Linear(head_hidden_dim, num_targets) # 256 -> 1
)
```

That is it. **132,609 trainable numbers.** Everything else in this chapter is either (a) what those five lines compute, or (b) how the 132,609 numbers get chosen.

Read the chapter in this order. Each block needs the ones above it.

```
 PART I — what the network computes
 1 neuron ─► 2 why activations are required ─► 3 the activation zoo
        └─► 4 MLP forward pass with shapes ─► 5 parameter count = 132,609 ─► 6 loss

 PART II — how the numbers get chosen
 7 gradients & backprop ─► 8 gradient descent by hand ─► 9 learning rate
        └─► 10 batches ─► 11 EPOCH vs STEP ─► 12 the full loop ─► 13 optimizers

 PART III — keeping it honest
 14 overfitting curves ─► 15 EARLY STOPPING ─► 16 dropout ─► 17 LayerNorm
        └─► 18 weight decay ─► 19 initialization

 PART IV — the verdict
 20 the four soft spots ─► 21 why Arm B lost ─► 22 freeze vs fine-tune
```

---
---

# PART I — What the network computes

## 1 · A neuron is a weighted sum plus a number

**Plain words.** A neuron (also called a *unit*) takes several numbers in and produces one number out. It multiplies each input by its own **weight**, adds them all up, and adds one extra number called the **bias**. That is the entire operation.

```
        x1 ──(w1)──┐
                   │
        x2 ──(w2)──┼──►  z = w1*x1 + w2*x2 + w3*x3 + b  ──►  a = activation(z)  ──► out
                   │            \_______________/    \_/
        x3 ──(w3)──┘             weighted sum        + bias
                   │
        1  ──(b )──┘   (the bias is a weight on a constant input of 1)
```

The weighted sum is written compactly as a **dot product**: `z = w·x + b`.

**Worked by hand, three inputs.** Let the weights be `w = (0.5, -0.2, 0.1)`, the bias `b = 0.4`, and one session's three inputs `x = (2.0, 3.0, -1.0)`.

```
w·x = (0.5)(2.0) + (-0.2)(3.0) + (0.1)(-1.0)
    =    1.0     +   (-0.6)    +   (-0.1)
    =    0.3
z   = w·x + b = 0.3 + 0.4 = 0.7
```

Then the activation is applied. With GELU (section 3): `GELU(0.7) = 0.5306`. With ReLU it would be `0.7`.

**Where this lives in our code.** `nn.Linear(512, 256)` is 256 neurons side by side, each with its own 512 weights and its own bias. Our first Linear layer computes 256 of these weighted sums from the 512 Chronos numbers. Our second, `nn.Linear(256, 1)`, is a *single* neuron over the 256 hidden values, and its output is the predicted cognitive score.

**A neuron and a linear-regression model are the same object.** Ridge (chapter `02_ML_FROM_ZERO.md`) is `prediction = b + Σ wᵢxᵢ` — one neuron with no activation. That equivalence is worth saying out loud, because it means the difference between Arm A and Arm B is not "linear algebra vs magic", it is "one neuron vs 257 neurons in two layers".

**Say it in your own words:** *"A neuron multiplies each input by a weight, adds them up, adds a bias, and passes the result through a bend — so `nn.Linear(512, 256)` is just 256 of those weighted sums computed at once."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: How many numbers does one neuron in our first Linear layer own?**
A: 513 — one weight for each of the 512 Chronos features, plus one bias.

**Q: What is the bias for? Why not drop it?**
A: Without a bias the neuron is forced through zero: if every input is 0 the output must be 0. The bias lets the neuron shift its output up or down freely, which is what lets the final layer output a score around 41.6 (the Prices average) rather than something near zero.

**Q: `w = (1, -1)`, `b = 2`, `x = (3, 5)`. What is z?**
A: `z = (1)(3) + (-1)(5) + 2 = 3 - 5 + 2 = 0`.

</details>

---

## 2 · Why an activation function is mandatory (two lines of algebra)

**Plain words.** If you stack weighted sums with nothing in between, the whole stack collapses into a *single* weighted sum. You gain parameters and gain nothing else. The activation function is the bend that prevents the collapse.

**The proof.** Let layer 1 be `h = W₁x + b₁` and layer 2 be `y = W₂h + b₂`. Substitute:

```
y = W₂(W₁x + b₁) + b₂
  = (W₂W₁)x + (W₂b₁ + b₂)
```

`W₂W₁` is just some matrix — call it `W*`. `W₂b₁ + b₂` is just some vector — call it `b*`. So `y = W*x + b*`: one linear layer. Stacking 2 layers, or 50, changes nothing about the set of functions you can express. **A network with no activation function is linear regression with extra steps.**

In our head, `LayerNorm → Linear(512→256) → GELU → Dropout → Linear(256→1)`, remove the `GELU` and the two Linear layers fuse into a single 512→1 linear map. Our 132,609 parameters would then describe a function with only 513 degrees of freedom — and Ridge finds the best such function in closed form, instantly, with a regularization guarantee. The GELU is the only reason Arm B is a different model class from Arm A at all.

**The flip side, and it matters for us.** Non-linearity buys the ability to represent curves and interactions — "high glucose hurts only when it is also *falling fast*", for example. That is worth having *if* such a pattern exists in the data. Our strongest raw glucose→score correlation is r = −0.092 (r² = 0.0085; see `01_FACT_SHEET.md` section E). There is almost no straight-line signal, so there was little reason to expect a curved one, and the extra freedom mostly bought extra ways to fit noise.

**Say it in your own words:** *"Two stacked linear layers multiply out to one linear layer — `W₂(W₁x+b₁)+b₂ = (W₂W₁)x + (W₂b₁+b₂)` — so without the GELU our 132,609 parameters would describe the same straight-line function Ridge already solves exactly."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Prove in one line that a 3-layer network with no activations is linear.**
A: `W₃(W₂(W₁x + b₁) + b₂) + b₃ = (W₃W₂W₁)x + (W₃W₂b₁ + W₃b₂ + b₃)`, which has the form `W*x + b*`.

**Q: So is a network without activations useless?**
A: Not useless, but pointless — it can only express what one linear layer expresses, while costing more memory, more compute, and a harder optimization problem than solving that linear problem directly.

**Q: Is our head with GELU exactly a linear model plus a bend?**
A: No, there are two non-linear pieces. GELU is one. LayerNorm is the other, because it divides each row by that row's own standard deviation — so our head is not even a bent version of Ridge's function of x. See section 17.

</details>

---

## 3 · The activation zoo, and why the output layer has none

An activation function takes one number and returns one number. It is applied element-wise, so a `(64, 256)` tensor stays `(64, 256)`.

| Name | Formula | Plain description | Output range |
|---|---|---|---|
| **ReLU** | `max(0, x)` | keep positives, flatten negatives to exactly 0 | `[0, ∞)` |
| **GELU** (ours) | `x·Φ(x) = 0.5x[1 + erf(x/√2)]` | a **smooth ReLU** that lets slightly-negative values leak through instead of hard-zeroing them | `≈[-0.17, ∞)` |
| **sigmoid** | `1/(1 + e^-x)` | squashes anything into 0…1; reads as a probability | `(0, 1)` |
| **tanh** | `(e^x − e^-x)/(e^x + e^-x)` | squashes into −1…1, centred on 0 | `(-1, 1)` |

`Φ` is the standard normal cumulative distribution function — the probability that a standard normal draw is below `x`. So GELU multiplies the input by "how far above average this input is", which is why small negatives survive a little and large negatives vanish.

Verified values (PyTorch `nn.GELU()`, default `approximate='none'`, i.e. the exact `erf` form):

```
   x    ReLU(x)   GELU(x)   sigmoid(x)  tanh(x)
 -3.0    0.0000   -0.0040     0.0474   -0.9951
 -2.0    0.0000   -0.0455     0.1192   -0.9640
 -1.0    0.0000   -0.1587     0.2689   -0.7616
 -0.5    0.0000   -0.1543     0.3775   -0.4621
  0.0    0.0000    0.0000     0.5000    0.0000
  0.5    0.5000    0.3457     0.6225    0.4621
  1.0    1.0000    0.8413     0.7311    0.7616
  2.0    2.0000    1.9545     0.8808    0.9640
  3.0    3.0000    2.9960     0.9526    0.9951
```

Read the GELU column: it dips to about −0.17 near x ≈ −0.75, then climbs back toward 0, and by x = 3 it is within 0.004 of ReLU. So "smooth ReLU with a small negative dip" is an accurate one-liner.

```
   ReLU                    GELU                   sigmoid                tanh
     |    /                  |    /                 |   ______            |   ______
     |   /                   |   /                  |  /                  |  /
 ____|__/____            ____|__/____           ____|_/______        ____|_/______
     |                     \_|                  __/ |                 __/|
     |                       |                      |                     |
   hard corner at 0      smooth, small dip       flattens both ends   flattens, centred
```

**Why the output layer has no activation.** Our target is a real number — Grids ranges 0…2.39, Symbols 0.15…4.19, Prices 0…90. Any squashing activation on the output would cap what the model can say: a sigmoid output could never exceed 1, so it could not predict a Prices score of 60. So the final `nn.Linear(256, 1)` output is used **as is**. This is called an **identity** or **linear** output. In our code there is simply nothing after the last Linear (`model.py:151`).

**Why Ben's classifier needed softmax and ours does not.** Ben's original model (`FoundationModelClassifier`) predicted a *category* — preterm birth, yes or no. For that you want the output to be a probability distribution over classes, so you put **softmax** on the output (turn the raw scores into non-negative numbers summing to 1) and use **cross-entropy** loss (penalise the model by how much probability it assigned to the wrong class). We predict a *quantity*, so we use identity output plus **MSE** loss (penalise the squared distance from the true number). The one-line change from his head to ours was `num_classes → num_targets=1` plus swapping the loss; everything else about the head is identical (`ben_adapter/README.md`).

| | Ben's classifier | Our regressor |
|---|---|---|
| Output width | number of classes (2) | `num_targets = 1` |
| Output activation | softmax | none (identity) |
| Loss | cross-entropy | `nn.MSELoss()` (`lightning_module.py:44`) |
| Reported metric | accuracy / AUC | R², RMSE, MAE |

**Say it in your own words:** *"GELU is a smooth ReLU that lets slightly-negative values leak through, and our output layer has no activation at all because the answer is a number on an open scale — a sigmoid could never say 60 on the Prices test."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: Why GELU rather than ReLU here?**
A: Honest answer: because Ben's head used GELU and we kept his architecture deliberately, so the comparison is a faithful adaptation rather than a redesign. On the merits GELU is smooth everywhere, so its gradient does not have a corner at 0 and it does not fully kill negative pre-activations. With 132,609 parameters on ~550 training sessions the choice is not what decided our result.

**Q: What is `GELU(0)`? What is `ReLU(0)`?**
A: Both 0. GELU(0) = 0·Φ(0) = 0·0.5 = 0.

**Q: If someone says "put a ReLU on the output so predictions can't go negative", what do you say?**
A: It would be defensible for Prices and Grids, whose true minimum is 0, but it changes the model class and it would clamp the gradient to zero for every session the model currently under-predicts, so those sessions would stop teaching it anything. We did not do it, and our predictions are not the problem — the near-absent signal is.

**Q: Does an activation function have parameters?**
A: `nn.GELU`, `nn.ReLU`, `sigmoid` and `tanh` have none — zero parameters. That is why GELU contributes 0 to our 132,609 count.

</details>

---

## 4 · An MLP, and the forward pass through our head with shapes

**Plain words.** An **MLP** (multi-layer perceptron) is the plainest kind of neural network: alternating Linear layers and activations, every input connected to every neuron. Also called a *fully-connected* or *dense* network. A **forward pass** is one run of data through the network from input to prediction.

Our head is a 2-layer MLP. `B` below is the **batch size** — how many sessions go through together. In training `B = 64` (except the last batch of an epoch, which is smaller).

```
  one batch of 64 sessions, each already turned into 512 Chronos numbers

  glucose window  ─────────────────────────────────┐
  (raw, per session: 3 to 288 readings)            │  computed ONCE upstream and
                                                   │  cached; FROZEN, no gradients
                     ┌─────────────────────────────▼──────────────┐
                     │  Chronos-bolt-small encoder  (47,718,016)  │
                     │  frozen: requires_grad = False everywhere  │
                     └─────────────────────────────┬──────────────┘
                                                   │
   embeddings  (B, 512)  ◄─────────────────────────┘
        │
        ▼
   LayerNorm(512)          (B, 512)   normalise each ROW across its own 512 values
        │                             1,024 parameters
        ▼
   Linear(512 -> 256)      (B, 256)   256 weighted sums per session
        │                             131,328 parameters
        ▼
   GELU()                  (B, 256)   element-wise bend, 0 parameters
        │
        ▼
   Dropout(p=0.1)          (B, 256)   training only: zero ~10%, scale rest by 1/0.9
        │                             0 parameters
        ▼
   Linear(256 -> 1)        (B, 1)     one weighted sum per session
        │                             257 parameters
        ▼
   .squeeze(-1)            (B,)       drop the width-1 axis -> one number per session
        │                             lightning_module.py:66
        ▼
   predicted z-scored cognitive score, one per session
```

Shapes as a single line: `(B,512) → (B,512) → (B,256) → (B,256) → (B,256) → (B,1) → (B,)`.

**The squeeze matters.** `lightning_module.py:66` is `logits = self(glucose, glucose_mask).squeeze(-1)`. Without it the prediction is `(64, 1)` while the labels are `(64,)`, and PyTorch's broadcasting would silently compute a `(64, 64)` matrix of all pairwise errors instead of 64 errors. That class of bug does not crash; it just makes the loss meaningless. Good thing to be able to point at.

**What "frozen encoder" means in the code.** Two independent mechanisms:

1. `model.py:141-142` — `for p in self.encoder.parameters(): p.requires_grad = False`. This tells PyTorch not to compute or store gradients for those tensors.
2. `lightning_module.py:89` — `params = [p for p in self.model.parameters() if p.requires_grad]`, and only those go to the optimizer. Even if something re-enabled a gradient, the optimizer would not be holding that tensor.

Plus `ChronosTorchEmbedder.forward` is decorated `@torch.no_grad()` (`model.py:69`).

**And in the runs we actually report, Chronos is not even in the loop.** `train.py:47` builds the model with a `PassthroughEmbedder`, which is a no-op "encoder" that returns the batch unchanged (`model.py:125-126`). The Chronos embeddings were computed once by `encoders.extract_embeddings` and cached to a `.npy` file; training reads those. Since the encoder is frozen, re-running it every epoch would produce identical numbers, so this is a pure speed optimisation with an identical result. Verified: the trainable-parameter count of the assembled model is exactly **132,609**, confirming the encoder contributes nothing trainable.

**Say it in your own words:** *"One batch of 64 sessions enters as a 64-by-512 block, LayerNorm rescales each row, the first Linear turns 512 numbers into 256, GELU bends them, Dropout blanks a tenth during training, the second Linear collapses 256 into 1, and squeeze turns the 64-by-1 result into 64 predictions."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: What are the tensor shapes at every stage for a batch of 64?**
A: `(64,512) → LayerNorm (64,512) → Linear (64,256) → GELU (64,256) → Dropout (64,256) → Linear (64,1) → squeeze (64,)`.

**Q: Why does the last batch of an epoch have a different size?**
A: With 550 training sessions and batch size 64, the DataLoader does not drop the remainder, so you get 8 batches of 64 and one of 38. Shapes become `(38, 512)` etc. The loss is a mean, so a smaller batch simply averages over fewer sessions.

**Q: If the Chronos encoder is 47.7 million parameters, why do we say the network has 132,609?**
A: Because 132,609 are the only **trainable** ones — 0.28% of the total. The 47,718,016 Chronos parameters are frozen and were excluded from the optimizer; in the reported runs their output was precomputed and cached, so training never touched them.

**Q: What is a "forward pass"?**
A: One run of a batch from input to prediction, with no learning — just arithmetic. Learning happens in the backward pass (section 7).

</details>

---

## 5 · Counting the parameters: 132,609

**The general rule.** `nn.Linear(in, out)` holds a weight matrix of shape `(out, in)` and a bias vector of shape `(out,)`, so:

```
parameters of Linear(in -> out)  =  in*out  +  out
```

`nn.LayerNorm(n)` with default `elementwise_affine=True` holds a learned scale and a learned shift, one each per feature: `2n` parameters. Activations and Dropout hold none.

**Our head, layer by layer** (verified by enumerating `named_parameters()`):

| # | Component | Tensor | Shape | Parameters |
|---|---|---|---|--:|
| 0 | `LayerNorm(512)` | weight (scale) | `(512,)` | 512 |
| 0 | `LayerNorm(512)` | bias (shift) | `(512,)` | 512 |
| 1 | `Linear(512→256)` | weight | `(256, 512)` | 131,072 |
| 1 | `Linear(512→256)` | bias | `(256,)` | 256 |
| 2 | `GELU()` | — | — | 0 |
| 3 | `Dropout(0.1)` | — | — | 0 |
| 4 | `Linear(256→1)` | weight | `(1, 256)` | 256 |
| 4 | `Linear(256→1)` | bias | `(1,)` | 1 |
| | | | **Total** | **132,609** |

Check the arithmetic yourself: `512 + 512 = 1,024`; `256 × 512 = 131,072`, `+256 = 131,328`; `256 + 1 = 257`. And `1,024 + 131,328 + 257 = 132,609`.

Notice where the parameters are: **98.8% of them are the first weight matrix** (131,072 / 132,609). If someone asks "where would you cut capacity first?", the answer is the hidden width. Dropping `head_hidden_dim` from 256 to 16 gives `512×16+16 = 8,208` plus `16+1 = 17` plus 1,024 = **9,249** parameters, a 14× reduction, and it is a one-argument change (`head_hidden_dim` in `train.py:47-49`). We have not run that.

**The ratio that explains the result.** The training split in each fold holds **538 to 576 sessions** (measured, section 11). Take 612 as the round figure used in `01_FACT_SHEET.md` or 550 as the measured figure — either way:

```
132,609 parameters / 612 training sessions  =  217 parameters per training example
132,609 parameters / 550 training sessions  =  241 parameters per training example
```

Chapter `01_FACT_SHEET.md` quotes **217×**. Use that number and know it is the optimistic version. Compare Arm A's Ridge: 513 parameters on ~765 training sessions, i.e. 0.67 parameters per example — 320× less freedom.

**Say it in your own words:** *"A Linear layer has in-times-out weights plus out biases, LayerNorm has two numbers per feature, activations have none — so our head is 1,024 plus 131,328 plus 257, which is 132,609 trainable numbers, and 98.8% of them sit in that first 512-by-256 matrix."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: How many parameters in `nn.Linear(768, 256)`?**
A: `768×256 + 256 = 196,608 + 256 = 196,864`. (That is what the head would cost on `chronos-bolt-base`, whose embedding width is 768.)

**Q: Does `nn.Dropout` have parameters?**
A: No. Its `p = 0.1` is a hyperparameter we set, not a number learned from data.

**Q: Our 132,609 versus Ridge's 513 — but the frozen Chronos model has 47.7 million. Isn't the real capacity 47.8 million?**
A: No, because those 47.7 million never change on our data. They are a fixed feature extractor, the same for every fold and every participant, so they cannot overfit *our* 956 sessions. Capacity that matters for overfitting is capacity that is fitted, and that is 132,609.

</details>

---

## 6 · The loss: what we optimize, versus what we report

**Plain words.** A **loss function** is a single number that says how wrong the model currently is. Training means changing weights to make that number smaller. Ours is **mean squared error** (`nn.MSELoss()`, `lightning_module.py:44`, default reduction `'mean'`):

```
MSE = (1/n) * Σ (prediction_i − truth_i)²
```

**Why squared?** Three reasons, all worth saying. First, squaring makes every error positive, so over- and under-prediction cannot cancel out. Second, it is smooth and differentiable everywhere, so gradient descent has a well-defined direction (absolute error has a corner at zero). Third, its derivative is beautifully simple — `d/dŷ (ŷ−y)² = 2(ŷ−y)` — which means the correction applied is proportional to the size of the mistake: a big miss pushes harder than a small one. The cost is sensitivity to outliers, since a mistake of 4 counts 16 times as much as a mistake of 1.

**Worked on five numbers.** Truths `y = (−1.0, 0.5, 0.0, 1.5, −0.5)`, predictions `ŷ = (−0.2, 0.1, 0.3, 0.4, 0.1)`:

```
errors  ŷ−y :   0.8   −0.4    0.3   −1.1    0.6
squared     :  0.64   0.16   0.09   1.21   0.36    sum = 2.46
MSE = 2.46/5 = 0.492      RMSE = √0.492 = 0.7014
```

A model that predicted 0 for all five (the mean of a z-scored target) would score `(1 + 0.25 + 0 + 2.25 + 0.25)/5 = 0.75`. So 0.492 beats the constant here — on this toy.

**Loss versus metric.** These are different jobs and it is a favourite exam trap:

| | Loss | Metric |
|---|---|---|
| Purpose | drives the weight updates | tells humans how good the model is |
| Must be | differentiable, smooth | interpretable |
| Ours | MSE on the **z-scored** target | **R²**, RMSE, MAE on the **original** score units |
| Where | `lightning_module.py:68` | `lightning_module.py:74`, `torchmetrics` |

The de-normalisation is explicit at `lightning_module.py:67-74`: the label is z-scored using **train-split** mean and standard deviation (`labels_norm = (labels - target_mean)/target_std`), the loss is computed there, and then the prediction is converted back (`preds = logits * target_std + target_mean`) before the metrics see it.

**Why z-score the target at all?** Prices ranges 0…90 and Grids 0…2.39. Unscaled, the same learning rate that is sensible for Grids would be far too small for Prices, so you would need per-target tuning. Z-scoring puts every target on an O(1) scale so one learning rate works for all three. It replaced Ben's gestational-age-specific `labels/365.0` hack (`ben_adapter/README.md`).

**Reading `val/loss` as a number you can interpret.** Because the target is z-scored with training statistics, a model that just printed the training mean would score `val/loss ≈ 1.0` — if the validation children resembled the training children. Measured for Symbols fold 4: predicting the training mean gives a validation MSE of **1.3206**, because those 4 validation children have a slightly different mean and a wider spread. The head's *best* validation loss on that fold was **1.2813** — barely better than the constant. Its loss at the epoch where training actually stopped was **1.3535** — worse than the constant. That single comparison is the negative R² story in one line.

**Say it in your own words:** *"We optimize mean squared error on the z-scored score because it is smooth and its gradient is proportional to the mistake, and we report R², RMSE and MAE in the score's real units because those are what a human can interpret — the loss is the steering wheel, the metric is the speedometer."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Why not train directly on R²?**
A: You can, since maximizing R² on a fixed test set is equivalent to minimizing MSE on it — R² is `1 − MSE/variance`, and the variance is a constant. So MSE is the same objective with the bookkeeping removed. R² would also need the target variance computed per batch, which changes batch to batch.

**Q: `val/loss` is 1.28 and `test/rmse` is 0.61. Why don't those match?**
A: They are on different scales. `val/loss` is MSE on the z-scored target; `test/rmse` is the square root of MSE after converting predictions back to the score's own units. Roughly: `rmse ≈ √val_loss × target_std`. For Symbols fold 4, `√1.28 × 0.522 = 0.59`, in the right region.

**Q: MAE versus RMSE — when do they disagree?**
A: When a few sessions are badly missed. RMSE squares before averaging, so it is dominated by the worst errors; MAE weights every session equally. If RMSE is much larger than MAE, a small number of sessions are carrying the error.

</details>

---
---

# PART II — How the numbers get chosen

## 7 · Gradients and backpropagation

**Plain words: what a derivative is.** The derivative of the loss with respect to one weight answers one question: *if I nudge this weight up by a tiny amount, does the loss go up or down, and how steeply?* Sign tells you the direction, magnitude tells you the steepness. A **gradient** is just all those derivatives collected together — for us, a list of 132,609 numbers, one per trainable parameter, describing the slope of the loss surface at the current position.

If the derivative with respect to weight `w` is `+24`, then increasing `w` a little increases the loss about 24× as fast, so to reduce the loss you must **decrease** `w`. That is the whole logic of the minus sign in gradient descent.

**Backpropagation, plain words.** Backpropagation is an algorithm for computing all 132,609 derivatives efficiently, in one sweep backwards through the network. The naive alternative — nudge each weight, re-run the forward pass, see what happened — would cost 132,609 forward passes per step. Backprop costs about one forward pass' worth of work for the whole gradient.

**Why it works: the chain rule.** If `L` depends on `y`, and `y` depends on `w`, then

```
dL/dw = (dL/dy) * (dy/dw)
```

A network is a long chain of simple functions. So the derivative with respect to an early weight is a product of local derivatives along the path back to the loss. Each layer only needs to know (a) the derivative arriving from the layer above and (b) its own local derivative. That is a purely local rule, which is why it can be implemented once per layer type and composed automatically.

**Worked two-layer example, by hand.** A scalar network: `h = w₁x`, then `ŷ = w₂h`, loss `L = (ŷ − y)²`. Take `x = 2`, `w₁ = 0.5`, `w₂ = 3`, true `y = 1`.

Forward:
```
h  = w₁x = 0.5 * 2 = 1.0
ŷ  = w₂h = 3 * 1.0 = 3.0
L  = (3.0 − 1)²    = 4.0
```

Backward, one link at a time:
```
dL/dŷ  = 2(ŷ − y)      = 2(3 − 1)      =  4
dŷ/dw₂ = h             = 1.0           → dL/dw₂ = 4 * 1.0  =  4
dŷ/dh  = w₂            = 3             → dL/dh  = 4 * 3    = 12
dh/dw₁ = x             = 2             → dL/dw₁ = 12 * 2   = 24
```

Both gradients are positive, so both weights should go down. One gradient-descent step with `lr = 0.01`:
```
w₂ ← 3.0  − 0.01(4)  = 2.96
w₁ ← 0.5  − 0.01(24) = 0.26
```
Re-run the forward pass: `h = 0.52`, `ŷ = 1.5392`, `L = (0.5392)² = 0.2907`. The loss fell from **4.0 to 0.291** in one step. (Verified against `torch.autograd`: `w1.grad = 24.0`, `w2.grad = 4.0`, new loss `0.290737`.)

Notice the structure: `dL/dw₁ = 24` is bigger than `dL/dw₂ = 4` because the signal to `w₁` was amplified by `w₂ = 3` on the way back and by `x = 2` locally. This is exactly why deep networks can suffer *exploding* or *vanishing* gradients — those local factors multiply.

**Forward pass versus backward pass.**

```
   FORWARD  (compute the prediction, and remember the intermediates)
   x ──[LayerNorm]──► a ──[Linear1]──► z ──[GELU]──► h ──[Dropout]──► d ──[Linear2]──► ŷ ──► L
       (store a)          (store z)        (store h)      (store mask)     (store d)

   BACKWARD (compute how much each weight is to blame)
   dL/dŷ ──[Linear2]──► dL/dd ──[Dropout]──► dL/dh ──[GELU]──► dL/dz ──[Linear1]──► dL/da ──[LayerNorm]──► (stop)
              │                                                      │
              └── dL/dW₂, dL/db₂                                     └── dL/dW₁, dL/db₁
```

The forward pass must **store** its intermediate values, because the backward pass needs them (`dL/dW₂` needs `d`, the input to Linear2). That stored stuff is the "activation memory" that makes training use much more memory than inference.

**What `torch.no_grad()` switches off.** Inside a `torch.no_grad()` block PyTorch does not build the graph of operations and does not store intermediates for backward. Consequences: no gradients can be computed through that region, and memory use drops. `ChronosTorchEmbedder.forward` is wrapped in `@torch.no_grad()` (`model.py:69`) because we never want to adjust Chronos. The 47.7 million encoder parameters therefore cost us no gradient buffers, no optimizer state and no activation memory — one reason 15 network trainings run comfortably on CPU.

**Say it in your own words:** *"A gradient says which way and how steeply the loss changes if I nudge a weight, backpropagation computes all 132,609 of those at once by multiplying local derivatives backwards through the network — that is the chain rule — and `torch.no_grad()` turns the whole bookkeeping off, which is how we keep the frozen Chronos encoder out of the learning entirely."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: In the worked example, why is `dL/dw₁` six times `dL/dw₂`?**
A: Because the derivative flowing back to `w₁` passes through `w₂ = 3` and then gets multiplied by the local `dh/dw₁ = x = 2`, so `4 × 3 × 2 = 24` against `4 × 1 = 4`.

**Q: Why not compute gradients numerically by nudging each weight?**
A: It would need one forward pass per parameter — 132,609 per step, times 9 steps per epoch — and it would be numerically inaccurate. Backprop gets the exact gradient for roughly the cost of one extra forward pass.

**Q: Does `requires_grad = False` do the same thing as `torch.no_grad()`?**
A: They overlap but are different. `requires_grad = False` marks specific tensors as not needing gradients. `torch.no_grad()` is a context that disables graph building for everything inside it. We use both on the encoder — belt and braces — and additionally filter the optimizer's parameter list at `lightning_module.py:89`.

**Q: What is stored during the forward pass, and why does it matter?**
A: The inputs to each parameterised operation plus things like the dropout mask, because the backward formulas need them. It matters because that memory, not the parameter count, is usually what limits batch size when training large models.

</details>

---

## 8 · Gradient descent, three steps by hand

**The rule.** For every parameter:

```
w  ←  w  −  learning_rate * (dL/dw)
```

Subtract, because the gradient points uphill and we want to go downhill.

```
   loss
    |                                     the walk with lr = 0.1
  1 |  *                    L(w) = w²
    |   \                            w=1.00  L=1.0000  grad=2.00
    |    \  *                        w=0.80  L=0.6400  grad=1.60
    |     \  \  *                    w=0.64  L=0.4096  grad=1.28
    |      \  \  \  *                w=0.51  L=0.2621  grad=1.02
    |       \__\__\__\__*___         w=0.41  L=0.1678
    |__________________________  w
   -1        0        1
       each step: move left by lr*grad, i.e. by 0.2*w
```

**Three iterations, `L(w) = w²`, `dL/dw = 2w`, `lr = 0.1`, start at `w = 1.0`:**

```
step 1:  grad = 2(1.00)  = 2.00   w ← 1.00 − 0.1(2.00) = 0.800   L = 0.6400
step 2:  grad = 2(0.80)  = 1.60   w ← 0.80 − 0.1(1.60) = 0.640   L = 0.4096
step 3:  grad = 2(0.64)  = 1.28   w ← 0.64 − 0.1(1.28) = 0.512   L = 0.2621
```

Each step multiplies `w` by `(1 − 2·lr) = 0.8`, so it shrinks geometrically toward 0, the minimum. Notice the steps get *smaller* as you approach — not because the learning rate changed, but because the gradient did. Gradient descent slows down automatically near a minimum.

**Now break it with a too-large learning rate.** Same loss, same start, four steps (all verified numerically):

| `lr` | w after 1, 2, 3, 4 steps | loss after 1, 2, 3, 4 | behaviour |
|---|---|---|---|
| **0.1** | 0.8, 0.64, 0.512, 0.4096 | 0.64, 0.410, 0.262, 0.168 | converges smoothly |
| **0.5** | 0, 0, 0, 0 | 0, 0, 0, 0 | lands exactly on the minimum in one step |
| **1.0** | −1, 1, −1, 1 | 1, 1, 1, 1 | oscillates forever, no progress |
| **1.1** | −1.2, 1.44, −1.728, 2.0736 | 1.44, 2.07, 2.99, 4.30 | **diverges** — the loss grows |

```
   lr too small                 lr just right                lr too large
     |  .                          |  *                        |        *
     | . \                         |   \                       |   *   /
     |.   \                        |    \  *                   |    \ /  *
     |.....\_____                  |     \__\____              |     X  /
     |___________ w                |___________ w              |___________ w
   crawls: many epochs           steady descent           overshoots, bounces out
```

The general fact behind that table: for `L = w²`, the update is `w ← w(1 − 2·lr)`, which shrinks only while `|1 − 2·lr| < 1`, i.e. `0 < lr < 1`. Real loss surfaces have no such clean threshold, but the shape of the failure is identical — cross a curvature-dependent threshold and you bounce out instead of settling in.

**Say it in your own words:** *"Gradient descent subtracts the learning rate times the gradient, so on the loss `w²` starting at 1 with a rate of 0.1 the weight goes 1 to 0.8 to 0.64 to 0.512 — each step multiplies it by 0.8 — but at a rate of 1.1 the same weight goes to −1.2 then 1.44 then −1.73 and the loss grows instead of shrinking."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Why the minus sign?**
A: The gradient points in the direction of steepest *increase*. We want to decrease, so we step the opposite way.

**Q: Same loss `w²`, `lr = 0.25`, start at `w = 2`. Give three steps.**
A: `w ← w(1 − 2(0.25)) = 0.5w`: 2 → 1 → 0.5 → 0.25. Losses 4 → 1 → 0.25 → 0.0625.

**Q: If the loss is increasing across steps, what are your first two hypotheses?**
A: Learning rate too high, or a sign error / wrong loss direction. In our project neither applies — the training loss falls fine; the problem is that the *validation* loss stops falling almost immediately.

</details>

---

## 9 · The learning rate

**Plain words.** The learning rate is how big a step you take in the direction the gradient points. It is the single most consequential hyperparameter in neural network training. Ours is **1e-3 = 0.001** (`train.py:73`, passed to `CGMRegressionModule`, `lightning_module.py:34`).

| Too small | Too large |
|---|---|
| Loss falls, but slowly; you run out of epochs before convergence | Loss jumps around or grows; may produce `NaN` |
| Looks like underfitting | Looks like a broken model |
| Early stopping may fire while still improving, because improvements shrink below the noise | Best validation loss is at epoch 0 and never beaten |
| Fix: raise it 3–10× | Fix: lower it 3–10× |

**Why 1e-3.** It is the conventional default for the Adam family. Two reasons. First, it is literally PyTorch's default: `torch.optim.AdamW(params, lr=0.001, ...)`. Second, Adam normalises each parameter's step by a running estimate of that parameter's gradient magnitude (section 13), so the step size is roughly `lr` in units of "typical gradient", not in the raw units of the loss. That makes a single value transfer across very different problems, which is not true for plain SGD, where sensible rates span several orders of magnitude.

**Did we tune it?** No. We ran one value, 1e-3, for all 15 networks. That is a real limitation, and the honest framing is: it is the standard default, and given the training loss does fall while the validation loss does not, the failure mode we observe is not "learning rate too small". A grid over `{3e-4, 1e-3, 3e-3}` would be a cheap next experiment — 45 network trainings, minutes on this machine — and it is not yet run.

**Say it in your own words:** *"The learning rate is the step size — ours is 0.001, PyTorch's default for Adam-family optimizers, which works across problems because Adam divides each step by that parameter's typical gradient size; too small and the loss crawls, too large and it bounces out and can go to NaN."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Someone says "you should have tuned the learning rate". Answer honestly.**
A: Correct, we did not — one value, 1e-3, the PyTorch default for AdamW. What I can say is that the observed failure is not a learning-rate-too-small failure: training loss falls, validation loss turns up after 1 to 9 epochs. A three-value sweep would cost 45 trainings and a few minutes, and it is the first thing I would run if asked to defend the setting rather than just report it.

**Q: What is a learning-rate schedule, and do we use one?**
A: A rule that lowers the learning rate as training proceeds — for example halving it whenever validation stops improving. We use none: `configure_optimizers` (`lightning_module.py:87-90`) returns a bare AdamW with no scheduler, so 1e-3 is constant for every step.

**Q: Why can one learning rate serve Grids (scores 0–2.39) and Prices (0–90)?**
A: Two reasons. The target is z-scored to an O(1) scale before the loss, and Adam rescales each parameter's step by its own gradient history. Either alone would help; together they make the setting portable.

</details>

---

## 10 · Batches: full-batch, stochastic, mini-batch

**Plain words.** A **batch** is the group of sessions you push through the network before you update the weights once.

| Name | Batch = | Gradient quality | Updates per epoch | Verdict |
|---|---|---|---|---|
| **Full-batch** (plain gradient descent) | all 550 training sessions | exact | 1 | too few updates; slow progress; needs all data in memory |
| **Stochastic** (SGD, strict sense) | 1 session | very noisy | 550 | very noisy; poor hardware use |
| **Mini-batch** (ours, 64) | 64 sessions | good estimate | 9 | the practical middle |

**Why mini-batches win.** Three separate reasons, and they are often conflated:

1. **More updates per pass over the data.** Full-batch on our fold gives 1 weight update per epoch; batches of 64 give 9. The weights move nine times as often per pass through the sessions.
2. **Hardware.** A matrix multiply of `(64, 512) × (512, 256)` uses a CPU or GPU far more efficiently than 64 separate `(1, 512) × (512, 256)` multiplies.
3. **Useful noise.** A batch gradient is an unbiased but noisy estimate of the full gradient. That noise is mild regularization: it discourages settling into a sharp, brittle minimum.

**Ours: 64 out of ~550.** `batch_size=64` (`train.py:71`), consumed by `CGMEmbeddingDataModule` (`datamodule.py:50`). With 550 training sessions that is 8 full batches of 64 plus one of 38.

**Why the data is reshuffled every epoch.** `datamodule.py:64` sets `shuffle=True` on the training DataLoader only (validation and test loaders leave it off, lines 66-70). Two reasons:

- If the order were fixed, the same 64 sessions would form a batch every single epoch, so the model would see the same 9 gradients repeatedly and could lock onto quirks of that particular grouping.
- Our data arrives grouped by participant and by time. Without shuffling, one batch might be entirely one child's sessions from one afternoon — a heavily biased gradient. Shuffling mixes participants within each batch.

Shuffling is off for validation and test because the metrics are averages over the whole set, so order cannot change them, and a fixed order makes runs easier to compare.

**Say it in your own words:** *"A batch is how many sessions we look at before nudging the weights once; we use 64 out of about 550, which gives 9 nudges per pass instead of 1, uses the hardware properly, and adds a little helpful noise — and we reshuffle every epoch so the same 64 sessions never form the same batch twice and no batch is one child's afternoon."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Is `batch_size` a parameter or a hyperparameter?**
A: Hyperparameter — we set it, it is not learned. We did not tune it either; 64 came from Ben's setup.

**Q: What would batch size 550 (full batch) do to our epoch arithmetic?**
A: 1 optimizer step per epoch instead of 9, so 100 epochs would be 100 steps. Early stopping with patience 8 would also be measured in far less optimization, so the model would stop much earlier in terms of real progress.

**Q: Why is shuffling *not* enabled for the validation loader?**
A: The validation loss is a mean over the whole validation set, which is order-independent. Leaving order fixed keeps the number exactly reproducible.

</details>

---

## 11 · Epoch versus step versus iteration — the arithmetic

**This section is the answer to a question you were asked and could not answer. Learn the derivation, not just the number.**

Three words, three meanings:

| Term | Definition | In our fold |
|---|---|---|
| **Epoch** | one complete pass through all the training sessions | 9 optimizer steps |
| **Step** (also **iteration**, also **optimizer step**) | one batch: forward, backward, update the weights once | 1 batch of 64 (last one 38) |
| **Validation check** | one pass over the validation sessions, no weight updates | once per epoch |

**The derivation, from the top.** Do this on a whiteboard.

```
956  sessions with usable glucose                      (01_FACT_SHEET.md, section B)
 20  participants
  ↓  GroupKFold(n_splits=5) on participant id          (train.py:89)
     20 participants / 5 folds = 4 held out per fold
     → test    ≈ 191 sessions from  4 participants
     → outer-train ≈ 765 sessions from 16 participants
  ↓  GroupShuffleSplit(test_size=0.2) on the 16 outer-train participants   (train.py:92)
     → validation and training are split BY PARTICIPANT, not by row
     → ceil(0.2 × 16) = 4 participants go to validation
     → validation ≈ 190 sessions from  4 participants
     → training   ≈ 550 sessions from 12 participants
  ↓  batch_size = 64, DataLoader does not drop the remainder
     ceil(550 / 64) = 9
  ⇒  1 EPOCH = 9 OPTIMIZER STEPS
  ⇒  max_epochs = 100  ⇒  at most 900 steps  (early stopping ends it far sooner)
```

**Two versions of this arithmetic, and the difference is worth knowing.**

*The row-proportional estimate* in `01_FACT_SHEET.md` treats the 20% validation split as 20% of the *rows*: `765 × 0.2 = 153` validation, `612` training, `ceil(612/64) = 10` batches. That is the version to reproduce if someone asks you to sketch it quickly.

*What the code actually does* is split by **participant**. `GroupShuffleSplit`'s `test_size` is a fraction of **groups**, and scikit-learn rounds it up: `ceil(0.2 × 16) = 4` participants, i.e. 25% of the children. Because participants contributed unequal numbers of sessions (32 to 67, `01_FACT_SHEET.md` section B), those 4 children carry 23–28% of the rows. Measured across all 15 fold-trainings:

| Target | Fold | Train sessions (participants) | Val sessions (participants) | Test sessions (participants) | Steps/epoch |
|---|--:|--:|--:|--:|--:|
| Grids | 0 | 553 (12) | 210 (4) | 188 (4) | **9** |
| Grids | 1 | 560 (12) | 200 (4) | 191 (4) | **9** |
| Grids | 2 | 576 (12) | 183 (4) | 192 (4) | **9** |
| Grids | 3 | 553 (12) | 208 (4) | 190 (4) | **9** |
| Grids | 4 | 548 (12) | 213 (4) | 190 (4) | **9** |
| Symbols | 0–4 | 538–560 (12) | 173–197 (4) | 170–190 (4) | **9** |
| Prices | 0–4 | 546–566 (12) | 183–201 (4) | 186–189 (4) | **9** |

Every one of the 15 trainings has exactly **9 training batches per epoch** (`trainer.num_training_batches = 9`), because 538–576 all land in the range that `ceil(n/64) = 9` covers, namely 513 to 576.

So: **quote 9, and be able to explain why the back-of-envelope says 10.** The reason is that the validation split takes 4 of 16 children, not 20% of the rows.

**The timeline, drawn.**

```
epoch 0   |--s1--s2--s3--s4--s5--s6--s7--s8--s9--|  VAL  ->  early-stopping check
epoch 1   |--s10-s11-s12-s13-s14-s15-s16-s17-s18-|  VAL  ->  check
epoch 2   |--s19 ............................ s27-|  VAL  ->  check
   ...
epoch 17  |--s154 ........................... s162|  VAL  ->  check -> STOP (patience 8 used up)

          \______________ 9 steps ______________/  \_ 3 val batches, no updates _/
          shuffled differently every epoch
```

| Epoch | Optimizer steps | Cumulative steps | Validation checks so far |
|--:|---|--:|--:|
| 0 | 1–9 | 9 | 1 |
| 1 | 10–18 | 18 | 2 |
| 2 | 19–27 | 27 | 3 |
| 3 | 28–36 | 36 | 4 |
| 4 | 37–45 | 45 | 5 |
| 9 | 82–90 | 90 | 10 |
| 17 | 154–162 | 162 | 18 |
| 99 (never reached) | 892–900 | 900 | 100 |

**Because validation runs once per epoch, "patience 8" means 8 epochs.** That equivalence is the bridge between this section and section 15, and it only holds because `check_val_every_n_epoch = 1` (Lightning's default) and `val_check_interval = 1.0`. If someone set `check_val_every_n_epoch = 5`, patience 8 would mean 40 epochs.

**How much training actually happens.** Measured epochs completed per fold:

```
Grids   : 12, 10, 10, 10,  9   → 51 epochs
Symbols :  9,  9, 15, 12, 18   → 63 epochs
Prices  : 12, 12, 11, 10, 10   → 55 epochs
                        total  = 169 epochs × 9 steps = 1,521 optimizer steps
```

That is the entire compute cost of Arm B: **about 1,500 weight updates spread over 15 networks.** Not one of the 15 got past epoch 18 of a permitted 100. Anyone who imagines Arm B "did not train long enough" should see this table — training stopped because validation loss stopped improving, not because it ran out of epochs.

One more Lightning detail: before epoch 0, Lightning runs a **sanity check** of 2 validation batches (`num_sanity_val_steps = 2` by default). It catches crashes in `validation_step` early. It does **not** count toward patience — `EarlyStopping._should_skip_check` returns `True` while `trainer.sanity_checking` is set.

**Say it in your own words:** *"An epoch is one full pass over the training sessions; a step is one batch, one weight update. We hold out 4 of 20 children as test and 4 of the remaining 16 as validation, which leaves about 550 training sessions, and 550 divided by a batch of 64 rounds up to 9 — so one epoch is 9 optimizer steps, validation runs once per epoch, and patience 8 therefore means 8 epochs."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: How many optimizer steps in 100 epochs?**
A: 9 × 100 = 900 at most. No fold got past 18 epochs (162 steps).

**Q: If I doubled the batch size to 128, what changes?**
A: `ceil(550/128) = 5` steps per epoch instead of 9, so the weights move less often per pass; each gradient is less noisy; and patience 8 now covers half as much optimization. To keep progress comparable you would typically raise the learning rate.

**Q: Is "iteration" the same as "epoch"?**
A: No. Iteration = step = one batch. Epoch = a full pass = 9 iterations here. Mixing them up is the single most common vocabulary slip on this topic.

**Q: Why is the number of steps per epoch identical in all 15 folds?**
A: Coincidence of the ranges. Training sizes span 538 to 576, and `ceil(n/64) = 9` for every `n` from 513 to 576, so all 15 land on 9.

</details>

---

## 12 · The full training loop, which Lightning hides

Lightning is convenient and it hides the loop. Here is the loop it runs, in plain PyTorch. You should be able to write this from memory.

```python
for epoch in range(max_epochs):

    model.train()                          # dropout ON, see section 16
    for batch in train_loader:             # 9 batches, reshuffled this epoch
        pred = model(batch["glucose"])     # 1. FORWARD  -> (64, 1)
        pred = pred.squeeze(-1)            #             -> (64,)
        y    = (batch["label"] - mu) / sd  # 2. z-score the target with TRAIN stats
        loss = mse(pred, y)                # 3. COMPUTE LOSS (one number)
        loss.backward()                    # 4. BACKWARD  -> fills .grad on 132,609 tensors
        optimizer.step()                   # 5. UPDATE    -> w <- w - lr * f(grad)
        optimizer.zero_grad()              # 6. CLEAR     -> .grad back to zero

    model.eval()                           # dropout OFF
    with torch.no_grad():                  # no graph, no gradients
        val_loss = mean(mse(model(b["glucose"]).squeeze(-1),
                            (b["label"] - mu) / sd) for b in val_loader)

    # --- early-stopping check, once per epoch (section 15) ---
    if val_loss < best_val_loss:           # min_delta = 0, so strict improvement
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:               # patience = 8
            break
```

**Why `zero_grad()` is not optional.** PyTorch *accumulates* into `.grad` rather than overwriting it. Skip the clear and step 4 of batch 2 adds to batch 1's gradient, so your updates use a stale running sum. The bug shows up as training that mysteriously destabilises rather than crashing. (Accumulation is deliberate: it is how you simulate a large batch on small memory.)

**Where the six steps live in Lightning.** Lightning owns the two `for` loops, `.backward()`, `.step()`, `.zero_grad()`, the `train()`/`eval()` switching and the `no_grad()` context. You supply only the parts that are specific to your model.

| Loop element | Who writes it | Our code |
|---|---|---|
| `for epoch ...` and `for batch ...` | Lightning | `trainer.fit(lm, dm)` — `train.py:57` |
| forward + loss on a training batch | **you** | `training_step` → `_shared_step(batch, "train")` — `lightning_module.py:78-79` |
| `loss.backward()` | Lightning | — |
| `optimizer.step()`, `zero_grad()` | Lightning | — |
| which optimizer, which learning rate | **you** | `configure_optimizers` — `lightning_module.py:87-90` |
| forward + loss + metrics on a validation batch | **you** | `validation_step` — `lightning_module.py:81-82` |
| `model.train()` / `model.eval()` / `no_grad()` | Lightning | — |
| the early-stopping check | Lightning callback | `EarlyStopping(...)` — `train.py:53` |
| final scoring on the test participants | **you** + Lightning | `test_step` (`:84-85`) called by `trainer.test` (`train.py:58`) |

`training_step`, `validation_step` and `test_step` in our module all delegate to one `_shared_step(batch, stage)` (`lightning_module.py:54-76`), which is why the loss is computed identically in all three phases and only the logging and metric bookkeeping differ.

**Say it in your own words:** *"For each epoch, for each batch: forward to get a prediction, compute the loss, backward to get gradients, step the optimizer, zero the gradients — then one validation pass with dropout off and no gradients, then the early-stopping check. Lightning writes the loops and the optimizer calls; we write `training_step`, `validation_step` and `configure_optimizers`."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: What breaks if you forget `optimizer.zero_grad()`?**
A: Gradients accumulate across batches, so every update uses a running sum of all previous gradients. Training becomes unstable and effectively uses a growing learning rate. No error is raised.

**Q: What does `trainer.fit(lm, dm)` do, in one sentence?**
A: Runs the whole nested loop — up to 100 epochs of 9 training steps each, one validation pass and one early-stopping check per epoch — using our `training_step`, `validation_step` and `configure_optimizers`.

**Q: Why does `_shared_step` exist instead of three separate methods?**
A: So the training, validation and test loss are provably the same computation. The only differences are the logging flags and whether the torchmetrics collection is updated (`lightning_module.py:71-75`).

</details>

---

## 13 · Optimizers: SGD, momentum, Adam, AdamW

**The problem all optimizers solve.** Gradient descent's plain rule `w ← w − lr·g` uses the same step size for every parameter and forgets everything it has seen. Real loss surfaces are steep in some directions and nearly flat in others, so one global step size is either too big for the steep directions or too small for the flat ones.

**1. Plain SGD.**
```
w  ←  w  −  lr * g
```
Simple, memory-free, and very sensitive to the learning rate. Bounces across steep valleys while crawling along their floor.

**2. SGD with momentum.** Keep a running average of past gradients and step along that instead.
```
v  ←  β*v  +  g              (β typically 0.9)
w  ←  w − lr * v
```
The picture: a ball rolling downhill rather than a hiker re-deciding at every point. Consistent directions accumulate speed; oscillating directions cancel out.

**3. Adam** = momentum **plus** per-parameter step sizes.
```
m  ←  β₁*m + (1−β₁)*g            first moment  — a running average of the gradient
v  ←  β₂*v + (1−β₂)*g²           second moment — a running average of the SQUARED gradient
m̂  =  m / (1 − β₁^t)             bias correction (m and v start at 0, so early values are too small)
v̂  =  v / (1 − β₂^t)
w  ←  w − lr * m̂ / (√v̂ + ε)
```
The key line is the last one. Dividing by `√v̂` means each parameter's step is measured **in units of its own typical gradient size**. A parameter with consistently huge gradients gets a proportionally smaller multiplier; a parameter with tiny gradients gets a larger one. That is why one learning rate — 1e-3 — transfers across problems. PyTorch defaults: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`. Cost: two extra numbers stored per parameter, so 3 × 132,609 numbers of optimizer state for us. Trivial here; a serious consideration at 47.7 million.

**4. AdamW** = Adam with **decoupled weight decay**, and it is what we use.

```python
torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)     # lightning_module.py:90
```

The AdamW update, from PyTorch's own algorithm block, applies the decay as its own separate shrink *before* the Adam step:

```
θ  ←  θ − lr*λ*θ                 <-- decoupled weight decay, λ = 0.01
then the ordinary Adam step:
θ  ←  θ − lr * m̂/(√v̂ + ε)
```

**What "decoupled" means, and why it matters.** The old way ("L2 regularization") was to add `λ‖w‖²` to the loss. Then `λ·2w` appears inside the gradient `g` — and Adam divides `g` by `√v̂`. So the amount of shrinkage a weight receives ends up depending on that weight's gradient history, which is not what anyone intended: parameters with large, noisy gradients get *less* regularization. Decoupled decay pulls the `λθ` term out of the gradient entirely and applies it as a plain multiplicative shrink, identical for every parameter. Same idea, cleanly separated from the adaptive machinery.

| Optimizer | Memory per parameter | Per-parameter step size? | Momentum? | Notes |
|---|--:|---|---|---|
| SGD | 0 | no | no | needs careful tuning |
| SGD + momentum | 1 | no | yes | still the choice for large-scale vision |
| Adam | 2 | yes | yes | robust default; L2-in-the-loss interacts badly with it |
| **AdamW** (ours) | 2 | yes | yes | Adam with the decay applied separately |

**Say it in your own words:** *"Plain SGD subtracts the learning rate times the gradient; momentum averages recent gradients so it rolls rather than re-decides; Adam adds a per-parameter step size by dividing by a running average of the squared gradient, which is why one learning rate works everywhere; and AdamW shrinks every weight by `lr × 0.01` as a separate move instead of hiding it inside the gradient, so the shrinkage does not get rescaled by Adam."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: Is "Adam" an acronym?**
A: It stands for **adaptive moment estimation** — "moments" being the running averages of the gradient and of its square.

**Q: Why does `weight_decay` in Adam behave differently from `alpha` in Ridge?**
A: Ridge's `alpha` multiplies an L2 penalty added to a closed-form objective, so its effect is exact and analysable. AdamW's `weight_decay` is an iterative shrink applied once per step; the total shrinkage depends on how many steps you take. With 9 steps per epoch and ~12 epochs, that is roughly 108 shrinks of `1 − 0.001 × 0.01 = 0.99999` each — a total contraction factor of about 0.9989. Essentially nothing. See section 18.

**Q: How much optimizer state does our run hold?**
A: Two numbers (`m`, `v`) per trainable parameter = 265,218 floats, about 1 MB in float32. The frozen encoder contributes none, because it is not in the optimizer's parameter list.

**Q: Would SGD have changed our result?**
A: Almost certainly not in any way that matters. The validation loss turns upward within 1 to 9 epochs — the failure is that there is nearly nothing to learn (strongest raw r = −0.092), not that the search was inefficient.

</details>

---
---

# PART III — Keeping it honest

## 14 · Overfitting in a network: the two-curve picture

**Plain words.** **Overfitting** is when the model gets better at the sessions it trained on and simultaneously worse at sessions it has not seen. It has started memorizing instead of generalizing. With 132,609 parameters and ~550 training sessions, it is the *expected* behaviour, not a surprise.

This is the most important diagram in the chapter. It answers "what is overfitting" and "why do you need early stopping" at the same time.

```
  loss
    ^
    |  \                                                     validation loss
    |   \                                              ......:''
    |    \.                                    ..:''''
    |      '..                        ...:'''
    |         ''..            ...:''''                     <-- getting WORSE:
    |             ''.....:''''                                 memorizing training
    |                  *                                       sessions
    |                  |  best validation loss
    |     \            |
    |      '-.         |
    |         '--.     |                                    training loss
    |             '---.|_____
    |                  |     '''-----.____________          <-- keeps FALLING,
    |                  |                          '''-----      always
    +------------------|-------------------------------------> epoch
    0                epoch of best val loss           STOP HERE
      \_____________/ \___________ patience 8 epochs __________/
       underfitting        overfitting territory
```

Left of the star: both curves fall. The model is still learning things that transfer. Right of the star: training loss keeps falling because the model keeps fitting the training sessions harder, while validation loss rises because what it is now fitting is noise specific to those 12 children. The star is where you want the weights.

**The real thing, from our runs.** Symbols, fold 4 — the validation loss per epoch, measured:

```
val/loss
 1.40 | *
 1.39 |                             *
 1.38 |
 1.37 |    *                              *
 1.36 |       *
 1.35 |                                       *   <- epoch 17: training STOPS here
 1.33 |                          *  *
 1.32 |          * (e5)       *
 1.30 |             *
 1.30 | (e3) *  *        *
 1.29 |          (e6)  *
 1.28 |                (e9) *  <- BEST validation loss, epoch 9, 1.2813
      +--------------------------------------------
        0  1  2  3  4  5  6  7  8  9 10 ...     17
```

Compare against the constant: predicting the training mean on this fold's validation children gives MSE **1.3206** on the z-scored target. So the best the network ever managed on validation, 1.2813, was 3% better than a constant, and the point at which training stopped, 1.3535, was 2.5% **worse** than a constant. That is what "no better than guessing the average" looks like from the inside.

**Say it in your own words:** *"Training loss falls forever because the model can always fit the sessions in front of it harder; validation loss falls for a while and then turns upward, and the turn is the moment it starts memorizing the 12 training children instead of learning about glucose — on Symbols fold 4 that turn came at epoch 9."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Can you detect overfitting from the training loss alone?**
A: No. The training loss looks excellent precisely when overfitting is worst. You need held-out data, which is what the validation split is for.

**Q: Our validation loss turns up after 1 to 9 epochs. What does that timing tell you?**
A: That the model runs out of transferable signal almost immediately. With a genuine relationship you would expect many epochs of joint improvement before the split. One epoch is 9 weight updates — the model barely moved before it started overfitting.

**Q: Which is worse for us, overfitting or underfitting?**
A: Overfitting, and we can point at the numbers: unregularized `LinearRegression` on the raw 512 features reaches R² ≈ −3 (`02_ML_FROM_ZERO.md`), and Arm B at −0.14 to −0.28 is worse than Ridge at −0.012 to −0.089. Every step toward *more* restraint has helped.

</details>

---

## 15 · Early stopping — the deep treatment

**This is the question you were asked and could not answer. This section is the complete answer.**

### 15.1 What it is, in one sentence

Early stopping means: watch a number computed on data the model is not training on, and halt training when that number stops getting better — even if you were permitted many more epochs.

### 15.2 Our exact configuration

```python
EarlyStopping(monitor="val/loss", patience=8, mode="min")      # train.py:53
```

| Setting | Value | Meaning |
|---|---|---|
| `monitor` | `"val/loss"` | the quantity watched: MSE on the **z-scored** target over the validation sessions, logged at `lightning_module.py:71` |
| `mode` | `"min"` | lower is better, so an improvement means a smaller value |
| `patience` | **8** | number of consecutive checks with no improvement that we tolerate before stopping |
| `min_delta` | **0.0** (Lightning default, not set by us) | how much better a value must be to count as an improvement — zero means *any* decrease counts |
| how often checked | **once per epoch** | because `check_val_every_n_epoch = 1` (Lightning default) |
| `max_epochs` | 100 | the ceiling if early stopping never fires — it always fired |

Because the check happens once per epoch, **patience 8 means 8 epochs**, which here means 72 optimizer steps. Lightning's own documentation flags this: patience counts *validation checks*, not epochs, and they coincide only when you validate every epoch.

### 15.3 The mechanics, exactly as Lightning implements them

From `lightning/pytorch/callbacks/early_stopping.py`:

```python
elif self.monitor_op(current - self.min_delta, self.best_score):   # for mode="min": current - 0 < best
    should_stop = False
    self.best_score = current
    self.wait_count = 0                     # improvement -> reset the counter
else:
    self.wait_count += 1                    # no improvement -> tick the counter
    if self.wait_count >= self.patience:    # 8
        should_stop = True
```

Three things to notice:

1. With `min_delta = 0` and `mode = "min"`, an improvement requires `current < best` — **strictly** less. Equal is not an improvement.
2. The counter **resets to 0** on any improvement. It counts *consecutive* failures, not total failures.
3. The comparison is always against the **best value ever seen**, not against the previous epoch's value.

### 15.4 A real 18-epoch walkthrough (Symbols, fold 4)

These are the measured `val/loss` values from a re-run of the exact code. Track the `wait` column and see where the stop lands.

| Epoch | `val/loss` | Improvement vs best so far? | `best_score` after | `wait` | Action |
|--:|--:|---|--:|--:|---|
| 0 | 1.4004 | yes (first) | 1.4004 | 0 | continue |
| 1 | 1.3655 | yes | 1.3655 | 0 | continue |
| 2 | 1.3571 | yes | 1.3571 | 0 | continue |
| 3 | 1.2999 | yes | 1.2999 | 0 | continue |
| 4 | 1.2967 | yes | 1.2967 | 0 | continue |
| 5 | 1.3220 | no | 1.2967 | 1 | continue |
| 6 | 1.2892 | yes | 1.2892 | **0 (reset)** | continue |
| 7 | 1.3048 | no | 1.2892 | 1 | continue |
| 8 | 1.2970 | no | 1.2892 | 2 | continue |
| 9 | **1.2813** | yes — **best of the whole run** | **1.2813** | **0 (reset)** | continue |
| 10 | 1.2879 | no | 1.2813 | 1 | continue |
| 11 | 1.3251 | no | 1.2813 | 2 | continue |
| 12 | 1.3226 | no | 1.2813 | 3 | continue |
| 13 | 1.2981 | no | 1.2813 | 4 | continue |
| 14 | 1.3887 | no | 1.2813 | 5 | continue |
| 15 | 1.3390 | no | 1.2813 | 6 | continue |
| 16 | 1.3974 | no | 1.2813 | 7 | continue |
| 17 | 1.3535 | no | 1.2813 | **8** | **`wait >= patience` → STOP** |

Read epochs 7 and 8 carefully: `wait` reached 2, then epoch 9 improved and reset it to 0. Without that reset the run would have stopped at epoch 15 instead of 17. Then epochs 10 through 17 are 8 consecutive non-improvements, and the stop fires at epoch 17. Training ran **18 epochs** (0 through 17) = **162 optimizer steps**, out of a permitted 900.

A second real example, Grids fold 2, which is the common pattern in our runs:

```
epoch:      0       1       2       3       4       5       6       7       8       9
val/loss: 1.4294  1.1806  1.2721  1.1862  1.2553  1.2200  1.2413  1.3215  1.2752  1.3252
best at:          ^^^^^^  (epoch 1)
wait:        0      0       1       2       3       4       5       6       7       8  -> STOP
```

Best at epoch 1; epochs 2 through 9 are 8 consecutive misses; stop at epoch 9, after 10 epochs. Note that epoch 3's 1.1862 was *close* to the best 1.1806 but not better, so it did not reset the counter. With `min_delta = 0`, near-misses count as failures.

Across all 15 fold-trainings, measured:

| Target | Epochs completed per fold | Epoch of best validation loss |
|---|---|---|
| Grids | 12, 10, 10, 10, 9 | 3, 1, 1, 1, 0 |
| Symbols | 9, 9, 15, 12, 18 | 0, 0, 6, 3, 9 |
| Prices | 12, 12, 11, 10, 10 | 3, 3, 2, 1, 1 |

The best epoch is **0 to 9**. Three of the fifteen networks peaked at epoch 0 — their very first validation check was never beaten.

### 15.5 Honest problem 1: we do NOT restore the best weights

**This is the most important honest point in the chapter.**

Look at `train.py:51-58`:

```python
trainer = pl.Trainer(
    max_epochs=max_epochs, accelerator=accelerator, devices=1,
    callbacks=[EarlyStopping(monitor="val/loss", patience=patience, mode="min")],
    logger=False, enable_checkpointing=False,          # <-- no checkpoints are written
    ...)
trainer.fit(lm, dm)
res = trainer.test(lm, dm, verbose=False)[0]           # <-- no ckpt_path argument
```

There is **no `ModelCheckpoint` callback** and `enable_checkpointing=False`, so nothing is ever saved to disk. `EarlyStopping` in Lightning only sets `trainer.should_stop`; it does not save or restore weights — that is `ModelCheckpoint`'s job. And `trainer.test(lm, dm)` passes the model object with `ckpt_path=None`, which Lightning resolves by returning `None` (`checkpoint_connector.py:138-140`: *"use passed model to function without loading weights"*), i.e. **the weights currently in memory**.

So the reported test metrics come from the weights at the moment training stopped — **up to 8 epochs past the best validation loss.** On Symbols fold 4 that means the weights from epoch 17 (`val/loss` 1.3535) are reported, not the weights from epoch 9 (`val/loss` 1.2813).

**How much does this cost us?** Measured, by re-running all 15 fold-trainings while snapshotting the best-validation-loss weights and scoring both:

| Score | R² reported (weights at stop) | R² if best weights were restored | Difference |
|---|--:|--:|--:|
| Grids | −0.195 | **−0.090** | 0.105 |
| Symbols | −0.279 | **−0.167** | 0.112 |
| Prices | −0.136 | **−0.058** | 0.078 |

(RMSE moves the same way: Grids 0.530 → 0.504, Symbols 0.610 → 0.586, Prices 18.27 → 17.62.)

So the fix would roughly **halve** Arm B's deficit and bring it level with Arm A's Ridge (−0.089 / −0.056 / −0.012). **Arm B currently looks worse than it should.** State that plainly if asked, and note what it does *not* change: even with best-weight restoration every value is still at or below zero, so the conclusion — no better than guessing the average — is unchanged. It is a fairness bug in the comparison, not the reason the project found no signal.

*(Those re-run figures come from today's environment. The published headline numbers in `results/headtohead_real.md` are −0.218 / −0.281 / −0.168 from the 2026-07-07 run. Fold-level values shift by a few hundredths between runs because weight initialization and batch shuffling depend on the library versions in the random-number stream; the means agree to within the fold-to-fold spread, which for Grids is ±0.10. Quote the published numbers as the headline and be able to explain the drift — with 550 training sessions, run-to-run variation of that size is itself a finding worth stating.)*

**The fix, concretely:**

```python
from lightning.pytorch.callbacks import ModelCheckpoint
ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
trainer = pl.Trainer(..., callbacks=[EarlyStopping(...), ckpt], enable_checkpointing=True)
trainer.fit(lm, dm)
res = trainer.test(lm, dm, ckpt_path=ckpt.best_model_path)[0]   # load the best weights
```

Three lines. Not yet applied — say "not yet applied", not "we should have".

### 15.6 Honest problem 2: the stopping criterion and the reported metric are on different scales

`val/loss` is MSE on the **z-scored** target (`lightning_module.py:67-68`). The reported R², RMSE and MAE are computed after converting predictions back to the score's real units (`lightning_module.py:70,74`). Nothing is wrong — the z-score is a fixed affine map using train-split statistics, so minimizing MSE on the z-scored target is equivalent to minimizing MSE on the raw target, and the ordering of models is preserved. But two consequences are worth knowing:

- You cannot compare a `val/loss` of 1.28 with a `test/rmse` of 0.61 directly. `rmse ≈ √(val_loss) × target_std` gets you into the right region.
- Stopping is chosen on a validation-MSE criterion while the headline is an R² on a *different* set (the test participants). Improving one does not have to improve the other, and on Symbols folds 0, 3 and 4 the best-validation weights actually scored a hair *worse* on test than the final weights (−0.304 vs −0.322, −0.005 vs −0.002, −0.216 vs −0.214). With 4 validation children the criterion is noisy enough for that to happen.

### 15.7 Honest problem 3: early stopping is itself tuning, and Arm B is regularized too

Two claims that appear in our earlier notes are too strong, and a careful reader will catch both.

**"Arm A tunes hyperparameters, Arm B tunes nothing."** Not right. Arm B chooses **when to stop** using validation data. The stopping epoch is a hyperparameter selected from data — it just happens to be selected by a rule rather than by a grid search. The honest version of the difference is:

| | Arm A | Arm B |
|---|---|---|
| What is chosen from held-out data | Ridge `alpha` from {0.1, 1, 10, 100, 1000}; SVR `C`, `gamma` | the stopping epoch, from {1, …, 100} |
| Chosen using | `GridSearchCV`, 3-fold grouped inner CV on the outer-train participants | one grouped validation split, 4 of 16 participants |
| Number of held-out estimates behind the choice | 3 inner folds averaged | 1 split |
| Selection budget | 5 or 6 candidate settings | up to 100 candidate stopping points |

Arm B's selection is if anything *less* well protected: one split instead of three averaged folds, and up to 100 candidates instead of 5.

**"Arm A is regularized and Arm B isn't."** Also not right. Arm B carries **three** regularizers at once: `weight_decay = 0.01` inside AdamW (an L2 shrink, section 18), `Dropout(0.1)` (section 16), and early stopping itself. The real difference is not presence but *character*: Ridge's `alpha` has a closed-form solution and a clean tuning path, whereas Arm B's three restraints interact and none of them is tuned.

### 15.8 Why early stopping is a form of regularization

Regularization means restricting the set of functions the model can end up expressing (`02_ML_FROM_ZERO.md`, section 9). Early stopping does this through the optimizer rather than through the objective.

Weights start small and random (section 19). Each step moves them a bounded distance. Stop after `k` steps and the weights cannot have travelled further than roughly `k × lr × (typical gradient)` from the initialization. So the reachable set of functions is a neighbourhood around the initialization, and `k` sets its radius. Fewer steps = smaller neighbourhood = less effective capacity — even though the parameter count never changes.

The point is sharp in our numbers. Nominally the head has 132,609 free parameters. In practice it took 10 to 18 epochs, i.e. 90 to 162 updates, from a small random start. The weights barely moved. The **effective** capacity used was a tiny fraction of the nominal capacity, and that is exactly what early stopping bought.

For linear models this connection is a theorem: gradient descent on squared error, stopped early, gives a solution close to the ridge-regression solution for a particular `alpha`, with more steps corresponding to smaller `alpha`. So "early stopping" and "L2 penalty" are two routes to overlapping places. This is why it is fair to say Arm B and Arm A are both regularized, just by different mechanisms with different guarantees.

**Say it in your own words:** *"Early stopping watches the validation loss once per epoch and stops when it has failed to beat its own best for 8 epochs in a row, with any decrease counting as an improvement because `min_delta` is 0. It is regularization because stopping after 90 to 160 updates keeps the weights near their small random start, so the model never reaches the wilder functions its 132,609 parameters could describe. And our version has a real flaw: we save no checkpoint, so we report the weights from the epoch we stopped at, not the best epoch — restoring the best weights would improve Arm B's R² by about 0.08 to 0.11 on each of the three scores."*

<details>
<summary><b>Drill</b> — 5 questions</summary>

**Q: With patience = 8, validation losses 1.0, 0.9, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89 — does it stop?**
A: No. The best is 0.9 at epoch 1; epochs 2 through 7 give `wait` = 1..6; epoch 8's 0.89 beats 0.90... careful — the best is 0.90 at epoch 1, and the values 0.95, 0.94, 0.93, 0.92, 0.91 are all worse, so `wait` climbs to 5; then 0.90 at epoch 7 ties the best, which with `min_delta = 0` is **not** an improvement (strict less-than), so `wait` = 6; then 0.89 at epoch 8 is a genuine improvement and resets `wait` to 0. Training continues.

**Q: Why does `min_delta = 0` matter?**
A: It makes the tiniest decrease count as progress, so the counter resets on noise. That makes stopping later and more erratic than with, say, `min_delta = 0.01`. It also means an exact tie counts as a failure.

**Q: Which weights are reported in our results?**
A: The ones in memory when training stopped — up to 8 epochs past the best validation loss. `enable_checkpointing=False` and there is no `ModelCheckpoint`, so Lightning restores nothing, and `trainer.test(lm, dm)` scores the current weights.

**Q: Does early stopping use the test participants?**
A: No, and this is asserted in code. `train.py:95-96` asserts that the participant sets for train, validation and test are pairwise disjoint, and crashes the run otherwise. The stopping decision sees only the 4 validation children.

**Q: How reliable is a stopping decision based on 4 children?**
A: Not very, and we should say so. The validation set is 4 of 20 participants (173 to 213 sessions), so `val/loss` moves with those particular children's session-to-session noise. The measured best epoch ranges from 0 to 9 across 15 otherwise-identical trainings, which is direct evidence of how unstable the signal is.

</details>

---

## 16 · Dropout

**Plain words.** During training, dropout randomly sets a fraction of a layer's outputs to exactly zero, choosing a fresh random set for every batch. Ours is `nn.Dropout(0.1)` — 10% of the 256 hidden values, so about 26 of them, blanked per session per batch.

**The mechanics, including the part people forget.** PyTorch uses "inverted dropout": survivors are scaled up by `1/(1−p)` so the layer's average output size does not change.

```
p = 0.1  ->  scale = 1/(1 − 0.1) = 1/0.9 = 1.1111

  training  (model.train()):
    input : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    output: [1.111, 1.111, 0, 1.111, 1.111, 1.111, 1.111, 1.111, 1.111, 1.111]
                            ^ dropped                          (verified in PyTorch)

  evaluation (model.eval()):
    input : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    output: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]      <-- identity. Nothing dropped, no scaling.
```

**Why it regularizes.** Three complementary explanations:

1. **No single hidden unit can be relied on.** If unit 137 encodes a useful pattern, it vanishes on roughly a tenth of the batches, so the network must spread the information across units. That discourages brittle single-feature detectors.
2. **It approximates an ensemble.** Each batch trains a randomly thinned sub-network; the weights are shared across all those sub-networks; at evaluation you use the full network, which behaves roughly like averaging them. Averaging many models reduces variance.
3. **It is noise injection.** Adding noise to intermediate representations makes the fitted function smoother.

**The classic exam point: dropout is ACTIVE in training and completely OFF at evaluation.** The switch is the module's mode:

- `model.train()` → dropout active, random zeroing
- `model.eval()` → dropout is the identity function

Get this wrong and you get one of two classic bugs: leave it in `train()` mode during evaluation and your metrics are noisy and pessimistic; leave it in `eval()` mode during training and you silently lose your regularizer. **Lightning switches this for you** — it calls `model.train()` before the training loop and `model.eval()` around validation and test. That is why our code never calls either one. If you ever write a raw PyTorch loop, you must call them yourself.

A useful consequence: dropout is one reason training loss can look *worse* than validation loss early on. The training loss is measured with a crippled network; the validation loss is measured with the full one.

**Say it in your own words:** *"Dropout zeroes a random 10% of the 256 hidden values on every batch and scales the survivors by 1/0.9, so no single unit can be depended on and the network is forced to spread information out — and it is completely switched off at evaluation time, which Lightning handles by calling `model.train()` and `model.eval()` for us."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: Why scale survivors by 1/(1−p)?**
A: So the expected sum of the layer's outputs is unchanged between training and evaluation. Without it, the next layer would see systematically smaller inputs during training than at test time and every learned weight would be miscalibrated.

**Q: Our dropout is 0.1. Is that a lot?**
A: No — 0.1 is on the light side; 0.2 to 0.5 is common in fully-connected layers. Given 132,609 parameters on ~550 training sessions, a stronger value would be a defensible experiment. It is a hyperparameter we did not tune; 0.1 is the value inherited from Ben's head.

**Q: Where exactly is dropout applied in our head, and why there?**
A: Between the GELU and the second Linear (`model.py:150`), so it perturbs the 256 hidden activations. Applying it to the 512 inputs instead would corrupt the frozen Chronos features; applying it after the final Linear would just add noise to the prediction.

**Q: Does dropout change the parameter count?**
A: No — zero parameters. `p = 0.1` is a setting, not something learned.

</details>

---

## 17 · LayerNorm, and how it differs from BatchNorm and StandardScaler

**Plain words.** LayerNorm takes one session's 512 numbers, subtracts their mean, divides by their standard deviation, then applies a learned scale and a learned shift — one of each per feature position.

**The formula.** For one sample `x` with `n = 512` features:

```
mu    = (1/n) Σ x_i                       mean over the 512 features of THIS sample
var   = (1/n) Σ (x_i − mu)²               variance over the same 512
x̂_i   = (x_i − mu) / sqrt(var + eps)      eps = 1e-5 (PyTorch default)
out_i = gamma_i * x̂_i + beta_i            gamma init 1, beta init 0; both LEARNED
```

`gamma` (`weight`) and `beta` (`bias`) are the 512 + 512 = 1,024 parameters from section 5. They are initialized to 1 and 0, so at the start LayerNorm is exactly "normalize", and training can move it away from that if useful.

**Worked on four numbers** (verified in PyTorch): input `[1, 2, 3, 4]`, mean 2.5, population standard deviation 1.1180.

```
(1 − 2.5)/1.1180 = −1.3416
(2 − 2.5)/1.1180 = −0.4472
(3 − 2.5)/1.1180 = +0.4472
(4 − 2.5)/1.1180 = +1.3416
```

Output `[−1.3416, −0.4472, 0.4472, 1.3416]`. The tiny `eps` inside the square root is why these are not the last-digit-exact analytic values — it exists to prevent division by zero on a constant input.

**The axis is the whole point.** This is the most confused topic in the area, so here is the three-way comparison in full.

| | **LayerNorm** (ours, Arm B) | **BatchNorm** | **StandardScaler** (Arm A) |
|---|---|---|---|
| Normalizes across | the **512 features of one sample** | the **samples in one batch, per feature** | the **rows of the training fold, per feature** |
| Statistics come from | that single sample | the current batch (training) / a running average (eval) | the training fold, computed once and stored |
| Depends on other samples? | **no** | yes | yes (training rows only) |
| Depends on batch size? | no | yes — unstable with small batches | no |
| Behaves differently in train vs eval? | **no** | **yes** | no |
| Learned parameters | 1,024 (512 scale + 512 shift) | 2 per feature | none |
| Can leak test information? | **impossible** | yes, if statistics are computed over data including test rows | only if fit outside the training fold — ours is fit inside the pipeline (`regression.py:116`) |
| In our code | `model.py:147` | not used | `regression.py:116`, inside the `Pipeline` |

**Why LayerNorm cannot leak.** Leakage means information from the test participants influencing the model. LayerNorm's statistics come from the *same row* being normalized — nothing else. So each session is normalized identically whether it is processed alone, in a batch of 64, in training, or at test time. There is no dataset-level statistic to fit and therefore nothing that could be fitted on the wrong rows. That is a genuine advantage worth stating, and it is one thing about Arm B that is unambiguously clean.

BatchNorm, by contrast, has caused real leakage bugs in published work: it needs batch statistics at training time and running averages at evaluation time, and if those running averages are ever updated on evaluation data, test information enters the model.

**Honest problem 4: Arm A and Arm B do not see identically preprocessed features.** It is tempting to say "both arms get the same 512 Chronos numbers, only the predictor differs". That overstates it. The actual difference:

| | Arm A | Arm B |
|---|---|---|
| Feature preprocessing | `StandardScaler` — **per feature, across the training-fold rows** (`regression.py:116`) | `LayerNorm` — **per sample, across the 512 features** (`model.py:147`), plus learned scale and shift |
| Optional dimensionality reduction | PCA available in the pipeline | none |
| Target preprocessing | **none** — Ridge and SVR fit the raw score | **z-scored** with train-split mean and standard deviation (`lightning_module.py:67`) |

These are different transformations, not the same transformation done twice. StandardScaler makes feature 7 comparable across sessions; LayerNorm makes one session's 512 numbers comparable to each other and destroys that session's overall magnitude and spread. If a session's *overall embedding magnitude* carried signal, Arm A could use it and Arm B could not. The honest sentence is: **"both arms start from the same 512 frozen Chronos numbers per session; after that the preprocessing differs as well as the predictor."**

**Say it in your own words:** *"LayerNorm normalizes each session's own 512 numbers to mean 0 and standard deviation 1, then applies a learned scale and shift, so it needs no dataset statistics and cannot leak anything from the test children — unlike BatchNorm, which normalizes each feature across the batch, and unlike StandardScaler, which normalizes each feature across the training fold and is what Arm A uses."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: Why is LayerNorm the right normalizer for a network and StandardScaler the right one for Ridge?**
A: They solve different problems. Ridge needs each *feature* on a comparable scale so one L2 penalty is fair to all 512 coefficients — that is a per-feature, per-dataset job. A network needs each *layer's* inputs in a well-conditioned range so gradients neither vanish nor explode — that is a per-sample job, and doing it per-sample makes it batch-size independent and leak-proof.

**Q: LayerNorm on a session whose 512 numbers are all identical — what happens?**
A: The variance is 0, so the output is `0/sqrt(0 + 1e-5) = 0` for every position, then `gamma·0 + beta = beta`. The `eps` is what stops it being a division by zero.

**Q: Someone asks "did your normalization leak test data?" Answer for both arms.**
A: Arm B: impossible — LayerNorm uses only the row it is normalizing, and the target z-score uses the training split's mean and standard deviation only (`datamodule.py:55-61`). Arm A: `StandardScaler` is a step inside the scikit-learn `Pipeline` (`regression.py:116-119`), so `GridSearchCV` refits it on each training split; it never sees the test fold. Both are clean, and there is an assertion on participant disjointness in both arms.

**Q: Does LayerNorm behave differently in `train()` and `eval()` mode?**
A: No. It is the same computation either way, which is another advantage over BatchNorm.

</details>

---

## 18 · Weight decay, and how it relates to Ridge's L2 penalty

**Plain words.** Weight decay shrinks every weight slightly toward zero on every optimizer step. Small weights mean a gentler, smoother function, so it is a way of preferring simple explanations.

**Ours.** `weight_decay = 0.01`, the default in `CGMRegressionModule.__init__` (`lightning_module.py:35`) and passed to AdamW at line 90. Note that `train.py` never overrides it, so 0.01 is what every one of the 15 networks used.

**The connection to Ridge.** Ridge minimizes

```
Σ (prediction − truth)²  +  alpha * Σ w²
```

Differentiate the penalty: `d/dw (alpha·w²) = 2·alpha·w`. So a gradient-descent step on Ridge's objective includes `−lr·2·alpha·w`, which is exactly a shrink proportional to `w`. **AdamW's `weight_decay` is that same shrink, applied directly instead of routed through the loss.** Same idea, two implementations:

| | Ridge `alpha` | AdamW `weight_decay` |
|---|---|---|
| Applied | as a term in the objective, solved in closed form | as a multiplicative shrink, once per step |
| Total effect | exact, independent of the optimizer | depends on how many steps you take |
| Tuned in our project | **yes** — {0.1, 1, 10, 100, 1000} via `GridSearchCV` | **no** — fixed at 0.01 |
| Interacts with Adam's per-parameter scaling | n/a | **no**, and that is the point of the "W" (section 13) |

**How much shrinkage do we actually get? Almost none.** Each step multiplies every weight by `(1 − lr·λ) = 1 − 0.001 × 0.01 = 0.99999`. Over a typical fold of 12 epochs × 9 steps = 108 steps:

```
0.99999^108 = 0.9989
```

A total contraction of about **0.1%**. So while it is true that Arm B is L2-regularized, the honest addition is that the amount is negligible at this learning rate and this step count. Arm B's real restraints are dropout and early stopping. If you wanted weight decay to do meaningful work at ~100 steps you would need it orders of magnitude larger — and the clean way to explore that is precisely the `alpha` sweep Arm A already runs.

**Say it in your own words:** *"Weight decay shrinks every weight a little on every step, which is the same idea as Ridge's L2 penalty — differentiate `alpha·w²` and you get a shrink proportional to `w`. Ours is 0.01, untuned, and over about 108 steps it contracts the weights by only 0.1%, so it is real but tiny; Ridge's alpha, which we do tune over five values, does far more work."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Is weight decay the same as L2 regularization?**
A: The same idea, two implementations. Adding `λ‖w‖²` to the loss puts the shrink inside the gradient, where Adam then rescales it per parameter. AdamW's decoupled decay applies the shrink separately so it is uniform. For plain SGD the two are equivalent up to a factor.

**Q: "Arm A is regularized and Arm B isn't." Correct the statement.**
A: Both are. Arm A uses a tuned L2 penalty (`alpha` over five values, chosen by grouped inner CV). Arm B uses weight decay 0.01, dropout 0.1 and early stopping, none of them tuned — and its weight decay contracts the weights by roughly 0.1% over a typical fold, so it is nearly inert. The difference is quality of regularization, not presence.

**Q: Should the bias terms be decayed?**
A: Usually not — decaying biases restricts the model's ability to shift its output without any generalization benefit. `torch.optim.AdamW` decays every parameter you hand it, so ours decays the 257 bias values and the 1,024 LayerNorm parameters too. Excluding them is a standard refinement we did not apply; at 0.1% total shrinkage it makes no practical difference here.

</details>

---

## 19 · Weight initialization, and why not all zeros

**Plain words.** Before training, the weights need starting values. They are set randomly, small.

**What PyTorch does by default for `nn.Linear`.** Weights and biases are drawn uniformly from `(−1/√in, +1/√in)`. For our `Linear(512, 256)` that is `±1/√512 = ±0.04419` (verified: the sampled weight range was −0.04419 to +0.04419). Two consequences of scaling the bound by `1/√in`: the weighted sum of `in` inputs has roughly the same size regardless of how wide the layer is, which keeps activations in a useful range; and the weights start small, so the function starts near-flat — early stopping's regularization argument (section 15.8) depends on this.

`nn.LayerNorm` initializes differently and deliberately: `weight` (scale) to all **ones** and `bias` (shift) to all **zeros**, so it starts as pure normalization.

**Why you cannot start all weights at zero.** Two failures, and the second is the deep one.

*Symmetry.* If every neuron in a layer has identical weights, every neuron computes the identical value and receives the identical gradient, so they update identically and stay identical forever. 256 hidden units behave as 1. Random initialization is what **breaks the symmetry** so different units can specialize.

*Total collapse in our specific head.* Set both Linear layers to all-zero weights and biases and trace it:

```
LayerNorm output  -> nonzero (LayerNorm's own weight starts at 1)
Linear1 (W = 0)   -> all zeros
GELU(0)           -> 0
Linear2 (W = 0, b = 0) -> prediction = 0 for every session

backward:
  dL/dW2 = (dL/dout) * h^T,  and h = 0            -> dL/dW2 = 0
  dL/db2 = dL/dout                                -> NONZERO, b2 can move
  dL/dh  = W2^T (dL/dout),   and W2 = 0           -> dL/dh = 0
  dL/dW1 = (dL/dh) * a^T                          -> 0
```

So the only parameter that ever moves is the output bias. It drifts toward the mean of the target, and the network predicts a constant forever. Verified experimentally: after 400 AdamW steps from an all-zero Linear initialization on synthetic data, both weight matrices were still **exactly** 0.0, the predictions had a standard deviation of 3e-8 (i.e. constant), and only the output bias had moved.

The punchline is a nice one: **a zero-initialized version of our head is precisely the mean-predictor baseline.** It would score R² ≈ 0 on training data — which, given our results, is uncomfortably close to what the properly-initialized version achieves on held-out children.

**Say it in your own words:** *"Weights start as small random numbers drawn from plus-or-minus one over the square root of the input width, which for our 512-input layer is about 0.044. They must be random, because identical weights give identical gradients and 256 hidden units would forever behave as one — and in our head, all-zero weights make the gradient into the first layer exactly zero, so only the output bias ever moves and the network is stuck predicting a constant."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: Why does the initialization bound depend on the input width?**
A: Because a neuron sums `in` terms. Without shrinking the individual weights as `in` grows, the sum's typical magnitude would grow like `√in`, pushing activations into the saturated or exploding regions and making gradients behave badly.

**Q: Can biases be initialized to zero?**
A: Yes, safely. Bias symmetry is broken by the weights already, since each neuron receives a different weighted sum. PyTorch happens to initialize `nn.Linear` biases randomly too, but zero would be fine.

**Q: Does initialization affect our results?**
A: Yes, measurably, and that is itself worth reporting. Fold-level R² shifts by a few hundredths between runs whose only difference is the library versions that determine the random-number stream. With ~550 training sessions and no real signal, run-to-run variation of that size is expected — which is why `seed_everything(42 + fold_idx)` is called at the top of every fold (`train.py:44`) to make a given run reproducible.

</details>

---
---

# PART IV — The verdict

## 20 · The four soft spots in Arm B, collected

Have these ready. A careful reader of the code will find all four, and knowing them first is the difference between "he understands his own limitations" and "he was caught".

| # | The problem | The precise statement | Fix | Effect on our conclusion |
|---|---|---|---|---|
| 1 | **Best weights are not restored** | `enable_checkpointing=False`, no `ModelCheckpoint`, and `trainer.test(lm, dm)` scores the in-memory weights — up to 8 epochs past the best validation loss | add `ModelCheckpoint(monitor="val/loss")` and pass `ckpt_path=ckpt.best_model_path` to `trainer.test` | Arm B looks **worse than it should** by about 0.08–0.11 R². Restoring the best weights gives −0.090 / −0.167 / −0.058. All still at or below zero |
| 2 | **Stopping criterion and reported metric are on different scales** | `val/loss` is MSE on the **z-scored** target; R²/RMSE/MAE are computed after converting back to real score units | none needed; it is correct, just needs stating | none — the z-score is a fixed affine map, so it preserves the ordering of models |
| 3 | **Early stopping is tuning, and Arm B is regularized** | Arm B selects its stopping epoch from validation data (up to 100 candidates, one split, 4 children), and carries weight decay 0.01 plus dropout 0.1 | describe the arms accurately: both tune, both regularize | none — but "Arm B tunes nothing" and "Arm B isn't regularized" are both wrong and must not be said |
| 4 | **The two arms do not see identically preprocessed features** | Arm A: `StandardScaler` per feature across training-fold rows, raw target. Arm B: `LayerNorm` per sample across 512 features plus learned scale/shift, z-scored target | say "same frozen embeddings, different preprocessing **and** different predictor" | none — but "only the predictor differs" is an overstatement |

None of the four changes the headline. Together they mean the fair version of Arm B is *level with* Arm A rather than clearly behind it — and level with Arm A still means no better than guessing the average.

---

## 21 · Why Arm B lost

Arm B is our least accurate arm. Real data, 20 participants, grouped 5-fold cross-validation, R² against a mean-predictor (0 = tied with guessing the average, negative = worse):

| Score | Arm A: Chronos + Ridge/SVR | **Arm B: Chronos + MLP head** | 43 hand-crafted features |
|---|--:|--:|--:|
| Grids | −0.089 | **−0.218** | −0.065 |
| Symbols | −0.056 | **−0.281** | −0.038 |
| Prices | −0.012 | **−0.168** | −0.009 |

Five reasons, in order of importance. This ordering is the answer to "why did the fancier model lose?"

**1. There is almost nothing to find.** The strongest correlation between any glucose statistic and any score across all 956 sessions is r = −0.092 (fraction of readings below 70, versus Symbols), so r² = **0.0085** — 0.85% of the variation. Every other pairing is under 0.6% (`01_FACT_SHEET.md`, section E). No model manufactures signal that is not there. Start here; every other reason is secondary.

**2. 132,609 parameters on ~550 training sessions.** That is 217 parameters per training example at the fact sheet's figure, 241 at the measured one. Arm A's Ridge has 513 parameters on ~765 training sessions — 0.67 per example, 320× less freedom. When there is no signal, extra freedom buys extra ways to fit noise, and that is precisely what negative R² measures.

**3. The early-stopping signal comes from 4 children.** The validation split holds 4 of 20 participants (173–213 sessions). Every decision about when to stop rests on those 4 children's session-to-session variability. The evidence that this is noisy is in our own numbers: across 15 otherwise-identical trainings, the epoch of best validation loss ranged from **0 to 9**, and three networks never beat their first check. Arm A's `alpha` is chosen from 3 averaged inner folds instead — a better-conditioned decision on the same data.

**4. No closed-form regularization path.** Ridge has an exact solution for every `alpha`, so a 5-value grid genuinely explores the whole strong-to-weak regularization spectrum and picks the best point on it. Arm B's restraint comes from three interacting, untuned mechanisms — dropout 0.1, weight decay 0.01 (which contracts weights by 0.1% over a fold, section 18), and a stopping rule watching 4 children. Arm B *can* express something close to Ridge's answer: with the right weights the hidden layer can be made nearly linear, so this is not a limit on expressiveness. It simply has no reliable route to find the well-regularized solution, whereas Ridge computes it directly.

**5. We report the weights from the stopping epoch, not the best epoch.** Soft spot 1 above. Worth about 0.08 to 0.11 of R² on each score — roughly half of Arm B's gap to Arm A.

**Say it plainly and without apology.** With 20 participants, 956 sessions, and a strongest raw association of r² = 0.0085, the *expected* ordering is: simplest model closest to the mean-predictor baseline, most flexible model furthest below it. That is exactly the ordering we observe — hand-crafted features (−0.009 to −0.065) then Chronos + Ridge (−0.012 to −0.089) then Chronos + MLP head (−0.168 to −0.281). This is a textbook result of applying a high-capacity model to a small sample with a near-absent relationship, not an anomaly and not a bug. And the machinery demonstrably works: the same embeddings and folds predict glucose **variability** at R² = 0.462 (`01_FACT_SHEET.md`, section I).

**Say it in your own words:** *"Arm B lost for a reason we could have predicted: the strongest raw glucose-to-score relationship in the whole dataset explains under 1% of the variation, and we pointed 132,609 parameters at 550 sessions with the stopping decision resting on 4 children. Ridge has 513 parameters and a tuned closed-form penalty, so it lands closer to the baseline. Roughly half of Arm B's remaining gap is a fixable bug — we report the weights from the epoch we stopped at rather than the best epoch."*

<details>
<summary><b>Drill</b> — 4 questions</summary>

**Q: "Your neural network was worse than linear regression. Doesn't that mean you implemented it wrong?"**
A: It is the expected ordering, not a symptom. When the true relationship is nearly flat, the model with more freedom does worse on held-out data because it fits more noise — that is what 217 parameters per training example produces. Three separate checks say the implementation is fine: the smoke test shows the training loss decreasing, the trainable-parameter count is exactly 132,609 as designed, and the same embeddings under the same folds predict glucose variability at R² = 0.462. There is one real bug and I know it: we report the weights from the epoch training stopped at rather than the best epoch, worth about 0.1 of R².

**Q: What is the single most promising change to Arm B?**
A: Shrink it. `head_hidden_dim` from 256 to 16 takes the head from 132,609 parameters to 9,249, a 14× cut, and it is a one-argument change. Adding the `ModelCheckpoint` fix would also recover about 0.1 of R². Neither is a substitute for the real fix, which is on the data side: a uniform pre-test window instead of the current 3-to-288-reading range.

**Q: Would more data fix Arm B?**
A: More *participants* would help the variance side — 20 children is a very small basis for grouped cross-validation, and the validation split resting on 4 children is a direct consequence. But it cannot create a relationship: r² = 0.0085 means a genuine effect of that size would need a much larger study to detect at all. The advisor has said more data is not guaranteed.

**Q: Is Arm B worth keeping in the write-up?**
A: Yes, for two reasons. It was the assigned task — adapt Ben's classifier to single-channel CGM regression — and it is a real comparison: it shows that the flat result is not an artefact of choosing a weak model, because the most flexible model available did worse, in the direction the theory predicts.

</details>

---

## 22 · Freezing versus fine-tuning versus training from scratch

| Approach | What is trainable | Data needed | Cost | Risk | Our situation |
|---|---|---|---|---|---|
| **Frozen encoder + trained head** (ours) | 132,609 (0.28% of the total) | small | minutes on CPU; embeddings computed once and cached | can only use what the frozen features already encode | **what we did** |
| **Fine-tuning** | all 47,850,625 | thousands of subjects | hours on GPU; gradients and optimizer state for 47.7M parameters | severe overfitting on 20 participants; you can destroy the pretrained representation in a few hundred steps | not attempted; the code path exists (`ChronosTorchEmbedder` accepts raw windows) |
| **Parameter-efficient fine-tuning** (LoRA, adapters) | ~0.1–1% of the encoder, injected as small extra matrices | moderate | close to frozen cost | milder than full fine-tuning but still real | flagged as a future option in `ben_adapter/README.md`, conditional on the frozen arm showing promise |
| **From scratch** | a full model, tens of millions | very large | large | no pretrained knowledge at all | never sensible at N=20 |

**Why we froze.** Compare the ratios. Frozen: 132,609 trainable parameters on ~550 training sessions from 12 participants — already 217× more parameters than examples. Full fine-tuning: 47,850,625 trainable parameters on the same ~550 sessions — about **87,000 parameters per training example**. There is no regularizer that makes that ratio informative. Freezing is not a shortcut here; it is the only defensible choice at this sample size.

**The other argument for freezing, which is about the science.** A frozen encoder makes the experiment interpretable. The question we are answering is *"do Chronos's pretrained representations of a glucose window contain information about cognitive performance?"* With the encoder frozen, the answer is clean: the features are fixed, and only a small head chooses how to read them. Fine-tune and the question becomes *"can 47.7 million parameters be bent to fit 550 rows?"*, whose answer is yes and is uninformative.

**Say it in your own words:** *"We froze Chronos and trained only a 132,609-parameter head — 0.28% of the total — because fine-tuning all 47.7 million parameters on 550 sessions from 12 children would be about 87,000 parameters per example, which no amount of regularization rescues. Freezing also keeps the experiment interpretable: the features are fixed, so the result is about what Chronos already encodes."*

<details>
<summary><b>Drill</b> — 3 questions</summary>

**Q: "Did you try fine-tuning Chronos?"**
A: No, deliberately. With 20 participants and ~550 training sessions per fold, unfreezing 47.7 million parameters is about 87,000 parameters per training example, and the frozen arm at 217 already overfits. The code path exists — `ChronosTorchEmbedder` takes raw windows and the regressor's forward pass runs end to end — so LoRA fine-tuning is a documented next step, conditional on the frozen arm showing something.

**Q: How does freezing save compute, specifically?**
A: Three ways. No gradients or optimizer state for 47.7M parameters (Adam would need two extra floats each). No stored activations through the encoder for the backward pass. And because the encoder's output never changes, the embeddings are computed once and cached to a `.npy` file, so each of the 169 total training epochs is a matrix multiply on a `(64, 512)` block instead of a transformer forward pass.

**Q: Where is the 0.28% from?**
A: 132,609 trainable divided by 47,850,625 total (47,718,016 frozen Chronos-bolt-small parameters plus the head) = 0.277%.

</details>

---
---

## The six hardest questions on this material

**1. "What is early stopping?"**

> It is a rule that halts training when a model stops improving on data it is not training on. In our code it is `EarlyStopping(monitor="val/loss", patience=8, mode="min")`. The monitored number is mean squared error on the z-scored cognitive score over the validation sessions — 4 children held out of the 16 training children, by participant, so no child appears in both. It is computed once at the end of each epoch. Lightning keeps the best value seen so far and a counter. Any strict decrease is an improvement and resets the counter to zero, because `min_delta` defaults to 0. Otherwise the counter ticks up, and when it reaches 8 training stops. Concretely, on Symbols fold 4 the best validation loss was 1.2813 at epoch 9, epochs 10 through 17 all failed to beat it, and training stopped at epoch 17 after 18 epochs and 162 optimizer steps out of a permitted 900. It is a regularizer, because stopping after roughly 100 updates from a small random initialization keeps the weights in a small neighbourhood of that initialization, so the model never reaches the wilder functions its 132,609 parameters could describe. And there is a flaw in our version I want to state myself: we set `enable_checkpointing=False` and add no `ModelCheckpoint`, so nothing is saved and `trainer.test` scores the weights in memory — the ones from the epoch we stopped at, up to 8 epochs past the best. I re-ran all 15 fold-trainings snapshotting the best weights: restoring them moves R² from −0.195 to −0.090 on Grids, −0.279 to −0.167 on Symbols, −0.136 to −0.058 on Prices. So Arm B currently looks worse than it should, and the fix is three lines. It does not change the conclusion, because every value stays at or below zero.

**2. "How does the epoch work?"**

> An epoch is one complete pass through the training sessions. A step, or iteration, is one batch: one forward pass, one backward pass, one weight update. For us an epoch is 9 steps, and I can derive that. 956 sessions from 20 participants. `GroupKFold(5)` on participant id holds out 4 participants, about 191 sessions, as the test fold, leaving about 765 sessions from 16 participants. Then `GroupShuffleSplit(test_size=0.2)` carves off a validation set — and it splits by participant, taking `ceil(0.2 × 16) = 4` children, so validation is 173 to 213 sessions and training is 538 to 576. Batch size is 64 and the DataLoader keeps the remainder, so `ceil(550/64) = 9` batches. One epoch is 9 optimizer steps: eight batches of 64 and one of about 38. With `max_epochs=100` that is at most 900 steps, and no fold got past epoch 18. Validation runs once per epoch, which is why patience 8 means 8 epochs here. If you scaled the 20% by rows instead of participants you would get 612 training sessions and 10 batches — that is the estimate in our fact sheet, and the difference is that the code splits by child, not by row.

**3. "Walk me through what happens to one batch of data."**

> Sixty-four sessions arrive as a 64-by-512 block of frozen Chronos embeddings, computed once upstream and cached. LayerNorm normalizes each row across its own 512 values and applies a learned scale and shift — 1,024 parameters — and the shape stays 64-by-512. `Linear(512→256)` computes 256 weighted sums per session — 131,328 parameters — giving 64-by-256. GELU bends each of those numbers, no parameters. Dropout zeroes about 10% of them and scales the survivors by 1/0.9, but only during training. `Linear(256→1)` collapses each session's 256 numbers into one — 257 parameters — giving 64-by-1, and `.squeeze(-1)` makes it 64 predictions. Those are compared against the 64 true scores, z-scored using the training split's mean and standard deviation, with mean squared error. `loss.backward()` runs the chain rule backwards to produce a gradient for each of the 132,609 parameters, `optimizer.step()` applies AdamW's update, and `optimizer.zero_grad()` clears the gradients for the next batch. That is one of 9 steps in an epoch.

**4. "You say the network is regularized. By what, and how much?"**

> Three things, and I would rank them honestly. Early stopping does most of the work: training ran 9 to 18 epochs, which is 81 to 162 updates from a small random initialization, so the weights barely moved from their starting point and the effective capacity used was far below the nominal 132,609. Dropout 0.1 is second: it blanks about 26 of the 256 hidden values on every batch, so no single unit can be relied on, and it is off at evaluation. Weight decay 0.01 inside AdamW is third and nearly inert — each step multiplies weights by 0.99999, and over a typical 108-step fold that is a total contraction of 0.1%. So it is wrong to say Arm B has no regularization, and it is also wrong to say it has good regularization. Compare Arm A, which tunes Ridge's alpha over five values with a closed-form solution and a 3-fold grouped inner search. That is the real difference: Arm B's three restraints interact and none of them is tuned.

**5. "Your architecture can represent Ridge's answer. Why couldn't it find it?"**

> Almost right, and worth being precise about. Ridge's answer is a straight-line function of the 512 inputs. My head is not exactly a straight line even in principle, because LayerNorm divides each row by that row's own standard deviation, so it discards each session's overall magnitude before the Linear layers ever see it. Setting that aside, the hidden layer can be made almost linear, so expressiveness is not the obstacle. The obstacle is the search. Ridge solves a convex problem in closed form, so for any alpha it lands exactly on the best regularized solution and I can scan the whole regularization spectrum with a five-value grid. My head reaches its solution by about 100 stochastic gradient steps from a random start, with restraint coming from an untuned dropout rate, a nearly inert weight decay, and a stopping rule watching 4 children whose best epoch varied from 0 to 9 across 15 runs. It can express the good solution; it has no reliable path to it.

**6. "Why should I believe your deep learning code is correct, given every number is negative?"**

> Four pieces of evidence. First, the trainable parameter count is exactly 132,609 — I can enumerate it layer by layer, and it matches what the architecture should give, which confirms the encoder is genuinely frozen and only the head trains. Second, the training loss falls reliably in every fold, so the gradients flow and the optimizer works; the smoke test in `ben_adapter/smoke_test.py` asserts this. Third, the same embeddings and the same grouped folds predict glucose variability at R² = 0.462, so the pipeline recovers real information when the target contains it — the flat cognition result is about the target, not the machinery. Fourth, there is an assertion at `train.py:95` that crashes the run if any participant appears in more than one of train, validation and test, so the negative numbers are not an artefact of a broken split. And the direction of the result is what the theory predicts: 132,609 parameters on 550 sessions with a strongest raw r² of 0.0085 should land further below the baseline than a 513-parameter tuned linear model, and that is exactly the ordering we see across all three scores.

---

## Every term introduced, with a one-line plain definition

| Term | Plain definition |
|---|---|
| **Activation function** | A bend applied to each number after a Linear layer; without one, stacked Linear layers collapse into one |
| **AdamW** | Our optimizer: Adam plus a weight shrink applied separately from the gradient |
| **Adam** | An optimizer that gives each parameter its own step size from running averages of its gradient and squared gradient |
| **Backpropagation** | The algorithm that computes every parameter's gradient in one backward sweep, using the chain rule |
| **Backward pass** | The sweep from loss back to inputs that fills in each parameter's gradient |
| **Batch / mini-batch** | The group of sessions processed before one weight update; ours is 64 |
| **BatchNorm** | Normalization across the samples in a batch, per feature; not used here |
| **Bias** | The extra number added to a neuron's weighted sum, letting its output shift freely |
| **Chain rule** | `dL/dw = (dL/dy)(dy/dw)` — the reason a local rule per layer computes global gradients |
| **Cross-entropy** | The loss for classification; penalises probability assigned to the wrong class |
| **Decoupled weight decay** | Applying the L2 shrink as its own step rather than inside the gradient, so Adam's rescaling does not distort it |
| **Dropout** | Randomly zeroing a fraction of activations during training and scaling the rest; ours 0.1; off at evaluation |
| **Early stopping** | Halting training when the validation metric stops improving; ours monitors `val/loss` with patience 8 |
| **Effective capacity** | How much freedom the model actually uses, as opposed to its parameter count |
| **Epoch** | One complete pass through the training sessions; ours is 9 optimizer steps |
| **Forward pass** | One run of a batch from input to prediction |
| **Freezing** | Marking parameters as not trainable so they never change; our whole Chronos encoder |
| **GELU** | `x·Φ(x)`; a smooth ReLU that lets slightly-negative values leak through |
| **Gradient** | The list of derivatives of the loss with respect to every parameter — direction and steepness |
| **Gradient descent** | `w ← w − lr × gradient`; the basic learning rule |
| **Hidden layer** | A layer whose output is not the prediction; ours has 256 units |
| **Identity output** | No activation on the final layer, so the prediction can be any real number |
| **Initialization** | The random starting values of the weights; `nn.Linear` uses `±1/√in` |
| **Iteration** | Synonym for step: one batch, one weight update |
| **LayerNorm** | Normalizing one sample across its own features, plus a learned scale and shift; cannot leak dataset statistics |
| **Learning rate** | The step size; ours is 1e-3, PyTorch's AdamW default |
| **Lightning** | The library that writes the training loop for us, given `training_step`, `validation_step` and `configure_optimizers` |
| **Linear layer** | `y = Wx + b`; `Linear(in→out)` has `in×out + out` parameters |
| **Loss** | The single number training minimizes; ours is MSE on the z-scored score |
| **Metric** | A number reported to humans; ours are R², RMSE, MAE in the score's real units |
| **min_delta** | How much better a monitored value must be to count as an improvement; defaults to 0 |
| **MLP** | Multi-layer perceptron — alternating Linear layers and activations, fully connected |
| **Momentum** | Stepping along a running average of past gradients rather than the latest one |
| **MSE** | Mean squared error: average of squared prediction errors |
| **Neuron / unit** | One weighted sum plus a bias, optionally followed by an activation |
| **Optimizer** | The rule that turns gradients into weight updates; ours is AdamW |
| **Optimizer state** | Extra numbers an optimizer keeps per parameter; Adam keeps two |
| **Overfitting** | Getting better on training data while getting worse on unseen data |
| **Parameter** | A number learned from data; we have 132,609 trainable ones |
| **Patience** | How many consecutive non-improving validation checks are tolerated; ours 8, so 8 epochs |
| **PassthroughEmbedder** | A no-op stand-in for the encoder that returns precomputed cached embeddings |
| **ReLU** | `max(0, x)`; keeps positives, zeroes negatives |
| **Sanity check** | Lightning's 2 validation batches run before training; does not count toward patience |
| **Sigmoid** | Squashes any number into 0…1; used for probabilities, not for our output |
| **Softmax** | Turns raw class scores into probabilities summing to 1; needed by Ben's classifier, not by us |
| **Squeeze** | Dropping a size-1 axis; turns our `(64,1)` output into `(64,)` so it matches the labels |
| **Step** | One batch: forward, backward, update |
| **StandardScaler** | Per-feature normalization across the training fold's rows; Arm A's preprocessing |
| **Stochastic gradient descent** | Gradient descent on batches rather than the full dataset |
| **tanh** | Squashes into −1…1, centred on zero |
| **`torch.no_grad()`** | Turns off graph building and gradient storage; wraps our frozen encoder |
| **Validation set** | Held-out data used to make choices during training; ours is 4 participants per fold |
| **Weight** | A multiplier a neuron applies to one input |
| **Weight decay** | Shrinking every weight slightly each step; ours 0.01, roughly 0.1% total over a fold |
| **z-scoring** | Subtract the mean, divide by the standard deviation; applied to our target with train-split statistics |

---

## Twelve-sentence chapter summary

1. A neuron computes `w·x + b` — one weighted sum plus a bias — and `nn.Linear(512, 256)` is 256 of those side by side, so a neuron and a linear-regression model are the same object.
2. An activation function is mandatory because `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`: without it, our 132,609 parameters would describe the same straight line Ridge solves exactly and instantly.
3. Ours is GELU, a smooth ReLU that lets slightly-negative values leak through, and the output layer has no activation at all because a cognitive score is a number on an open scale — a sigmoid could never predict 60 on Prices.
4. Our head is `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`, shapes `(B,512) → (B,512) → (B,256) → (B,256) → (B,256) → (B,1) → (B,)`, and 132,609 = 1,024 + 131,328 + 257 with 98.8% of it in that first weight matrix.
5. We minimize mean squared error on the z-scored score because it is smooth and its gradient is proportional to the mistake, and we report R², RMSE and MAE in the score's own units — the loss steers, the metric informs.
6. A gradient says which way and how steeply the loss moves if a weight is nudged; backpropagation gets all 132,609 of them in one backward sweep by multiplying local derivatives, which is the chain rule, and `torch.no_grad()` disables that bookkeeping entirely for the frozen encoder.
7. Gradient descent is `w ← w − lr × gradient`: on `L = w²` from `w = 1` with `lr = 0.1` the weight goes 1 → 0.8 → 0.64 → 0.512, while `lr = 1.1` sends it to −1.2 → 1.44 → −1.73 and the loss grows.
8. An epoch is one pass over the training sessions and a step is one batch — for us `ceil(550/64) = 9` steps per epoch, derived from 956 sessions, 4 test participants, 4 validation participants and 12 training participants — so 100 epochs is at most 900 steps and no fold got past 18.
9. The loop is: for each epoch, for each batch, forward, compute loss, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`; then one validation pass with dropout off; then the early-stopping check — and Lightning writes all of it except `training_step`, `validation_step` and `configure_optimizers`.
10. Early stopping monitors `val/loss` once per epoch, resets its counter on any strict improvement because `min_delta` is 0, and stops after 8 consecutive failures, which on Symbols fold 4 meant best at epoch 9 and a stop at epoch 17 — and because we save no checkpoint we report the epoch-17 weights, which costs about 0.1 of R² on every score.
11. Arm B carries three untuned regularizers — early stopping, dropout 0.1, and a weight decay that contracts weights by only 0.1% over a fold — while Arm A tunes a closed-form L2 penalty over five values, so the honest difference between the arms is the quality of the regularization, not its presence.
12. Arm B is our least accurate arm at −0.218, −0.281 and −0.168, and this is the expected outcome rather than a surprise: 132,609 parameters on ~550 training sessions, a stopping decision resting on 4 children, and a strongest raw glucose-to-score association of r² = 0.0085 — no model manufactures a signal that is not in the data.

---

*Cross-references: `01_FACT_SHEET.md` (every number with its derivation, including the head's parameter table and the epoch arithmetic), `02_ML_FROM_ZERO.md` (features and targets, overfitting, regularization as one idea with many names), `03_PREPROCESSING.md` (how a glucose window becomes 512 numbers), `07_TRANSFORMERS.md` (what the frozen Chronos encoder computes internally), `09_THE_TWO_ARMS.md` (Arm A and Arm B side by side), `13_PROBLEM_SETS.md` (exercises).*

*Provenance. Architecture and settings read directly from `cgm_tsfm/ben_adapter/model.py` (lines 141-157), `lightning_module.py` (lines 29-90), `train.py` (lines 37-109) and `datamodule.py` (lines 37-71); early-stopping semantics read from `lightning/pytorch/callbacks/early_stopping.py` and the best-weights behaviour from `lightning/pytorch/trainer/connectors/checkpoint_connector.py:138-140`. The 132,609 parameter count, the 47,718,016 Chronos-bolt-small count, every activation and LayerNorm and dropout value, the chain-rule example, the gradient-descent tables and the zero-initialization collapse were all computed in this repository's environment (`/data_1_8TB_ssd/brian_workspace/envs/cgm`) on 2026-07-30. The per-fold split sizes, the 9 steps per epoch, the per-epoch validation-loss traces, the epochs-completed table, and the best-weights-versus-final-weights comparison come from a re-run of all 15 Arm B fold-trainings on the cached `chronos-bolt-small` embeddings on 2026-07-30. Headline R² values (−0.218 / −0.281 / −0.168) are from `results/headtohead_real.md`, generated 2026-07-07; the re-run's means (−0.195 / −0.279 / −0.136) differ within the fold-to-fold spread because weight initialization and batch shuffling depend on the library versions in the random-number stream.*
