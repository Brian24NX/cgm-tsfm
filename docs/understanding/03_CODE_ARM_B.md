# Arm B — The Trainable Neural Network (`ben_adapter/`)

*Read `01_CONCEPTS.md` and `02_CODE_ARM_A.md` first. Arm B does the same job as Arm A's `regression.py` — turn the 512-number embedding into a predicted score — but with a **small trainable neural network** instead of a classical sklearn model. It's built with **PyTorch + Lightning**, and it's the literal task "**adapt Ben's code**."*

Files: `model.py` (the network), `lightning_module.py` (the training logic), `datamodule.py` (feeding data), `train.py` (the grouped-CV loop), `smoke_test.py` (a self-check).

---

## First: what is this arm, and why does it exist?

Ben is a labmate whose code puts a foundation model + a trainable network head on top for *his* project (2 sensors → predict a category). The advisor asked Brian to **adapt Ben's code** to our case: **1 channel (glucose) → predict a number**. That adaptation is `ben_adapter/`.

The design is **"frozen encoder + trainable head"**:
- The **encoder** (Chronos) stays **frozen** — we never change it, we only read out the 512-number embedding (same as Arm A).
- On top we add a small **trainable head** — a few neural-network layers that *learn*, on our data, how to map `512 numbers → 1 score`.

So Arm A and Arm B differ only in the "predictor" box: Arm A = a fixed sklearn formula (Ridge/SVR); Arm B = a little network trained by gradient descent.

**Result reminder:** Arm B did *worse* than Arm A on the real data (it overfit the tiny 20-subject dataset). That's a legitimate finding — more flexibility with no signal + little data = worse generalization. Explaining *why* that happens is a great thing to say to your professor.

---

## PyTorch vocabulary you need (5 terms)

Arm B uses PyTorch, so here are the words:
- **`nn.Module`** — the PyTorch base class for "a piece of a neural network." You subclass it and define a `forward()` method that says how inputs flow to outputs. Everything (the head, the whole model) is an `nn.Module`.
- **`forward(x)`** — the function that runs inputs through the network to produce outputs. Calling `model(x)` runs `forward`.
- **Layers** — the building blocks stacked inside a network:
  - **`nn.Linear(in, out)`** — a "fully connected" layer: `output = W·input + b` (a matrix multiply + bias). The `W` and `b` are the **weights** that get trained.
  - **`nn.GELU`** — an **activation function** (a smooth non-linear bend). Stacking Linear + activation is what lets a network learn *non-straight* relationships. Without an activation, stacking Linear layers would collapse to one straight line.
  - **`nn.LayerNorm`** — normalizes the numbers flowing through (keeps them well-scaled), which stabilizes training.
  - **`nn.Dropout(p)`** — during training, randomly zeroes a fraction `p` of the values. Sounds odd, but it's a **regularizer**: it stops the network from leaning too hard on any one feature, reducing overfitting.
  - A stack like `Linear → activation → Linear` is called an **MLP** (multi-layer perceptron) — the simplest kind of neural network.
- **`requires_grad`** — a flag on each weight: `True` = "train this," `False` = "freeze this." We set the encoder's weights to `False` (frozen) and leave the head's `True`.
- **Optimizer / loss / epoch** — training = repeatedly compute the **loss** (how wrong we are, here **MSE**), and have the **optimizer** (**AdamW**) nudge the trainable weights to reduce it; one full pass over the data is an **epoch**.

---

## `model.py` — the network itself

### The three embedders (encoder options)
`model.py` defines three interchangeable "encoders," all `nn.Module`s with an `embedding_dim` and a `forward`:
- **`ChronosTorchEmbedder`** — the real one: loads `BaseChronosPipeline`, and its `forward` does the same recipe as `encoders.py` (`embed` → mean-pool → `(B, 512)`), wrapped in `@torch.no_grad()` (frozen — no training of Chronos).
- **`MockTorchEmbedder`** — a no-download stand-in (summary stats), for offline tests.
- **`PassthroughEmbedder`** — the clever one for **speed**. Because the encoder is frozen, its output never changes during training — so we compute the embeddings **once** (cached), and then this "encoder" just **returns the already-computed embedding unchanged**:
  ```python
  def forward(self, embeddings, mask=None):
      return embeddings
  ```
  This means Arm B trains only the head on precomputed embeddings — **much faster** than re-running Chronos every epoch, and identical in result. (This is what `train.py` actually uses.)

`_valid_series(...)` and `collate_windows(...)` are plumbing: split/pad variable-length windows into a padded `(B, T)` batch + a `mask` marking which entries are real (needed only when feeding *raw glucose*; the passthrough path uses precomputed embeddings and ignores the mask).

### `class FoundationModelRegressor` — frozen encoder + trainable head
```python
self.encoder = encoder                      # frozen
for p in self.encoder.parameters():
    p.requires_grad = False                 # <- freeze the encoder
d = encoder.embedding_dim                   # 512 (or 16 for the mock)
self.head = nn.Sequential(
    nn.LayerNorm(d),
    nn.Linear(d, head_hidden_dim),          # 512 → 256
    nn.GELU(),
    nn.Dropout(head_dropout),               # regularize
    nn.Linear(head_hidden_dim, num_targets),# 256 → 1 (one score)
)
def forward(self, glucose, glucose_mask=None):
    emb = self.encoder(glucose, glucose_mask)   # (B, 512), no grad
    return self.head(emb)                        # (B, 1)
```
- The `for p ... requires_grad = False` loop is **how "frozen" is enforced** — those weights are excluded from training.
- The **head** is the MLP: `LayerNorm → Linear(512→256) → GELU → Dropout → Linear(256→1)`. This is the *same shape as Ben's classifier head*, with the output width changed from "number of classes" to **`num_targets=1`** (single-target regression, per the advisor). That one change ("num_classes → num_targets=1") is the essence of "adapt Ben's classifier into a regressor."
- `forward`: run the frozen encoder → get `(B, 512)` → run the head → `(B, 1)` prediction.

---

## `lightning_module.py` — the training logic (`CGMRegressionModule`)

**Lightning** is a wrapper that removes the boilerplate of PyTorch training loops. You fill in a few named methods and Lightning runs the loop, validation, early stopping, GPU handling, etc. `CGMRegressionModule` is our filled-in version.

### `__init__` — set up loss, metrics, normalization
```python
self.register_buffer("target_mean", torch.tensor(float(target_mean)))
self.register_buffer("target_std",  torch.tensor(max(float(target_std), 1e-6)))
self.loss_fn = nn.MSELoss()
base = {"mse": ..., "rmse": ..., "mae": ..., "r2": R2Score()}
```
- `target_mean`/`target_std` are the **training-set** mean and standard deviation of the score. We **z-score the target** during training (subtract mean, divide by std) so the loss is on a tidy O(1) scale for *any* score. This **replaces Ben's `/365.0` hack** (his was gestational-age-specific; ours is general). `register_buffer` = "store this number with the model, but don't train it."
- `MSELoss` = mean squared error, the regression loss. `torchmetrics` gives ready-made RMSE/MAE/R² that accumulate correctly across batches.

### `_shared_step(batch, stage)` — one training/val/test step
```python
labels = batch["label"].float()
valid = ~torch.isnan(labels)                     # skip sessions missing this score
glucose, labels = glucose[valid], labels[valid]
logits = self(glucose, glucose_mask).squeeze(-1) # model prediction (B,)
labels_norm = (labels - self.target_mean) / self.target_std   # z-score the target
loss = self.loss_fn(logits, labels_norm)          # MSE on the normalized scale
preds = logits * self.target_std + self.target_mean  # de-normalize for reporting
metrics.update(preds, labels)                     # RMSE/R² on real-unit predictions
```
- Filter out sessions whose score is missing (`~isnan`).
- Predict, then compute **loss on the normalized target** (stable training), but **de-normalize predictions back to real units** before computing the reported metrics (so R²/RMSE are on the score's real scale). `.squeeze(-1)` turns the `(B,1)` output into `(B,)` to match the labels.
- `training_step`/`validation_step`/`test_step` just call `_shared_step` with the right stage — Lightning calls these automatically during `trainer.fit()`/`.test()`.

### `configure_optimizers`
```python
params = [p for p in self.model.parameters() if p.requires_grad]   # head only
return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
```
Crucially, it optimizes **only the trainable parameters** (`requires_grad` True) — i.e. the **head**, not the frozen Chronos encoder. **AdamW** is the optimizer (a good default); `weight_decay` is another regularizer (keeps weights small, like Ridge's penalty).

---

## `datamodule.py` — feeding one CV fold in batches

**Lightning** wants data delivered through a `LightningDataModule`. `CGMEmbeddingDataModule` holds one fold's **precomputed embeddings** split into train/val/test and serves them as `{"glucose": embedding, "label": score}` batches (the keys `_shared_step` reads — "glucose" here is actually the embedding, via the `PassthroughEmbedder`).

- A **`DataLoader`** batches and (for training) shuffles the data. `batch_size=64` = 64 sessions per step.
- `target_mean`/`target_std` are computed from the **training split only** (`self._tr.y`) — so the normalization never leaks val/test info into training. This is the same leakage-discipline as Arm A's scaler.

---

## `train.py` — the grouped-CV training loop (`run_arm_b`)

**Job:** run Arm B under the **same** 5-fold grouped CV as Arm A, so their R²s are directly comparable, and return results in the **same shape** as `run_arm_a` (so they slot into the head-to-head table).

The structure mirrors `nested_group_cv`, but instead of `GridSearchCV` it trains a network per fold with early stopping:
```python
outer = GroupKFold(n_splits=outer_splits)                 # same folds as Arm A
for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y, g)):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_local, va_local = next(gss.split(X[train_idx], ...))   # a grouped val split for early stopping
    assert ... isdisjoint ...                              # no subject in train/val/test at once
    res.folds.append(_train_one_fold(X[tr_abs], y[tr_abs], X[va_abs], y[va_abs],
                                     X[test_idx], y[test_idx], ...))
```
- **Outer loop:** `GroupKFold(5)` on `subid` — identical folds to Arm A (leakage-free).
- Inside each outer-train set, a grouped **`GroupShuffleSplit`** carves off a **validation** set (again subject-disjoint) used for **early stopping** — Arm A tuned a knob with an inner grid search; Arm B instead uses a fixed head + early stopping as its regularizer. The `assert` guards that no subject is shared across train/val/test.

`_train_one_fold(...)` builds the model + `CGMRegressionModule` + `CGMEmbeddingDataModule`, then:
```python
trainer = pl.Trainer(max_epochs=..., accelerator=accelerator, devices=1,
                     callbacks=[EarlyStopping(monitor="val/loss", patience=patience, mode="min")],
                     logger=False, enable_checkpointing=False, enable_progress_bar=False)
trainer.fit(lm, dm)                       # train the head (early-stopped on val loss)
res = trainer.test(lm, dm, verbose=False)[0]   # evaluate on the held-out test fold
return FoldResult(rmse=res["test/rmse"], mae=res["test/mae"], r2=res["test/r2"])
```
- **`EarlyStopping(monitor="val/loss", patience=8)`** — stop training when the validation loss stops improving for 8 epochs, to avoid overfitting.
- `accelerator=` is `"cuda"` for GPU. `trainer.fit` trains only the head; `trainer.test` measures on the untouched test fold.

`run_arm_b` collects per-fold RMSE/MAE/R², averages them, and returns the same `{target: {n, n_subjects, rows}}` dict shape as Arm A — which is why `run_headtohead --with-arm-b` can drop it into the same comparison table.

---

## `smoke_test.py` — the "is it wired correctly?" self-check

A tiny offline test (no data, no download): make a mock model, run a forward pass, then train the head for 30 steps on a trivial batch and **assert the loss went down**:
```python
assert losses[-1] < losses[0], "head did not learn on a trivial batch"
```
If the loss decreases, the network + loss + optimizer are correctly connected. It's a **plumbing check**, not science — but a good habit (and a nice thing to show: "I have a self-test that verifies the model can learn").

---

## Arm A vs Arm B — the one-paragraph comparison for your professor

> Both arms take the same frozen 512-d Chronos embedding and predict the score. **Arm A** (sklearn) fits a *regularized linear/kernel model* in one shot — simple, robust, and it tunes its one knob with nested grouped cross-validation. **Arm B** (PyTorch/Lightning) trains a *small MLP head* by gradient descent with early stopping, under the *same* grouped folds. On our 20-subject data with no real signal, Arm A ≈ the mean baseline and **Arm B overfits and does slightly worse** — which is the expected behavior of a more flexible model on small, signal-poor data, and is exactly why simple regularized models are the right default here.

**Next:** `04_WALKTHROUGH_AND_QA.md` follows one glucose window through all of this, end to end, and lists questions your professor may ask.
