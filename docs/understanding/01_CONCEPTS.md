# Concepts From Scratch — Everything You Need Before the Code

*Assumes no machine-learning background. Read top to bottom; each idea builds on the last. Every technical term is in **bold** the first time it's defined.*

---

## Part 1 — What are we even doing? (Machine learning, in plain words)

**Machine learning (ML)** is: *writing a program that finds a pattern in example data, so it can make predictions on new data it hasn't seen.* You don't write the rule by hand — you show the computer many examples and it fits a rule.

Our specific flavor is **supervised learning**: we have examples where we know both the **input** (the glucose readings before a test) and the **answer** (the score the child got). We want a program that learns *input → answer*, so that for a **new** child it can guess the answer from the input.

Two words you'll hear constantly:
- **Target** (a.k.a. label, outcome, `y`): the thing we're trying to predict — here, a cognitive test score (a number).
- **Features** (a.k.a. inputs, `X`): the numbers we feed the model to predict the target — here, something derived from the glucose curve.

Because our target is a **continuous number** (a score like 1.83, not a category like "cat/dog"), this is a **regression** problem. (If we were predicting a yes/no category, that would be **classification**. We do regression.)

> **The whole project in one line:** learn a function `f(glucose window) → cognitive score`, then check honestly whether `f` is any good.

---

## Part 2 — The hard part: turning a glucose curve into "features"

A model like linear regression eats a **fixed-length list of numbers** and outputs one number. But our input is a **glucose curve** — a sequence like `[140, 138, 135, 150, …]` that is a *different length* for every test (some have 15 minutes of readings, some have hours). You can't hand a variable-length curve directly to a simple model. So we must convert each curve into a **fixed-size list of numbers**. There are two philosophies for doing this — and comparing them is the scientific point of the project.

### Philosophy A — Feature engineering (the OLD way)
A human expert decides which summary numbers matter and writes formulas to compute them: average glucose, standard deviation (how bumpy), % of time below 70 (low), the slope, etc. A labmate (Mack) built **43** such **hand-crafted features** (that's what `handcrafted.py` is). The model then only ever sees those 43 numbers.
- **Pro:** interpretable, simple.
- **Con:** it can only capture what a human thought to measure. If the real signal is in *the exact shape or timing* of the curve, and nobody wrote a formula for that, it's lost.

### Philosophy B — Raw-data / representation learning (OUR way)
Instead of hand-picking, we feed the **raw curve** to a big pre-trained AI model and let *it* produce the summary numbers. That AI is a **foundation model** (next section). Its output — a fixed-size list of numbers that captures the curve — is called an **embedding**.
- **Pro:** can capture shape/timing patterns nobody hand-coded.
- **Con:** the numbers aren't human-readable, and there are a lot of them (512), which needs care.

The project runs **both** philosophies through the *exact same* evaluation, so any difference is due to the representation, not the test. (Per the advisor's latest guidance we now focus on Philosophy B on its own; the hand-crafted features stay only as a sanity-check reference.)

---

## Part 3 — Foundation models, TSFM, and Chronos

A **foundation model** is a very large neural network that has been **pre-trained** on a huge amount of data, so it has already learned general patterns and can be **reused** for many tasks without training your own from scratch. ChatGPT is a foundation model for *language* — it read enormous amounts of text and learned the patterns of language.

A **TSFM = Time-Series Foundation Model** is the same idea but for **time series** — data that is *numbers changing over time* (stock prices, weather, heart rate… or glucose). A TSFM has been pre-trained on millions of time series, so it already "knows" the general shapes that sequences of numbers take (trends, spikes, cycles, noise). You can point it at *your* time series (glucose) and it produces something useful with **no extra training**.

**Chronos** is the specific TSFM we use. It's made by Amazon, free to download, and its trick is clever: it treats a time series *like a sentence in a language*. It chops the numbers up, turns each chunk into a "token" (like a word), and runs a **transformer** (the same kind of network behind chatbots) over those tokens. So Chronos is, almost literally, *"a language model whose language is sequences of numbers over time."*

- **Chronos-Bolt** (the version we use, `amazon/chronos-bolt-small`): a fast variant that reads the series in **patches** (chunks of 16 readings). Much faster; runs fine on CPU and flies on GPU.
- **Chronos-T5**: an older/slower variant that reads **one token per reading**. (The research paper the advisor cited used this one — so it's worth knowing both.)

We use Chronos in an unusual way: **we do NOT ask it to forecast the future.** We feed it the glucose curve and grab its **internal understanding** of the curve — the **embedding** — and use *that* as our features. (Details + exact tensor shapes: `docs/06_CHRONOS_INPUT_FORMAT.md`.)

### Embedding and encoder — the 512 numbers
- **Embedding**: a fixed-size list of numbers that represents something complex. For us, feeding one glucose window into Chronos gives back **512 numbers**. Think of it as a **"fingerprint" of the curve's shape** — two similar curves get similar fingerprints. The individual numbers have no human meaning; collectively they're a rich summary a simple model can eat. (Analogy: face-recognition turns a photo into ~128 numbers describing the face; same idea.)
- **Encoder**: the thing that produces the embedding. In our code `encoders.py` is the encoder; it wraps Chronos and does the "curve → 512 numbers" step. We say the encoder is **frozen** — we never change/train Chronos itself; we only run it forward to read out embeddings.

Why exactly **512**? That's the internal "width" of the `chronos-bolt-small` model (its `d_model`). Bigger Chronos models are wider; the `-small` one is 512. It's just how many dials that model uses internally to describe a series.

---

## Part 4 — "Window": what it means

A **window** is *the slice of glucose readings we feed the model for one test*. The real data stores, for each cognitive test, the glucose readings recorded **before** that test (column `Glucose_Before_Test`). One window = one list of glucose numbers = the input for one prediction.

- Readings come every **5 minutes** (the CGM device, a Dexcom G6). So 12 readings ≈ 1 hour.
- Windows vary in length — in our real data, from **3 readings (~15 min) to 288 (~24 h)**, median ~36 (~3 h). ("Windowing" code in `data.py` can also *cap* a window to, say, the most-recent 30 readings = 2.5 h, if we want a fixed lookback.)

"One window → one embedding → one predicted score" is the unit the whole pipeline processes, repeated for every test session (956 of them in the real data).

---

## Part 5 — The tools (which library does what)

You'll see these imported at the top of the `.py` files. Know what each is for:

| Library | One-line job | Where used |
|---|---|---|
| **NumPy** (`import numpy as np`) | Fast arrays of numbers + math (mean, std, matrix ops). The bedrock everything sits on. | everywhere |
| **pandas** (`import pandas as pd`) | Tables (spreadsheets in code) — reading CSV files, columns, rows. | `data.py` |
| **SciPy** (`scipy.stats`) | Extra math/stats functions (e.g. skew, kurtosis). | `handcrafted.py` |
| **scikit-learn** (`sklearn`) | Ready-made **classical ML**: Ridge, SVR, PCA, StandardScaler, cross-validation, R². | `regression.py` (Arm A) |
| **PyTorch** (`torch`) | Build & train **neural networks**; also what Chronos itself is built in. | `encoders.py`, `ben_adapter/` |
| **Lightning** (`lightning`) | A tidy wrapper around PyTorch that handles the training loop for you. | `ben_adapter/` (Arm B) |
| **torchmetrics** | Ready-made metrics (RMSE, R²…) that work with PyTorch. | `ben_adapter/lightning_module.py` |
| **chronos** (`chronos-forecasting`) | The Chronos TSFM itself (loads the pretrained model, `.embed()`). | `encoders.py`, `ben_adapter/model.py` |

### sklearn vs PyTorch — the key distinction
- **scikit-learn** = a toolbox of **finished** classical models. You give it your features `X` and targets `y`, call **`.fit(X, y)`**, and it computes the best model in essentially one shot (solving a math equation). No training loop, no GPU needed. Great for **simple, robust** models. **This is Arm A.**
- **PyTorch** = a toolkit for **building your own** neural network and training it by **gradient descent** — repeatedly showing it data and nudging its internal numbers ("weights") to reduce error, over many passes ("epochs"). More powerful and flexible, but more code and more ways to go wrong. **This is Arm B** (and Chronos itself).

*Why use both?* Arm A deliberately wants the **simplest** predictor on top of the embedding — sklearn's one-line, well-tested models are exactly right, and its cross-validation tools are built in. Arm B is the "trainable neural network" experiment (and the specific task of adapting Ben's code), which is inherently a PyTorch job. Using PyTorch for Arm A would be more work for no benefit; using sklearn for Arm B isn't possible (it's a custom network). **Right tool per job.**

---

## Part 6 — How a model "learns", and overfitting

- **sklearn `.fit()`**: for a model like Ridge, "learning" means solving a formula that finds the straight-line weights minimizing the prediction error on the training data. One shot, deterministic.
- **PyTorch training**: start with random weights; feed a **batch** of examples; measure how wrong the predictions are with a **loss function** (we use **MSE = mean squared error**); compute which direction to nudge each weight to reduce the loss (**backpropagation**); take a small step (**gradient descent**, via an **optimizer** like **AdamW**); repeat for many **epochs** (full passes over the data).

**Overfitting** — the central danger. A model **overfits** when it *memorizes* the training examples (including their noise) instead of learning the real pattern. It then looks great on training data but fails on new data. With **512 features** and only ~950 examples from **20 kids**, overfitting is very easy — which is exactly why we use:
- **Regularization** (Ridge's penalty — Part 7),
- **PCA** (shrink 512 → ~32 numbers — Part 8),
- **early stopping** in Arm B (stop training when performance on held-out data stops improving),
- and above all **honest evaluation** (Part 9).

We literally saw overfitting in action: plain (unregularized) linear regression on the 512 embeddings scored **R² ≈ −3** (much worse than guessing) — a textbook overfit. That's *why* the code always uses regularized models.

---

## Part 7 — The specific prediction models we use (and why)

In Arm A (`regression.py`) we try several small models per target and report the best. Each is a different way to draw the "512 numbers → score" relationship:

- **`DummyRegressor(mean)` — the baseline.** The dumbest strategy: ignore the glucose entirely and always predict the *average* score. Not a real model — it's the **bar to beat**. Every real model is compared against it (that's what R² does, Part 9). If a model can't beat "always guess the average," it learned nothing.

- **Ridge regression.** A straight-line (**linear**) model: `score = w₁·e₁ + w₂·e₂ + … + w₅₁₂·e₅₁₂ + b`, i.e. a weighted sum of the 512 embedding numbers. The catch: with 512 weights and little data, plain linear regression overfits. **Ridge adds a penalty that keeps the weights small** ("regularization"), which prevents the wild overfitting. This is the workhorse — simple, robust, the right default for high-dimensional features. (The penalty strength `alpha` is a knob the code tunes automatically.)

- **SVR (Support Vector Regression).** A model that can fit **curved (non-linear)** relationships, not just straight lines (using an "RBF kernel" — think of it as flexible bendy fitting). We include it in case the true relationship isn't a straight line.

- **Linear regression (plain).** The same weighted sum as Ridge but with **no penalty**. We keep it *specifically to demonstrate the failure mode* — on 512-d embeddings it overfits catastrophically (R²≈−3), which is a teaching point: *"this is why regularization is mandatory."*

- **XGBoost** (optional, only if installed). A "gradient-boosted trees" model — a strong, popular non-linear model. Included when available for completeness.

**Why these and not something fancier?** Because with ~20 subjects and a weak/absent signal, **simpler + regularized generalizes better**; complex models just overfit (we confirmed this — the trainable neural net in Arm B did *worse*). The scientifically honest choice is the simplest models that can't fool themselves.

---

## Part 8 — Two supporting tricks: StandardScaler and PCA

- **StandardScaler (standardization / z-scoring).** Different features live on different scales (glucose ~140, a slope ~3, a fraction ~0.05). Many models work better if every feature is rescaled to "mean 0, standard deviation 1." `StandardScaler` does that. **Crucially, it's fit on the training data only** and then applied to test data — so the test data's scale never leaks into training (see leakage, Part 9).

- **PCA (Principal Component Analysis).** A math technique that **compresses** a big set of numbers into a smaller set that keeps most of the information. It turns the 512 embedding numbers into, say, the 32 most-informative combinations. Why: 512 features on ~950 samples invites overfitting and makes linear algebra unstable; PCA reduces that. In our tests PCA helped a little (it cleaned up the instability) but didn't create signal that wasn't there. Like StandardScaler, PCA is **fit per training fold only** (no leakage).

---

## Part 9 — Evaluating honestly (this is where the science lives)

Getting a good score *on data the model already saw* is meaningless — it could just be memorizing. The real question is: **how well does it predict data it has never seen?** The tools:

### Train/test split & cross-validation
Split the data into a **training** part (the model learns from it) and a **test** part (we only measure on it). To not depend on one lucky split, use **cross-validation (CV)**: split the data into *k* pieces ("folds"); each round, train on *k−1* folds and test on the held-out one; rotate so every piece is the test set once; average the results. We use **5-fold** CV.

### Leakage & grouping by subject — the most important rule
**Leakage** = when information from the test set sneaks into training, making the model look better than it really is. Our specific danger: **the same child appearing in both training and test.** Each child has a personal baseline (some just score higher). If a child's sessions are in both sets, the model can secretly learn *"this is kid #7, kid #7 scores ~1.9"* and predict well **without learning anything about glucose** — then fail completely on a new child.

The fix: **group by subject.** We split the *children* into folds (using scikit-learn's **`GroupKFold`**, grouping on the subject id `subid`), guaranteeing **a child is never in both training and test.** This answers the only honest question: *"can we predict a brand-new child we've never seen?"* Every result in the project uses this grouped CV.

We even keep the *wrong* way around as a diagnostic: **session-level CV** (shuffle all sessions ignoring which child they came from — which *does* leak). If the leaky "session" score is much better than the honest "grouped" score, that's a fingerprint that *"most of the apparent signal is just per-child baseline, not a real glucose effect."* Comparing the two is a deliberate check.

### Nested CV
Some models have a knob to tune (Ridge's penalty strength). To tune it *without* peeking at the test fold, we use **nested CV**: inside each training fold, do a *smaller* inner cross-validation (`GridSearchCV`) to pick the best knob, then evaluate once on the untouched outer test fold. This keeps tuning honest too.

### The metrics: R², RMSE, MAE
After predicting the test fold, we measure how good the predictions are:
- **R² ("R-squared") — the main one.** *"How much better are we than always guessing the average?"*
  `R² = 1 − (our squared error) / (baseline's squared error)`.
  - **1.0** = perfect. **0.0** = no better than guessing the average (useless but not harmful). **below 0** = *worse* than guessing the average (usually overfitting). Yes, R² can go negative on held-out data.
  - Tiny example: true scores `2, 4, 9` (average 5). Baseline "guess 5" squared errors: 9+1+16 = 26. If our model's squared errors sum to 3, then R² = 1 − 3/26 = **0.88** (great). If they sum to 26, R² = 0. If 101, R² = 1 − 101/26 = **−2.88** (much worse than guessing).
- **RMSE (root mean squared error)** — the typical size of our prediction error, in the score's own units (lower is better).
- **MAE (mean absolute error)** — similar, the average absolute miss (lower is better).

We report each as **mean ± spread across the 5 folds**, e.g. `−0.06 ± 0.06`, so you see both the level and how much it wobbles.

### One more idea: within-subject normalization
Because between-child baseline differences dominate, we also test: *instead of predicting the raw score, predict how much a child did better/worse than **their own** usual average.* Subtracting each child's personal mean ("centering") strips out the baseline, so any leftover predictability is a **real within-child glucose effect**. (In our real data this was also ~0 — no within-child effect either.) This is the `target_norm="center"` option.

---

## The 12 words to be able to define on the spot
**ML / supervised learning · regression vs classification · target · features · feature engineering vs raw-data learning · foundation model / TSFM · Chronos · embedding / encoder · window · overfitting · cross-validation & leakage (grouped CV) · R² & baseline.**

If you can explain each of these in a sentence, you understand the *why* behind every line of code. Next: `02_CODE_ARM_A.md` maps these ideas onto the actual files.
