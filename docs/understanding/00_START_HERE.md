# Understand This Project — Start Here

*A from-scratch course so you can explain the whole pipeline and every line of code to your professor. It assumes **zero** machine-learning background. Read the files in order.*

| File | What it teaches |
|---|---|
| **00_START_HERE.md** (this) | The big picture + a map of how the code fits together |
| **01_CONCEPTS.md** | Every idea and tech term from scratch (ML, TSFM, sklearn, regression, window, embedding, R²…) |
| **02_CODE_ARM_A.md** | Every Arm A `.py` file explained meticulously, with *why* it's written that way |
| **03_CODE_ARM_B.md** | The `ben_adapter/` files (Arm B) + the PyTorch/Lightning ideas |
| **04_WALKTHROUGH_AND_QA.md** | One glucose window followed through the real code, end to end + likely professor questions |

---

## 1. The one-sentence project

> **We take the continuous-glucose-monitor (CGM) readings from the ~hours before a child takes a quick thinking test, and we try to predict their score on that test — using a modern AI model instead of hand-picked glucose statistics.**

The scientific question: *does short-term blood sugar affect momentary thinking in kids with Type 1 Diabetes?* If glucose predicts the score, that's evidence it does.

Our **first real answer is "no measurable effect"** (the R² scores are ~0 or slightly negative). That is a **legitimate, honest result** — a fancy model cannot invent a signal that isn't in the data. Your job this week is not to fix that number; it's to **understand exactly how the machine that produced it works.**

---

## 2. The mental model: an assembly line

Think of the code as a factory line. Raw material goes in one end, a prediction comes out the other:

```
  ┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
  │  1. DATA   │──▶│ 2. ENCODER  │──▶│ 3. FEATURES  │──▶│ 4. PREDICT │──▶│ 5. SCORE IT  │
  │ glucose +  │   │  (Chronos)  │   │  512 numbers │   │ (regressor)│   │ (R², fairly) │
  │ test score │   │ turns curve │   │ per session  │   │  number in │   │ cross-       │
  │ per session│   │ into numbers│   │ = "embedding"│   │ number out │   │ validation   │
  └────────────┘   └─────────────┘   └──────────────┘   └────────────┘   └──────────────┘
        │                 │                  │                 │                 │
     data.py         encoders.py      (the output of    regression.py    regression.py
                                       encoders.py)      (Arm A) /        + run_*.py
                                                         ben_adapter (B)
```

Every `.py` file is one station (or a helper) on this line. Once you can point at each file and say *"this is the station that does X, and it's written this way because Y,"* you understand the project.

---

## 3. The file map (what each `.py` is for)

Everything lives in the `cgm_tsfm/` folder. Here's the whole package, grouped by job:

```
cgm_tsfm/
│
├── __init__.py          "This folder is a Python package." + a docstring describing the project.
├── config.py            SETTINGS. The 3 score names, file paths, CV settings, which Chronos model.
├── data.py              STATION 1. Load real CSVs OR make fake data; cut glucose into "windows".
├── encoders.py          STATION 2. Turn each glucose window into 512 numbers using Chronos (or a mock).
├── handcrafted.py       The OLD way (43 hand-picked glucose stats) — kept only as a comparison baseline.
├── regression.py        STATIONS 4+5 (Arm A). Fit simple models (Ridge/SVR) + score them fairly (R², grouped CV).
│
├── run_demo.py          A "main" script: runs the whole line once and prints results.
├── run_headtohead.py    A "main" script: runs Chronos vs hand-crafted (+ Arm B) side by side.
├── run_sweep.py         A "main" script: tries different settings (model size, window, PCA) to see what helps.
│
└── ben_adapter/         ARM B — the "trainable neural network" version of stations 4+5.
    ├── __init__.py      Marks ben_adapter as a sub-package.
    ├── model.py         The neural network: frozen Chronos encoder + a small trainable "head".
    ├── lightning_module.py  The training logic (loss, metrics, optimizer) wrapped for the Lightning library.
    ├── datamodule.py    Feeds one CV fold's data to the trainer in batches.
    ├── train.py         Runs the grouped-CV training loop for Arm B.
    └── smoke_test.py    A tiny self-check that Arm B is wired correctly (no data/download needed).
```

**Arm A vs Arm B** — both do the same job (turn the 512-number embedding into a predicted score), two ways:
- **Arm A** (`regression.py`) uses **classical statistics models** from the `scikit-learn` library — fast, simple, nothing to "train" beyond fitting a formula.
- **Arm B** (`ben_adapter/`) uses a **small trainable neural network** built with **PyTorch/Lightning** — more flexible, adapted from a labmate's (Ben's) code. This was a specific task ("adapt Ben's code").

They share stations 1–2 (same data, same Chronos embeddings).

---

## 4. Two libraries, two worlds (this trips people up — know it cold)

The code uses **two different families of tools**, and knowing *which does what* is half of understanding the repo:

| | **`scikit-learn` (sklearn)** | **`PyTorch` (+ Lightning)** |
|---|---|---|
| What it is | A library of **classic, ready-made ML models** (linear regression, SVM, PCA…) | A library for building/training **neural networks** from scratch |
| Who "learns" | You call `.fit()` and it solves a math formula in one shot | You loop over the data many times, nudging weights with gradient descent |
| Used in | **Arm A** (`regression.py`) — Ridge, SVR, the R² scoring, PCA, cross-validation | **Arm B** (`ben_adapter/`) — the trainable head; **and** Chronos itself is a PyTorch model |
| Why both | Arm A only needs simple, robust models → sklearn is perfect and fast. Arm B needs a custom trainable network → that's PyTorch's job. Chronos is a big pretrained PyTorch network, so PyTorch is always in the room. |

*"Why not just use PyTorch for everything?"* — Because for Arm A we deliberately want the **simplest possible model** (a regularized linear fit). sklearn gives that in one line with battle-tested cross-validation built in. Reaching for PyTorch there would be more code, slower, and easier to get wrong — with no benefit. Right tool for each job. (Full explanation in `01_CONCEPTS.md`.)

---

## 5. How to use these docs

1. Read **`01_CONCEPTS.md`** first — it defines every word (`TSFM`, `embedding`, `regression`, `window`, `cross-validation`, `R²`, `sklearn`, `PyTorch`, …). Don't skip it; the code files assume these.
2. Read **`02_CODE_ARM_A.md`** and **`03_CODE_ARM_B.md`** with the actual `.py` files open beside them — they go function by function.
3. Read **`04_WALKTHROUGH_AND_QA.md`** last — it ties everything together by following one real glucose window through the code, and lists questions your professor may ask with answers.

By the end you should be able to open any file in `cgm_tsfm/`, point at any function, and say what it does and why. That's the goal.
