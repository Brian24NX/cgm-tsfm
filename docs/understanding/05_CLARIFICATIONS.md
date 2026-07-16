# Clarifications (Round 1) — Cross-Validation, Centering, and the Config

*Deeper, slower answers to the points you pulled from `01_CONCEPTS.md` and `02_CODE_ARM_A.md`, with tiny worked examples and ASCII infographics. Topics: (1) the whole cross-validation family + leakage + the session-vs-grouped diagnostic + nested CV; (2) within-subject centering; (3) the config settings — including the CPU-vs-GPU question; (4) windowing; (5) a couple of small code bits.*

---

# 1. Cross-validation, built up slowly (the big one)

You had the most questions here, so let's build it from the ground up. Five steps.

## 1.1 Why we can't just measure on the training data

Imagine a student who **memorizes** the answers to a practice test. Give them that *same* test → 100%. Give them a *new* test → they flunk. Measuring a model on the data it learned from is the same trap: it rewards **memorizing**, not **learning**. So we must always measure on data the model has **not seen**.

## 1.2 Train/test split

Split the examples into two piles:
```
   ALL 956 sessions
   ┌───────────────────────────┬───────────┐
   │        TRAIN (learn)      │ TEST (grade)│
   │         ~765              │    ~191     │
   └───────────────────────────┴───────────┘
   model learns here            we ONLY measure here
```
The model learns from TRAIN; we report the score on TEST. Honest — but it depends on which sessions landed in TEST (you might get lucky/unlucky).

## 1.3 k-fold cross-validation (removes the luck)

Cut the data into **k equal pieces ("folds")**. Take turns: each fold is the TEST once, the other k−1 are TRAIN. Average the k scores. We use **k = 5**:
```
 Round 1:  [TEST ][train][train][train][train]
 Round 2:  [train][TEST ][train][train][train]
 Round 3:  [train][train][TEST ][train][train]
 Round 4:  [train][train][train][TEST ][train]
 Round 5:  [train][train][train][train][TEST ]
                     average the 5 test scores → final R²
```
Every session gets to be in the test set exactly once. More stable than a single split.

## 1.4 The leakage trap: a random split lets a child appear in both piles ⚠️

This is the key point from your excerpts [002–014, 023–027]. Here's the danger, drawn out.

Each **child** takes ~40 tests, so each child contributes ~40 sessions. Children differ a lot in their *baseline* — some just score higher than others, for reasons that have nothing to do with glucose:
```
   Child A's ~40 scores hover around 1.9   (a "high scorer")
   Child B's ~40 scores hover around 1.2   (a "low scorer")
```
Now suppose we split **sessions randomly** (ignoring who they came from). Child A's 40 sessions get scattered into BOTH piles:
```
              TRAIN pile                     TEST pile
   ┌──────────────────────────┐     ┌──────────────────────────┐
   │ A A A A A  B B B  C C ... │     │ A A A   B B   C C C ...  │
   └──────────────────────────┘     └──────────────────────────┘
        ▲ some of A's sessions            ▲ MORE of A's sessions
```
The model can now **cheat**: it learns *"when I recognize the fingerprint of child A, guess ~1.9."* On the TEST pile it sees child A again → it guesses ~1.9 → looks accurate! But it **never learned anything about glucose** — it just memorized *who each child is*. This is **leakage**: information about the test sessions ("this is child A") leaked into training through the child's identity.

Then the moment a **brand-new child** (never seen) shows up, the model has no memorized baseline for them and **fails**. So the good-looking score was a lie.

## 1.5 The fix: split by CHILD, not by session (GroupKFold) ✅

Assign **whole children** to folds, so a child is **entirely** in TRAIN or **entirely** in TEST — never split across both:
```
 20 children: 01 02 03 ... 20     (each child = a solid block of ~40 sessions)

 Round 1:  TEST = { 01 02 03 04 }        TRAIN = { 05 ... 20 }
 Round 2:  TEST = { 05 06 07 08 }        TRAIN = { 01-04, 09-20 }
 Round 3:  TEST = { 09 10 11 12 }        TRAIN = { 01-08, 13-20 }
 Round 4:  TEST = { 13 14 15 16 }        TRAIN = { 01-12, 17-20 }
 Round 5:  TEST = { 17 18 19 20 }        TRAIN = { 01-16 }

 RULE (enforced by an assert in the code): a child NEVER appears in both piles.
```
In the code this is scikit-learn's **`GroupKFold`**, grouping on `subid`. Now the model *cannot* memorize a test child's baseline (it has never seen that child), so the only way to do well is to actually learn a **glucose → score** rule that works on strangers. That's the honest question: *"can we predict a brand-new child?"* **Every scientific number in the project uses this.**

## 1.6 The session-vs-grouped "fingerprint" diagnostic [items 007–014, 027]

We deliberately run **both** ways and compare — not to trust the leaky one, but because *the gap between them is informative*:

```
                          honest        leaky
                        (GroupKFold)   (random by session)
   grids   R² :            -0.09          -0.04
   symbols R² :            -0.06           0.00
   prices  R² :            -0.01          -0.02
                              │              │
                              └──── small gap, and BOTH ≈ 0 ────┘
```

How to read the gap:
- **Big gap** (session R² *much* higher than grouped R²) → the model was mostly exploiting **who the child is** (baseline), not a real glucose effect. The size of the gap ≈ how much "cheating on identity" was inflating things.
- **Small gap AND both ≈ 0** (our real result) → there wasn't much signal *either* way; even the identity trick barely helps. Consistent with "the glucose→cognition effect is very weak here."

> One-liner for the professor: *"Grouped CV is the number I trust for any claim; session CV is a diagnostic — the gap between them measures how much apparent performance is just per-child baseline rather than a generalizable glucose effect."*

## 1.7 Nested CV: tuning a knob without cheating [items 030–032, 045]

Some models have a **knob** to set before training. Ridge's knob is `alpha` (how hard to penalize big weights — its regularization strength). We must *choose* `alpha`, but **if we pick the alpha that scores best on the TEST fold, that's cheating** (we peeked at the test answers).

The fix is a **loop inside the loop** — **nested** cross-validation:

```
 OUTER loop (5 folds) — estimates honest performance:
 ┌─────────────────────────────────────────────────────────────┐
 │ Outer Round 1:  TRAIN = children 05–20     TEST = children 01–04   │
 │                                                               │
 │   INNER loop (3 folds), run ONLY inside the TRAIN children:   │
 │   ┌───────────────────────────────────────────────────────┐ │
 │   │ try alpha = 0.1, 1, 10, 100, 1000                        │ │
 │   │ for each alpha: 3-fold CV *within* children 05–20        │ │
 │   │ → pick the alpha with the best INNER score               │ │
 │   └───────────────────────────────────────────────────────┘ │
 │                                                               │
 │   refit Ridge (with the winning alpha) on ALL of 05–20        │
 │   → predict children 01–04 ONCE  → record that R²             │
 └─────────────────────────────────────────────────────────────┘
     ... repeat for outer rounds 2–5, average the 5 test R²s
```

- The **inner** 3-fold CV (`GridSearchCV` in the code) picks `alpha` using *only* the outer-training children — the outer TEST children (01–04) are never touched during tuning.
- This is why `config.py` has **`OUTER_CV_SPLITS = 5`** (the outer honesty loop) and **`INNER_CV_SPLITS = 3`** (the inner tuning loop). "5-fold outer / 3-fold inner" = exactly this picture.
- Both loops use grouped-by-child splitting, so leakage is blocked at both levels.

---

# 2. Within-subject centering (target_norm="center") [items 038–040, 071–078]

Grouped CV tells us we can't predict a **new child**. But there's a *different*, still-interesting question: *within a child we already know, do their glucose ups/downs track their score ups/downs?* To ask that, we remove each child's personal baseline first.

**Worked example.** Two children, three sessions each:
```
   RAW scores                        CENTERED (subtract each child's own mean)
   Child A: 1.8, 2.0, 1.9  (mean 1.9)   →   -0.1, +0.1,  0.0
   Child B: 1.2, 1.4, 1.3  (mean 1.3)   →   -0.1, +0.1,  0.0
             ▲ A always scores higher         ▲ now BOTH children look identical!
```
After centering, "which child is this" carries **no information** (every child is centered on 0) — so the model can no longer win by memorizing baselines. Any predictability that *survives* must come from the **within-child** glucose signal. That's exactly the effect the study hypothesizes.

The code (`within_subject_normalize`):
```python
for s in np.unique(subjects):     # for each distinct child...
    idx = subjects == s           # ...find that child's sessions...
    vals = y[idx]
    out[idx] = vals - vals.mean() # ...subtract that child's own average score
```

⚠️ **The honest caveat** (say this — it shows rigor): centering uses **each child's own sessions** to compute their mean ("**oracle** centering"). For a brand-new child you wouldn't know their mean yet. So this answers *"is there a within-child effect at all?"* — a **characterization**, not a deployable new-child predictor. In our real data the centered R² was also ≈ 0, i.e. **no within-child effect either.**

---

# 3. The config settings decoded [items 045–047, 049]

## 3.1 `RANDOM_STATE = 42` — the "seed"
Lots of steps involve randomness (how folds are drawn, model initialization). A **random seed** fixes that randomness so the code gives the **exact same result every time you run it** — essential for reproducibility (your professor can re-run and get your numbers). `42` is just a conventional arbitrary number (a programmer in-joke). Change it and the *splits* shuffle slightly, but conclusions shouldn't.

## 3.2 `OUTER_CV_SPLITS = 5`, `INNER_CV_SPLITS = 3`
These are the two loops from §1.7: **5** outer folds (honest performance estimate) and **3** inner folds (knob tuning). Bigger = more thorough but slower; 5/3 is the standard, and it matches the original study so the arms are comparable.

## 3.3 `WindowConfig` — `min_readings=3`, `max_readings=None` [item 046]
Controls how a glucose window is trimmed before it goes in:
- `min_readings=3`: throw away sessions with fewer than 3 readings (too little to be meaningful).
- `max_readings=None`: keep the **whole** window. Set it to a number (e.g. 30) to keep only the most-recent 30 readings = 2.5 h. (`None` = "no cap.")

## 3.4 `EncoderConfig` — and your CPU/GPU question [item 049] ⭐
You wrote: *"Cpu? I think it should be set to 'gpu', we are training on the GPU."* Great catch — here's the precise answer, and it's a good thing to be able to explain:

```python
device: str = "cpu"     # the DEFAULT in the config
```
Three things to know:

1. **This is only the *default*.** At run time we **override** it from the command line: `--device cuda`. So when you run on the server, the encoder *does* run on the GPU — the config default is just the fallback if nobody says otherwise. (You can verify: our real runs all used `--device cuda`, and Chronos loaded onto the GPU.)

2. **Why keep the default as `cpu` and not change it to GPU?** Because the code must also run on machines with **no GPU** (a laptop, a CI checker, a quick offline test). If the default were a GPU and you ran it where there's no GPU, it would **crash**. So the safe, portable default is `cpu`, and you explicitly ask for the GPU when you have one. This is standard practice.

3. **The exact string matters: it's `"cuda"`, not `"gpu"`.** PyTorch/Chronos name NVIDIA GPUs `"cuda"` (after NVIDIA's CUDA platform). `torch.device("gpu")` is **invalid** and would error. So if you ever set it by hand, use `"cuda"` (the comment in the file lists the valid options: `"cpu" | "cuda" | "mps"`; `mps` is Apple-Mac GPUs).

> So: **the config says `cpu`, but we run with `--device cuda`, so training *is* on the GPU.** Nothing to fix — but now you can explain it. *(Small accuracy note for the meeting: our card is an **NVIDIA RTX 6000 Ada** per `nvidia-smi` — not an "A6000", which is a different card. Worth getting right in front of a picky professor.)*

Should we change the default to `"cuda"`? I'd **keep it `cpu`** for the portability reason above — and just always pass `--device cuda` on the server (which we do). If you'd prefer the default flipped, I can do it, but then the code won't run on a machine without an NVIDIA GPU.

---

# 4. The windowing code [items 053–060]

```python
def apply_window(glucose, window):
    if glucose is None or glucose.size < window.min_readings:   # too short → drop
        return None
    if window.max_readings is not None and glucose.size > window.max_readings:
        glucose = glucose[-window.max_readings:]                 # keep most-recent N
    return glucose
```
- `glucose.size` = how many readings this window has.
- First `if`: if the window is missing or shorter than `min_readings` (3), return `None` → this session gets dropped later.
- Second `if`: if a cap is set and the window is longer, keep only the **most-recent** `max_readings`.
- **`glucose[-N:]` = "the last N elements."** Negative indexing counts from the end:
```
   glucose = [140, 138, 135, 150, 161, 158, 149, 152]   (8 readings)
   glucose[-3:]  →  [149, 152]  ... wait, that's the last 3:  [158, 149, 152]
                     └── the 3 most-recent readings (closest to the test) ──┘
```
We keep the *most-recent* readings because they're the ones closest in time to the cognitive test (the physiologically relevant window).

---

# 5. Two small code bits [items 065–067, 083]

- **`target_mask(target)`** returns a True/False array of "which sessions have a real (non-missing) value for this score." We model only those sessions — e.g. Symbols has 920 non-missing scores out of 956, so we use those 920. (Different scores have slightly different counts because some are occasionally missing.)

- **`eff_pca = min(pca_components, X.shape[1])`** just prevents asking for more PCA components than there are features. PCA can output at most as many components as the input width. For Chronos that's 512, for hand-crafted it's 43 — so if you asked for `PCA=64` on the 43-feature matrix, this clamps it to 43. A safety guard, nothing conceptual.

---

## The three sentences to lock in from this round
1. **Grouped CV** (split by child) is the honest test — "can we predict a new child?"; a **random session split leaks** because the model memorizes each child's baseline.
2. The **session-vs-grouped gap** is a diagnostic: a big gap = performance was mostly per-child baseline, not glucose; ours is small and both ≈ 0 = weak signal either way.
3. **Nested CV** = an inner loop (3-fold) picks the model's knob using only training children, then the outer loop (5-fold) grades once on untouched test children — so even *tuning* never peeks. And the config's `device="cpu"` is just a portable default we override with `--device cuda` on the GPU.
