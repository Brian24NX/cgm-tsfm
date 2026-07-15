# End-to-End Walkthrough + What the Professor Might Ask

*The payoff file. Part 1 follows ONE glucose window through the actual code, naming every function and shape. Part 2 is a Q&A drill for the meeting. Part 3 is a suggested way to present.*

---

## Part 1 — One command, traced through every file

When you run the real analysis:
```bash
python -m cgm_tsfm.run_headtohead --encoder chronos --real --device cuda
```
here is exactly what happens, file by file, following **one session** (say, child `217000`'s 3rd test, which had a 45-reading glucose window and a real Symbols score of 1.83):

**Step 0 — `run_headtohead.py` starts.** It reads the command-line options with `argparse`, sees `--real`, and calls `data.load_real_data()`.

**Step 1 — `data.py` loads and windows the data.**
- `pd.read_csv` reads the two cohort CSVs into a table.
- For our session, the `Glucose_Before_Test` cell holds the text `"[141.0, 138.0, 135.0, …]"`. `parse_glucose()` turns it into a NumPy array `[141, 138, 135, …]` of length 45 and drops any `NaN`s.
- `apply_window()` keeps it (≥ 3 readings). The score `prices_cognitive_score` is negated (per `PRICES_IS_INVERTED`); `symbols_cognitive_score` = 1.83 is kept as-is.
- Result: a `CGMDataset` with `windows` (956 arrays), `subjects` (956 `subid`s), and `targets` (3 score arrays). Our session is one entry: `window = [141,138,…]` (len 45), `subject = "217000"`, `symbols = 1.83`.
- `ds.summary()` prints "**956 sessions, 20 subjects**…".

**Step 2 — `encoders.py` turns every window into 512 numbers.**
- `extract_embeddings(ds.windows, cfg)` checks the cache; first time, it builds `ChronosEncoder`, which loads `amazon/chronos-bolt-small`.
- Our 45-reading window becomes a 1-D tensor `(45,)`, batched with others. `pipeline.embed(batch)` returns shape `(B, 4, 512)` for our window (45 → 3 patches + 1 token). `emb.mean(dim=1)` mean-pools the 4 tokens → **one `(512,)` embedding** for our session.
- All 956 sessions → a matrix **`X` of shape `(956, 512)`**. Saved to `.cache/…​.npy` so next time it's instant.
- (`handcrafted.extract_handcrafted_features` also runs → a `(956, 43)` matrix, for the sanity-check comparison.)

**Step 3 — `regression.py` predicts and scores fairly (Arm A).**
- `run_arm_a(ds, X)` loops over the 3 scores. For Symbols: it takes the sessions that have a Symbols score → `X` (embeddings), `y` (scores), `g` (subject ids).
- For each model (baseline, Ridge, SVR, Linear) it calls `nested_group_cv`:
  - **`GroupKFold(5)`** splits the **20 children** into 5 groups. In one fold, child `217000` is in the **test** group, so **none of `217000`'s sessions are in training** (the `assert` guarantees it).
  - A `Pipeline(StandardScaler → [PCA] → Ridge)` is `.fit()` on the training children only; `GridSearchCV` (inner GroupKFold) picks Ridge's best `alpha`.
  - The fitted model `.predict(X[test])` produces a guessed Symbols score for `217000`'s sessions. Compare guesses to the real 1.83 (and the others) → per-fold **R²/RMSE/MAE**.
  - Average over the 5 folds → Symbols: **R² ≈ −0.06** (Ridge/SVR) — i.e. *no better than guessing the average*.

**Step 4 — `run_headtohead.py` prints + saves.** It assembles the per-target table (Chronos vs hand-crafted vs, with `--with-arm-b`, the Arm-B network), prints it, and writes `results/headtohead_real.md`. The verdict for every score: *"all ≈ baseline — no generalizable signal."*

**The whole journey in one sentence:** *CSV text → parsed glucose array (`data.py`) → 512-number Chronos embedding (`encoders.py`) → regularized model scored under leakage-free grouped CV (`regression.py`) → an honest R² ≈ 0.*

---

## Part 2 — Questions the professor may ask (with answers)

Rehearse these out loud. Short, confident, honest.

**Q: What is a TSFM, and why Chronos?**
A: A Time-Series Foundation Model — a large network pre-trained on millions of time series, so it already knows general temporal patterns and can produce useful features for *our* series with no extra training. We use Amazon's Chronos because it's strong, free, treats a series like a language sequence, and — practically — it's NaN-tolerant, handles variable length, and scales each series internally, so we feed raw glucose with no preprocessing.

**Q: What exactly is the input to Chronos? What shape?**
A: It's **univariate** — one variable over time. Input is conceptually `(batch, length)`; there's **no channel dimension**, which is *why* our project is single-channel glucose. Output is `(batch, tokens, 512)`; we mean-pool the tokens to get **one 512-number embedding per window: `(batch, 512)`**. (Chronos-Bolt makes one token per 16-reading patch; Chronos-T5 one per reading — measured shapes are in `docs/06`.)

**Q: What is the "embedding"? Do those 512 numbers mean anything?**
A: It's a fixed-size fingerprint of the glucose curve's shape, produced by Chronos's internal representation. The individual numbers aren't human-interpretable — collectively they're a rich summary that a simple model can predict from.

**Q: Why is the encoder "frozen"?**
A: We use Chronos zero-shot — we never update its weights, only read out embeddings. That's the standard, fast "representation" approach, and with only 20 subjects we don't have the data to fine-tune a huge model without overfitting.

**Q: Why mean-pool the tokens instead of using them all / the last one?**
A: We need one fixed vector per window; averaging over tokens summarizes the whole window and is exactly the recipe the reference ICML paper used. "Last token" is available as a knob (`pooling`); the sweep can compare — mean was marginally better.

**Q: Why sklearn for Arm A instead of PyTorch?**
A: Arm A deliberately wants the *simplest* predictor on top of the embedding — a regularized linear/kernel model. sklearn provides those in one line with battle-tested nested cross-validation. PyTorch would be more code and more overfitting risk for no benefit. We *do* use PyTorch where it's the right tool: Chronos itself, and Arm B's trainable head.

**Q: Why these specific models (Ridge, SVR)? Why the plain Linear one too?**
A: Ridge = linear with a penalty that prevents overfitting on 512 features — the robust default. SVR = allows a curved relationship in case it's non-linear. Plain Linear has no penalty and overfits to R²≈−3 — we keep it *to demonstrate* that regularization is mandatory. And a mean baseline as the bar to beat.

**Q: What is R², and why are your values negative?**
A: R² = how much better than "always guess the average." 0 = no better; below 0 = worse than guessing (usually overfitting or genuinely no signal). Our ~0/slightly-negative values mean the glucose window doesn't predict the score for a new child — expected given a weak effect and large between-child differences.

**Q: Why "grouped" cross-validation? What would go wrong otherwise?**
A: If the same child were in both training and test, the model could recognize the child and use their personal average — inflating the score without learning any glucose→cognition rule, then failing on new children. Grouping by `subid` guarantees a child is never in both, so we measure honest new-child prediction. (We also run the leaky "session" version as a diagnostic: it's only slightly better, confirming the signal is mostly per-child baseline.)

**Q: What's a "window"?**
A: The glucose readings before one test — the input for one prediction. In our data they range 15 min to 24 h (median ~3 h). The wide range is actually a concern I want to fix next (a uniform window).

**Q: Why did the trainable neural network (Arm B) do *worse* than the simple model?**
A: More flexibility + only 20 subjects + little real signal = overfitting. A bigger model can memorize noise; the simpler regularized model generalizes better. It's the expected outcome, and it's why simple models are the right default here.

**Q: Is 20 subjects enough to conclude anything?**
A: It's an honest interim result on the data we have; the full study targets ~92. I'd re-run as more arrive. Power is a real caveat, which I'd state in any writeup.

**Q: What is the synthetic data for — isn't that fake?**
A: It's a plumbing test — fake data with the real schema, used to build and validate the pipeline before the real data arrived (and to prove the pipeline *can* detect a within-child signal when one is injected). Synthetic numbers are never presented as findings.

**Q: If the result is null, what did you actually accomplish?**
A: A complete, leakage-free pipeline (data → Chronos embeddings → regularized models → honest grouped-CV scoring), run on all 20 patients, giving a rigorous null that corroborates the earlier feature-based attempt with a stronger method — plus a clear diagnosis (subject-baseline dominates) and a concrete plan to give the hypothesis a fairer shot (a uniform pre-test window, a within-person analysis).

---

## Part 3 — How to walk the professor through it (suggested order)

1. **The question + the honest result** (30 s): glucose-before-test → cognitive score; first real answer is a rigorous null.
2. **The assembly line** (2 min): show the diagram from `00_START_HERE` — data → Chronos encoder → 512-number embedding → regularized model → R² under grouped CV. Name the file at each station.
3. **Zoom into the two hard/interesting bits** (5 min):
   - *The encoder / Chronos input format* — univariate `(B, L)` in, `(B, 512)` out; frozen; use `docs/06`.
   - *Honest evaluation* — grouped CV + R² + why leakage would fake it. This is where the science lives.
4. **Arm A vs Arm B** (2 min): sklearn simple model vs trainable MLP; Arm B overfit → simpler is better here.
5. **Result + why + next steps** (2 min): null, subject-baseline dominates, plan = fix the window + within-person analysis.

If you can do steps 2–3 pointing at the actual files and functions (`data.parse_glucose`, `encoders.extract_embeddings`, `regression.nested_group_cv`), you'll have shown you understand the code, not just ran it. That's the whole assignment.

---

## The five-sentence summary (memorize this)
1. We predict a child's cognitive-test score from the glucose readings just before the test.
2. Each variable-length glucose window is turned into a fixed 512-number "embedding" by a frozen time-series foundation model (Amazon Chronos) — that's the encoder.
3. A simple regularized model (Ridge/SVR via scikit-learn = Arm A) or a small trainable network (PyTorch = Arm B) predicts the score from those 512 numbers.
4. We score it honestly with R² under subject-grouped cross-validation, so a child is never in both training and test — the only fair test of predicting a new child.
5. The first real result is a rigorous null (R² ≈ 0), meaning no detectable short-term glucose→cognition signal in these 20 patients — an honest finding, with a clear plan to test the hypothesis more fairly next.
