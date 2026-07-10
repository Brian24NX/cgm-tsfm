# Meeting Prep — First Real Results (private notes)

*Private + gitignored (Liuyi won't see this). This is your script + cheat-sheet for the meeting. Read §1 and §6 if nothing else.*

---

## 1. The 30-second version (say this first, out loud)

> "I finished setting up the pipeline on the lab server, got the real data from Phil — all **20 patients** — and ran the first real analysis. The honest result is a **null**: using the ~hours of CGM before each test, we **cannot predict the cognitive scores better than just guessing the average**, and that holds for our modern foundation-model approach *and* the classical hand-crafted features, under a leakage-free evaluation. That corroborates the earlier attempt with a stronger method. I also found a **data issue with the glucose window that I think is worth fixing next**, and I have a concrete plan."

**The three sentences that carry the meeting:**
1. "Everything is built and the first real run is done — this is a real result, not a plumbing test."
2. "The result is a rigorous null; a fancy model can't create signal that isn't there, so I'm not overselling."
3. "But the pre-test window is inconsistent (15 min to 24 h per test) — fixing that is my #1 next step and could change the answer."

Do **not** apologize for the null. A clean, honest, leakage-free null is a legitimate scientific finding — that's the framing.

---

## 2. What each file in `results/` means (so you can answer "what's in there?")

| File | What it is | The number that matters |
|---|---|---|
| **`README.md`** | The narrative that ties it all together. **Open this one to present.** | Has the real headline + how to read R² |
| **`headtohead_real.md`** | ⭐ THE main result. Our 3 approaches on the real 20 patients, "predict a new kid" (grouped CV): Chronos+Ridge/SVR (Arm A), Chronos+MLP (Arm B), hand-crafted (sanity). | All R² **≈ 0 to −0.28** → null |
| **`headtohead_real_centered.md`** | The **within-subject** test: remove each kid's personal average, ask "when THIS kid's glucose moves, does their score move?" | All R² **≈ 0** → no within-kid signal either |
| **`sweep_real.md`** | Does shrinking the 512-number fingerprint (PCA) help? | Marginal: mean R² **−0.052 → −0.045**, still negative |
| `headtohead_synthetic.md` | Same comparison but on **fake** data — a pipeline harness check. | **Ignore for the meeting** (not scientific) |
| `sweep_synthetic.md` | Sweeps on fake data — harness check. | **Ignore for the meeting** |

> If Liuyi opens the folder and asks "why are there synthetic files?" → "Those were the plumbing tests I built the pipeline against before the data arrived; the `_real` files are the actual findings."

**Reminder on R²:** 1.0 = perfect, **0 = no better than guessing the average**, below 0 = worse than guessing. So "≈ 0 or slightly negative" = "the glucose isn't helping."

---

## 3. The headline numbers (the one table to show)

**"Predict a new kid" (subject-grouped CV) — `headtohead_real.md`:**

| score | Chronos (Arm A) | Chronos+MLP (Arm B) | hand-crafted |
|---|--:|--:|--:|
| Grids | −0.089 | −0.218 | −0.065 |
| Symbols | −0.056 | −0.281 | −0.038 |
| Prices | −0.012 | −0.168 | −0.009 |

**"Within the same kid" (personal baseline removed) — `headtohead_real_centered.md`:**

| score | Chronos (Arm A) | hand-crafted |
|---|--:|--:|
| Grids | −0.018 | −0.004 |
| Symbols | −0.022 | −0.004 |
| Prices | −0.002 | −0.002 |

**How to read it in one breath:** every number is at or just below 0 → nothing beats the baseline; the trainable MLP (Arm B) is *worse* because it overfits 20 kids; centering doesn't rescue it → there isn't a hidden within-person effect either.

---

## 4. Step-by-step: how to present (≈12–15 min)

Turn each step into a slide, or just talk it through with `results/README.md` and `docs/06` open.

**Step 1 — The question (1 min).**
"Do short-term blood-sugar swings affect how a T1D kid does on a thinking test minutes later? I'm testing the *modern* approach: feed the raw glucose to an AI foundation model instead of hand-picking features."

**Step 2 — What I built & set up (2 min).**
- "The full pipeline runs on the lab server (`cpsl-mds`) on GPU — that's Task 1 done."
- "I got the data from Phil and verified it: **20 patients**, 14 + 6 across the two cohorts. One important catch — the first Cohort-2 file was missing so much glucose that it silently dropped a patient to 19; Phil's **Updated** file fixed it, so we're at the full 20."
- *(This shows diligence — you caught a data bug.)*

**Step 3 — How it works (2–3 min).** *(Have `docs/06_CHRONOS_INPUT_FORMAT.md` ready — this is the Task-3 material Liuyi may quiz you on.)*
"Each test session's glucose window goes into **Chronos** (Amazon's time-series foundation model), which turns it into a **512-number fingerprint**; a small model predicts the score from that fingerprint. I evaluate with **subject-grouped cross-validation** — never testing on a kid seen in training — so the result is honest."
- If asked about input format: "Chronos is univariate — input is **(batch, length)**, no channel dimension; output is **(batch, 512)** after pooling. Bolt reads the series in 16-reading patches; T5 reads one token per reading."

**Step 4 — The result (3 min).** Show the §3 table.
"Under honest CV, **no approach beats guessing the average** for any of the three scores — not the Chronos fingerprint, not a trainable head, not the 43 hand-crafted features. Removing each kid's personal baseline doesn't reveal a within-person effect either. Plain linear regression actually blows up to R² ≈ −3, which is why regularization is essential. **This is a rigorous null** — and it lines up with the earlier feature-based attempt, now confirmed with a stronger method."

**Step 5 — What it means + next steps (3–4 min).** (See §6.) Lead with the window issue.
"A model can't invent signal that isn't there — but before we conclude the effect is truly absent, there's a **methodological fix I want to make**: the glucose window is wildly inconsistent (15 min to 24 h per test), so we may be feeding the model apples and oranges. That's my #1 next step, and I have a short list after it."

**Step 6 — Questions for you (1–2 min).** Pull up the A–F questions (they're in `docs/04_OPEN_QUESTIONS.md`). Lead with **B (the window)**, then **A (what counts as success)** and **D (prediction vs association framing)**.

---

## 5. Questions Liuyi may ask — and solid answers

- **"So there's no effect at all?"** → "No *generalizable, new-subject* effect and no detectable *within-subject* effect at N=20, under honest CV. But it could be masked by the inconsistent window — that's why fixing the window is my next step before drawing a hard conclusion."
- **"Why are the R² values negative?"** → "Negative just means 'worse than guessing each score's average.' With little real signal and big differences between kids, that's expected — even the baseline scores slightly negative on held-out kids."
- **"Did the AI beat the hand-crafted features?"** → "They're tied — both sit at baseline. The AI didn't lose; neither can find signal that isn't there. So 'raw-data learning' is at least on par with feature engineering here."
- **"Why did the trainable model (Arm B) do worse?"** → "It's more flexible, so on only 20 kids with no signal it overfits. Simpler + regularized (Arm A) is the right tool at this scale."
- **"What is Chronos' input format?"** *(the Task-3 quiz)* → §Step 3 above / `docs/06`. "(batch, length), univariate; → (batch, 512) after mean-pooling; Bolt patches 16 readings, T5 one token per reading."
- **"Is 20 patients enough?"** → "It's what we have now; the K01 targets ~92. So I'd treat this as an honest interim result and re-run as more arrive — power is a real caveat."
- **"What's this window problem?"** → §6 item 1.

---

## 6. Next steps to get a better result (present these as *your plan*)

Ordered by expected impact. Frame as "here's how I'd strengthen this."

1. **★ Fix the pre-test glucose window (biggest lever).** Right now `Glucose_Before_Test` ranges from ~15 min to ~24 h per test (median ~3 h) — with the `Prev_Test_Time` column, it looks like it's "all glucose since the previous test," not a fixed lookback. Feeding the model such wildly different windows is a confound that could be *hiding* a real effect. **Plan:** get the raw Dexcom stream / merge script from Liuyi/Phil and define a **uniform window (e.g. exactly the 2 h before each test**, per the K01 hypothesis), then re-run. *(This is open question B — lead with it.)*
2. **Reframe from prediction to within-person association.** "Predict a brand-new kid" (grouped CV) is a very hard bar. The biological question is really *within* a person — "when a given kid's glucose swings, does their performance move?" A **mixed-effects model / within-subject correlation** targets that directly and is standard for this kind of repeated-measures data. *(Open question D.)*
3. **Zoom in on hypoglycemia.** Cognitive effects of glucose are most plausible during **lows**. Restricting to sessions preceded by a hypoglycemic excursion (or adding a "was there a low?" feature) is a focused, physiologically-motivated test.
4. **Run the full sweep on real data.** We only swept PCA on real; running the checkpoint / window / pooling sweeps (`run_sweep --kind all --real`, a few min on GPU) characterizes the ceiling and confirms bigger models don't help.
5. **Add obvious confounders as controls.** Time-of-day and practice/learning effects (tests are 5×/day for ~10 days; `Prev_Test_Time` is available) plausibly move the scores; controlling for them could sharpen any real glucose signal.
6. **Scale with more data** as it arrives (toward N≈92), and only *then* consider heavier methods (e.g. LoRA fine-tuning of Chronos) — not worth it while the frozen approach shows nothing.

**One-liner to close the meeting:** "So: the pipeline is done and the first honest result is a null, but I don't think we've given the hypothesis its fairest shot yet — a clean, fixed glucose window and a within-person analysis are the next two things I'd do."

---

## 7. Cheat-sheet numbers (glance before you walk in)
- **20 patients, 956 sessions**, 3 scores (Grids / Symbols / Prices).
- Windows: **3–288 readings ≈ 15 min–24 h**, median 36 ≈ **3 h**.
- Best real R² anywhere ≈ **−0.01** (Prices, Chronos) → **null**; within-subject ≈ **0**; PCA best mean **−0.045**.
- Chronos-Bolt fingerprint = **512 numbers**; input **(batch, length)**, univariate.
- Server: env on the SSD, torch+CUDA on an **RTX 6000 Ada**.
