# Open Questions for Liuyi (v2 — post-answers)

*Supersedes the pre-answer `QUESTIONS_FOR_LIUYI.md` (kept for history). These are the questions raised by actually building the pipeline and reading `diabetes-fitbit`. Ordered by how much they block progress.*

---

### Q1 — Access to the real merged CSVs  🔴 *blocks the real run*
The pipeline is built and validated; the only thing between me and a real result is the data. Can I get **`Cohort1_scores_merged_with_glucose.csv`** and **`Cohort2_scores_with_glucose.csv`** (they're gitignored in `diabetes-fitbit`, so not in the clone)? Or should I run in an environment where they already live? *This is your project's own data (Mack used it), separate from the pending IRB for Ben's data — so I assume it's accessible, but confirming.*

### Q2 — The pre-test window & the raw CGM stream  🔴 *shapes the model input*
In `diabetes-fitbit`, `Glucose_Before_Test` is **already clipped to some lookback upstream** (the merge script isn't in the repo), and only that clipped array is stored — not the raw continuous CGM. For a TSFM I'd ideally control the window. Two sub-questions:
- (a) Is the **upstream merge script / raw Dexcom export** available, so I can define the lookback myself (e.g. exactly 2 h before each test, per the K01 hypothesis)?
- (b) If not, what **lookback is baked into the current arrays**, so I can document it? (I'll default to "use the full clipped window" until told otherwise; I can also cap to the most-recent 30 readings = 2.5 h.)

### Q3 — Prices inversion + what the three scores measure  🟠 *affects modeling & interpretation*
Your `next_steps` notebook flags that **`prices_cognitive_score` is inverted (higher = worse)**. I currently **negate it** so all three targets read "higher = better" (toggle: `config.PRICES_IS_INVERTED`). Please confirm that's what you want. Also: the code never documents **what Grids / Symbols / Prices actually measure** (accuracy? reaction time? a composite?) or their units — knowing this helps me sanity-check predictions and choose sensible normalization. (Dr. Hassenstab's PARC/ARC docs, maybe?)

### Q4 — What does "success" look like?  ✅ *RESOLVED (2026-07)*
> **Resolved:** Liuyi said to **focus on our own TSFM approach and drop the comparison with Mack's feature-engineering arm.** So success = an honest, leakage-free evaluation of the raw-CGM→Chronos approach (Arm A + Arm B) for all three scores, plus the within-subject test of the baseline theory — reported on its own merits, whatever the answer. Mack's 43 features stay in the code only as an optional internal sanity-check. (We keep the "signal is likely weak, don't oversell" caveat — that's just honest.)

*(Original question, kept for history:)* Mack's 43-feature arm has negative R²/chance AUROC everywhere, and session-CV > LOPO says the signal is mostly subject-baseline. The open choice was between (a) a fair confirm/refute vs hand-crafted features, or (b) chasing a specific hypothesis (hypoglycemia windows / within-subject effects). Liuyi's answer: neither as a *comparison* — just develop and honestly evaluate our own approach (the within-subject test in (b) is still a valuable analysis to run).

### Q5 — Chronos family & checkpoint size  ✅ *DECIDED by us (2026-07), open to your override*
> **Decision:** standardize on **`amazon/chronos-bolt-small`** (frozen) as primary, + **`amazon/chronos-bolt-base`** as a GPU sensitivity check. Rationale: not "biggest = best" — on ~900 sessions with a baseline-dominated signal, larger embeddings mainly add overfitting risk (the checkpoint sweep will confirm empirically); Bolt over T5 for speed. Easy to change if you prefer otherwise.

*(Original question:)* The ICML paper studied Chronos-**T5**; Ben used Chronos-**Bolt**. My code supports both via `BaseChronosPipeline`. Larger = richer embeddings but more compute.

### Q6 — compute2 / lab-server environment (Task 1)  ✅ *RESOLVED (2026-07)*
> **Resolved:** The environment is **built and GPU-validated on `cpsl-mds`** — Miniforge + a prefix env on the SSD, a CUDA build of torch on an RTX 6000 Ada, and real Chronos running on the GPU (see `docs/05_SERVER_SETUP.md`). Task 1 is effectively done. *One thing to confirm with Liuyi:* if "compute2" in the original task list is a **different specific machine** than `cpsl-mds`, say so and I'll replicate the (now fully scripted) setup there.

### Q7 — Frozen vs fine-tuned encoder  🟢 *later*
Currently the encoder is **frozen** (zero-shot embeddings — the ICML approach). Worth trying **LoRA fine-tuning** later, or keep it frozen for the first comparison? (I'd only invest here if the frozen arm shows promise.)

---

*Resolved by your last round of answers (recorded so we don't re-litigate): the "two CSVs" were Ben's watch data; Task 3 uses the ICML repo, not our data; single channel; single target × 3 models; TSFM arm only; the ICML paper is the relevant Chronos reference.*

*Resolved at the 2026-07 meeting: **drop the Mack head-to-head — focus on our own TSFM approach** (Q4); **Task 1 / lab-server env is done** on `cpsl-mds` (Q6). Data is on the way after labmate Phil's correlation analysis of the 20 patients (Q1). Chronos version decided by us: **`chronos-bolt-small` primary, frozen**, + `bolt-base` sensitivity (Q5). Full current status: [`PLAIN_ENGLISH_SUMMARY.md`](../PLAIN_ENGLISH_SUMMARY.md).*
