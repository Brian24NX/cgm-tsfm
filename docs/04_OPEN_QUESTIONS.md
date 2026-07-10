# Open Questions for Liuyi (v2 — post-answers)

*The open questions raised by actually building the pipeline and reading `diabetes-fitbit`, ordered by how much they block progress.*

---

## Current questions — after the first real run (sent to Liuyi 2026-07-08)

*These extend the earlier round (Q1–Q7 below), now that the env is up, the 20-patient data is in, and the first real run is done — an **honest null** (no representation beats the mean baseline; no within-subject effect). Each notes our current default so Liuyi can confirm or redirect. Most-needed steer: **A, B, D.***

### A — Direction, given the null result  🔴
- **A1.** Is a rigorous **null + characterization** the intended deliverable, or should we actively hunt for signal (hypoglycemia/hyperglycemia windows, subgroups)? *Default: treat the null as a real finding + a few targeted follow-ups — not force a positive.*
- **A2.** Is **N=20 the sample for this phase**, or is more coming (toward the K01's ~92)? Decides if this null is interim or a stopping point. *Default: report on 20; auto-re-run as more arrive.*

### B — The pre-test glucose window  🔴 *(new finding from the real data; extends Q2)*
Real `Glucose_Before_Test` windows span **~15 min–24 h (3–288 readings), median ~3 h** — with the `Prev_Test_Time` column, this looks like **"all glucose since the previous test," not a fixed lookback** (overnight gaps → 24 h windows). That inconsistency is a confound and **could itself be dampening any signal.**
- **B1.** Is that interpretation correct?
- **B2.** Can we get the **raw continuous Dexcom stream / the upstream merge script** to define a uniform window (e.g. exactly 2 h pre-test)? *Default: full clipped window + a 2.5 h sensitivity check until then.*

### C — What the three scores measure  🟠 *(extends Q3)*
- **C1.** Units + definitions of Grids/Symbols/Prices? Prices looks like a **latency/error** measure (inverted, large scale). Raw scores, or already **age-/practice-adjusted**? *Default: WM / processing-speed / latency reading; Prices negated so higher = better.*
- **C2.** Tests are 5×/day for ~10 days (+ timestamps + `Prev_Test_Time`) → **practice & time-of-day are plausible confounds.** Control for time-of-day / test order / inter-test interval? *Default: glucose-only for now; can add these as covariates.*

### D — Modeling framing  🟠 *(top priority)*
- **D1.** Keep the **new-subject prediction** bar (subject-grouped CV), or add a **within-person association / mixed-effects** analysis — closer to "when a given child's glucose swings, does their performance move?" *Default: add the association analysis alongside the ML prediction.*
- **D2.** Stay **glucose-only** (per the 1-channel scope), or eventually fold in covariates (insulin / meals / activity)? *Default: glucose-only for now.*

### E — Provenance / confirmations  🟢
- **E1.** Confirm **`Updated_Cohort2` is the final file** (it yields the full 20; the original drops one → 19), and how was the missing glucose backfilled (raw re-export vs interpolation)? *(Partly for Phil.)*
- **E2.** Confirm we should **negate `prices_cognitive_score`** so "higher = better" for all three.

### F — Deliverable & timeline  🟢
- **F1.** What output do you want (short writeup / figures / a specific results table), and by when / for what (progress update / K01 aim / paper)? So I package it right.

---

### Q1 — Access to the real merged CSVs  ✅ *RECEIVED (2026-07)*
> **Received** from Phil: 20 patients (14 Cohort1 + 6 Cohort2), now in `data/Merged_glucose_data/` (gitignored). Use **`Updated_Cohort2`** (the original drops one subject → 19). The first real run is done (honest null — see `results/README.md`). Provenance follow-up (how the glucose was backfilled) is item **E1** above.

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

*Resolved at the 2026-07 meeting: **drop the Mack head-to-head — focus on our own TSFM approach** (Q4); **Task 1 / lab-server env is done** on `cpsl-mds` (Q6). Data received from Phil — 20 patients, using `Updated_Cohort2` (Q1); first real run done = an honest **null result** (see `results/README.md`). Chronos version decided by us: **`chronos-bolt-small` primary, frozen**, + `bolt-base` sensitivity (Q5). Full current status: [`PLAIN_ENGLISH_SUMMARY.md`](../PLAIN_ENGLISH_SUMMARY.md).*
