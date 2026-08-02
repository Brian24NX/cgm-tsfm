# 12 · The exam drill

> ⚠️ **Numbers updated 2026-08-01.** The pooling bug described in [`15_FINDINGS_TO_REPORT.md`](15_FINDINGS_TO_REPORT.md) has been **fixed in the code and all results regenerated**. Current headline values under grouped cross-validation are **Arm A −0.114 / −0.052 / −0.012** and **Arm B −0.313 / −0.412 / −0.360** (grids / symbols / prices). Worked examples in this chapter may still quote the pre-fix figures (Arm A −0.089 / −0.056 / −0.012, Arm B −0.218 / −0.281 / −0.168) — the arithmetic and the teaching point are unaffected, but [`01_FACT_SHEET.md`](01_FACT_SHEET.md) and `results/` are authoritative for current values. **The conclusion did not change: everything is still at or below zero.**


> **How to use this.** Cover the answer. Say yours **out loud** — not in your head; speaking is the skill being tested. Then compare. If you were vague where the answer is specific, that one is not learned yet.
>
> Questions are marked **[HE ASKED THIS]** where Liuyi has literally asked it in writing, **[core]** for the ones any examiner would ask, **[hard]**, and **[trap]** where the intuitive answer is wrong.
>
> Target before the meeting: every **[HE ASKED THIS]** and every **[trap]**, cold.

---

## Tier 1 — Questions he has already asked in writing

These are not hypothetical. Get these perfect.

**Q1.1 [HE ASKED THIS] "What does this 'predict the average' baseline do?"**

> It's the dumbest possible predictor and we use it as the thing to beat. It ignores the glucose completely — it takes all the scores from the training children, computes one average, and predicts that same number every time. For Grids it would say 0.478 for every session.
>
> We need it because an error on its own means nothing. If I say "my error was 0.50," you can't tell if that's good. If I say "guessing the average also gives 0.50," you know my model learned nothing. That comparison is what R² measures: 0 means exactly as good as guessing the average.

**Q1.2 [HE ASKED THIS] "Do we have a good understanding of the dimension/shape format of the data input to the TSFM?"** *(asked twice — draw it, don't say it)*

> One session goes in as a single list of glucose numbers. Nothing else — no timestamps, no participant id, no second channel. Length varies from 3 to 288 readings, median 36.
>
> Chronos standardizes it, pads it to a multiple of 16, cuts it into blocks of 16 readings, and turns each block into 512 numbers. A 36-reading session becomes 3 blocks plus one summary token, so a 4-by-512 block. We average down the token direction to get 512 numbers for the session.
>
> Stack all 956 sessions and you get a 956-by-512 table. That plus a column of 956 scores is what the regression sees.

**Q1.3 [HE ASKED THIS] "Why is a uniform 2-hour window the cleanest test of the hypothesis?"** *(also asked twice, and never answered)*

> Because right now every session has a different amount of glucose data — 3 readings to 288. The stored list looks like everything since that child's previous test, so an overnight gap gives ~24 hours and two tests an hour apart give ~12 readings.
>
> That means the length itself carries information unrelated to glucose. A long list means a long gap, probably morning after sleep. So the model can't tell whether it's reacting to glucose shape or just to time of day.
>
> I can show it: the strongest correlation with the Grids score isn't a glucose statistic at all, it's the **number of readings**, r = −0.077. That's window length leaking in.
>
> If every session used the same span, only the glucose would vary. On whether we already have what we need — you're right these are real Dexcom readings. What I don't have is the uncut timeline with timestamps; the CSV is already cut per session, and I can't re-cut to 2 hours because the short ones don't contain 2 hours. Phil's script would tell us if re-cutting is possible.

**Q1.4 [HE ASKED THIS] "What is a token? A patch? An epoch?"**

> **Patch** — a block of 16 consecutive readings. Readings are 5 minutes apart, so one patch is 80 minutes of glucose. Non-overlapping.
> **Token** — one patch after it's been turned into 512 numbers. It's the unit the transformer processes. A 36-reading session becomes 4 tokens (3 patches + 1 summary token).
> **Epoch** — one complete pass through the training data. In Arm B one fold has ~612 training sessions fed 64 at a time, so one epoch is 10 weight updates, then we check the validation error. Up to 100 epochs, usually stopping earlier.

**Q1.5 [HE ASKED THIS] "Is it 19 or 20 participants, and how was the missing glucose recovered?"**

> I'm using 20 because there's a corrected Cohort 2 file that fills glucose which was blank in the original. In the original, 44 of 240 rows had no glucose and that wiped out one child entirely — which is why you had 19. The corrected file has 1 blank row, so all 6 Cohort 2 children survive.
>
> What I can't tell you is **how** it was filled, because I haven't seen the code. If it was read back from the raw stream, fine. If it was interpolated, I'm partly modelling invented numbers. I've asked Phil for that script.

**Q1.6 [HE ASKED THIS] "What does 'power' mean? What does 'sensitivity' mean?"**

> Both are about how big an effect this study could even notice. With 20 children and 5 folds of 4 children each, the fold-to-fold variation in R² alone is ±0.04 to ±0.12. So any true effect smaller than roughly that is invisible to us — it would be buried in the noise between folds.
>
> Put plainly: we can rule out a **large** relationship between glucose and these scores. We cannot rule out a small one.

**Q1.7 [HE ASKED THIS] "'Confound' — is that the right word?"**

> No, and you were right to correct me. A confounder is a *third* variable affecting both things you're relating. Time of day is a genuine confounder — it plausibly affects glucose through meals and sleep, and affects test scores through alertness. But the variable window length isn't a confounder; it's one measurement that isn't standardized across sessions. The right words are "inconsistent measurement."

**Q1.8 [HE ASKED THIS] "Why do you say 'null'?"**

> I shouldn't, and I've stopped. The accurate phrase is **lower accuracy** — the model didn't predict better than guessing the average. "Null" imports a statistical meaning nobody needs here.

---

## Tier 2 — The data and the numbers

**Q2.1 [core] How many participants and sessions, and where does the session count come from?**
> 20 participants — 14 in Cohort 1, 6 in Cohort 2. 956 sessions. Derived: 740 Cohort 1 rows + 240 updated Cohort 2 rows = 980 raw rows, minus 24 with an empty or unparseable glucose cell (23 + 1) = 956. No rows were dropped for being too short; the 3-reading minimum never bit.

**Q2.2 [core] What is one row?**
> One cognitive test session: one child, one moment, one set of three scores, plus the list of glucose readings before it.

**Q2.3 [core] Sampling rate, and how do you convert readings to time?**
> One reading every 5 minutes (Dexcom G6). So 12 readings per hour, 288 per day. Our median session is 36 readings = 3.0 hours.

**Q2.4 [core] The window length distribution?**
> Minimum 3 readings, median 36, maximum 288, mean 71.4, standard deviation 64.1. A 96-fold spread between shortest and longest. 14 sessions have under an hour.

**Q2.5 [core] Glucose statistics?**
> 68,206 readings. Mean 177.24 mg/dL, SD 80.17, median 157. Range exactly 40 to 400.

**Q2.6 [trap] Are there outliers in the glucose?**
> No — and that's the interesting part. Nothing falls outside 40–400 because those are the Dexcom G6 reporting limits. Instead there's **censoring**: 1,470 readings (2.16%) sit at exactly 400 meaning "≥400", and 99 (0.15%) at exactly 40 meaning "≤40". So the tops of big excursions are flattened, which matters for a model trying to read curve shape.

**Q2.7 [core] Time in range?**
> By reading: 2.31% below 70, 58.14% in 70–180, 39.54% above 180, 17.66% above 250. Clinical guidance targets over 70% in range, so this cohort is hyperglycemic a lot. Low readings are rare — which matters, because hypoglycemia is where a cognitive effect is most expected and we barely have any.

**Q2.8 [core] Sessions with an excursion?**
> Any reading below 70: 201 sessions (21%). Any above 250: 449 (47%). Entirely within 70–180: 143 (15%).

**Q2.9 [core] The three scores?**
> Grids n=951, mean 0.478, SD 0.500. Symbols n=920, mean 1.846, SD 0.560. Prices n=936, mean 41.59, SD 17.17. All three lower = better. Missing counts 5 / 36 / 20.

**Q2.10 [trap] Do the three targets use the same sessions?**
> No. Each target masks its own missing scores, so they use 951 / 920 / 936 sessions — and the folds get **re-split** afterwards, so Symbols' folds are not Grids' folds with rows removed. Per-target test-fold sizes differ: grids 188/191/192/190/190, symbols 170/190/188/187/185.

**Q2.11 [hard] What is the Grids score, physically?**
> The average straight-line distance, in grid cells, between where the child placed each of 6 objects and where it belonged, on a 5×5 grid. 0 means all six perfect. I verified this from the data: all 94 distinct Cohort 1 values are exactly the mean of 6 Euclidean distances on a 5×5 grid, and the item count is uniquely determined — 4, 5 or 7 items explain only 2–6%.

**Q2.12 [hard] Anything unusual about the Grids distribution?**
> Yes — **31.0% of sessions score exactly 0**, perfect recall of all six. That's a hard floor, and it breaks the smooth-continuous assumption behind least squares. It's a limitation to disclose, but it doesn't explain the result: Prices has no floor and is equally flat.

**Q2.13 [core] What is the Prices score?**
> Only 10 distinct values — 0, 10, up to 90 — in both cohorts. So it's a count out of 10 items as a percentage. Since lower is better it's presumably percent incorrect; I'd like that confirmed.

**Q2.14 [core] Sessions per participant?**
> Minimum 32, median 47.5, maximum 67, mean 47.8. No participant dominates.

**Q2.15 [hard] How much of the score variation is between children vs within?**
> Between children: Grids 0.153, Symbols 0.376, Prices 0.077. So **62% to 92% is within a child**, session to session. This corrects our own older notes, which said between-child differences dominate. They don't — and that makes the finding stronger, because session-to-session is exactly where a glucose effect would live, and we still can't predict it.

**Q2.16 [hard] What's the strongest raw relationship in the dataset?**
> r = −0.092, between the fraction of readings below 70 and the Symbols score. That's r² = 0.0085 — **0.85% of the variance**. Everything else is under 0.6%. Before any model, glucose and score are essentially unrelated in this data.

---

## Tier 3 — Machine learning basics

**Q3.1 [core] What is supervised learning?**
> Learning a function from examples where you're shown the answer. We have 956 sessions, each with 512 input numbers and one known score, and we're looking for a rule mapping input to score that also works on children we haven't seen.

**Q3.2 [core] Regression or classification, and why?**
> Regression — the output is a continuous number (a distance in grid cells, a time in seconds). Classification would be for a category.

**Q3.3 [core] Why can't you score a model on its training data?**
> Because a flexible model can memorize the training answers, which measures recall rather than understanding. It's like grading an exam using the questions the student already had the answer key for.

**Q3.4 [core] Difference between a parameter and a hyperparameter?**
> Parameters are learned from the data by fitting — Ridge's 512 coefficients and its intercept. Hyperparameters are set before fitting and control *how* fitting happens — Ridge's alpha, the number of PCA components, the learning rate, the batch size. Parameters are chosen by the algorithm; hyperparameters are chosen by us, on validation data.

**Q3.5 [core] What is overfitting, and show me an example from your work?**
> The model fits the training data's noise instead of a general pattern, so it looks good on data it's seen and fails on new children. Concrete example: plain LinearRegression on our 512 features gives R² ≈ −3, while Ridge on the identical features gives −0.089. Same data, same folds; the only difference is a penalty on coefficient size.

**Q3.6 [core] What's underfitting?**
> The opposite — the model is too constrained to capture even the real pattern. Ridge with alpha at 1000 approaches this: coefficients are squeezed so hard the model approaches just predicting the mean.

**Q3.7 [hard] What's the bias–variance tradeoff?**
> Prediction error splits into bias (systematic error from the model being too simple), variance (error from the model being over-sensitive to which particular training data it saw), and irreducible noise. More flexibility lowers bias and raises variance. Alpha is the dial: small alpha = low bias, high variance; large alpha = high bias, low variance. The inner cross-validation loop picks where to sit.

**Q3.8 [core] Why is 512 features on 956 sessions a problem?**
> Grouped folds leave ~764 training rows. Fitting 512 coefficients plus an intercept means 513 unknowns from 764 rows — about 1.5 rows per unknown. That's barely enough to pin them down, and since the Chronos dimensions are correlated, the solution is unstable and doesn't transfer.

**Q3.9 [hard] Name the regularization methods and say what they have in common.**
> L2/Ridge (penalize squared coefficients), L1/Lasso (penalize absolute coefficients, which zeroes some out), dropout (randomly disable activations during training), weight decay (L2 for networks), early stopping (limit how long optimization runs), and dimensionality reduction like PCA. All of them restrict what the model is allowed to do, trading a worse fit on training data for better transfer to new data.

**Q3.10 [core] You train on MSE but report R². Why the difference?**
> The **loss** is what the optimizer minimizes and it needs to be smooth and differentiable — squared error is. The **metric** is what humans interpret, and R² is on a scale where 0 means "same as guessing the average" and 1 means perfect, which MSE isn't. The inner tuning loop uses `neg_mean_squared_error` because sklearn maximizes, so it flips the sign.

**Q3.11 [hard] Can a model find a relationship that isn't in the data?**
> No. That's the ceiling on this whole project: the strongest raw relationship here is 0.85% of the variance. No representation or architecture can manufacture structure that isn't present. Which is why "we tried a bigger model" is not a real answer to a flat result.

---

## Tier 4 — Preprocessing and normalization

**Q4.1 [HE ASKED THIS / core] What is normalization and why must you process data before training?**
> Putting different measurements on a comparable scale so one doesn't dominate just because its numbers are bigger. Glucose is around 177; a slope might be 0.3. Three concrete reasons it matters here:
> - **Ridge** applies one penalty to every coefficient, so if features have very different sizes the same penalty shrinks them unevenly and the fit depends on measurement units.
> - **SVR** measures distances between points, so a large-numbered feature dominates the distance and the small ones stop mattering.
> - **Gradient descent** uses one step size in every direction, so mismatched scales make it zig-zag down a narrow valley.

**Q4.2 [core] What exactly is z-scoring?**
> Subtract the mean, divide by the standard deviation: z = (x − μ)/σ. Result is centred on 0 with a typical range about −2 to +2. Same shape, comparable scale. Worked: Grids has mean 0.4782 and SD 0.4999, so a raw score of 0.333 becomes (0.333 − 0.4782)/0.4999 = −0.29.

**Q4.3 [trap] Do you normalize the glucose before feeding Chronos?**
> We don't do anything ourselves — we pass raw mg/dL. **Chronos does it internally**, and specifically Chronos-Bolt applies instance normalization: it subtracts each series' own mean and divides by its own population standard deviation. Note our older docs say "mean-scales, divides by average magnitude" — that's the T5 variant, not the model we use. Bolt centres *and* scales.

**Q4.4 [hard] So how many normalizations are there in your pipeline?**
> Four, at four stages. (1) Inside Chronos — instance normalization of each glucose series, done by the library. (2) `StandardScaler` on the 512 features before the regressor, fit on the training fold only. (3) In Arm B, z-scoring the *target* using training-fold statistics, then converting predictions back to real units. (4) Inside the Arm B head, `LayerNorm` across the 512 values within each single session. Our docs never stated these in one place, which is why they look contradictory.

**Q4.5 [hard] What's the consequence of Chronos's internal normalization for your science?**
> Since both level and spread are divided out, the features are largely blind to how high or low the glucose actually was. That's awkward, because hypo and hyper are defined by absolute thresholds — 70, 180, 250 mg/dL. It shows up in measurement: the same features predict glucose *variability* at R² 0.462 but *mean glucose* at only 0.070. And `embed()` returns those two removed numbers, which our code was discarding. I added them back and it didn't change the result.

**Q4.6 [core] What is data leakage and how do you prevent it?**
> Leakage is when information from the test data influences training, which inflates the score. The classic version is fitting the scaler on all the data before splitting — then the test rows' mean and standard deviation are baked into the transform. We prevent it by putting the scaler, PCA and model in one sklearn `Pipeline`, so `.fit()` only ever sees the training fold. Plus an explicit `assert` on every fold that train and test participants are disjoint.

**Q4.7 [hard] LayerNorm vs BatchNorm vs StandardScaler?**

| | Axis | Statistics from | Can it leak? |
|---|---|---|---|
| StandardScaler | per feature | the training fold's rows | Yes, if fit wrongly |
| BatchNorm | per feature | the current batch (running averages at eval) | Yes, via batch composition |
| LayerNorm | **per sample, across features** | that one sample only | **No** — uses no dataset statistics |

**Q4.8 [trap] Our docs say "Chronos is NaN-tolerant so we need no gap-filling." Is that true of your pipeline?**
> It's true of Chronos and irrelevant to us. `data.py:47` does `arr[~np.isnan(arr)]` — we **delete** missing readings before Chronos sees them, so its mask mechanism is never exercised. And there's a real cost: deleting a reading breaks the even 5-minute spacing, and Chronos gets no timestamps, so it assumes evenly spaced points. A session with a 30-minute sensor gap becomes one where two adjacent numbers are half an hour apart and the model can't tell.

**Q4.9 [core] Does normalizing or negating the target change R²?**
> No. R², RMSE and MAE are unchanged by negating the target, and R² is unchanged by any shift or rescale, because both the residuals and the deviations from the mean transform the same way and then get squared. That's why keeping all three scores in "lower = better" orientation doesn't alter a single reported number — it only affects interpretation.

---

## Tier 5 — Evaluation

**Q5.1 [core] What is R²?**
> 1 − (sum of squared errors) / (sum of squared deviations from the mean). 1 is perfect. 0 means exactly as good as predicting the average. Negative means worse than that.

**Q5.2 [core] Work out an R² for me.**
> Say true values are [0, 0.333, 0.5, 1.0, 2.39], mean 0.8446, so the total sum of squares is 3.5062. If predictions are off by +0.2 each, the error sum of squares is 5 × 0.04 = 0.2000, so R² = 1 − 0.2/3.5062 = **+0.943**. If predictions are [1.5, 1.2, 0.2, 0.1, 0.3], the error sum is 8.2698, so R² = 1 − 8.2698/3.5062 = **−1.359**.

**Q5.3 [trap] Your R² is negative. Doesn't that mean the model is broken?**
> No, and this is the most important nuance in our results. **The mean-predictor itself scores negative under our folds**: −0.078 on Grids, −0.050 on Symbols, −0.004 on Prices. Our model gets −0.089 / −0.056 / −0.012, which is within 0.01 of the trivial baseline. That's a *tie*, not a collapse.
>
> The reason the baseline is negative: R² measures each test fold against *that fold's* own average, but the model was trained on the *training* fold's average. With only 4 children held out, those differ. Fold 1 exactly: train mean 0.5193, test mean 0.3117, offset 0.2076, test variance 0.1721, giving R² = −0.2076²/0.1721 = −0.2503, matching the measured value. So under leave-participants-out with 4 per fold, slightly-below-zero *is* the neutral point.

**Q5.4 [core] Why cross-validation instead of one split?**
> A single split depends on which rows happened to land where. K-fold rotates so every row is tested exactly once, and gives you a spread across folds as well as an average.

**Q5.5 [core] Why group by participant?**
> Because otherwise the same child appears in training and test. Each child has 32–67 sessions, so the model can learn "this particular child usually scores about 1.8" and score well without knowing anything about glucose. Grouping holds out **whole children**, so the question becomes "predict a child you've never seen."

**Q5.6 [core] Describe your folds concretely.**
> 5 folds, 20 participants, so exactly 4 participants held out each time. Test sessions per fold: 189, 193, 191, 191, 192. Training: 767, 763, 765, 765, 764. Plus an assert on every fold that the participant sets are disjoint.

**Q5.7 [core] What is nested cross-validation and why?**
> Two loops. The outer 5 folds give the honest performance estimate. Inside each outer training fold, an inner 3-fold split chooses the hyperparameters — Ridge's alpha, SVR's C and gamma. If you chose alpha by looking at the test fold, the test fold would have influenced the model and the estimate would be optimistic.

**Q5.8 [trap] You report "mean ± standard deviation across 5 folds." Is that a confidence interval?**
> No. It's the spread across 5 folds, and with only 4 participants per fold it's dominated by which children landed where. Grids is ±0.123, which is larger than the effect we'd be looking for. A confidence interval would need a proper accounting of the participant-level uncertainty.

**Q5.9 [trap] You set seed 42. Does that determine your folds?**
> No — and this catches people. sklearn's `GroupKFold` defaults to `shuffle=False`, so the outer folds are deterministic regardless of the seed. The seed does affect the `GroupShuffleSplit` validation split in Arm B and the models' internal randomness.

**Q5.10 [hard] How many model fits per Arm A run?**
> 185 per target: 5 for the baseline, 80 for Ridge (5 alphas × 3 inner folds = 15, plus 1 refit = 16, × 5 outer folds), 95 for SVR (6 settings × 3 = 18, plus 1 = 19, × 5), and 5 for plain Linear. Times 3 targets = **555**.

**Q5.11 [hard] You tried many settings. Aren't you at risk of a lucky result?**
> In general yes — with 3 targets, 4 models, 5 model sizes, 4 window lengths, 2 poolings, 6 PCA values, 3 target normalizations and 5 glucose subgroups, something eventually looks good by chance. But **every single result came out at or below zero**, so there's no lucky positive to worry about. The risk is real for any *future* hunt through subgroups, which is why those should be pre-committed or the threshold corrected.

**Q5.12 [hard] What would convince you there IS a relationship?**
> Four things, committed to in advance: a positive R² on held-out participants; the same sign in most folds rather than one lucky fold; clearly outside the range we get from scrambled scores; and larger than the fold-to-fold spread. None of those happened.

---

## Tier 6 — Deep learning

**Q6.1 [HE ASKED THIS] What is an epoch?**
> One complete pass through the training data. Ours: about 612 training sessions, batch size 64, so ceil(612/64) = 10 batches. One epoch is those 10 weight updates. Then validation runs once, and the early-stopping check happens.

**Q6.2 [core] Epoch vs step vs batch?**
> A batch is a group of examples processed together (64 for us). A step or iteration is one weight update, which happens after each batch. An epoch is a full pass, which for us is 10 steps. Max 100 epochs = at most 1,000 steps.

**Q6.3 [HE ASKED THIS] What is early stopping?**
> While training, we watch the error on a held-out validation set. Training error keeps falling forever because the model starts memorizing. Validation error falls at first, then turns back up — that's where memorizing begins. We stop there. Concretely: if validation loss hasn't improved for 8 epochs in a row, stop.

**Q6.4 [trap] Patience is 8 — eight what? And which weights get tested?**
> Eight **epochs**, because validation runs once per epoch. And here's what I'd volunteer: **the best weights are not restored.** `enable_checkpointing=False` and there's no ModelCheckpoint callback, so Lightning restores nothing. The weights we test are the ones from the moment training stopped — up to 8 epochs past the best validation loss. That makes Arm B look slightly worse than intended. One-callback fix.

**Q6.5 [hard] Why is early stopping a form of regularization?**
> Because it limits how far optimization can travel. A network with enough capacity will eventually fit the training noise, but it needs many steps to get there. Cutting training short keeps the weights nearer their small initial values, which limits the effective complexity — the same goal as a penalty term, achieved by limiting time rather than by adding a cost.

**Q6.6 [core] Draw the two curves.**
```
 loss
  │╲
  │ ╲___                     validation  ← rises again: memorizing starts
  │     ╲___        ______/
  │          ╲____/  ↑
  │  ╲___            stop here
  │      ╲______     training  ← keeps falling forever
  │             ╲_________
  └──────────────────────────────▶ epochs
```

**Q6.7 [core] What is a gradient, and what is backpropagation?**
> A gradient is the derivative of the loss with respect to each weight — which direction and how steeply the loss changes if you nudge that weight. Backpropagation is the efficient way to compute all of them: run forward to get the loss, then apply the chain rule backwards through the layers, reusing intermediate results, so you get every gradient in roughly the cost of one more forward pass.

**Q6.8 [core] What is gradient descent, and what's the learning rate?**
> Each weight moves against its gradient: w ← w − lr × gradient. The learning rate is the step size. Too small and training crawls; too large and it overshoots and can diverge. Ours is 1e-3, the conventional default for Adam-family optimizers.

**Q6.9 [core] What optimizer, and what does it do differently from plain gradient descent?**
> AdamW. It keeps running averages of each weight's gradient and of the gradient squared, and uses them to give every weight its own effective step size — so weights with consistently small gradients still move. The "W" is decoupled weight decay: the shrinkage is applied directly to the weights rather than added into the loss, which behaves better with adaptive step sizes. Ours is lr 1e-3, weight decay 0.01.

**Q6.10 [core] What does dropout do, and when is it active?**
> During training it randomly zeroes a fraction of the values passing through — 10% for us — so the network can't rely on any single one. **At evaluation it's completely off** and predictions are deterministic. PyTorch switches it via `model.train()` / `model.eval()`; Lightning calls those automatically. That on/off distinction is the classic exam point.

**Q6.11 [core] Why does the head need an activation function?**
> Without one, consecutive linear layers collapse: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂), which is just one linear layer. So 512→256→1 with no activation is mathematically identical to 512→1 and the hidden layer is pointless. GELU makes it genuinely non-linear.

**Q6.12 [trap] Why is there no activation on the output layer?**
> Because we're predicting a continuous number that can take any value, and any activation would restrict the range. This is also the real difference from Ben's classifier we adapted: his ended in softmax with a cross-entropy loss; ours ends in identity with squared error. Our docs describe the change as just the output *width*, which undersells it — the loss changed too.

**Q6.13 [core] What's the loss, and on what scale?**
> Mean squared error, computed on the **z-scored** target so it's on an O(1) scale for any of the three scores. Predictions are converted back to real units before the reported metrics. So the stopping criterion and the reported R² live on different scales — worth knowing.

**Q6.14 [core] Frozen vs fine-tuned vs from scratch — which and why?**
> Frozen. Chronos's 47,718,016 parameters never change; only the head's 132,609 train. Fine-tuning 47.7 million parameters on 20 participants would be indefensible — you'd memorize instantly. Training from scratch is even more hopeless: the whole reason we use Chronos is that it already learned what time series look like from data we don't have.

---

## Tier 7 — Transformers and Chronos

**Q7.1 [HE ASKED THIS] Why do you need a transformer for this project?**
> The honest primary reason is **pretraining**, not architecture. With 20 children we can't train any sequence model from scratch, so we need one that already learned what time series look like from enormous amounts of other data. Every strong publicly available pretrained time-series model happens to be a transformer, so choosing "pretrained" effectively chose "transformer."
>
> The architecture does help too: a transformer can relate any reading directly to any other in one step, so a dip 90 minutes before the test can interact with the final reading immediately. An LSTM would have to carry that through every intermediate step, where it tends to fade.

**Q7.2 [core] Why not an LSTM?**
> An LSTM reads one point at a time carrying a hidden state, so information from early in the series must survive many sequential updates and gradients tend to vanish over long spans. It also can't be parallelized over time. But the decisive reason is simpler: there's no strong pretrained LSTM for time series, and pretraining is what we actually need.

**Q7.3 [core] What is self-attention, in words?**
> Each position forms a query about what it needs, every position advertises a key describing what it has, and the match between queries and keys becomes weights. The output at each position is a weighted blend of all the values. Formally softmax(QKᵀ/√d_k)V.

**Q7.4 [hard] Why divide by √d_k?**
> Because dot products of longer vectors grow with dimension, and large scores push softmax towards nearly one-hot, where gradients almost vanish. Dividing by √d_k keeps the scores in a range where softmax stays soft and trainable.

**Q7.5 [hard] Attention doesn't care about order. So how does the model know the sequence?**
> Attention on its own is permutation-invariant — shuffle the tokens and you'd get the same set of outputs. T5, which Chronos-Bolt is built on, uses **relative position bias**: a learned number added to each attention score depending on how far apart the two positions are, bucketed into 32 buckets up to distance 128, shared across layers. Not the sinusoidal absolute encodings of the original transformer paper. The practical consequence for us: the model knows how far apart two patches are but not the clock time — and it assumes even spacing, which our NaN-dropping breaks.

**Q7.6 [core] How many heads, and why more than one?**
> 8 heads of 64 dimensions each, totalling 512. Multiple heads let the model attend to different things simultaneously — like several people reading the same curve looking for different features — instead of forcing one averaged attention pattern.

**Q7.7 [core] What's in one encoder block?**
> A self-attention sub-layer then a position-wise feed-forward sub-layer (Linear 512→2048, ReLU, Linear 2048→512), each wrapped with a residual connection and layer normalization. Six of these stacked.

**Q7.8 [trap] Does Chronos turn glucose values into words from a vocabulary?**
> Not the version we use. That's **Chronos-T5**, which bins values into 4096 discrete tokens. **Chronos-Bolt** keeps values continuous and groups them into patches of 16; its `vocab_size` is literally 2, holding only a decoder-start marker and one special token. Our own `PLAIN_ENGLISH_SUMMARY.md` describes the T5 mechanism by mistake — I need to fix that.

**Q7.9 [core] Give me the shape formula.**
> `embed()` returns `(B, ceil(min(T, 2048)/16) + 1, 512)`. So T=3 → 2 tokens; T=36 → 4; T=288 → 19. It's `ceil`, verified against exact multiples.

**Q7.10 [hard] What is the "+1"?**
> The `[REG]` token — a single learned 512-number vector appended at the **end** of the sequence, always attended to. Its input is identical for every series, but its output becomes a summary of the whole series via self-attention, exactly like BERT's `[CLS]`. It sits at index −1.

**Q7.11 [trap] Your config has `pooling="last"`. Last what?**
> Not the last time step — the **`[REG]` token**, because it's appended last and `emb[:, -1, :]` selects index −1. Our docs call it "last-token pooling," which reads as "most recent patch." It isn't, and that's worth correcting.

**Q7.12 [hard] Which way is the padding, and why?**
> **Left.** When T isn't a multiple of 16, exactly 16 − (T mod 16) NaN values are prepended. That keeps the most recent reading at the right-hand edge of the final patch — Chronos forecasts forward from the end of the series, so the end must stay aligned to a patch boundary. The ragged edge goes to the oldest data.

**Q7.13 [hard] How big is the model, and how much of it do you use?**
> chronos-bolt-small is 47,718,016 parameters: 6 encoder layers, 6 decoder layers, 8 heads, d_model 512, d_ff 2048. `.embed()` runs only the input patch embedding plus the encoder = **20,015,872 parameters, 41.9%**. The decoder and output head (27,703,168) are loaded but never executed on our path.

**Q7.14 [trap] Is it encoder-only or encoder-decoder?**
> Encoder-decoder in the T5 style, but the decoder runs exactly **one** step — nothing autoregressive happens. It's best described as a learned single-query pooler. And we never run it at all; `.embed()` stops at the encoder's last hidden state.

**Q7.15 [trap] If you never forecast, why does `prediction_length=64` exist?**
> It's part of the pretrained checkpoint — it sets the output head's width, 9 quantiles × 64 horizons = 576. It has no effect on our embeddings because we stop at the encoder. It's inert configuration on our path.

**Q7.16 [hard] Did you try a bigger model?**
> Yes, four sizes across a 24× parameter range. Mean R²: tiny −0.049, mini −0.052, small −0.052, **base −0.063 — the worst**. Plus t5-small at −0.057. When the limit is the data rather than the model, extra capacity only gives the downstream regressor more columns to overfit on ~764 rows.

**Q7.17 [core] Chronos is univariate. What would adding insulin take?**
> Bolt has no channel axis at all, so you can't simply add a second input. Options: encode each channel separately and concatenate the embeddings, or switch to a genuinely multivariate model like Moirai. But you've gated this — no extra channels until the glucose-only pipeline is understood and exhausted.

---

## Tier 8 — The two arms

**Q8.1 [HE ASKED THIS] What is the MLP head and how does it work?**
> A small trainable network on top of the frozen Chronos model. `LayerNorm(512) → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→1)`. It normalizes the 512 numbers, maps them to 256 through a linear layer, bends them non-linearly, randomly drops 10% during training, then maps down to one predicted score. **132,609 trainable parameters.** Chronos's 47.7 million stay frozen, so 0.28% of the model trains.

**Q8.2 [core] Count those parameters for me.**
> LayerNorm: 512 scales + 512 shifts = 1,024. Linear(512→256): 512×256 = 131,072 weights + 256 biases = 131,328. Linear(256→1): 256 + 1 = 257. Total **132,609**. General rule: Linear(in→out) has in×out + out.

**Q8.3 [hard] Why did Arm B do worst?**
> Four reasons, in order. **Capacity**: 132,609 parameters on ~612 training rows is 217 parameters per example, versus Ridge's 513 total. **No signal**: the strongest raw relationship is 0.85% of the variance, so extra flexibility can only fit noise. **Weaker regularization, chosen worse**: Ridge searches 5 alphas per fold with a proper inner loop; Arm B has one untuned weight decay and an early-stopping decision based on ~3 children. **No closed form**: Ridge solves exactly; Arm B must find its way there in ≤1,000 steps.
>
> Worth adding: an MLP with GELU can represent Ridge's solution exactly — it's strictly more expressive. So this isn't a limit on what it *could* do; it's a failure to find and hold the good solution.

**Q8.4 [trap] "Arm A tunes hyperparameters, Arm B doesn't." Right?**
> Not quite, two ways. Arm B chooses its stopping epoch using validation data, which is tuning. And AdamW's `weight_decay=0.01` means Arm B **is** L2-regularized — the same family as Ridge's alpha. The difference is Ridge searches five values while Arm B uses one that nobody tuned.

**Q8.5 [trap] "Only the predictor differs between the arms." Right?**
> Also not quite. Arm A applies StandardScaler — per feature, across training rows. Arm B applies LayerNorm — per sample, across the 512 features — plus z-scores the target. Different transforms on different axes. They share the encoder and the fold structure; the preprocessing differs slightly.

**Q8.6 [core] Did you fine-tune Chronos in Arm B?**
> No. Chronos is frozen, and in the actual runs it isn't even executed during training — embeddings are computed once upstream, cached, and a pass-through stand-in replaces the encoder inside the network. So Arm B trains a small network on a fixed table of numbers.

**Q8.7 [core] Why did you use such simple models in Arm A?**
> Because with 20 participants and essentially no raw relationship, a simple regularized model is the defensible choice, and a complicated one would be hard to justify. The evidence backs it: the more flexible option, Arm B, did worse, and plain unregularized least squares was catastrophic at −3.

**Q8.8 [trap] Did you try gradient boosting?**
> No. XGBoost is listed in the code but the import is wrapped in try/except and the package **isn't installed** in this environment, so it silently never ran. I'd rather tell you that than have you find it in the code.

---

## Tier 9 — Rigor

**Q9.1 [core] How do you know the low accuracy is real and not a broken pipeline?**
> Two checks. **Scramble the scores** 200 times, so any real relationship is destroyed, and re-run everything: the real result lands inside the scrambled pile every time, p = 1.000, 1.000, 0.995. **Ask the same setup to predict something we know is in the data** — how variable the glucose was — and it does that well, R² 0.462. So the machinery extracts real information when there is some; the glucose-to-score relationship just isn't in this data.

**Q9.2 [hard] Why is the scrambled average −0.05 rather than 0?**
> Same reason the mean-predictor's R² is negative: R² is measured against each test fold's own average, and with only 4 children per fold that differs from the training average. So a useless model scores slightly below zero by construction. It means the correct comparison point isn't 0 — it's about −0.05.

**Q9.3 [trap] Your real R² is −0.106 but the results table says −0.089. Which is it?**
> Both, from different models. The shuffle test deliberately uses a **fixed** model — StandardScaler → PCA(32) → Ridge(alpha=10), no tuning — so the 200 repeats are comparable and affordable. The headline table tunes alpha per fold via the inner loop, which does slightly better. Same data, different protocol.

**Q9.4 [trap] Did the known-signal check pass?**
> Not cleanly, and I won't claim it did. Glucose variability came back at 0.462, but mean glucose only 0.070 and time-above-180 only 0.075. The code only prints "passed" if all three exceed 0.5, so that message never printed and isn't in the results file. The honest reading: variability is recovered well, absolute level is not — and we know exactly why, because Chronos subtracts each series' mean before the encoder sees it.

**Q9.5 [hard] Is 20 participants enough to conclude anything?**
> Enough to rule out a large relationship; not enough to rule out a small one. The fold-to-fold spread alone is ±0.04 to ±0.12 in R², so anything smaller than that is invisible. And the effective sample size for a claim about children is closer to 20 than 956, because sessions from one child aren't independent. That's exactly why we group by participant.

**Q9.6 [trap] You have 956 sessions. Isn't that a decent sample?**
> For sessions, yes. For a claim about children, no — sessions from one child are correlated, so the effective sample size is much closer to 20. Conflating them would overstate the evidence. This is also why I say "sessions" and not "patients" when I mean sessions.

**Q9.7 [hard] Where did you look for a relationship and not find one?**
> Five axes. Model size: four Chronos sizes plus T5, all flat, biggest worst. Window length: capping to 2, 2.5, 3 hours is *worse* than using everything. Pooling: mean beats last, marginally. Dimensionality: PCA 8–32 improves mean R² from −0.052 to −0.045, still negative. Glucose regimes: restricting to sessions with a reading below 70 (201 sessions, 19 participants) or above 250 (449 sessions, 20 participants) gives −0.19 to −0.08. And removing each child's own average — the within-child test — moves R² to −0.018 / −0.022 / −0.002, towards zero but never above it.

**Q9.8 [core] What's the difference between "we found no effect" and "there is no effect"?**
> The first is what we can say. The second we can't. We can say that in this data, with 20 children and this model, we could not predict the scores better than guessing the average, and that a large effect would have shown up. A small effect could be there and be invisible to us.

---

## Tier 10 — The killers

**Q10.1 [trap] Are your embeddings correct?**

This is the one to volunteer rather than wait for.

> They weren't, and I found it this week. Our pooling step averaged over all token positions, but Chronos left-pads short series out to the longest one in the batch — so a short session's features got diluted by padding, and depended on which other sessions happened to share its batch. Measured: a 13-reading session alone versus in a batch with a 288-reading session gives vectors with cosine similarity 0.22. Across our 956 sessions, 861 (90%) had cosine similarity below 0.9 against correctly-pooled features, median 0.37.
>
> The `[REG]` token is immune — bit-identical across batchings.
>
> I re-ran the whole evaluation with corrected features. **The conclusion didn't change**: still at or below zero on all three scores. If anything the bug was flattering the results slightly — corrected Grids goes from −0.089 to −0.114.

**Q10.2 [hard] Is your 512-number representation actually adding anything?**
> No, and I measured it. Under the same protocol: Chronos's 512 features give −0.114 / −0.093 / −0.069, while **one number — the average glucose — gives −0.075 / −0.050 / −0.004**. The single number does as well or better. Nothing beats guessing the average either way, but it tells us the elaborate representation isn't buying anything here. Which fits: the raw relationship is 0.85% of the variance at best. Fewer features means less room to overfit, which is why the simpler ones land closer to zero.

**Q10.3 [hard] Anything else you found wrong in your own work?**
> Three things. Our docs described Chronos-T5's value-binning as if it were Bolt's mechanism — it isn't, Bolt patches continuous values. Our docs said between-participant differences dominate the score variance — measured, it's 62–92% *within* participant. And PCA(32) was silently discarding features I'd deliberately added: predicting mean glucose from [mean, SD] gives R² 1.0000 with no PCA and 0.0853 with PCA(32), because PCA keeps high-variance directions, not useful ones.

**Q10.4 [hard] What's the weakest part of this pipeline?**
> The unequal input lengths. Every session gets a different amount of glucose — 3 to 288 readings — and the length itself carries time-of-day information. It's the one thing I can point to that's structurally wrong rather than just underpowered, and fixing it needs the uncut timeline.

**Q10.5 [hard] What would you do differently if starting over?**
> Fix the input lengths first, before any modelling. Keep the missing readings so Chronos's mask mechanism can do its job instead of breaking the even spacing. Keep the level and amplitude numbers that Chronos hands back. And start with the one-number-versus-512 comparison, because it establishes the ceiling immediately and would have told us early where the limit is.

**Q10.6 [hard] Should we keep going on this?**
> The glucose-only path with this model is close to exhausted — I've swept size, window, pooling, dimensionality and glucose regimes, and everything sits at or below zero. There are a few cheap things left: correct the pooling, keep the level numbers, and fix the unequal windows if Phil's script allows it. After that I think the honest answer is that this data doesn't contain the effect, and the question becomes how to write that up.

**Q10.7 [trap] Did the AI write this?**
> I used Claude to learn the parts I didn't know and to check my numbers. Everything on these slides I can derive at the whiteboard — ask me anything and I'll work it through. I also recomputed every number from the CSVs rather than trusting the older documents, and that's how I found the three errors I just described.

---

## The "I don't know" list — say these plainly

Knowing your boundaries is part of knowing the material. These are genuinely open:

1. **How the Cohort 2 glucose was backfilled.** Haven't seen Phil's code. Could be recovered from the raw stream, could be interpolated. It matters.
2. **Whether the scores are age- or practice-adjusted.** Asked, not answered. With ~5 tests a day for 10 days, practice effects are plausible and would be a session-order trend competing with glucose.
3. **The exact definition of the Prices score.** I know it's a count out of 10 as a percentage; I infer percent-incorrect from "lower is better."
4. **Whether the uncut CGM timeline with timestamps exists anywhere accessible.**
5. **Whether more participants are coming.** You said not guaranteed and not to hope for it.

---

## The night before

Ten minutes, out loud, no notes:

1. State the ten numbers from `01_FACT_SHEET.md` §K.
2. Draw the pipeline with shapes at every stage.
3. Define: epoch, batch, step, patch, token, early stopping, normalization, the mean baseline.
4. Explain why R² is negative **and** why the baseline is negative too.
5. Explain the padding bug and what you measured.
6. Say the three summary sentences from `00_SAY_IT_SIMPLY.md` §6.
7. Check you haven't used a single banned word.

If any of those stumbles, that's tonight's work.
