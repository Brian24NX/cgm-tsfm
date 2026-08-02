# 00 · Say It Simply — the translation layer

> **Read this file first, and read it again the morning of the meeting.**
>
> The technical chapters teach you the machinery. This file teaches you how to *talk* about it. That is a separate skill, and right now it is the one you're being marked on.

---

## 1. What actually went wrong last time

It is worth being precise about the criticism, because it was not only "you don't know ML." Liuyi wrote two separate complaints, and they need two different fixes.

**Complaint 1 — the knowledge gap.** You couldn't explain early stopping, epochs, normalization, why a transformer, how the MLP head works. Fix: chapters 02–11 of this bootcamp.

**Complaint 2 — and this is the one people miss.** In his written comments he said:

> *"I feel your reliance on AI is beyond the extent I was expecting. First, I would think **asking questions using your own language/words** would make it easier to understand. At least, you could **edit the AI's text into your language** so that I can spend less time understanding your points. Second, **whenever AI produce any logic/code, you would be able to understand before present it to me.** If you don't first understand, your direct presentation of AI's logic/code seems **equal to asking me to make sure AI is doing right thing.**"*

Read that last sentence again. His objection is not that you used a tool. It is that he felt he was being asked to **audit** output that you hadn't audited yourself. From his side of the table, unexplained polished text is not evidence of work — it is a request that he do the checking.

**So there is a trap in front of you right now.** If you take these bootcamp files and read polished sentences out of them at the meeting, you will trigger exactly the same reaction, even though you'll be *correct*. The fluency itself is what he flagged.

**The fix is mechanical, and it works:**

1. Learn the mechanism from these files until you could redraw it on a whiteboard.
2. Then **close the file** and write the explanation again in your own voice — shorter, rougher, with your own comparison.
3. Bring *that* version to the meeting.
4. When you don't know something, say *"I don't know, I'll check"* — that sentence buys you more credibility than any paragraph. He explicitly values it: his whole complaint is about unverified claims.

A rough sentence you actually own beats a perfect sentence you're reciting. He can tell the difference instantly — that's what happened last time.

---

## 2. The banned-words list

These are phrases Liuyi has specifically objected to or said he doesn't use. The left column is what's currently in our documents. **The right column is what you say out loud.**

| ❌ Don't say | ✅ Say instead | Why |
|---|---|---|
| "a null result" | **"lower accuracy"**, or "it didn't predict better than guessing the average" | His standing instruction: *"to avoid confusion, let's always say 'lower accuracy' instead of 'null' in the future."* |
| "no signal beyond chance" | **"we got the same result from scrambled data as from the real data"** | He flagged this exact phrase as not being language he uses. |
| "positive control" | **"a check where we ask the model to predict something we already know is in the data"** | He flagged this exact phrase. |
| "statistical power" | **"how big an effect this study is even capable of noticing"** | He asked directly: *"What does 'Power note' mean?"* |
| "sensitivity" | **"the smallest real effect we'd have been able to see"** | He asked directly: *"What does 'sensitivity' mean?"* |
| "pre-test window" | **"the CGM readings before the test"**, or just **"the session"** | He asked whether this was just a new name for "CGM sessions." |
| "the one lever not yet pulled" | **"the one thing we haven't tried yet"** | He named "lever" as invented vocabulary. |
| "permutation test" | **"the shuffle test"** (say the real name once, then use this) | Not banned, but nobody needs the Latin. |
| "patients" (when you mean sessions) | **"sessions"** | He corrected this directly: *"Should this 'patients' be 'sessions'?"* |
| "confound" (loosely) | see §4 below — he has a specific definition and he's right | He corrected your usage once and endorsed it once. |
| "head-to-head", "drop the head-to-head" | **"the comparison against the older feature-based method"** | He asked what "drop the head-to-head" even meant. |
| "oracle centering" | **"we used each child's own average, which we wouldn't have for a new child"** | Jargon nobody outside our docs uses. |
| "the null hypothesis" | **"the assumption that glucose has no relationship to the score"** | |
| "TSFM" (unexplained) | **"a big pretrained time-series model"** — expand it the first time | |
| "embedding" (unexplained) | **"512 numbers that summarise the glucose curve"** | Fine to use *after* you've defined it once. |

His overall instruction, verbatim: *"Unless necessary, in all docs/slides/meetings/code, can we try to stop introducing new terms… let's use commonly used terms to minimize potential confusion and communication costs."*

**Practical rule:** you may use a technical term *once you have defined it in one plain sentence*. Never use one you haven't defined.

---

## 3. The five explanations he has explicitly asked for

These are not hypothetical questions. He asked these in writing and they have not been answered to his satisfaction. **Learn these five cold.** They are the most likely opening questions.

### 3.1 "What does this 'predict the average' baseline do?"

He wrote: *"We did not talk about 'predict the average' baseline before, so I cannot understand what this 'predict the average' baseline does."*

**Say this:**

> "It's the dumbest possible predictor, and we use it as the thing to beat. It ignores the glucose data completely. It just takes all the scores from the children in the training set, computes one average, and predicts that same number for every session of every new child.
>
> So for the Grids score it would predict 0.478 every single time, no matter what the glucose looked like.
>
> We need it because a prediction error on its own means nothing. If I tell you 'my error was 0.50', you can't tell whether that's good. But if I tell you 'guessing the average gives an error of 0.50 and my model also gives 0.50', then you know my model learned nothing useful. That comparison is what R² measures — R² of 0 means exactly as good as guessing the average, and a negative R² means worse than that."

Then, if he wants the number: our best model is at **R² = −0.012**, i.e. slightly worse than that dumb predictor.

### 3.2 "Do we have a good understanding of the dimension/shape format of the data input to the TSFM?"

He has asked this **twice**. It's clearly a sore point. Have this ready as a drawing, not a sentence:

```
ONE SESSION = one child, one cognitive test, and the glucose readings before it

  Raw:      [142, 145, 139, ... ]   a list of mg/dL numbers,
                                     one every 5 minutes
                                     length varies: 3 to 288 readings
                                     (median 36 readings = 3 hours)

  Into Chronos:  ONE list of numbers. That's it.
                 No second channel. No timestamps. No participant id.
                 Chronos is univariate — one variable over time.

  Chronos cuts it into chunks of 16 readings ("patches"):
      36 readings → pad to 48 → 3 patches → +1 extra token = 4 tokens
      Each token becomes 512 numbers.
      So one session → a (4, 512) block.

  We average down the token direction (4 → 1):
      → 512 numbers for this session.

  Stack all 956 sessions:
      → a table of 956 rows × 512 columns.  That is the model's input.

  Target: one column of 956 scores.
```

**Say this:** *"One session goes in as a single list of glucose numbers, nothing else. It comes out as 512 numbers. Stack all 956 sessions and you get a 956-by-512 table, which is what we feed the regression."*

Full detail and the verified formula are in `08_CHRONOS.md`.

### 3.3 "Why is a uniform 2-hour window the cleanest test of the K01 hypothesis?"

He asked this and you didn't answer it. He also pushed back on the premise, saying *"our current CGM data IS the raw continuous CGM stream."* So you need to handle both parts, carefully, without contradicting him.

**Say this:**

> "The stored glucose list is a different length for every session — anywhere from 3 readings to 288. It looks like it holds everything since that child's previous test, so an overnight gap gives about 24 hours of data and two tests an hour apart give about 12 readings.
>
> That matters because the length itself carries information that has nothing to do with glucose. A long list means a long time since the last test — probably morning, after sleep. A short list means they just tested recently. So when the model looks at a long list, it can't tell whether it's reacting to the glucose shape or just to the time of day.
>
> I can actually show this. The strongest correlation I found with the Grids score isn't a glucose statistic at all — it's the **number of readings**, at r = −0.077. That's the window length leaking in.
>
> If every session used exactly the same span — say the 24 readings covering two hours before the test — then all the sessions would be comparable and the only thing varying would be the glucose. That's what makes it a cleaner test.
>
> On whether we already have what we need: you're right that these are real Dexcom readings, not simulated. What I don't have is the *uncut* timeline with timestamps. What's in the CSV is already cut into per-session lists, and I can't re-cut them to a fixed 2 hours because the short ones don't have 2 hours in them. If Phil has the script that produced these lists, that would tell us whether re-cutting is possible."

That's honest, it concedes his point, it makes the argument with a real number, and it ends with a concrete ask.

### 3.4 "What is an epoch / a token / a patch?"

He named these three specifically as things you must fully understand before he'll consider new ideas: *"This could be promising after you fully understand your python code and all technical terms, e.g., token, patch, epoch."*

- **Patch** — *"A chunk of 16 consecutive glucose readings. Chronos doesn't look at readings one at a time; it groups them into blocks of 16 first. Since a reading is 5 minutes, one patch is 80 minutes of glucose."*
- **Token** — *"One patch after it's been turned into 512 numbers. It's the unit the model actually processes. A 36-reading session becomes 4 tokens."*
- **Epoch** — *"One complete pass through the training data. In our Arm B, one fold has about 612 training sessions and we feed them 64 at a time, so one epoch is 10 small updates and then we check the validation error. We allow up to 100 epochs, but training usually stops earlier."*

### 3.5 "Is the 20th participant real, and how was the missing glucose recovered?"

He said: *"I did not work with 20 patients before, I only worked with 19 patients before"* and *"the CGM monitoring is continuous, i.e., always on. So there is no need for imputation."*

**Do not paper over this.** Say:

> "I'm using 20 because there's a corrected Cohort 2 file that fills in glucose which was blank in the original. In the original, 44 of 240 rows had no glucose, and that wiped out one child entirely — which is why you had 19. The corrected file has only 1 blank row, so all 6 Cohort 2 children survive.
>
> What I can't tell you is *how* that glucose was filled in, because I haven't seen the code that did it. If it was read back out of the raw stream, that's fine. If it was interpolated, then I'm partly modelling made-up numbers and I need to know. I've asked Phil for that script."

This is the single most likely place to be caught overstating. Being the one who *raises* the doubt is much stronger than being the one who gets caught.

---

## 4. The "confound" correction — he is right, so get it right

You used "confound" for the variable window length. He pushed back:

> *"I don't think the term 'confound' is correctly used here. 'Confound', to me, usually mean **two** confounding variables; while here you are referring to **one variable but two different values**."*

He's correct. A **confounder** is a *third* variable that influences both the thing you're studying and the outcome, creating a fake link between them.

```
        time of day
         /        \
        v          v
  glucose  ......  score
   (the apparent link may be entirely time-of-day)
```

- **Time of day IS a confounder.** It plausibly affects glucose (meals, insulin, sleep) *and* affects test scores (alertness). He agreed: *"Here the usage of the word 'confounds' seems correct to me."*
- **Variable window length is NOT a confounder.** It's one measurement that isn't standardised across sessions. The right words are **"inconsistent measurement"** or **"the input isn't comparable across sessions."**

**Say this:** *"I used the wrong word before. The window length isn't a confounder — it's just an inconsistency in how the input was measured. Time of day would be a genuine confounder, because it could affect both the glucose and the score."*

Admitting the correction in your own words costs you nothing and buys a lot.

---

## 5. Ten explanations, in plain words

Practise saying these out loud. If you can't say one without reading, you don't have it yet.

**Normalization.** *"Putting different measurements on the same scale so one doesn't drown out the others just because its numbers happen to be bigger. Glucose is around 177; a slope might be 0.3. Without rescaling, the model effectively only sees glucose."*

**Standardization / z-score.** *"Subtract the average and divide by the spread. Then every feature is centred on 0 and typically ranges about −2 to +2. Same shape, comparable size."*

**Why normalize before training.** *"Three reasons. Ridge applies one penalty to every coefficient, so if the features have wildly different sizes the penalty punishes them unevenly. SVR measures distances, so a big-numbered feature dominates the distance. And gradient descent takes one step size in every direction, so mismatched scales make it zig-zag."*

**Epoch.** *"One full pass through the training data."*

**Batch.** *"We don't feed all 612 training sessions at once. We feed 64 at a time and update after each group. Each group is a batch."*

**Step.** *"One update to the weights, i.e. one batch. Our epoch is 10 steps."*

**Early stopping.** *"While training, we watch the error on a held-out validation set. Training error keeps falling forever because the model starts memorizing. Validation error falls at first, then turns back up — that's where memorizing begins. We stop there. Concretely: if the validation error hasn't improved for 8 epochs in a row, stop."*

**Overfitting.** *"The model memorizes the training children instead of learning something general. It looks great on data it's seen and fails on a new child. Our giveaway: plain linear regression on 512 features got R² of about −3, far worse than guessing."*

**Why a transformer.** *"Because we wanted a model already trained on huge amounts of time-series data, so it could describe a glucose curve without us having to learn that from 20 children. Every good pretrained time-series model happens to be a transformer. The mechanism helps too — a transformer can relate any reading directly to any other, so a dip 90 minutes before the test can interact with the final reading in one step."*

**The MLP head.** *"A small neural network sitting on top of the frozen Chronos model. It takes the 512 numbers and squeezes them down to one predicted score, through a 256-unit middle layer. It has 132,609 adjustable numbers, and it's the only part that trains — Chronos itself never changes."*

---

## 6. The three sentences that summarise the whole project

If you're asked "what did you do," lead with these, then stop and let him ask.

1. **"We took the glucose readings before each cognitive test and asked a big pretrained time-series model to summarise each one as 512 numbers, then tried to predict the test score from those numbers."**
2. **"It didn't work — no better than just guessing the average score. That's true for all three tests, for five model sizes, for every window length and setting we tried, and for the older hand-built-features method too."**
3. **"So we checked whether that's a real finding or a broken pipeline. When we scramble the scores, we get the same accuracy as with the real scores. And when we ask the same setup to predict something about the glucose itself — how variable it was — it does that well, R² of 0.46. So the machinery works; the relationship just isn't in this data."**

---

## 7. Answers for the hard moments

**"Did you write this yourself or did the AI write it?"**
> "I used Claude to help me learn the parts I didn't know and to check my numbers. Everything I'm presenting I can derive at the whiteboard — ask me anything on the slide and I'll work it through. I also recomputed every number in the fact sheet from the CSVs myself rather than trusting what was in the old documents, and I found two things wrong with them, which I'll come to."

That's honest, it doesn't hide the tool, and it immediately offers to prove the understanding. Then **actually be able to do it.**

**"I don't think you understand X."**
> "You might be right. Let me try, and stop me when I go wrong."

Then explain it. Being willing to be corrected mid-sentence reads as confidence, not weakness.

**A question you can't answer.**
> "I don't know. Let me write that down and come back to you."

Write it down visibly. Then actually follow up. Three honest "I don't know"s plus one good follow-up email is a far better meeting than twelve confident guesses.

**"Why should I trust this number?"**
> "Because I recomputed it from the raw CSV rather than copying it. Here's the derivation: 740 rows in Cohort 1 plus 240 in the updated Cohort 2 is 980; 24 have an empty glucose cell; that leaves 956."

**"Is this worth continuing?"**
Don't answer for him. Give him the state of things and one recommendation:
> "The glucose-only approach with this model is close to exhausted — I've swept size, window, pooling, dimensionality, and glucose-regime subsets, and everything sits at or below zero. There are two things I'd still try before concluding, and I found a bug this week that I've fixed. After that I think the honest answer is that this data doesn't contain the effect, and the question becomes how to write that up."

---

## 8. Before the meeting — a checklist

- [ ] I can state the 10 numbers in §K of `01_FACT_SHEET.md` from memory.
- [ ] I can draw the pipeline, with shapes, on a blank whiteboard.
- [ ] I can explain epoch, batch, step, early stopping, normalization, token, patch — without notes.
- [ ] I can explain the mean baseline in the plain words of §3.1.
- [ ] I can draw the shape walkthrough of §3.2 from memory.
- [ ] I have my answer to the 2-hour-window question (§3.3) and it concedes his point.
- [ ] I can say what I *don't* know: the Cohort 2 backfill code, the exact definition of the Prices score, whether more data is coming.
- [ ] I have rewritten my three summary sentences **in my own handwriting**, not copied.
- [ ] I have the two corrections ready (see `15_FINDINGS_TO_REPORT.md`) — the padding bug and the level-information issue. **Leading with your own corrections is the strongest possible opening.**
- [ ] I have not used a single word from the banned list.

---

## 9. One last thing

You're walking in with something better than a rehearsed defence: you found a **real bug** in the embedding code this week, you measured whether it changed the conclusion, and it didn't. You also found that our 512-number representation adds nothing over the single number "average glucose."

That is what actual understanding looks like — not fluency, but finding the thing that was wrong. Lead with it. Nobody who is faking it walks in and says "here's a mistake I found in my own pipeline, here's what I did about it, and here's why the conclusion still holds."

→ Next: `01_FACT_SHEET.md` for the numbers, then `02_ML_FROM_ZERO.md` to start the mechanism.
