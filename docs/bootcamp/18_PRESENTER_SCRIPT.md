# Presenter script — `Progress_Incremental_Value.pptx`

*Six slides, about 5 minutes. The same text is in the PowerPoint speaker notes, so
presenter view works as a prompter. Bold lines are the ones that must land.*

## Slide 1 — Title — the headline

Quick update on the glucose and cognition project.

The headline is that I got the first positive result we've had. The speed game went from minus 0.33 to plus 0.41.

But I want to be upfront about where that came from. It did not come from glucose. It came from using the child's own earlier scores. Glucose contributed about minus 0.01.

So the finding about glucose has not changed. But I think we're in a much stronger position than we were, and I'll show you why.

> *IF ASKED "so glucose works now?" — No. Nothing about glucose changed. What changed is that I can now prove the pipeline is capable of a positive number.*

---

## Slide 2 — Why the old numbers were never going to work

Here's what I think we were doing wrong.

The way we were measuring it, R-squared was asking: can you predict a child you have never met, better than the average child?

But look at the split. For the speed game, only 38% of the variation is one child being different from another. 62% is the same child having a good round or a bad round. For the other two games it's even more lopsided — 85% and 92%.

So we were measuring the hard question, and the data is mostly the other one. And two hours of glucose readings can't tell the model which child it's looking at.

> *IF ASKED where the split comes from — a one-way variance split of each score column by child. Straight from the data, no model involved.*

---

## Slide 3 — What I changed — one new input

So I gave the model one new input: that child's own earlier scores.

Three numbers. Their last score. The running average of their earlier rounds. And how many earlier rounds there were, so the model knows how much to trust that average.

Nothing else changed — same 2-hour window, same model, same folds.

And to be clear, this is not leakage. For a session on Tuesday I only use that child's sessions from before Tuesday. No future information, nothing borrowed from other children. It's what a real system would already have on file.

> *IF ASKED "isn't that circular?" — No. A child's past score is an input you genuinely have at prediction time. It's the standard way to ask whether a NEW measurement adds anything.*

---

## Slide 4 — The result

And it works.

Speed game: minus 0.33 to plus 0.41. Memory and prices go from slightly negative to about plus 0.04.

The other number I'd point at is rho equals 0.70. That asks whether we put a child's rounds in the right order, slowest to fastest — and 0.70 means we largely do. So we're not just close on average, we're getting the order of their good and bad rounds mostly right.

One note on the sample. I drop each child's very first session, because it has no earlier score to use. So it's 891, 862 and 877 sessions — and both bars are measured on those same rows, so the comparison is fair.

> *IF ASKED why Symbols reads −0.326 here and −0.290 in the results file — the results file is on all sessions; this is on the matched rows. Same thing, measured on the same set.*

---

## Slide 5 — Where the gain came from — and where it did not

Now the part that actually matters.

The number went up — but was it glucose?

You can't answer that by throwing everything into one model, because three useful columns get drowned out by 512 that aren't.

So I did it in two steps. First I let the history model predict. Then I took what it got wrong — the leftover — and asked glucose to predict that. If glucose knew something history didn't, it should be able to.

It explained none of it. Minus 0.011 on all three scores.

And I checked it a second way — squeezing the glucose down to 8 numbers first so it couldn't be drowned out. Same answer.

> *IF ASKED why the contribution is slightly negative — that's the cost of fitting 512 columns that carry no information. It's not a signal in the wrong direction.*

---

## Slide 6 — Two more things I checked

Two last things.

First, the memory game. 31% of those rounds are perfect, so asking for the exact score is the wrong question. I asked it as yes-or-no instead: was this round perfect? Half is a coin flip. Glucose gets 0.45 to 0.48. So even asked properly, with the right score, nothing.

Second, the limits — and I'd rather say these than have you find them.

The model needs at least one earlier session from that child, so it can't score a brand-new child cold. Only the speed game holds up strongly. And I tried leave-one-child-out expecting it to be steadier — it got much worse, so I went back to groups of four and I'm reporting that it didn't work.

So where that leaves us: glucose still tells us nothing about these scores. But I can now show the pipeline is capable of a positive number, which I think is a stronger way to state the same finding.

> *IF ASKED "what do you need from us?" — two decisions: is a rigorous characterisation the deliverable, or do we keep hunting; and is N=20 final. Full numbers: results/incremental_value_real.md*

---
