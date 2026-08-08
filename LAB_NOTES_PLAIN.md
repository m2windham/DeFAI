# DeFAI — The Plain-Language Lab Notes

A parallel copy of the project record, written for people who will never
open `organism.py`. Same facts, same honesty, no mathematics.

`ROADMAP.md` is the technical lab notebook — precise, dense, and hostile to
newcomers by necessity. This file is its twin. Nothing here is softer than
the real record: where a result failed, this says it failed. Where a number
is impressive only under a condition, the condition is in the sentence.

**Who this is for:** anyone being told about the project — a colleague, an
investor, a journalist, a collaborator deciding whether to care. Also for
the owner, as a set of ready-made explanations that don't require
re-deriving the framing every time.

**Rule for keeping it honest:** every claim here must trace to a row in
`ROADMAP.md` and a number in `NOVELTY.md`. If those change, this changes.
No claim lives here that doesn't live there.

---

## The one-sentence version

**We're building one continuously running system that watches a stream of
experience, forms its own memories from it, works out how the world tends
to go, reasons about it, and can be switched off and back on without losing
a thing — with nobody labelling anything and no training runs.**

## The thirty-second version

Almost every AI you've heard of learns by being shown millions of examples
over and over, adjusting billions of numbers a fraction at a time, with
humans supplying the right answers. Ours doesn't do any of that. It watches
a stream once, and when something is new it makes a memory for it; when
something is familiar it sharpens the memory it already has. It notices,
by itself, which things tend to follow which, and that becomes a map it can
reason over. Nobody labels anything. There's no training phase and no
deployment phase — it's the same thing running continuously.

The honest part, which we say before anyone asks: **it does not beat the
best conventional systems at benchmark scores.** We measured that carefully
rather than avoiding it. What it does that they can't is learn continuously
from unlabelled experience without forgetting, and survive being switched
off as the *same* system rather than a copy.

## The whiteboard version

Picture a still pond. Drop something in, and ripples settle into a pattern.
Drop the same thing in again, and the pond falls into the same pattern
faster — it's learned the shape. That's a memory, and it isn't stored in a
file anywhere; it's a shape the water prefers.

Now: the pond notices which patterns tend to follow which. That's a map of
how the world goes — "this usually comes after that." Once it has that map,
it can do something odd and useful: **stop looking at the world entirely**
and just walk the map. That's reasoning, and it happens without the pond
part running at all.

Everything else in this project is either making that work reliably, or
finding out where it breaks.

---

## The story so far, in chapters

Each chapter names the phases behind it so anyone can dig into the real
notes.

### Chapter 1 — Can this hold memories at all? *(phases 1–4)*

The first question was whether a field like this can store several patterns
without them collapsing into one blur, and whether it can wander among them
on its own with no input. It can — but only with a fatigue mechanism, where
a memory tires after being visited so the system moves on. Without fatigue
it gets stuck on a single memory forever. That detail turns out to be
load-bearing everywhere downstream.

### Chapter 2 — Does it hold up on real data? *(phases 2–5)*

Tested on handwritten digits against standard methods, and then on a small
made-up language. Two things stood out. It handles **new information
without wrecking the old** far better than a conventional neural network —
learning a second vocabulary cost only about 15% of the first. And it
detects when the world has changed underneath it.

### Chapter 3 — The hard problem: words with two jobs *(phases 8–12, 21)*

This is the heart of the project, and it started as a failure.

Take a word like "fish" — sometimes a thing, sometimes an action. We wanted
the system to notice that on its own and keep two separate memories. The
obvious approach is to check whether the word "looks different" in the two
uses. **Every version of that failed**, and we eventually proved *why* it
must fail: the difference is necessarily too small, because if it were
large the word would stop being recognizable as itself.

The breakthrough was to stop looking at the word. Ask instead: **does
knowing the context change what I'd expect to come next?** If yes, the word
is doing two jobs. That question is answerable from the system's own
predictions — no dictionary, no labels, no human.

It works. On real books, the system picked out "right" as a multi-role word
on its own, and we checked it against a directly measured "what would luck
produce" baseline so it isn't wishful thinking. At larger scale it found
two hundred more.

This is the project's strongest idea. The obvious next question — **does
noticing actually help?** — now has an answer, and it is a partial one.
See Chapter 8.

### Chapter 4 — Real language, and the wall we hit *(phases 19–28, 37)*

Moving from a made-up language to real books surfaced two problems, and
both had unglamorous answers.

Small amounts of text fragmented the system's self-made word categories;
large amounts collapsed them into one giant blob. Everyone's instinct —
including ours — was "it needs more data." **We tested that instinct and it
was wrong.** The real cause was that common words dominated the arithmetic
purely by being common. Correcting for that fixed both ends at once. The
standing lesson: check the obvious explanation before believing it.

The second problem is still partly open. The system's more sophisticated
perception mode has internal thresholds that were tuned on clean, tidy data
and don't transfer to messy real text — coverage collapsed. We've measured
two separate causes, fixed a good part of it (coverage went from 85 words
to 275 of 395), and the rest is honest unfinished business.

### Chapter 5 — Thinking without perceiving *(phase 30)*

We separated the system into a "recognizing" half and a "knowing what
follows what" half, then made a prediction: if the separation is real, we
should be able to switch off the recognizing half and still reason.

We could. With perception entirely absent it still inferred multi-step
consequences almost perfectly, still imagined plausible sequences, and
still planned routes — finding the genuinely shortest path every single
time, about twenty times faster than with perception running.

The closest human analogy is someone who has lost the ability to recognize
or name things but can still think, plan, and reason perfectly well. We're
going to measure that properly by deliberately damaging the system and
watching what survives (T4.1).

### Chapter 6 — Too many memories in too small a space *(phases 34–36)*

A scaling test asked a blunt question: as the vocabulary grows and the
space stays the same, what breaks? Quality fell off a cliff — 99% to 28%
going from 50 words to 800 — while a trivially simple baseline held steady
at 85% for a fraction of the cost. Not a comfortable result to write down.

We tracked down the cause: it isn't that the memories degrade, it's that
**choosing between them** gets harder as they crowd, and one wrong choice
poisons the next. The fix was to choose in two stages — first what *kind*
of thing comes next, then which specific one within that kind. That held
quality at ~85% all the way up.

Then we removed the cheat. The first version was told what the categories
were; the second worked them out itself — and did **slightly better** than
being told. That closed the loop: a real fix, no human input anywhere.

### Chapter 7 — The reckoning *(phases 33, 33b–33h)*

The owner set a hard bar: match the best available methods, or be cheaper
at the same quality, or we're not ready to ship. Then we went and measured
it properly against standard baselines.

**We didn't clear it.** We score 71% where a simple supervised method gets
87% and a memory-replay method gets 91%. We kept twice the retention of the
gradient-based methods, which was the claim we cared about, but we lost the
headline number outright.

What followed is the part worth telling. Rather than tuning to look better,
we spent five experiments finding out exactly *where* the gap lives:

- It was partly capacity — give the system four times the memory and it
  reaches 90%, clearing the bar. But that's changing the rules of the test,
  and we said so.
- It wasn't the read-out method. A promising improvement looked like it
  helped, then **evaporated when we re-ran it from different starting
  points**. We threw it away.
- It wasn't a flaw in how memories get recycled — we built a full audit
  trail to check, and cleared it.
- It was partly storage efficiency, and we made the system nearly four
  times smaller with **exactly zero** loss in quality.
- And finally we measured the floor: even at its best, it costs about 2.2×
  more memory per point of accuracy than the simple baseline. That's
  arithmetic, not effort.

So the honest summary is: **we know precisely what we can't do and why, to
the decimal.** That's a better position than most projects are in, and it's
the input to a decision about whether to re-scope the goal around what this
architecture is uniquely good at rather than a benchmark it wasn't built
for.

### Chapter 8 — Does noticing actually help? *(phase 41)*

Chapter 3 ends with a system that can *notice* a word doing two jobs. That
is a detector, not an ability. This chapter asks the only question that
turns one into the other: if we act on what it noticed, does anything get
better?

**What we asked.** When a word clears the "this word is doing two jobs" bar
— measured against a shuffled-up version of its own history, so luck is
ruled out at that word's own sample size — let the system split its memory
of that word in two. Then check whether the split memories are genuinely
different, and whether the system predicts or writes any better.

**The trap we had to avoid, and it is the whole reason this took a control
arm.** We already knew from earlier work that simply giving the system *more
memory* makes it score better, whatever you do with it. So "split it and see
if it improves" proves nothing — the improvement could be the extra memory.
The fix: build a second system with **exactly** the same amount of extra
memory, split into exactly the same number of pieces for exactly the same
words, but with the occurrences shuffled into those pieces **at random**.
Same resources, no information. Anything the real split does beyond that is
attributable to the noticing.

**What happened.** Three things, and only the first is a clean win.

1. The split memories are *real*. What follows one sense of the word is
   genuinely different from what follows the other, every time, checked
   against a shuffled baseline. A nice bonus we didn't design for: the
   detector occasionally flags a word that isn't actually two-jobbed, and
   this same test correctly refuses those — they split, but into pieces
   that behave identically.
2. Guessing **what kind of thing** comes next got better — by a margin that
   held on every repeat run, and worth about half the distance to a perfect
   score. That gain survives the random-split control, so it is the
   noticing, not the memory.
3. Guessing **the exact next word** did not improve, and neither did the
   sentences the system writes. On writing quality the result wobbled around
   zero and even went negative on one run, which by our own rules means
   there is no effect to claim.

**What it cost, and the unflattering part.** The system does not split a
word cleanly in two. It shatters it into about eleven pieces. Each piece
sees roughly a tenth of the evidence, and thin evidence predicts a specific
word *worse* than one pooled memory does — which is exactly why item 3 came
out flat. Worse, our own gating made this problem bigger: by only allowing
the flagged words to split, we handed the entire spare-memory budget to
three words. The earlier, ungated version of this mechanism split them into
three or four pieces; ours makes eleven.

**What we still don't know.** Everything above is on a made-up language
built so that two-jobbed words really are two-jobbed. On real books, the
same detector mixes up "this word has two meanings" with "this word behaves
differently depending on grammar," and only about one in seven cases is the
former. So this is a demonstration that the mechanism works, not a claim
that it works on English.

**The honest one-liner:** noticing a word does two jobs helps the system
predict *what sort of thing* comes next. It does not, yet, help it write.
And the next problem to solve is no longer the noticing — it's teaching it
to split a word into *two* memories instead of eleven.

### Chapter 9 — Where we are now

Deliberately not chasing new frontiers. The current block deepens the four
things the system is genuinely differentiated on: continual learning
without lesson boundaries, the reasoning layer, deliberate damage studies,
and making the polysemy discovery actually pay off — the last of which now
has its first real answer (Chapter 8: partly, on one axis) — plus
re-measuring performance now that the system has changed shape.

---

## What people push back on, and the honest answer

**"It doesn't beat the benchmarks, so why does it matter?"**
Because the benchmark measures one axis and the system is differentiated on
others. Every method that beats us needs someone to mark where each lesson
ends and, in most cases, to keep the old training data around to re-study.
Ours needs neither, sees each item once, and nobody labels anything. Real
data doesn't arrive in labelled chapters.

**"Isn't 'no gradients' just a limitation you're dressing up?"**
It's a constraint we chose, and it costs us accuracy — that's on the record.
What it buys is learning that doesn't overwrite itself, which is exactly
the failure mode the mainstream approach spends enormous effort patching.

**"Couldn't a large language model do all of this?"**
For the language tasks specifically, a large model would score better. It
would also need enormous training compute, a fixed vocabulary decided in
advance, and it can't add a new memory at 3pm and still have it at 4pm
without retraining. Different tool, different shape of problem.

**"How do I know these results are real?"**
Fair question, and the answer is unusually concrete. Predictions are written
down *before* runs. Results are checked against simulated luck. There's an
automated suite of sixty-five checks that must pass on two independent
computation backends before any change is believed. Failures are kept in
writing. Three separate promising results were thrown out because they
didn't survive re-running from different random starting points — including
one that would have looked good in a press release.

**"What's the actual product?"**
The first one is a memory module: something that ingests a stream of events
continuously, keeps what matters, doesn't forget the old when the new
arrives, and can be shut down and restarted as the *identical* system
rather than an approximation. That last property is unusual and is exactly
what a memory product is for.

---

## A small glossary

| What we say | What it means |
|---|---|
| the organism | the whole running system |
| the field | the "pond" — where recognition happens |
| a slot / memory | one learned pattern the system holds |
| recruit | make a new memory for something novel |
| consolidate | tidy up: merge duplicates, discard junk |
| recall | let it run with no input and see what it produces |
| the transition graph | its map of what tends to follow what |
| polysemy | one word carrying more than one job |
| a null / noise floor | what pure luck would produce, measured not guessed |
| pre-registration | writing down the prediction before seeing the answer |
| the harness | the automated suite that must pass before we believe anything |
| SOTA | "state of the art" — the best published result |
| an honest negative | a result that says our idea didn't work, kept on purpose |

---

## Analogies that hold up

Reusable, and each one is accurate rather than merely vivid:

- **The pond and the ripples** — memory as a shape the system prefers, not
  a file it stores.
- **The sheet vs. the filing cabinet** — conventional learning writes every
  lesson on one sheet, so lessons smudge each other; ours opens a new
  folder.
- **The full cabinet** — when it's full you must throw something out, and
  the whole trick is throwing out your own recent clutter rather than your
  oldest memories.
- **Watching what comes next** — you learn a word has two meanings not by
  staring at the word but by noticing your predictions split.
- **Choose the kind, then the thing** — how to pick from eight hundred
  options without being overwhelmed: pick the category first.
- **Thinking with your eyes closed** — reasoning that continues with
  perception switched off.

Two to avoid, because they overclaim: calling it a brain, and calling it
conscious or sentient. The internal project name reaches in that direction;
the measurements do not, and the gap between those two is exactly what a
skeptic will go for first.

---

## Keeping this file honest

Update it whenever a result changes what can be claimed — the same sweep
that touches `ROADMAP.md` and `NOVELTY.md` (see the Standing Operating
Protocol in `AGENT_TARGETS.md`, section C). Two specific duties:

1. **A weakened claim gets rewritten here too.** If a number in
   `NOVELTY.md` is edited down, the plain-language version follows it down
   in the same commit. This file must never be the optimistic copy.
2. **New chapters, not new adjectives.** When something genuinely new
   lands, it gets a chapter with the same structure: what we asked, what
   happened, what it cost, what we still don't know.
