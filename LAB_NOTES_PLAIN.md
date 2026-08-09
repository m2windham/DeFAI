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

### Chapter 9 — Teaching the reasoning half to think harder *(phase 40)*

Chapter 5 left the reasoning layer able to do three things with the
perceiving half switched off. Three is not a lot. This chapter asks how much
further that half can go on its own.

**Planning that knows what it doesn't know.** The old planner treated "I saw
this route work three times out of three" as certainty. It isn't — three
tries is thin evidence, and treating it as solid quietly drags the planner
toward the corners of the map it has barely visited. So we taught it to
plan on a pessimistic estimate instead: not "how often did this work" but
"how often can I be confident this works, given how little I've watched it."

Against a planner that just counts hops, this is a clear win, and it holds
up on fresh runs. **Against the planner we already had, it mostly isn't.**
The two disagree on about one plan in seven, and when they disagree the
cautious one is better only while evidence is scarce — the advantage shrinks
as the system watches more, and reverses once it has watched a lot. On fresh
starting points the win didn't reproduce. So we did not adopt it. What we
kept is the *report*: the planner can now tell you how reliable a route is
and which single step is the weakest link, which is useful whether or not it
changes the route.

**Plans with conditions attached.** It can now answer "get to A without
going through B," and "reach both A and B." We checked these against
brute-force search on a small map: it finds the genuinely best route every
time, and it never once trespasses on the thing it was told to avoid. A
random walker doing the same trip trespasses about three-quarters of the
time — so the constraint is really being enforced, not just usually
satisfied.

**The idea that didn't work, which we predicted.** We tried letting the
system learn shortcuts — "this five-step route comes up often, treat it as
one move." The gain was **exactly zero**, and it has to be: a shortcut's
reliability is just the reliability of the steps inside it, which the
planner was already accounting for. Worth knowing rather than assuming. But
the interesting part is *why*, and it points somewhere: shortcuts are
worthless when the world only depends on where you are right now. On a world
that depends on the last *two* places you were, the same shortcuts pay off
well. So it's not a dead end, it's a signpost for the level-up experiment.

**Speed.** We predicted a 2× speedup from skipping the empty parts of the
map, at a size we actually use. We got it — but only at maps five times
larger than predicted, and one operation got 5–10× *slower*. A different
operation got hundreds of times faster. The honest version is: this helps a
lot, in specific places, at large sizes, and you must say which operation
you mean.

### Chapter 10 — Re-measuring ourselves, and calling off a build *(phase 42)*

Every speed number we had was recorded before three changes that altered the
system's shape. This chapter re-measures, and its main product is a decision
*not* to build something.

**A pinned number we couldn't reproduce, and what that taught us.** Our
recorded throughput figure didn't come back. Before blaming the machine, we
ran the current system and two older versions of it head-to-head, on the
same machine, in the same session, interleaved. They came out the same. The
machine itself, doing byte-for-byte identical work, varies by about 20% run
to run — a spread wide enough to swallow the number we had pinned. So the
finding isn't "we got slower," it's that **a single absolute speed number
was never a usable alarm**, and we've replaced it with same-session
comparisons.

**We were about to build a GPU tier. We're not going to.** The case for it
was that the statistical safety checks — the shuffled-up comparisons that
keep us honest — were becoming the dominant cost. We finally measured
instead of assuming: they're **22% of a full run, not the 50% we'd set as
the bar**. Two-thirds of the time goes to the main perceiving loop, which is
inherently step-by-step and was never a GPU target in the first place. Even
making the safety checks *instantly free* would speed up a full run by about
1.3×.

Then it got better. Two of those checks turn out to be re-expressible as
ordinary arithmetic that gives **the same answers** — in one case identical
to the last decimal, taking 0.18 seconds where it used to take 283. After
those two fixes the safety checks fall under 1% of a run. Days of work
instead of building a whole graphics-card component, and they remove the
reason it existed.

**Where the time actually goes** — never measured before, only inferred.
Three-quarters of every step is one operation. Our long-standing guess about
*why* was subtly wrong: we thought it was limited by fetching from main
memory, and it's actually limited by the much faster cache. That correction
matters, because it changes which optimizations would pay.

### Chapter 11 — A claim of ours, tested and withdrawn *(phase 38)*

This is the chapter where we go looking for our own best argument and find
out it doesn't hold. It is the most useful thing in this file.

**The claim.** Standard learning methods are usually taught in blocks — all
of topic one, then all of topic two — and they need a teacher to say "topic
one ends here." One popular method uses that moment to write down which
parts of itself matter most; another uses it to keep a balanced sack of old
examples. Ours needs no such announcement. We had been saying that this is a
real advantage, because real data doesn't arrive with chapter headings.

**How we tested it.** Three versions of the same experiment, identical in
every other respect — same amount of data, same amount of study time. In the
first, topics come in blocks and the other methods are told where the breaks
are. In the second, **the same blocked stream, but nobody is told** — the
methods have to guess, on a fixed timetable that doesn't line up with the
real breaks. In the third, topics **blend into each other**, old ones drift
back, and new ones appear with no announcement at all. The middle version is
the one that does the real work, and building it is what made the answer
clear.

**The result: we were wrong.** Taking the announcements away costs the other
methods almost nothing. One of them got slightly *better* without them —
guessing on a regular timetable beat being told the truth. So the
announcement was never what they needed.

**What they actually needed was for the topics not to be blocked.** When we
let the topics blend — which is what a natural stream does — the other
methods recovered nearly all of their lost ground on their own, going from
about 30% to about 92%. Ours moved by less than half a percentage point.
The teacher's markings were never the load-bearing thing; being forced to
study one topic at a time was, and the markings are just a label stuck on
that.

**A second thing we hoped for, also absent.** We expected the system to give
away the topic changes by recruiting new memory in bursts right when
something new appeared — computing for itself the signal the other methods
have to be handed. It doesn't. Recruitment at those moments is, if anything,
slightly *below* average. When topics blend gradually there is no sharp
moment to detect, so there is no burst.

**What is left, stated honestly.** One thing survived, and it is exact
rather than approximate: our system scores *identically* — to the last
decimal — whether or not it is told where the breaks are, because it never
had anywhere to put that information. That is a genuine property of how it
is built. It is just worth much less than we were claiming, because the
thing it is immune to wasn't hurting anybody else much either.

We have taken the claim out of the pitch. What replaces it is a better
question we now know how to ask: *how blocked does a stream have to be
before the announcement is worth anything to anyone?*

### Chapter 12 — Half of every memory was never being used *(phase 43)*

Every memory this system stores has two halves — think of each one as a
note with both a *shape* and a *timing offset*. The shape is what the
memory looks like; the timing offset was meant to let several memories be
held at once without smearing together, the way several people humming
different notes can still be told apart. Storing both halves is the single
biggest reason our memories cost more than the simple baseline we measure
against: roughly twice the bytes per memory, and that factor is most of the
gap the release decision has been stuck on.

**So we went and looked at what is actually in there.** Not argued about
it — measured it, with a number that is zero exactly when a memory has no
timing content at all and one-half when it is completely full of it. The
answer, across three different bodies of data: **about 0.00026**. Better
than 99.8% of memories are essentially all shape and no timing. Then we
checked the obvious follow-up: does anything downstream even read the
timing? We scrambled it — gave every memory a random offset — and the
system's score did not move by a single digit. Nothing reads it.

**Why is it empty?** The work order we were handed said the answer was
structural: the machinery simply cannot put content there. We checked by
deriving what the update rule actually does instead of describing it, and
**that answer is wrong**. It can put content there. Turn one dial (the
system's internal rotation speed) up, or let each input settle for less
time, and the timing channel fills right up — we measured it going from
0.00026 to 0.33. It is empty because of where two dials happen to be set,
not because of how the thing is built. This matters: it means the
"add timing information" upgrade someone might propose next is a settings
change, not an invention.

**And one honest catch about our own test.** The obvious check — throw the
timing away and see if accuracy drops — showed no drop at all. That is a
weaker result than it sounds. We built a version where the timing channel
*is* full and threw it away too: still no drop, until the channel was
almost entirely full. Our scoring method is simply blind to this until the
effect gets large. So the reason we believe the channel is empty is that we
measured the channel, not that the score didn't move.

**What it is worth.** If we store only the shape and one small number for
the offset, the memory cost falls from about 2.24× the baseline to about
**1.32×** — comfortably inside the fallback target the release decision
named — and, in the setup we measure on, nothing gets worse. The catch is
that this is a one-way door: throw the timing channel away and the
"several notes at once" capability from Chapter 3's neighborhood goes with
it, permanently, along with anything we might later build on it.

We deliberately did not choose. Both options are priced and the decision is
the owner's, because it is not really a storage question — it is a question
about what this system is for.

### Chapter 13 — A rule we had backwards about the connection table *(phase 44)*

Alongside its memories, the system keeps a table of which memory tends to
follow which. That table is mostly empty — most memories never follow most
others — so we store only the entries that exist. The work order we were
given assumed there was still money on the table here: sparsify the
connections and the storage bill comes down. **We checked before spending
anything, and there was nothing to spend: the bill we have been quoting for
months already assumed the sparse form.** Making it "the default" moves the
number by exactly zero. Saying so first, before doing the interesting part,
is the whole discipline in one paragraph.

There *was* money somewhere else, and it was in the boring details: the
table stores each entry's address and count in wider number formats than
they need. Narrowing both — with a check each time that nothing is lost, so
it is not a rounding trick — halves that part of the bill. We worked out
what it should come to on paper first (5058 bytes) and then measured it
(5058 bytes). Combined with Chapter 12's change, the storage bill goes from
2.24× the simple baseline to about **1.15×**. Which, we now think, is where
the sandbox's unexplained "1.13×" came from — it was quietly counting this
too.

**But the pre-registered test still failed, and it matters.** Even with the
connection table squeezed as hard as it goes, the bill only falls from 2.24×
to **2.07×** — still outside the 2× target. So the target was never going to
be won or lost on the connection table. It is decided entirely by the choice
in Chapter 12. That is a cleaner answer than a saving would have been.

**And one assumption of ours turned out to be wrong.** We had been treating
"the table is 6% full" as a fact about the system. It is a fact about *how
much text it has read*. Feed it sixteen times as much and the table goes from
6% to 37% full on one benchmark and from 0.8% to 4.7% on another. It still
pays to store it sparsely — that stops paying only around 50% full — but
every storage number we have ever published needs the amount of data
attached to it, and we have gone back and said so about the older ones too.

### Chapter 14 — How long to look at each word *(phase 45)*

When the system reads, each word is held in front of it for a fixed number
of moments — eight, in every experiment we have run. Eight is about twice as
long as it takes the system to settle on what it is looking at. The question
this chapter asks is what that costs: after eight moments, does anything
remain of the words that came *before*?

The sandbox's answer was that nothing does — that the system ends up in a
pure "last word" state — and that a shorter look (three moments) opens a
window where the earlier words still matter. The first half is right. On the
similarity scale the system itself uses, two sentences ending in the same
word look 97% alike after eight moments. Shorten the look and they separate.

**The second half is wrong in an interesting way.** We tested whether what
comes back at a short look is really the earlier words or just the system not
having settled yet — noise dressed up as memory. The test is whether it
*repeats*: start from a different random state and see if you land in the
same place. It does, and more than that: **you can identify which earlier
words produced a state perfectly, at every look length we tried, including
twelve.** The earlier words are always there. What changes with look length
is not whether the information exists but *how loud it is* — and every part
of the machinery that reads it works by comparing against a fixed threshold.
So "does the system notice the earlier words" is a question about the
readers, not about the state. That is a different problem from the one we
thought we had.

**The trade is real and it is steep.** Shorten the look to three moments and
the system covers 265 of 376 words instead of 327, and — the part that
actually decides it — the categories it discovers **stop passing their own
statistical test entirely**. There is no free lunch here and eight is not too
long. What this points at instead is keeping the fast reader and adding a
second, slower one alongside it: the slow one remembers the earlier words
without anything having to give up its threshold.

### Chapter 15 — Where we are now

Deliberately not chasing new frontiers. The current block set out to deepen
the four things we believed the system was differentiated on: continual
learning without lesson boundaries, the reasoning layer, deliberate damage
studies, and making the polysemy discovery actually pay off.

All but one now have first answers, and the pattern across them is worth
saying plainly: **each came back partial or negative, and the project is
better for it.** The reasoning layer went deeper, with its most promising
new idea measured and then *not* adopted (Chapter 9). Polysemy pays off on
one axis and not the one we most wanted (Chapter 8). Performance was
re-measured and a planned build was called off as a result (Chapter 10).
And the continual-learning differentiator — the one we were most confident
about — was tested and **withdrawn** (Chapter 11).

That last one cost us an argument we liked. It also replaced it with a
sharper question we can actually answer, which is the better trade. What
remains untouched: a proper study of what protects old memories, and the
damage-and-recovery experiment.

---

## What people push back on, and the honest answer

**"It doesn't beat the benchmarks, so why does it matter?"**
Because the benchmark measures one axis and the system is differentiated on
others: it sees each item once, nobody labels anything, and it learns the
structure of what it sees rather than just a score.

This answer used to continue: "and every method that beats us needs someone
to mark where each lesson ends." **We tested that claim and it did not
survive** — see Chapter 11. Take the markings away and the other methods
barely notice. We have edited the answer rather than quietly keeping the
better-sounding version, because a file that only ever gets more optimistic
is not a lab record.

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
