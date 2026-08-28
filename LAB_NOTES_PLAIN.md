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

### Chapter 15 — Chunking: a measure that paid out for nothing *(phase 47)*

Before building anything that groups common word-pairs into single units —
the obvious next step for a system that reads — we ran three checks. This
chapter is mostly about the check that saved us from all three.

To decide whether chunking helps you need a score: how many fewer bits does
the text take once the chunks exist. We built that score, and then we ran it
on **shuffled text**, where by construction there are no real word-pairs to
find. It should have scored zero. Instead the simplest chunk-picking rule —
just take the most common pairs — scored **+2286**, a large positive number
on text with no structure in it whatsoever. The score was rewarding
something else entirely: merging pairs makes the text shorter, and shorter
text is cheaper to describe no matter what is in it.

We had written down in advance that if this happened, nothing else in the
experiment counts. So it didn't. Everything we had measured up to that point
went into the file as a diagnosis of a broken ruler rather than as an answer.

**The repair, and why we don't just get to keep it.** Subtracting the
shuffled-text score from the real-text score cancels the length effect and
leaves only what real structure buys. That is a fix invented *after* seeing
the failure, which is exactly the kind of fix that fools people. So we split
the text in half, developed the fix on one half, and confirmed it on a half
it had never seen. It holds there, and on the corrected score the
"principled" chunk-picking rule beats the naive one on every single fold —
which is what the original claim said, and which the broken ruler had been
reporting backwards.

**The result that changes a plan.** The most exciting claim we were handed
was that chunking works even better one level up, on *categories* of words
rather than words. It has a control that has to be run: compare against
categories assigned **at random**, keeping the group sizes the same. Any
grouping of ordinary text produces apparent structure through frequency
alone, and only this control removes it. The discovered categories scored
810. The random ones scored **1280** — better. Not a small margin, and not
on one fold out of five: on all five.

So there is nothing there, and we have said so. A planned piece of work that
was waiting on this stays parked, on its own honest negative rather than on
a guess. One check we did keep: chunk inventories **saturate and then go
backwards** — more chunks help up to about sixty-four and actively hurt
beyond that — and that reversal survives even when we stop charging for the
chunk list itself. So it is a fact about language, not about our accounting,
and it gives a stopping rule an earlier chapter's work was missing.

### Chapter 16 — We checked our own headline, and it was too modest *(phase 46)*

One of the things we say about this system is that it learns from a single
pass — read the text once, the way a person does, rather than grinding over
it repeatedly the way most machine learning does. It is one of the few
claims that would survive a decision to stop competing on raw accuracy, so
it had better be true.

It sat on an awkward fact. The recipe that produced our language results
reads the same text **fifteen times**. If the results only appear on the
fifteenth pass, the claim is not one we get to make.

So we re-ran everything at one pass and compared. On the three things we
measure — how many words get their own memory, whether the categories the
system invents pass their statistical test, and whether it spots words doing
two jobs — **all three survive a single pass**. We had predicted that word
coverage would be the casualty, and gave a specific reason why. We were
wrong: coverage after one pass is 327 words out of 376, and after fifteen it
is **320**. The extra passes don't add words; they quietly merge memories
that one pass had kept apart.

The one thing extra passes do buy is confidence in the categories — the
statistical test goes from comfortably passed to very comfortably passed.
That is worth having and it is now the only thing we claim for them.

**Two limits we are not glossing over.** The one-versus-fifteen comparison
was only affordable on the small text; on the large one we compared one pass
against three. And the word-sense-detection count bounces around a lot
between random starts (39, 39, 71 on one pass), so the right statement is
"three runs cannot see a difference", not "there is none".

### Chapter 17 — Where we are now

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

**Added 2026-08-09.** A second block (Chapters 12–16) went after the
*economics* of the design rather than its capabilities, and it followed the
same pattern — most of what we were handed did not survive contact with a
measurement. Half of every memory turned out to be unused, which is worth a
lot of storage but only if we agree to give the capability up permanently
(Chapter 12); a saving we were told to go find had already been taken
(Chapter 13); a "window" we were told to open turned out to be a question
about our own thresholds rather than about the system (Chapter 14); a
chunking result we were asked to replicate came apart on a control it had
never been run against (Chapter 15). The one that went the other way is
Chapter 16, where our own headline claim turned out to be too modest.

**The open question, and it is not ours to settle.** The storage decision in
Chapter 12 is a one-way door: take it and the system gets much cheaper and
permanently loses a capability we have demonstrated but never yet used. We
priced both sides carefully and deliberately did not choose. Everything that
would be built on top of that decision is parked until it is made.
*(Update 2026-08-14: the missing measurement has now been run — Chapter 20.
The cheap store held up; a recommendation went up; the call is still the
owner's and the parked work stays parked until it lands.)*

---

### Chapter 18 — We changed the exam, and said so out loud *(decisions, 2026-08-11)*

**Added 2026-08-11.** Two decisions had been sitting unmade while work piled
up around them. Both are now made, and this chapter exists because the first
one is the kind of decision a project can quietly cheat on.

**We changed what counts as ready — and we did not pass the old test.**
A year ago the rule was: *at least as good as the state of the art, or
cheap enough to make up for it, or we are not ready.* We did not clear
either bar. On the standard benchmark the system scores 0.712 where a
much simpler supervised method scores 0.872, and a third method (replay)
beats us on **both** measures and tops the entire field. On cost, we spent
four separate efforts trying to get under a 2× budget and finished at 2.24×,
for a reason that turned out to be arithmetic rather than sloppiness: our
memories are complex numbers and theirs are ordinary ones, so ours are twice
the size before anything else happens.

So we changed the exam. The new one asks whether the system does four things
at once that the benchmark winner does not do **at all**: learn its own
representations with nobody labelling anything, discover structure, survive
being saved and reloaded exactly, and do all of it seeing each item **once**.

Changing your own exam after failing it is exactly the move that should make
a reader suspicious, so here is the honest defence and the honest admission,
in that order.

*The defence.* We audited the new exam **before** adopting it, and we
audited the part most likely to be embarrassing. Our own text recipe had
been quietly reading each corpus fifteen times — which would have made
"sees each item once" a false claim about our own production code. We ran
that check expecting to lose, and wrote down in advance that a big drop
would be a real and important finding. Instead the system did *better* on a
single pass than on fifteen (Chapter 16). The one axis that could have
collapsed under scrutiny got stronger. Re-scoping onto a claim you just
stress-tested is a different act from re-scoping onto a claim you find
convenient.

*The admission.* It is still a concession. We are not claiming we met the
old bar; we are claiming the old bar measured one axis and we are
differentiated on others. Those are different sentences and only the second
one is ours. The standing rule we have written into the project is that
**every** description of what this system can do must carry the "not
state of the art, cost target conceded" line in the same breath — not in a
footnote, not on a later page. If a piece of writing about this project
falls apart when you add that sentence, the writing is wrong, not the
sentence.

**The second decision: we refused to guess.** The one-way storage door from
Chapter 12 is still shut. The obvious move was to walk through it — the
cheaper side meets the cost target comfortably and costs almost nothing in
accuracy. But "almost nothing" was measured on a *different test* from the
one where forgetting actually shows up, and forgetting is precisely what a
lossier memory should damage first. So before choosing, we are running the
cheap version through the test that would expose it. If it holds, we take
the door; if old memories degrade, we keep the expensive memories and live
with the cost. *(That test has now been run — Chapter 20. It held.)*

Two smaller things fell out of looking at it properly. The cost saving no
longer decides anything important, because we just conceded that branch of
the exam — so this is now a design choice, not a scramble for a number. And
the door is narrower than we had been telling ourselves: of the four things
we thought we would be giving up, two turn out not to depend on the door at
all, including the cheapest and most promising one. We had been pricing the
decision as more painful than it is.

---

### Chapter 19 — Which folder do you throw away? *(phase 39)*

**Added 2026-08-11.** The system has a fixed number of memory folders. When
they are all full and something genuinely new shows up, it has to throw one
away to make room. We built that rule a while back and found a law for it,
and this chapter is about discovering that we had only checked half of it.

The rule has two parts. First: a folder is only a candidate if nobody has
touched it for a while. Second: among those candidates, throw away the one
with the least written in it. Earlier work swept the first part hard — how
long is "a while"? — and found something genuinely surprising, which we
wrote up as the whole story: set the bar too long and the only untouched
folders are your *oldest* ones, so you burn your own history and end up
worse than if you had never tidied up. Set it short and the junk you created
minutes ago is what goes stale first, so you recycle your own clutter and
the old memories survive.

That was true, and we then quietly assumed the second part barely mattered.
It was never tested. This phase tested it, by adding a deliberately stupid
version — **throw away a random untouched folder** — and running it against
the real rule at every setting of the first knob, on ten separate runs with
five of them held back so we could not fool ourselves.

**The stupid version is a disaster, and that is the result.** At every
setting, throwing away a random stale folder instead of the emptiest one
costs about as much as getting the first knob completely wrong. Old-memory
retention collapses. It went the same direction on all ten runs, which is
the strongest agreement the test can produce at that number of runs.

Why? Because we had been picturing the stale pile as a bin of equally
worthless scraps, in which case "pick the emptiest" and "pick any" would be
the same act. It is not. When we logged what actually gets thrown out, the
real rule discards folders with a handful of entries, and the random version
discards folders with **four to fifteen times more** written in them. The
stale pile is a mix of genuine clutter and perfectly good old memories, and
the second part of the rule is what tells them apart. It was doing the heavy
lifting the whole time and we had credited the first part with all of it.

So **the prediction we wrote down in advance was wrong**, and we are saying
so plainly: we predicted the folder-choosing rule would matter *less* than
the timing knob. Measured, the two matter about equally. The one-line law
gains a second clause — *the stale pile must contain the present, **and you
must throw out the emptiest thing in it**, or you eat your own history
either way* — and the blunt practical version is that **a badly configured
tidying rule is worse than never tidying at all.** Both wrong settings score
worse than simply switching the whole mechanism off.

Two other things fell out. We had a specific "improvement" on the shelf —
adjust for the fact that newer folders have had less time to fill up — that
an earlier investigation had declined to endorse. We tested it properly
rather than assuming, and it is not neutral, it is **actively worse**. That
shelf is now clear. And we found a real retention gain hiding in plain
sight: a caution mechanism built years ago, which makes brand-new folders
prove themselves before they count as permanent, is **switched off** in the
recipe this benchmark uses. Turning it on improves old-memory retention on
every run. We are flagging it rather than adopting it, because we found it
by searching a grid, and this project has watched three grid-found results
evaporate on re-testing.

**What we could not measure, and are not pretending otherwise.** One of the
four knobs we set out to test — how fast the system forgets *connections
between* memories, as opposed to the memories themselves — turned out to be
invisible to this experiment. The knob works; we verified it does something
drastic to the connection table. But the exam we are grading with only ever
looks at the memories, never the connections, so every setting scores
identically. That is a hole in the study, not a finding, and writing it down
as "connection decay doesn't matter" would have been a lie of exactly the
kind this file exists to prevent. Measuring it needs a different exam.

**The caveat that travels with all of it.** An earlier chapter recorded that
the forgetting in this benchmark is caused by teaching things in strictly
separated blocks. This whole study lives at that one setting. Everything
above is "how to protect old memories *when the lessons are fully
separated*" — it is not a general statement about memory, and the phase that
varies that setting has not been run yet.

---

### Chapter 20 — The one-way door, measured before walking through it *(phase 50)*

**Added 2026-08-14.** Chapter 12 found that half of every stored memory —
the "timing" half of each complex number — is essentially empty, and that
storing only the used half would nearly halve the price of memory. It also
said, correctly, that taking that saving is a one-way door: do it and the
system permanently gives up a capability we demonstrated once in a lab
setting and have never used since. We priced both sides and refused to
choose. The owner then refused to choose *blind*, and ordered the missing
measurement instead: nobody had ever run the cheap store and the accuracy
benchmark **in the same experiment**, and the old safety check had only
looked at final accuracy — never at *forgetting*, which is exactly where a
slightly-lossy memory should show damage first, and exactly the axis this
project's remaining claims stand on.

So this phase ran the full continual-learning benchmark twice at every
setting — once with the normal store, once with the cheap one — at two
memory sizes, ten runs each, half the runs held back so we could not fool
ourselves, with the comparison baselines recomputed fresh on every run's
own data split. We wrote down in advance the number that would kill the
idea: if the cheap store costs 0.02 or more of forgetting, it is not worth
taking at **any** price, because that would roughly double the forgetting
the system actually shows.

**The answer: the cheap store is indistinguishable from the full one.**
Accuracy and forgetting both move by less than half a percent — within the
noise, in both directions, at both sizes, on the held-back runs. The
kill-number was never approached. And the discount is exactly what
arithmetic said it would be: the cheap store costs about 0.58 of the full
one wherever you measure it, because the saving lives entirely in the one
term it halves. In plain terms: we get the memory system at 1.15× the cost
of the simple supervised baseline, instead of 2.07×, for free — *on
everything we currently measure*.

Two honest footnotes, because this file is not the optimistic copy. First,
the single most interesting-looking number in the early probe — at the
larger memory size, one configuration appeared to beat experience replay,
the method that tops our whole leaderboard — **evaporated on the held-back
runs**, going the wrong way on all five. That is the third time this
project has watched a one-run result die on reseeding, and it is why the
held-back runs exist. Replay still beats us. Second, one sub-measurement
did move in the worrying direction (forgetting, one decoder, one size, up
by 0.003) — six times smaller than the kill-number, absent in every other
slice, but written down as the number to watch rather than rounded away.

**The recommendation we sent up, and what it actually gives up.** Take the
cheap store — as the format the system *saves to disk in*, while computing
at full width — but leave the switch off in the code until the owner flips
it. The "permanent loss" turns out to be smaller than Chapter 12's framing
suggested: two of the four planned future upgrades don't need the timing
channel at all, and the two that do were already the weakest bets on the
board — the system's own reading mechanism is blind to that channel below
a measured threshold, and the capability it would carry collapses within
dozens of steps by an older measurement. And the door is less one-way than
it looks while the channel stays empty: you can always re-compress a full
store into the cheap format later, whereas going back only loses what the
empty channel had accumulated — which, measured, is nothing. The decision
is still the owner's. What changed is that it is now a decision about
measured numbers instead of feared ones.


**The decision, 2026-08-14.** The owner took it: from now on the system
**saves to disk in the cheap format**, while still *thinking* in the full
one. The saving is real and the measured cost is indistinguishable from
zero on everything we currently test.

Three things are worth saying about *how* this got decided, because the
process is the part worth copying. The measurement was **ordered before
the choice**, not after it — the number that settled the question did not
exist until someone insisted on having it. The run then **killed its own
best headline**, and we kept the kill. And the one measurement that pointed
the wrong way was **written down rather than rounded away**; it now sits on
a watch-list that any future change to the storage format has to re-check.

We also have to correct ourselves about the door. We called it one-way
three times in this file, and that was wrong in both directions: going back
is a re-conversion you can do whenever you like, and going forward gives up
only what the unused channel had accumulated, which was measured to be
nothing. It is a valve, not a door, and it only closes if that channel ever
starts carrying something. What we are actually spending is an *option* —
and that option already had two measured obstacles standing in front of it.

One last piece of honesty: **the code does not do this yet.** Ratifying a
format is not the same as changing the default, and this project pins
changes like that rather than assuming them. Until that work lands, "1.15×"
is the format we have chosen, not the thing running today.

### Chapter 21 — We found the right question, and then found we cannot ask it yet *(phases 51-53)*

**Added 2026-08-28.** Three experiments, and the most useful one failed.

**The idea.** The system decides that two things are "the same memory" by
asking whether they *look* alike. There is an older idea in physics and
computer science that says the right question is different: two situations
are the same if they imply the same *future*. Our strongest result — the one
about words with two jobs — already uses that second question. It just uses
it as a detector, bolted on after the memories have already been formed the
first way. So we asked whether it belongs in the memory-forming rule itself.

**It does, on paper.** On a stream built so the two questions genuinely
disagree, grouping by predicted-future is measurably better: it captures more
of what is predictable, using *fewer* memories than the ideal answer needs.
And the gain is not spread thinly — it sits entirely on the one symbol we
deliberately made ambiguous, and is exactly zero everywhere else. That is
what a real mechanism looks like as opposed to a statistical accident.

**Then we tried to make it run live, and it would not.** Not because the idea
failed, but because the system could not hold the stream steady enough to
test it. One of the five symbols never gets a memory of its own at all. It is
smeared across three. We checked whether it was a space problem; it is not,
at any size we tried.

The reason is the thing this project has been circling for weeks. That
particular symbol is always *in transit* — it only ever appears between other
symbols, and the system carries a fraction of whatever came before into
whatever comes next. So it settles somewhere slightly different every time
and never becomes a stable thing. **A symbol that occurred fourteen hundred
times has no memory.** That is the settling problem, no longer as an
abstraction but as a missing file in the cabinet.

**Two mistakes of ours, recorded because that is the rule here.** The first
run of the live test reported "no effect" — but only because we had wired two
values in the wrong order, so the mechanism could never fire at all. And we
had quietly dropped a safety check from the earlier experiment. Worse, when
we went back to that safety check we found it was never strong enough: it
confirmed the system was *consistent* about which memory it used, which stays
true even when two different things share one memory. It could not detect a
collision, and it did not.

**A third result, small and clean.** An audit found that the half of the
system that *generates* — imagining what comes next — ignores how much
evidence stands behind each of its beliefs, while the half that *plans* does
not. We measured it: a connection seen three times, all agreeing, is followed
40% more often than it should be. The fix exists in the code already and is
simply not connected to that half. We have not connected it, because doing so
changes what the numbers mean and that deserves its own decision.

---

### Chapter 22 — We set three tests, and two of them said stop *(phases 54-56)*

**Added 2026-08-28. This is the last chapter of the research line.**

We had spent weeks circling one problem: the system forgets almost
everything instantly because of how it works, not because of how it is
tuned. Three cheap experiments were specified to decide whether it was worth
rebuilding, and **two of them were named in advance as the ones that had to
move**. The bars were written down before any of them ran.

**Test one: can we read the half of each memory we've been paying for?**
Every part of this system reads memories by *size* and throws away *timing*.
Half of every stored memory is timing. Nobody had ever tried reading it. So
we built a reader that can, and swept the one dial that fills the timing
channel up. **It does not help.** Even at settings where the channel is
demonstrably full, reading it does no better than ignoring it — and the
ordinary reader actually gets *better* at those settings, so we cannot even
blame the comparison. Half the memory is not just unused. It is not worth
using.

**Test two: is our binding limit real?** We have long recorded that the
system can only hold about five things at once. That turns out to be an
artifact of working in 64 dimensions. Capacity grows in step with dimension —
at 1024 it is eighty things. **This one moved**, and it corrects something we
had written down wrongly for a long time. But it was not one of the two that
mattered, and it does not touch the other half of the problem: how *long*
those things survive before collapsing, which is unchanged.

**Test three: does the published fix work?** The wider literature has a
theorem for our exact problem — nonlinear systems like ours always trade away
memory — and a standard remedy: run a slow, simple memory alongside the fast,
clever one. We built it. **It made things slightly worse.** The symbol that
had no memory still had no memory.

**So we stopped.** Not because we ran out of ideas — because we spent the two
we had the most confidence in, and both came back negative against bars we
had set ourselves in advance.

**What we are keeping.** The idea from Chapter 20's neighbour still stands:
grouping memories by *what they predict* rather than *what they look like* is
measurably better, and that is a real result about the mathematics even
though this particular machine could not carry it. The polysemy discovery is
untouched. The engineering — save it, reload it, get the identical system
back — is untouched. And the way we work is untouched, which is the part that
made all of the above findable.

**And what we are doing instead.** Everything this system is genuinely good
at — remembering continuously without being retrained, surviving being
switched off, building its own map of what follows what — is exactly what
conventional AI systems are worst at. So it becomes the *memory* for one of
those, rather than trying to become one. That path needs none of the things
that just failed.

Three of the experiments in this final stretch had to be thrown away and
re-run because we got the setup wrong. Each time it was our own advance
safety check that caught it. That is worth saying, because the conclusion
here rests on the corrected runs, and the reason to trust them is that the
same checks passed on those.

---

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
