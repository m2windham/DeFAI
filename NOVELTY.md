# DeFAI — Novelty Register

What this project can actually claim, what each claim rests on, what it
does **not** cover, and the mini-roadmap to take each from *implemented* to
*perfected*. Written 2026-08-07 against the tree carrying phases 27/28/33h
and the symbol registry.

**Purpose.** Three jobs: (1) keep public claims scoped to what is measured
— phase 33 established that this project will say "not SOTA" out loud, and
this file is where that discipline lives; (2) give any new session or
reviewer the honest summary in one read; (3) source material for the D2
launch essay, so marketing never outruns the harness.

> **BINDING, added 2026-08-11 — the gate was re-scoped and the RELEASE HOLD
> lifted.** The readiness gate is now the capability axes: *unsupervised
> representation + structure learning + bitwise persistence, in one
> single-pass online mechanism*. The full decision record, with the
> per-axis evidence table and the not-claimed list, is at the top of
> `ROADMAP.md`.
> **This makes job (1) above load-bearing rather than aspirational.** The
> re-scope **concedes** the cost branch instead of passing it, so the
> caveats are now the price of the claims and travel with them: **not SOTA**
> on class-incremental accuracy (0.712 vs the 0.872 prototype bar; replay
> tops the ladder at 0.913/0.105), **cost parity and the ≤2× fallback both
> missed** at a measured 2.24× (2.07× narrowed), equal-accuracy
> cost-effectiveness **not claimable**. *(Those are the numbers the gate
> was conceded on, and they stay on the record as such. Separately, on
> 2026-08-14 the owner ratified the (A) persistence layout, measured at
> **1.148×** — see ROADMAP row 50. It is **the ratified layout, not the
> current default**. T7.8 was amended on 2026-08-28: the layout is now a
> NAMED spec, `organism_compress.deployment_spec()`, rather than a changed
> `CompressionSpec` default — because the bare spec is the measurement
> baseline the harness uses as the c64 arm of its own fork comparison, so
> flipping it would have turned that A-vs-B test into A-vs-A. The layout is
> persistable and pinned (harness §20); what is still open is T7.8's own
> bar, phase 50 re-measured on the deployment path with dFORG at K=112
> re-reported, which needs torch. **Until that runs, no artifact may quote
> 1.148× as what the code does today.**)* Every claim below is quoted with
> that line in the same breath — including in the D2 essay. A capability
> claim published without it misrepresents the decision that authorized it.

> # CLOSED REGISTER — Path A retired 2026-08-28
>
> This register is **closed for new Path A claims**. See the terminal
> decision record at the top of `ROADMAP.md`. The mini-roadmaps below are
> **historical**: they describe what each claim would have needed, not work
> that is pending. Do not read an open "next step" as a live plan.
>
> Two edits the 2026-08-28 experiments force, both recorded rather than
> deleted:
> - **N8's stated reason is wrong.** Phase 55 measured superposition capacity
>   scaling linearly with dimension (m\*/N ≈ 0.07 flat; m\*(64) = 5.0 exactly,
>   matching phase 16). The "about five items" ceiling is an **artifact of
>   N=64**, not a property of the mechanism. What is NOT rescued is
>   **lifetime** — phase 16's 9–56-step collapse under attractor pull is a
>   dynamics property phase 55 does not address, and it is why N8 stays
>   unsolved.
> - **The complex representation is now unjustified on BOTH axes.** The cost
>   axis was conceded 2026-08-11; phase 54 closed the capability axis by
>   measuring the phase channel unreadable *even when full*.
>
> **N1 is unaffected**, and phase 51's causal-state result stands as the
> strongest finding of the closing line.

**Rules for editing.** A claim enters this file only with (a) a phase
script, (b) measured numbers against a null or baseline, (c) a scope
caveat written by the person who ran it. Numbers here must match the
ROADMAP row they cite. If a claim is weakened by a later run, it is
**edited down, never deleted** — the weakening is the record.

| # | Novelty | Strength | Next step |
|---|---|---|---|
| N1 | Predictive (Myhill–Nerode) polysemy detection | **Strongest** | sense *counts*, then real text (T6.4 done) |
| N2 | Field-free logic layer that does real work | **Strong**, with a measured qualifier: the GENERATIVE half of it is evidence-blind (phase 53) | wire `confidence` past the planners, as its own target with the semantics decision made explicitly |
| N3 | Attractor Crowding Collapse, named and fixed label-free | **Strong** | real-text impurity test |
| N4 | Gradient-free continual learning | **Solid** | T6.1/T6.2 — boundary-free + retention |
| N5 | Category validity without geometry | **Solid** | k-selection at scale (phase 27 residual) |
| N6 | Bitwise persistence + stable symbol identity | **Solid (engineering)** | T4.1 — survive a lesion |
| N7 | Eviction under recruitment pressure | **Narrow but crisp** (factorized; the law now has two clauses — T6.2 done) | self-calibrate **both** knobs from online statistics (N7 step 3) |
| N8 | Phase-superposition binding | **Unsolved — and the reason is now CORRECTED.** Capacity was never the limit: phase 55 measured m\*/N ≈ 0.07 flat, so "~5 items" is an artifact of N=64. **LIFETIME is the limit** (phase 16: 9–56-step collapse). Phase 54 then closed the channel on capability grounds too — unreadable even when full | RETIRED. Reopening must first clear phase 54's bar |
| N9 | The measurement discipline itself | **Underrated** | codify as a harness tier |

---

## N1 — Predictive polysemy detection (Myhill–Nerode criterion)

> **In plain terms.** How do you notice that "bank" means two different
> things, when nobody ever tells you? The obvious approach is to look at the
> word and ask whether it *looks* different in different sentences. It
> doesn't — and we didn't just fail at that, we proved it can't work. What
> does work: watch what comes **next**. If knowing the words before it
> changes what you'd bet on the word after, the word is doing more than one
> job. The system notices its own predictions splitting in two. No labels,
> no dictionary, no human in the loop — and it found "right" as a
> multi-role word in eight novels by itself, then again in forty-two.
>
> **And now: does noticing actually help?** Partly. We wired the discovery
> back in, so a word caught doing two jobs gets its memory split in two, and
> then we checked whether that buys anything — against a control given the
> *same* extra memory but told to split at random, because more memory helps
> on its own and we would otherwise have credited the wrong thing. Guessing
> the *kind* of word coming next got measurably better. Guessing the exact
> word did not, and neither did the sentences it writes. The reason is
> unflattering and worth stating: it doesn't split a word cleanly in two, it
> shatters it into about eleven pieces, and thin evidence spread over eleven
> memories predicts a specific word worse than one pooled memory does. So the
> honest sentence is "noticing helps it predict *what sort of thing* comes
> next" — not "noticing makes it write better."


**Claim.** To detect that a word carries two senses, do not ask whether its
representation drifts in context — ask whether knowing the context changes
*what you predict next*. A state-splitting test on successor-category
entropy, fully online, zero labels anywhere in the mechanism.

**Evidence.** `phase12_predictive_split_test.py` (0.30-bit margin, zero
false positives on synthetic dual-role words) → `phase21_working_polysemy_
detection.py` (real text: "right", 622 occurrences over 8 books, gain 0.070
vs a directly measured 99th-percentile null of 0.043, plus 37 plausible
multi-role words as an honest full ranking) → `phase23_unified_large_corpus.py`
(114 words clear per-word permutation nulls at 547K words; "right"
reproduced against a ~3× tighter null, 0.115 vs p99 0.015) →
`phase27_5m_word_scale_run.py` (**P2 confirmed at 5.22M words**: "right"
clears its tighter null again, 205/354 candidates clear theirs) →
`phase28_polysemy_vs_context_sensitivity.py` (decomposition: 18/119 words
lexical polysemy, 101/119 grammatical context-sensitivity) →
`phase41_detection_driven_split.py` (**T6.4: detection now drives the field
and the result is measured, not assumed** — see the downstream paragraph
below). Pinned in harness §5 (dual-role gain 0.806 vs monosemous 0.003) and
§13 (the closed loop end to end, including the matched-capacity invariant).

**Downstream utility — the open question, now answered, and narrower than
the claim assumed (phase 41, 2026-08-08).** Detection driving phase-10
context-primed splitting **does** buy downstream utility on one axis and
**does not** on the others. Measured against an EXACTLY matched-capacity
control (same per-word slot count for every word, occurrences routed at
random — mandatory because T1.5 showed extra slots alone buy accuracy):
held-out next-*category* accuracy **+0.063**, positive on all five paired
seeds, 56% of the headroom between the control (0.672) and an eval-only
oracle ceiling (0.786). Successor distinctness holds 15/15 against a
same-word permutation null, and — an internal validation the phase did not
have to pass — 0/3 of the detector's own false positives clear that test.
**But generation does not improve at all** (corpus-bigram hit +0.001 and
modal-role grammaticality −0.005, both failing the survival rule by sign
flip), and word-level likelihood, while beating the matched control by
+0.145 nats/token, is **worse than the unsplit organism** (−0.081): the
split fragments each word across ~11 slots, and that costs more than its
information buys on the word axis. So the honest form of the claim is
**"detection-driven splitting improves category-level prediction at matched
capacity"** — not "improves generation", and not "improves prediction"
without the qualifier.

**Why it's novel.** The ingredients have ancestry — Myhill–Nerode state
splitting, Brown clustering, distributional sense induction. What is not
standard: doing it **online inside a running dynamical system**, with no
labels, no gradient, no offline clustering pass, against nulls measured at
the actual sample size — and arriving there by *first proving the obvious
representational approach cannot work* (`verify_residual_gating.py`: the
recruit gate provably cannot fire under additive context blending, and
consolidation destroys any split that does form).

**Scope caveats.** Predictive gain measures *distributional* context
sensitivity, which is broader than lexical polysemy — phase 28 quantified
the split (only ~15% is lexical at POS level) and its own test is a
bigram-level proxy with thresholds chosen in-phase, no null of its own
(bootstrap on chance cross-bucket conflicts is the open follow-up).
Same-POS polysemes (bank/bank) are classed as context-sensitivity by
construction. Detection is now shown to improve **category-level prediction**
downstream (phase 41), and **not** generation or word-level likelihood — the
downstream claim is real but single-axis, and it is measured on a SYNTHETIC
corpus where polysemy is lexical by construction, so it is a mechanism
result, not a real-text capability. Phase 41 also made phase 10's open
"exact 2/1 slot structure" problem *worse*, not better: gating concentrates
the free-slot budget on the detected words, so each splits into ~11 slots
(mean role-purity 0.87) where ungated phase 10 produced 3–4.

**Mini-roadmap.**
1. ~~*Implement* — **T6.4 / phase 41**: close the loop.~~ **DONE
   2026-08-08.** Detection triggers phase-10 context-primed splitting; the
   matched-capacity control arm was built and its slot budget matched
   EXACTLY at every seed. Outcome: prediction (a) confirmed 15/15;
   downstream utility survives on next-category accuracy (+0.063) and fails
   on both generation metrics by sign flip. The blanket honest-negative
   ("flat at matched capacity ⇒ a representational nicety") did NOT fire,
   but neither did the strong reading — see ROADMAP row 41 and harness §13.
2. *Improve* — **the binding problem is now fragmentation, not the trigger**:
   move beyond binary splits to **n-way sense counts** (how many senses, not
   just "more than one"), which is what would let the ~11 slots per split
   word collapse to the 2 the corpus actually contains — phase 41 measured
   that concentrating the budget makes this worse, so it is the highest-value
   next step for N1, ahead of any further work on the gate. Also: give phase
   28's polysemy/context call its own measured null (bootstrap the chance
   cross-bucket conflict rate); re-derive the decomposition on the 5M corpus.
3. *Perfect* — transfer evidence: does the criterion port to a second
   language or a non-linguistic stream (code tokens, event logs)? A
   criterion that survives a domain change is a much stronger claim than
   one tuned to English prose. Then wire sense-splitting into the unified
   loop's stage B so it is a standing capability, not a phase script.

---

## N2 — A logic layer that runs with the field absent

> **In plain terms.** The system keeps two things apart: the part that
> recognizes things, and the map of what-tends-to-follow-what. Switch the
> recognizing part completely off and the map still works — it can still
> infer, plan a route, and imagine a sequence. It's the machine version of
> someone who has lost the ability to recognize or name things but can
> still reason and plan perfectly well. That was a *prediction* the design
> made; we then measured it: planning found the exact shortest route every
> single time, and imagining ran about twenty times faster with the
> perceptual machinery switched off. The split isn't cosmetic — it earns
> its keep.


**Claim.** Transition learning, extracted behind a confidence-gated
`EventBoundary`, supports multi-step inference, imagination, and planning
**with the perceptual field switched off** — and the separation is
measurably load-bearing, not cosmetic.

**Evidence.** `phase30_symbolic_reasoning.py`: k-step inference corr
0.995–0.999 vs a permutation null ~0.45; field-free rollout bigram corr
0.999 against field recall's 0.939 at ~20× the speed; directed recall
reaches every goal at exactly true shortest-path length (1.8 hops, 100%)
vs 23.2 hops undirected. Pinned in harness §6. Refactor was behavior-
identical (demo bitwise, harness values unchanged). *Scope on the "~20×":
it is a pre-E2 numpy measurement of rollout against field recall; re-run
2026-08-08 it is 17–18× on numpy and 2× on numba — the field side got
faster, the graph side did not change. Quote it as "rollout is far cheaper
than field recall", never as a fixed ratio.*

`phase40_logic_depth.py` (T6.3) extended the layer along the three axes the
caveats below used to name as missing, and priced each: **planning under
uncertainty** (Wilson lower bounds on edge probability + a `PathReport`
carrying reliability, weakest link and that link's observation count) beats
hop-count planning by **+1.04 nats/plan** of true log-reliability, 0/5 sign
flips, above a shuffled-shrinkage permutation null and held-out verified;
**compositional goals** (negation, conjunction) are certified against
brute-force optimal walks — 0/6 avoidance violations, optimality gap 4e-16
and 0.0; **sparse ports** of next_hops/rollout are **bitwise** identical to
dense including the RNG stream, and `kstep_row` — the query planners
actually issue — runs 4×→356× from K=112 to K=1580. Pinned in harness §12,
which also pins `next_hops` bitwise against its pre-phase-40 body.

**Why it's novel.** Neuro-symbolic hybrids usually bolt a symbolic module
onto a learned encoder. Here the graph is *the same object the field
built*, mutated only through `observe`/`merge`/`retire`/`fold`, and the
demonstrated consequence is an **aphasia property**: damage perception,
reasoning continues. That is a falsifiable architectural prediction that
was made and then confirmed.

**Scope caveats.** Still demonstrated on synthetic worlds — a 6-regime
hub-and-branch world and phase 40's Zipf/second-order/12-symbol worlds —
**not on the real-text graph**. The "aphasia" claim is qualitative until
phase 31 measures it. Four caveats phase 40 added or sharpened, all of
which a skeptic should be handed rather than have to find:

- *Correction, phase 40*: phase 30's planner was never "Dijkstra over hop
  counts" — it already minimized −log transition probability, i.e. the
  most-probable path. This register said otherwise and was wrong.
- Confidence weighting's robust win is therefore over a **hop-count**
  planner, which is a weak baseline. Against phase 30's own MLE planner it
  is a **thin-evidence correction only**: the two disagree on ~14% of plans
  and confidence wins +0.379 nats when they do at 2000 observations,
  decaying to −0.074 at 60 000. It clears the survival rule on selection
  seeds and **fails on held-out seeds**, so it is recorded as a direction
  and not banked. The durable contribution is the *report*, not the router.
- Temporal abstraction is a **measured negative at first order**: a mined
  macro-edge cannot beat planning over its own parts (exactly 0.0, forced
  by construction), and the pass-through census finds nothing to contract.
  Macros pay only when the world is **second-order** (+0.443 nats, held-out
  +0.581) — the boundary is an ORDER, not a graph size.
- The sparse speedup is real but arrives late: 2× at **K=800**, not at
  K=112 as predicted (6.1× at K=1580). Sparse *full* `kstep` is 5–10×
  **slower** than dense. Never quote the 356× without naming `kstep_row`
  as the op.

**Mini-roadmap.**
1. ~~*Implement* — **T6.3 / phase 40**~~ — **DONE 2026-08-08.** Delivered
   confidence-weighted planning with reliability reporting, compositional
   (negated + conjunctive) goals certified against brute force, macro-edges
   in both the `fold` and overlay readings, and the sparse-P ports. Two of
   three pre-registered predictions returned partial or negative answers and
   are recorded above rather than smoothed. Note for whoever picks up
   temporal abstraction: `fold` turned out to be the *wrong* primitive for
   it — folding merges identities, whereas a macro names a route between
   symbols that stay distinct, so macros landed as a non-mutating planning
   overlay and the four auditable graph mutators are untouched.
2. *Improve* — **T4.1 / phase 31**: graded lesion curves turning the
   aphasia property from anecdote into measurement, now with registry-
   identity survival and bitwise-E3 recovery as controls.
3. *Perfect* — run the logic layer on the **real-text** graph (phase 27's
   598 slots): does planning/inference hold up when the world model came
   from prose rather than a designed world? That is the honest scaling
   test, and nobody has run it. Phase 40 raised the stakes on it rather
   than reducing them: prose is exactly the under-evidenced, heavy-tailed,
   plausibly-higher-than-first-order regime where its two live findings —
   the thin-evidence correction and the second-order macro gain — should
   either pay or be falsified, and at K≈598 the sparse ports are past their
   measured crossover. It is now the cheapest high-information test of N2.

---

## N3 — Attractor Crowding Collapse: a named failure mode, fixed label-free

> **In plain terms.** Memories here are like dips in a landscape that a
> ball rolls into. Put too many dips in a small space and they blur
> together; pick the wrong one and your next pick starts from the wrong
> place, so mistakes compound. Going from 50 words to 800, quality fell
> from 99% to 28%. The fix was not a bigger landscape. It was to choose in
> two steps — first *what kind* of thing (noun-ish? verb-ish?), then which
> specific item within that kind. Five choices, then a handful, instead of
> 800 at once. Quality held at ~85% all the way up. And the "kinds" are
> ones the system worked out for itself; nobody told it what a noun is.


**Claim.** A fixed-dimensional attractor store degrades as the number of
stored items grows — but the binding constraint is **selection, not
storage**, and routing selection hierarchically fixes it at inference time
with categories the system discovers itself.

**Evidence.** `phase34_capacity_scaling.py`: grammaticality collapses
0.994 → 0.284 as vocabulary grows 50→800 at fixed N=128, while a bigram
table holds ~0.85–0.87 flat at up to ~900× less compute (the honest
comparison that makes the finding sting). `phase35_hierarchical_recall.py`:
category-then-word routing holds ~0.84–0.85 flat across the same sweep,
recovering 98% of the flat-vs-oracle gap at 800 words — but with oracle
categories. `phase36_unsupervised_hierarchical_recall.py`: **oracle caveat
closed** — `discover_categories_v2` + distinctness k-selection gives
ρ = 1.02 (discovered routing matches or beats oracle at every vocabulary),
k=5 selected correctly at all scales, MI-null z up to 25 624.

**Why it's novel.** Two parts. Naming and isolating the mechanism —
crowding is amplified by one-shot K-way selection into cascading errors,
which a resampling baseline doesn't suffer — and then showing the fix needs
**no retraining and no labels**: same Hebbian weights, same attractors,
only the selection procedure changes.

**Scope caveats.** Synthetic cyclic grammar only. Discovery was *perfect*
there (purity 1.000), so the pre-registered purity floor was never
exercised — whether routing survives real-text category quality
(V ≈ 0.55) is **open**. Phase 27's k-collapse (below) makes that question
sharper, not softer.

**Mini-roadmap.**
1. *Implement* — done (phase 36).
2. *Improve* — run hierarchical routing on real-text categories at
   phase-23/27 quality and **measure the purity floor** the synthetic run
   couldn't reach. This is the single most informative open experiment
   attached to N3.
3. *Perfect* — recursion: if one level of hierarchy rescues selection, does
   a second (T2.4 / phase 29, phrase-level motifs)? Pre-registered risk
   stands — "level 2 learns nothing" is a legitimate outcome.

---

## N4 — Gradient-free continual learning

> **In plain terms.** A normal neural network is one big sheet, and every
> lesson writes on the same sheet — so lesson two smudges lesson one. Ours
> uses folders: something new gets a new folder, something familiar refines
> an existing one. New knowledge goes in new places instead of on top of
> old places. No error-correction pass, no second look at the data, nobody
> labelling anything. Result: it keeps about **twice** what the standard
> methods keep. The honest half: it doesn't win the benchmark — a simple
> supervised method scores 87% to our 71%.
>
> We used to add: "and everyone else needs a teacher marking where each
> lesson ends, while we don't." **We tested that in phase 38 and it is
> false.** Take the teacher's markings away and the other methods barely
> notice — one of them actually did slightly *better* without them. What
> hurts them is not the missing markings, it's being made to study one
> topic at a time; the markings are just a label on that. Let the topics
> blend, the way real streams do, and they mostly recover on their own.
> Ours is genuinely indifferent to the markings — provably, to the last
> decimal — but that turns out to be worth much less than we claimed,
> because it wasn't costing anyone else much either.


**Claim.** Online, single-pass, unsupervised continual learning with
**2× the retention of gradient baselines** — no rehearsal buffer, no
importance weights, no second pass over data.

**Evidence.** `phase33c_gate_retest.py` (class-incremental split-digits):
organism 0.712 ACC / 0.169 FORG vs SGD 0.323/0.842 and EWC 0.322/0.844.
`phase33b_slot_budget.py` lifted the flooding ceiling (task-0 retention
0.732 → 0.939). `phase33d_capacity_sweep.py`: ACC rises monotonically with
capacity to 0.900 / FORG 0.033 at K=160. Earlier: `phase2_forgetting.py`
beat a neural net; `simulate_scenarios.py` measured a second vocabulary
learned to 100% coverage with ~15% forgetting of the first.

**Scope caveats — the important ones.** **Not SOTA.** A supervised
prototype baseline wins raw accuracy outright (0.872) and experience replay
tops the entire ladder on both axes (0.913 / 0.105). Bar-crossing accuracy
is reachable only as a protocol variation (K=160), and the cost branch has
a **measured floor of 2.24× the bar's KB-per-accuracy-point** (phase 33h,
held-out seeds) — arithmetic, not tuning. Every public statement about
continual learning must carry the "not benchmark dominance" clause that
phase 33 wrote.

**The boundary-freedom claim is retired (phase 38, T6.1).** This item used
to argue that the differentiator was needing no task boundaries, since EWC
must snapshot Fisher at one and replay must balance its buffer at one.
Measured, that is wrong: removing the boundary *signal* with the stream held
fixed costs the gradient arms essentially nothing — mlp-seq +0.000, **EWC
+0.056 (it gains** from a misaligned fixed-period schedule versus being told
the true task ends**)**, replay −0.005 for reservoir sampling. The
forgetting is produced by **sequential blocking of the stream**, and a task
boundary is only the annotation of that blocking. On a drifted,
re-emerging — i.e. interleaved — stream, the gradient arms recover almost
the whole gap unaided (mlp-seq 0.296→0.924, EWC 0.318→0.910) while the
organism moves +0.003. What remains true and is now pinned exact in harness
§14: the organism's accuracy is **bitwise identical** with and without the
boundary signal, because it consumes none. That invariance is architecturally
real and worth stating; it is not worth *much*, and this item no longer
claims otherwise. Do not reintroduce "needs no task boundaries" as a selling
point without a protocol where boundary information demonstrably pays.

**Mini-roadmap.**
1. *Implement* — ~~**T6.1 / phase 38**: re-run the ladder on a
   **boundary-free** stream... EWC needs boundaries to snapshot Fisher
   information, replay needs them to balance its buffer, the organism needs
   neither.~~ **DONE 2026-08-08 and the premise was falsified** — see the
   scope caveat above. The gradient arms do not need the boundary; they
   need the stream not to be blocked. Recruitment also does *not* localize
   transitions (z = −1.53, wrong sign, 0/5 seeds). The honest-framing
   clause was exercised: the gap did narrow, but because the gradient arms
   *improved*, which is not a result for us. **Replacement item, and the
   sharper question this exposed**: sweep how *blocked* a stream must be
   before boundary annotation pays for anyone. That is the experiment that
   would show whether this axis has a differentiator on it at all.
2. *Improve* — **T6.2 / phase 39**: factorize what actually protects old
   memories (window, victim rule, `p_decay`, headroom) against a
   random-eviction control, so retention becomes tunable rather than
   emergent.
3. *Perfect* — longer task sequences and re-emergence (classes that come
   back), which is where slot recycling either shines or breaks; then
   restate the claim on the axes the ladder cannot score: unsupervised
   representation + structure learning + persistence in one mechanism.

---

## N5 — Category validity without geometry

> **In plain terms.** When the system sorts words into groups on its own,
> how do you know the groups are real and not wishful thinking? The usual
> test asks whether the clusters sit far apart in space — the wrong
> question for word categories, which overlap heavily but *behave*
> differently. Better question: do the groups predict each other? Then
> shuffle the labels thousands of times to see what pure luck looks like.
> Ours beat luck by a margin you'd essentially never see by chance. We also
> found that the standard way of deciding *how many* groups quietly fails
> here — it always answers "more, please," forever. Our replacement asks
> which number keeps every group doing a distinct job. It said six; a
> human-labelled grammar check it had never seen also said six.


**Claim.** Silhouette is the wrong certificate for soft distributional
categories. Class-bigram mutual information against a measured permutation
null certifies validity; category-profile *distinctness* selects k — and
**k-selection is a parsimony problem, not a prediction problem**.

**Evidence.** `phase24_category_validity.py`: MI clears its null by 14–98σ
at every k≥4 where silhouette is flat (~0.03–0.044); MDL *and* held-out
class-bigram perplexity both run monotonically to the finest k (a
class-based bigram model with k²+V params never overfits at 408K pairs);
distinctness peaks at k=6, which **equals** the V-measure argmax against
universal POS tags (0.547) that the label-free criterion never saw.

**Why it's novel.** The negative is the contribution: two standard
model-selection criteria are shown to be structurally incapable of picking
k here, with the reason measured. The replacement is label-free and
validated against an oracle it never touched.

**Scope caveats.** Phase 27 is a real dent: at 5.22M words distinctness-
argmax **collapsed to k=2** (the pre-registered falsification fired) and
z fell at every comparable k, because the null's MI floor rises with k
faster than measured MI. Validity itself survived overwhelmingly (17–61σ)
— categories are real, they did not *sharpen*. A confound is on record
(MIN_COUNT 150→1500 changed vocabulary composition toward function words),
so the failure is not yet attributable.

**Mini-roadmap.**
1. *Implement* — done (phase 24), pinned.
2. *Improve* — resolve phase 27's k-collapse: re-run at matched vocabulary
   composition to separate "scale breaks distinctness" from "the MIN_COUNT
   confound broke it". Until that is done, k-selection at scale is **open**,
   and N3/N5 both inherit the uncertainty.
3. *Perfect* — hierarchy-aware k-selection (the phase-29 dependency), and a
   distinctness variant whose null is stable in k.

---

## N6 — Bitwise persistence and stable symbol identity

> **In plain terms.** Save it, shut it down, load it back, carry on — and
> what you get is *identical, digit for digit*, to never having stopped.
> That sounds mundane and isn't: it means the thing you reloaded is
> provably the same individual, not a good-enough copy. On top of that,
> every memory keeps a stable name even as memories merge or get recycled,
> so "memory #47" still means the same memory next month. For a product
> whose entire promise is "it will remember you," this is the whole
> ballgame — and it's a promise a normal checkpointed model can't honestly
> make.


**Claim.** Save → load → continue is **bitwise identical to never
stopping**, and symbol identity survives fusion, recycling, and
consolidation without perturbing a single bit of organism state.

**Evidence.** `organism_state.py` (E3): schema-versioned .npz, rng state
included; pinned in harness §8 — max |Δxi| = |ΔP| = 0, deterministic
replay, cross-backend restore. `organism.py::SymbolRegistry` (T3.3): driven
only from EventBoundary's existing `commit`/`remap`/`invalidate`, opt-in,
and registry-on state is bitwise identical to registry-off on both
backends; both backends emit identical identity events in the same order.
Schema now v3; v1 and v2 files still load. `organism_compress.py` (33g)
adds 3.85× smaller stores at **exactly zero** accuracy cost.

**Why it matters.** Individually ordinary engineering; together they mean
an episodic memory that is *provably the same organism* after a restart —
which is the defining property of the first intended product, and the thing
a checkpointed neural net cannot honestly say.

**Scope caveats.** Persistence is proven for the mechanism's own state, not
for downstream artifacts; the registry is opt-in and observational, so
nothing yet *depends* on stable IDs. Compression is lossless at measured
scales, not proven so in general.

**Mini-roadmap.**
1. *Implement* — done (E3 v3 + registry + compression).
2. *Improve* — **T4.1 / phase 31**: make identity survive *damage*, not
   just serialization — registry-identity under lesion, with bitwise-E3
   restore as the sharpest possible control.
3. *Perfect* — migrate downstream scripts off raw slot indices onto symbol
   IDs (the registry's whole point), and make one phase script depend on
   identity across a save/lesion/restore cycle end-to-end.

---

## N7 — Eviction under recruitment pressure

> **In plain terms.** The filing cabinet has a fixed number of folders. When
> it's full and something new arrives, which folder do you throw out? Only
> one that is both *unused recently* and *never got properly established*.
> There are **two** ways to get this wrong, and we have now measured both.
> The first is what "recently" means: set the bar at "untouched for ages"
> and the only candidates are your oldest memories, so you flush your own
> history. Set it short and the junk you made five minutes ago goes stale
> first, so you recycle clutter and keep the past. The second — which we
> originally expected to be a detail, and it is not — is **which** of the
> stale folders you pick. Picking at random instead of picking the least
> filled-in one is just as destructive as setting the window wrong, because
> the stale pile is not uniformly worthless: a random pick throws out
> folders that are typically **4–15× more filled in**. The two-line law:
> **the stale pile has to contain the present, or throwing out stale things
> eats your history — and you have to throw out the emptiest folder in it,
> or the same thing happens anyway.** The blunt version we can now state:
> **a badly set eviction policy is worse than never evicting at all.**


**Claim.** A slot budget where eviction fires **only** under recruitment
pressure, reclaiming the least-established *stale* slot, lifts the
capacity-flooding ceiling. **Two** parameters are load-bearing and neither
dominates: the staleness window *and* the victim rule. Both have a stateable
law, and either one mis-set is worse than running no budget at all.

**Evidence.** `phase33b_slot_budget.py`: pre-registered E=2000 **failed**
(ACC 0.665 → 0.511) because with a long window only previous eras are ever
stale, so eviction degenerates into a recency flush; at E=250 the stream's
own churn goes stale mid-task and is recycled first — ACC 0.712, FORG
0.169, fresh recruits every task. Pinned in harness §9.
`phase33f_eviction_ledger.py` exonerated the victim rule from the task-1
regression (H3: not an eviction pathology).
`phase39_retention_factors.py` (T6.2) then **factorized the mechanism**, one
factor at a time, 10 paired/held-out seeds, exact paired sign-flip nulls,
against a random-eviction control — and **corrected this entry's earlier
claim that the window is *the* load-bearing parameter**. Random eviction
loses to argmin-count at every window (dACC −0.28…−0.34, dFORG +0.44…+0.47,
10/10 seeds, exact p=0.0020), and the two axes' FORG spreads are
indistinguishable (0.4386 across rules vs 0.4244 across windows; 0.0536 vs
0.0554 with each axis's catastrophic setting dropped). The count-normalized-
by-era rule was tested as a falsification and is affirmatively **worse**
(dFORG +0.035…+0.060 at all four windows). All three pre-registered mundane
accounts were rejected. Pinned in harness §18.

**Scope caveats.** **Everything here is measured at 33c's blockedness** —
five disjoint 2-class tasks in strict sequence — and phase 38 measured that
the forgetting on this benchmark is produced by that sequential blocking.
None of it is a general retention claim; T6.6 (phase 49) is the target that
varies the axis. Also: one protocol, one dataset, and a window still chosen
post-hoc within a measured plateau (100–750). **`p_decay` remains
unmeasured** as a retention factor — 33c's readout decodes from `xi` alone
and never consults the transition graph, so the protocol has no instrument
for it; this is a gap, not a null result.

**Mini-roadmap.**
1. *Implement* — done, mechanism + E3-serialized clock.
2. *Improve* — **done, T6.2 / phase 39**, with the headline going the
   opposite way from the prediction: the victim rule is **not** secondary to
   the window, and must not be simplified away. Two knobs that *buy*
   retention were identified — slot headroom K, and phase 14's
   probation/confirm machinery, which 33c's own recipe leaves switched off
   (the latter grid-selected, held-out-confirmed, not yet a recommendation).
3. *Perfect* — make **both** parameters self-calibrating: the window from
   the measured live-slot revisit interval, and the victim rule from the
   measured spread of establishedness in the stale pool (phase 39 showed
   that spread is what argmin-count is exploiting, and it is measurable
   online). That would turn the law into a mechanism instead of two tuned
   constants, and close a Path-B requirement — products cannot hand-tune
   either knob per deployment, and phase 39 showed that getting either one
   wrong is worse than shipping no budget.

---

## N8 — Phase-superposition binding *(partial — honest)*

> **In plain terms.** A way of holding several things in mind at once by
> giving each a different timing offset — like several people humming
> different notes and still being able to pick out each voice. It works
> cleanly for about five things at a time. The catch: the system's own
> recall process squashes the chord back into a single note within a few
> dozen steps. So the capability is real but we can't yet hold it open.
> It's written down here specifically so we neither forget it nor oversell
> it.
>
> **Added 2026-08-09.** There is a second catch, and we found it by
> looking rather than by arguing: in every setup we actually run, the
> system never hums more than one note in the first place. We measured the
> stored memories and the timing channel is essentially empty — and the
> part of it that is there is not read by anything downstream. That is not
> a flaw in the design; it is a consequence of two dials being set the way
> they are, and we now know which dials and roughly what turning them
> would cost.
>
> **Added 2026-08-14 — the owner has now ruled, and this claim is edited
> down rather than deleted.** Since the channel was measured empty and
> nothing downstream reads it, the project has decided to **stop saving it
> to disk**, which nearly halves the cost of stored memory. So this
> capability is now **deferred, not merely unsolved**: it is no longer
> something the system is quietly carrying and might one day switch on. To
> revive it you would have to clear the two obstacles already measured —
> the readout cannot see the channel until it is ~230× stronger than it is,
> and the chord collapses within a few dozen steps — *and* change the
> storage format back. That last part is genuinely reversible, so nothing
> is destroyed; it is simply no longer free. **The system still computes in
> full complex numbers.** What changed is only what gets written down.


**Claim, scoped down.** Relative phase is a real binding code — perfect
identity readout to ~5 items at N=256 and 1.00 pair-grouping — **but** the
recall pull collapses superpositions in 9–56 steps.

**Evidence.** `phase16_phase_binding.py`, including the collapse
measurement, which is the constraint rather than a caveat.

**Scope, tightened 2026-08-09 (T7.1, phase 43) — measured, not
premise-derived.** Every superposition phase 16 binds is **hand-built**;
nothing in the perception path has ever produced one. Phase 43 audited the
stored memories directly and found the imaginary channel empty to a median
residual of **2.6e-04** (99.8% of live slots below 1e-2) on the 33h digits
arm, and to 1.2e-04–1.7e-04 on both text corpora — and the per-slot phase
is **gauge**: rotating every slot by an independent random phase moves the
scored readout by exactly zero. So the binding code is real (phase 16) but
**the organism as configured never writes to it**. The reason is measured
and it is a parameter regime rather than a structure: `perceive` computes a
phasor sum `Σ_j b a^j x_{t−j}`, and its phase content scales as omega² and
peaks at shallow settling, so the committed settings (omega/g_in = 0.0375,
hold=8) suppress it. Two consequences this claim must carry: making the
channel carry content needs **no new dynamics**, and any arm that does so
must also change the readout, which phase 43 measured to be blind below a
residual of **R ≈ 0.06**.

**Status.** A capability with an unsolved protection problem **and an
unused channel**. It is listed here so it is not quietly forgotten or
quietly oversold.

**Mini-roadmap.**
1. *Implement* — a protection mechanism (gating the recall pull during
   binding windows; or a separate slow-timescale channel).
2. *Improve* — show a downstream task that *needs* binding and improves
   with it — otherwise this stays a curiosity.
3. *Perfect* — bind at the symbol layer (N6's registry) rather than the
   field, if field protection proves impossible; record that pivot as a
   measured negative if it comes to it.

---

## N9 — The measurement discipline *(the underrated one)*

> **In plain terms.** Before running an experiment we write down what we
> expect **and** what result would prove us wrong — then commit that to the
> record before we see the answer. We simulate what pure luck would produce
> and refuse to believe anything that doesn't beat it. We keep our failures
> in writing, in the same detail as our wins. And we re-run everything from
> different random starting points, because three separate times now a
> beautiful-looking result turned out to be a fluke — including one we
> caught and threw away *after* it looked like a win. Most research code
> cannot tell you which of its numbers would survive that treatment. This
> one can, and that's the reason any of the claims above are worth
> anything.


**Claim.** This codebase can tell you which of its numbers would survive
reseeding. Most cannot.

**Evidence.** Pre-registration before committed runs (phases 25/26/27/33b–h
all record predictions *and* their misses); nulls simulated at the actual
sample size; negative results preserved as deliverables (phases 8, 9, 17,
33b's E=2000, 33g's low-rank arm); a 65-check regression harness pinning
every headline behavior across two backends; and the selection-bias lesson
paid for three times — T1.6's decoder win vanished on reseeding, T1.8's
low-rank arm sign-flipped, and **T1.9 correctly declined to bank a 1.96×
result that failed held-out confirmation**. Also on record: two sessions
independently ran phase 27 on different hosts and reproduced every number —
and a third (T6.5, 2026-08-08) re-ran it end-to-end on a fresh 42-book fetch
with **every committed anchor exact**.

**Scope caveat, added 2026-08-08 (T6.5/phase 42) — the discipline had a
blind spot on the engineering side.** Mechanism numbers are reseeded and
held out; *performance* numbers were pinned as single absolute values from
single runs. Phase 42 measured the same perceive load 43.7–58.2K frames/s on
one host on bit-identical work (±20%), a band that straddles the pinned 58K,
so that pin could neither confirm nor refute a regression. The fix is the
same discipline applied one domain over: a **same-session, interleaved A/B
against a reference tree** (phase 42 ran three trees × six rounds and found
the current tree indistinguishable from pre-33g and pre-T3.3, medians
49.8/50.1/50.6K). A performance claim quoted without its run-to-run spread
is the timing analogue of an unreseeded result.

**Mini-roadmap.**
1. *Implement* — done in practice, now written down here.
2. *Improve* — **T3.2**: corpus-tier harness checks (no longer blocked on a
   reproducible Gutenberg fetch — T6.5 re-fetched all 42 phase-27 books with
   live title verification, so the fetch is demonstrated reproducible), so
   the real-text headline numbers are pinned like the synthetic ones.
   Alongside it: pin *performance* as a same-session A/B ratio, never as a
   bare absolute number.
3. *Perfect* — codify the reseed + held-out rule as a harness tier any
   phase can call, rather than a convention each agent re-implements.

---

## What is *not* novel (state this before a reviewer does)

- **The substrate.** Complex-valued oscillator attractor networks sit in a
  deep lineage (Hopfield, Kuramoto, complex-valued associative memories).
  Novelty lives in what was built on it and what was measured — not the
  physics.
- **Distributional category induction.** Brown clustering and PPMI are
  standard; using PPMI to kill frequency-magnitude bias was a *fix for a
  known trap*, well applied, not a new idea.
- **Prototype/exemplar continual learning.** The prototype baseline that
  beats us is old and simple. That is the point of keeping it on the ladder.
- **Benchmark performance.** Nothing here is SOTA on a standard benchmark,
  and phases 33/33c–h say so with floors attached.

## The one-sentence claim

**One continuously-running system that perceives, remembers, learns world
structure, reasons with its perception switched off, and persists exactly —
online, single-pass, label-free, gradient-free.** Every clause has a phase
script and pinned harness numbers behind it; the clause that is *not* there
is "and wins benchmarks."

> **"Single-pass" audited 2026-08-09 (T7.4, phase 46), because the language
> recipe that produced the headline numbers runs fifteen epochs and a claim
> must not rest on an axis the production recipe violates.** All three
> measured axes survive ONE pass, and the audit's own prediction — that
> coverage would be the axis that pays — was wrong in the useful direction.
> Word coverage at 1 epoch vs 15 (fables): **327/376 → 320/376**, i.e. the
> extra epochs do not buy coverage, they consolidate it away. At scale
> (gutenberg8, 1 vs 3 epochs): **387.3 → 388.3 of 395**. Category validity
> clears its permutation null on a single pass at **z = 9.1** (fables) and
> **z = 32.0** (gutenberg8), strengthening to 12.7 / 38.3 with more passes.
> Polysemy detection at 1 epoch already fires on **50 words** above phase
> 21's measured noise floor (gutenberg8), 3% from the 3-epoch count.
> **Scope, load-bearing**: the 1-vs-15 comparison is measured on the small
> corpus only (the large one was run to 3 epochs — 3.3M frames/epoch), and
> the detection axis has a wide 3-seed spread (39–71 at one pass), so it is
> "not epoch-sensitive at the resolution three seeds resolve", not
> "epoch-invariant". The honest wording is therefore **single-pass on every
> axis we can measure, with multi-pass strengthening the category
> certificate and nothing else**.
