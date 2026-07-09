# DeFAI

Research codebase exploring whether complex-valued oscillator attractor
networks can do online, unsupervised cognition: form memories from a noisy
stream, learn the world's transition structure, and generate plausible
trajectories with no input — all in one continuously-running dynamical system.

The core abstraction is the **organism** (`organism.py`): a field of
Stuart-Landau-style complex oscillators that stores patterns as attractors,
recruits new memory slots competitively when input is novel, learns Hebbian
transitions between slots, and recalls by fatigue-driven itinerancy biased by
the learned transition prior.

Every experiment is a standalone script: `python <file>.py` prints its own
setup, baselines, results, and an honest verdict. Negative results are kept
deliberately — they are part of the scientific record.

## Setup

```
pip install -r requirements.txt
python organism.py          # core demo: memories + structure + recall
```

## Project arc

| Stage | Files | Question |
|---|---|---|
| Dynamics | `multi_attractor.py`, `itinerancy.py` | Can intrinsic dynamics itinerate among stored patterns without input? |
| Learning the world | `learn_core.py`, `learn_world.py`, `train_*.py`, `integrated_model.py`, `scaled_model.py` | Can the generative core be learned from the stream, and where does it stop generalizing? |
| Phase 1 | `organism.py` | Consolidated pipeline: perceive → consolidate → recall. |
| Phase 2 | `phase2_*.py` | Real data (digits): vs k-means, drift detection, catastrophic forgetting vs a neural net, joint integration. |
| Phase 3 | `phase3_*.py` | Prediction through the learned transition graph; isolating mechanism vs perception quality. |
| Phase 4 | `phase4_*.py` | Mini language model: 30-word vocabulary, cyclic grammar, generation quality, transition fine-tuning. |
| Phase 5 | `phase5*.py` | Scale (100 words), first polysemy attempt, learned (word2vec) embeddings. |
| Phase 6–7 | `phase6_continuous_manifold.py`, `phase7_context_perceive.py` | Context carry during recall (wrong place), then conjunctive coding during perceive (right place, but corpus had no real ambiguity). |
| Phase 8 | `phase8_true_polysemy.py` | Genuinely dual-role words (`fish`/`duck`/`bear` as both ANIMAL and ACTION, identical embedding). **Negative result, root-caused.** |
| Phase 9 | `phase9_centroid_consolidation.py` | Occurrence-centroid consolidation (2/3 exact splits, best yet) + replay refinement (negative result: EM inherits online mixing). |
| Phase 10 | `phase10_context_primed_settling.py` | Context as field carry-over: full functional disambiguation with no alpha parameter, seed-stable; attractor pull during perception is harmful (negative result, both orderings). |
| Phase 11 | `phase11_transition_decay.py` | Transition-count decay fixes concept-drift inertia: adaptation 0.44 → 0.87 at `p_decay=0.001`, generation unaffected. |
| Phase 12 | `phase12_capacity_scaling.py` | Storage scales perfectly to 1000 words (coverage/purity 1.00, N=128); generation under crowded embeddings is the scaling casualty, not slot count. |
| Phase 13 | `phase13_recall_dynamics.py` | Recall overhaul (`recall2`: lateral inhibition + hop commitment): plateau 0.62 → 0.995, crowding collapse 0.05 → 0.78; noise case improved 0.26 → 0.47, residual is perception-side. |
| Phase 14 | `phase14_noise_robust_perception.py` | Probationary recruitment: junk memories eliminated, strict win to σ=0.2, with an analytic boundary at σ*≈0.24 where single-shot recruitment provably fails (revisits fall below the confirmation bar). |
| Phase 15 | `phase15_action_conditioned.py` | Action-conditioned transitions (efference copy): forward model 1.00 vs 0.31 action-blind; navigation by planning entirely inside the learned model, 1.00 vs 0.34 random. |
| Phase 16 | `phase16_phase_binding.py` | Phase-superposition binding is a real code: perfect identity readout to ~5 items (N=256), 1.00 pair-grouping by relative phase; constraint measured — the recall pull collapses superpositions in 9–56 steps, so binding needs protection, not readout. |
| Phase 17 | `phase17_pooled_recruitment.py` | Pooled probationary recruitment (`perceive(pool=True)`): evidence pooled across occurrences *before* keep/discard. Strict win through σ=0.25 (past σ\*≈0.24; at σ=0.2: zero junk hops, generation 0.67 → 0.81), clean case improved (coverage 0.88 → 1.00), storage crosses σ\* to a new boundary at σ≈0.4 exactly where the pairwise-margin analysis predicts. Honest residual: generation beyond σ\* is polluted by frozen mixture slots from bootstrap routing chaos — five cures tried, each failure measured. |
| Phase 18 | `phase18_ambiguity_gate.py` | Soft ambiguity-gated routing (`perceive(amb=...)`): evidence into provisional pools weighted by attribution confidence (margin over the runner-up), acting *before* bootstrap mixing instead of trying to undo it. Clean case identical; σ=0.2 win extended (0.810 → 0.862, exactly 26 slots); at σ=0.25 generation 0.509 → 0.66–0.67; at σ=0.3 dominates phase 17 at full coverage (0.271 → 0.406) *and* the phase-14 gate on both axes (0.584/0.69 vs 0.489/0.50). Measured constraint: 77% of greedy assignments are wrong at σ=0.3 during bootstrap and margins barely discriminate — `amb` tunes an information-limited purity–coverage frontier; every hard variant slides along it. |

Side experiment: `strain_propagation.py` (Kuramoto "code bath" refactor-wave test).

## Language track (parallel line — own phase numbering)

A second research line forked at phase 4–5 and was developed independently
(branch `claude/defai-investigation-gsn2oy`, merged back 2026-07); its files
(`phase9_track_a_residual.py` … `phase21_working_polysemy_detection.py`,
`polysemy_organism.py`, `real_text_corpus.py`) use their **own phase
numbering** — see `FABLE_HANDOFF.md` for the full brief and its commit
messages for the lab notebook. Headline results:

- **Predictive (Myhill–Nerode) polysemy test** (its phase 12): split senses
  by whether context changes *successor predictions*, not by representation
  distance. 0.30-bit margin, zero false positives, fully online
  (reproduced post-merge: `phase12_predictive_split_test.py`).
- **Category discovery fixed for real corpora** (`discover_categories_v2`):
  PPMI-transform the transition profile before clustering — fragmentation at
  small scale and the 2-blob collapse at 547K words were both
  frequency-magnitude bias, not data problems.
- **First unsupervised polysemy detection in real natural language** (its
  phase 21): "right" (622 occurrences over 8 books) clears a directly
  measured noise floor (0.070 vs 99th-percentile null 0.043), plus 37
  plausible multi-role English words — the honest full ranking.
- **Hybrid generation** (its phase 18/18b): pure continuous generation fails
  (role signal ~3% of similarity scale); grammar as a small unsupervised
  discrete FSM + everything else continuous reaches 0.818 grammaticality at
  full coverage with live disambiguation.

## Unified track (phase 22+, numbering continues from the language track)

`phase22_unified_realtext.py` wires both tracks into one end-to-end loop on
the committed fables corpus (2.4K tokens, 376 words): perceive → emergent
categories → polysemy vs a per-word permutation null → layered generation,
zero labels anywhere. Findings: perception covers 335/376 words
(recruit floor re-swept to 0.85; the core pooled/ambiguity stack collapses
here — measured cause: 68% of real PPMI/SVD embeddings have a neighbor above
the 0.7 online-fusion bar, so pool-mode constants calibrated for
near-orthogonal patterns merge distinct words); generated word bigrams hit
the corpus at ~11× random with 221/376 words covered; and **category
emergence is the measured bottleneck at this corpus scale** (best silhouette
0.011, 291-word blob) — the same scale wall the language track measured in
its phases 19–20, so the next lever is corpus scale, not mechanism.

`phase23_unified_large_corpus.py` pulls that lever: the *same four stages,
same scorers* on ~547K words of public-domain prose (the 8-book phase 20/21
corpus, re-fetched from Gutenberg into `/tmp/gutenberg_corpus/`; the script
prints the `curl` block if the corpus is missing). The pre-registered
question was narrow — does ~100× more text turn phase 22's *wired* verdict
into *closed*? The category-emergence bottleneck it named is **relieved**:

- **Perception** covers 378/395 words at recruit=0.75 (phase 22's fables-tuned
  0.85 peak does *not* transfer — it drops to 375; the core pooled/ambiguity
  stack still collapses, 85/395, confirming the correlated-embedding cause
  persists at scale).
- **Categories** go from phase 22's single 87% blob to a **balanced 3-way
  split** (largest 42%) that is grammatically legible — a verb/modal cluster
  (`am, are, be, been, can, come, could`) and a noun-phrase cluster (`a,
  another, beautiful, bed, big, bird, boy`). Silhouette only doubles
  (0.011 → 0.024) and stays below any conventional cluster-validity bar:
  silhouette measures geometric separation, the wrong certificate for soft
  distributional categories.
- **Polysemy** is now robust: **114 words clear their own per-word permutation
  null**, `right` reproduced with a ~3× tighter null than phase 21
  (gain 0.115 vs p99 0.015; phase 21 was 0.070 vs 0.043), and the top of the
  ranking is highly plausible (`long, soon, end, dead, right, young, old,
  like`). Honest caveat: predictive gain measures *distributional*
  context-sensitivity, broader than lexical polysemy (function words like
  `not`, `with`, `or` also clear).
- **Generation** reuses **57.6% of corpus bigrams verbatim** (vs 0.234
  random — the sparse-vocab 5× multiplicative gate is a scoring artifact and
  is corrected in the verdict) and its **category flow now beats random**
  (−1.334 vs −1.607), reversing the phase-22 blob artifact where random
  outscored the generator.

**Verdict: the scale lever works — relief, not unqualified closure.** Every
stage now clears chance end-to-end and unsupervised; the residual honest
constraints move from "categories don't form" to (a) they form and are
grammatically legible but not geometrically well-separated (silhouette stays
low), and (b) predictive gain conflates lexical polysemy with grammatical
context-sensitivity. Two of phase 22's verdict gates were shown to be
calibrated for the sparse fables vocabulary and recalibrated in the script,
transparently (see its verdict block).

`phase24_category_validity.py` attacks phase 23's sharpest residual — the
categories are grammatically legible but silhouette (geometric) can neither
certify them nor pick *k*. It replaces silhouette with the project's own
prediction-first framing (phase 12's "good splits change what you predict";
phase 18b's "grammar is a small state machine"), using an oracle POS tagging
**for evaluation only, never in any mechanism**. Result — **both threads
close, with a measured negative between them**:

- **Validity** — class-bigram mutual information `I(C_t; C_{t+1})` (the Brown-
  clustering objective) against a **directly measured permutation null**
  clears it by **14–98σ** at every k≥4 (z=65 at the selected k), where
  silhouette is flat (~0.03–0.044): the certificate silhouette couldn't give.
- **k-selection, the informative failure** — two-part MDL *and* held-out
  class-bigram perplexity **both run monotonically to the finest k (k=12)**.
  Measured cause: a class-based bigram model with k²+V params never overfits
  at 408K pairs, so a finer partition always predicts at least as well —
  k-selection is a **parsimony** problem, not a prediction one.
- **k-selection, the fix** — category-profile **distinctness** (min pairwise
  distance between categories' transition profiles) **peaks at k=6** then
  collapses when a split creates a predictively-redundant category.
- **Oracle confirmation (eval only)** — distinctness's k=6 **equals** the
  V-measure-argmax against universal-POS tags (0.547), which the label-free
  criterion never saw; silhouette's k=7 scores 0.458, MDL's k=12 scores 0.511.

So **open threads #1 (wrong metric) and #3 (automatic k-selection) close
together**: MI-vs-null certifies validity, distinctness selects k, and both
drop into the unified loop's stage B. The experiment is deliberately isolated
from the oscillator (the criterion is a property of a category assignment plus
corpus bigram statistics — the same stage-B object), the same isolation
discipline as phase 3.

## Polysemy (core track, phases 8–10): functionally solved

Goal: one word-form occurring in two senses should recruit **two** memory
slots, driven only by context. Phase 8 shows the additive context blend
(`z_store = normalize(z + alpha*ctx)`) never splits a slot, and
`verify_residual_gating.py` preserves why this is analytic, not a tuning
failure:

1. **Recruit gate can't fire.** Stored-vs-new overlap for one word under two
   contexts is `1/(1+alpha^2)`, which stays above the 0.5 recruit threshold
   for all `alpha < 1` — and at `alpha >= 1` word identity itself collapses.
2. **Consolidation destroys any split.** Two same-word/different-context
   slots overlap ≈ 0.89 at `alpha=0.35`, above the 0.84 merge threshold.
3. **The metric is blind.** Occurrences are assigned to slots by pure word
   embedding, which cannot attribute two senses to two slots.

The fix is **word-conditional residual gating**: compare same-word patterns
in the context residual (component orthogonal to the word direction), where
distinct contexts separate cleanly and independently of `alpha`. Status:

- `verify_residual_gating.py` — the theorem + fix, verified numerically.
- Track A (`phase8b_track_a.py`) — the real prev-attractor context signal IS
  role-aligned (cluster purity 0.74–0.83 vs ~0.52 chance) but thin: residuals
  track the specific previous word, and the role signal is only the shared
  category component (within-role overlap ~0.30 vs ~0.22 across).
- Track B (`phase8c_residual_gating.py`) — partial success, split along that
  margin: with a fast slot EMA all three dual words get role-specific slots
  whose successor statistics match their role's grammar (functional
  disambiguation, the phase-8 goal), at the cost of same-role over-splitting;
  a slow EMA yields an exact 2-way role-aligned split for `fish` but leaves
  `duck`/`bear` unsplit. The `alpha=0` control never splits.
- Phase 9 (`phase9_centroid_consolidation.py`) — consolidation by
  occurrence-centroid residuals lifts the best structure to 2/3 exact 2-way
  splits with 3/3 functional coverage. The remaining failure is online
  mixing: a slot that absorbs both roles early cannot be unmixed afterward —
  replay/EM re-sorting (stage 2) provably inherits that mixing through its
  initialization (negative result, preserved).
- Phase 10 (`phase10_context_primed_settling.py`) — context becomes
  organism-native: the field state carried over between words IS the context
  (no alpha blend, so the dilution theorem never applies). Full functional
  disambiguation (3/3 role coverage, 10/10 grammatically correct dual-slot
  successors), stable across seeds; the no-carry control never splits. Twice-
  confirmed negative result: attractor pull during perception suppresses
  splitting in both orderings — perception should read the field, not bend
  it. Open: exact 2/1 slot structure (same-role duplicates persist at
  merge thresholds that don't endanger the cross-role split).

The language track then carried polysemy to real text with a different,
prediction-based criterion (see its phases 12 and 21 above) — the two
approaches are complementary: phase 10 splits *representations* online
inside the field; the language track *detects* sense structure from
corpus statistics and now needs the field mechanism to act on it.

## Capability envelope (measured)

`simulate_scenarios.py` exercises the stock organism under local-AI usage
patterns (26-word world, cyclic grammar; oracle ceiling 0.88, chance ~0.33):

| Scenario | Result |
|---|---|
| Cold start | 85% vocabulary coverage after only **500 words**; generated-sequence grammaticality plateaus at ~0.55–0.62 regardless of further data — the ceiling is architectural (recall dynamics), not data-limited. |
| Noise | Memory formation survives sigma=0.3 corruption (92% coverage) but **generation collapses** (0.19); at sigma=0.6 coverage degrades (65%). Perception is robust; recall is noise-fragile. |
| Concept drift | Grammar reversal mid-stream: the organism adapts online (0.12 → 0.44 on the new regime after 4000 words) but with heavy inertia — Hebbian counts never decay, so old evidence outvotes new roughly in proportion to its volume. |
| Continual learning | The standout: second vocabulary learned to 100% coverage with only **15% forgetting** of the first, and combined generation at 0.70 grammaticality — the architecture's strongest deployment property (consistent with `phase2_forgetting.py` beating a neural net). |

Actionable gaps this exposed — status after phases 11–12:

- ~~Transition-count decay (drift inertia)~~ **fixed**: `p_decay=0.001`
  (phase 11) restores near-ceiling adaptation for a 7pp stationary cost.
- Capacity: **not a bottleneck** — storage is perfect at 1000 words / N=128
  (phase 12); prediction degrades gracefully (0.89 → 0.79).
- ~~Generation under crowded embeddings~~ and ~~the grammaticality
  plateau~~ **fixed** (phase 13): `recall2` (top-k lateral inhibition +
  debounced hop commitment) takes the clean small world to 0.995 — above
  the corpus's own 0.88 noise level — and the V=300 crowding collapse from
  0.05 to 0.78. Both mechanisms are required at scale.
- Noise: **boundary pushed three times, now split by capability**
  (phases 14, 17, 18).
  Phase 14's probationary recruitment (`perceive(confirm=3)`) eliminates
  junk and wins strictly to σ=0.2, with an analytic wall at σ*≈0.24 where
  token-vs-memory overlap 1/√(1+σ²N) falls below the 0.6 confirmation bar
  and any single-shot scheme must fail. Phase 17 implements the conjectured
  fix — pool evidence across occurrences *before* keep/discard
  (`perceive(pool=True)`: saccade-gated evidence, annealed acceptance bars,
  running-mean pooling, held-out graduation, online fusion) — and it
  dominates phase 14 at every σ measured: strict win through σ=0.25,
  **storage** (coverage/purity) intact to σ=0.35 with the new wall at
  σ≈0.4, exactly where same-word token-token overlap stops clearing the
  cross-word fluctuation floor. Phase 17's residual — frozen mixture slots
  from bootstrap routing chaos polluting **generation** (0.27 at σ=0.3),
  which no post-hoc per-slot statistic could undo — is attacked at the
  source by phase 18: evidence routing weighted by attribution confidence
  (`perceive(amb=...)`). Clean case identical, σ=0.2 win extended
  (0.810 → 0.862), σ=0.25 generation 0.51 → 0.66–0.67, and at σ=0.3 it
  dominates phase 17 at full coverage (0.27 → 0.41) and the phase-14 gate
  on both axes (0.584/0.69 vs 0.489/0.50). Open: the purity–coverage
  frontier itself — Part A of phase 18 measures 77% greedy-assignment
  error during bootstrap at σ=0.3 with near-uninformative margins, so
  single-token attribution is information-limited; the conjectured way
  off the frontier is temporal-context attribution (let the learned
  transition prior lend confidence to ambiguous tokens).

## Current target

The project's standing goal: one continuously-running oscillator field
that perceives, remembers, learns world structure, and generates — now
validated on synthetic worlds (core track, phases 1–18) with first
footholds on real language (language track, through its phase 21). The
open threads, roughly ordered by leverage:

1. **Unify the tracks on real text** — *wired* (phase 22) then *scaled*
   (phase 23): the full loop runs unsupervised, and at the language track's
   547K-word scale (`phase23_unified_large_corpus.py`, the 8-book Gutenberg
   corpus) the category-emergence bottleneck is **relieved** — the 87% blob
   becomes a balanced, grammatically legible 3-way split, 114 words clear
   per-word polysemy nulls (`right` reproduced with a ~3× tighter null), and
   generation reuses 57.6% of corpus bigrams with category flow now beating
   random. Relief, not closure: silhouette stays below any cluster-validity
   bar — but that was the *wrong* certificate, and *phase 24* replaced it
   (see thread #3). The remaining residual: predictive gain conflates lexical
   polysemy with grammatical context-sensitivity (needs disentangling). Also
   still open from phase 22: pool-mode constants (fusion 0.7, 0.8-scale bars)
   need decorrelation or embedding-aware calibration before the core
   perception stack works on real embeddings — phase 23 confirmed the collapse
   (85/395) persists at scale.
2. **Temporal-context attribution** (core phase 18's residual): let the
   learned transition prior lend confidence to ambiguous tokens during
   routing — the conjectured way off the information-limited
   purity–coverage frontier beyond σ*.
3. ~~**Automatic k-selection + a category-validity metric**~~ **resolved
   (phase 24)**: silhouette is geometric — the wrong objective for soft
   distributional categories (it's flat ~0.03–0.04 and picks k=7).
   `phase24_category_validity.py` replaces it with two label-free criteria
   grounded in the project's prediction-first framing: class-bigram *mutual
   information vs a measured permutation null* certifies **validity** (14–98σ),
   and category-profile *distinctness* (a parsimony criterion) selects **k=6**,
   confirmed by an oracle POS V-measure (eval only) that the criterion never
   saw. Measured en route: predictive/compression criteria (MDL, held-out
   perplexity) monotonically over-split and cannot select k — k-selection is a
   parsimony problem, not a prediction one. Both selectors drop into stage B.
4. **Exact polysemy slot structure** (core phase 10): same-role duplicates
   persist at merge thresholds that keep the cross-role split safe.
5. **Phase-binding protection** (core phase 16): superpositions are a real
   code but the recall pull collapses them in 9–56 steps; binding needs a
   protection mechanism before it can be used.
