# DeFAI — Handoff Brief

> **Addendum, 2026-08-04 (onboarding verification pass).** Fresh clone
> re-verified on an independent machine: E1 harness 27/27 PASS on both
> backends (numpy 59.8s / numba 10.2s), `test_fastpath_equivalence.py`
> all green, `e2_benchmark.py` meets the pinned numba throughput
> (67.3K frames/s; 1.21 min full corpus perceive — the "13.0×" ratio is
> hardware-relative, the absolute numba number is the portable claim),
> and phase 35 reproduced exactly at 50–400 words (800-word point not
> re-run: memory-limited host). PR state: **0 open, 26 closed** — PR #23
> has landed (phases 32/33 are on main); the "in-flight PR #23" note in
> open thread 8 below is historical. New since this brief was written:
> phases 34 (Attractor Crowding Collapse) and 35 (hierarchical recall
> rescue, oracle caveat), and the **Path B RELEASE HOLD** pending the
> phase-33 gate — see the ROADMAP rows. Current top thread: phase 36
> (un-oracle phase 35 via `discover_categories_v2` + slot-budget policy,
> then re-run the phase-33 ladder).

Written 2026-07-15, replacing the original brief in full. You're taking over
an in-progress research project with no prior context — everything you need
is below. Where this brief and the code disagree, **trust the code** and flag
the mismatch to the user. `git log --oneline` is the full lab notebook: every
phase commit message is a self-contained entry explaining what was tried,
what broke, and why.

**Before writing any code: check the open PRs.** Multiple Claude sessions
work this repo in parallel, and on 2026-07-15 two sessions independently
built the same engineering item (E2) and had to reconcile after the fact.
`ROADMAP.md` is the coordination surface — claim work by its roadmap row,
and look at what's in flight first.

## Repo and goal

`m2windham/DeFAI`, ~70 phase/engine scripts, 30+ phases of work. The goal is
a **General Sentient Exploration model**: a continuous dynamical system built
on coupled-oscillator physics, meant to perceive, remember, learn world
structure, reason, and generate language without discrete tokenization —
meaning is a *position* in a continuous field, not a lookup in a vocabulary
table. Everything is online, label-free, single-pass, and gradient-free
inside the mechanism. That premise is load-bearing: external reviews keep
proposing gradient-era machinery (SSM cores, MoE, EWC, differentiable
memory) and the roadmap records exactly why each was rejected.

## Architecture (as of 2026-07)

`organism.py` — the core, split into two systems behind a narrow boundary
(the MIT/McGovern language/logic functional separation, PR #20):

- **`Organism` (perception, "language")**: complex Stuart-Landau-style
  oscillator field `z ∈ C^N`, `‖z‖ = √N`. `perceive(stream)` drives the
  field toward input embeddings; Hebbian competitive learning recruits and
  refines memory slots `xi`. Modes, each layering on the last: plain;
  `confirm` (phase 14: provisional slots must recur to survive);
  `pool` (phase 17: evidence pooling at saccade boundaries with annealed
  per-slot acceptance bars, online fusion, use-it-or-lose-it recycling);
  `amb` (phase 18: soft attribution-confidence gate on contested evidence).
  `fuse_bar` (phase 26) exposes the duplicate-fusion threshold for
  calibration. **`recruit` is a similarity floor, not a novelty threshold —
  higher = more eager recruitment.** This inverted reading cost real time
  once; don't re-derive it.
- **`TransitionGraph` (logic, "reasoning")**: owns the transition matrix P;
  mutations only via `observe`/`merge`/`retire`/`fold`. Phase-30 ops run on
  the graph with the field absent: `kstep` (multi-step inference), `rollout`
  (field-free imagination), `next_hops`/`plan` (Dijkstra planning).
  `Organism.P` is a delegating property so old phase scripts still work.
  Phase 40 (T6.3) went deeper on the same field-free footing, all additive:
  `dijkstra` (one algorithm, several notions of "best" — `next_hops` now
  delegates to it and is pinned bitwise identical to its old body),
  `confidence` (Wilson lower bounds — **the point estimate is
  evidence-blind: three visits all going one way reads as certainty**),
  `edge_quality`/`plan_reliable`/`path_report` (planning on reliability and
  reporting it, with the weakest link and its observation count),
  `plan_visit` + `avoid=` (conjunctive and negated goals),
  `passthrough_census`/`contracted`, plus `MacroGraph` (temporal
  abstraction) and `SparseTransitions` (a bitwise-equal sparse port — see
  the ROADMAP row for where it does and does not pay).
- **`EventBoundary`**: the single call-site where recognitions become
  relational knowledge — only confident, non-provisional events cross;
  fusion/recycling arrive as explicit `remap`/`invalidate` notifications, so
  recognition errors cannot silently rewrite the world graph.
- **`recall(steps)` / `recall2(...)`**: autonomous generation. Habituation
  (`h[k]` fatigue) is essential — anything downstream that skips it
  collapses onto one attractor. `recall2` adds top-k lateral inhibition and
  commit/debounce hop acceptance. `recall_directed(goal)`: the logic layer
  plans, the field renders — reaches goals at true shortest-path length.
- **`consolidate()`**: merges duplicate slots, prunes junk, builds `mem`
  (compact bank), `Pn` (normalized transitions), `kept_idx` (for
  reconstructing raw counts, needed by PPMI category discovery).

`polysemy_organism.py` — `PolysemyOrganism(Organism)`:
- **`discover_categories_v2(...)`** is THE category-discovery method:
  PPMI-transform of transition profiles (kills frequency-magnitude bias,
  the root cause of every category-discovery failure this project hit) +
  k-means with silhouette/balance-checked k. The older
  `discover_categories()` survives only for old scripts — known-broken on
  real text; never build on it.
- **`perceive_polysemy()` / `consolidate_polysemy()`**: online
  predictive-gain-gated, residual-gated sense splitting (see below).

Engineering spine (all landed, all pinned):
- **E1** `regression_harness.py`: 92 tolerance-based checks pinning every
  headline behavior (§1-5 core/noise/pool/categories/gain, §6 phase-30
  reasoning, §7 phase-26 calibration, §8 E3 serialization, §9 T1.2 slot
  budget/eviction, §10 T1.8 store compression, §11 T3.3 symbol registry,
  §12 T6.3 logic-layer depth, §13 T6.4 detection-driven sense splitting,
  §14 T6.1 task-free continual learning). Every backend, port, and
  calibration change must pass it before being believed.
- **E2** `fastpath.py`: Numba JIT backend for perceive/recall/recall2 +
  vectorized consolidate. Selected via `Organism(backend="auto"|"numba"|
  "numpy")` or `DEFAI_BACKEND` env; default `auto` (JIT when numba is
  installed). `DEFAI_BACKEND=numpy` gives the historical pure-NumPy
  reference. Measured **13.0× at phase-23 corpus shape (18.2 min →
  1.40 min)**; equivalence pinned by `test_fastpath_equivalence.py`
  (state agreement ~1e-15 on short streams; recall sequences match
  exactly — same RNG consumption order). **Do not treat "13.0×" or "58K
  frames/s" as a pin** — T6.5 (phase 42, 2026-08-08) measured a ±20% band
  (43.7–58.2K) on bit-identical work on one host, straddling that number.
  Detect throughput regressions with a same-session A/B against a reference
  tree; phase 42's did, and found the current tree indistinguishable from
  pre-33g and pre-T3.3. `organism_numba.py`'s
  `NumbaOrganism` is a thin alias kept for the harness/E3 seams — two
  sessions built E2 in parallel and the implementations were unified
  (the ROADMAP E2 row records both). Remaining bound is the K×N overlap
  matvec, not the interpreter: the next speedup is algorithmic, not more
  JIT. Phase 42 measured the matvec at 75% of a 14 614 ns frame at K=1580
  and corrected the reason: the 1.2 MB operand is cache-resident, so this is
  cache bandwidth rather than DRAM — which is why halving the operand width
  buys 1.48–1.77× (priced, not taken; compute width is pinned at
  complex128 by `Organism.perceive`'s dtype guard, deliberately).
- **E3** `organism_state.py`: schema-versioned .npz save/load, rng state
  included. Pinned: mid-stream save→load→continue is bitwise identical to
  never stopping; deterministic replay; cross-backend restore. Schema is
  now **v3** (T3.3 symbol registry); v1 and v2 files still load.
- **E5** `organism.py::SymbolRegistry` (T3.3): stable symbol IDs decoupled
  from slot indices, at the EventBoundary seam. Opt-in and observational.
  See open thread 6 below.

## The core intellectual results (read this even if you read nothing else)

1. **The polysemy result (phase 12, the single most important idea)**: to
   detect that "fish" is both noun and verb, do NOT ask "does its
   representation look different in context" — every version of that failed,
   for principled reasons (identity-preserving drift is necessarily small;
   thresholds don't transfer across scales). Ask instead **"does knowing the
   context change what I'd predict happens next"** — a Myhill-Nerode-style
   state-splitting test on successor-category entropy. Zero labels anywhere.
   Validated on synthetic data (0.30-bit margin, zero false positives,
   fully online), then on real text (phase 21): the word **"right"** (622
   occurrences, 8 Gutenberg books) cleared a directly measured noise floor
   (gain 0.070 vs 99th-pct null 0.043), plus 37 plausible multi-role words
   (turn, long, far, fast, hard, near, old, way) — the honest full ranking,
   not cherry-picked.
2. **Frequency bias, not data volume, was the category killer (phases
   19-21)**: small corpora fragmented category discovery, large corpora
   collapsed it into 1-2 giant blobs. The cause was frequency-magnitude bias
   in clustering raw transition profiles; PPMI transformation fixed both
   ends. The general lesson is baked into the rules below: verify the
   obvious "needs more data" hypothesis before believing it.
3. **Grammar is discrete; everything else is continuous (phases 17-18b)**:
   pure continuous generation failed measurably (role signal ~3% of the
   similarity scale — invisible to k-NN, and amplification doesn't help).
   The working architecture keeps grammar as a small unsupervised discrete
   category-FSM and keeps word identity/sense/meaning continuous
   (grammaticality 0.818 vs 0.839 discrete baseline, full coverage, live
   disambiguation).
4. **Reasoning runs without the field (phase 30)**: on the decoupled graph,
   k-step inference corr 0.995-0.999 vs permutation null ~0.45; symbolic
   rollout matches field recall at ~20× speed; directed recall reaches every
   goal at true shortest-path length. The logic/language split is not
   cosmetic — it does computational work.
5. **Perception collapse at corpus scale has two measured causes (phases
   25-26)**: embedding correlation (phase 25: decorrelating recovers
   85→216/395 coverage, then plateaus) and the absolute acceptance-bar
   constants (phase 26: percentile calibration passes its synthetic gate;
   see caveats below). Neither alone is the whole story.
6. **Honest external benchmark (phase 33, in-flight on PR #23)**: on
   class-incremental split-digits the organism holds 2× the retention of
   gradient baselines but is **not SOTA** — a supervised prototype baseline
   wins raw accuracy. The defensible public claims are scoped to:
   unsupervised representation + structure learning + bitwise persistence
   in one single-pass online mechanism.

## Rules of engagement (violating these re-derives closed dead ends)

- **Never use ground-truth labels inside the mechanism.** Oracles are
  capability ceilings for evaluation only, always labeled as such.
- **Negative results are the deliverable.** Write down what broke and why,
  with the same care as wins; that discipline is how phases 12, 21, and 26
  were findable.
- **Measure noise floors; never eyeball signal.** Simulate the null at the
  actual sample size. Multiple impressive-looking gains died this way.
- **When it looks like a data-volume problem, verify first.** Phase 20
  disproved its own predecessor's hypothesis by experiment.
- **Pre-register predictions** before committed runs (see phases 25/26/27
  rows for the format), and record partial/failed outcomes in the ROADMAP
  row rather than hiding them.
- **Tolerance bands, not bitwise equality, for ports** — but pin exact
  invariants where they're cheap (E3 round-trips ARE bitwise; recall RNG
  streams ARE identical across backends).

## Strategy: Path A / Path B

This repo stays **Path A**, the research line. Products are **hard forks**
(separate repos), never branches; discoveries flow downstream only through a
versioned engine when they have (1) a phase script, (2) pinned harness
numbers, (3) a release tag. **The fork gate (E1+E2+E3+Phase 26) is OPEN as
of 2026-07-13.** Remaining before the first Path B commit: cut a release tag
on main, create the product repo (first product: deployable episodic-memory
module — persistence is its defining feature, hence E3).

## Open threads, in rough priority order

1. **Phase 27 — 5M-word scale run** (50-100 books): unblocked by E2; wire
   phase 24's MI-vs-null k-selection into stage B. Pre-registered questions
   in its roadmap row.
2. ~~**Phase 26 real-text arm (V ≫ N)**~~ — **CLOSED 2026-08-06 (T2.2,
   phase 37).** Ported closed PR #19's pairwise-quantile `qcal` estimator
   (rank-free; minus its `bar_guard` recruit floor, which needed an
   organism.py hook main lacks); real corpus (395 words, N=50) coverage
   85 → 275/395 at alpha=0.5, well above phase 25's 216/395 decorrelation
   plateau, and 289/395 composed with decorrelation. Synthetic V=250≫N=50
   gate passed at all sigmas — σ=0 by genuine estimation; σ≥0.2 via the
   known-sigma fallback (non-separated modes decline to estimate rather
   than oscillate the way phase 32's fixed point did; qcal is one-shot). See ROADMAP row 37.
3. **Phase 28 — polysemy vs grammatical context-sensitivity** (the
   disentangling test, gold-POS for eval only).
4. **Phase 29 — recursive hierarchy** (the recruit/consolidate primitive one
   level up; the answer to "no structure above the category FSM").
   **BLOCKED 2026-08-09 on its own pre-gate (T7.5, phase 47), and the
   blocker is a measurement, not an opinion.** A chunk-inventory census at
   the category level clears a stream-permutation null (810.3 vs 350.4
   nats) and **fails a LABEL-permutation null outright — 810.3 against
   1279.8 on 5/5 folds**: a RANDOM partition at fixed category sizes
   compresses the stream better than the discovered one. That is the
   clustering artifact any Zipfian stream manufactures, and only the
   label-permutation control removes it. Phase 29's own pre-registered
   "level 2 learns nothing" therefore FIRES on this corpus. Do not open
   this thread without either a corpus where the label-permutation null is
   cleared, or a level-2 unit that is not a category bigram. Harness §17
   pins the artifact so a reversal has to be earned.
5. **Corpus-tier harness checks**: cheap now (E2), blocked on making the
   Gutenberg fetch reproducible inside the harness.
6. ~~**Stable symbol registry** decoupled from slot indices~~ — **CLOSED
   2026-08-07 (T3.3).** `SymbolRegistry` in `organism.py`, driven only from
   the EventBoundary's existing commit/remap/invalidate notifications, so
   there are no new call sites in the perceive loop. Opt-in
   (`Organism(symbols=True)`) and observational — registry-on state is
   bitwise identical to registry-off on both backends. IDs survive fusion
   (moved or aliased), recycling (tombstoned, never reissued),
   consolidation (a view, `mem_row`), and save/load (E3 schema v3; v1 and
   v2 files still load). Downstream scripts may still index `org.P` by
   slot; `registry.slot_of(sid)` is the index that stays correct.
   Migration proof: `phase33f_eviction_ledger.py`'s script-side birth-era
   replay reproduced on 40/40 slots, both backends, phase numbers
   unchanged. Honest scope: identity is lineage, not content -- pool-mode
   plasticity re-centers mature traces at the same rate by ID as by slot.
   See ROADMAP row E5 and harness §11.
7. ~~**E4 — GPU statistics tier**~~ — **DEFERRED 2026-08-08 (T6.5, phase 42),
   on measurement.** With phase 27 run, E4's premise is testable and fails:
   permutation nulls are 265s of the 1184s corpus-tier run (22.4%), so making
   them free caps total speedup at 1.29× (1.45× including all-pairs
   similarity, E4's other named target). The bill is stage A perceive at
   785s = 66.3% — which E4 correctly never claimed. And phase 42 priced two
   *exact* CPU reformulations that take the null work under 10s: the stage-B
   category-bigram null is `G^T W G` over precomputed word-pair counts
   (bit-identical, 282.6s → 0.18s), and the stage-C per-word null is a
   multivariate-hypergeometric draw of the same contingency table (O(k²)
   rather than O(n); 4–255× by word size). Do those before considering a GPU
   crate; re-open E4 only if a future phase makes nulls dominant again, and
   re-measure first. See ROADMAP rows 42 and E4.
8. **In-flight PR #23**: demo/outreach track (D1-D4, phase 31 self-lesion
   protocol) + phases 32/33. Reconcile with it before touching those areas.
9. **THE STORAGE FORK IS OPEN AND IT IS THE OWNER'S** (T7.1, phase 43;
   priced further by T7.2, phase 44). The cost floor's dominant term is
   that a field memory is a COMPLEX N-vector where the prototype bar's is
   REAL. Phase 43 measured the imaginary half **empty** — median residual
   2.6e-04 on the 33h arm, 1.2–1.7e-04 on both text corpora, and the
   per-slot phase is **gauge** for every `np.abs` consumer. Two mutually
   exclusive options, both priced and neither chosen:
   **(A) spend it** — store `(real N-vector, phase scalar)` and the floor
   goes 2.24× → **1.32×** (**1.15×** with phase 44's narrow CSR), meeting
   the ≤2× fallback at a measured behavior cost of ≤0.005 held-out; this
   **permanently forfeits** N8's binding channel and any complex-input
   pipeline. **(B) cash it in** — keep the width, the floor stands at
   2.24× (2.07× narrowed), the option-(a) gate re-scope is forced, and the
   width becomes usable **by parameter change rather than new mechanism**,
   because phase 43 falsified the structural account: `perceive` already
   computes a phasor sum and the channel is empty only because
   `omega/g_in = 0.0375` with deep settling.
   **Nothing below the fork should be built until it is resolved** —
   phase 48 is reserved for T7.6 and deliberately unused. If (B) is
   chosen, the cheapest arm is the **dual time constant** (a second O(N)
   field state at a slower `g_in`, ~0.001× of the K×N store), because
   phase 45 measured that moving the ONE existing state costs 62 words of
   coverage and the category certificate outright — and any phase-carrying
   arm must also change the readout, which phase 43 measured to be blind
   below a residual of **R ≈ 0.06**.

## Practical notes

- Backends: default `auto` = JIT. `DEFAI_BACKEND=numpy` for reference runs.
  Run `regression_harness.py` under BOTH before trusting any mechanism or
  backend change (`numpy` ~81s, `numba` ~14s + one-time JIT compile, cached
  on disk). `test_fastpath_equivalence.py` is the sharper tier-0 check for
  kernel edits; `e2_benchmark.py` reproduces the performance numbers.
- Corpus: `/tmp/gutenberg_corpus/*.txt` (8 public-domain books) is ephemeral
  — re-fetch with the curl block printed by `phase26_percentile_bars.py`
  (or see `phase20_large_corpus.py` history). Post-E2, the 3-epoch corpus
  perceive is ~1.4 min; still run big sweeps in background.
- Saved run state: prefer `organism_state.save_state/load_state` (E3) over
  ad-hoc pickles; `/tmp/phase20_*.{npy,pkl}` artifacts are legacy and
  ephemeral.
- requirements: numpy always; numba+scipy for the fastpath (optional);
  scikit-learn/torch only for specific phase baselines.

Good luck. This project rewards someone who tests the obvious hypothesis
before trusting it, writes down what broke as carefully as what worked —
and now, who checks what the other sessions are doing before starting.
