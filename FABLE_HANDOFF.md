# DeFAI — Handoff Brief

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
- **E1** `regression_harness.py`: 27 tolerance-based checks pinning every
  headline behavior (§1-5 core/noise/pool/categories/gain, §6 phase-30
  reasoning, §7 phase-26 calibration, §8 E3 serialization). Every backend,
  port, and calibration change must pass it before being believed.
- **E2** `fastpath.py`: Numba JIT backend for perceive/recall/recall2 +
  vectorized consolidate. Selected via `Organism(backend="auto"|"numba"|
  "numpy")` or `DEFAI_BACKEND` env; default `auto` (JIT when numba is
  installed). `DEFAI_BACKEND=numpy` gives the historical pure-NumPy
  reference. Measured **13.0× at phase-23 corpus shape (18.2 min →
  1.40 min)**; equivalence pinned by `test_fastpath_equivalence.py`
  (state agreement ~1e-15 on short streams; recall sequences match
  exactly — same RNG consumption order). `organism_numba.py`'s
  `NumbaOrganism` is a thin alias kept for the harness/E3 seams — two
  sessions built E2 in parallel and the implementations were unified
  (the ROADMAP E2 row records both). Remaining bound is the K×N overlap
  matvec (memory bandwidth), not the interpreter: the next speedup is
  algorithmic or E4 (GPU statistics tier), not more JIT.
- **E3** `organism_state.py`: schema-versioned .npz save/load, rng state
  included. Pinned: mid-stream save→load→continue is bitwise identical to
  never stopping; deterministic replay; cross-backend restore.

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
2. **Phase 26 real-text arm (V ≫ N)**: the landed spectral estimator needs
   vocabulary rank < N. Two recorded starting points: closed PR #19's
   pairwise-quantile `qcal` estimator (rank-free; its unfinished Part B was
   exactly this run), and phase 32 (PR #23, partial) — a fixed-point
   self-calibration that engages at V ≥ N but misses its σ=0.2 bands.
3. **Phase 28 — polysemy vs grammatical context-sensitivity** (the
   disentangling test, gold-POS for eval only).
4. **Phase 29 — recursive hierarchy** (the recruit/consolidate primitive one
   level up; the answer to "no structure above the category FSM").
5. **Corpus-tier harness checks**: cheap now (E2), blocked on making the
   Gutenberg fetch reproducible inside the harness.
6. **Stable symbol registry** decoupled from slot indices (designed at the
   EventBoundary seam; downstream scripts still index P by slot).
7. **E4 — GPU statistics tier** (permutation nulls, all-pairs similarity;
   Rust/wgpu shape for Windows+AMD reality). Timed for phases 27/28.
8. **In-flight PR #23**: demo/outreach track (D1-D4, phase 31 self-lesion
   protocol) + phases 32/33. Reconcile with it before touching those areas.

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
