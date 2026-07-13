# DeFAI — Handoff Brief for Fable

You're taking over an in-progress research project with no prior context — everything you need is below. Where this brief and the code disagree, **trust the code** and flag the mismatch to the user.

## Repo

`m2windham/DeFAI`, branch `claude/defai-investigation-gsn2oy`. 21+ phases of work, all committed and pushed. Read `git log --oneline` for the full trail — every phase commit message is a self-contained lab notebook entry explaining what was tried, what broke, and why.

## The goal

A **General Sentient Exploration model**: a continuous dynamical system built on coupled oscillator physics, meant to generate language without discrete tokenization — a unified cognitive substrate where meaning is a *position* in a continuous field, not a lookup in a vocabulary table.

## Core architecture

`organism.py` — `Organism` class. Complex-valued Stuart-Landau-style oscillator field `z ∈ C^N`, normalized to `‖z‖ = √N`.

- **`perceive(stream)`**: drives the field toward input embeddings (Hebbian competitive learning). Recruits a new memory slot when novelty exceeds a threshold (`recruit` param — **this is a similarity floor for updating, not a novelty threshold; higher = more eager to recruit new slots**, a real bug we found and fixed on real text), else updates the nearest existing slot via EMA. Also accumulates a Hebbian transition-count matrix `P` between whichever slots are sequentially active.
- **`consolidate()`**: merges near-duplicate slots, prunes rarely-used ones, builds `self.mem` (compact memory bank) and `self.Pn` (row-normalized transition matrix) from the accumulated `P`. Also sets `self.kept_idx` (added recently — lets you reconstruct raw, unnormalized transition counts restricted to the kept memories, needed for PPMI-style category discovery).
- **`recall(steps)`**: generates sequences autonomously. Uses **habituation** (`h[k]`, a fatigue term) so the field doesn't lock onto one attractor forever — this is essential; anything you build downstream that skips habituation will collapse to repeating one output.

Since 2026-07 (PR #20), `organism.py` is split into two systems with a narrow boundary — the language/logic functional separation (MIT/McGovern) applied to the pipeline:

- **`TransitionGraph`** (logic layer): owns the transition matrix; the only mutations are its interface ops (`observe`/`merge`/`retire`/`fold`/`normalized`), plus phase-30 reasoning ops that run on the graph with the field absent: `kstep` (multi-step inference), `rollout` (field-free imagination), `next_hops`/`plan` (Dijkstra planning). `Organism.P` is a property delegating to it — read/write, slot-indexed, so older phase scripts are unaffected.
- **`EventBoundary`**: the single call-site where recognitions become relational knowledge; `perceive()` commits only confident, non-provisional events, and slot fusion/recycling arrive as explicit `remap`/`invalidate` notifications. Recognition errors can't silently rewrite the world graph.
- **`Organism.recall_directed(goal)`**: goal-directed generation — logic proposes the next hop toward the goal, the field renders it (keeps habituation; see the warning above).
- **`perceive(..., fuse_bar=0.7)`**: the duplicate-fusion threshold, formerly hardcoded; phase 26's label-free calibration (`phase26_percentile_bars.calibrate`) can now measure `active_bar`, `s_hat`, and `fuse_bar` from a stream prefix instead of being handed them. Known limit: its spectral noise estimator needs vocabulary rank < N; re-derive that arm before trusting it on real text.
- Symbols are currently still 1:1 with slots; a stable symbol registry that survives slot churn is the designed next step (module docstring documents the seam).

`polysemy_organism.py` — `PolysemyOrganism(Organism)`, the validated polysemy pipeline as a reusable class:
- **`discover_categories_v2(k_range, raw_counts)`**: THE CURRENT, FIXED category-discovery method. PPMI-transforms the raw transition-count profile (removes frequency-magnitude bias — the actual root cause of every category-discovery failure this project hit), then real k-means with k chosen by silhouette + manual balance-checking. **Use this one.** There's an older `discover_categories()` (greedy threshold-gated clustering on raw `Pn`) — it's kept for backward compatibility with older phase scripts but is known-broken on real text (fragments on small corpora, collapses into 1-2 giant blobs on large ones, because it never corrects for word-frequency skew). Don't build new work on it.
- **`perceive_polysemy()` / `consolidate_polysemy()`**: the online, single-pass, predictive-gain-gated, residual-gated sense-splitting mechanism (see below).

## The core intellectual result (read this even if you read nothing else)

**The polysemy problem**: a word like "fish" can be a noun or verb with no orthographic difference. Early mechanisms (Phases 7-11) tried to detect this by asking "does the word's context-shifted *representation* look different?" — every version of this failed, because context-driven positional drift is necessarily small (to preserve word identity), and any threshold on "looks different" either splits everything or nothing, and the failure mode is scale-dependent, not a tunable constant.

**The fix (Phase 12, the single most important idea in this project)**: stop asking "does it look different," ask **"does knowing the context change what I'd predict happens next"** — a Myhill-Nerode / formal-automaton style state-splitting test. Concretely: compute entropy of the successor category unconditionally vs. conditioned on the previous word's (emergent, unsupervised) category. Real polysemous words show a large, clean **predictive gain**; monosemous words show ~zero. This test uses only observable corpus statistics — no ground-truth labels, ever.

**Validated on synthetic data** (Phases 8-16): clean separation, 0.30-bit margin between true dual-role words and controls, zero false positives, fully online (single streaming pass, no full-corpus pre-pass).

**Ported to real text** (Phases 19-21) and it broke in two different, informative ways depending on scale, both eventually fixed or diagnosed:
1. Small real corpus (~1,400-2,800 tokens): category discovery *fragmented* (15 categories, mostly 1-4 words) — too little data.
2. Large real corpus (~547,000 words, 8 classic books): category discovery *collapsed* (2 giant blobs holding 97% of vocabulary) — NOT a data problem, a **frequency-magnitude bias in the clustering itself** (common words dominate by raw count regardless of true category). Fixed by PPMI-transforming the transition profile before clustering (`discover_categories_v2`).

**The actual win (Phase 21)**: with the fix, on real text, the word **"right"** (622 real occurrences across 8 books) cleared a directly-measured statistical noise floor (gain 0.070 vs. 99th-percentile noise 0.043) — genuine unplanted polysemy detected in real natural language, fully unsupervised, for the first time in this project. 37 other words cleared the bar too, and they're largely plausible multi-role English words (turn, long, far, fast, hard, near, old, way) — this is the honest full ranking, not cherry-picked.

## Other validated capabilities

- **Grammar learning** (Phase 4-5): unsupervised Hebbian perception learns category-cyclic grammar structure from raw embedding streams; a "grammar-mask" fine-tune step can push generation grammaticality up to/past oracle level on synthetic data.
- **Category emergence** (Phase 10): running the *same* recruit/update primitive over transition profiles instead of embeddings makes grammatical categories emerge unsupervised, at 100% purity on synthetic data.
- **Continuous meaning manifold** (Phase 17): replaced discrete polysemy "slots" with continuous, gain-scaled positional drift — structurally eliminates slot explosion, first-mover lock-in, and brittle thresholds, since there's nothing discrete to explode or threshold. Control words show ~0 drift, dual-role words show real, role-correlated drift, entirely as a mathematical consequence of the same gain signal from Phase 12 — no separate rule needed.
- **Continuous generation** (Phase 18/18b): pure continuous kernel-regression generation (no discrete state anywhere) **failed** — diagnosed precisely: role signal is ~3% of the similarity scale, invisible to raw k-NN retrieval, and amplifying it doesn't help (tested to 3x). The working fix: keep grammar as a small, *unsupervised but discrete* category-transition FSM (the correct level for grammar, which genuinely is a small state machine), and keep everything else — word identity, sense, fine meaning — continuous. Achieved grammaticality 0.818 vs. 0.839 discrete baseline, full vocabulary coverage, live context-sensitive disambiguation during actual generation.

## Rules of engagement (violate these and you'll re-derive already-closed dead ends)

- **Never use ground-truth role/category labels inside the mechanism.** Oracle tests (Phase 9B) are fine as *capability ceilings* to compare against, explicitly labeled as oracles — never as part of a proposed solution.
- **Negative results are the deliverable, not noise to hide.** This project's commit history is full of "X failed, here's precisely why" — that discipline is why Phase 12 and Phase 21's fixes were findable. Keep doing it.
- **When something looks like a data-volume problem, verify before assuming.** Phase 20 disproved Phase 19's own hypothesis ("just needs more data") with a real experiment — the actual bottleneck was a clustering bug, not data. Measure, don't guess.
- **Directly measure noise floors, don't eyeball a number and call it signal.** Multiple times in this project, an impressive-looking gain number turned out to be indistinguishable from pure noise at that sample size/category-count — always simulate the null.
- **`recruit` in `Organism.perceive` is a similarity floor, not a novelty threshold** — higher values recruit new slots *more* eagerly, the opposite of the naive reading. This bug cost real time once already.

## Immediate open threads (pick one, or propose your own)

1. **Automatic k-selection for category discovery isn't solved.** `discover_categories_v2`'s silhouette-argmax picked k=19 on the large real corpus (bad — reinflates the noise floor), while manually checking category *balance* found k=3 worked well. Silhouette and "the right number of categories" aren't the same thing yet on real text — needs a real fix, not another manual override.
2. **Scale the real-text validation.** Current result is one lexical item ("right") + 37 plausible extras on 547K words. More books, more candidate words, ideally cross-checked against a real POS-tagged corpus (even just for evaluation, never fed into the mechanism) to get a precision/recall number instead of eyeballing plausibility.
3. **Wire Phase 21's real-text polysemy detection into Phase 17's continuous manifold + Phase 18b's generation**, closing the loop on real text the way it's already closed on synthetic data.
4. **Longer-range context** — disambiguation currently looks one token back; real ambiguity often needs more.

## Practical notes

- Large-corpus runs (`phase20_large_corpus.py`) take ~15 minutes (3 training epochs over ~408K tokens) — run in background, don't block on it.
- Corpus source: `real_text_corpus.py` (small, hand-written fables) and `/tmp/gutenberg_corpus/*.txt` (8 public-domain books, fetched live via curl from Gutenberg — not committed to the repo since they're large and re-fetchable; re-run the `curl` block near the top of `phase20_large_corpus.py`'s history if `/tmp` has been cleared).
- Saved intermediate state from the large-corpus run lives in `/tmp/phase20_*.{npy,pkl}` — also ephemeral, regenerate by rerunning `phase20_large_corpus.py` if needed.

Good luck. This project rewards someone who tests the obvious hypothesis before trusting it, and who writes down what broke as carefully as what worked.
