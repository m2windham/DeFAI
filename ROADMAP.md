# DeFAI Roadmap

Adopted 2026-07-10, after a critical review of two external AI audits (an
engineering-focused executive summary and a strategic review). What survived
that review is below; what didn't (SSM/Mamba core swap, MoE gating, EWC,
differentiable-memory rewrite — all gradient-era machinery that would break
the online/no-backprop premise and the measured continual-learning advantage)
is deliberately absent. Full reasoning lives in the session notes and the
review discussion; the standing rules of engagement in `FABLE_HANDOFF.md`
still bind everything here.

## Strategic structure: Path A main, product forks downstream

- **This repository stays Path A**: the research line. One continuously
  running oscillator field that perceives, remembers, learns world structure,
  and generates — online, label-free, no gradients inside the mechanism.
  All discovery happens here, phase scripts remain the lab notebook.
- **Path B and later products are hard forks, not branches.** First fork: a
  deployable episodic-memory module (online ingestion, measured ~15%-forgetting
  continual learning, persistent state). Later forks: agent products built on
  capabilities as they are discovered. Each fork is a separate product
  repository.
- **Discoveries flow downstream through a versioned engine, never through
  git merges.** A result graduates from main to forks only when it has
  (1) a phase script, (2) pinned numbers in the regression harness, and
  (3) a release tag. Forks upgrade by bumping the engine version.
- **Fork gate for Path B** (event-driven, not a date): E1 + E2 + E3 done and
  Phase 26 landed — the earliest moment the engine is fast, persistent,
  calibration-robust, and regression-guarded.

## Research track (this repo)

| Phase | Question | Status |
|---|---|---|
| 25 — decorrelation α-sweep | Is embedding correlation the whole cause of the pool-mode perception collapse (85/395 at scale)? Sweep the PPMI-SVD weighting `emb = U·S^α`, α ∈ {0.5, 0.25, 0}; report the >0.7-neighbor fraction, transitive-component count, and core-arm coverage per α. **Run 2026-07-10 on the live 547K-word corpus — result: PARTIAL, not full closure.** Neighbor fraction fell monotonically (23% → 13% → 9%) and core-arm coverage more than doubled (85 → 168 → 216 / 395, a 2.5× recovery) as α dropped to 0 — embedding correlation is confirmed as a real, measured cause. But even fully decorrelated, coverage plateaus at 216/395 (55%), well short of both full vocabulary and the plain-recipe arm's 378/395 (phase 23) — a large residual gap that decorrelation alone does not close. Verdict: correlation is *a* cause, not *the* cause; the acceptance-bar constants (0.7 fusion, 0.6 active_bar, the 0.8/√ anneal) are very likely an independent second cause, exactly as pre-registered for this outcome. Feeds directly into Phase 26. | `phase25_decorrelation_sweep.py` — done |
| 26 — percentile acceptance bars | Replace the absolute perception constants (0.7 fusion, 0.6 active_bar, the 0.8/√ anneal) with quantiles of the measured similarity distribution, generalizing the existing `s_hat` hook. Must reproduce phases 14/17/18 on synthetic near-orthogonal data before being believed on real text. Closes handoff open thread #1 as a mechanism and makes perception embedding-source-agnostic (a Path B requirement — products won't control their input embeddings). | planned |
| 27 — 5M-word scale run | 50–100 Gutenberg books, with phase 24's MI-vs-null + distinctness criteria wired into stage B. Pre-registered: does category structure sharpen (MI z up, distinctness-selected k stable or gracefully growing), and do the polysemy detections hold under tighter nulls? Depends on E2 for runtime. | planned |
| 28 — polysemy vs grammatical context-sensitivity | The disentangling test: cluster each detected word's occurrences by context signature (prev/next category), ask whether the clusters land in *different* induced categories (lexical polysemy) or the same category with shifted successors (grammatical context-sensitivity). Evaluated against gold-POS occurrence entropy — oracle for evaluation only, per standing rules. Closes handoff threads: the phase-23 conflation residual *and* the POS precision/recall thread, in one run. Scope caveat pre-registered: same-POS polysemes (bank/bank) will be classed as context-sensitivity; the claim narrows to POS-level multi-role detection. | planned |
| 29 — recursive hierarchy | Run the recruit/consolidate primitive one level up: "tokens" are category n-gram signatures, recruitment discovers recurring category-sequence motifs (phrase-level states). Certified by the same level-agnostic MI-vs-permutation-null machinery from phase 24; oracle eval only against shallow-parse chunk boundaries. Pre-registered risk: at k≈6 there are only ~36 category bigrams — the honest negative is "level 2 learns nothing beyond the level-1 transition matrix," which would push toward hierarchy-aware k-selection. This is the direct answer to the strongest external critique (no structure above the category FSM). | planned |

## Engineering track (parallel; feeds both paths)

| Item | What | Status |
|---|---|---|
| E1 — regression harness | `regression_harness.py`: multi-seed, tolerance-based checks pinning the mechanism's headline behaviors (core perceive/recall/consolidate in plain, confirm, pool, and amb modes; category discovery; predictive-gain margin). Statistical tolerances, not bitwise equality (GPU reductions will legitimately perturb low-order bits). Every backend/port must pass before it is trusted. Corpus-tier checks (phase 22/23 headline numbers) join the harness once E2 makes them cheap. | `regression_harness.py` |
| E2 — Numba port | JIT the sequential hot loops (perceive, recall/recall2, consolidate inner loops) — they are interpreter-bound, not arithmetic-bound; target 15 min → ~1 min at phase-23 scale. CPU is the right tool: the perceive loop is inherently sequential over the stream. Validated against E1. | planned |
| E3 — state serialization | Save/load of the full organism state (`xi`, `P`, habituation, pool/probation state) with a schema version so stored memories survive mechanism upgrades. Not product polish — an episodic-memory product is *defined* by persistence. Plus: bounded memory (slot cap + eviction policy), deterministic replay. | planned |
| E4 — GPU tier (hipfire) | The embarrassingly parallel stages only — permutation nulls first (the statistical backbone, thousands of independent shuffles), then per-word gain sweeps, all-pairs similarity, PPMI/SVD at V≥5–20K. The sequential perceive loop is *not* a GPU target. Windows+AMD reality: Python-side ROCm (numba-hip, CuPy-ROCm) is Linux-only, so the intended shape is a Rust/wgpu (Vulkan/DX12) compute crate with PyO3 bindings — which doubles as the dependency-free native runtime a product fork can ship. Timed to land around Phase 27/28 when the statistics workload grows. | planned |

## Sequencing

```
E1 harness ──┬─────────────► E2 Numba ──► Phase 27 (5M words) ──► Phase 28 ──► Phase 29
             │                   │
Phase 25 ────┴──► Phase 26 ──────┤
                                 ├──► E3 serialization ──► FORK GATE (Path B repo)
                                 └──► E4 GPU tier (by Phase 27/28)
```

Phase 25 and E1 are independent and both unblock everything else; they go
first. Phase 26 consumes Phase 25's diagnosis. E2 consumes E1's guarantees.
The fork gate consumes E1+E2+E3+Phase 26.

## What the external reviews got right (kept) and wrong (dropped)

Kept: vectorize so the corpus can scale (E2), external baselines (a small
gradient-trained model on the same corpus as an *oracle ceiling*, phase-9B
style — attach to Phase 27), demonstrate downstream utility of polysemy
detection (Phase 28 + wiring into generation), regression/testing discipline
(E1), hierarchy/compositionality above the category FSM (Phase 29).

Dropped, with reasons recorded so they aren't re-derived: SSM/Mamba core
(replaces the research question; gradient-trained), MoE gating (routing here
is information-limited, not cost-limited — phase 18 measured it), EWC/replay
(solves a failure mode this architecture measurably doesn't have),
differentiable external memory (a different project), post-hoc consolidation
clustering (re-opens phase 9's closed negative result: re-sorting provably
inherits online mixing).
