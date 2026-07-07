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

Side experiment: `strain_propagation.py` (Kuramoto "code bath" refactor-wave test).

## Current frontier: polysemy (phase 8)

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
- Noise: **solved to a measured boundary** (phase 14). Probationary
  recruitment (`perceive(confirm=3)`) eliminates junk memories and strictly
  improves prediction and generation up to σ=0.2. The boundary is analytic:
  token-vs-memory overlap 1/√(1+σ²N) crosses the 0.6 confirmation bar at
  σ*≈0.24, beyond which genuine revisits are unrecognizable and any
  single-shot recruitment scheme must fail. Open beyond σ*: pool evidence
  across occurrences *before* the keep/discard decision.
