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

The proposed fix is **word-conditional residual gating**: compare same-word
patterns in the context residual (component orthogonal to the word
direction), where distinct contexts separate cleanly and independently of
`alpha`. Work plan:

- `verify_residual_gating.py` — the theorem + fix, verified numerically.
- Track A (`phase8b_track_a.py`) — is the real prev-attractor context signal
  bimodal enough to separate senses? (feasibility gate, run first)
- Track B (`phase8c_residual_gating.py`) — residual-gated recruit/merge and
  an occurrence-level assignment metric; threshold sweep.
