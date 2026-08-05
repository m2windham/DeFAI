# DeFAI — Agent Targets

Actionable work packages derived from `ROADMAP.md`, structured for parallel
agent sessions. **Claim a target by its ID in your PR title / roadmap row
before starting** (two sessions once built E2 twice — don't repeat that).
Read order for any cold start: `FABLE_HANDOFF.md` → `ROADMAP.md` → this file
→ the target's listed context files. `git log --oneline` is the lab notebook.

Standing rules bind every target (see FABLE_HANDOFF "Rules of engagement"):
no labels inside mechanisms; negative results are deliverables; measure
noise floors; pre-register predictions; run `regression_harness.py` under
BOTH backends (`DEFAI_BACKEND=numpy|numba`) before trusting any change;
`test_fastpath_equivalence.py` for kernel edits.

Verification state as of 2026-08-04: E1 27/27 both backends, equivalence
green, e2_benchmark meets pinned numba throughput. See ROADMAP
"Verification log".

---

## Category 1 — Release-gate critical path (highest leverage)

The Path B RELEASE HOLD (owner, 2026-07-15) is lifted only when the
phase-33 gate passes ("at least SOTA / cost-effective"). Phase 35 showed
the collapse is fixable at inference time. These targets are sequential.

### T1.1 — Phase 36: un-oracle hierarchical recall  `[claimed: claude/claim-execute-t1-1-wthzip, 2026-08-05]`
- **Objective**: replace phase 35's ground-truth category labels with
  `discover_categories_v2` (PPMI + distinctness k-selection) in the
  hierarchical router; re-run the 50→800-word sweep with identical scorers.
- **Context**: `phase35_hierarchical_recall.py` (router + oracle caveat in
  its docstring), `phase34_capacity_scaling.py` (shared corpus/scorers),
  `polysemy_organism.py::discover_categories_v2` (THE method; never the old
  `discover_categories`), `phase24_category_validity.py` (distinctness
  k-selection, MI-vs-null certificate).
- **Pre-registered predictions** (ROADMAP row 36): (a) gap recovery
  degrades ∝ category impurity; (b) honest negative = a purity floor below
  which hierarchy loses to flat — measure the floor either way.
- **Done when**: `phase36_*.py` prints setup/baselines/verdict; sweep rows
  + verdict recorded in ROADMAP; harness still green both backends.

### T1.2 — Slot-budget / eviction policy  `[claimed: —]`
- **Objective**: stop task-1 slot flooding (phase 33: K=40 exhausted by the
  first class, later classes learn only by slot drift → FORG 0.20). Natural
  home: the pool-mode use-it-or-lose-it recycling machinery in
  `organism.py` (phase 17 mode).
- **Context**: `phase33_industry_baselines.py` (flooding measurement),
  `organism.py` perceive pool-mode paths, `organism_state.py` (eviction
  state must serialize; E3 pins are bitwise).
- **Constraint**: mechanism, not benchmark tuning — must not use task/label
  information. Must leave harness §1–§5 green.
- **Done when**: measured re-run of the phase-33 flooding diagnostic shows
  later classes recruiting fresh slots; harness green; E3 round-trip green.

### T1.3 — Online label-evidence readout as mechanism  `[claimed: —]`
- **Objective**: phase 33 fixed its frozen-label readout artifact in-phase
  (ACC 0.25 → 0.665); promote online per-slot label evidence to a proper
  mechanism (readout layer, labels still never inside perception/learning).
- **Context**: `phase33_industry_baselines.py` (the artifact + in-phase fix).
- **Done when**: readout lives outside the phase script, documented as
  eval-side; phase-33 script consumes it.

### T1.4 — Re-run the phase-33 ladder (gate re-test)  `[claimed: —]`
- **Objective**: after T1.1–T1.3, re-run the full ladder (naive SGD, EWC,
  replay, joint oracle, growing prototypes, organism-on-numba) on
  class-incremental split-digits; report ACC/FORG vs the 0.872 supervised
  prototype bar.
- **Blocked by**: T1.1, T1.2, T1.3. **Owner decision point**: gate re-opens
  only at ≥SOTA or demonstrated cost-effectiveness at equal accuracy.
- **Note**: needs torch installed (baselines only).

---

## Category 2 — Scale & real text

### T2.1 — Phase 27: 5M-word scale run  `[claimed: —]`
- **Objective**: 50–100 Gutenberg books through the unified loop with
  phase 24's MI-vs-null + distinctness wired into stage B. Pre-registered:
  does category structure sharpen (MI z up, selected k stable/graceful),
  do polysemy detections hold under tighter nulls?
- **Context**: `phase23_unified_large_corpus.py` (the 547K template; prints
  the corpus curl block), `phase24_category_validity.py`,
  `fastpath.py`/`e2_benchmark.py` (runtime budget: ~1.2–1.4 min per corpus
  perceive at 547K → plan for ~10× that), E4 status (nulls are the cost).
- **Done when**: phase script + ROADMAP row with pre-registered outcomes
  recorded, including partials.

### T2.2 — Phase 26 real-text arm: calibration at V ≫ N  `[claimed: —]`
- **Objective**: acceptance-bar calibration without the rank<N spectral
  assumption — the blocker for the core perception stack on real
  embeddings (85/395 collapse, confirmed at scale).
- **Context**: ROADMAP rows 26 + 32 (read both fully — estimator lessons
  and a documented 1.7–2.1 oscillation at σ=0.2 are recorded to save you
  time), closed **PR #19**'s `_calibrate_bars` (pairwise-quantile `qcal`,
  rank-free — the designated starting point; do NOT invent a third
  estimator first), `phase32_selfconsistent_calibration.py` (partial),
  `phase26_percentile_bars.py`, `phase25_decorrelation_sweep.py`.
- **Done when**: calibrated bars reproduce oracle coverage/junk within
  pre-registered bands at V ≫ N on synthetic, then lift real-text core-arm
  coverage measurably above the 216/395 decorrelation plateau.

### T2.3 — Phase 28: polysemy vs grammatical context-sensitivity  `[claimed: —]`
- **Objective**: the disentangling test — cluster detected words'
  occurrences by context signature; different induced categories = lexical
  polysemy, same category with shifted successors = context-sensitivity.
  Gold-POS for **evaluation only**.
- **Context**: `phase21_working_polysemy_detection.py`,
  `phase23_unified_large_corpus.py` (the 114-word detection list + the
  conflation caveat), ROADMAP row 28 (scope caveat: same-POS polysemes
  classed as context-sensitivity; claim narrows to POS-level multi-role).
- **Done when**: per-word classification with a measured null; precision/
  recall vs gold-POS entropy reported; caveats recorded.

### T2.4 — Phase 29: recursive hierarchy  `[claimed: —]`
- **Objective**: recruit/consolidate one level up — tokens are category
  n-gram signatures; discover phrase-level states. Certify with phase 24's
  level-agnostic MI-vs-null.
- **Context**: ROADMAP row 29 (pre-registered risk: at k≈6 only ~36
  category bigrams — "level 2 learns nothing" is the honest negative and
  pushes toward hierarchy-aware k-selection), `phase24_category_validity.py`.
- **Blocked by**: sensible to follow T2.1 (more bigram mass at 5M words).

---

## Category 3 — Engineering spine

### T3.1 — E4: GPU statistics tier  `[claimed: —]`
- **Objective**: permutation nulls first (thousands of independent
  shuffles), then per-word gain sweeps, all-pairs similarity, PPMI/SVD at
  V≥5–20K. **Not** the perceive loop (sequential by nature).
- **Context**: ROADMAP E4 row — Windows+AMD reality means Rust/wgpu
  (Vulkan/DX12) + PyO3, not CUDA/ROCm-Python; doubles as a dependency-free
  native runtime for product forks. `A:\dev\hipfire` may hold prior art.
- **Timing**: land alongside T2.1/T2.3 when the statistics bill arrives.
- **Done when**: harness gains a GPU-vs-CPU tolerance check (statistical
  bands, not bitwise); null-generation wall-clock measured.

### T3.2 — Corpus-tier harness checks  `[claimed: —]`
- **Objective**: pin phase 22/23 headline numbers in the harness (cheap
  post-E2). Blocker: make the Gutenberg fetch reproducible inside the
  harness (pinned URLs + hashes, cached under a stable path, offline skip).
- **Context**: `regression_harness.py` (§ structure), fetch block printed
  by `phase23_unified_large_corpus.py` / `phase26_percentile_bars.py`.
- **Done when**: `regression_harness.py --corpus` (or equivalent tier flag)
  runs the pinned checks from a cold cache and skips gracefully offline.

### T3.3 — Stable symbol registry  `[claimed: —]`
- **Objective**: symbol IDs decoupled from slot indices, designed at the
  `EventBoundary` seam (downstream scripts still index `org.P` by slot;
  fusion/recycling already emit remap/invalidate notifications to build on).
- **Context**: `organism.py` (`EventBoundary`, `TransitionGraph`),
  `organism_state.py` (registry must serialize).
- **Done when**: registry survives consolidate/recycle/save-load with
  stable IDs; at least one phase script migrated as proof; harness green.

---

## Category 4 — Demonstration & outreach (feeds Path B adoption)

All demos render only capabilities with pinned harness numbers; nothing
staged. Order decided in triage (ROADMAP): D1 first.

### T4.1 — Phase 31: self-lesion study (research half of D4)  `[claimed: —]`
- **Objective**: pre-registered lesion protocol — ablate the perceptual
  field at graded severities (zero/noise input coupling, damage `xi`),
  measure what survives on the logic layer (kstep, rollout, plan) vs
  perception-side metrics, plus recovery after E3 restore. Quantifies the
  aphasia property phase 30 showed qualitatively.
- **Context**: `phase30_symbolic_reasoning.py`, `organism_state.py`,
  ROADMAP row 31. Timing: after T2.1/T2.3.
- **Done when**: severity-vs-survival curves + verdicts pinned; only then
  may the stage-demo version (D4) be built.

### T4.2 — D1 "It learns you" interactive demo  `[claimed: —]`
- **Objective**: page where an organism ingests the visitor's interaction
  stream live; memories crystallize in ~2 min, then it predicts their next
  move; E3 makes return visits literal ("it will remember you").
- **Context**: `organism.py` (perceive/recall2), `organism_state.py`,
  ROADMAP demo table. Constraint: earmarked as the interactive front door
  to D3 — keep the feed adapter generic (D3's only new engineering).
- **Done when**: running demo whose displayed numbers trace to harness-
  pinned behaviors; no capability shown that lacks a phase script.

### T4.3 — D2/D3 (marketing essay / living organism)  `[claimed: —]`
- **Status**: D2 is launch-essay material (organism ingests its own repo
  history); D3 is the public continuous-life server. Both intentionally
  deferred until Category 1 lifts the release hold — don't build ahead of
  the gate.

---

## Category 5 — Path B (blocked)

### T5.1 — Release tag + episodic-memory product fork  `[claimed: —]`
- **Status**: **BLOCKED by RELEASE HOLD** until T1.4 passes the owner's
  bar. Engineering gate (E1+E2+E3+Phase 26 synthetic) already OPEN.
- **When unblocked**: cut release tag on main → create separate product
  repo (hard fork, never a branch) → discoveries flow only via the
  versioned engine (phase script + pinned harness numbers + release tag).

---

## Dependency sketch

```
T1.1 ─┬─► T1.4 ─► T5.1 (hold lifts) ─► T4.3
T1.2 ─┤
T1.3 ─┘
T2.1 ─► T2.4          T2.2, T2.3 independent (T2.3 benefits from T2.1 corpus)
T3.1 ─► cheaper nulls for T2.1/T2.3     T3.2, T3.3 independent
T4.1 ─► D4 demo       T4.2 independent of the hold (demo, not product)
```
