# DeFAI — Agent Targets

Actionable work packages derived from `ROADMAP.md`, structured for parallel
agent sessions. **Claim a target by its ID in your PR title / roadmap row
before starting** (two sessions once built E2 twice — don't repeat that).
Read order for any cold start: `FABLE_HANDOFF.md` → `ROADMAP.md` → this file
→ the target's listed context files. `NOVELTY.md` is the claims register —
what the project can defend, with scope caveats and per-item roadmaps; keep
it current when a target changes what a claim can say. `git log --oneline` is the lab notebook.

Standing rules bind every target (see FABLE_HANDOFF "Rules of engagement"):
no labels inside mechanisms; negative results are deliverables; measure
noise floors; pre-register predictions; run `regression_harness.py` under
BOTH backends (`DEFAI_BACKEND=numpy|numba`) before trusting any change;
`test_fastpath_equivalence.py` for kernel edits.

Verification state as of 2026-08-07, on the merged tree carrying T1.9
(phase 33h), T2.1 (phase 27's outcome) and T3.3 (the symbol registry):
E1 ALL PASS both backends (65 checks: 31 + T1.8's section 10 + T3.3's
section 11), `test_fastpath_equivalence.py` green incl. the narrowed-store
section 7 and the symbol-registry section 8, `test_label_readout.py` green,
E3 round-trips green for schema v3 uncompressed/compressed and v1 + v2
backward load. Torch is NOT installed on any of these hosts, so the ladder's
torch arms rest on 33c's committed values. See ROADMAP "Verification log".

---

## Standing operating protocol (every target, every session)

This is the professional baseline. Individual targets add requirements;
none of them relax these. Every line below exists because this project paid
for it — E2 was built twice, a shared checkout corrupted its git index mid-
merge, two agents both named a script `phase33e`, and three separate
positive results evaporated under reseeding.

### A. Before you touch anything
1. **Sync and survey.** `git pull`; check open PRs and the `[claimed: …]`
   slots here. Work already in flight is not yours to redo.
2. **Claim the lock.** Edit your target's claim slot to
   `[claimed: <branch>, <date>]`, commit that one-line change, and **push it
   before starting work**. The push is the lock; an unpushed claim is not a
   claim.
3. **Work isolated.** Your own clone, your own branch (`claude/<topic>-<id>`).
   Never run concurrently with another agent in the same working tree.
4. **Reserve your phase number.** Grep `phase*.py` and the ROADMAP for the
   next free number *and* announce it in the claim commit. Two agents took
   `33e` on the same day; the collision cost a rename and a doc sweep.
5. **Baseline the tree.** Run `regression_harness.py` under **both**
   backends before you change anything. If it is not green on arrival, stop
   and report — do not build on an unknown baseline.

### B. While you work
6. **Pre-register before any committed run.** Predictions — including the
   named honest-negative branch and its decision rule — go into the script
   docstring or ROADMAP row *and get committed* before the run that tests
   them. A prediction written after the numbers is not a prediction.
7. **Nulls, not eyeballs.** Simulate the null at the actual sample size.
8. **Reseed everything, and hold out.** Any arm that was chosen from a grid
   is a *selected* result: confirm it on fresh held-out seeds (s=5–9) before
   banking it. Reseed the **baseline on the same split** — an unpaired
   fixed-seed baseline is a bug, not a shortcut.
9. **Stay in your lane.** Touch only the files your target owns. If you need
   a change in a shared file (`organism.py`, `fastpath.py`,
   `regression_harness.py`), check the claim slots first and keep the edit
   minimal, additive, and behind a default-off flag where possible.
10. **Do not re-open closed negatives.** Phase 9 re-attribution, low-rank
    compression (33g), the count-normalized victim rule (33f), gradient-era
    machinery (ROADMAP's dropped list). If you believe one deserves
    reopening, argue it in the PR *before* spending the compute.

### C. Definition of done — no target is complete without all of it
11. **Green tree**: `regression_harness.py` both backends;
    `test_fastpath_equivalence.py` if any kernel path was touched;
    `test_label_readout.py` if the readout was touched; E3 round-trip if
    state or schema changed (and backward load for every prior schema).
12. **New behavior is pinned.** A new mechanism gets a harness section with
    tolerance bands. Unpinned behavior is not landed behavior.
13. **Anchors reproduce.** Every committed anchor your work could disturb is
    re-run and reported exact (or the drift is explained in the PR).
14. **Documentation sweep — all four surfaces:**
    - `ROADMAP.md` — the phase row: question, method, measured result,
      **including partials, misses, and confounds**;
    - `AGENT_TARGETS.md` — your claim slot to `DONE` plus an outcome block
      a stranger could act on;
    - `NOVELTY.md` — the N-item your result touches, **edited down as
      readily as up**; a weakened claim is edited, never deleted, and its
      "In plain terms" sidebar is updated in the same commit;
    - `LAB_NOTES_PLAIN.md` — the plain-language twin of the lab record.
      Update it whenever your result changes what can be claimed. **A
      weakened claim follows down here too** — this file must never be the
      optimistic copy. New capability gets a chapter, not an adjective;
    - `FABLE_HANDOFF.md` — only if architecture, invariants, or the open-
      thread ordering changed.
15. **Verification log.** Add the ROADMAP "Verification log" entry: host,
    versions, what was re-run, what passed.
16. **PR hygiene.** Title carries the target ID. Description states what was
    verified and how, names any prediction that missed, and lists follow-ups
    you deliberately deferred. Rebase/merge origin/main and re-run the
    harness on the **merged** tree before requesting merge — green on your
    branch is not green on main.

### D. Reporting standard
17. **Report the miss first.** Lead with what failed, what was falsified,
    and what you could not verify; then the wins. A verdict with no caveats
    is a verdict that was not examined.
18. **Never quote a number without its scope sentence.** 33h's
    unconstrained KB/ACC-pt is the standing example: cheap-looking numbers
    that answer a different question than the one asked.
19. **Hand off cleanly.** If you stop mid-target, push your branch, write
    what is done vs open into the claim slot, and leave the tree green.

---

## Category 1 — Release-gate critical path (highest leverage)

The Path B RELEASE HOLD (owner, 2026-07-15) is lifted only when the
phase-33 gate passes ("at least SOTA / cost-effective"). Phase 35 showed
the collapse is fixable at inference time. These targets are sequential.

### T1.1 — Phase 36: un-oracle hierarchical recall  `[DONE 2026-08-05: claude/claim-execute-t1-1-wthzip, PR #27]`
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
- **Outcome (2026-08-05)**: PASS at ρ = 1.02 — discovered-label routing
  matches/beats the oracle router at every vocab (0.858 vs 0.845 at 800
  words; flat 0.284); distinctness picked the true k=5 at all scales,
  purity/V-measure 1.000, MI-null z up to 25 624; phase-35 anchors
  reproduced exactly. Caveat: the purity floor (prediction b) was never
  exercised — discovery was perfect on this synthetic grammar; whether
  routing survives real-text impurity (phase-24 quality, V ≈ 0.55) is
  open and attaches to T2.1/T2.3. Full row in ROADMAP 36.

### T1.2 — Slot-budget / eviction policy  `[claimed: claude/slot-budget-eviction-pool-mode-9vd1u8, 2026-08-05 — DONE]`
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
- **RESULT (2026-08-05)**: `perceive(evict=E)` — persistent staleness clock
  (`org.age`, E3-serialized) + eviction under recruitment pressure (novelty
  > 0.8× recruit floor, no free slot → argmin-count stale slot recycled via
  graph.retire/boundary.invalidate). Phase 33b: baseline reproduces the
  flood + ACC 0.665/FORG 0.200 verbatim; pre-registered E=2000 window was
  an honest NEGATIVE (recency flush, ACC 0.511); finding — the stale pool
  must contain the present, or eviction eats the past — short windows
  (100–750) beat baseline on BOTH axes; primary E=250: ACC 0.712,
  FORG 0.169, fresh recruits every task, era census [9,7,4,6,14]. Harness
  31/31 both backends (new §9); equivalence check 6; E3 bitwise. See
  ROADMAP row 33b + `phase33b_slot_budget.py`.

### T1.3 — Online label-evidence readout as mechanism  `[DONE 2026-08-05: claude/phase-33-label-evidence-eval-zac8hq, PR #28]`
- **Objective**: phase 33 fixed its frozen-label readout artifact in-phase
  (ACC 0.25 → 0.665); promote online per-slot label evidence to a proper
  mechanism (readout layer, labels still never inside perception/learning).
- **Context**: `phase33_industry_baselines.py` (the artifact + in-phase fix).
- **Done when**: readout lives outside the phase script, documented as
  eval-side; phase-33 script consumes it.
- **Outcome (2026-08-05)**: `label_readout.py` — `LabelEvidenceReadout`,
  documented eval-side (labels enter at the readout only; observe/predict
  are pinned to leave organism state bitwise unchanged). Phase-33 script
  consumes it and reproduces the committed run exactly (ACC 0.665 /
  FORG 0.200, task-accuracy matrix bit-identical to the pre-refactor
  inline logic); torch is now optional there, so the organism/prototype
  arms run on baseline-less hosts. `test_label_readout.py` pins inline
  equivalence, eval-side purity, and the remap/invalidate lifecycle
  (EventBoundary's contract, mirrored) — T1.2's eviction can forward
  notifications to it at phase-script level; no organism.py hook needed
  (organism.py untouched). Harness 27/27 both backends post-change.

### T1.4 — Re-run the phase-33 ladder (gate re-test)  `[DONE 2026-08-05: claude/t1-4-tg2wpu, PR #30]`
- **Objective**: after T1.1–T1.3, re-run the full ladder (naive SGD, EWC,
  replay, joint oracle, growing prototypes, organism-on-numba) on
  class-incremental split-digits; report ACC/FORG vs the 0.872 supervised
  prototype bar.
- **Blocked by**: T1.1, T1.2, T1.3. **Owner decision point**: gate re-opens
  only at ≥SOTA or demonstrated cost-effectiveness at equal accuracy.
- **Note**: needs torch installed (baselines only).
- **Outcome (2026-08-05)**: `phase33c_gate_retest.py` — full ladder with
  T1.2 (evict=250) + T1.3 (`LabelEvidenceReadout`, eviction invalidations
  forwarded at phase-script level) folded into the organism arm; evict=0
  kept as reproduction anchor. All anchors reproduced exactly (prototypes
  0.872/0.021, organism evict=0 0.665/0.200 incl. the flood, budget arm
  0.712/0.169, census [9,7,4,6,14]); torch arms on record: SGD 0.323/0.842,
  EWC 0.322/0.844, replay 0.913/0.105, joint\* 0.974/0.000. **Verdict:
  NOT SOTA** — 0.712 vs the 0.872 bar (23% of the gap closed by T1.2;
  FORG 0.200→0.169; task-0 retention 0.732→0.939). Recorded miss of one
  pre-registered clause: replay beats the organism on both axes and tops
  the whole ladder. Equal-accuracy cost-effectiveness branch not
  claimable. Full row: ROADMAP 33c. **The RELEASE HOLD stands — owner
  decision point reached, not passed.**

### Gap-closing follow-ons (opened 2026-08-05, after T1.4's verdict)

The remaining gap is 0.160 (0.712 vs 0.872) and T1.4's diagnostics localize
it: not flooding anymore — capacity (K=40 = 4 slots/class vs the prototype
bar's 120 prototypes) and readout richness (best-overlap majority label vs
per-class prototype geometry), plus one unexplained regression (task 1).
T1.5 and T1.6 are independent and parallelizable; T1.7 is a small
diagnostic that feeds T1.5. All three re-score on the phase-33c protocol
verbatim, evict=250 + LabelEvidenceReadout as the baseline arm.

### T1.5 — Cost-matched capacity sweep (phase 33d)  `[DONE 2026-08-05: claude/phase-33d-capacity-sweep-ja2etk]`
- **Objective**: the ladder pins K=40 by protocol, but the 0.872 prototype
  bar spends 30.7KB on 120 prototypes while the organism spends 54.4KB on
  40 slots. Sweep K ∈ {40, 60, 80, 120, 160} at evict=250, reporting
  ACC/FORG *and state bytes* per point; the headline comparison is the
  cost-matched one (organism at ≤30.7KB-equivalent state and organism at
  matched slot count 120). Phase 12 showed storage itself scales — the
  question is whether the gate gap is mostly a capacity artifact.
- **Context**: `phase33c_gate_retest.py` (protocol + baseline arm — reuse
  its stream and scorers verbatim), `phase33b_slot_budget.py` (evict
  characterization; note the E=250 window was tuned at K=40 — re-check
  staleness dynamics at higher K, the live-slot revisit interval grows),
  ROADMAP rows 33/33b/33c. T1.7's verdict (ROADMAP 33e): keep the victim
  rule AS-IS (H1 absent — no count normalization needed) and track the
  task-1 content census per K point (prediction: the 42-error capture
  floor shrinks with capacity).
- **Pre-registered predictions**: (a) ACC rises monotonically with K and
  the flood becomes irrelevant (fresh slots without eviction pressure);
  (b) honest negative to watch: if ACC plateaus below ~0.87 by K=120, the
  gap is NOT capacity — it's readout geometry, and T1.6 carries the load;
  (c) at K=120 the state-bytes comparison may favor prototypes — report
  the cost-per-accuracy curve either way, that IS the cost-effectiveness
  measurement the gate's second branch asks for.
- **Done when**: `phase33d_*.py` prints the sweep + cost curve + verdict
  vs both gate branches; ROADMAP row; harness green both backends
  (no organism.py changes expected — protocol-level only).
- **Outcome (2026-08-05)**: `phase33d_capacity_sweep.py` — 33c imported
  (stream/scorers verbatim), evict=0 twins at every K, cost-matched
  point K=24 computed against the bar's measured 30.7KB. All anchors
  EXACT. **Prediction (a) held: ACC monotone 0.712/0.735/0.837/0.854/
  0.900, crossing the 0.872 bar at K=160 (FORG 0.033) — the gap IS
  mostly capacity; prediction (b)'s plateau never appeared** (0.854 at
  K=120, still rising). Flood dissolves at K≥60 but eviction stays
  load-bearing (budget beats its twin everywhere but K=120's −0.005;
  evict=0 ladder non-monotone, 0.497 at K=60); window contingency (d)
  untriggered — E=250 transfers. **Prediction (c) held: cost branch NOT
  met at any K** (bar 35.2 KB/ACC-pt vs 45.9→412.4; K=24 cost-matched
  0.644 vs 0.872; matched-count K=120 0.854 at 7.8× bytes — the byte gap
  is representation width, not memory count). Raw-accuracy branch met
  only as a protocol variation (K=160, 12.1× bytes); the pinned-K=40
  verdict stands NOT SOTA; replay still tops the ladder. Feeds T1.6
  (readout is no longer the presumed load-carrier; byte-efficiency +
  residual K=40 gap remain its case) and T1.7 (task-1 final heals with
  capacity: 0.393→0.964 by K=160 — consistent with H1's
  born-under-pressure reading; the K=40 mechanism still needs the
  ledger). Full row: ROADMAP 33d.

### T1.6 — Readout geometry upgrade (eval-side)  `[DONE 2026-08-05: claude/label-readout-decoder-upgrades-m2mczj, PR #32]`
- **Objective**: `LabelEvidenceReadout.predict` is argmax-overlap → slot
  majority label. Prototypes win partly on readout geometry, not memory
  content. Add richer eval-side decoders over the SAME organism state:
  (i) evidence-weighted soft vote over top-m overlapping slots,
  (ii) per-slot label distribution (not majority collapse),
  (iii) optional distance-calibrated variant. Labels still enter only at
  the readout; observe/predict must stay bitwise non-mutating (pinned by
  `test_label_readout.py` — extend it, don't weaken it).
- **Context**: `label_readout.py` (T1.6 owns this file),
  `test_label_readout.py`, `phase33c_gate_retest.py` (re-score protocol).
  T1.7's measured targets (ROADMAP 33e, phase 33e section D3): the task-1
  regression's 9 label-lost errors are the exact cases prediction (a)
  describes — a 0.061 mean routing margin for a soft top-m vote to
  overcome (slot-9 type), and preserved-but-outvoted label mass on the
  routed slot itself (slot-33 type: 8 class-2 votes under 13 class-5).
- **Pre-registered predictions**: (a) soft top-m vote recovers part of the
  gap (drifted/mixed slots carry label mass the majority collapse throws
  away — the ACC 0.25→0.665 history says slot-label ambiguity is real);
  (b) honest negative: if no eval-side decoder moves ACC by >0.02, the
  memory content itself is the limit and the lever is T1.5/mechanism, not
  readout — that null is worth pinning.
- **Done when**: decoder comparison table on the 33c protocol (evict=0 and
  evict=250 arms), `test_label_readout.py` extended and green, harness
  green; the winning decoder becomes the default only if it also leaves
  the 33c anchors reproducible under a flag.
- **Boundary rule**: this is eval-side improvement, not mechanism — any
  temptation to feed readout confidence back into perception is out of
  scope (and out of premise).
- **RESULT (2026-08-05)**: the pre-registered NULL (b) is the verdict,
  robustly established. `label_readout.py` gained `predict(decoder=...)`:
  `soft` (evidence-weighted top-m vote), `dist` (count-normalized per-slot
  label distributions), `calib` (softmax(β·overlap)-weighted
  distributions) — all eval-side, bitwise non-mutating, each reducing
  EXACTLY to argmax at m=1 (`test_label_readout.py` §4, 13 new checks).
  Phase 33e (both 33c organism arms, anchors exact): seed-0 budget arm
  had five configs clear +0.02 (best calib-b8 +0.063 → 0.775) while the
  flooded evict=0 arm was dead-null (best +0.000) — but the
  pre-registered reseeding supplement sign-flipped at seed 3 (calib-b8
  mean +0.036, min −0.031; the gain is anti-correlated with argmax's own
  per-seed strength). Per the pre-registered survival rule: no decoder
  moves ACC > 0.02 ROBUSTLY — memory content, not readout geometry, is
  the gate-gap limit; the lever is T1.5/mechanism, as prediction (b)
  named. argmax stays the default; null pinned in
  `test_label_readout.py` §5; harness 31/31 both backends post-change.
  See ROADMAP row 33e + `phase33e_readout_geometry.py`.

### T1.7 — Task-1 regression diagnostic  `[DONE 2026-08-05: claude/eviction-ledger-task1-diagnostic-37exsm]`
- **Objective**: evict=250 lifted task-0 retention 0.732→0.939 but DROPPED
  task-1 final accuracy 0.464→0.393, and the era census [9,7,4,6,14] shows
  task 1 holding the fewest surviving slots. Explain it: log per-eviction
  victim (birth task, count, age, evicting-token task) across the 33c run
  and reconstruct who evicted whom, when. Small scope — instrumentation +
  analysis, no mechanism change unless the data names one.
- **Context**: `phase33b_slot_budget.py` (per-task traces),
  `phase33c_gate_retest.py`, `organism.py`/`fastpath.py` eviction paths
  (read-only; if a log hook is needed, keep it behind a debug flag and
  prove evict-mode equivalence unchanged).
- **Pre-registered hypotheses to discriminate**: (H1) task-1 slots are
  evicted disproportionately (they were born under maximum pressure with
  the lowest counts — the flood's successor pays the bill); (H2) task-1
  slots survive but their labels are stale in the readout after nearby
  evictions; (H3) task 1 is intrinsically hardest (its baseline 0.464 was
  already the ladder's low) and eviction merely fails to help. Each
  hypothesis names a different fix (count-normalized-by-era victim choice /
  readout invalidation scope / nothing).
- **Done when**: eviction ledger + verdict naming which hypothesis the
  data supports; feeds the victim-selection refinement into T1.5's sweep
  if H1 holds.
- **Outcome (2026-08-05)**: `phase33f_eviction_ledger.py` (33d reserved
  for T1.5) + a debug-only `perceive(evict_debug=...)` ledger hook in
  organism.py/fastpath.py — equivalence proven (all 33c anchors exact
  with the ledger attached; ledger on/off bitwise identical; numpy/numba
  ledgers identical row-for-row, 171 evictions; harness 31/31 both
  backends). **Verdict: H3 — H1 and H2 ABSENT, the victim-choice rule is
  exonerated** (123/171 evictions are same-task self-churn; era-1-born
  slots suffered only 4 cross-era evictions vs era-0's 31; task-1
  content coverage is HIGHER than baseline at every snapshot). Task 1
  acquires fine (A[1,1] 0.929/0.881, tokens the stream's most novel) but
  is the hardest task to RETAIN: 42/51 budget errors (45/45 baseline)
  are captured by foreign-CONTENT slots — the K=40 interference floor.
  The net −0.071 delta is readout geometry: 9 errors on two slots still
  holding task-1 content under foreign labels (one out-voted majority,
  one argmax-routing loss by 0.061 to the healthy majority-2 slot);
  class 3 recovers 0.152→0.424 while class 2 pays 0.667→0.373. **Fix
  named: none at the mechanism.** Feed-forward: T1.5 keeps the victim
  rule as-is and tracks the task-1 content census as K grows; T1.6 gains
  two measured recovery targets (routing margin 0.061 for a soft top-m
  vote; preserved-but-outvoted label mass). Full row: ROADMAP 33e.

### T1.8 — Representation-width byte reduction (phase 33g)  `[DONE 2026-08-06: claude/phase-33g-representation-width-wf4nsv]`
- **Objective**: T1.5 measured the cost-branch failure as representation
  width, not memory count (K=120 vs 120 protos: ACC -0.018 at 7.8× bytes;
  complex128 `xi` + dense K×K P). Cut bytes without touching behavior:
  (i) complex64 storage for `xi`/`mem` (compute may stay float64),
  (ii) sparse/pruned P above a count floor, (iii) optional low-rank mem
  factorization. Consolidation-as-COMPRESSION only — nothing re-attributes
  occurrences to slots, so phase 9's closed negative (post-hoc re-sorting
  inherits online mixing) does NOT apply and must not be re-opened.
- **Context**: `phase33d_capacity_sweep.py` (cost curve + state-bytes
  accounting — reuse its measurement verbatim), `organism_state.py` (E3
  schema versioning — compressed state needs a schema bump, old saves must
  load), `fastpath.py` (kernel dtype assumptions), ROADMAP row 33d.
- **Pre-registered predictions**: (a) complex64 xi halves the dominant
  term with ACC drift within harness tolerance bands (attractor capture is
  ~0.99 overlaps; 7 significant digits is plenty); (b) P sparsification
  above a count floor cuts the K×K term by >5× at K=160 (measure the
  count distribution first); (c) honest negative: if compressed-state
  ACC at K=160 drops below the 0.872 bar, report the byte-accuracy
  frontier and where it crosses; (d) target: K=160-equivalent accuracy
  inside ~2× the prototype bar's bytes — landing near replay's ~90KB
  footprint would make the cost branch arguable.
- **Done when**: `phase33g_*.py` re-runs the 33d cost curve with
  compressed arms; harness green BOTH backends + equivalence + E3
  round-trip incl. schema migration; ROADMAP row with the new curve.
- **Constraint**: inference/storage engineering only — no mechanism or
  learning-rule changes; the 33c/33d anchors must stay reproducible in
  uncompressed mode.
- **RESULT (2026-08-06)**: `organism_compress.py` (the three levers, split
  by whether they can lose anything) + E3 **schema v2** (compressed saves,
  v1 files still load, uncompressed round-trip still bitwise) +
  `phase33g_representation_width.py`. **3.85× fewer bytes at ZERO accuracy
  cost**: K=160 goes 371.2KB → 96.4KB with max |Δ task-accuracy| = 0.0000
  at every swept K, store-mode (quantize each task boundary and carry it
  forward) identical to eval-only, and paired reseeds s=0–4 drifting
  exactly 0.000. Every 33d anchor EXACT uncompressed
  (0.712/0.735/0.837/0.854/0.900). Prediction (a) HELD more strongly than
  written (predicted "within tolerance", measured identically zero);
  (b) HELD at 15.5× and turned out LOSSLESS (P is 6.1% dense at K=160, so
  CSR at floor 0 reconstructs bit-for-bit — no count floor needed);
  (c) did not fire (0.900 still above the 0.872 bar). **(d) NOT MET — the
  phase's honest negative**: only low-rank reaches inside ~2× the bar's
  bytes and it fails reseeding (seed 0 says rank-20 = 0.887 at 1.61×,
  which would have met (d); paired seeds 0–4 swing it −0.052…+0.033, and
  rank-16 −0.071…+0.019, while c64+CSR is 0.000 everywhere) — best-of-grid
  on one seed, exactly T1.6's measured selection bias. Levers (i)+(ii)
  carry the whole 3.85×; lever (iii) is a negative at this scale. Cost
  branch still NOT met but repriced: 33d's cost-matched K=24/0.644 becomes
  K=48/0.728 in the same 30.7KB; 412.4 → 107.1 KB/ACC-pt at K=160; 0.900
  now sits at 3.14× the bar (was 12.1×) and 1.08× replay's footprint.
  Dtype work: `Organism.perceive` gained a guard pinning compute width at
  complex128 — the numba kernel already promoted its inputs, so a narrowed
  store would otherwise have made the numpy path compute at a different
  width (a latent cross-backend divergence, now closed and pinned by
  `test_fastpath_equivalence.py` §7). Harness §10 (14 checks) green both
  backends. Phase 9's negative untouched: compression re-encodes a
  finished state, nothing re-attributes occurrences. Full row: ROADMAP 33g.

### T1.9 — Cost-branch follow-up: KB-per-accuracy-point parity (phase 33h)  `[DONE 2026-08-07: claude/phase-33h-cost-branch-w62bcj]`
- **Objective**: T1.8 repriced the gate's cost branch but did not meet it:
  the bar-crossing organism arm (K=160, ACC 0.900) now costs 96.4KB =
  107.1 KB/ACC-pt vs the prototype bar's 35.2 (0.872 @ 30.7KB), a ~3×
  per-point gap. Close that gap, or measure its floor as the owner's
  decision input for re-scoping the RELEASE HOLD. Either outcome is the
  deliverable — a measured "this mechanism cannot reach parity because X"
  is as valuable as parity itself.
- **Context**: `phase33g_representation_width.py` +
  `organism_compress.py` (the compressed cost curve; `store_bytes` is the
  agreed formula), `phase33d_capacity_sweep.py` (protocol + the K sweep),
  `phase33e_readout_geometry.py` (T1.6's null AND its two measured
  recovery targets: routing margin 0.061 for a soft top-m vote;
  preserved-but-outvoted label mass), `phase33f_eviction_ledger.py`
  (task-1 regression is NOT eviction — don't spend bytes there).
- **Levers, in priority order** (width is exhausted — 33g's c64+CSR is
  already zero-cost; low-rank is a measured NEGATIVE, do not retry it
  without a reseeding-robust selection protocol):
  (i) the knee of the COMPRESSED cost curve — 33d chose K on the
  uncompressed curve; re-find the accuracy-per-byte-optimal K now that
  33g moved cost-matched from K=24/0.644 to K=48/0.728 in the same
  30.7KB; (ii) readout richness at fixed K (T1.6's two recovery targets,
  eval-side, no mechanism change); (iii) slot-count reduction via
  consolidation at task boundaries (merge near-duplicate slots with the
  existing `consolidate()` machinery — fold, don't re-attribute; phase
  9's negative stays closed).
- **Pre-registered predictions**: (a) the compressed cost curve has a
  knee below K=160 where KB/ACC-pt improves but likely stays above the
  bar's 35.2 — measure where; (b) honest negative: if no lever reaches
  ≤2× the bar's KB/ACC-pt at ACC ≥ 0.872, record the measured floor and
  the reason, as the decision input for an owner re-scope of the gate to
  the capability axes. All arms reseed-verified (s=0–4 paired), per
  T1.6/T1.8's measured selection-bias lesson.
- **Constraint**: no task/label information inside mechanisms (readout
  levers stay eval-side); 33c/33d/33g anchors must stay reproducible;
  harness green both backends; `test_fastpath_equivalence.py` if any
  kernel path is touched.
- **Done when**: `phase33h_*.py` prints the measured cost frontier
  (KB/ACC-pt per arm, reseeded); ROADMAP row records parity, progress, or
  the floor; AGENT_TARGETS + the RELEASE-HOLD block updated with the
  verdict.
- **RESULT (2026-08-07)**: prediction (b)'s honest negative is the
  verdict — **parity NOT met, the ≤2× fallback NOT met, and the floor is
  measured at 76.1 KB/ACC-pt = 2.24× the bar** (ACC 0.903 vs the bar's
  0.865, K=112 + `calib-b8`), on HELD-OUT seeds. Per-point cost fell
  3.15× → 2.24× (−29%). `phase33h_cost_frontier.py`; no library file
  touched; all anchors exact (bar 0.872/30.7KB/120, 33d's ladder, 33g's
  96.4KB); harness 45/45 both backends before and after.
  - Two protocol points that generalize past this target. (1) Every arm
    reseeds the **prototype bar on the same split** — a fixed seed-0 bar
    against reseeded organism arms is unpaired, and T1.6/T1.8's lesson
    applies to the baseline as much as to the mechanism. (2) Rule (d)
    (min paired delta > 0 on all seeds) is a filter over a grid, so it
    is still selection; the phase pre-registered a **held-out
    confirmation** on fresh seeds 5–9 and it earned its keep — the
    cheapest arm (K=96, 1.96×) sign-flipped out of sample (min −0.004)
    and was recorded, not banked, exactly as 33g's rank-20 should have
    been.
  - Lever (i) knee: prediction (a) held in its first clause and FAILED
    in its second, usefully. The knee is at K=24 (23.9 KB/ACC-pt =
    **0.70×** the bar) and the organism beats the bar per point at every
    K ≤ 40 — but only because the metric is minimized where accuracy is
    worst (K=24 scores 0.583). The gate says "cost-effective AT EQUAL
    ACCURACY"; only the constrained number answers it, and unconstrained
    KB/ACC-pt must not be quoted from this phase without that caveat.
    Constrained, lever (i) alone is 3.25× — the 17-point grid bought
    resolution, not cost.
  - Lever (ii) readout: **T1.6's null is BOUNDED, not overturned.** It
    reproduces at low capacity (K=24: nothing survives) but at K=120
    five decoders clear the survival rule on all five seeds (dist-m2/m3/
    m5, calib-b8/b32; calib-b8 +0.053 [+0.012, +0.081]) — via the exact
    mechanism T1.6 named and could not exercise at K=40: per-slot label
    DISTRIBUTIONS instead of majority collapse, which only have mass to
    recover once slots are plentiful enough to split it. Zero bytes, so
    it moves the whole curve down. **Anyone citing T1.6's null should
    now cite it as "at K=40".**
  - Lever (iii) folding: a measured NEGATIVE, called by the census
    before a fold ran (prediction (c) held). At `consolidate()`'s own
    0.8 threshold the bank is near-orthogonal — 8 duplicate pairs out of
    12 720 at K=160, 13/160 slots with any partner, median pairwise
    overlap 0.169. Every byte folding saves is bought with accuracy: th
    ≤ 0.60 costs −0.147 ACC, th = 0.85 frees 0.2 slots (noise). No fold
    arm is both at/above the bar and cheaper per point. **Do not retry
    this lever at this scale without new evidence of redundancy.**
  - **The floor's reason, and it is arithmetic**: a prototype is a REAL
    float32 N-vector (256 B), a field memory is a COMPLEX one (512 B) +
    8 B meta = 2.03× per stored memory before the graph, at parity slot
    count (112 slots clear what 115 prototypes clear). Decomposition:
    68.7KB/29.4KB = 2.34× bytes × (0.865/0.903) = 2.24×, of which 1.98×
    is complex-vs-real and 0.36× is the CSR graph. Parity needs the bar
    cleared on ≤ 57 slots (K=56 measures 0.800); the ≤2× fallback needs
    ≤ 114 slots with a FREE graph — **missed by 9.9KB against a graph
    costing 10.5KB**. 33g already took the lossless width, so the
    residue is the premise, not an implementation.
  - Scope variation on record, NOT banked: this benchmark's readout
    never queries the transition graph, so a graph-free store is 58.2KB
    at 0.903 = 64.5 KB/ACC-pt = **1.90×, which would meet the
    fallback** — honest only for a product that never reasons, since the
    graph is the logic layer (phase 30, phases 35/36) and underpins
    every capability a re-scoped gate would point at. Quoting it
    unqualified is the artifact 33g refused to bank on P pruning.
  - Direction with its caveat: K=112 (0.903 at 76.1) is now cheaper per
    accuracy point than 33c's replay (0.913 at ~89.6KB = 98.1), having
    been 3.4× dearer at 33d — but replay is a single-seed torch number,
    not reseeded and not re-measured (no torch on this host).
  - **Owner decision input**: the cost branch is not closable by storage
    engineering and its floor is ~2.2×. Re-scoping the gate to the
    capability axes is the decision the data supports. Full row:
    ROADMAP 33h.

---

## Category 2 — Scale & real text

### T2.1 — Phase 27: 5M-word scale run  `[DONE 2026-08-07, session claude/t2-1-phase-27-scale-run-ycro8b — see ROADMAP row 27]`
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
- **Status (2026-08-06, PR #34)**: instrument landed —
  `phase27_5m_word_scale_run.py` (42 books / ~5.22M raw words,
  MIN_COUNT 1500, phase-24 criteria in stage B, stage-A E3 checkpoint at
  `/tmp/phase27_stageA_checkpoint.npz`, coverage_map OOM fixed by
  chunking). Stage A measured once (~14.3 min, 598 slots, 3 577 289
  in-vocab tokens); **the committed run's stage B–D outcomes are NOT yet
  recorded — that is the open piece.** ROADMAP row 27 stays "in
  progress" and this target flips to DONE only when P1/P2 verdicts land
  (partials included). Fold these merge-review follow-ups into the
  completion commit: (1) use the already-built `train_arr` in
  `coverage_map`'s member loop instead of re-allocating per slot; (2) add
  a corpus/vocab fingerprint to the stage-A checkpoint and refuse a stale
  load; (3) commit the fetch-time title-verification the docstring leans
  on (the printed curl block has no title check); (4) reconcile
  `sharpens_k`'s 4–10 window with the docstring's falsification clause
  (k=11 currently fails without being the sweep ceiling of 12) — argue
  the fix from the frozen pre-registration text BEFORE looking at the
  measured k.
- **Result (2026-08-07, committed run, all four follow-ups applied
  pre-run)**: **P1 NOT CONFIRMED** — distinctness-argmax collapsed to
  k=2 (the frozen clause's own falsification mode) and z=61 at the
  selected k < phase 24's 65 (z down at every comparable k; VALIDITY
  itself still 17–61 sigma at all k). **P2 CONFIRMED** — 'right' clears
  its tighter per-word null (n=3340, gain 0.008 vs p99 0.001);
  205/354 candidates clear theirs. Caveats + the vocabulary-composition
  confound recorded in ROADMAP row 27.

### T2.2 — Phase 26 real-text arm: calibration at V ≫ N  `[DONE 2026-08-06, session claude/v-n-acceptance-bar-calibration-jrlk86 — see ROADMAP row 37]`
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

### T2.3 — Phase 28: polysemy vs grammatical context-sensitivity  `[claimed: claude/polysemy-context-sensitivity-phase28-h0shnt, 2026-08-05 — DONE]`
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
- **Result (2026-08-05/06)**: 119-word gain-detected list (phase 23's
  stage A + stage-C gain code verbatim, but conditioned on
  `discover_categories_v2` slot categories rather than phase 23's
  word-level k-means — same population, not a bit-exact reproduction)
  split 18 polysemy (15%) / 101 context-sensitivity (85%) by
  predecessor-category-bucket successor-conflict (a proxy with no
  measured null of its own — see ROADMAP row 28). Gold-POS
  (nltk, eval-only): polysemy mean entropy 0.533 vs context-sensitivity
  0.309 bits (right direction, soft not clean); whole-detector vs gold
  minority-POS precision 0.336 / recall 0.284. 'right' itself lands in
  context-sensitivity (gold-entropy 1.667) — a measured instance of the
  pre-registered scope caveat, not an assumption. Full numbers: ROADMAP
  row 28.

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

### T3.3 — Stable symbol registry  `[DONE 2026-08-07: claude/symbol-registry-eventboundary-avk5l0]`
- **Objective**: symbol IDs decoupled from slot indices, designed at the
  `EventBoundary` seam (downstream scripts still index `org.P` by slot;
  fusion/recycling already emit remap/invalidate notifications to build on).
- **Context**: `organism.py` (`EventBoundary`, `TransitionGraph`),
  `organism_state.py` (registry must serialize).
- **Done when**: registry survives consolidate/recycle/save-load with
  stable IDs; at least one phase script migrated as proof; harness green.
- **RESULT (2026-08-07)**: `SymbolRegistry` in `organism.py`, driven only
  from `EventBoundary` — `commit` mints an ID the first time a slot crosses
  the boundary, `remap` moves it to the surviving slot or aliases it, and
  `invalidate` tombstones it permanently. Those are the three notifications
  the boundary already emitted, so there are **no new call sites in the
  perceive loop**. Consolidation is a VIEW (`mem_row` → row of `org.mem`),
  never an identity mutation, because callers snapshot/restore raw counts
  around it. Opt-in (`Organism(symbols=True)`) and observational: on both
  backends, registry-on state is **bitwise identical** to registry-off. The
  JIT kernel maintains the slot→symbol array in place and journals
  alias/tombstone events for replay, so both backends emit the same
  identity events in the same order (`test_fastpath_equivalence.py` §8).
  E3 **schema v3** serializes it (v1 and v2 files still load, into an
  organism with no registry — additive migration; `save_state_v2` added so
  the backward load is tested against a real file). Harness §11 measures
  the hazard before pinning the fix: in the oversubscribed regime 6 of 9
  epoch-A slots come to hold a different word and 0 IDs are ever reissued;
  in the fusion regime every aliased ID still resolves while its recorded
  slot has been re-let. **Honest scope**: identity is LINEAGE, not content
  — pool-mode refinement re-centers a mature trace on an ~1/eta-visit
  window, so a long-lived memory can drift onto a different word; measured,
  that happens at the same rate by ID (17/25) as by slot (16/25), so the
  registry is banded there as a tripwire, not sold as a content guarantee.
  **Migration proof**: `phase33f_eviction_ledger.py` — its script-side
  birth-era replay (which rested on a prose ordering argument) is
  reproduced by the registry on 40/40 slots on both backends, with 171
  tombstones = 171 ledger rows and 211 mints = K=40 + 171 rebirths; every
  committed number of that phase is unchanged (output byte-identical apart
  from the four new gate lines). Harness 65/65 both backends. See ROADMAP
  row E5.

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
  (T1.4 ran 2026-08-05: bar NOT met — 0.712 vs 0.872, ROADMAP row 33c;
  the hold stands pending the owner's decision.) **The gap-closing
  sequence T1.5–T1.9 is now exhausted and the decision is fully
  specified: raw-accuracy is buyable as a protocol variation (K=160,
  0.900), and the cost branch has a measured ~2.2× floor with an
  arithmetic reason (33h). What remains is an owner call on re-scoping
  the gate to the capability axes — not another agent target.**
- **When unblocked**: cut release tag on main → create separate product
  repo (hard fork, never a branch) → discoveries flow only via the
  versioned engine (phase script + pinned harness numbers + release tag).

---

## Category 6 — Consolidation & depth (opened 2026-08-07, owner posture)

**Posture, owner decision 2026-08-07: refine and enhance before pushing.**
Categories 1–3 established the measured floors; the gate decision is
specified and waiting. Rather than open new frontier scale work, this
category deepens the four things the organism is actually differentiated
on — gradient-free continual learning, the field-free logic layer,
perception ablation, and polysemy (the strongest asset) — and re-baselines
performance now that compression (33g) and the symbol registry (T3.3)
changed the shapes everything was measured on.

Standing rules apply unchanged, plus two lessons this project paid for:
**reseed every arm (s=0–4 paired) AND hold out fresh seeds for any
grid-selected claim** (T1.6/T1.8/T1.9), and **reseed the baseline on the
same split** — an unpaired fixed-seed baseline is a bug (T1.9).

### T6.1 — Task-free continual learning (phase 38)  `[claimed: —]`
- **Why this first**: the split-digits ladder hands every learner explicit
  task boundaries — the one thing the organism does not need and every
  gradient arm does. EWC needs boundaries to snapshot Fisher information;
  replay needs them to balance its buffer. Measuring on a **boundary-free
  stream** tests the differentiated claim instead of the borrowed one, and
  it is the deployment condition for the episodic-memory product (streams
  do not announce their tasks).
- **Objective**: re-run the 33c ladder on a stream with no boundary signal
  — gradual class drift / interleaved re-emergence / unannounced novelty —
  and report ACC/FORG plus a boundary-detection axis: does the organism's
  own recruitment rate localize the transitions a gradient arm must be
  told about?
- **Context**: `phase33c_gate_retest.py` (ladder + metrics),
  `phase33b_slot_budget.py` (eviction under pressure; recruitment rate is
  the boundary signal), `phase2_drift.py` + `phase11_transition_decay.py`
  (this project's own drift history — `p_decay` matters here),
  `phase33f_eviction_ledger.py` (ledger tooling for the census).
- **Pre-registered predictions**: (a) gradient arms degrade further
  without boundaries (EWC loses its snapshot trigger) while the organism's
  numbers move little — the gap narrows or reverses *without* the organism
  improving, which is the honest framing; (b) recruitment-rate spikes
  localize true transitions above a measured permutation null;
  (c) honest negative: if the organism also degrades sharply, task-free
  operation is not a free differentiator and the claim must be dropped.
- **Done when** (plus the Standing Operating Protocol, section C): `phase38_*.py` with the boundary-free protocol, all arms
  reseeded and the baseline reseeded on the same split; ROADMAP row;
  harness green both backends.

### T6.2 — Retention mechanism study (phase 39)  `[claimed: —]`
- **Objective**: characterize *what actually protects old memories*, so
  retention becomes a tunable property instead of an emergent accident.
  Factorize the contributions: eviction window E, victim rule
  (argmin-count vs count-normalized-by-era vs random ablation), `p_decay`,
  probation/confirm, and slot-count headroom. One-factor-at-a-time on the
  33c protocol with a random-eviction control arm.
- **Context**: `phase33b_slot_budget.py` (E sweep + the "stale pool must
  contain the present" finding), `phase33f_eviction_ledger.py` (H3
  verdict: task-1 loss is NOT eviction — do not re-litigate it),
  `phase11_transition_decay.py`, ROADMAP rows 33b/33f.
- **Pre-registered predictions**: (a) the victim rule matters less than
  the window (33b's finding implies the window is the load-bearing knob);
  (b) a count-normalized-by-era victim rule shows no advantage — 33f
  affirmatively did not indicate it, so this is a falsification test, not
  a fix; (c) random eviction underperforms argmin-count at every window,
  or the victim rule is doing nothing and should be simplified.
- **Done when** (plus protocol section C): factor table with effect sizes, reseeded; the tunable
  retention story written into the ROADMAP row.

### T6.3 — Logic-layer depth & optimization (phase 40)  `[claimed: —]`
- **Objective**: the field-free logic layer is the least-developed part of
  a differentiated capability. Extend and optimize it *as reasoning*, not
  as plumbing: (i) **multi-step planning under uncertainty** — plan with
  edge confidence, report path reliability, not just shortest hops;
  (ii) **compositional queries** — conjunctive/negated goals ("reach A
  without passing B") over `next_hops`; (iii) **temporal abstraction** —
  macro-edges for frequently traversed paths (`fold` already exists as the
  primitive); (iv) **optimization** — phase 30's ~20× field-free speedup
  was measured on a dense P; 33g showed P is ~6% dense at K=160, so sparse
  kstep/rollout/plan should be materially faster — measure it.
- **Context**: `organism.py` `TransitionGraph` (`kstep`, `rollout`,
  `next_hops`, `plan`, `fold`, `merge`, `retire`) + `recall_directed`,
  `phase30_symbolic_reasoning.py` (protocol + the 0.995–0.999 vs ~0.45
  null anchors — must stay exact), `organism_compress.py` (CSR P),
  T3.3's `SymbolRegistry` (stable IDs make macro-edges expressible).
- **Pre-registered predictions**: (a) confidence-weighted planning beats
  hop-count planning on a world with unreliable edges, against a measured
  null; (b) sparse-P reasoning is ≥2× faster at K≥112 with bitwise-equal
  results; (c) honest negative: macro-edges may not beat plain planning at
  this graph size — the phase-29 lesson (thin structure above the base
  level) applies, and "level 2 adds nothing" is a legitimate outcome.
- **Constraint**: reasoning ops are graph-only — the field must stay
  absent, per phase 30's isolation discipline.
- **Done when** (plus protocol section C): `phase40_*.py` with each op measured against a null +
  a timing table; phase-30 anchors exact; harness green both backends.

### T6.4 — Polysemy: act on detection (phase 41)  `[claimed: —]`
- **Why**: README names this thread explicitly — the language track
  *detects* sense structure from corpus statistics and "now needs the
  field mechanism to act on it." Phase 10 splits representations inside
  the field; phase 12/21/28 detect senses from predictions. **Closing the
  loop is the single highest-value extension of the project's strongest
  asset**, and it converts a detector into a capability.
- **Objective**: feed predictive-gain detections back into the field as a
  *split trigger* — when a word's predictive gain clears its own measured
  null, drive the phase-10 context-primed splitting for that word, then
  measure whether the resulting sense-specific slots (i) have grammatically
  distinct successor statistics and (ii) **improve downstream generation
  or next-token prediction** vs the unsplit organism. Downstream utility
  is the point; detection alone is already banked.
- **Context**: `phase12_predictive_split_test.py`,
  `phase10_context_primed_settling.py` (splitting mechanism; note the
  twice-confirmed negative — attractor pull during perception suppresses
  splitting, perception reads the field, does not bend it),
  `polysemy_organism.py` (`perceive_polysemy`/`consolidate_polysemy`),
  `phase21_working_polysemy_detection.py`, phase 28's disentangling
  verdict (split only what is lexical, not merely context-sensitive).
- **Pre-registered predictions**: (a) sense-split slots show distinct
  successor distributions vs a same-word permutation null; (b) generation
  grammaticality / next-token accuracy improves measurably over the
  unsplit organism at matched capacity — *matched*, since T1.5 showed
  extra slots alone buy accuracy and would confound the comparison;
  (c) honest negative: if utility is flat at matched capacity, sense
  splitting is a representational nicety, not a functional gain — say so
  plainly, it is a publishable negative about the whole polysemy line.
- **Done when** (plus protocol section C): `phase41_*.py` end-to-end (detect → split → score);
  matched-capacity control arm present; reseeded; ROADMAP row.

### T6.5 — Performance re-baseline & profile (phase 42 / E5)  `[DONE 2026-08-08: claude/phase-42-perf-rebaseline-8v80kw — E4 DEFERRED on measurement]`
- **Objective**: every performance number on record predates compression
  (33g), the symbol registry (T3.3), and the 5M-word corpus (phase 27).
  Re-measure and re-profile: `e2_benchmark.py` at current shapes and with
  CSR P; harness wall-clock both backends (it has grown 27 → 65 checks);
  phase-27 stage timings; and **where the time actually goes now** — the
  K×N overlap matvec was the bound at K≈1580 dense, but sparse P and
  narrowed dtypes move it. Output an honest optimization shortlist ranked
  by measured cost, which is also the **E4/GPU go/no-go input**: only
  commit to the GPU tier if permutation nulls are genuinely the bill at
  5M-word scale.
- **Context**: `e2_benchmark.py`, `fastpath.py`, `organism_compress.py`,
  `phase27_5m_word_scale_run.py` (stage timings), ROADMAP E2/E4 rows.
- **Pre-registered predictions**: (a) the numba absolute throughput holds
  or improves post-compression; (b) permutation nulls dominate corpus-tier
  wall-clock, making E4 justified — or they do not, and E4 should be
  deferred, which is an equally useful answer; (c) harness runtime growth
  is sublinear in check count (shared setup) — if not, tier the harness.
- **Done when** (plus protocol section C): benchmark table + profile + ranked shortlist + an E4
  recommendation with numbers behind it; ROADMAP E-row updated.
- **Outcome (2026-08-08)** — `phase42_perf_rebaseline.py`, protocol-level
  only (no library file touched). Misses first, per section D.
  - **(a) MISSED as written.** The pinned 58K frames/s was not reached
    (four full-load runs at the phase-23 shape: 43.8/50.1/51.4/54.3K,
    median ~51K; 1.50–1.86 min vs the pinned 1.40 min). The honest-negative
    branch required localizing rather than blaming the host, so a same-host,
    interleaved, six-round A/B ran three trees on identical work — current,
    pre-T3.3 (`7e0b639~1`), pre-33g (`92e9cc9~1`): medians **49.8/50.1/50.6K,
    indistinguishable**. Registry ON −2.4% vs a 17% within-arm spread; c64
    store +3.6% vs a 4% spread — both inside noise. **No tree-attributable
    regression.** The finding that matters is methodological: this host spans
    **43.7–58.2K on bit-identical work**, straddling the pin, so a single
    absolute throughput number cannot detect regressions. Use a same-session
    A/B. (This retires the 2026-08-04 "pin the absolute numba number" advice.)
  - **(b) FALSIFIED → E4 DEFER**, the pre-registered useful answer. Nulls are
    **265s of 1184s = 22.4%** of phase-27 wall-clock, not ≥50%: stage B 163s
    (169s observed − 5.8s measured k-means/silhouette) + stage C 102s. Amdahl
    ceiling with nulls free = **1.29×** on TOTAL corpus-tier time; E4's full
    scope incl. all-pairs (`coverage_map`, 105s) = 31.3% → **1.45×**. The bill
    is stage A perceive, **785s = 66.3%**, which E4 explicitly excludes.
  - **(c) CONFIRMED**, same-host: 27-check tree (`15cfa65`) vs 65-check tree
    = numpy 82.2s → 101.0s (1.23×), numba 12.4–13.5s → 12.5–12.7s (~1.00×)
    for **2.41×** the checks. No tiering needed.
  - **Profile** (never done before): at K=1580/N=50 the frame is 14 614 ns —
    matvec **10 927 (75%)**, free-slot scan+refine 2 135 (15%), argmax 1 441
    (10%), z update 111 (0.8%). Matvec bound HOLDS; but it is **cache**-bound
    (1.2 MB operand, 84–94 GB/s), not DRAM-bound as the E2 row said.
  - **Ranked shortlist, priced not argued** (part 6 reproduces all three):
    1. **Stage-B null as `G^T W G`** over precomputed word-pair counts —
       **bit-identical** tables (max diff 0.0), 282.6s → **0.18s**.
    2. **Stage-C null as a multivariate-hypergeometric draw** of the same
       table, O(k²) not O(n) — p99 agrees to 3 s.f. at n=3 340/37 495/259 324,
       **4.0×/41.3×/255×**. (1)+(2) take 265s of nulls under 10s.
    3. **complex64 matvec operand**: **1.48–1.77×** on 75% of the frame.
       PRICED ONLY — compute width is pinned at complex128 on purpose;
       taking it moves every committed number and needs its own target.
    4. **`coverage_map`** (105s, 8.9%) — chunked all-pairs, parallelizable.
    5. **numpy reference path**: `.conj()` is 44% of its frame (3.12s of
       7.09s); maintain a conjugate as `fastpath.py` already does. Zero risk
       to numba numbers, helps numba-less hosts and the equivalence suite.
  - **Deliberately deferred**: applying any lever (this target measures; each
    fix wants its own claim), and corpus-tier harness checks (T3.2) — though
    the 42-book fetch is now known-reproducible, which is T3.2's blocker.
  - **CSR/compression at corpus K**: P is 3.64% dense at K=1580 → CSR is
    **18.2× smaller, lossless**; full store 15.43× in 20 ms. CSR is
    **storage only** — perceive writes P in place, so the kernel needs dense.

### T4.1 (re-scoped) — Phase 31 perception ablation
Now unblocked and materially better instrumented than when written:
E3 restore is bitwise, `SymbolRegistry` survives lesions (identity is
separable from slots), and `organism_compress.py` gives a second ablation
axis (precision degradation as a graded lesion). Keep the original
pre-registered protocol; add: (i) registry-identity survival under
lesion, (ii) precision-lesion arm, (iii) recovery-after-E3-restore as the
sharpest control — the same organism, provably, before and after damage.

---

## Dependency sketch

```
T1.1 ─┬─► T1.4 ─► T5.1 (hold lifts) ─► T4.3
T1.2 ─┤
T1.3 ─┘
T2.1 ─► T2.4          T2.2, T2.3 independent (T2.3 benefits from T2.1 corpus)
T3.1 ─► cheaper nulls for T2.1/T2.3     T3.2, T3.3 independent
T4.1 ─► D4 demo       T4.2 independent of the hold (demo, not product)

Category 6 (consolidation posture, 2026-08-07) — all independent:
T6.1 task-free CL ─┐
T6.2 retention     ├─► inform any future gate re-test / product claims
T6.3 logic depth   │
T6.4 polysemy→act ─┘   T6.5 re-baseline ─► E4 (T3.1) go/no-go
T4.1 ablation (re-scoped, now unblocked)
```
