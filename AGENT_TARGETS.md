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

Verification state as of 2026-08-11, on the merged tree at `a341cc5`,
which carries everything through Category 7 — T1.9 (phase 33h), T2.1
(phase 27's outcome), T3.3 (the symbol registry), T6.1/T6.3/T6.4/T6.5
(phases 38/40/41/42) and T7.1–T7.5 (phases 43–47, architecture
economics):
E1 ALL PASS both backends (**119 checks**: 31 + T1.8's section 10 +
T3.3's section 11 + T6.3's section 12 + T6.4's section 13 + T6.1's
section 14 + T7.1's section 15 + T7.2's section 16 + T7.3/T7.5's
section 17), `test_fastpath_equivalence.py` green incl. the
narrowed-store section 7 and the symbol-registry section 8,
`test_label_readout.py` green, E3 round-trips green for schema v3
uncompressed/compressed and v1 + v2 backward load.
**Re-verified independently by the repo owner on 2026-08-11**, on a
different host from the one that ran Category 7 (Python 3.11, numpy
2.4.6 / numba 0.66.0 / scipy 1.17.1 / sklearn 1.9.0, **no torch**):
119/119 both backends (numpy 111.5s, numba 39.1s) plus both auxiliary
suites green. PR #49's verification claims therefore reproduce on a
second host — the first time this project has held a Category to that
standard.
**Torch status is host-dependent and has now changed twice**: the T6.1
host (2026-08-08) had 2.13.0+cpu and re-ran `phase33c_gate_retest.py`
reproducing EXACTLY, torch arms included — prototypes 0.872, organism
0.665/0.200, ORGANISM+B 0.712/0.169 — so the ladder's torch arms are no
longer resting on committed values alone. The Category 7 host and this
one have **no torch**. State which you have rather than assuming this
one's state. See ROADMAP "Verification log".

**Both owner decisions were ruled on 2026-08-11.**
(1) **The gate is re-scoped to the capability axes and the RELEASE HOLD is
LIFTED** — full decision record at the top of `ROADMAP.md`, required
reading before you write any release-facing artifact. It **concedes** the
cost branch rather than passing it, and the "NOT SOTA / cost conceded"
line travels with every capability claim. T5.1 and T4.3 are unblocked;
unblocked is not "released".
(2) **T7.1's storage fork was measured, not chosen blind** — T7.7 (phase
50) reported 2026-08-14: the ≤0.005 bound transfers to the ladder, every
prediction held, recommendation (A).
(3) **RULED 2026-08-14 — the fork resolves to (A).** The `real_phase` +
narrow-CSR layout is the **deployment persistence encoding** (1.148× the
bar vs (B)'s 2.066×); **complex128 stays the compute width** and the E2
contract is untouched. Decision record at the top of `ROADMAP.md`.
**This ruling does not flip a default.** `real_phase` stays **default-OFF
in code** until **T7.8** lands the flip with full anchor re-verification,
and **until then no artifact may quote 1.148× as "the current default"** —
only as *the ratified layout*. **T7.6 unblocks for arms (c)/(d) only**;
arms (a)/(b) are closed by the ruling and phase 48 is released for the
survivors.

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
14. **Documentation sweep — all five surfaces:**
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
  in-vocab tokens); at the time of writing, the committed run's stage B–D
  outcomes were not yet recorded — that was the open piece, and the
  Result block below closed it on 2026-08-07 (ROADMAP row 27 now reads
  `done`). The completion commit was required to fold in these
  merge-review follow-ups, and did: (1) use the already-built `train_arr` in
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

### T2.4 — Phase 29: recursive hierarchy  `[BLOCKED 2026-08-09 by its own pre-gate — T7.5, phase 47]`
- **BLOCKER (2026-08-09, measured not argued)**: phase 47's category-level
  chunk census clears a stream-permutation null (810.3 vs 350.4 nats) and
  **FAILS a label-permutation null outright — 810.3 against 1279.8 on 5/5
  folds**, i.e. a RANDOM partition at fixed category sizes compresses the
  stream better than `discover_categories_v2`'s. That is exactly the
  clustering artifact any Zipfian stream manufactures, and only the
  label-permutation control removes it. This target's own pre-registered
  "level 2 learns nothing" therefore FIRES on the gutenberg8 corpus.
  Category-level allocation measures 12.66 nats/slot against word-level
  24.84 — the reverse of the sandbox anchor that motivated re-opening it.
  **Do not open this without either (a) a corpus where the
  label-permutation null is cleared, or (b) a level-2 unit that is not a
  category bigram** (phase 40's finding that the boundary is ORDER, not
  graph size, points at the second). Harness §17 pins the artifact so a
  reversal has to be earned.
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
- **Status**: **UNBLOCKED 2026-08-11 — the RELEASE HOLD is lifted.** The
  owner re-scoped the gate to the capability axes; the full decision
  record, with the evidence table and the not-claimed list, is at the top
  of `ROADMAP.md` and is **required reading before any release artifact is
  written**. History: T1.4 ran 2026-08-05 and the bar was NOT met (0.712 vs
  0.872, row 33c); T1.5–T1.9 exhausted the gap-closing sequence; 33h
  measured the cost floor at ~2.2× with an arithmetic reason. The re-scope
  **concedes** the cost branch rather than passing it — do not write
  otherwise.
- **Unblocked is not "released".** Cutting the tag is a separate owner
  action requiring its own go-ahead. Do not cut a tag, create the product
  repo, or publish anything on the strength of this status line alone.
- **Binding on every artifact this target produces**: the four capability
  axes and the "NOT SOTA on class-incremental accuracy / cost branch
  conceded" line appear **together**, in the same breath, every time.
  Replay scores 0.913/0.105 and tops the ladder above even the prototype
  bar; the ≤2× fallback is missed at a measured 2.24× (2.07× narrowed). A
  capability claim quoted without its benchmark caveat misrepresents the
  decision. If a draft cannot survive stating both, the draft is wrong —
  not the caveat.
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

### T6.1 — Task-free continual learning (phase 38)  `[DONE 2026-08-08: claude/agent-target-open-items-uyyq8o — BOTH predictions falsified, see ROADMAP row 38]`
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
- **RESULT (2026-08-08, `phase38_task_free_continual.py`, 5 paired seeds,
  torch 2.13.0+cpu)**: **both headline predictions FALSIFIED, and what
  failed is this target's own premise.** Read the premise above — "EWC
  needs boundaries to snapshot Fisher information, replay needs them to
  balance its buffer" — then the measurement: removing the boundary
  **signal** with the stream held fixed costs the gradient arms essentially
  nothing. mlp-seq +0.000 (it never used boundaries), **mlp-ewc +0.056** —
  it *gains* from a misaligned period-5 snapshot schedule versus being told
  the true task ends — mlp-replay −0.005 for reservoir vs a per-task
  balanced buffer. Prediction (a) fails on **0 of 5 seeds** and in the
  opposite direction: on the drifted stream the gradient arms improve
  enormously (mlp-seq 0.296→0.924, mlp-ewc 0.318→0.910) while ORGANISM+B
  moves +0.003. Prediction (b) fails with the **sign reversed** —
  recruitment is *lower* at unannounced-novelty chunks (pooled −1.39
  slots/chunk, z = −1.53 vs 2000 permutations, p = 0.94, 0/5 seeds
  positive), because gradual drift smears novelty and leaves no spike to
  localize. (c)'s collapse clause did not fire (+0.003 > −0.10).
- **Why, and it is worth internalizing before writing the next target**: a
  three-condition design (A blocked+told, B same stream untold, C drifted)
  decomposes A→C into "signal removed" and "stream unblocked", and **all of
  the movement is in the second**. Catastrophic forgetting is caused by
  **sequential blocking**; a task boundary is only the annotation of the
  blocking, and the annotation is nearly worthless once the blocking
  exists. Any drifting, re-emerging stream is *interleaved* — the textbook
  cure for forgetting — so "task-free" as specified here hands the gradient
  arms a gift. That is a confound in the framing, not in the measurement.
- **What survives, pinned exact in harness §14**: ORGANISM+B is **bitwise
  invariant** between A and B (max |ΔACC| = 0.0e+00 across seeds) because
  it consumes no boundary information at all. mlp-seq and prototypes are
  exactly invariant too, which is the internal-validity check that the
  condition plumbing measures what it claims rather than leaking. The
  invariance is architecturally real; it is just not worth much here.
- **What was dropped**: "task-free operation is a differentiator" is not
  claimable from this protocol. NOVELTY N4 edited down (not deleted): its
  "everyone else needs a teacher marking where each lesson ends" line is
  now measured false and says so. In condition C, replay reaches
  0.968/0.031 against ORGANISM+B's 0.759/0.140 while storing 200 raw
  labelled samples. Phase 33c's boundaried verdict is untouched.
- **Follow-on this exposes** (deliberately NOT smuggled into this target):
  the question worth asking is the reverse one — *how blocked must a stream
  be before gradient methods need the annotation at all?* That is a sweep
  over blocking, and it is the honest version of what T6.1 was reaching
  for. It would also give T6.2 its stream.

### T6.2 — Retention mechanism study (phase 39)  `[DONE 2026-08-11: claude/phase-39-retention-mechanism-r7k2qm — (a) NOT SUPPORTED, (b) holds, (c)-strong CONFIRMED; see ROADMAP row 39]`
**Outcome in one paragraph.** Retention at 33c's blockedness has **two**
load-bearing knobs, not one. **(a) is NOT SUPPORTED**: the victim rule does
not matter less than the window — pooled n=10 FORG spread of the level means
is 0.4386 across rules vs 0.4244 across windows (full range), and 0.0536 vs
0.0554 with each axis's catastrophic setting dropped, a 0.0018 gap that is
rounding rather than a ranking. 33b's "the window is the load-bearing knob"
was an artifact of having swept only the window. **(c)-strong is CONFIRMED
and is the headline**: uniform-random eviction loses to argmin-count at every
window (dACC −0.279/−0.282/−0.327/−0.339, dFORG +0.457/+0.439/+0.465/+0.446
at E=100/250/500/750), 10/10 sign census, exact paired p=0.0020 — the n=10
floor — so the simplification branch does **not** fire and the victim rule
must not be simplified away. **(b) holds and harder than written**: the
count-normalized-by-era rule has no advantage anywhere and is affirmatively
**worse** on FORG at all four windows (+0.035…+0.060), task-0 final falling
0.90 → 0.70; 33f's refusal to indicate H1 is vindicated by direct test. All
three pre-registered **mundane accounts were rejected**, so none of this is
true by construction — decisively, random evicts memories **4–15× better
established** than argmin-count does (victim lifetime count 217.1 vs 14.4 at
E=100), so the stale pool has real spread and argmin-count exploits it.

**What a stranger should do with this.**
- **33b's law now has a second clause.** "The stale pool must contain the
  present, or eviction eats the past" **AND the pool must be ordered by how
  established its members are, or eviction eats the past just as fast.**
  Sharp form, symmetric across both knobs: **a mis-set budget is worse than
  no budget.** E=0 gives ACC 0.551 / FORG 0.291; E=2000 gives 0.439 / 0.577;
  random at E=250 gives 0.449 / 0.591; E=250 with argmin-count gives
  0.730 / 0.152. Never ship `evict>0` without checking BOTH knobs.
- **Two knobs buy retention** (pooled n=10, held-out sign-consistent): slot
  headroom K (K=96: dFORG −0.111 / dACC +0.127, 10/10; roughly monotone, but
  K=56 wobbles out of sample so monotonicity is a trend, not a pin), and —
  **not pre-registered, found by the sweep** — probation/confirm, phase 14's
  provisional-slot machinery which **33c's own recipe leaves off at
  confirm=0** (confirm=3/probation=1000: dFORG −0.062 10/10, dACC +0.056).
  Quoted at its held-out value; it is a grid-selected result, not a tuned
  recommendation, and anyone banking it should reseed again.
- **New default-off hook**: `perceive(evict_victim='count'|'rate'|'random')`
  in organism.py + fastpath.py, with `org.tenure` and `org.evict_rng`
  round-tripping additively in E3 (no schema bump). `'count'` is the default
  and its path is **bitwise identical to pre-T6.2 main on both backends**.
  Pinned by harness §18 (13 checks).
- **`p_decay` (F3) is a GAP, not a null.** Every level returns byte-identical
  ACC/FORG. The parameter is wired (graph mass 2782 → 93.6 at 0.01) but
  touches only the transition graph, and 33c's readout decodes from `xi`
  alone. **Do not record "transition decay does not affect retention"** —
  it was never measured. Measuring it needs a metric that consumes P
  (next-symbol prediction / planning). Deliberately deferred rather than
  patched by swapping protocols mid-study.
- **SCOPE, non-negotiable**: every number is retention **at 33c's
  blockedness**. Phase 38 measured that the forgetting here is produced by
  sequential blocking; 33c is one fully-blocked point and **T6.6 (phase 49)**
  is what varies the axis. Nothing here transfers to a less blocked stream
  without re-measurement.
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

### T6.3 — Logic-layer depth & optimization (phase 40)  `[DONE 2026-08-08: claude/phase-40-logic-depth-o2w1p5 — see ROADMAP row 40]`
**Outcome in one paragraph.** (a) CONFIRMED: confidence-weighted planning
beats hop-count planning by +1.04 nats/plan of true log-reliability (0/5
sign flips) above a shuffled-shrinkage null, held-out verified at the
pre-registered alpha. Against the MLE planner phase 30 already had — a
sharper comparison that was NOT pre-registered — the arms disagree on 13.7%
of plans and confidence wins +0.379 nats when they do at 2000 observations,
decaying to −0.074 at 60 000; it clears T1.6's survival rule on the
selection seeds and FAILS on held-out seeds, so it is recorded as a
direction and not banked. The durable half of the op is `PathReport`
(reliability, weakest link, that link's observation count), not the router.
(b) SPLIT: equality MET (next_hops and rollout bitwise incl. the RNG stream,
kstep to 1e-17), speed NOT MET as written — 2× arrives at K=800, not K=112,
reaching 6.1× at K=1580; sparse full `kstep` is a measured NEGATIVE (0.1–0.2×
— a dense K² output is BLAS's home ground) and the real win is `kstep_row`
at 4×→356×. (c) The negative fired and its condition is now measured: mined
first-order macros move reliability by exactly 0.0 (forced by construction)
and the pass-through census finds nothing to fold, but observed trigram
macros gain +0.443 nats on a second-order world (held-out +0.581, survives)
while collapsing to |Δ|≤4e-04 on a first-order control. **The boundary is an
ORDER, not a graph size** — worth carrying into T2.4 (phase 29), whose
"level 2 learns nothing" risk is the same question one level up. Library
additions are purely additive on `TransitionGraph` plus `MacroGraph` /
`SparseTransitions`; `next_hops` moved onto a shared Dijkstra and is pinned
bitwise against its pre-refactor body in harness §12 (12 new checks, 77
total, both backends).
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

### T6.4 — Polysemy: act on detection (phase 41)  `[DONE 2026-08-08: claude/phase-41-detection-phase-10-9u1bkc — see ROADMAP row 41]`
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
- **Outcome (2026-08-08)**: **the loop closes, asymmetrically.** Misses
  first. Generation does NOT improve — both generation metrics fail the
  survival rule by sign flip (corpus-bigram hit +0.001, modal-role
  grammaticality −0.005 with a −0.053 seed), so that half of prediction (b)
  is falsified. Word-level likelihood beats the matched control (+0.145
  nats/token) but LOSES to the unsplit organism (−0.081): fragmentation
  costs more than the split's information buys on that axis. Held-out
  next-*word* accuracy is saturated (Bayes ceiling 0.144, matched control
  already 0.132) and cannot discriminate — recorded as below-threshold, not
  as a null. Fragmentation got WORSE under gating, not better: 11.3 slots
  per split dual word vs ungated phase 10's 3–4, because gating hands the
  whole free-slot budget to three words — phase 10's open "exact 2/1 slot
  structure" problem is amplified here, not closed. Detector false-positived
  on 3/26 controls at one seed of five (margins +0.001…+0.011).
  What held: prediction (a) CONFIRMED 15/15 — every split dual word's
  sense-slots are distinct from a same-word permutation null (mean pairwise
  TV 0.304–0.420 vs p99 0.107–0.127) — and 0/3 of the detector's own false
  positives clear that test, so distinctness rejects exactly what the
  detector should not have flagged. Detection: 3/3 dual words at every seed,
  0.34–0.42 bits against ~0.006 nulls, on self-discovered categories (k=3
  every seed). **Prediction (b) survives on held-out next-CATEGORY accuracy:
  +0.063 over an EXACTLY matched-capacity control (positive on all five
  seeds; 56% of the 0.114 headroom to the eval-only ceiling 0.786).** The
  capacity match is exact at every seed and pinned in harness §13. Note the
  threshold caveat: +0.02 is T1.6's ACCURACY bar, so the ll/token survival
  is corroboration, not a second independent win.
- **What a follow-on should take from this**: (i) the fragmentation is now
  the binding problem, not the trigger — a sense-COUNT criterion (how many
  senses, per NOVELTY N1's mini-roadmap step 2) would do more than any
  further work on the gate; (ii) do NOT quote the +0.073 split-vs-unsplit
  delta — only +0.063 is capacity-controlled, and the control is what showed
  the confound was small (matched beats unsplit by just +0.010) rather than
  something that could be assumed; (iii) every number is on the SYNTHETIC
  phase-8/10 corpus where polysemy is lexical by construction — phase 28
  measured that real text conflates lexical polysemy with grammatical
  context-sensitivity (~15% lexical at POS level), so a real-text arm is a
  re-derivation, not a port.

### T6.5 — Performance re-baseline & profile (phase 42 / E5)  `[DONE 2026-08-08: claude/phase-42-perf-rebaseline-8v80kw — E4 DEFERRED on measurement]`
- **Objective**: every performance number on record predates compression
  (33g), the symbol registry (T3.3), and the 5M-word corpus (phase 27).
  Re-measure and re-profile: `e2_benchmark.py` at current shapes and with
  CSR P; harness wall-clock both backends (it has grown 27 → 72 checks, and
  T6.4's §13 is the most expensive single section — ~16s of the numpy run);
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

### T6.6 — The blocking sweep (phase 49)  `[claimed: —]`
**Opened by the repo owner 2026-08-11.** T6.1 ended by exposing a
follow-on and deliberately refusing to smuggle it in; it was left with no
home for three merges. This is that home. Read T6.1's RESULT block first —
this target exists *because* T6.1's framing carried a confound, and the
whole point is to measure the axis the confound was hiding.
- **The question**: T6.1 decomposed A→C into "boundary signal removed"
  (worth almost nothing: mlp-seq +0.000, mlp-ewc **+0.056**) and "stream
  unblocked" (worth almost everything: mlp-seq +0.629). Catastrophic
  forgetting is therefore produced by **sequential blocking**, and a task
  boundary is only the annotation on it. So the honest question is not
  "what happens without boundaries" but ***how blocked must a stream be
  before gradient methods need the annotation at all?***
- **Objective**: sweep blockedness as a continuous axis — from fully
  interleaved (i.i.d. class mixture) to fully blocked (33c's hard task
  switches) — with the boundary annotation ON and OFF at each level.
  The deliverable is the **crossover point**: the blockedness at which
  (i) gradient arms begin to collapse, and (ii) the annotation starts
  being worth more than nothing. Report both as curves with paired seeds,
  not as a verdict.
- **Why it is worth running**: every continual-learning claim this project
  can make, including the ones N4 still carries, is a claim about *where
  on this axis* the organism wins. T6.1 showed we have been quoting a
  single point (fully blocked) and a single mis-specified alternative
  (fully drifted) as if they were the whole space. Locating the crossover
  turns N4 from a slogan into a scoped, defensible statement — and it is
  the only route left to one, since T6.1 removed the unscoped version.
- **Context**: `phase38_task_free_continual.py` (its three conditions are
  three points on this axis and its harness §14 invariances are the
  internal-validity check to keep), `phase33c_gate_retest.py` (the fully
  blocked endpoint and the ladder), `phase2_drift.py`,
  `phase11_transition_decay.py`.
- **Pre-registered predictions required before any committed run**, plus
  the **named mundane account** Category 7 made standard. State in advance
  where you expect the crossover and what would falsify a crossover
  existing at all — a monotone curve with no knee is a real and publishable
  outcome, and it would mean "blockedness" is not a threshold phenomenon.
- **Standing warning**: ORGANISM+B was **bitwise invariant** to the
  annotation in T6.1 and should remain so at every level of this sweep —
  it consumes no boundary information. If it ever moves, that is a plumbing
  leak, not a finding. Pin that invariance in the harness section.
- **Done when** (plus protocol section C): `phase49_*.py`, paired seeds
  with the baseline reseeded on the same split, both crossover curves,
  ROADMAP row, harness section, and **N4 rewritten to the measured scope**
  — edited down as readily as up.

### T4.1 (re-scoped) — Phase 31 perception ablation
Now unblocked and materially better instrumented than when written:
E3 restore is bitwise, `SymbolRegistry` survives lesions (identity is
separable from slots), and `organism_compress.py` gives a second ablation
axis (precision degradation as a graded lesion). Keep the original
pre-registered protocol; add: (i) registry-identity survival under
lesion, (ii) precision-lesion arm, (iii) recovery-after-E3-restore as the
sharpest control — the same organism, provably, before and after damage.

---

## Category 7 — Architecture economics (owner work order, 2026-08-08)

**Provenance, and it is load-bearing.** T7.1–T7.5 originate from an
exploratory sandbox session (2026-08-08) that cloned `main`, ran on the
phase 20/21 Gutenberg corpus and the phase 19 recipe, and produced **no
branch and no repo changes**. Nothing it reported is a result. Every
sandbox number quoted in these targets is an **indicative anchor to be
re-derived under the SOP** — hypothesis, not finding. Where a measurement
here disagrees with a sandbox anchor, **the measurement wins and the
disagreement is reported as a finding**, not reconciled away.

Standing additions for every target in this category, on top of the SOP:
pre-register a **named mundane account** (the most boring reason the
prediction fails) alongside every prediction; check baseline/competitor
behavior against the literature *before* predicting it; tag any NOVELTY
claim as measured vs premise-derived; band negatives so a reversal must be
earned.

Phase numbers reserved by this claim: **43, 44, 45, 46, 47** (48 held for
T7.6 and NOT to be used until the fork is resolved).

### T7.1 — Phase-channel audit and the storage fork (phase 43)  `[DONE 2026-08-09: claude/repo-agent-arch-economics-4adfij — the fork is priced and WITH THE OWNER; see ROADMAP row 43]`
- **RESULT (2026-08-09), misses first.** (1) **The work order's own
  mechanistic account is FALSIFIED** — deriving the perceive recursion
  rather than describing it gives `z_t ~ Σ_j b a^j x_{t−j}` with
  `a = 1 − g_in·dt + i·omega·dt`, a phasor sum, so the reachable set is NOT
  {rotation of a real vector}. Measured: R(omega=0) = exactly 0 at every
  hold, R ∝ omega², 0.33 by omega ≥ 4, and R PEAKS at shallow settling
  (hold 2–4). The channel is empty by **parameter regime**, not
  construction. **T7.6(a) is therefore a parameter setting, not a new
  mechanism.** (2) **The behavior null is weaker than it looks**: the
  pre-registered positive control measured the readout's resolution floor
  at **R ≈ 0.06** (substitution costs exactly 0.0000 there and −0.1256 only
  at R = 0.462), and the store sits ~230× inside it — so `dACC = 0` is not
  the evidence; the residual measurement is. **T7.6(b) inherits this as a
  bar**: a phase-carrying arm must move the readout past R ≈ 0.06 or score
  as a null whatever the dynamics do. (3) **The sandbox's ~1.13× is
  contradicted**: derived 1.32× from the layout before measuring, measured
  **1.318×** held-out; 1.02× is the memory term alone and the sandbox total
  drops 33h's graph term. (4) Parity remains unreachable by this lever.
  **Held**: median R **2.55e-04** held-out over 560 live slots (92% < 1e-3,
  99.8% < 1e-2); cross-corpus at the phase 19/20 recipe (fables 410/413 <
  1e-3 — the sandbox's 416/418 reproduces; gutenberg8 1199/1200);
  store-mode `dACC` held-out −0.0005 [−0.0050, +0.0025]; the per-slot phase
  is **GAUGE** (exactly 0 ACC delta under random per-slot rotation); 33h's
  76.10 KB/ACC-pt = 2.236× reproduced exactly. **M1 measured and left
  standing**: no committed pipeline builds a complex-valued stream (census
  of 54 files), and a constructed one reaches R = 0.462 — the width is
  usable, just unused. **THE FORK: (A) 44.83 KB/ACC-pt = 1.32×, ≤2×
  fallback MET, imaginary channel forfeited permanently; (B) 76.10 = 2.24×,
  floor stands and the option-(a) re-scope is forced.** Priced, not chosen.
  Harness §15 (10 checks) green both backends.
- **Original target text below.**
- **Why first**: every other cost lever is small next to this one, and
  T7.6's design depends on which branch the owner picks.
- **Objective**: T1.9 recorded "nothing left to engineer" for the cost
  floor. That is established for **lossless width narrowing** (33g,
  complex128→complex64). It is NOT established for a **representation
  change**. Audit whether the stored `xi` actually uses the imaginary
  channel; if it does not, price the `(real N-vector, phase scalar)` layout
  against the floor's 2.03× complex-vs-real term.
- **Sandbox anchor (hypothesis)**: after optimal global de-rotation the
  residual imaginary energy had median 1e-4, 416/418 slots below 1e-3;
  substituting the de-rotated real part left word→slot assignment agreement
  1.0000 and overlap-profile correlation 0.9996 — on the phase 19 fables
  recipe.
- **Mechanistic account to test alongside it**: `perceive`'s only phase
  source is `dz = 1j*omega*z + g_in*(x - z)`. If that is a content-blind
  global rotation, the reachable set is exactly {e^{iθ}·real} for real
  inputs and the emptiness is structural, not incidental.
- **Do**: (1) reproduce the audit at the **33h configuration** (K=112, the
  arm the floor was measured on), reporting the residual DISTRIBUTION not
  just its mean; (2) test the mechanistic account — residual under (a) a
  different corpus, (b) `omega` swept including 0, (c) complex-valued
  inputs if any pipeline produces them; (3) cost the layout (256 B + 4 B vs
  512 B) and **derive the floor from the layout before measuring it**;
  (4) verify losslessness against every pinned anchor on both backends.
- **Mundane account**: the residual is at floor only because the inputs are
  real; a pipeline with genuinely complex inputs would use the width and
  the saving is an artifact of one embedding choice. Step (2) must be able
  to distinguish this.
- **Deliverable**: an **owner decision artifact**, not a recommendation.
  The fork is mutually exclusive — (A) spend it: halve the store, meet the
  ≤2× fallback, permanently forfeit phase-16 binding including the
  path-encoding T7.5/T7.6 would need; (B) cash it in: keep the width and
  make it load-bearing (T7.6), leaving 2.24× standing and forcing the
  option-(a) gate re-scope. **Price both, choose neither.**

### T7.2 — Sparse P as the default path (phase 44)  `[DONE 2026-08-09: claude/repo-agent-arch-economics-4adfij — see ROADMAP row 44]`
- **RESULT (2026-08-09), misses first.** (1) **A premise of the target was
  already spent**: 33h's 0.36× graph term IS the CSR number, not a dense
  number awaiting sparsification (dense float64 100 352 B vs the measured
  10 116 B), so **CSR-as-default-storage moves the floor by exactly
  0.000×**. (2) **The mundane account FIRED — density is not scale-free.**
  Over 16× observations at fixed K: digits 0.099 → **0.374 (3.79×)**,
  gutenberg8 0.0077 → **0.0466 (6.08×)**. P3 MISSED as written; growth is
  sublinear but unsaturated. **Standing consequence: every CSR byte number
  in this project needs an observation count attached**, phase 42's 3.64%
  at K=1580 included. (3) The compute crossover HELD inside its band but
  the honest reading is that sparse **loses at every shape this project
  reasons at**: `next_hops` dense/sparse 0.72× at K=112, 0.95× at 400,
  **1.15× at 500** (crossover), 2.84× at 1580; `rollout` negative at every
  K; full `kstep` 0.08–0.23×; `kstep_row` wins from K≈200.
  **Held**: the narrow-CSR derivation landed to the byte (derived
  `4·nnz + 2·(K+1) = 5058 B`, measured 5058 B, P reconstructed identically),
  and the floor moves 2.236× → **2.066×** (c64) and 1.318× → **1.147×**
  (real+phase). **The pre-registered rule fired: the ≤2× fallback is still
  MISSED by graph engineering alone**, which settles that the fallback is
  decided by the complex-vs-real term and never by the graph. It also
  reconciles T7.1's contradiction — **1.147× is the sandbox's unreproduced
  ~1.13×**, so the sandbox was pricing a narrowed graph too and said so
  only for the store. **Default changed, deliberately inert**:
  `next_hops(sparse=None)` dispatches on `SPARSE_MIN_K = 800` **and on
  integer counts** (T6.3's bitwise condition), set at the ≥2× point rather
  than the measured 500 break-even because phase 42 measured ±20% host
  variance; phases 30/40 reason at K ≤ 60, so the dispatch cannot move an
  anchor. Harness §16 (10 checks) green both backends.
- **Original target text below.**
- **Objective**: the graph contributes 0.36× to the cost floor; 33g
  measured P at ~6% density at K=160. T6.3 already built
  `SparseTransitions` and moved `next_hops` onto a shared Dijkstra with the
  dense body pinned bitwise (harness §12).
- **Do**: make CSR the default storage/compute path **where T6.3 measured
  it to win**, keeping dense where T6.3 measured a negative (full `kstep` —
  a dense K² output is BLAS's home ground; the win was `kstep_row`).
  Extend the bitwise pinning to every op whose default path changes.
  Re-derive the graph term in the cost floor.
- **Mundane account**: density is 6% at K=160 but P may densify with corpus
  size; if density rises with observation count the saving evaporates at
  scale. Measure density **as a function of observation count**, not at one
  point.

### T7.3 — Settling-depth sweep (phase 45)  `[DONE 2026-08-09: claude/repo-agent-arch-economics-4adfij — see ROADMAP row 45]`
- **RESULT (2026-08-09), misses first.** (1) **A PROCESS MISS**: this
  phase's predictions were authored before its first run but committed
  after it. Phases 43/44/46/47 pre-registered in the required order; this
  one did not, and the row says so. (2) **The mundane account is
  FALSIFIED, and the way it fails re-frames the question.** 1-NN prefix
  decoding at a fixed final token is **1.000 at every hold from 1 to 12**
  (null p99 0.083), with ceiling ≥ 0.99 from hold=2 — so the retained
  information is prefix IDENTITY, not un-converged noise, and it is not
  confined to a window. (3) What the sweep moves is the prefix signal's
  MAGNITUDE in similarity units — `ceiling − suffix` = 0.375/0.406/0.307/
  0.207/0.078/**0.027**/0.003 at hold 1/2/3/4/6/8/12 — and every consumer
  in the mechanism is a THRESHOLD on that similarity (`recruit` 0.75,
  `active_bar` 0.6, `fuse_bar` 0.7, `merge_thresh` 0.8). **Path sensitivity
  is a property of the consumer, not of the state.** (4) The work order's
  own flag is off by one step: `merge_thresh = 0.8` crosses suffix between
  **hold=4 (0.7931) and hold=6 (0.9219)**, not "from hold=4 upward". The
  flag stands and remains untested for sequence-state consolidation.
  **Held**: P1 exactly (hold=8 ceiling 1.0000, suffix 0.9735, PSI 0.0302);
  P2 at 21.7× (bar was 5×); the `g_in·dt` knob reproduces the curve at
  fixed frame count, so this is settling depth and not exposure length.
  **P4 HELD and the trade is steep**: coverage 128/199/265/301/320/**327**/
  321 of 376, and category-validity z = 1.3/2.1/2.6/6.8/10.5/**9.5**/9.3 —
  **at hold ≤ 3 the categories do not clear their own null at all**. No
  free lunch; hold=8 is not over-settling. **Consequence for T7.6**: arm
  (c), the dual time constant, is the only one that buys path sensitivity
  without paying this bill — it adds a second O(N) state instead of moving
  the one state everything else reads.
- **Original target text below.**
- **Objective**: sandbox measurement (hypothesis) — at the phase 19
  exposure setting (`hold=8`) the end-of-sequence field state has a
  same-sequence ceiling of exactly 1.0000 and a suffix-sharing similarity
  of 0.980: a pure last-token attractor that has also forgotten its own
  initial condition. A window at hold≈3 held reproducibility 0.945 with
  suffix-sharers at 0.703.
- **Do**: sweep `hold` (equivalently `g_in·dt`) against slot coverage,
  category validity (phase 24 MI-vs-null), and a path-sensitivity index
  `(ceiling − suffix)/(ceiling − floor)`. Report the trade on the axes the
  project already measures.
- **Flag for consolidation**: the default `merge_thresh=0.8` sits between
  ceiling and suffix at hold=3 and *below* suffix from hold=4 up, where
  distinct suffix-sharing states would be merged. If sequence states are
  ever consolidated this threshold is load-bearing and is currently
  untested for that use.
- **Mundane account**: the time constant `1/(g_in·dt)` ≈ 4 frames fully
  explains both numbers; there is no window, only an EMA, and hold=3
  "works" solely because settling is incomplete — i.e. the state is noisy
  rather than path-aware. Distinguish by testing whether the retained
  information is prefix IDENTITY or just un-converged noise.

### T7.4 — Single-pass audit (phase 46)  `[DONE 2026-08-09: claude/repo-agent-arch-economics-4adfij — see ROADMAP row 46]`
- **RESULT (2026-08-09), miss first.** **P1 MISSED, in the direction that
  strengthens the claim.** Predicted: one epoch loses ≥10 points of
  coverage. Measured on fables, 1 vs 15 epochs: **327/376 → 320/376** — the
  extra fourteen passes *cost* seven words, with consolidated memories
  falling 405 → 356, i.e. the epochs fuse memories one pass had already
  separated. On gutenberg8, 1 vs 3: **387.3 → 388.3 of 395**. **Held**: P2
  on both corpora (a single pass clears the permutation null at **z = 9.1**
  fables / **z = 32.0** gutenberg8, strengthening to 12.7 / 38.3), and P3
  (words above phase 21's noise floor **50.0 → 51.3**, +3%, inside the
  ±20% rule — so re-reading a fixed corpus does NOT manufacture detection
  signal, the failure mode the rule was written to catch). **Two scope
  sentences that must travel**: the 1-vs-15 comparison exists only on the
  2 431-token fables corpus (the large corpus was audited 1-vs-3 at 3.3M
  frames/epoch), and the detection axis has a wide three-seed spread
  (39/39/71 at one pass), so it is "not epoch-sensitive at the resolution
  three seeds resolve". On fables the detection axis is not measurable at
  all — one candidate clears the n≥100 bar in 2 431 tokens — which is
  phase 19's own n=200–500 finding reappearing. **NOVELTY's one-sentence
  claim now carries the audit and both scope sentences**, per the
  pre-registered rule.
- **Original target text below.**
- **Objective**: phase 19's recipe runs 15 epochs. "Single-pass online" is
  one of the capability axes any gate re-scope would stand on; a re-scoped
  gate must not rest on an axis the production recipe violates.
- **Do**: re-measure the headline results (coverage, category validity,
  polysemy detection) at **1 epoch** against the multi-epoch numbers, paired
  seeds. Report which survive. **This is an audit, not an improvement
  target** — a large drop is a legitimate and important finding.
- **Mundane account**: multi-epoch is doing ordinary optimization work any
  online learner would need; the honest claim is "single-pass capable,
  multi-pass tuned", and the fix is a wording change in NOVELTY rather than
  an experiment.

### T7.5 — Macro-Recruit pre-gates (phase 47)  `[DONE 2026-08-09: claude/repo-agent-arch-economics-4adfij — T2.4 STAYS BLOCKED; see ROADMAP row 47]`
- **RESULT (2026-08-09), misses first — four of them.** (1) **P2 FAILED and
  its pre-registered consequence was honored**: the frequency arm gained
  **+2286 nats under the predecessor-permutation null**, so the raw measure
  is broken and sections 1–5's numbers are diagnostics, not answers.
  Diagnosis: merging shortens the stream and enlarges the table, and an
  add-α model refitted on the longer table fits its own data better — a
  length/estimation effect present with no sequential structure at all.
  (2) **P1 MISSED on the pre-registered statistic** — held out, plain
  frequency beats expected-total-gain at every M (M=64: 2519.8 vs 1589.9),
  reversing the sandbox's in-sample ordering; the M1 check clears
  total-gain of being frequency in disguise (overlap 0.55–0.62). (3) **P3
  MISSED — the sandbox's own UNK confound does not reproduce**: the UNK
  share is exactly 0.253 as reported, but the gain ratio is **1.15×**, not
  ≥2×, and the payoff is **0.93% of total cross-entropy** either way.
  (4) **P5 MISSED / P6 FIRED, and this is the consequential one.** The
  category arm clears the stream-permutation null (810.3 vs 350.4) and
  **FAILS the label-permutation null outright — 810.3 against 1279.8 on
  5/5 folds**, i.e. a random partition at fixed sizes compresses better
  than the discovered one. That is exactly the clustering artifact M5
  named. **Phase 29's "level 2 learns nothing" FIRES on this corpus:
  T2.4 stays blocked on its own honest negative and no chunk mechanism
  should be built at the category level.** Category-level allocation is
  12.66 nats/slot vs word-level 24.84 — the reverse of the sandbox's
  ~+91 vs +11.7. **Held**: P4 (peak at **M=64**, marginal block negative
  from M=128 **even with the table cost removed**, so the reversal is
  language and not accounting — the stopping rule phase 41 lacked); and
  P7, the null-corrected statistic, **committed before the run that tested
  it and confirmed on a corpus half no selection had touched** —
  expected-total-gain beats frequency on every confirmation fold at
  M=32/64/128 (M=64: 1984.2 vs 51.1). **Net: the chunk-selection rule
  survives on a repaired measure, at the WORD level only, worth ~1% of
  cross-entropy.**
- **Original target text below.**
- **Objective**: replicate three sandbox pre-gates **under the SOP** before
  any chunk mechanism is built. Sandbox anchors, all indicative:
  expected-total-gain chunk selection beat frequency-only and
  divergence-only over 5 seeds (+2908 / +1740 / +880 nats at 64 slots), all
  arms negative under a predecessor-permutation null; **the top chunks were
  UNK-heavy** (UNK 25.3% of the stream under OOV→UNK mapping) and excluding
  UNK contexts cut the word-level gain +2908 → +748, i.e. ~74% of the
  headline was a vocabulary artifact; inventory saturates near M≈128 and
  REVERSES (−472 nats at M=256, −1481 at M=512); a category-stream census
  gave +1458 nats vs +65 null (+556 vs +33 UNK-dropped), 0/5 sign flips.
- **Do**: pre-register, then replicate with **stream construction fixed and
  stated** (OOV mapped vs deleted give different answers), the project's own
  `discover_categories_v2` (slot-level, not the word-level PPMI k-means
  proxy the sandbox used), and harness banding for the negative.
- **Mundane account**: the category-stream gain is a k-means artifact — any
  clustering of a Zipfian stream produces apparent second-order structure
  through frequency effects alone. The permutation null is the control;
  verify it is constructed to kill exactly that.

### T7.7 — Fork decision measurement: the (A) store on the ladder (phase 50)  `[DONE 2026-08-14: claude/t7-7-fork-ladder — the fork is measured, recommendation (A); see ROADMAP row 50]`
- **RESULT (2026-08-14), misses and caveats first.** (1) **The probe's
  seed-0 "K=160 organism above replay" observation DIED held-out** —
  registered as an open question with the probe value disclosed (0.9301 vs
  0.9100), it reversed on ALL FIVE held-out seeds (organism − replay mean
  −0.0215); replay still tops the ladder at 0.9187/0.0834 held-out.
  Selection, caught by the discipline that exists to catch it. (2) **The
  largest fork delta is exactly where the charter said to look**: dFORG
  argmax K=112 held-out +0.0034 [+0.0000, +0.0089] — rp forgets marginally
  more, 6× inside the kill line, and it does NOT replicate at calib-b8
  (−0.0006) or K=160 (−0.0050, rp better). The number to watch, not a
  cost. (3) K=160 calib-b8 (held-out 0.8972) is not a better operating
  point than K=112 (0.9029) — 33h's knee holds. **EVERY PRE-REGISTERED
  PREDICTION HELD, and M1 — the named mundane account — IS the outcome:
  phase 43's ≤0.005 bound TRANSFERS to the continual ladder** (worst
  held-out mean dACC −0.0012 against the 0.005 band; worst dFORG +0.0034
  against the 0.010 band; the pre-named kill line dFORG ≥ +0.020 / dACC ≤
  −0.020 never approached — both K, both decoders, store mode, bootstrap
  CIs). P1 anchors exact (bar 0.872/120/30720 B; K=160 argmax s0 0.9002;
  K=112 floors 76.10 = 2.236× / 70.30 = 2.066× reproducing phase 44 to the
  second decimal; rp 44.88 = 1.319× / 39.07 = 1.148× inside the registered
  store-mode-vs-uncompressed attribution band). P3: the (A) discount is
  K-independent — rp/c64 = 0.590 / 0.581 at K=112/160 vs the derived
  0.58 ± 0.02. P4: replay reseeded held-out 0.9187 in band. K=160 cost for
  the record: rp+narrow 55.08 KB/ACC-pt = 1.619× vs c64+narrow 100.13 =
  2.943×. **RECOMMENDATION DELIVERED (the artifact is ROADMAP row 50):
  take (A) — real_phase + narrow CSR as the deployment persistence layout
  (39.07 KB/ACC-pt = 1.148× vs (B)'s 70.30 = 2.066×), compute width stays
  complex128, `real_phase` stays default-OFF until the owner flips it** —
  with the mandated scoping note: T7.6 (c)/(d) are fork-independent, so
  (A) forfeits only arms (a)+(b), which carry the R ≈ 0.06 readout bar
  and the 9–56-step collapse; the fork text overstates what (A) forfeits,
  and (B)→(A) is a re-compression at any time while (A)→(B) loses only
  what phase 43 measured to be empty. NOVELTY deliberately untouched (no
  claim changes until the owner rules). Harness §19 (4 checks) pins the
  store-mode equivalence bands and the byte-ratio arithmetic; 136/136
  both backends. (Owner audit 2026-08-14: the run artifacts below were
  never committed and the container was ephemeral, so they **no longer
  exist** — the numbers survive in ROADMAP row 50 and
  `phase50_fork_ladder.py` regenerates them. An outcome block may not cite
  evidence that lives only in a working tree.) Runs were `phase50_results_*.txt` +
  `phase50_cells.json` in the working tree.
- **Original target text below.**
**Opened by the repo owner 2026-08-11 as the ruling on T7.1's fork: measure
before choosing.** T7.1 priced the fork and correctly refused to choose it.
The owner declines to choose it blind either — nobody has yet run the (A)
store and the accuracy arm **in the same experiment**, and this project's
own rule is not to bank a number where it was not measured.
- **What changed the stakes, and read this before you scope the work**: the
  gate was re-scoped to the capability axes on 2026-08-11 (ROADMAP decision
  record), which **removes the fork's gate stakes**. Meeting the ≤2×
  fallback no longer decides a release. So this target is NOT "can (A) clear
  the gate" — that question is closed and the branch is conceded. It is:
  **does the (A) layout hold up behaviorally where the organism is actually
  evaluated, and what does it really cost there?** The fork is now an
  architecture decision on measured merit.
- **Do**: run the `real_phase` store (currently a default-OFF
  `CompressionSpec` lever from phase 43) through the **33c ladder** at
  K=112 and K=160, against the identical c64 arm, paired seeds s=0–4 with
  held-out confirmation on s=5–9 and **the prototype bar recomputed on each
  seed's own split** (an unpaired fixed-seed bar is the bug T1.6/T1.8 were
  burned by). Report ACC **and FORG** — phase 43's ≤0.005 behavior bound was
  measured on the cost-frontier protocol, not on the continual ladder, and
  FORG is exactly where a lossy store would be expected to hurt first.
  Report the KB/ACC-pt for both arms with the narrow-CSR variant, each with
  its observation count attached (phase 44: density is not scale-free).
- **Pre-register**, with the named mundane account: (a) whether you expect
  the ≤0.005 bound to transfer to the ladder, and what size of FORG
  regression would make (A) not worth taking at any cost saving; (b) the
  mundane account — the bound transfers fine and the whole exercise merely
  re-derives phase 43 on a second protocol, which is a legitimate and
  useful outcome to record; (c) the honest-negative branch: if (A) costs
  real FORG, the fork resolves to **(B)** and the cost floor stands at
  2.07×, which the re-scoped gate can now absorb.
- **Deliver an owner decision artifact again, but this time with a
  recommendation** — T7.1 was right to withhold one because the fork was
  unmeasured on this axis; you will have measured it.
- **Do NOT** touch phase 48 or start any T7.6 arm. Note for scoping: T7.6's
  arms **(c) dual time constant** and **(d) graph order** are
  **fork-independent** by T7.6's own text, so (A) costs only arms (a)+(b),
  and those carry phase 43's R ≈ 0.06 readout-blindness bar and phase 16's
  measured 9–56 step collapse. State that in the recommendation; the fork
  text as written overstates what (A) forfeits.
- **Done when** (plus protocol section C): `phase50_*.py`, the paired
  ACC/FORG/cost table held out, ROADMAP row, harness section, and a
  recommendation the owner can act on in one reading.

### T7.8 — Land the (A) persistence default (no new phase number)  `[claimed: —]`
**Opened by the repo owner 2026-08-14 as the execution half of the (A)
ruling.** The fork is decided; the code still does (B). This target closes
that gap and nothing else — it is deliberately an **engineering** target
with no new science, and it must not acquire any.
- **Do**: flip `real_phase` from default-OFF to the default persistence
  encoding, together with phase 44's narrow CSR. Compute width stays
  **complex128** — if this target changes an arithmetic path, it has
  exceeded its scope.
- **Why it is not a one-line change**: a persistence default touches every
  E3 round-trip, every byte number in the project, and the schema. Expect
  a schema bump with **backward load for v1/v2/v3**, and expect the byte
  accounting in phases 33g/33h/43/44/50 to need its convention restated
  rather than its numbers changed.
- **Bar for landing** (in addition to SOP section C): E3 round-trip green
  for the new default **and** every prior schema; the phase-50 equivalence
  bands re-measured on the default path, not just the opt-in path; and
  **dFORG at K=112/argmax re-reported explicitly** — the ruling put that
  number on a watch-list precisely so a format change re-checks it.
- **Do NOT** quote 1.148× as "the current default" in any artifact until
  this lands. Until then it is *the ratified layout*.
- **Done when**: default flipped, harness pinning the new default's byte
  arithmetic and equivalence, all five doc surfaces updated to say the code
  now does what the ruling says, and the "ratified layout / not the current
  default" caveat **removed** from NOVELTY, ROADMAP and FABLE_HANDOFF in the
  same commit that makes it false.

### T7.6 — Deep-architecture arms (phase 48)  `[UNBLOCKED 2026-08-14 for arms (c) and (d) ONLY — the owner ruled (A); arms (a) content-dependent phase and (b) phase-aware readout are CLOSED by that ruling]`
**Read before starting.** The (A) ruling closed half of this target. Arms
**(a)** and **(b)** are not "deferred", they are **closed**: the persistence
layout no longer carries the channel they would write to, and both already
had to clear phase 43's R ≈ 0.06 readout-blindness bar and phase 16's
9–56-step collapse. Reopening them is a persistence-format change plus two
measured obstacles, and is an owner decision, not an agent's.
**Arms (c) and (d) are fork-independent and are what this target now is.**
Take **(c) first** — it is the cheapest measured route to sequence memory
and, per T8.1, it is also the first candidate mechanism for Category 8's
Gate 0. **Coordinate with T8.1 before starting: if T8.1 is claimed, (c)
belongs to it and this target is (d) alone.**
Do not start until the owner has resolved T7.1. Under branch (A) most of
this is moot; under branch (B) it is the work that earns the width.
Ordered by cost-to-information: (a) content-dependent phase — a per-item or
per-position phase increment makes the accumulated state a phasor sum
`Σ e^{iφ_j} x_j`, order-sensitive, at zero additional stored bytes;
pre-register against phase 16's measured constraint that superpositions
collapse in 9–56 steps under attractor pull, so LIFETIME is the binding
risk, not accuracy. (b) readout must change too — `overlaps` returns
`(M.conj() @ z)/N` and every consumer takes `np.abs`, so a phase-carrying
dynamics would silently score as a null against a phase-blind readout; any
binding arm is TWO changes. (c) dual time constant — a companion field
state with a slower `g_in` is O(N), ~0.001× of the K×N store, the cheapest
route to sequence memory and independent of the phase fork. (d) graph
order — phase 40 established the boundary is ORDER not size; a dense
third-order tensor at K=112 is ~1.4M entries and dead on arrival, so the
affordable form is low-rank (R·3K) or T7.5's sparse chunk-node inventory,
whose saturation curve caps the rank worth buying. Price both against
T7.5's measured gain before building either.

---

## Category 8 — Agentic language model (opened by the repo owner, 2026-08-14)

**Why this category exists, and read this before claiming anything in it.**
The owner asked what it would take to turn this architecture into a true
agentic language model. This category is the honest answer, decomposed into
**gates with kill rules** rather than a build plan — because three of the
relevant results are already measured, and they are negative.

**The starting position, stated without flattery.** An agentic LM needs
long-range context, compositional binding, word-level generation, goal
conditioning, an action-conditioned world model, scale, and outcome-driven
credit assignment. Of those, the architecture today has **one** in good
shape (the action-conditioned world model and planning — phases 15/30/40),
**one** measured weak (word-level generation — phase 41 measured splitting
*hurting* word-level likelihood), **one** measured empty and now closed by
ruling (phase-superposition binding — N8), and **one** that is a hard
structural limit: the logic layer is a **first-order Markov model over slot
indices**. Phase 40 established the binding constraint is graph **order**,
not size, and that a dense third-order tensor at K=112 is ~1.4M entries and
dead on arrival.

**The number that should govern this category's ambition.** On the phase
34/35 sequence-prediction protocol, a **bigram table** held ~0.85–0.87 flat
at a small fraction of the compute while the organism fell 99% → 28% from 50
to 800 words; two-stage hierarchical routing (phase 36) restored ~85%. A
first-order transition model over discrete slots *is* approximately a bigram
model, and language modelling is exactly where that ceiling binds hardest.
**Any agent working here who finds themselves reporting that the system
"does language" without having cleared Gate 0 has made an error.**

**Owner's standing recommendation, recorded so it is argued with rather
than rediscovered.** The measurements point at a **hybrid**, not a
replacement: every axis this architecture is differentiated on is one
conventional LMs are worst at, and every axis it is weak on is one they are
best at. That complementarity is four measurement campaigns' worth of
evidence, not a hunch. **T8.6 is the recommended path and clears no research
gates at all.** T8.1 is the honest test of whether the pure path exists.
Both are worth having; only one of them is likely to ship.

Phase numbers reserved by this category: **51, 52, 53, 54, 55**.

Standing additions for every target here, on top of the SOP and Category
7's: **pre-register a kill rule, not just a prediction** — a gate you
cannot fail is not a gate; and **name the conventional baseline you are
measured against before you run**, because "better than our previous
version" is not evidence of language capability.

### T8.1 — GATE 0: context depth beyond first order (phase 51)  `[claimed: claude/repo-owner-26nu39, 2026-08-28 — phase 51 reserved. RE-SCOPED on the 2026-08-28 literature pass: see the causal-state reframe below]`

**RE-SCOPE, repo owner, 2026-08-28 — read this before the original text.**
A literature pass changed what this gate should test, in two ways.

**(i) The dual-time-constant arm is published work, not discovery.** The
memory–nonlinearity trade-off is a known and general result in reservoir
computing: nonlinear dynamics degrade stored memory regardless of the form
of the nonlinearity (Inubushi & Yoshimura, *Sci. Rep.* 2017), and the
remedy that paper proposes is a **mixture reservoir carrying both linear
and nonlinear dynamics** — which is what T7.6(c) is. It should be run as
engineering and **must not be claimed as novel**. DeFAI's settling is its
nonlinearity; the memory loss is the theorem's prediction, not a tuning
failure.

**(ii) The sharper question is the equivalence relation, not the depth.**
Computational mechanics (Crutchfield/Shalizi) defines **causal states**:
the equivalence classes of pasts that induce identical conditional
distributions over futures — the coarsest partition retaining full
predictive power, with optimality theorems for prediction, minimality and
uniqueness. That is the minimal sufficient statistic, constructively
defined. Now compare what the organism does:

- **`recruit` partitions on APPEARANCE** — a similarity floor on overlap.
- **N1 partitions on PREDICTED FUTURE** — the Myhill–Nerode criterion, and
  causal states are exactly its generalization from deterministic automata
  to stochastic processes.

**This project already found the right equivalence relation and used it
only as a detector.** N1 is the strongest, most-replicated claim in the
register; it is also the one place the correct criterion is applied. The
hypothesis this target now tests is that the criterion belongs in
`recruit`, not bolted on afterward — and that a first-order graph over
*causal* states is maximally predictive by construction (the ε-machine
optimality theorem), which would mean phase 40's "the constraint is graph
ORDER" was measuring the wrong nodes rather than the wrong order.

**Phase 51 is the cheapest possible test of that**, and it is
protocol-level: no change to `organism.py`. Run the real organism, take its
real recruited partition, build the causal-state partition on the same
observation sequence, and compare **predictive information at matched
cardinality on held-out data**.

**The make-or-break, and the only target in this category worth running
before the others.** Everything downstream assumes it clears.
- **Question**: can the state carry usable context beyond one step, above a
  measured permutation null? Not "is there a mechanism that could" — phase
  43 already established that `perceive` computes a phasor sum and is
  *capable* of order sensitivity. The question is whether any of that
  reaches a consumer.
- **Do (a) FIRST and stop if it clears or clearly fails**: **the dual time
  constant** — a second field state at a slower `g_in`, O(N), ~0.001× of the
  K×N store, **fork-independent** (T7.6(c); coordinate with T7.6, this arm
  belongs to whichever target claims first). Measure effective context depth
  against a permutation null over shuffled histories, and report it as a
  **curve over depth**, not a single number.
- **Then (b) if and only if (a) is ambiguous**: the low-rank or sparse
  higher-order graph. Note before you spend anything: T7.5 already measured
  the chunk route's category level **failing its label-permutation null**,
  so that path is wounded before you start, and phase 40 priced the dense
  form as dead on arrival.
- **CLOSED, do not attempt**: content-dependent phase and the phase-aware
  readout. The 2026-08-14 ruling closed them; reopening is the owner's.
- **KILL RULE, pre-register it verbatim**: if effective context depth does
  not exceed **2–3 tokens above the permutation null** on any surviving
  mechanism, **the architecture is not a language model**, this category
  stops at T8.6, and that is a genuine and publishable result rather than a
  failure. Write down before running what "effective context depth" means
  operationally and what number would falsify a depth claim.
- **Mundane account to name**: any depth signal is carried by unigram
  frequency structure rather than order, and a shuffled-history control at
  matched marginals removes it. Build that control **into** the first run.
- **Done when**: `phase51_*.py`, the depth curve with its null, the kill
  rule adjudicated **out loud**, ROADMAP row, harness section.

### T8.2 — GATE 1: compositional binding (phase 52)  `[BLOCKED on T8.1]`
Do not start until Gate 0 has been adjudicated. Representing "X did Y to Z"
is required by both language and agency, and the architecture's own binding
mechanism (N8) is measured empty, closed by ruling, and carries a
**9–56-step collapse** — lifetime, not accuracy, is the binding risk. Any
proposal here must say in advance what it does about lifetime, and must be
measured against a conventional baseline (a slot-filling or
role-labelling baseline), not against the project's own earlier attempts.

### T8.3 — GATE 2: word-level generation (phase 53)  `[BLOCKED on T8.1 + T8.2]`
The weakest measured axis. **Do not attempt first** — phase 41 already
measured detection-driven splitting improving next-*category* prediction
while making word-level likelihood **worse than not splitting** (−0.081
nats/token), and generation not improving at all on either metric. Starting
here re-derives that negative at full cost. Scored against a conventional
LM baseline at matched compute, with perplexity reported, or it does not
count.

### T8.4 — GATE 3: vocabulary scale (phase 54)  `[claimed: —]`
10³ → 10⁵ vocabulary is two orders of magnitude past anything measured.
Attractor Crowding Collapse (N3) is named and fixed label-free, and
hierarchical routing (phase 36) holds ~85% to 800 items — **whether that
survives 100× is unknown, and it is a research question rather than an
engineering one.** Independent of Gates 0–2, so it can run in parallel;
its result is informative whichever way Gate 0 goes.

### T8.5 — GATE 4: agency proper (phase 55)  `[claimed: —]`
**The architecture's most natural extension, and it is independent of Gates
0–3.** Model-based planning over a learned transition graph is standard
model-based RL, and the pieces exist: `phase15_action_conditioned.py`, plus
phases 30/40's `kstep`/`rollout`/`plan`/`plan_reliable`/`plan_visit`. What
is missing is **goal conditioning** and **outcome-driven credit assignment**
— the agent currently plans but never learns from whether the plan worked.
Add those, measure against a tabular model-based RL baseline on a task with
a real reward signal, and report regret, not just success rate. This is the
target most likely to produce something that works, and it should probably
run **before** T8.2/T8.3 regardless of Gate 0's outcome.

### T8.6 — The hybrid: episodic-memory substrate for an external agent  `[claimed: —]`
**The owner's recommended path, and it clears no research gates.** Expose
the organism as the **persistent episodic memory and world model** for an
agent whose language faculty is a conventional LM: the LM generates and
follows instructions; the field does what it is measured to do — continuous
unlabelled experience, retention without task boundaries or replay buffers,
exact resume, and a queryable transition graph over the agent's own history.
- This is **already what T5.1 names** (the episodic-memory product fork),
  and T5.1 is unblocked as of the 2026-08-11 re-scope. Coordinate rather
  than duplicating: if T5.1 is claimed, this is its engineering half.
- **The honest framing to keep**: this is a memory substrate, not a language
  model. Every claim it makes is a claim about memory, retention and
  persistence — the axes with evidence — and none about generation.
- **Bar**: an integration where the memory's contribution is *measured*
  (ablate it and show the agent degrades), not merely present. An
  unablated integration is a demo, not a result.

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
T6.1 task-free CL ─► T6.6 blocking sweep (phase 49, opened 2026-08-11)
T6.2 retention     ├─► inform any future gate re-test / product claims
T6.3 logic depth   │
T6.4 polysemy→act ─┘   T6.5 re-baseline ─► E4 (T3.1) go/no-go
T4.1 ablation (re-scoped, now unblocked)

Category 7 (architecture economics, owner work order 2026-08-08) — ORDERED:
T7.1 phase-channel audit + storage fork ─► T7.7 (phase 50) ─► T7.6 (blocked)
T7.2 sparse-P default   T7.3 settling depth   T7.4 single-pass audit
T7.5 macro-recruit pre-gates ─► T2.4 (phase 29) if the category arm survives

Owner rulings 2026-08-11:
gate RE-SCOPED to capability axes ─► RELEASE HOLD LIFTED ─► T5.1, T4.3 open
T7.1 fork ─► measure first (T7.7) ─► then choose ─► T7.6 unblocks

Owner ruling 2026-08-14 (fork resolved to (A)):
T7.7 measured ─► RULED (A) ─┬─► T7.8 land the default (code still does (B))
                            ├─► T7.6 arms (c)/(d) only; (a)/(b) CLOSED
                            └─► N8 edited down (deferred, not deleted)

Category 8 (agentic LM, opened 2026-08-14) — GATED, not a build plan:
T8.1 GATE 0 context depth ─► if it fails, the category stops at T8.6
     └─ shares arm (c) with T7.6 — coordinate before claiming
T8.2 GATE 1 binding [blocked on T8.1] ─► T8.3 GATE 2 generation [blocked]
T8.4 GATE 3 scale        ─┐ independent of Gates 0-2, run any time
T8.5 GATE 4 agency       ─┘ closest to working; run before T8.2/T8.3
T8.6 HYBRID substrate ─► the recommended path; clears no research gates
     └─ overlaps T5.1's product fork — coordinate, do not duplicate
```

**Phase-number ledger (check before reserving).** Used through **50**.
**39** USED by T6.2 (`phase39_retention_factors.py`, landed 2026-08-11);
**50** USED by T7.7 (`phase50_fork_ladder.py`, landed 2026-08-14).
**48** reserved for T7.6 — **released 2026-08-14 for arms (c)/(d) only**,
arms (a)/(b) closed by the fork ruling. **49** reserved for T6.6, **31**
for T4.1. **51–55** reserved for Category 8 (T8.1–T8.5, in that order).
**T7.8 needs no phase number** — it is an engineering change.
Next free number for anything else: **56**.

**Open, unclaimed and unblocked right now** (as of 2026-08-14):
**T7.8** — land the (A) persistence default; the ruling's execution half,
and the code still does (B) until it lands.
**T8.1** — Category 8's Gate 0, the make-or-break for the agentic-LM
question. **T8.5** — Gate 4 (agency), the closest thing to working.
**T8.6** — the hybrid substrate, the owner's recommended path.
Also: **T8.4**, **T7.6 arms (c)/(d)** (phase 48), T6.6 (phase 49),
T4.1 (phase 31), T3.1, T3.2, T4.2, and — by the 2026-08-11 re-scope —
T5.1 and T4.3.

**Closed recently**: T6.2 (phase 39, 2026-08-11) — its `p_decay` gap is an
open follow-on, since measuring transition decay as a retention factor
needs a protocol whose metric consumes the transition graph, which the
class-incremental readout does not. T7.7 (phase 50, 2026-08-14) — the fork
was measured and **ruled (A) on 2026-08-14**.

**Blocked**: T2.4 (by T7.5's own pre-gate); T8.2 and T8.3 (by T8.1's gate);
T7.6 arms (a)/(b) (**closed**, not blocked — reopening is the owner's).

**Coordination hazards, check before claiming**: T7.6(c) and T8.1's first
arm are the **same dual-time-constant experiment** — claim one, not both.
T8.6 and T5.1 overlap on the episodic-memory product — coordinate rather
than duplicating. This project has built the same thing twice before.
