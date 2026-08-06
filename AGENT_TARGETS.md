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

Verification state as of 2026-08-06: E1 ALL PASS both backends (45 checks:
31 + T1.8's section 10), `test_fastpath_equivalence.py` green incl. its new
narrowed-store section 7, `test_label_readout.py` green, E3 round-trips
green for schema v2 uncompressed/compressed and v1 backward load. Torch is
NOT installed on that host, so the ladder's torch arms rest on 33c's
committed values. See ROADMAP "Verification log".

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

### T1.9 — Cost-branch follow-up: KB-per-accuracy-point parity (phase 33h)  `[claimed: —]`
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

---

## Category 2 — Scale & real text

### T2.1 — Phase 27: 5M-word scale run  `[DONE 2026-08-06: claude/phase-27-gutenberg-mi-run-afdm3u, PR #34]`
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
- **Outcome (2026-08-06)**: 42-book / 5.22M-raw-word corpus (title-verified
  at fetch), 354-word vocab (MIN_COUNT scaled 150→1500 to hold phase-23/24's
  computational regime). **P1 (sharpens): NOT CONFIRMED** — distinctness
  collapsed to k=2 (z=61, below phase 24's z=65 at k=6), a coarser
  function/content-word split rather than a finer grammatical partition;
  MI itself still climbed through k=11 exactly as phase 24's predictive
  criteria did, so the failure mode matches phase 24's shape at a coarser
  k — open, plausibly corpus-register heterogeneity (verse/philosophical
  prose mixed with narrative fiction), attaches to T2.3. **P2 (polysemy
  holds): CONFIRMED** — 'right' clears its own null under a ~43×-tighter
  bar (margin 0.007 vs 547K's 0.027, but null p99 fell 0.043→0.001);
  205/354 words (58%) clear their own null, up from 114/395 at 547K. One
  bug found/fixed en route: `coverage_map`'s one-shot assignment matmul
  tried to allocate ~32 GiB at this token count — the same
  one-shot-broadcast blowup phase 36 hit in `_silhouette_real` — fixed by
  chunking (values unaffected). Harness 45/45 both backends throughout.
  Merge-review follow-ups from the PR #34 status note, folded in: (1)
  `coverage_map`'s member loop now reuses the already-built `train_arr`
  instead of re-allocating `np.array(train_seq)` per slot; (2) the E3
  stage-A checkpoint now stores a corpus/vocab fingerprint (token count +
  a hash of the sorted vocab) and `load_state` is followed by a check that
  refuses (and deletes) a stale checkpoint rather than silently training
  on the wrong corpus; (3) the corpus-missing fetch instructions now print
  the same title-verification the live fetch used, as a comment above the
  curl block, so a re-fetch without this session's fetch script still gets
  the ebook-id guard; (4) `sharpens_k`'s window reconciled: argued from
  the frozen pre-registration text alone (not the measured k=2, which
  fails either way) — "stable or gracefully larger" than k=6 reads most
  naturally as up to but not including the sweep ceiling, so the window is
  widened to 4–11 (was 4–10, which wrongly failed a hypothetical k=11).
  Harness re-verified 45/45 both backends after these fixes (doc/script
  changes only, no organism.py/fastpath.py edits, no equivalence re-check
  needed). Full row: ROADMAP 27.

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
  (T1.4 ran 2026-08-05: bar NOT met — 0.712 vs 0.872, ROADMAP row 33c;
  the hold stands pending the owner's decision.)
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
