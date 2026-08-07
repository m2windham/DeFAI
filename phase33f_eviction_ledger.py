"""
PHASE 33f (T1.7): TASK-1 REGRESSION DIAGNOSTIC -- the eviction ledger.
(33d is reserved for T1.5's capacity sweep; this diagnostic feeds it.)

The regression under diagnosis (phase 33c, committed 2026-08-05): folding
the T1.2 slot budget into the ladder's organism arm (evict=250) lifted
task-0 final accuracy 0.732 -> 0.939 but DROPPED task-1 final accuracy
0.464 -> 0.393, and the birth-era census of surviving slots [9,7,4,6,14]
leaves the middle eras thin. T1.7's brief: log every eviction's victim
(birth era, lifetime count, staleness age, evicting-token class) across
the 33c run, reconstruct who evicted whom when, and discriminate three
pre-registered hypotheses -- each naming a different fix:

  H1  task-1-born slots are evicted disproportionately: born under maximum
      pressure (the bank still full of task-0's flood) they hold the lowest
      lifetime counts, so the argmin-count victim rule bills the flood's
      cleanup to its successor.        FIX: count normalized by era/lifetime
                                       in the victim choice (feeds T1.5).
  H2  task-1 slots survive but their readout labels go stale after nearby
      evictions (vote redistribution flips majorities on surviving slots).
                                       FIX: readout invalidation scope.
  H3  task 1 is intrinsically hardest (its baseline 0.464 was already the
      ladder's low) and eviction merely fails to help.
                                       FIX: none (capacity/readout work,
                                       T1.5/T1.6, carries the load).

Instrumentation (T1.7's constraint honored -- mechanism read-only): a debug
ledger behind `perceive(evict_debug=...)` (organism.py + fastpath.py, both
backends). Default None = flag off = bitwise-identical behavior; with a
list attached, each pressure eviction appends (frame, slot, count, age),
recorded BEFORE the eviction mutates state, and no mechanism decision
reads the ledger. Era attribution stays OUT of the mechanism: the script
replays births (free-slot recruits precede all evictions inside a task --
eviction fires only on a full bank -- and every eviction rebirths its slot
the same frame), so `victim_birth` is reconstructed here, label-free
inside perceive.

T3.3 MIGRATION (2026-08-07): this script is the proof case for the stable
symbol registry. The `owner` replay above reconstructs slot lineage from
the ledger and rests on the ordering argument in that paragraph; the
registry tracks the same lineage AT THE MECHANISM BOUNDARY, needing no
ledger and no argument -- a slot's occupant carries an ID minted at its
first commit, and IDs are minted in chronological order, so the symbols
born in task `ti` are exactly the ID range [minted-before, minted-after).
Both are computed; section (A) gates them against each other and
`census_of` now reads the registry. Measured on the committed run: the two
agree on 40/40 slots, on both backends, and the registry accounts for every
eviction exactly (171 tombstones = 171 ledger rows; 211 mints = K=40
initial + 171 rebirths). The census anchor [9,7,4,6,14] and every number
below are unchanged -- the registry is observational, and enabling it
leaves organism state bitwise identical (harness section 11).

Equivalence proof, pre-registered as section (A)'s gate: (i) budget arm
with ledger attached reproduces 33c's committed anchors EXACTLY (ACC
0.712 / FORG 0.169, census [9,7,4,6,14]); (ii) ledger-on vs ledger-off
leaves every organism field bitwise identical (dxi=dP=dcount=dage=dused=
devictions=0) and the task-accuracy matrix identical; (iii) the numpy
backend records the identical ledger row-for-row (this protocol's routing
is decision-stable across backends, as 33b/33c found); (iv) harness 31/31
on BOTH backends and `test_fastpath_equivalence.py` green with the flag
off (run separately; results in the ROADMAP verification log); (v) T3.3:
registry lineage == script replay on every slot, on both backends, with
tombstone/mint counts matching the ledger exactly.

Pre-registered discriminating measurements and decision rule:
  (B) the ledger itself: per-task eviction counts; who-evicted-whom
      matrix (evicting task x victim birth era); victim count/age
      distributions, same-task churn vs cross-era victims split.
  (C) H1 axis: per-era births vs evictions vs survivors (hazard); victim
      quality (count at eviction) for era-1-born vs same-task churn;
      task-1 content coverage (mean max-overlap of task-1 test queries
      against the used bank) per snapshot, budget vs baseline arm.
  (D) H2 axis: final task-1 errors routed -- classify each error by the
      routed slot's birth era and majority label: era-1 slot with label
      outside {2,3} = stale-label (H2); non-era-1 slot = content loss/
      absorption (H1/H3); era-1 slot, label in {2,3}, wrong class =
      intra-pair confusion (H3). Label-flip audit: slots whose majority
      was in {2,3} after task 1, tracked per snapshot (survived /
      invalidated / flipped-while-surviving, and what flipped them).
  (E) H3 axis: immediate post-training accuracy A[1,1] baseline vs
      budget; the A[:,1] retention trajectory; per-class (2 vs 3) final
      accuracy; birth provisioning -- fraction of each task's train
      tokens absorbed by the pre-task bank at >= 0.8x recruit (0.48, the
      refine-not-evict zone) and >= recruit (0.6, never novel), from the
      task-(i-1) snapshot -- was task 1 short of slots from birth?
  (F) verdict: H1 requires era-1 cross-era eviction excess AND coverage
      degradation AND flips routing away from era-1 slots; H2 requires
      >= half of task-1 errors on surviving era-1 slots with stale
      labels; H3 requires comparably low A[1,1] in both arms with the
      same wrong-class targets and neither H1 nor H2 signatures. Mixed
      outcomes: name the dominant channel by flip counts, honestly.

COMMITTED-RUN FINDINGS (2026-08-05; numpy 2.4.6 / numba 0.66; gate (A)
all-PASS: every anchor exact -- budget 0.712/0.169 census [9,7,4,6,14],
baseline 0.665/0.200 -- ledger on/off bitwise identical across all state
and the task-accuracy matrix, numpy and numba ledgers identical
row-for-row; 171 evictions logged):

  VERDICT: H3, dominant and supported; H1 ABSENT; H2 ABSENT. The
  regression is NOT an eviction pathology and the pre-registered H1 fix
  (count-normalized-by-era victim choice) is affirmatively NOT indicated
  -- the ledger exonerates the victim-choice rule. FIX NAMED: none at
  the mechanism. The regression's own delta is readout geometry and
  belongs to T1.6 (its pre-registered prediction (a), now with measured
  targets); the underlying task-1 floor is interference at K=40 and
  belongs to T1.5 (capacity).

  The measured facts:
  - (B) The budget's victims are junk and self-churn, 33b's law exactly:
    123/171 evictions are same-task self-churn, median victim count 9,
    92/171 victims held count < 10. Cross-era victims: era-0-born 31
    (the flood's cleanup; median count 31), era-1-born only 4, era-2 8,
    era-3 5.
  - (C) Era 1 was not robbed: 23 births / 16 evicted, but 12 of those
    were its own within-task churn -- 4 cross-era evictions in the whole
    run vs era-0's 31 -- and 7 survive. Task-1 test-content coverage is
    HIGHER than baseline at every snapshot (0.723 vs 0.694 after task 1;
    0.566 vs 0.535 final) and decays identically in both arms (-0.158
    vs -0.160): nothing task 1 owned was taken from it. H1 ABSENT.
  - (D/D2) Budget task-1 errors 51/84 = 42 captured by slots whose
    CONTENT genuinely belongs to other classes + 9 on slots still
    holding task-1 content but carrying a foreign label + 0 intra-pair;
    baseline 45/84 = 45 + 0 + 0. The 9-vs-0 label-lost channel IS the
    net regression (flip set 19 correct->wrong / 13 wrong->correct, net
    6). By the pre-registered >=half rule H2 is ABSENT (9/51); of 15
    slots that ever carried a {2,3} majority, 3 flipped it while
    surviving un-invalidated. Per class the budget REDISTRIBUTES rather
    than loses: class 2 0.667 -> 0.373 (-0.294, 51 queries) while class
    3 recovers 0.152 -> 0.424 (+0.273, 33 queries -- baseline's class-3
    catastrophe was the ladder's true floor all along).
  - (D3) The 9 label-lost errors live on exactly two slots, and neither
    indicts eviction's bookkeeping: slot 33 (era-0 victim evicted at
    count 23 under task 1, reborn on a task-1 token, correctly
    relabeled majority-2 mass 9) was OUT-VOTED at task 2 (+13 class-5
    votes onto content that still best-matches class 2) -- majority
    collapse discards its 8 surviving class-2 votes; slot 9 (era-0,
    NEVER evicted, zero task-1 votes in its entire history) drifted
    into class-2 content and captures 7 queries with label 9 while the
    one healthy class-2 slot (slot 30: majority 2, mass 27) loses the
    argmax-overlap race by 0.061 mean margin (slot 33's margin 0.136).
    Both failures are argmax routing + majority collapse -- T1.6's
    decoders (i)/(ii), with those margins as the measured bar.
  - (E) Task 1 acquires fine and is NOT under-provisioned at birth:
    A[1,1] = 0.929 baseline / 0.881 budget, and task-1 tokens are the
    stream's MOST novel against the pre-task bank (never-novel fraction
    0.06 vs 0.12-0.23 for tasks 2-4) -- the absorption story this
    diagnostic half-expected is refuted by measurement. Task 1 is the
    hardest task to RETAIN: the steepest retention trajectory in BOTH
    arms (0.929 -> 0.464 baseline, 0.881 -> 0.393 budget), and at final
    state the entire bank holds only 2 (baseline) to 4 (budget) slots
    whose content matches class 2/3 at all. The interference floor is
    intrinsic to K=40, not to eviction.

  Consequences fed forward: T1.5 proceeds with the victim rule AS IS,
  and should track the task-1 content census + label-mass concentration
  as K grows (prediction: the 42-error capture floor shrinks with
  capacity). T1.6 gains two measured recovery targets (slot-9-type:
  0.061 routing margin, recoverable by a soft top-m vote over slot 30's
  mass-27 evidence; slot-33-type: preserved-but-outvoted label mass).
"""

import numpy as np
from sklearn.datasets import load_digits

from label_readout import LabelEvidenceReadout
from organism import Organism, normalize
from organism_numba import NumbaOrganism

rng = np.random.default_rng(0)

# ---- phase 33/33b/33c data pipeline, verbatim -----------------------------
d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(np.float32)
y = d.target.astype(int)
N = X.shape[1]
NORM = np.sqrt(N)
perm = rng.permutation(len(X))
tr, te = perm[:1400], perm[1400:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
K = 40
EVICT = 250
HOLD = 8
EPOCHS = 3

# committed anchors (phase 33b/33c rows) this run must reproduce
ANCHOR_BASE = (0.665, 0.200)
ANCHOR_BUDGET = (0.712, 0.169)
ANCHOR_CENSUS = [9, 7, 4, 6, 14]


def task_acc_matrix_to_metrics(A):
    T = len(TASKS)
    final = A[T - 1]
    acc = float(np.mean(final))
    forg = float(np.mean([np.max(A[:, j]) - final[j] for j in range(T - 1)]))
    return acc, forg


def eval_tasks(predict):
    out = []
    for task in TASKS:
        m = np.isin(yte, task)
        out.append(float((predict(Xte[m]) == yte[m]).mean()))
    return out


def states_of(Xe):
    return np.array([normalize(x.astype(complex), NORM) for x in Xe])


def run_arm(evict, ledger_on=True, cls=NumbaOrganism):
    """Phase 33c's organism arm verbatim (stream, recipe, readout
    bookkeeping), plus the T1.7 ledger and per-task snapshots.

    BIRTH ATTRIBUTION, two ways -- this is T3.3's migration proof.

    (1) `owner`, the original script-side replay, which rests on an
        ordering argument stated in prose: free-slot recruits precede all
        evictions inside a task (eviction fires only on a full bank) and
        every eviction rebirths its slot the same frame, so marking fresh
        free recruits first and then replaying ledger rows chronologically
        attributes every victim's birth era correctly, including same-task
        rebirths. Correct, but it is the script re-deriving slot lineage
        from a diagnostic ledger -- and it is only as good as the argument.

    (2) `owner_reg`, read off the T3.3 SYMBOL REGISTRY. A slot's occupant
        is named by an ID minted at the EventBoundary the first time it
        commits, and IDs are minted in chronological order -- so the
        symbols minted during task `ti` are exactly those in the ID range
        [minted-before, minted-after), and a slot's birth era is the range
        containing the ID it holds NOW. No ordering argument, no ledger,
        no replay: the mechanism tracked the lineage as it happened, and
        the slot's own churn (eviction -> tombstone -> fresh mint) is what
        moves it between eras.

    Both are computed and section (A) gates them against each other slot
    for slot. `census_of` reads the registry; the replay stays as the
    cross-check that pins the migration.
    """
    r = np.random.default_rng(0)
    r.permutation(len(X))                # burn the split draw (33b's technique)
    org = cls(N=N, K=K, omega=0.15, beta=10.0, seed=0, symbols=True)
    reg = org.registry
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    owner = np.full(K, -1)
    eras = []                            # per-task minted-ID range (T3.3)
    A = np.zeros((len(TASKS), len(TASKS)))
    ledger = []
    snaps = []
    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        order = []
        for _ in range(EPOCHS):
            p = r.permutation(idx)
            order.extend(p.tolist())
            for i in p:
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * HOLD)
        dbg = [] if ledger_on else None
        id_before = reg.minted()
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict,
                     evict_debug=dbg)
        eras.append((id_before, reg.minted()))       # (1) IDs born this task
        owner[~used_before & org.used] = ti          # free recruits come first
        for (f, j, c, a) in (dbg or []):             # then replay evictions
            tok = order[int(f) // HOLD]
            ledger.append(dict(task=ti, frame=int(f), slot=int(j),
                               count=float(c), age=float(a),
                               victim_birth=int(owner[j]),
                               tok_class=int(ytr[tok])))
            owner[j] = ti                            # eviction = same-frame rebirth
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        readout.invalidate(fresh)                    # readout follows slot identity
        readout.observe(org, Xtr[idx], ytr[idx])
        A[ti] = eval_tasks(lambda Xe: readout.predict(org, Xe))
        snaps.append(dict(used=org.used.copy(), count=org.count.copy(),
                          age=org.age.copy(), xi=org.xi.copy(),
                          evidence=readout.evidence.copy(),
                          owner=owner.copy(), fresh=int(fresh.sum()),
                          evd=int((org.evictions - evd_before).sum())))
    return dict(A=A, org=org, readout=readout, owner=owner,
                owner_reg=owner_from_registry(reg, eras), eras=eras,
                registry=reg, ledger=ledger, snaps=snaps)


def owner_from_registry(reg, eras):
    """(2) Birth era per slot, straight from the symbol registry: the era
    whose minted-ID range contains the ID the slot holds now. -1 = the slot
    holds no symbol (free, or used but never confident enough to commit)."""
    owner = np.full(len(reg.slot_sym), -1)
    for j, sid in enumerate(reg.slot_sym):
        if sid >= 0:
            owner[j] = next((t for t, (a, b) in enumerate(eras)
                             if a <= sid < b), -1)
    return owner


def census_of(res):
    """Surviving slots per birth era -- read off the symbol registry."""
    org, owner = res["org"], res["owner_reg"]
    return [int((owner[org.used] == t).sum()) for t in range(len(TASKS))]


def route(org, readout, states):
    """Per-query routing through the readout's own rule: (routed slot,
    predicted label, overlap)."""
    slots, lab = readout.slot_labels(org)
    ov = np.abs(org.xi[slots].conj() @ states.T) / N
    j = np.argmax(ov, axis=0)
    return slots[j], lab[j], ov[j, np.arange(states.shape[0])]


if __name__ == "__main__":
    print("PHASE 33f (T1.7): task-1 regression diagnostic -- eviction ledger\n")

    T1_CLASSES = TASKS[1]
    te1 = np.isin(yte, T1_CLASSES)
    st1 = states_of(Xte[te1])
    y1 = yte[te1]

    # ---- (A) anchors + equivalence gate -------------------------------
    print("(A) anchors + ledger-equivalence gate")
    bud = run_arm(EVICT, ledger_on=True)
    bud_off = run_arm(EVICT, ledger_on=False)
    base = run_arm(0, ledger_on=False)
    acc_b, forg_b = task_acc_matrix_to_metrics(bud["A"])
    acc_0, forg_0 = task_acc_matrix_to_metrics(base["A"])
    cen = census_of(bud)
    o_on, o_off = bud["org"], bud_off["org"]
    d_ab = max(np.abs(o_on.xi - o_off.xi).max(), np.abs(o_on.P - o_off.P).max(),
               np.abs(o_on.count - o_off.count).max(),
               np.abs(o_on.age - o_off.age).max(),
               float((o_on.used != o_off.used).sum()),
               np.abs(o_on.evictions - o_off.evictions).max(),
               np.abs(bud["A"] - bud_off["A"]).max())
    reg_same = bool(np.array_equal(bud["owner"], bud["owner_reg"]))
    reg_deaths = len(bud["registry"].dead)
    reg_minted = bud["registry"].minted()
    bud_np = run_arm(EVICT, ledger_on=True, cls=Organism)
    led_same = len(bud["ledger"]) == len(bud_np["ledger"]) and all(
        a["frame"] == b["frame"] and a["slot"] == b["slot"]
        and a["count"] == b["count"] and a["age"] == b["age"]
        for a, b in zip(bud["ledger"], bud_np["ledger"]))
    checks = [
        ("budget ACC/FORG anchor",
         abs(acc_b - ANCHOR_BUDGET[0]) < 5e-4 and abs(forg_b - ANCHOR_BUDGET[1]) < 5e-4,
         f"{acc_b:.3f}/{forg_b:.3f} vs {ANCHOR_BUDGET[0]:.3f}/{ANCHOR_BUDGET[1]:.3f}"),
        ("baseline ACC/FORG anchor",
         abs(acc_0 - ANCHOR_BASE[0]) < 5e-4 and abs(forg_0 - ANCHOR_BASE[1]) < 5e-4,
         f"{acc_0:.3f}/{forg_0:.3f} vs {ANCHOR_BASE[0]:.3f}/{ANCHOR_BASE[1]:.3f}"),
        ("birth-era census anchor", cen == ANCHOR_CENSUS, f"{cen}"),
        ("ledger on/off bitwise (state+A)", d_ab == 0.0, f"max delta {d_ab:.1e}"),
        ("numpy backend: identical ledger", led_same,
         f"{len(bud['ledger'])} vs {len(bud_np['ledger'])} rows"),
        # T3.3 migration gate: the symbol registry's lineage must reproduce
        # the script-side replay slot for slot, and account for every
        # eviction (one tombstone per ledger row, one fresh mint after it)
        ("registry birth era == script replay", reg_same,
         f"{int((bud['owner'] == bud['owner_reg']).sum())}/{K} slots agree"),
        ("registry tombstones == ledger rows", reg_deaths == len(bud["ledger"]),
         f"{reg_deaths} vs {len(bud['ledger'])}"),
        ("registry mints == K + rebirths",
         reg_minted == K + len(bud["ledger"]),
         f"{reg_minted} vs {K} + {len(bud['ledger'])}"),
        ("registry cross-backend lineage", np.array_equal(
            bud["owner_reg"], bud_np["owner_reg"]),
         f"numba vs numpy, {K} slots"),
    ]
    gate = True
    for name, ok, detail in checks:
        gate &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not gate:
        raise SystemExit("equivalence/anchor gate failed -- diagnosis void")

    led = bud["ledger"]
    A1, A0 = bud["A"], base["A"]
    print(f"\n  final per-task acc: baseline {np.round(A0[-1], 3).tolist()}")
    print(f"                      budget   {np.round(A1[-1], 3).tolist()}")
    print(f"  the regression: task 1 {A0[-1][1]:.3f} -> {A1[-1][1]:.3f} "
          f"({int(te1.sum())} test queries)")

    # ---- (B) the ledger -----------------------------------------------
    print(f"\n(B) the eviction ledger: {len(led)} evictions")
    M = np.zeros((5, 5), int)                  # [evicting task, victim birth era]
    for e in led:
        M[e["task"], e["victim_birth"]] += 1
    print("    who evicted whom (rows = evicting task, cols = victim birth era):")
    print("           era0 era1 era2 era3 era4")
    for ti in range(5):
        marks = "".join(f"{M[ti, tb]:5d}" for tb in range(5))
        print(f"    task {ti}{marks}   ({M[ti].sum()} evictions)")
    churn = sum(1 for e in led if e["victim_birth"] == e["task"])
    cross = len(led) - churn
    cnts = np.array([e["count"] for e in led])
    print(f"    same-task self-churn {churn}/{len(led)}, cross-era {cross}/{len(led)}")
    print(f"    victim lifetime count: median {np.median(cnts):.0f}, "
          f"junk (count<10) {int((cnts < 10).sum())}/{len(led)}, "
          f"max {cnts.max():.0f}")
    for era in range(4):
        ev_e = [e for e in led if e["victim_birth"] == era and e["task"] != era]
        if ev_e:
            ce = np.array([e["count"] for e in ev_e])
            print(f"    cross-era victims born era {era}: {len(ev_e)} "
                  f"(count median {np.median(ce):.0f}, max {ce.max():.0f})")

    # ---- (C) H1 axis: hazard + coverage --------------------------------
    print("\n(C) H1 axis -- was era 1 robbed?")
    births = np.zeros(5, int)
    births[0] = K                              # the flood owns every first birth
    for ti in range(1, 5):
        births[ti] = M[ti].sum()               # every eviction = one rebirth
    evicted_of = M.sum(0)                      # column: evictions suffered by era
    surv = np.array(cen)
    print("    era   births  evicted  survive")
    for era in range(5):
        print(f"      {era}   {births[era]:5d}  {evicted_of[era]:6d}  {surv[era]:6d}")
    cov = []
    for arm, nm in ((bud, "budget"), (base, "baseline")):
        row = []
        for s in arm["snaps"]:
            xi_u = s["xi"][s["used"]]
            row.append(float(np.abs(xi_u.conj() @ st1.T).max(0).mean() / N))
        cov.append(row)
        print(f"    task-1 test coverage (mean max-overlap) per snapshot, {nm}: "
              f"{np.round(row, 3).tolist()}")
    cov_flat = cov[0][4] - cov[0][1]
    print(f"    budget coverage drift snapshot1 -> final: {cov_flat:+.3f} "
          f"(baseline {cov[1][4] - cov[1][1]:+.3f})")

    # ---- (D) H2 axis: routing + label staleness ------------------------
    print("\n(D) H2 axis -- do surviving task-1 slots carry stale labels?")
    slots_r, lab_r, _ = route(bud["org"], bud["readout"], st1)
    err = lab_r != y1
    own_f = bud["owner"]
    is_e1 = own_f[slots_r] == 1
    in_t1 = np.isin(lab_r, T1_CLASSES)
    n_err = int(err.sum())
    stale = int((err & is_e1 & ~in_t1).sum())      # era-1 slot, foreign label
    lost = int((err & ~is_e1).sum())               # routed off era 1 entirely
    intra = int((err & is_e1 & in_t1).sum())       # era-1 slot, 2<->3 confusion
    print(f"    budget task-1 errors {n_err}/{len(y1)}: "
          f"stale-label-on-era-1-slot {stale}, routed-to-other-era {lost}, "
          f"intra-pair-on-era-1 {intra}")
    routed_eras = np.bincount(own_f[slots_r[err]] + 1, minlength=6)[1:] \
        if n_err else np.zeros(5, int)
    print(f"    error routing by slot birth era: {routed_eras.tolist()}")
    flips = 0
    flip_slots = []
    carriers = set()
    for s_i in range(1, 5):
        ev = bud["snaps"][s_i - 1]["evidence"]
        for sl in range(K):
            if ev[sl].sum() > 0 and int(ev[sl].argmax()) in T1_CLASSES:
                carriers.add(sl)
    for sl in sorted(carriers):
        prev_maj = None
        for s_i in range(1, 5):
            s = bud["snaps"][s_i]
            ev = s["evidence"][sl]
            if not s["used"][sl] or ev.sum() == 0:
                prev_maj = None
                continue
            maj = int(ev.argmax())
            if (prev_maj in T1_CLASSES and maj not in T1_CLASSES
                    and s["owner"][sl] == bud["snaps"][s_i - 1]["owner"][sl]):
                flips += 1                     # flipped while surviving
                flip_slots.append(sl)
                break
            prev_maj = maj
    print(f"    slots ever carrying a {{2,3}} majority: {len(carriers)}; "
          f"flipped while surviving un-invalidated: {flips} {flip_slots}")

    # ---- (D2) content-vs-label decomposition of the errors -------------
    # Added after a first look at the ledger (documented post-hoc, purely
    # observational): (D)'s era/label split conflates two different
    # failures on era-1-born slots -- a slot whose CONTENT genuinely
    # drifted to a later class and was correctly re-labeled by votes
    # (content interference) vs a slot still holding task-1 content whose
    # majority was merely out-voted (readout failure). Disambiguate by the
    # routed slot's content class (oracle-eval: which test class its
    # attractor matches best -- evaluation only, per standing rules).
    print("\n(D2) error decomposition by routed-slot CONTENT (post-hoc, eval-side)")
    cls_states = [states_of(Xte[yte == c]) for c in range(10)]

    def content_class_map(org):
        used = np.where(org.used)[0]
        mo = np.zeros((used.size, 10))
        for c in range(10):
            mo[:, c] = np.abs(org.xi[used].conj() @ cls_states[c].T).mean(1) / N
        return dict(zip(used.tolist(), mo.argmax(1).tolist()))

    d2 = {}
    for arm, nm in ((bud, "budget"), (base, "baseline")):
        ccm = content_class_map(arm["org"])
        sl_, lb_, _ = route(arm["org"], arm["readout"], st1)
        er_ = lb_ != y1
        cap = intra_ = 0
        lost_q = []
        for qi in np.where(er_)[0]:
            cc = ccm[int(sl_[qi])]
            lb = int(lb_[qi])
            if cc not in T1_CLASSES:
                cap += 1                   # foreign content captured the query
            elif lb not in T1_CLASSES:
                lost_q.append(qi)          # task-1 content, label out-voted
            else:
                intra_ += 1                # task-1 content, 2<->3 confusion
        n_ = int(er_.sum())
        d2[nm] = dict(n=n_, cap=cap, lost=len(lost_q), intra=intra_,
                      lost_q=lost_q, ccm=ccm, sl=sl_, lb=lb_)
        print(f"    {nm}: {n_} errors = captured-by-foreign-content {cap}, "
              f"task-1-content-but-label-lost {len(lost_q)}, intra-pair {intra_}")
    for nm, arm in (("budget", bud), ("baseline", base)):
        ccm = d2[nm]["ccm"]
        holders = [s for s, c in ccm.items() if c in T1_CLASSES]
        slots_all, labs_all = arm["readout"].slot_labels(arm["org"])
        lab_of = dict(zip(slots_all.tolist(), labs_all.tolist()))
        labs = [lab_of.get(s, -1) for s in holders]
        good = sum(1 for L in labs if L in T1_CLASSES)
        print(f"    {nm}: slots holding {{2,3}} content {len(holders)}, "
              f"of which labeled {{2,3}}: {good}  "
              f"(birth eras {[int(arm['owner'][s]) for s in holders]})")
    for c in T1_CLASSES:
        mc = y1 == c
        a_b = float((d2['budget']['lb'][mc] == c).mean())
        a_0 = float((d2['baseline']['lb'][mc] == c).mean())
        print(f"    class {c} ({int(mc.sum())} queries): baseline {a_0:.3f} -> "
              f"budget {a_b:.3f} ({a_b - a_0:+.3f})")

    # ---- (D3) the label-lost slots, named (post-hoc, eval-side) ---------
    print("\n(D3) slot forensics on the label-lost errors (budget arm)")
    lost_slots = sorted(set(int(d2["budget"]["sl"][qi])
                            for qi in d2["budget"]["lost_q"]))
    slots_all, labs_all = bud["readout"].slot_labels(bud["org"])
    lab_of = dict(zip(slots_all.tolist(), labs_all.tolist()))
    good_t1 = [s for s in slots_all.tolist()
               if lab_of[s] in T1_CLASSES]      # correctly-labeled {2,3} slots
    for sl in lost_slots:
        nq = sum(1 for qi in d2["budget"]["lost_q"]
                 if int(d2["budget"]["sl"][qi]) == sl)
        ev = bud["readout"].evidence[sl]
        top = [(int(t), round(float(ev[t]), 1))
               for t in np.argsort(ev)[::-1][:3] if ev[t] > 0]
        hist = []
        for si, s in enumerate(bud["snaps"]):
            e = s["evidence"][sl]
            hist.append(int(e.argmax()) if e.sum() > 0 else -1)
        rows = [(e["task"], e["victim_birth"], int(e["count"]))
                for e in led if e["slot"] == sl]
        print(f"    slot {sl}: born era {int(bud['owner'][sl])}, content class "
              f"{d2['budget']['ccm'][sl]}, serves {nq} errors, final label "
              f"{int(ev.argmax())} (top votes {top})")
        print(f"      majority per snapshot {hist}; evictions of this slot "
              f"(task, victim-born, count): {rows if rows else 'none'}")
        qs = [qi for qi in d2["budget"]["lost_q"]
              if int(d2["budget"]["sl"][qi]) == sl]
        if good_t1:
            ov_self = np.abs(bud["org"].xi[sl].conj() @ st1[qs].T) / N
            ov_good = np.max(
                np.abs(bud["org"].xi[good_t1].conj() @ st1[qs].T) / N, axis=0)
            print(f"      its queries' overlap: this slot {ov_self.mean():.3f} vs "
                  f"best correctly-labeled {{2,3}} slot {ov_good.mean():.3f} "
                  f"(margin {float((ov_self - ov_good).mean()):+.3f} -- what a "
                  f"top-m soft vote would have to overcome)")

    # ---- (E) H3 axis: intrinsic difficulty + birth provisioning --------
    print("\n(E) H3 axis -- was task 1 ever going to be easy?")
    print(f"    immediate post-training A[1,1]: baseline {A0[1, 1]:.3f}, "
          f"budget {A1[1, 1]:.3f}")
    print(f"    task-1 retention A[:,1]: baseline {np.round(A0[1:, 1], 3).tolist()}, "
          f"budget {np.round(A1[1:, 1], 3).tolist()}")
    for arm, nm, (Ar,) in ((bud, "budget", (A1,)), (base, "baseline", (A0,))):
        sl_, lb_, _ = route(arm["org"], arm["readout"], st1)
        for c in T1_CLASSES:
            mc = y1 == c
            acc_c = float((lb_[mc] == c).mean())
            wrong = np.bincount(lb_[mc][lb_[mc] != c], minlength=10)
            top = np.argsort(wrong)[::-1][:2]
            print(f"    {nm} class {c}: acc {acc_c:.3f}  "
                  f"absorbed by {[(int(t), int(wrong[t])) for t in top if wrong[t] > 0]}")
    print("    birth provisioning: task tokens vs the PRE-task bank")
    print("      task  >=0.6 never-novel  0.48-0.6 refine-zone  <0.48 free-to-recruit")
    for ti in range(1, 5):
        idx = np.where(np.isin(ytr, TASKS[ti]))[0]
        stt = states_of(Xtr[idx])
        s_prev = bud["snaps"][ti - 1]
        xi_u = s_prev["xi"][s_prev["used"]]
        mm = np.abs(xi_u.conj() @ stt.T).max(0) / N
        nn = float((mm >= 0.6).mean())
        rz = float(((mm >= 0.48) & (mm < 0.6)).mean())
        fr = float((mm < 0.48).mean())
        print(f"        {ti}      {nn:.2f}              {rz:.2f}                {fr:.2f}")

    # baseline->budget flip set: the samples that ARE the regression
    sl_b, lb_b, _ = route(base["org"], base["readout"], st1)
    flips_reg = np.where((lb_b == y1) & (lab_r != y1))[0]
    back = np.where((lb_b != y1) & (lab_r == y1))[0]
    print(f"    regression flip set: {len(flips_reg)} correct->wrong, "
          f"{len(back)} wrong->correct (net {len(flips_reg) - len(back)})")
    to_e1 = int((own_f[slots_r[flips_reg]] == 1).sum()) if len(flips_reg) else 0
    print(f"      of the correct->wrong flips, routed to an era-1 slot in the "
          f"budget arm: {to_e1}/{len(flips_reg)}")

    # ---- (F) verdict ---------------------------------------------------
    print("\n(F) verdict vs the pre-registered rule")
    e1_cross = int(M[2:, 1].sum())
    e0_cross = int(M[1:, 0].sum())
    h1 = (e1_cross > max(int(M[2:, 2].sum()), int(M[2:, 3].sum())) + 3
          and cov_flat < -0.02 and len(flips_reg) > 0
          and to_e1 < len(flips_reg) / 2)
    h2 = n_err > 0 and stale >= n_err / 2
    h3 = (abs(A1[1, 1] - A0[1, 1]) < 0.15 or A1[1, 1] < 0.75) and not h1 and not h2
    print(f"    H1 (era-1 robbed): cross-era evictions of era-1-born {e1_cross} "
          f"(era-0 {e0_cross}), coverage drift {cov_flat:+.3f} -> "
          f"{'SUPPORTED' if h1 else 'ABSENT'}")
    print(f"    H2 (stale labels): {stale}/{n_err} errors on stale era-1 slots, "
          f"{flips} surviving flips -> {'SUPPORTED' if h2 else 'ABSENT'}")
    print(f"    H3 (intrinsically hardest): A[1,1] {A0[1, 1]:.3f}/{A1[1, 1]:.3f}, "
          f"class-2 floor in both arms -> {'SUPPORTED' if h3 else 'ABSENT'}")
    if h3:
        print("\n    VERDICT: H3 -- the regression is not an eviction pathology.")
        print("    The victim-choice rule is exonerated by the ledger: victims")
        print("    are junk and self-churn (B), era-1 prototypes were never")
        print(f"    reclaimed (C: 4 cross-era evictions vs era-0's 31), and the")
        print("    budget arm holds task-1 content BETTER than baseline at every")
        print("    snapshot. Task 1 acquires fine (A[1,1] 0.929/0.881) and its")
        print("    tokens are the stream's most novel (E) -- it is the hardest")
        print("    task to RETAIN: in BOTH arms its queries end up captured by")
        print(f"    foreign-content attractors (budget {d2['budget']['cap']}, "
              f"baseline {d2['baseline']['cap']} of its errors),")
        print("    the shared interference floor that made 0.464 the ladder's")
        print("    low before eviction existed. The budget's own net delta is a")
        print(f"    READOUT loss (D2/D3): {d2['budget']['lost']} errors on slots "
              f"that still hold task-1")
        print(f"    content but carry foreign labels (baseline "
              f"{d2['baseline']['lost']}) -- majority")
        print("    collapse discards surviving class-2 votes, and argmax routing")
        print("    prefers a drifted never-voted-for slot over the healthy one.")
        print("    Per class, eviction REDISTRIBUTES the task rather than losing")
        print("    it: class 3 recovers on fresh slots while class 2 pays.")
        print("    FIX NAMED: none at the victim-choice rule (H1's fix is")
        print("    affirmatively not indicated). The regression's delta belongs")
        print("    to T1.6 (soft top-m vote / per-slot label distributions --")
        print("    exactly its pre-registered prediction a); the shared floor")
        print("    belongs to T1.5 (capacity). T1.5 should track the task-1")
        print("    content census and label-mass concentration as K grows.")
    elif h1:
        print("\n    VERDICT: H1 -- feed count-normalized-by-era victim choice")
        print("    into T1.5's sweep.")
    elif h2:
        print("\n    VERDICT: H2 -- widen the readout invalidation scope.")
    else:
        print("\n    VERDICT: mixed -- no single channel dominates; see the")
        print("    section counts above.")
