"""
PHASE 33d (T1.5): COST-MATCHED CAPACITY SWEEP -- is the remaining gate gap
(0.712 vs the 0.872 supervised prototype bar, phase 33c) a capacity
artifact of the protocol's pinned K=40?

The ladder pins K=40 by protocol (4 slots/class), while the prototype bar
GROWS its store on demand and ends at 120 prototypes. The two arms also
spend very different bytes per unit of store: the prototype bar's 0.872
costs 30.7KB (120 float32 unit vectors), the organism's 0.712 costs 54.4KB
(complex128 field memories + the K x K transition matrix + bookkeeping).
Phase 12 measured that storage itself scales; phase 34 measured that
SELECTION over a crowded store does not. This phase asks which regime the
gate gap lives in, and prices the answer -- the cost-per-accuracy curve IS
the measurement the gate's second branch ("cost-effective at equal
accuracy") asks for.

Protocol: phase 33c's stream and scorers VERBATIM -- this script imports
the 33c module and uses its data split, task schedule, metrics, and
prototype arm as the same objects the committed gate re-test ran. The only
degree of freedom is K (and the evict window's engagement, see below).
No organism.py changes; protocol-level only.

Arms:
  prototypes        phase 33c verbatim, re-run in-process (the 0.872 /
                    30.7KB / 120-proto bar this sweep is scored against)
  organism K-sweep  K in {40, 60, 80, 120, 160} at evict=250 +
                    LabelEvidenceReadout (33c's ORGANISM+B recipe), plus a
                    COST-MATCHED point: the largest K whose measured state
                    bytes fit inside the prototype bar's budget.
  evict=0 twins     every K also runs evict=0 -- the staleness re-check
                    T1.5 names (the E=250 window was tuned at K=40; its
                    live-slot revisit interval grows with K). Mechanism
                    fact fixing what the twins can show: eviction fires
                    only under `used.all()` recruitment pressure, so until
                    the bank fills, evict=250 differs from evict=0 by
                    bookkeeping only and the twin arms must match EXACTLY.

Reproduction anchors (numpy-deterministic, expected exact): prototypes
0.872 with 120 protos; K=40 evict=0 = 0.665/0.200 incl. the 40/40 flood
with zero later recruits; K=40 evict=250 = 0.712/0.169 with census
[9,7,4,6,14] (phase 33b/33c committed values).

Pre-registered predictions (written before the committed run):
  (a) ACC rises monotonically with K, and the flood becomes irrelevant:
      by some K the first task no longer fills the bank, later tasks
      recruit fresh slots with NO eviction pressure, and the twin arms
      converge to identical numbers (the mechanism fact above makes this
      exact, not approximate, once evictions hit zero).
  (b) Honest negative to watch: if ACC plateaus below ~0.87 by K=120, the
      gap is NOT capacity -- it is readout geometry (argmax-overlap +
      majority-label vs per-class prototype geometry), and T1.6 carries
      the load. Report the plateau either way.
  (c) At K=120 the state-bytes comparison may favor prototypes (240KB vs
      30.7KB is 7.8x): report the cost-per-accuracy curve either way --
      that curve is the gate's second-branch measurement, whichever way
      it points.
  (d) Window-transfer risk at intermediate K (bank still fills, but
      slower): with more slots per class the live-slot revisit interval
      grows beyond phase 33b's measured ~80 frames at K=40, so E=250 may
      mark live slots stale and evict them. Measured by the twins: if an
      evict=250 arm trails its evict=0 twin by > 0.01 ACC while evictions
      fired, a post-hoc diagnostic window sweep (E in {100, 250, 500,
      1000, 2000}) runs at the worst such K -- labeled post-hoc, not a
      primary result.

COMMITTED-RUN FINDINGS (2026-08-05; numpy 2.4.6 / numba 0.66 / sklearn
1.9; harness 31/31 both backends + test_label_readout green before the
run; no library file touched -- protocol-level only, as T1.5 requires).

Anchors: every one EXACT (prototypes 0.872/0.021 with 120 protos; K=40
evict=0 0.665/0.200 incl. the 40/40 flood; K=40 evict=250 0.712/0.169,
census [9,7,4,6,14]).

THE HEADLINE: the gate gap IS mostly a capacity artifact. On the budget
arm ACC rises monotonically in K -- 0.712 / 0.735 / 0.837 / 0.854 /
0.900 at K = 40/60/80/120/160 -- and CROSSES the 0.872 bar at K=160
(0.900, FORG 0.033), every task represented in the final bank (census
[43,34,52,17,14]), with the K=40 task-1 regression (0.393, T1.7's
subject) healing on its own as pressure relaxes (0.726 / 0.881 / 0.905 /
0.964 at 60/80/120/160). Prediction (a) monotone rise: HELD. Prediction
(b) plateau: NOT triggered -- 0.854 at K=120 sits below ~0.87 but the
curve is still rising and crosses at 160. Capacity, not readout
geometry, is the dominant gap component; T1.6 stays open as an
independent lever (byte-efficiency and the residual K=40 gap), but it
does not carry the load prediction (b) would have handed it.

Prediction (a)'s second half MISSED, informatively: the flood does
dissolve (task 0 recruits only 53-54 slots at K >= 60, vs 40/40 at
K=40), but the bank still fills by tasks 1-3 at every swept K (final
used = K everywhere) and eviction stays LOAD-BEARING, not moot: the
evict=250 arm beats its evict=0 twin at every K except 120 (0.854 vs
0.859, -0.005, inside the noise band) -- by +0.124 at K=24, +0.047 at
40, +0.238 at 60, +0.076 at 80, +0.046 at 160 -- while eviction counts
fall 191/171/126/97/50/37 and never reach zero. The evict=0 ladder is
NOT monotone (0.665 at K=40 but 0.497 at K=60: task 0 takes 53 slots,
task 1 the last 7, census [53,7,0,0,0], and everything later learns by
drift); the budget restores monotonicity. Prediction (d): contingency
NOT triggered -- no K>40 budget arm trails its twin by > 0.01, so
E=250's K=40 characterization transfers across the sweep unchanged.

Prediction (c) held -- the cost branch is NOT met, and the curve says
so without ambiguity. The bar delivers 0.872 at 30.7KB (35.2
KB/ACC-point); the organism's cost-matched point (K=24, 29.6KB)
delivers 0.644/0.285; its bar-crossing point (K=160) spends 371.2KB =
12.1x the bar's bytes (412 KB/ACC-point); and 33c's committed replay
(0.913 at ~89.6KB incl. its raw sample buffer) still tops the ladder
on both axes. Matched slot count is the sharp localization: at K=120
vs the bar's 120 prototypes, accuracy nearly matches (0.854 vs 0.872,
-0.018) at 7.8x the bytes -- the byte gap is REPRESENTATION WIDTH
(complex128 field states + the dense K x K transition graph), not
memory count. Marginal cost is nonuniform: K 60->80 is the cheap
stretch (+2.36 ACC-pt/MB), 80->120 the dearest (+0.16).

Verdict vs the gate: raw-accuracy branch MET at K=160 as a PROTOCOL
VARIATION (the pinned-K=40 ladder itself still stands at 0.712 NOT
SOTA, and replay still tops everything); cost-effectiveness branch NOT
met at any swept K. What this buys the owner's decision: the remaining
K=40 gap is now measured to be capacity, purchasable at a ~12x byte
premium, with FORG collapsing as K grows (0.169 -> 0.033); the
cost-per-accuracy curve printed below IS the gate's second-branch
measurement, and it points at representation width (complex128 xi +
dense P) as the place byte-efficiency would have to come from.
"""

import time

import numpy as np

# Phase 33c's protocol objects, verbatim -- importing executes its data
# pipeline (rng(0) split, task schedule); torch arms live behind its
# __main__ guard and are not touched here (33c's committed run keeps them
# on record: replay 0.913/0.105 at ~38.4KB params + 51.2KB raw buffer,
# joint* 0.974, SGD 0.323, EWC 0.322).
import phase33c_gate_retest as p33c
from phase33c_gate_retest import (
    ANCHOR_BUDGET_ACC, ANCHOR_BUDGET_FORG, ANCHOR_ORG_ACC, ANCHOR_ORG_FORG,
    ANCHOR_PROTO_ACC, EVICT, N, NORM, TASKS, X, Xtr, ytr,
    run_prototypes, task_acc_matrix_to_metrics,
)
from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_numba import NumbaOrganism

K_SWEEP = [40, 60, 80, 120, 160]
ANCHOR_CENSUS = [9, 7, 4, 6, 14]


def org_state_bytes(K):
    """The organism arm's measured state accounting at evict>0 (33c's
    formula): xi (complex128 K x N) + P (K x K float64) + count + age."""
    return 16 * N * K + 8 * K * K + 16 * K


def run_organism_K(K, evict, hold=8, epochs=3):
    """Phase 33c's run_organism VERBATIM with K parameterized (T1.5's one
    protocol degree of freedom). The stream is identical across K: the
    burn draw and per-task shuffles consume the local rng the same way."""
    r = np.random.default_rng(0)
    r.permutation(len(X))                    # burn the split draw (33b's technique)
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=0)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    owner = np.full(K, -1)
    A = np.zeros((len(TASKS), len(TASKS)))
    rows = []
    t0 = time.time()
    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in r.permutation(idx):
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict)
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        owner[fresh] = ti
        readout.invalidate(fresh)            # readout follows slot identity
        readout.observe(org, Xtr[idx], ytr[idx])
        A[ti] = p33c.eval_tasks(lambda Xe: readout.predict(org, Xe))
        rows.append(dict(task=ti, used=int(org.used.sum()),
                         evictions=int((org.evictions - evd_before).sum()),
                         fresh=int(fresh.sum())))
    census = [int((owner[org.used] == t).sum()) for t in range(len(TASKS))]
    state_bytes = (org.xi.nbytes + org.P.nbytes + org.count.nbytes
                   + (org.age.nbytes if evict > 0 else 0))
    return A, time.time() - t0, state_bytes, int(org.used.sum()), rows, census


if __name__ == "__main__":
    print("PHASE 33d (T1.5): cost-matched capacity sweep on the 33c protocol\n")

    # -- the bar --------------------------------------------------------
    A, t, proto_bytes, n_proto = run_prototypes()
    proto_acc, proto_forg = task_acc_matrix_to_metrics(A)
    print(f"  bar: prototypes ACC {proto_acc:.3f}  FORG {proto_forg:.3f}  "
          f"state {proto_bytes / 1e3:5.1f}KB  ({n_proto} protos)")
    ok = abs(proto_acc - ANCHOR_PROTO_ACC) < 5e-4 and n_proto == 120
    print(f"       anchor 0.872 / 120 protos: {'REPRODUCED' if ok else 'MISMATCH'}\n")

    # cost-matched point: largest K whose measured state fits the bar's bytes
    K_COST = max(k for k in range(1, 512) if org_state_bytes(k) <= proto_bytes)
    print(f"  cost-matched K (state <= {proto_bytes / 1e3:.1f}KB): K={K_COST} "
          f"({org_state_bytes(K_COST) / 1e3:.1f}KB)\n")

    # -- the sweep ------------------------------------------------------
    ks = sorted(set([K_COST] + K_SWEEP))
    results = {}
    for K in ks:
        for evict in (0, EVICT):
            A, t, bytes_, nm, rows, census = run_organism_K(K, evict)
            acc, forg = task_acc_matrix_to_metrics(A)
            results[(K, evict)] = dict(A=A, acc=acc, forg=forg, t=t,
                                       bytes=bytes_, nm=nm, rows=rows,
                                       census=census)
            tag = f"evict={evict}" if evict else "evict=0 "
            ev_total = sum(r['evictions'] for r in rows)
            fresh_later = [r['fresh'] for r in rows[1:]]
            print(f"  K={K:>3} {tag:<9} ACC {acc:.3f}  FORG {forg:.3f}  "
                  f"state {bytes_ / 1e3:6.1f}KB  used@t0 {rows[0]['used']:>3}  "
                  f"final {nm:>3}  evictions {ev_total:>3}  "
                  f"fresh-later {fresh_later}  census {census}")
        b, z = results[(K, EVICT)], results[(K, 0)]
        if b['acc'] == z['acc'] and b['forg'] == z['forg']:
            print(f"        twins EXACT (no eviction pressure at K={K})")
    print()

    # -- reproduction anchors ------------------------------------------
    print("reproduction anchors (numpy-deterministic, expected exact):")
    r0, rB = results[(40, 0)], results[(40, EVICT)]
    checks = [
        ("K=40 evict=0 ACC", r0['acc'], ANCHOR_ORG_ACC),
        ("K=40 evict=0 FORG", r0['forg'], ANCHOR_ORG_FORG),
        (f"K=40 evict={EVICT} ACC", rB['acc'], ANCHOR_BUDGET_ACC),
        (f"K=40 evict={EVICT} FORG", rB['forg'], ANCHOR_BUDGET_FORG),
    ]
    for label, got, want in checks:
        ok = abs(got - want) < 5e-4
        print(f"  {label:<22} {got:.3f} vs committed {want:.3f}  "
              f"{'REPRODUCED' if ok else 'MISMATCH'}")
    flood_ok = (r0['rows'][0]['used'] == 40
                and sum(r['fresh'] for r in r0['rows'][1:]) == 0)
    print(f"  K=40 flood (40/40, zero later recruits): "
          f"{'REPRODUCED' if flood_ok else 'MISMATCH'}")
    print(f"  K=40 evict={EVICT} census {rB['census']} vs {ANCHOR_CENSUS}: "
          f"{'REPRODUCED' if rB['census'] == ANCHOR_CENSUS else 'MISMATCH'}")

    # -- per-task finals (where does added capacity go?) ---------------
    print("\nper-task final accuracy, budget arms (task cols 0-4):")
    for K in ks:
        print(f"  K={K:>3}  {np.round(results[(K, EVICT)]['A'][-1], 3).tolist()}")

    # -- cost-per-accuracy curve ---------------------------------------
    print("\ncost-per-accuracy curve (the gate's second-branch measurement):")
    print(f"  {'arm':<22} {'ACC':>6} {'FORG':>6} {'stateKB':>8} {'KB/ACCpt':>9}")
    print(f"  {'prototypes (bar)':<22} {proto_acc:6.3f} {proto_forg:6.3f} "
          f"{proto_bytes / 1e3:8.1f} {proto_bytes / 1e3 / proto_acc:9.1f}")
    for K in ks:
        r = results[(K, EVICT)]
        label = f"organism K={K}" + ("  (cost-matched)" if K == K_COST else "")
        print(f"  {label:<22} {r['acc']:6.3f} {r['forg']:6.3f} "
              f"{r['bytes'] / 1e3:8.1f} {r['bytes'] / 1e3 / r['acc']:9.1f}")
    print("  (33c committed, not re-run: replay 0.913/0.105 at ~89.6KB "
          "incl. 200 raw samples; joint* 0.974)")
    print("\n  marginal cost along the sweep (delta-ACC per delta-KB, evict=250):")
    for k0, k1 in zip(K_SWEEP[:-1], K_SWEEP[1:]):
        a0, a1 = results[(k0, EVICT)], results[(k1, EVICT)]
        dacc = a1['acc'] - a0['acc']
        dkb = (a1['bytes'] - a0['bytes']) / 1e3
        print(f"    K {k0:>3} -> {k1:>3}: {dacc:+.3f} ACC for {dkb:+6.1f}KB "
              f"({1000 * dacc / dkb:+.2f} ACCpt/MB)")

    # -- pre-registered contingency (d): window transfer ----------------
    worst = None
    for K in ks:
        if K == 40:
            continue                          # 33b already characterized K=40
        b, z = results[(K, EVICT)], results[(K, 0)]
        fired = sum(r['evictions'] for r in b['rows']) > 0
        if fired and z['acc'] - b['acc'] > 0.01:
            if worst is None or z['acc'] - b['acc'] > worst[1]:
                worst = (K, z['acc'] - b['acc'])
    if worst is not None:
        K_bad = worst[0]
        print(f"\npost-hoc diagnostic (pre-registered contingency d): at K={K_bad} "
              f"evict={EVICT} trails its evict=0 twin by {worst[1]:.3f} with "
              f"evictions fired -- window sweep:")
        for E in (100, 250, 500, 1000, 2000):
            A, t, bytes_, nm, rows, census = run_organism_K(K_bad, E)
            acc, forg = task_acc_matrix_to_metrics(A)
            ev = sum(r['evictions'] for r in rows)
            print(f"    E={E:>4}: ACC {acc:.3f}  FORG {forg:.3f}  "
                  f"evictions {ev:>3}  census {census}")
    else:
        print("\nwindow-transfer contingency (pre-registered d): NOT triggered --")
        print("  no K>40 arm with evictions fired trails its evict=0 twin by >0.01.")

    # -- verdict --------------------------------------------------------
    accs = [results[(K, EVICT)]['acc'] for K in K_SWEEP]
    best_i = int(np.argmax(accs))
    best_K, best_acc = K_SWEEP[best_i], accs[best_i]
    best = results[(best_K, EVICT)]
    acc120 = results[(120, EVICT)]['acc']
    r120 = results[(120, EVICT)]
    rc = results[(K_COST, EVICT)]
    print("\nverdict (against the owner's gate, both branches):")
    mono = all(a1 >= a0 - 1e-9 for a0, a1 in zip(accs[:-1], accs[1:]))
    print(f"  prediction (a) monotone rise: {'HELD' if mono else 'FAILED -- ACC is not monotone in K'}"
          f"  (sweep: {[round(a, 3) for a in accs]})")
    # prediction (b) is a conjunction: a PLATEAU, sitting below ~0.87, by
    # K=120. Test both halves separately -- a curve merely passing through
    # a sub-0.87 value at K=120 while still rising is NOT the negative.
    still_rising = best_acc - acc120 > 0.01 and best_K > 120
    if acc120 < 0.87 and not still_rising:
        print(f"  prediction (b) plateau: ACC {acc120:.3f} at K=120 < ~0.87 and flat --")
        print("    the gap is NOT capacity; readout geometry (T1.6) carries the load.")
    elif acc120 < 0.87:
        print(f"  prediction (b) plateau: NOT a plateau -- ACC {acc120:.3f} at K=120 is")
        print(f"    below ~0.87 but the curve is still rising ({best_acc:.3f} at K={best_K}):")
        print("    capacity, not readout geometry, is the dominant gap component.")
    else:
        print(f"  prediction (b): no plateau -- ACC {acc120:.3f} at K=120.")
    print(f"  raw-accuracy branch: best organism {best_acc:.3f} at K={best_K} vs bar "
          f"{proto_acc:.3f}")
    if best_acc >= proto_acc:
        print(f"    -- MEETS the bar at K={best_K} (a protocol variation: the ladder")
        print(f"       pins K=40), spending {best['bytes'] / proto_bytes:.1f}x the bar's state bytes")
        print(f"       ({best['bytes'] / 1e3:.1f}KB vs {proto_bytes / 1e3:.1f}KB). "
              f"33c's committed replay (0.913) still tops it.")
    else:
        print("    -- NOT SOTA at any swept K.")
    print(f"  matched slot count (K=120 vs {n_proto} protos): ACC {acc120:.3f} vs "
          f"{proto_acc:.3f} ({acc120 - proto_acc:+.3f})")
    print(f"    at {r120['bytes'] / proto_bytes:.1f}x the bytes -- at equal memory COUNT the")
    print("    accuracy nearly matches; the byte gap is representation width")
    print("    (complex128 field states + the K x K graph), not memory count.")
    print(f"  cost branch (cost-effective at equal accuracy): NOT met. At matched")
    print(f"    bytes (K={K_COST}, {rc['bytes'] / 1e3:.1f}KB <= bar's {proto_bytes / 1e3:.1f}KB) ACC is "
          f"{rc['acc']:.3f} vs {proto_acc:.3f};")
    print(f"    at matched-or-better accuracy (K={best_K}) the organism spends")
    print(f"    {best['bytes'] / proto_bytes:.1f}x the bar's bytes ({best['bytes'] / 1e3 / best_acc:.0f} vs "
          f"{proto_bytes / 1e3 / proto_acc:.0f} KB/ACC-point). The bar")
    print("    delivers more accuracy per byte at every swept K.")
    print("  OWNER DECISION POINT unchanged from 33c: the measured facts are above;")
    print("    this sweep localizes the remaining gap (capacity, buyable at a byte")
    print("    premium -- not readout) -- the decision on the hold is the owner's.")
