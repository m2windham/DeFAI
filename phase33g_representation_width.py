"""
PHASE 33g (T1.8): REPRESENTATION-WIDTH BYTE REDUCTION -- can the organism
buy phase 33d's accuracy at a fraction of phase 33d's bytes?

Where this comes from. Phase 33d (T1.5) priced the gate's cost branch and
localized the failure exactly: at MATCHED memory count (K=120 slots vs the
supervised prototype bar's 120 prototypes) accuracy nearly matches (0.854
vs 0.872) while the organism spends 7.8x the bytes; at K=160 it CROSSES
the bar (0.900) at 12.1x the bytes (371.2KB vs 30.7KB). Its closing
sentence names the target: the byte gap is REPRESENTATION WIDTH --
complex128 field memories plus a dense K x K transition matrix -- not
memory count. This phase attacks the width.

Scope, stated up front. This is inference/storage engineering
(`organism_compress.py`): compression maps a finished state to a narrower
encoding of THE SAME state. No mechanism, no learning rule, and no
re-attribution of occurrences to slots is touched -- phase 9's closed
negative (post-hoc re-sorting inherits the online mixing it meant to undo)
is about a different operation and is neither invoked nor re-opened here.
Compute width stays complex128 on both backends
(`Organism.perceive`'s dtype guard, pinned by `test_fastpath_equivalence.py`
section 7); the 33c/33d anchors must reproduce EXACTLY in uncompressed mode,
and the run checks that they do.

The three levers, split by whether they can lose anything:
  LOSSLESS   sparse (CSR) P above a count floor of 0; float32 count/age
             (integer-valued and far below 2**24). The state decompresses
             bit-for-bit -- these move bytes and CANNOT move accuracy, and
             the run verifies that as an equality, not a tolerance.
  LOSSY      complex64 xi (halves the dominant term); a count floor > 0 on
             P (drops rare transitions); low-rank xi factorization.

Compression granularity: the store is compressed at each TASK BOUNDARY and
carried forward quantized -- the persistence path an episodic-memory
product actually takes between sessions (save narrow, load, keep learning),
so any loss compounds across the five tasks exactly as it would in
deployment. The `eval-only` arm compresses just before scoring instead, and
the two together separate storage loss from compounded loss.

Protocol: phase 33c's stream and scorers VERBATIM via phase 33d's runner
(same data split, task schedule, metrics, prototype bar). K and the
compression spec are the only degrees of freedom.

Pre-registered predictions (T1.8's, restated verbatim from AGENT_TARGETS
before the committed run):
  (a) complex64 xi halves the dominant term with ACC drift within harness
      tolerance bands (attractor capture is ~0.99 overlaps; 7 significant
      digits is plenty).
  (b) P sparsification above a count floor cuts the K x K term by >5x at
      K=160 (measure the count distribution first).
  (c) Honest negative: if compressed-state ACC at K=160 drops below the
      0.872 bar, report the byte-accuracy frontier and where it crosses.
  (d) Target: K=160-equivalent accuracy inside ~2x the prototype bar's
      bytes -- landing near replay's ~90KB footprint would make the cost
      branch arguable.

Integrity note on lever (b), fixed before the run: the 33c readout scores
label evidence over xi overlaps and NEVER queries the transition graph, so
on this protocol P pruning is accuracy-free by construction -- however hard
it prunes. Reporting "floor>5 costs nothing" on that basis would be a
measurement artifact, so every pruned arm also reports GRAPH FIDELITY
(mass kept, next-hop argmax agreement, mean total-variation drift of the
row-normalized transitions). The logic layer's value is measured by phase
30 (kstep/rollout/plan), not by this benchmark; the fidelity columns are
what a phase-30 consumer would have to accept.

COMMITTED-RUN FINDINGS (2026-08-06; numpy 2.4.6 / numba 0.66 / sklearn 1.9;
harness 31+14 checks both backends + equivalence (incl. the new narrowed-
store section 7) + E3 v1/v2 round-trips green before the run).

Anchors: EVERY 33d budget-arm point reproduced EXACTLY in uncompressed mode
(ACC 0.712 / 0.735 / 0.837 / 0.854 / 0.900 at K = 40/60/80/120/160, FORG
0.169 at K=40 and 0.033 at K=160).

THE HEADLINE: 3.85x fewer bytes at ZERO accuracy cost. At K=160 the store
goes 371.2KB -> 96.4KB while every single task-accuracy entry is unchanged
(max |d task-accuracy| = 0.0000 at every swept K, and the paired reseed
supplement puts the drift at exactly 0.000 on all five seeds). 33d's 12.1x
byte premium over the prototype bar becomes 3.14x, and the organism now
holds 0.900 inside 1.08x replay's ~89.6KB footprint.

Prediction (a) HELD, and more strongly than written: complex64 xi was
predicted to cost drift "within tolerance bands"; it measured drift of
IDENTICALLY ZERO. The quantization moves the stored digits (~1e-7
relative) and never once moves a routing decision, a recruitment, an
eviction, or a label vote -- attractor capture at ~0.99 overlaps has
margins orders of magnitude above float32 epsilon, exactly as the
prediction's reasoning said. Store-mode (quantize at every task boundary,
carry it forward, let loss compound) and eval-only mode agree to 0.000 at
every K, so there is nothing to compound.

Prediction (b) HELD at 15.5x, and the lever turned out to be LOSSLESS: P
is 6.1% dense at K=160 (1571/25600 nonzeros), so CSR at a count floor of 0
stores exactly the nonzeros -- 204.8KB -> 13.2KB with the matrix
reconstructing bit-for-bit. No count floor is needed to beat the >5x bar,
and the run verifies the lossless levers leave the whole run bit-identical
(not merely within tolerance) at every K.

Prediction (c)'s negative did NOT fire: compressed ACC at K=160 is 0.900,
still above the 0.872 bar. The frontier table is printed regardless.

Prediction (d) NOT MET, and the reason is the phase's honest negative.
Lever (iii), low-rank xi factorization, is the only one that reaches
inside ~2x the bar's bytes, and it does not survive reseeding. On seed 0
the rank sweep looks like a win (rank-20: 0.887 at 49.4KB = 1.61x the bar,
which would have MET (d)); the pre-registered paired supplement over seeds
0-4 shows every low-rank arm swinging far beyond its own margin (rank-20
spans -0.052 to +0.033; rank-16 -0.071 to +0.019; even rank-40 -0.047 to
+0.011), while the c64+lossless arm is identically 0.000 on every seed.
Picking rank-20 was best-of-grid selection on one seed -- precisely the
bias T1.6 measured and the reason its supplement exists. LEVER (iii) IS A
MEASURED NEGATIVE at this scale; levers (i) and (ii) carry the whole 3.85x.
(A related seed-0 artifact, recorded so nobody chases it: rank-48 scored
0.910, ABOVE the full-width run. It is the same noise -- and it costs more
bytes than dense complex64 xi anyway, since K=160 slots span an N=64 field
and factors only pay below r ~ 45.)

Cost branch, updated: NOT met, but the price moved a lot. 33d's
cost-matched point (largest K inside the bar's 30.7KB) was K=24 at ACC
0.644; under compression it is K=48 at 29.1KB and ACC 0.728 (+0.084),
still -0.144 short of the bar. Per-point cost falls from 412.4 to 107.1
KB/ACC-point at K=160 and from 76.4 to 33.3 at K=40 -- the K=40 compressed
arm (23.7KB) is the first organism configuration to sit UNDER the bar's
byte budget AND under its cost-per-accuracy-point (33.3 vs 35.2), though
at 0.712 accuracy it is not a bar-clearing arm. The bar still delivers
more accuracy per byte at every K that clears it.

Integrity note that survived contact with the data: P pruning above a
count floor is accuracy-free on this protocol at every floor tested --
because the label readout never queries the transition graph. The graph
fidelity columns price it properly: floor>5 at K=160 keeps 81% of the
mass but empties 31% of the live rows of their next hops. Nobody should
read those arms as free; they are free FOR A CONSUMER THAT DOES NOT USE
THE GRAPH, and phase 30's consumers do.
"""

import copy

import numpy as np

import phase33e_readout_geometry as p33e
from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_compress import CompressionSpec, compress, store_bytes
from organism_numba import NumbaOrganism
from phase33c_gate_retest import (
    EVICT, N, NORM, TASKS, X, run_prototypes, task_acc_matrix_to_metrics,
)

# 33e's seeded split -- build_split(0) is the committed 33c protocol
# byte-for-byte (33e pins that), so seed 0 reproduces the anchors while
# seeds 1..4 give the paired robustness supplement T1.6's precedent asks for
build_split = p33e.build_split

K_SWEEP = [40, 60, 80, 120, 160]

# phase 33d's COMMITTED budget-arm ladder (evict=250) -- the anchors this
# phase's uncompressed arm must reproduce EXACTLY. Only values actually
# recorded in the 33d row are pinned: the full ACC ladder, and FORG at the
# two K points 33d printed it for (K=40's 0.169, K=160's 0.033). The other
# FORGs are reported by this run, not asserted against.
ANCHOR_33D_ACC = {40: 0.712, 60: 0.735, 80: 0.837, 120: 0.854, 160: 0.900}
ANCHOR_33D_FORG = {40: 0.169, 160: 0.033}
REPLAY_BYTES = 89.6e3          # 33c committed: 38.4KB params + 51.2KB buffer

# compression arms carried through the run (store mode)
SPEC_C64 = CompressionSpec(xi_dtype=np.complex64, meta_dtype=np.float64)
SPEC_FULL = CompressionSpec(xi_dtype=np.complex64, meta_dtype=np.float32)
SPEC_LOSSLESS = CompressionSpec(xi_dtype=np.complex128, meta_dtype=np.float32)


def graph_fidelity(P0, P1):
    """What a pruned transition graph costs a phase-30-style consumer:
    (mass kept, next-hop argmax agreement, mean TV drift, fraction of live
    rows pruned to nothing) over the rows the uncompressed graph had mass
    in. The agreement column is computed only over rows that SURVIVE with
    mass -- a count floor removes the smallest entries first, so the row
    argmax is the last thing to go and agreement stays ~1 by construction;
    the emptied-row fraction is where a pruned graph actually loses its
    next hops, and TV drift is where it loses its distribution."""
    live = P0.sum(1) > 0
    if not live.any():
        return 1.0, 1.0, 0.0, 0.0
    mass = float(P1.sum() / P0.sum()) if P0.sum() > 0 else 1.0
    R0 = P0[live] / P0[live].sum(1, keepdims=True)
    s1 = P1[live].sum(1, keepdims=True)
    R1 = np.divide(P1[live], s1, out=np.zeros_like(P1[live]), where=s1 > 0)
    agree = float((R0.argmax(1) == R1.argmax(1))[s1[:, 0] > 0].mean()
                  if (s1 > 0).any() else 0.0)
    # rows pruned to nothing count as maximal drift -- they lost everything
    tv = 0.5 * np.abs(R0 - R1).sum(1)
    tv[s1[:, 0] == 0] = 1.0
    emptied = float((s1[:, 0] == 0).mean())
    return mass, agree, float(tv.mean()), emptied


def run_arm(K, evict=EVICT, spec=None, mode="store", seed=0, hold=8, epochs=3):
    """Phase 33d's `run_organism_K` with a compression hook.

    mode='store'     compress the store at every task boundary and carry it
                     forward (the deployment persistence path; loss compounds)
    mode='evalonly'  score a compressed COPY, keep learning at full width
                     (isolates storage loss from compounded loss)
    spec=None        uncompressed -- must reproduce 33d exactly
    seed             0 = the committed protocol; >0 reseeds split, stream
                     shuffles and organism together (robustness supplement)
    """
    Xtr, ytr, Xte, yte = build_split(seed)
    r = np.random.default_rng(seed)
    r.permutation(len(X))                    # burn the split draw (33b's technique)
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=seed)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    owner = np.full(K, -1)
    A = np.zeros((len(TASKS), len(TASKS)))
    P_raw = None
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
        if spec is None:
            state = None
            scored = org
        elif mode == "store":                # quantize and carry forward
            P_raw = org.P.copy()
            state = compress(org, spec=spec)
            state.apply(org)
            scored = org
        else:                                # eval-only: score a compressed copy
            P_raw = org.P.copy()
            state = compress(org, spec=spec)
            scored = state.apply(copy.deepcopy(org))
        A[ti] = p33e.eval_tasks(lambda Xe: readout.predict(scored, Xe),
                                Xte, yte)
    census = [int((owner[org.used] == t).sum()) for t in range(len(TASKS))]
    if spec is None:
        bytes_ = store_bytes(org)
        fid = (1.0, 1.0, 0.0, 0.0)
    else:
        bytes_ = compress(org, spec=spec).nbytes
        fid = graph_fidelity(P_raw, org.P if mode == "store" else
                             compress(org, spec=spec).P_full())
    acc, forg = task_acc_matrix_to_metrics(A)
    return dict(A=A, acc=acc, forg=forg, bytes=bytes_, census=census,
                used=int(org.used.sum()), fid=fid, org=org)


def fmt(r, label, raw_bytes, width=30):
    return (f"  {label:<{width}} ACC {r['acc']:.3f}  FORG {r['forg']:.3f}  "
            f"state {r['bytes'] / 1e3:6.1f}KB  ({raw_bytes / r['bytes']:4.2f}x "
            f"smaller)")


if __name__ == "__main__":
    print("PHASE 33g (T1.8): representation-width byte reduction on the "
          "33c/33d protocol\n")

    # -- the bar ---------------------------------------------------------
    Ap, _, proto_bytes, n_proto = run_prototypes()
    proto_acc, proto_forg = task_acc_matrix_to_metrics(Ap)
    print(f"  bar: prototypes ACC {proto_acc:.3f}  FORG {proto_forg:.3f}  "
          f"state {proto_bytes / 1e3:.1f}KB  ({n_proto} protos)")
    print(f"  reference: 33c replay 0.913 at ~{REPLAY_BYTES / 1e3:.1f}KB "
          f"(tops the committed ladder)\n")

    # -- lever (b) evidence: measure the count distribution FIRST ---------
    print("P count census (prediction (b) asks for this before the lever):")
    census_specs = {}
    for K in (40, 160):
        base = run_arm(K, spec=None)
        P = base['org'].P
        nz = P > 0
        dense = P.nbytes
        sparse = compress(base['org'], spec=SPEC_LOSSLESS).p_bytes
        print(f"  K={K:>3}: nnz {int(nz.sum()):>5}/{K * K} ({nz.mean():.3f} dense)  "
              f"dense P {dense / 1e3:6.1f}KB -> CSR {sparse / 1e3:5.1f}KB "
              f"({dense / sparse:5.2f}x, LOSSLESS)")
        for floor in (1, 2, 5):
            sp = CompressionSpec(xi_dtype=np.complex64, p_floor=floor)
            st = compress(base['org'], spec=sp)
            m, ag, tv, mt = graph_fidelity(P, st.P_full())
            print(f"         floor>{floor}: P {st.p_bytes / 1e3:5.1f}KB "
                  f"({dense / st.p_bytes:5.2f}x)  graph fidelity: mass {m:.3f}  "
                  f"surviving-row next-hop agree {ag:.3f}  TV drift {tv:.3f}  "
                  f"rows emptied {mt:.3f}")
        census_specs[K] = base
    print()

    # -- main sweep ------------------------------------------------------
    print("main sweep (evict=250; compression applied at every task boundary):")
    rows = {}
    for K in K_SWEEP:
        base = run_arm(K, spec=None)
        raw = base['bytes']
        rows[(K, 'raw')] = base
        a_acc = ANCHOR_33D_ACC[K]
        ok = abs(base['acc'] - a_acc) < 5e-4
        anchor = f"ACC {a_acc:.3f}"
        if K in ANCHOR_33D_FORG:
            ok = ok and abs(base['forg'] - ANCHOR_33D_FORG[K]) < 5e-4
            anchor += f" / FORG {ANCHOR_33D_FORG[K]:.3f}"
        print(f"\n  K={K}  (33d anchor {anchor}: "
              f"{'REPRODUCED' if ok else 'MISMATCH'})")
        print(fmt(base, "uncompressed (33d)", raw))
        for tag, spec in (("lossless (Psparse+meta32)", SPEC_LOSSLESS),
                          ("c64 xi", SPEC_C64),
                          ("c64 + Psparse + meta32", SPEC_FULL)):
            r = run_arm(K, spec=spec)
            rows[(K, tag)] = r
            print(fmt(r, tag, raw))
        # the lossless lever must move bytes and NOTHING else
        L = rows[(K, "lossless (Psparse+meta32)")]
        same = (L['acc'] == base['acc'] and L['forg'] == base['forg']
                and np.array_equal(L['A'], base['A']))
        print(f"        lossless lever leaves the run bit-identical: "
              f"{'YES' if same else 'NO -- INVESTIGATE'}")
        dA = float(np.abs(rows[(K, "c64 xi")]['A'] - base['A']).max())
        print(f"        c64 store vs full width, max |d task-accuracy| {dA:.4f}")
        C = rows[(K, "c64 xi")]
        F = rows[(K, "c64 + Psparse + meta32")]
        same2 = C['acc'] == F['acc'] and np.array_equal(C['A'], F['A'])
        print(f"        adding the lossless levers to c64 changes only bytes: "
              f"{'YES' if same2 else 'NO -- INVESTIGATE'}")

    # -- store vs eval-only: does the loss compound? ----------------------
    print("\nstore (carried forward, loss compounds) vs eval-only "
          "(full-width learning):")
    for K in K_SWEEP:
        s = rows[(K, "c64 + Psparse + meta32")]
        e = run_arm(K, spec=SPEC_FULL, mode="evalonly")
        rows[(K, 'evalonly')] = e
        print(f"  K={K:>3}  store ACC {s['acc']:.3f}   eval-only ACC "
              f"{e['acc']:.3f}   delta {s['acc'] - e['acc']:+.3f}"
              f"   (uncompressed {rows[(K, 'raw')]['acc']:.3f})")

    # -- lossy levers at the bar-crossing point ---------------------------
    print("\nlossy levers at K=160 (the 33d bar-crossing point), store mode:")
    raw160 = rows[(160, 'raw')]['bytes']
    lossy = {}
    for floor in (1, 2, 5):
        sp = CompressionSpec(xi_dtype=np.complex64, p_floor=floor,
                             meta_dtype=np.float32)
        r = run_arm(160, spec=sp)
        lossy[f"floor>{floor}"] = r
        m, ag, tv, mt = r['fid']
        print(fmt(r, f"c64 + P floor>{floor}", raw160)
              + f"\n{'':>34}graph: mass {m:.3f} agree {ag:.3f} "
                f"TV {tv:.3f} rows-emptied {mt:.3f}")
    # low-rank: K=160 slots live in an N=64 field, so rank 64 is EXACT by
    # construction (the check below) and everything under it is a real
    # approximation. Swept densely because this is the lever that decides
    # where the frontier crosses the bar (prediction d).
    for rank in (8, 12, 16, 20, 24, 32, 40, 48, 56, 64):
        sp = CompressionSpec(xi_dtype=np.complex64, rank=rank,
                             meta_dtype=np.float32)
        r = run_arm(160, spec=sp)
        lossy[f"rank{rank}"] = r
        tag = ""
        if rank >= N:
            tag = ("  <- rank >= N: exact factorization, ACC must equal the "
                   "c64 arm")
        elif r['bytes'] > rows[(160, "c64 + Psparse + meta32")]['bytes']:
            tag = "  <- costs MORE than dense c64 xi; a lever only below the crossover"
        print(fmt(r, f"c64 + rank-{rank} xi", raw160) + tag)
    exact = lossy["rank64"]['acc'] == rows[(160, "c64 + Psparse + meta32")]['acc']
    print(f"  rank-{N} sanity (exact factorization reproduces the c64 arm): "
          f"{'YES' if exact else 'NO -- INVESTIGATE'}")
    print("  (P floor is accuracy-free ON THIS PROTOCOL by construction --")
    print("   the label readout never queries the graph; the fidelity columns")
    print("   are the price a phase-30 consumer would pay for those bytes.)")

    # -- where the frontier crosses the bar -------------------------------
    print("\nfrontier crossing (cheapest arm at or above the "
          f"{proto_acc:.3f} bar, seed 0):")
    cands = ([(f"K={K} compressed", rows[(K, "c64 + Psparse + meta32")])
              for K in K_SWEEP]
             + [(f"K=160 {k}", v) for k, v in lossy.items()])
    above = sorted([(lbl, r) for lbl, r in cands if r['acc'] >= proto_acc],
                   key=lambda t: t[1]['bytes'])
    below = sorted([(lbl, r) for lbl, r in cands if r['acc'] < proto_acc],
                   key=lambda t: -t[1]['bytes'])
    if above:
        lbl, r = above[0]
        print(f"  cheapest AT/ABOVE bar: {lbl}  ACC {r['acc']:.3f}  "
              f"{r['bytes'] / 1e3:.1f}KB = {r['bytes'] / proto_bytes:.2f}x the bar")
    if below:
        lbl, r = below[0]
        print(f"  dearest BELOW bar:     {lbl}  ACC {r['acc']:.3f}  "
              f"{r['bytes'] / 1e3:.1f}KB = {r['bytes'] / proto_bytes:.2f}x the bar")

    # -- 33d's cost-matched point, recomputed under compression -----------
    # 33d's headline cost comparison was "the largest K whose state fits the
    # bar's 30.7KB" -- K=24, ACC 0.644. Compression moves that point, and
    # moving it is the whole purpose of this phase.
    print(f"\ncost-matched point under compression (state <= the bar's "
          f"{proto_bytes / 1e3:.1f}KB):")
    matched = None
    for K in (40, 44, 48, 52, 56, 60):
        r = run_arm(K, spec=SPEC_FULL)
        rows[(K, 'costmatch')] = r
        fits = r['bytes'] <= proto_bytes
        print(f"  K={K:>3}  ACC {r['acc']:.3f}  FORG {r['forg']:.3f}  "
              f"{r['bytes'] / 1e3:5.1f}KB  {'fits' if fits else 'over budget'}")
        if fits and (matched is None or K > matched[0]):
            matched = (K, r)
    if matched:
        K_m, r_m = matched
        print(f"  -> cost-matched K={K_m} at {r_m['bytes'] / 1e3:.1f}KB: ACC "
              f"{r_m['acc']:.3f} vs the bar's {proto_acc:.3f}")
        print(f"     (33d's uncompressed cost-matched point was K=24, ACC 0.644 "
              f"-- compression buys\n      {K_m - 24} more slots inside the same "
              f"budget, worth {r_m['acc'] - 0.644:+.3f} ACC, still "
              f"{r_m['acc'] - proto_acc:+.3f} vs the bar)")

    # -- robustness supplement: is "compression is free" a seed-0 story? --
    # T1.6's precedent: a single seed sign-flipped its headline. Here the
    # comparison is PAIRED (same seed, same stream, compressed vs not), which
    # is the sharper test -- if compression is genuinely free, every paired
    # delta is ~0, not merely small on average.
    print("\nrobustness supplement (paired, full-protocol reseeds s=0..4 at "
          "K=160):")
    supp_specs = [("c64+Psparse+meta32", SPEC_FULL)]
    supp_specs += [(f"rank{r_}", CompressionSpec(xi_dtype=np.complex64, rank=r_,
                                                 meta_dtype=np.float32))
                   for r_ in (40, 32, 24, 20, 16)]
    print(f"  {'seed':>4} {'uncompr.':>9}" +
          "".join(f" {n:>16}" for n, _ in supp_specs))
    deltas = {n: [] for n, _ in supp_specs}
    supp_acc = {n: [] for n, _ in supp_specs}
    for s_ in range(5):
        base_s = run_arm(160, spec=None, seed=s_)
        cells = []
        for n, sp in supp_specs:
            r = run_arm(160, spec=sp, seed=s_)
            d = r['acc'] - base_s['acc']
            deltas[n].append(d)
            supp_acc[n].append(r['acc'])
            cells.append(f"{r['acc']:.3f}({d:+.3f})".rjust(17))
        print(f"  {s_:>4} {base_s['acc']:9.3f}" + "".join(cells))
    print("  paired delta vs uncompressed, across seeds "
          "(pre-registered survival bar: |delta| <= 0.02 on EVERY seed):")
    robust = set()
    for n, _ in supp_specs:
        d = np.array(deltas[n])
        ok = np.abs(d).max() <= 0.02
        if ok:
            robust.add(n)
        print(f"    {n:<20} mean {d.mean():+.4f}  min {d.min():+.4f}  "
              f"max {d.max():+.4f}  -> {'FREE' if ok else 'NOT free (seed-unstable)'}")
    print("  Note: the low-rank ranks were swept and the frontier point chosen "
          "on seed 0.\n  Selecting the best of a grid on one seed is exactly "
          "the selection bias T1.6\n  measured; that is what this paired "
          "supplement exists to catch.")

    # -- cost-per-accuracy curve / frontier -------------------------------
    print("\nbyte-accuracy frontier (compare with 33d's cost curve):")
    print(f"  {'arm':<34} {'ACC':>6} {'FORG':>6} {'stateKB':>8} {'KB/ACCpt':>9} "
          f"{'vs bar':>7}")

    def line(label, acc, forg, b):
        print(f"  {label:<34} {acc:6.3f} {forg:6.3f} {b / 1e3:8.1f} "
              f"{b / 1e3 / acc:9.1f} {b / proto_bytes:6.1f}x")

    line("prototypes (bar)", proto_acc, proto_forg, proto_bytes)
    line("replay (33c committed)", 0.913, 0.105, REPLAY_BYTES)
    for K in K_SWEEP:
        line(f"organism K={K} uncompressed", rows[(K, 'raw')]['acc'],
             rows[(K, 'raw')]['forg'], rows[(K, 'raw')]['bytes'])
    for K in K_SWEEP:
        r = rows[(K, "c64 + Psparse + meta32")]
        line(f"organism K={K} compressed", r['acc'], r['forg'], r['bytes'])
    for tag in ("rank40", "rank32", "rank24", "rank16"):
        line(f"organism K=160 {tag}", lossy[tag]['acc'], lossy[tag]['forg'],
             lossy[tag]['bytes'])

    # -- verdict against the pre-registered predictions --------------------
    print("\nverdict (pre-registered predictions (a)-(d)):")
    best_c = rows[(160, "c64 + Psparse + meta32")]
    raw160r = rows[(160, 'raw')]

    drift = max(abs(rows[(K, "c64 + Psparse + meta32")]['acc']
                    - rows[(K, 'raw')]['acc']) for K in K_SWEEP)
    drift_seeds = np.abs(np.array(deltas["c64+Psparse+meta32"])).max()
    print(f"  (a) complex64 xi: dominant term halved "
          f"({raw160r['bytes'] / 1e3:.1f}KB -> {best_c['bytes'] / 1e3:.1f}KB "
          f"at K=160, {raw160r['bytes'] / best_c['bytes']:.2f}x overall); "
          f"max |dACC| across the sweep {drift:.3f}")
    print(f"      -> {'HELD' if drift <= 0.02 else 'MISSED'} "
          f"(bar: drift inside the harness tolerance band, <= 0.02); "
          f"worst paired\n         reseed drift {drift_seeds:.3f} over s=0..4, "
          f"so this is not a seed-0 story")

    dense160 = 8 * 160 * 160
    csr160 = compress(raw160r['org'], spec=SPEC_LOSSLESS).p_bytes
    print(f"  (b) P sparsification at K=160: {dense160 / 1e3:.1f}KB -> "
          f"{csr160 / 1e3:.1f}KB = {dense160 / csr160:.1f}x, at a count floor "
          f"of 0 (LOSSLESS)")
    print(f"      -> {'HELD' if dense160 / csr160 > 5 else 'MISSED'} "
          f"(bar: >5x at K=160)")

    crossed = best_c['acc'] >= proto_acc
    print(f"  (c) compressed ACC at K=160: {best_c['acc']:.3f} vs the "
          f"{proto_acc:.3f} bar -- {'still ABOVE' if crossed else 'BELOW'}")
    if not crossed:
        print("      -> the honest negative (c) fires; frontier printed above.")
    else:
        print("      -> (c)'s negative does not fire; the compressed state "
              "keeps 33d's crossing.")

    tgt = 2 * proto_bytes
    def is_robust(lbl):
        """An arm may only be CLAIMED if the supplement cleared it. The
        dense-c64 arms are the lossless-plus-quantization family (tested
        directly at K=160, and its paired delta is identically zero at every
        K on seed 0); the low-rank arms must appear in `robust`."""
        if "rank" not in lbl:
            return "c64+Psparse+meta32" in robust
        return lbl.split()[-1] in robust
    inside = [(lbl, r) for lbl, r in cands
              if r['bytes'] <= tgt and r['acc'] >= proto_acc and is_robust(lbl)]
    inside_seed0 = [(lbl, r) for lbl, r in cands
                    if r['bytes'] <= tgt and r['acc'] >= proto_acc]
    print(f"  (d) target: >= K=160-equivalent accuracy (>= {proto_acc:.3f}, the bar) "
          f"inside ~2x the bar's bytes ({tgt / 1e3:.1f}KB)")
    print(f"      compressed K=160: {best_c['acc']:.3f} at "
          f"{best_c['bytes'] / 1e3:.1f}KB = {best_c['bytes'] / proto_bytes:.2f}x the bar, "
          f"{best_c['bytes'] / REPLAY_BYTES:.2f}x replay's footprint "
          f"(33d spent 12.1x)")
    if inside:
        lbl, r = min(inside, key=lambda t: t[1]['bytes'])
        print(f"      -> MET by a ROBUST arm: {lbl} at {r['bytes'] / 1e3:.1f}KB "
              f"({r['bytes'] / proto_bytes:.2f}x), ACC {r['acc']:.3f}")
    else:
        print("      -> NOT met by any arm that survives reseeding.")
        if inside_seed0:
            lbl, r = min(inside_seed0, key=lambda t: t[1]['bytes'])
            print(f"         On seed 0 alone {len(inside_seed0)} arm(s) land "
                  f"inside 2x -- cheapest {lbl},\n         ACC {r['acc']:.3f} at "
                  f"{r['bytes'] / 1e3:.1f}KB ({r['bytes'] / proto_bytes:.2f}x) -- but "
                  f"the paired supplement above\n         puts the low-rank lever's "
                  f"seed spread at or beyond its own margin, so that\n         "
                  f"crossing is NOT claimable. Lever (iii) is a measured NEGATIVE.")
        if above:
            lbl, r = above[0]
            print(f"         Robust position instead: the c64+lossless family at "
                  f"{best_c['bytes'] / 1e3:.1f}KB\n         "
                  f"({best_c['bytes'] / proto_bytes:.2f}x the bar, "
                  f"{best_c['bytes'] / REPLAY_BYTES:.2f}x replay) holding "
                  f"33d's {best_c['acc']:.3f} exactly.")

    print("\n  cost branch (cost-effective at equal accuracy) vs 33d:")
    for K in K_SWEEP:
        r0, r1 = rows[(K, 'raw')], rows[(K, "c64 + Psparse + meta32")]
        print(f"    K={K:>3}: {r0['bytes'] / 1e3 / r0['acc']:7.1f} -> "
              f"{r1['bytes'] / 1e3 / r1['acc']:7.1f} KB/ACC-point "
              f"(bar {proto_bytes / 1e3 / proto_acc:.1f})")
    print("  OWNER DECISION POINT unchanged: this phase moves the price, not "
          "the\n    mechanism; the hold decision remains the owner's.")
