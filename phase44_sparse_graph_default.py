"""
PHASE 44 (T7.2) -- SPARSE P AS THE DEFAULT PATH.

WHAT THIS ASKS. The work order puts this second because "the graph
contributes 0.36x to the cost floor" and 33g measured P at ~6% density.
T6.3 (phase 40) already built `SparseTransitions` and moved `next_hops`
onto a shared Dijkstra with the dense body pinned bitwise (harness 12).
The ask: make CSR the default storage/compute path where T6.3 measured it
to WIN, keep dense where it measured a NEGATIVE (full `kstep` -- a dense
K^2 output is BLAS's home ground), extend the bitwise pinning to every op
whose default path changes, and re-derive the graph term in the floor.

THE FIRST THING TO SAY IS THAT ONE PREMISE OF THE ASK IS ALREADY SPENT.
33h's 0.36x graph term is not a dense number waiting to be sparsified -- it
IS the CSR number (33g's headline spec is c64 + CSR at floor 0, and 33h
priced the floor with it). At K=112 a dense float64 P is 100 352 B; the
floor's graph term is 10.46 KB. So "CSR as the default STORAGE path" moves
the cost floor by exactly zero, and this phase has to look somewhere else
for bytes. It looks in two places the project has never priced: the CSR
ENCODING itself (index and count dtypes), and the COMPUTE default, which
is a speed question rather than a byte question.

======================================================================
PRE-REGISTRATION (committed before the run that tests it)
======================================================================

P1 -- THE PREMISE CHECK. The floor's graph term is already CSR, so making
CSR the default storage path moves the floor by 0.000x. Measured as: dense
P bytes at K=112 vs the graph term phase 43 measured (10.46 KB), and the
floor recomputed with a CSR-by-default store (identical by construction).
  MUNDANE ACCOUNT (M1): 33h might have quoted a dense-with-float32 graph,
  in which case there IS a sparsification saving left. Falsified by
  arithmetic if dense-float32 (K^2*4 = 50 176 B) is also >> 10.46 KB.

P2 -- WHERE THE GRAPH BYTES ACTUALLY ARE, DERIVED BEFORE MEASURING. The
CSR triple currently stores data as float32 (4 B), indices as int32 (4 B)
and indptr as int32. From phase 43's measured graph term,
10 460 = 8*nnz + 4*(K+1) at K=112 gives nnz = 1251 (9.97% density). Two
LOSSLESS narrowings the project has never taken, both guarded exactly the
way 33g guards float32 counts:
    indices/indptr -> int16   whenever K <= 32767 and nnz <= 32767
    data           -> uint16  whenever every count is an integer < 65536
              (uint32 when < 2^32; float32 otherwise -- current behavior)
giving 4*nnz + 2*(K+1) = 5 230 B, i.e. the graph term 0.355 -> 0.178.
PREDICTION, carrying phase 43's other terms unchanged:
    c64 + narrow CSR         1.981 + 0.178 = 2.159 x 0.958 = 2.07x
    real+phase + narrow CSR  1.021 + 0.178 = 1.199 x 0.958 = 1.15x
DECISION RULE: if the measured narrow-CSR floor on the c64 store is
<= 2.00x, the <=2x fallback is met by GRAPH engineering alone and branch
(B) of T7.1's fork changes character. Predicted NOT -- it misses by ~0.07x,
which would say the fallback is decided by the complex-vs-real term and
never by the graph. (Note: 1.15x is suspiciously close to the work order's
sandbox anchor of "~1.13x", which T7.1 could not reproduce at 1.32x. If
this arm lands there, the likeliest reconciliation is that the sandbox was
pricing a narrowed graph as well as a real store, and said so only for the
store.)

P3 -- THE WORK ORDER'S OWN MUNDANE ACCOUNT: does density rise with
observation count? Measured at FIXED K by feeding 1x, 2x, 4x, 8x, 16x the
observations, on the digits protocol (K=112) and on the gutenberg8 text
recipe (K=1200). PREDICTION: density rises and then SATURATES well below
1, because a symbol's successor set is bounded by how many distinct
symbols actually follow it, not by K. Literature check before predicting
the baseline's behavior: bigram graphs over a FIXED vocabulary have
out-degree bounded by V and empirically sublinear in corpus size (the
same Heaps-law-shaped saturation that makes n-gram LMs sparse at every
scale), so saturation is the expected behavior and a linear rise would be
the surprise.
  DECISION RULE: if density at 16x observations exceeds 2x the density at
  1x, the CSR saving erodes with scale and every byte number here must be
  re-quoted with an observation count attached. If it saturates, CSR's
  advantage is structural and phase 42's corpus-scale 3.64% at K=1580 is
  the endpoint, not a snapshot.

P4 -- THE COMPUTE DEFAULT. T6.3 measured sparse `next_hops` >= 2x only at
K >= 800 (6.1x at K=1580) and full `kstep` as a NEGATIVE (0.1-0.2x).
PREDICTION: the CROSSOVER -- the K where sparse first beats dense INCLUDING
the O(K^2) build -- lies between K=200 and K=600 on this host; a size-gated
auto-dispatch above it is bitwise identical whenever P holds integer
counts (T6.3's own condition), and `kstep` must be left dense at every K.
  MUNDANE ACCOUNT (M4): phase 42 measured +-20% wall-clock variance on
  bit-identical work on this host, so a single crossover constant is not
  portable. The deliverable is the measured CURVE plus a deliberately
  CONSERVATIVE threshold, not a tuned constant -- and the threshold is
  chosen so that the dispatch is INERT on every committed benchmark, which
  makes the default change unable to move any anchor.

P5 -- HONEST NEGATIVE BRANCH. If the crossover is above K=1580 (i.e.
sparse never wins at any shape this project runs), the correct outcome is
to NOT change the default and to record the measurement, exactly as T6.5
recorded E4's deferral. A default change that helps nothing is a
regression surface for free.
"""

import time

import numpy as np

import organism_compress as oc
import phase33h_cost_frontier as p33h
import text_recipe as tr
from organism import Organism, SparseTransitions, TransitionGraph, normalize
from organism_compress import CompressionSpec, compress
from phase33c_gate_retest import EVICT, N, NORM, TASKS
from phase43_phase_channel_audit import (DECODER, K_FLOOR, SEEDS, SEEDS_HELD,
                                         SPEC, SPEC_RP, score, train_digits)

SPEC_NARROW = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0,
                              meta_dtype=np.float32, p_narrow=True)
SPEC_RP_NARROW = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0,
                                 meta_dtype=np.float32, real_phase=True,
                                 p_narrow=True)


def synth_graph(K, out_degree=10, seed=0, counts_max=200):
    """A transition graph with K symbols and a FIXED out-degree -- the shape
    `SparseTransitions` documents (out-degree does not grow with K, so the
    matrix gets emptier the bigger the organism is). Integer counts, so the
    sparse/dense equality is the bitwise one."""
    rng = np.random.default_rng(seed)
    g = TransitionGraph(K)
    P = np.zeros((K, K))
    for i in range(K):
        cols = rng.choice(K, size=min(out_degree, K), replace=False)
        P[i, cols] = rng.integers(1, counts_max, size=cols.size)
    g.P = P
    return g


def time_op(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def density_curve(build, multiples, label):
    rows = []
    for m in multiples:
        P, k, nobs = build(m)
        nnz = int((P > 0).sum())
        rows.append((m, nobs, k, nnz, nnz / max(k * k, 1)))
        print(f"    {label} x{m:<3} obs {nobs:>9}  live K {k:>5}  nnz {nnz:>7}  "
              f"density {rows[-1][4]:.5f}")
    return rows


if __name__ == "__main__":
    t_all = time.time()
    print("PHASE 44 (T7.2): sparse P as the default path\n")

    # ---------------- 1. the premise check --------------------------------
    print("=" * 70)
    print("(1) IS THE FLOOR'S GRAPH TERM ALREADY CSR? -- P1")
    org, ro, (Xte, yte) = train_digits(seed=5)
    st = compress(p33h.StoreView(org), spec=SPEC)
    Kl = int(org.used.sum())
    dense64 = Kl * Kl * 8
    dense32 = Kl * Kl * 4
    print(f"  live K {Kl}   dense float64 P {dense64:>8} B   dense float32 P "
          f"{dense32:>8} B")
    print(f"  33h/33g CSR graph term (measured here)   {st.p_bytes:>8} B "
          f"= {st.p_bytes / 1e3:.2f} KB   nnz {st.p_data.size}  "
          f"density {st.p_data.size / (Kl * Kl):.5f}")
    print(f"  P1 {'HELD' if st.p_bytes < dense32 / 2 else 'MISSED'}: the floor's "
          f"graph term is the CSR number already -- CSR-as-default-storage moves "
          f"the floor by {0.0:.3f}x")

    # ---------------- 2. the CSR encoding, derived then measured ----------
    print("\n" + "=" * 70)
    print("(2) NARROWING THE CSR ENCODING -- P2 (lossless, guarded)")
    nnz = st.p_data.size
    derived = 4 * nnz + 2 * (Kl + 1)
    print(f"  derived: 4*nnz + 2*(K+1) = 4*{nnz} + 2*{Kl + 1} = {derived} B")
    stn = compress(p33h.StoreView(org), spec=SPEC_NARROW)
    print(f"  measured narrow CSR graph term: {stn.p_bytes} B "
          f"({'MATCHES' if stn.p_bytes == derived else 'DIFFERS FROM'} the derivation)")
    print(f"  dtypes: data {stn.p_data.dtype}  indices {stn.p_indices.dtype}  "
          f"indptr {stn.p_indptr.dtype}")
    exact = np.array_equal(stn.P_full(), np.asarray(p33h.StoreView(org).P))
    print(f"  lossless: reconstructed P identical to the live P: {exact}")

    rows = []
    for s in SEEDS + SEEDS_HELD:
        o, r_, (Xt, yt) = train_digits(seed=s)
        acc = score(o, r_, Xt, yt, DECODER[1])
        bar = p33h.run_prototypes_seed(s)
        view = p33h.StoreView(o)
        rows.append(dict(
            seed=s, acc=acc, bar_acc=bar["acc"], bar_b=bar["bytes"],
            c64=compress(view, spec=SPEC).nbytes,
            c64n=compress(view, spec=SPEC_NARROW).nbytes,
            rp=compress(view, spec=SPEC_RP).nbytes,
            rpn=compress(view, spec=SPEC_RP_NARROW).nbytes))

    def floor(key, held=True):
        rr = [x for x in rows if (x["seed"] in SEEDS_HELD) == held]
        ok = np.mean([p33h.kbpp(x[key], x["acc"]) for x in rr])
        bk = np.mean([p33h.kbpp(x["bar_b"], x["bar_acc"]) for x in rr])
        return ok, ok / bk

    print("\n  the floor, held-out seeds 5-9 (33h's metric):")
    for key, lbl, pred in (("c64", "c64 + CSR (33h)", 2.24),
                           ("c64n", "c64 + narrow CSR", 2.07),
                           ("rp", "real+phase + CSR (43)", 1.32),
                           ("rpn", "real+phase + narrow CSR", 1.15)):
        ok, ratio = floor(key)
        print(f"    {lbl:26s} {ok:6.2f} KB/ACC-pt = {ratio:.3f}x   "
              f"(pre-registered {pred:.2f}x)")
    _, r_c64n = floor("c64n")
    print(f"  P2 fallback by graph engineering alone: "
          f"{'MET' if r_c64n <= 2.0 else 'NOT met'} at {r_c64n:.3f}x")

    # ---------------- 3. density vs observation count ---------------------
    print("\n" + "=" * 70)
    print("(3) DOES DENSITY RISE WITH OBSERVATIONS? -- P3 (the mundane account)")

    def digits_build(m):
        o, _, _ = train_digits(seed=0, epochs=3 * m)
        idx = np.where(o.used)[0]
        return np.asarray(o.P)[np.ix_(idx, idx)], len(idx), 3 * m * 1400 * 8

    print("  digits protocol, K=112 (epochs scaled):")
    d_dig = density_curve(digits_build, (1, 2, 4, 8, 16), "digits")

    R = tr.Recipe(tr.load_gutenberg8(verbose=False), **tr.RECIPE_20, verbose=False)
    full = R.train_seq

    def text_build(m):
        frac = len(full) * m // 16
        o = Organism(N=R.N, K=1200, omega=0.15, beta=10.0, seed=0, backend="numba")
        o.perceive(R.stream(tr.HOLD_19, seq=full[:frac]), **tr.PERCEIVE_19)
        idx = np.where(o.used)[0]
        return np.asarray(o.P)[np.ix_(idx, idx)], len(idx), frac
    print("  gutenberg8 text recipe, K=1200 (stream fraction scaled):")
    d_txt = density_curve(text_build, (1, 2, 4, 8, 16), "text  ")

    for name, rws in (("digits", d_dig), ("text", d_txt)):
        ratio = rws[-1][4] / max(rws[0][4], 1e-12)
        print(f"  {name}: density x{rws[0][0]} -> x{rws[-1][0]} is "
              f"{rws[0][4]:.5f} -> {rws[-1][4]:.5f} = {ratio:.2f}x")
    p3 = all(rws[-1][4] / max(rws[0][4], 1e-12) <= 2.0 for _, rws in
             (("digits", d_dig), ("text", d_txt)))
    print(f"  P3 {'HELD (saturates)' if p3 else 'MISSED (density rises >2x)'}")

    # ---------------- 4. the compute crossover ----------------------------
    print("\n" + "=" * 70)
    print("(4) WHERE DOES SPARSE ACTUALLY WIN? -- P4 (build cost INCLUDED)")
    print(f"  {'K':>6} | {'next_hops d/s':>22} | {'rollout d/s':>20} | "
          f"{'kstep_row d/s':>20} | {'kstep(2) d/s':>20}")
    cross = None
    for K in (112, 200, 400, 800, 1580):
        g = synth_graph(K)
        idx = np.arange(K)
        reps = max(1, int(2e7 / (K * K)))
        t_nh_d = time_op(lambda: g.next_hops(idx, 0, sparse=False), reps)
        t_nh_s = time_op(
            lambda: SparseTransitions.from_graph(g, idx).next_hops(0), reps)
        rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
        t_ro_d = time_op(lambda: g.rollout(idx, 0, 50, rng1), reps)
        t_ro_s = time_op(
            lambda: SparseTransitions.from_graph(g, idx).rollout(0, 50, rng2), reps)
        t_kr_d = time_op(lambda: g.kstep(idx, 2)[0], max(1, reps // 4))
        t_kr_s = time_op(
            lambda: SparseTransitions.from_graph(g, idx).kstep_row(0, 2),
            max(1, reps // 4))
        t_ks_d = time_op(lambda: g.kstep(idx, 2), max(1, reps // 4))
        t_ks_s = time_op(lambda: SparseTransitions.from_graph(g, idx).kstep(2),
                         max(1, reps // 4))
        print(f"  {K:>6} | {t_nh_d * 1e3:8.2f}/{t_nh_s * 1e3:7.2f} ms "
              f"{t_nh_d / t_nh_s:5.2f}x | {t_ro_d * 1e3:6.2f}/{t_ro_s * 1e3:6.2f} "
              f"{t_ro_d / t_ro_s:5.2f}x | {t_kr_d * 1e3:6.2f}/{t_kr_s * 1e3:6.2f} "
              f"{t_kr_d / t_kr_s:5.2f}x | {t_ks_d * 1e3:6.2f}/{t_ks_s * 1e3:6.2f} "
              f"{t_ks_d / t_ks_s:5.2f}x")
        if cross is None and t_nh_d / t_nh_s > 1.0:
            cross = K
        # equality, at every K, on the ops whose default path could change
        nd, dd = g.next_hops(idx, 0, sparse=False)
        ns, ds = SparseTransitions.from_graph(g, idx).next_hops(0)
        assert np.array_equal(nd, ns), f"next_hops nxt mismatch at K={K}"
        assert np.array_equal(dd, ds), f"next_hops dist mismatch at K={K}"
    print(f"\n  measured crossover for next_hops (incl. build): "
          f"K = {cross if cross else '> 1580 (never wins)'}")
    print(f"  TransitionGraph.SPARSE_MIN_K (conservative default) = "
          f"{TransitionGraph.SPARSE_MIN_K}")

    # ---------------- 5. the default, and what it touches -----------------
    print("\n" + "=" * 70)
    print("(5) THE DEFAULT PATH -- P4/P5")
    for K in (40, 112, 800, 1580):
        g = synth_graph(K)
        idx = np.arange(K)
        print(f"  K={K:>5}  auto-dispatch picks "
              f"{'SPARSE' if g._sparse_worth(idx) else 'dense '}"
              f"   (committed benchmarks run at K <= 1580; phases 30/40 at K <= 60)")
    g = synth_graph(300)
    g.P = g.P * 0.5                      # non-integer counts: p_decay's regime
    print(f"  non-integer counts at K=300: auto-dispatch picks "
          f"{'SPARSE' if g._sparse_worth(np.arange(300)) else 'dense'} "
          f"-- bitwise identity is only available on integer counts (T6.3)")
    print(f"\nTOTAL {time.time() - t_all:.1f}s")
