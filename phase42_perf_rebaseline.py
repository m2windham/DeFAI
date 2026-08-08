"""
PHASE 42 -- PERFORMANCE RE-BASELINE AND PROFILE (T6.5 / E-track)

Every performance number on record predates three changes to the shapes the
engine actually runs at:

  * T1.8 / phase 33g (`organism_compress.py`): complex64 stores and CSR
    transition matrices -- a narrowed store now takes a promote/restore path
    through `Organism.perceive` that did not exist when E2 was measured;
  * T3.3 (`SymbolRegistry`): the perceive kernel gained registry branches
    (dead when the registry is off, but they are in the compiled loop) and
    E3 gained schema v3;
  * phase 27 (T2.1): the corpus-tier workload is now 42 books / 3.58M
    in-vocab tokens / V=354 / K_CAP=1416, not phase 23's 408K / V=395 /
    K_CAP=1580, and its stage timings were reported once, as a byproduct.

The pinned E2 numbers (58K frames/s numba, 1.40 min full 3-epoch perceive,
"13.0x") date from 2026-07-15; the 2026-08-04 onboarding pass already
measured 67.3K frames/s and a 6.0x ratio on different hardware, which is the
standing warning that the RATIO is hardware-relative and only the absolute
numba throughput is portable. This phase re-measures all of it on one host,
in one sitting, and -- the part that has never been done -- profiles WHERE
the time goes, so the optimization shortlist is ranked by measured cost
rather than by the 2026-07 intuition that the K x N overlap matvec is the
bound.

The deliverable that matters downstream is the E4 (GPU statistics tier)
go/no-go. E4 was scoped as "permutation nulls first, timed to land around
phase 27/28". Phase 27 has now run, so the question is answerable with
measurement instead of a plan: if permutation nulls are the bill at 5M-word
scale, E4 is justified; if they are not, E4 should be deferred and the
shortlist should point somewhere else. Deferring is an equally useful
answer and is recorded as such.

--------------------------------------------------------------------------
PRE-REGISTERED PREDICTIONS (frozen before any measurement in this phase;
AGENT_TARGETS T6.5 (a)-(c), with the decision rule for each)
--------------------------------------------------------------------------

(a) NUMBA ABSOLUTE THROUGHPUT HOLDS OR IMPROVES post-compression/registry.
    Operationalization: at the pinned phase-23 shape (V=395, DIM=50,
    K_CAP=1580, 408K tokens x hold 4 x 3 epochs), the numba backend with an
    uncompressed complex128 store and the registry OFF sustains >= 58K
    frames/s (the pinned E2 number).
    Honest-negative branch: if throughput has REGRESSED below 58K on a host
    that is otherwise comparable, that is a T3.3/T1.8 tax and the phase must
    localize it (registry branches vs promote/restore vs neither) rather
    than report a number and move on. Hardware confound named in advance:
    this is a different host than either prior measurement, so the ratio to
    numpy is NOT the test -- the absolute numba rate is.

(b) PERMUTATION NULLS DOMINATE CORPUS-TIER WALL-CLOCK, making E4 justified.
    Operationalization: in the phase-27 run, the permutation-null work
    (stage B's 11 k-values x 500 MI nulls, stage C's per-word 500-shuffle
    nulls) is >= 50% of total wall-clock. Decision rule: >= 50% -> E4 GO
    (and the Amdahl ceiling is quotable); < 50% -> E4 DEFER, and the phase
    must say what the bill actually is instead. Either way the ceiling is
    reported as a speedup bound on TOTAL corpus-tier time, never as a
    speedup on the null stage alone -- 33h's unconstrained-KB/ACC-point
    lesson applied to a different metric.

(c) HARNESS RUNTIME GROWTH IS SUBLINEAR IN CHECK COUNT (shared setup).
    Operationalization: the harness went 27 -> 65 checks (2.41x). If
    wall-clock grew by less than 2.41x on the same backend, growth is
    sublinear and the harness does not need tiering.
    Honest-negative branch: if growth is superlinear, the phase reports
    per-section wall-clock and names which sections would move to a slow
    tier -- a concrete tiering proposal, not a complaint.

Additional measurements with no prediction attached (they exist to make the
shortlist rankable, and are reported whatever they say): per-section harness
cost, the K-scaling of perceive throughput, a kernel-level decomposition of
the frame budget against the BLAS matvec ceiling, the real cost of a
compressed (c64 + CSR) store on the perceive path, and dense-vs-CSR
transition-matrix bytes at true corpus K.

--------------------------------------------------------------------------
SCOPE AND METHOD NOTES
--------------------------------------------------------------------------
* Protocol-level only. No library file is touched: this script imports
  `organism`, `fastpath`, `organism_compress`, `regression_harness` and
  measures them as they are. Nothing here can change a committed number.
* The phase-23-shape benchmark reuses `e2_benchmark.py`'s exact
  configuration and stream construction (same V/DIM/HOLD/K_CAP/seed/zipf
  draw) so its numbers are directly comparable to the pinned ones rather
  than merely similar.
* numpy is timed on a slice and extrapolated, exactly as `e2_benchmark.py`
  does -- running the full 4.9M frames on the reference backend is the cost
  E2 removed. Extrapolation is linear in frames and is labelled "projected"
  everywhere it appears.
* Timings are wall-clock on a 4-core host with no other load; each shape is
  measured after a JIT warmup so compilation is never inside a timed
  region (compile cost is reported separately, once).

Run: python phase42_perf_rebaseline.py [part ...]
     parts: shapes bench micro csr harness nulls levers   (default: all)

--------------------------------------------------------------------------
MEASURED OUTCOME (2026-08-08; 4-core Linux host, numpy 2.4.6 / numba 0.66.0;
full numbers and their scope sentences in ROADMAP row 42)
--------------------------------------------------------------------------
FIRST, THE MISSES.

(a) MISSED AS WRITTEN, and the honest-negative branch fired. The pinned 58K
    frames/s was not reached: four full-load runs at the phase-23 shape gave
    43.8 / 50.1 / 51.4 / 54.3K (median ~51K, 1.50-1.86 min for the 3-epoch
    load vs the pinned 1.40). The branch required localizing it rather than
    blaming the host, so a same-host, interleaved, six-round A/B was run
    across THREE trees -- current, pre-T3.3 (7e0b639~1), pre-33g (92e9cc9~1)
    -- on identical work: medians 49.8 / 50.1 / 50.6K, indistinguishable.
    The registry costs -2.4% (median, ON vs OFF, base `Organism` both arms)
    against a 17% within-arm spread, and a complex64 store costs +3.6%
    against a 4% spread; both are INSIDE the noise. So there is no
    tree-attributable regression -- but the real finding is sharper than
    "the host was slow": this host spans 43.7-58.2K on bit-identical work,
    a +-20% band that STRADDLES the pin. A single absolute throughput number
    cannot function as a regression detector. Use the same-session A/B.

(b) FALSIFIED, and the pre-registered DEFER branch is the answer. Nulls are
    22.4% of phase-27 wall-clock (265s of 1184s), not the >= 50% that would
    have justified E4: stage B 163s (169s observed minus 5.8s of measured
    k-means/silhouette) + stage C 102s. Amdahl ceiling with nulls made FREE
    is 1.29x on TOTAL corpus-tier time; E4's full named scope (nulls plus
    all-pairs similarity, i.e. coverage_map's 105s) reaches 31.3% and a
    1.45x ceiling. The bill is stage A perceive: 785s = 66.3%, which E4
    explicitly does not target. **E4: DEFER.**

(c) CONFIRMED, on the same host rather than across hosts: the 27-check tree
    (15cfa65) and the 65-check tree cost numpy 82.2s vs 101.0s (1.23x) and
    numba 12.4-13.5s vs 12.5-12.7s (~1.00x) for 2.41x the checks. T1.8's and
    T3.3's 34 new checks add 8.5s numpy / 0.7s numba. No tiering needed.

WHERE THE TIME ACTUALLY GOES (the thing that had never been profiled): at
K=1580/N=50 the frame is 14 614 ns -- overlap matvec 10 927 (75%), free-slot
scan + refine 2 135 (15%), argmax 1 441 (10%), the fused z update 111 (0.8%).
So the 2026-07 hypothesis survives: the matvec is still the bound. One
correction to the E2 row's wording, though: at these shapes the operand is
1.2 MB and therefore CACHE-resident, so the bound is cache bandwidth
(84-94 GB/s effective), not DRAM -- which is why narrowing the operand pays.

THE SHORTLIST IS PRICED, NOT ARGUED (part 6). The two biggest wins are
algorithmic, CPU-only, and exact -- and they delete most of E4's own
premise: stage B's null is `G^T W G` over precomputed word-pair counts
(bit-identical contingency tables, max diff 0.0, 282.6s -> 0.18s), and
stage C's null is a multivariate-hypergeometric draw of the same table,
O(k^2) instead of O(n) (p99 agreement to 3 significant figures at n = 3 340
/ 37 495 / 259 324; 4.0x / 41.3x / 255x). Together they take the 265s of
null work to under 10s. Narrowing the matvec operand to complex64 is worth
1.48-1.77x on the matvec, but compute width is pinned at complex128 on
purpose and this phase only prices it -- it does not propose taking it.
"""

import gc
import json
import os
import sys
import time

import numpy as np

import fastpath
from organism import Organism, normalize
from polysemy_organism import PolysemyOrganism

OUT = {}


def hdr(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def hostinfo():
    import platform
    try:
        import numba
        nb = numba.__version__
    except Exception:
        nb = "absent"
    n_cpu = os.cpu_count()
    print(f"host: {platform.platform()}  python {platform.python_version()}  "
          f"cores {n_cpu}")
    print(f"numpy {np.__version__}  numba {nb}  "
          f"HAVE_NUMBA={fastpath.HAVE_NUMBA}")
    OUT["host"] = dict(platform=platform.platform(), cores=n_cpu,
                       numpy=np.__version__, numba=nb)


# ======================================================================
# shared shape definitions
# ======================================================================
# phase-23 shape: e2_benchmark.py's configuration, verbatim.
S23 = dict(name="phase-23", V=395, DIM=50, HOLD=4, N_TOKENS=408_000, EPOCHS=3,
           K_CAP=min(2000, 395 * 4), recruit=0.75)
# phase-27 shape: phase27_5m_word_scale_run.py's measured configuration
# (V=354 at MIN_COUNT=1500, 3 577 289 in-vocab tokens, K_CAP=min(2000,4V)).
S27 = dict(name="phase-27", V=354, DIM=50, HOLD=4, N_TOKENS=3_577_289, EPOCHS=3,
           K_CAP=min(2000, 354 * 4), recruit=0.75)


def make_corpus(shape, seed=0):
    """Zipf-ish token stream over unit-norm random embeddings -- e2_benchmark's
    synthetic stand-in for the corpus, at the given shape."""
    V, DIM = shape["V"], shape["DIM"]
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((V, DIM))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb_c = np.array([normalize(e.astype(complex), np.sqrt(DIM)) for e in emb])
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    seq = rng.choice(V, size=shape["N_TOKENS"], p=p)
    return emb_c, seq


def stream_of(emb_c, seq, hold, n_tokens=None):
    for w in (seq if n_tokens is None else seq[:n_tokens]):
        s = emb_c[w]
        for _ in range(hold):
            yield s


def time_perceive(shape, emb_c, seq, backend, n_tokens, epochs, K=None,
                  symbols=None, org=None):
    """Wall-clock one perceive load. Returns (organism, seconds, frames).

    `symbols=None` builds the `PolysemyOrganism` e2_benchmark uses (which
    does not forward the T3.3 registry flag); `symbols=True/False` builds a
    base `Organism`, the only class that takes the flag -- so the registry
    tax is measured Organism-vs-Organism, never across classes."""
    K = K or shape["K_CAP"]
    if org is None:
        if symbols is None:
            org = PolysemyOrganism(N=shape["DIM"], K=K, omega=0.15, beta=10.0,
                                   seed=0, backend=backend)
        else:
            org = Organism(N=shape["DIM"], K=K, omega=0.15, beta=10.0,
                           seed=0, backend=backend, symbols=symbols)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(epochs):
        org.perceive(stream_of(emb_c, seq, shape["HOLD"], n_tokens),
                     g_in=5.0, dt=0.05, eta=0.02, recruit=shape["recruit"])
    dt = time.perf_counter() - t0
    return org, dt, n_tokens * shape["HOLD"] * epochs


# ======================================================================
# PART 1 -- throughput bench
# ======================================================================
def part_bench():
    hdr("PART 1 -- perceive throughput, current tree, both backends")
    res = {}

    if fastpath.HAVE_NUMBA:
        t0 = time.perf_counter()
        fastpath.warmup()
        jit = time.perf_counter() - t0
        print(f"JIT warmup (compile or cache load): {jit:.1f}s "
              f"-- excluded from every timed region below\n")
        res["jit_warmup_s"] = jit

    for shape, np_tokens, nb_tokens, nb_epochs in (
            (S23, 25_000, S23["N_TOKENS"], S23["EPOCHS"]),
            # phase-27 shape: numba on a bounded slice (the full 3-epoch load
            # is measured for real by the phase-27 re-run, part 5), numpy on
            # the same slice ratio as e2_benchmark uses.
            (S27, 25_000, 400_000, 1),
    ):
        print(f"--- {shape['name']} shape: V={shape['V']} DIM={shape['DIM']} "
              f"K_CAP={shape['K_CAP']} hold={shape['HOLD']} "
              f"tokens={shape['N_TOKENS']} epochs={shape['EPOCHS']} "
              f"= {shape['N_TOKENS'] * shape['HOLD'] * shape['EPOCHS'] / 1e6:.1f}M frames")
        emb_c, seq = make_corpus(shape)
        full = shape["N_TOKENS"] * shape["HOLD"] * shape["EPOCHS"]

        _, t_np, f_np = time_perceive(shape, emb_c, seq, "numpy", np_tokens, 1)
        r_np = f_np / t_np
        print(f"  numpy : {f_np / 1e3:>6.0f}K frames in {t_np:>6.1f}s = "
              f"{r_np / 1e3:>5.1f}K frames/s -> full load projected "
              f"{full / r_np / 60:.1f} min")

        row = dict(numpy_rate=r_np, numpy_proj_min=full / r_np / 60)
        if fastpath.HAVE_NUMBA:
            o_nb, t_nb, f_nb = time_perceive(shape, emb_c, seq, "numba",
                                             nb_tokens, nb_epochs)
            r_nb = f_nb / t_nb
            measured = (nb_tokens == shape["N_TOKENS"]
                        and nb_epochs == shape["EPOCHS"])
            print(f"  numba : {f_nb / 1e6:>6.2f}M frames in {t_nb:>6.1f}s = "
                  f"{r_nb / 1e3:>5.1f}K frames/s "
                  f"({'measured, full load' if measured else 'measured on slice'})")
            print(f"  ratio : {r_nb / r_np:.1f}x  |  full {shape['EPOCHS']}-epoch "
                  f"perceive: {full / r_np / 60:.1f} min (numpy, projected) -> "
                  f"{full / r_nb / 60:.2f} min (numba"
                  f"{', measured' if measured else ', projected from slice'})")
            print(f"  sanity: recruited {int(o_nb.used.sum())} slots "
                  f"(bounded by K_CAP={shape['K_CAP']})")
            row.update(numba_rate=r_nb, ratio=r_nb / r_np,
                       numba_full_min=full / r_nb / 60,
                       numba_measured_full=measured,
                       slots=int(o_nb.used.sum()))
        res[shape["name"]] = row
        print()

    # --- registry / compressed-store taxes on the SAME shape and stream
    if fastpath.HAVE_NUMBA:
        print("--- T3.3 registry tax (phase-23 shape, 100K tokens, numba, "
              "base Organism both arms)")
        emb_c, seq = make_corpus(S23)
        _, t_off, f_off = time_perceive(S23, emb_c, seq, "numba", 100_000, 1,
                                        symbols=False)
        _, t_on, _ = time_perceive(S23, emb_c, seq, "numba", 100_000, 1,
                                   symbols=True)
        print(f"  registry OFF : {f_off / t_off / 1e3:>5.1f}K frames/s  ({t_off:.1f}s)")
        print(f"  registry ON  : {f_off / t_on / 1e3:>5.1f}K frames/s  ({t_on:.1f}s)"
              f"   tax {100 * (t_on / t_off - 1):+.1f}%")
        res["registry_off_rate"] = f_off / t_off
        res["registry_on_rate"] = f_off / t_on

    # --- K-scaling: is throughput 1/K (matvec-bound) or flatter?
    if fastpath.HAVE_NUMBA:
        print("\n--- K-scaling of perceive throughput (phase-23 embeddings, "
              "DIM=50, 60K tokens, numba)")
        emb_c, seq = make_corpus(S23)
        print(f"  {'K':>6}  {'frames/s':>10}  {'K x rate':>12}   "
              f"(flat 'K x rate' == pure 1/K, i.e. matvec-bound)")
        ks = {}
        for K in (40, 160, 400, 800, 1580, 3000):
            _, t, f = time_perceive(S23, emb_c, seq, "numba", 60_000, 1, K=K)
            r = f / t
            ks[K] = r
            print(f"  {K:>6}  {r / 1e3:>9.1f}K  {K * r / 1e6:>11.1f}M")
        res["k_scaling"] = ks
    OUT["bench"] = res


# ======================================================================
# PART 2 -- kernel-level decomposition: where does a frame go?
# ======================================================================
def part_micro():
    hdr("PART 2 -- frame budget vs the BLAS matvec ceiling")
    res = {}

    print("The plain-mode perceive frame (confirm=0, pool=False -- what "
          "e2_benchmark\nand phase 27 both run) does exactly one K x N "
          "complex matvec `np.dot(xic, z)`.\nThat matvec's standalone rate is "
          "therefore a hard ceiling on frames/s, and\nthe gap between it and "
          "measured throughput is everything else in the loop.\n")

    print(f"  {'shape':>10}  {'K':>6}  {'N':>4}  {'matvec/s':>10}  "
          f"{'frames/s':>10}  {'frame/matvec':>13}  {'GB/s':>7}")
    rows = {}
    for shape in (S23, S27):
        K, N = shape["K_CAP"], shape["DIM"]
        xic = np.ascontiguousarray(
            (np.random.default_rng(1).standard_normal((K, N))
             + 1j * np.random.default_rng(2).standard_normal((K, N))))
        z = np.ascontiguousarray(np.random.default_rng(3).standard_normal(N)
                                 + 1j * np.random.default_rng(4).standard_normal(N))
        out = np.empty(K, dtype=complex)
        for _ in range(200):
            np.dot(xic, z, out=out)
        n_it = 20_000
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(n_it):
            np.dot(xic, z, out=out)
        mv = n_it / (time.perf_counter() - t0)
        # bytes touched per matvec: the K x N complex128 operand dominates
        gbs = mv * K * N * 16 / 1e9
        # measured frame rate for this shape: from part 1 if it ran in this
        # process, else from P42_RATES (part 1's committed numbers)
        fr = OUT.get("bench", {}).get(shape["name"], {}).get("numba_rate")
        if fr is None and os.environ.get("P42_RATES"):
            fr = json.loads(os.environ["P42_RATES"]).get(shape["name"])
        ratio = (mv / fr) if fr else float("nan")
        print(f"  {shape['name']:>10}  {K:>6}  {N:>4}  {mv / 1e3:>9.1f}K  "
              f"{(fr or 0) / 1e3:>9.1f}K  {ratio:>12.2f}x  {gbs:>7.1f}")
        rows[shape["name"]] = dict(matvec_rate=mv, frame_rate=fr,
                                   ceiling_ratio=ratio, gbs=gbs)
    res["matvec"] = rows
    print("\n  'frame/matvec' = how many matvec-equivalents one perceive frame "
          "costs.\n  1.0x would mean the matvec IS the frame; higher means the "
          "rest of the\n  loop (argmax scan, free-slot scan, the fused z "
          "update, recruit/refine)\n  costs more than the matvec it was "
          "assumed to be dominated by.")

    # --- kernel ladder: add one piece of the real frame at a time, in the
    # order `_perceive_chunk` executes them, and measure the marginal cost.
    # Same shapes, same dtypes, JIT-compiled the same way -- the point is the
    # DELTAS between rungs, which are the per-component frame budget.
    if fastpath.HAVE_NUMBA:
        from numba import njit

        @njit(cache=True, fastmath=True)
        def _rung(frames, xi, xic, used, z, omega, dt, g_in, norm, rung):
            K, N = xi.shape
            best_acc = 0.0
            for t in range(frames.shape[0]):
                x = frames[t]
                if rung >= 1:                     # the fused z update
                    s = 0.0
                    for i in range(N):
                        zi = z[i]
                        zi = zi + dt * (1j * omega * zi + g_in * (x[i] - zi))
                        z[i] = zi
                        s += zi.real * zi.real + zi.imag * zi.imag
                    sc = norm / (np.sqrt(s) + 1e-9)
                    for i in range(N):
                        z[i] *= sc
                if rung >= 2:                     # the K x N overlap matvec
                    o = np.dot(xic, z)
                    if rung >= 3:                 # squared-magnitude argmax
                        k = 0
                        best = -1.0
                        for k2 in range(K):
                            m2 = o[k2].real * o[k2].real + o[k2].imag * o[k2].imag
                            if m2 > best:
                                best = m2
                                k = k2
                        best_acc += best
                        if rung >= 4:             # free-slot scan + refine
                            f = -1
                            for k2 in range(K):
                                if not used[k2]:
                                    f = k2
                                    break
                            ok = o[k]
                            z_al = z * np.exp(-1j * np.arctan2(ok.imag, ok.real))
                            acc = xi[k] + 0.02 * (z_al - xi[k])
                            s2 = 0.0
                            for i in range(N):
                                s2 += acc[i].real * acc[i].real + acc[i].imag * acc[i].imag
                            sc2 = norm / (np.sqrt(s2) + 1e-9)
                            for i in range(N):
                                xi[k, i] = acc[i] * sc2
                                xic[k, i] = np.conj(xi[k, i])
                            best_acc += f
                    else:
                        best_acc += o[0].real
            return best_acc

        print("\n--- kernel ladder: marginal cost of each frame component "
              "(numba, DIM=50)")
        print(f"  {'K':>6}  {'rung':>34}  {'ns/frame':>9}  {'marginal':>9}  "
              f"{'share':>6}")
        ladder = {}
        names = {0: "stream read only",
                 1: "+ fused z update (O(N))",
                 2: "+ K x N overlap matvec",
                 3: "+ squared-magnitude argmax (O(K))",
                 4: "+ free-slot scan & refine (O(K+N))"}
        for K in (S27["K_CAP"], S23["K_CAP"]):
            rgen = np.random.default_rng(11)
            nf = 40_000
            frames = np.ascontiguousarray(
                (rgen.standard_normal((nf, 50)) + 1j * rgen.standard_normal((nf, 50))))
            xi0 = np.ascontiguousarray(
                (rgen.standard_normal((K, 50)) + 1j * rgen.standard_normal((K, 50))))
            prev = None
            ladder[K] = {}
            for rung in range(5):
                xi = xi0.copy()
                xic = np.conj(xi).copy()
                used = np.ones(K, np.bool_)
                used[-1] = False
                z = np.ascontiguousarray(rgen.standard_normal(50)
                                         + 1j * rgen.standard_normal(50))
                _rung(frames[:200], xi, xic, used, z, 0.15, 0.05, 5.0,
                      np.sqrt(50.0), rung)      # compile
                gc.collect()
                t0 = time.perf_counter()
                _rung(frames, xi, xic, used, z, 0.15, 0.05, 5.0,
                      np.sqrt(50.0), rung)
                ns = (time.perf_counter() - t0) / nf * 1e9
                marg = ns - prev if prev is not None else ns
                ladder[K][names[rung]] = ns
                prev = ns
                print(f"  {K:>6}  {names[rung]:>34}  {ns:>9.0f}  "
                      f"{marg:>9.0f}  {100 * marg / ns:>5.0f}%")
            prev = None
            print()
        res["ladder"] = {str(k): v for k, v in ladder.items()}

    # --- numpy-side profile: function-level attribution on the reference path
    print("\n--- cProfile, numpy backend, phase-23 shape, 8K tokens "
          "(reference path only)")
    import cProfile
    import pstats
    import io
    emb_c, seq = make_corpus(S23)
    org = PolysemyOrganism(N=S23["DIM"], K=S23["K_CAP"], omega=0.15, beta=10.0,
                           seed=0, backend="numpy")
    pr = cProfile.Profile()
    pr.enable()
    org.perceive(stream_of(emb_c, seq, S23["HOLD"], 8_000), g_in=5.0, dt=0.05,
                 eta=0.02, recruit=S23["recruit"])
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    body = [ln for ln in s.getvalue().splitlines() if ln.strip()][:20]
    print("\n".join("  " + ln for ln in body))
    OUT["micro"] = res


# ======================================================================
# PART 3 -- compressed store (c64 + CSR P) on the perceive path
# ======================================================================
def part_csr():
    hdr("PART 3 -- compressed store: bytes, and what it costs perceive")
    import organism_compress as oc
    res = {}

    # Train a real organism at phase-23 shape so P has real corpus structure.
    emb_c, seq = make_corpus(S23)
    backend = "numba" if fastpath.HAVE_NUMBA else "numpy"
    if fastpath.HAVE_NUMBA:
        fastpath.warmup()
    org, t_tr, _ = time_perceive(S23, emb_c, seq, backend, 200_000, 1)
    live = int(org.used.sum())
    K = org.xi.shape[0]
    nnz = int((org.P != 0).sum())
    print(f"trained at phase-23 shape (200K tokens, {backend}): "
          f"{live}/{K} slots live, P {nnz}/{K * K} nonzero "
          f"({100 * nnz / (K * K):.2f}% dense)\n")

    dense_p = org.P.nbytes
    csr_p = nnz * (8 + 4) + (K + 1) * 4          # data f64 + indices i32 + indptr
    print(f"  transition matrix P at K={K}:")
    print(f"    dense float64 : {dense_p / 1024:>9.1f} KB")
    print(f"    CSR (floor 0) : {csr_p / 1024:>9.1f} KB   "
          f"{dense_p / max(csr_p, 1):.1f}x smaller, lossless")
    print(f"  xi store at K={K}, N={S23['DIM']}:")
    print(f"    complex128    : {org.xi.nbytes / 1024:>9.1f} KB")
    print(f"    complex64     : {org.xi.nbytes / 2048:>9.1f} KB")
    res.update(K=K, live=live, nnz=nnz, dense_p_kb=dense_p / 1024,
               csr_p_kb=csr_p / 1024)

    for spec_kw in (dict(xi_dtype=np.complex64, p_floor=0.0),):
        spec = oc.CompressionSpec(**spec_kw)
        gc.collect()
        t0 = time.perf_counter()
        cs = oc.compress(org, spec)
        t_c = time.perf_counter() - t0
        print(f"\n  compress({spec.label}): {t_c * 1e3:.1f} ms -> "
              f"{cs.nbytes / 1024:.1f} KB total store "
              f"(from {oc.store_bytes(org) / 1024:.1f} KB uncompressed, "
              f"{oc.store_bytes(org) / max(cs.nbytes, 1):.2f}x)")
        res["compress_ms"] = t_c * 1e3
        res["compressed_kb"] = cs.nbytes / 1024
        res["uncompressed_kb"] = oc.store_bytes(org) / 1024

        # What does perceiving FROM a compressed store cost? Organism.perceive
        # promotes a narrowed store to complex128 on entry and restores it on
        # exit (the T1.8 dtype guard) -- a per-call O(K*N) copy, plus P has to
        # be dense for the kernel's in-place row/column writes.
        # What does perceiving FROM a compressed store cost? `Organism.perceive`
        # promotes a narrowed store to complex128 on entry and restores it on
        # exit (the T1.8 dtype guard) -- an O(K*N) copy per CALL, not per frame.
        # Both arms start from the SAME state (the narrow arm's store is just
        # round-tripped through c64) and the two are INTERLEAVED across reps,
        # because this host's run-to-run spread is larger than the effect.
        def _fresh(narrow):
            o = PolysemyOrganism(N=S23["DIM"], K=K, omega=0.15, beta=10.0,
                                 seed=0, backend=backend)
            o.xi = (org.xi.astype(np.complex64) if narrow else org.xi.copy())
            o.used = org.used.copy()
            o.P = org.P.copy()
            return o

        def _one(narrow, n_tok=40_000):
            o = _fresh(narrow)
            gc.collect()
            t0 = time.perf_counter()
            o.perceive(stream_of(emb_c, seq, S23["HOLD"], n_tok), g_in=5.0,
                       dt=0.05, eta=0.02, recruit=S23["recruit"])
            return time.perf_counter() - t0

        _one(False, 2_000)
        wide, narrow = [], []
        for _ in range(5):
            wide.append(_one(False))
            narrow.append(_one(True))
        f = 40_000 * S23["HOLD"]
        mw, mn = float(np.median(wide)), float(np.median(narrow))
        print(f"  store dtype on entry: complex64 -> promoted, restored on exit "
              f"(the T1.8 guard); P must be dense for the kernel's in-place writes")
        print(f"  perceive {f // 1000}K frames, complex128 store: median {mw:.2f}s "
              f"({f / mw / 1e3:.1f}K frames/s)  reps "
              f"{' '.join(f'{t:.2f}' for t in sorted(wide))}")
        print(f"  perceive {f // 1000}K frames, complex64  store: median {mn:.2f}s "
              f"({f / mn / 1e3:.1f}K frames/s)  reps "
              f"{' '.join(f'{t:.2f}' for t in sorted(narrow))}")
        spread = (max(wide) - min(wide)) / mw
        print(f"  median difference {100 * (mn / mw - 1):+.1f}%, against a "
              f"within-arm spread of {100 * spread:.0f}% -- "
              f"{'INSIDE the noise' if abs(mn / mw - 1) < spread else 'outside the noise'}")
        res["perceive_wide_rate"] = f / mw
        res["perceive_narrow_rate"] = f / mn
        res["wide_reps"] = wide
        res["narrow_reps"] = narrow

    # consolidate + recall at this K, the other two hot paths
    print("\n--- the other two hot paths at this K")
    o2 = PolysemyOrganism(N=S23["DIM"], K=K, omega=0.15, beta=10.0, seed=0,
                          backend=backend)
    o2.xi, o2.used, o2.P = org.xi.copy(), org.used.copy(), org.P.copy()
    if hasattr(org, "count"):
        o2.count = org.count.copy()
    gc.collect()
    t0 = time.perf_counter()
    o2.consolidate()
    t_cons = time.perf_counter() - t0
    n_mem = o2.mem.shape[0] if getattr(o2, "mem", None) is not None else 0
    print(f"  consolidate() at K={K} ({live} live -> {n_mem} memories): "
          f"{t_cons * 1e3:.0f} ms")
    res["consolidate_ms"] = t_cons * 1e3
    if n_mem > 1:
        gc.collect()
        t0 = time.perf_counter()
        o2.recall(steps=20_000)
        t_rec = time.perf_counter() - t0
        print(f"  recall(20 000 steps) over {n_mem} memories: {t_rec:.2f}s "
              f"({20_000 / t_rec / 1e3:.1f}K steps/s)")
        res["recall_rate"] = 20_000 / t_rec
    OUT["csr"] = res


# ======================================================================
# PART 4 -- harness wall-clock, per section
# ======================================================================
def part_harness():
    hdr("PART 4 -- regression harness cost, per section (prediction c)")
    import regression_harness as rh

    sections = [(n, getattr(rh, n)) for n in dir(rh)
                if n.startswith("section_")]
    sections.sort(key=lambda kv: int(kv[0].split("_")[1]))
    backend = os.environ.get("DEFAI_BACKEND", "auto")
    print(f"backend: DEFAI_BACKEND={backend}\n")
    print(f"  {'section':<28}  {'checks':>6}  {'seconds':>8}  {'% total':>7}")
    rows = {}
    total0 = time.perf_counter()
    for name, fn in sections:
        gc.collect()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        rows[name] = dt
    total = time.perf_counter() - total0
    if rh.FAILURES:
        print(f"  !! {len(rh.FAILURES)} harness checks FAILED during timing: "
              f"{rh.FAILURES}")
    # count checks per section by re-reading the harness source (cheap, exact
    # enough for the tiering question: a section's check() call sites)
    src = open(rh.__file__).read()
    import re as _re
    for name, dt in rows.items():
        body = src.split(f"def {name}(")[1]
        body = body.split("\ndef ")[0]
        n_checks = len(_re.findall(r"\bcheck\(", body))
        print(f"  {name:<28}  {n_checks:>6}  {dt:>8.1f}  {100 * dt / total:>6.1f}%")
    print(f"  {'TOTAL':<28}  {'':>6}  {total:>8.1f}")
    OUT["harness"] = dict(total=total, sections=rows, backend=backend)


# ======================================================================
# PART 5 -- permutation-null cost at 5M scale (the E4 input)
# ======================================================================
def part_nulls():
    hdr("PART 5 -- permutation-null cost at 5M-word scale (E4 go/no-go input)")

    print("Re-creating phase 27's two null workloads at their true measured "
          "shapes.\nStage B: 11 k-values x 500 shuffles of a pair-space label "
          "array.\nStage C: one 500-shuffle null per candidate word, over that "
          "word's own\noccurrence pairs. Both are pure-numpy statistics over "
          "arrays whose sizes\ncome from the phase-27 run itself.\n")

    # ---- shapes from the phase-27 corpus, extracted exactly by part `shapes`
    stats_path = os.environ.get("P42_SHAPES", "/tmp/phase42_p27_shapes.json")
    if not os.path.exists(stats_path):
        print(f"  no phase-27 shape file at {stats_path} -- run the `shapes` "
              f"part first (needs the corpus + stage-A checkpoint)")
        return
    with open(stats_path) as f:
        d = json.load(f)
    n_pairs, word_ns, V_COV = d["n_pairs"], d["word_ns"], d["v_cov"]
    print(f"  phase-27 shapes (measured, not assumed): {n_pairs} "
          f"category-bigram pairs over {V_COV} covered words; "
          f"{len(word_ns)} candidate words with n>=100\n")

    K_CATS = 2                     # phase 27's selected k

    def _entropy(c):
        p = c[c > 0] / c.sum()
        return float(-(p * np.log2(p)).sum())

    rng = np.random.default_rng(7)

    # ---- stage B null: shuffle the per-WORD label vector (length V_COV),
    # then gather it through the pair-space index arrays (length n_pairs) --
    # phase 27's own construction, so the cost model is its cost, not a proxy.
    labels = rng.integers(0, K_CATS, size=V_COV)
    pred_arr = rng.integers(0, V_COV, size=n_pairs)
    succ_arr = rng.integers(0, V_COV, size=n_pairs)

    def cat_bigram(lab, k):
        lp, ls = lab[pred_arr], lab[succ_arr]
        return np.bincount(lp * k + ls, minlength=k * k).reshape(k, k).astype(float)

    def mutual_information(M):
        p = M / M.sum()
        pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = p * np.log2(p / (pi @ pj + 1e-300) + 1e-300)
        return float(term[p > 0].sum())

    n_rep = 100
    for _ in range(5):
        mutual_information(cat_bigram(rng.permutation(labels), K_CATS))
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        mutual_information(cat_bigram(rng.permutation(labels), K_CATS))
    per_b = (time.perf_counter() - t0) / n_rep
    b_total = per_b * 500 * 11
    print(f"  stage B: one MI null at {n_pairs} pairs = {per_b * 1e3:.1f} ms "
          f"-> 11 k x 500 shuffles = {b_total:.0f}s (modelled)")

    # The model is only used to ATTRIBUTE; the authoritative stage-B number is
    # the observed one, and the non-null part of stage B is measured directly
    # (k-means + silhouette at the real profile shape) so nulls come out by
    # subtraction rather than by extrapolation.
    from polysemy_organism import _kmeans_real, _silhouette_real
    prof = np.random.default_rng(0).standard_normal((V_COV, 2 * V_COV))
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9
    t_nonnull = 0.0
    for k in range(2, 13):
        t0 = time.perf_counter()
        lab_k, _ = _kmeans_real(prof, k, seed=3)
        _silhouette_real(prof, lab_k)
        t_nonnull += time.perf_counter() - t0
    print(f"  stage B: k-means + silhouette, k=2..12 at the real ({V_COV} x "
          f"{2 * V_COV}) profile shape = {t_nonnull:.1f}s NON-null")

    # ---- stage C null: per-word, over that word's occurrence pairs
    def cond_gain(pred, succ):
        H_u = _entropy(np.bincount(succ, minlength=K_CATS))
        n = len(succ)
        H_c = 0.0
        for pc in set(pred.tolist()):
            sub = succ[pred == pc]
            H_c += len(sub) / n * _entropy(np.bincount(sub, minlength=K_CATS))
        return max(H_u - H_c, 0.0)

    # cost is ~linear in n, so calibrate on a few sizes and sum over the real
    # per-word n distribution rather than re-running all 354 words
    cal = {}
    for n in (100, 1_000, 10_000, 100_000):
        pred = rng.integers(0, K_CATS, size=n)
        succ = rng.integers(0, K_CATS, size=n)
        reps = max(3, min(200, int(2e6 / n)))
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(reps):
            cond_gain(rng.permutation(pred), succ)
        cal[n] = (time.perf_counter() - t0) / reps
        print(f"  stage C: one shuffle+gain at n={n:>7} = "
              f"{cal[n] * 1e6:>9.1f} us")

    xs = np.array(sorted(cal))
    ys = np.array([cal[x] for x in xs])
    slope, intercept = np.polyfit(xs, ys, 1)
    c_total = sum(500 * (slope * n + intercept) for n in word_ns)
    print(f"  stage C: fitted {intercept * 1e6:.1f} us + {slope * 1e9:.2f} ns/occurrence")
    print(f"  stage C: 500 shuffles x {len(word_ns)} words "
          f"(sum n = {sum(word_ns)}) = {c_total:.0f}s")

    # --- the verdict, against the OBSERVED phase-27 stage wall-clock
    obs = dict(total=1184, stage_a_perceive=785, stage_a_coverage=105,
               stage_b=169, stage_c=102, stage_d=3, corpus_emb=20)
    b_nulls = obs["stage_b"] - t_nonnull
    c_nulls = obs["stage_c"]          # stage C is 500 nulls per 1 real gain
    nulls = b_nulls + c_nulls
    frac = nulls / obs["total"]
    print(f"\n  OBSERVED phase-27 wall-clock, this host, this run "
          f"(total {obs['total']}s):")
    for k, v in (("corpus load + PPMI/SVD embeddings", obs["corpus_emb"]),
                 ("stage A perceive (3 epochs)", obs["stage_a_perceive"]),
                 ("stage A consolidate + coverage_map", obs["stage_a_coverage"]),
                 ("stage B categories (incl. 5500 MI nulls)", obs["stage_b"]),
                 ("stage C per-word nulls (354 x 500)", obs["stage_c"]),
                 ("stage D generation", obs["stage_d"])):
        print(f"    {k:<44} {v:>5}s  {100 * v / obs['total']:>5.1f}%")
    print(f"\n  permutation-null work = {b_nulls:.0f}s (stage B, by subtraction) "
          f"+ {c_nulls}s (stage C) = {nulls:.0f}s = {100 * frac:.1f}% of total")
    ceiling = obs["total"] / (obs["total"] - nulls)
    print(f"  pre-registered rule: >= 50% -> E4 GO, < 50% -> E4 DEFER  ->  "
          f"{'GO' if frac >= 0.5 else 'DEFER'}")
    print(f"  Amdahl ceiling on TOTAL corpus-tier wall-clock with nulls made "
          f"FREE: {ceiling:.2f}x")
    allpairs = nulls + obs["stage_a_coverage"]
    print(f"  E4's full named scope (nulls + all-pairs similarity: "
          f"coverage_map's {obs['stage_a_coverage']}s) = "
          f"{100 * allpairs / obs['total']:.1f}% -> ceiling "
          f"{obs['total'] / (obs['total'] - allpairs):.2f}x")
    print(f"  the actual bill: stage A perceive, {obs['stage_a_perceive']}s = "
          f"{100 * obs['stage_a_perceive'] / obs['total']:.1f}% -- explicitly "
          f"NOT a GPU target (sequential over the stream)")
    OUT["nulls"] = dict(n_pairs=n_pairs, stage_b_modelled=b_total,
                        stage_b_nonnull=t_nonnull, stage_b_nulls=b_nulls,
                        stage_c_modelled=c_total, stage_c_nulls=c_nulls,
                        null_frac=frac, amdahl=ceiling, observed=obs,
                        n_words=len(word_ns), sum_n=int(sum(word_ns)))


# ======================================================================
# PART 0 -- exact phase-27 null shapes, from the corpus + stage-A checkpoint
# ======================================================================
def part_shapes():
    """Extract the two array shapes phase 27's permutation nulls actually run
    at, from the real corpus and the real stage-A organism -- so part 5's
    cost model is calibrated on measured sizes rather than estimates.

    Deliberately duplicates phase 27's corpus prep and `coverage_map` rather
    than importing them: `phase27_5m_word_scale_run.py` has no __main__ guard
    (importing it would re-run the whole 21-minute phase), and this script is
    not allowed to edit it."""
    hdr("PART 0 -- phase-27 permutation-null shapes (exact)")
    import glob
    import re
    from collections import Counter
    import organism_state

    corpus_dir = os.environ.get("P42_CORPUS", "/tmp/gutenberg_corpus_5m")
    ckpt = os.environ.get("P42_CKPT", "/tmp/phase27_stageA_checkpoint.npz")
    if not glob.glob(f"{corpus_dir}/*.txt") or not os.path.exists(ckpt):
        print(f"  need corpus at {corpus_dir} and stage-A checkpoint at {ckpt}")
        return

    g_start = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                         re.IGNORECASE | re.DOTALL)
    g_end = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
                       re.IGNORECASE | re.DOTALL)
    all_text = []
    for path in sorted(glob.glob(f"{corpus_dir}/*.txt")):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m_s, m_e = g_start.search(text), g_end.search(text)
        all_text.append(text[m_s.end() if m_s else 0:
                             m_e.start() if m_e else len(text)])
    raw_tokens = re.findall(r"[a-zA-Z']+", "\n".join(all_text).lower())
    MIN_COUNT = 1500
    counts = Counter(raw_tokens)
    vocab = sorted([w for w, c in counts.items() if c >= MIN_COUNT])
    w2i = {w: i for i, w in enumerate(vocab)}
    train_seq = [w2i[w] for w in raw_tokens if w in w2i]
    N_WORDS, DIM = len(vocab), 50
    print(f"  corpus: {len(raw_tokens)} raw, {len(train_seq)} in-vocab, "
          f"V={N_WORDS}")

    # embeddings: phase 27's recipe, needed only to re-run coverage_map
    WINDOW = 4
    cooc = np.zeros((N_WORDS, N_WORDS))
    for i, w in enumerate(train_seq):
        for j in range(max(0, i - WINDOW), min(len(train_seq), i + WINDOW + 1)):
            if j != i:
                cooc[w, train_seq[j]] += 1.0
    tot = cooc.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((cooc * tot) / (cooc.sum(1, keepdims=True)
                                     @ cooc.sum(0, keepdims=True) + 1e-12) + 1e-12)
    U, S, _ = np.linalg.svd(np.maximum(pmi, 0.0), full_matrices=False)
    DIM = min(50, U.shape[1])
    emb = U[:, :DIM] * np.sqrt(S[:DIM])
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    emb_c = emb.astype(complex)

    org = organism_state.load_state(ckpt, cls=PolysemyOrganism)
    t0 = time.perf_counter()
    org.consolidate(merge_thresh=0.84, prune_frac=0.0005)
    t_cons = time.perf_counter() - t0
    n_mem = org.mem.shape[0]
    train_arr = np.array(train_seq)
    assigns = np.empty(len(train_seq), dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(0, len(train_seq), 50_000):
        states = emb_c[train_arr[i:i + 50_000]]
        assigns[i:i + 50_000] = np.abs((org.mem.conj() @ states.T) / DIM).argmax(0)
    t_cov = time.perf_counter() - t0
    slot_word = {}
    for k in range(n_mem):
        members = train_arr[assigns == k]
        if len(members):
            slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
    covered = sorted(set(slot_word.values()))
    cidx = {w: i for i, w in enumerate(covered)}
    pair_seq = [cidx[w] for w in train_seq if w in cidx]
    n_pairs = len(pair_seq) - 1
    print(f"  stage A checkpoint: {n_mem} memories, coverage "
          f"{len(covered)}/{N_WORDS}   (consolidate {t_cons:.1f}s, "
          f"coverage_map {t_cov:.1f}s)")
    print(f"  stage B null runs on: {len(covered)}-element label vector "
          f"gathered through {n_pairs} pair-space indices")

    # stage C: per-word occurrence-pair counts, phase 27's own filter
    cov_set = set(covered)
    n_by_word = np.zeros(N_WORDS, dtype=np.int64)
    ts = train_arr
    inner = ts[1:-1]
    ok = np.isin(ts[:-2], list(cov_set)) & np.isin(ts[2:], list(cov_set))
    for w in range(N_WORDS):
        n_by_word[w] = int(((inner == w) & ok).sum())
    word_ns = [int(n) for n in n_by_word if n >= 100]
    print(f"  stage C null runs on: {len(word_ns)} words with n>=100, "
          f"sum n = {sum(word_ns)}, max n = {max(word_ns)}")

    dest = os.environ.get("P42_SHAPES", "/tmp/phase42_p27_shapes.json")
    with open(dest, "w") as f:
        json.dump(dict(n_pairs=n_pairs, v_cov=len(covered), word_ns=word_ns,
                       n_mem=n_mem, consolidate_s=t_cons, coverage_map_s=t_cov),
                  f)
    print(f"  -> {dest}")
    OUT["shapes"] = dict(n_pairs=n_pairs, v_cov=len(covered),
                         n_words=len(word_ns), sum_n=int(sum(word_ns)),
                         n_mem=n_mem, consolidate_s=t_cons, coverage_map_s=t_cov)


# ======================================================================
# PART 6 -- price the shortlist: measure the levers, don't estimate them
# ======================================================================
def part_levers():
    """Three candidate optimizations, measured rather than argued. Nothing
    here is APPLIED -- this phase does not touch a mechanism file; the point
    is to rank the shortlist by measured payoff so whoever takes the work has
    numbers to start from."""
    hdr("PART 6 -- shortlist levers, priced")
    res = {}

    # ---- lever 1: matvec operand width (the 75%-of-frame component)
    print("LEVER 1 -- overlap-matvec operand width at corpus K.")
    print("  The matvec is cache-resident at these shapes (a 1.2 MB operand), "
          "so its\n  rate tracks operand BYTES. Compute width is deliberately "
          "pinned at\n  complex128 by `Organism.perceive`'s dtype guard -- this "
          "prices what\n  narrowing it would buy, it does NOT propose narrowing "
          "it silently.\n")
    lev1 = {}
    for K, N in ((S23["K_CAP"], S23["DIM"]), (S27["K_CAP"], S27["DIM"])):
        rng = np.random.default_rng(1)
        A = np.ascontiguousarray(rng.standard_normal((K, N))
                                 + 1j * rng.standard_normal((K, N)))
        z = np.ascontiguousarray(rng.standard_normal(N)
                                 + 1j * rng.standard_normal(N))
        row = {}
        for dt, lab in ((np.complex128, "complex128"), (np.complex64, "complex64")):
            Ad, zd = A.astype(dt), z.astype(dt)
            out = np.empty(K, dtype=dt)
            for _ in range(500):
                np.dot(Ad, zd, out=out)
            gc.collect()
            t0 = time.perf_counter()
            for _ in range(30_000):
                np.dot(Ad, zd, out=out)
            r = 30_000 / (time.perf_counter() - t0)
            row[lab] = r
            print(f"  K={K} N={N} {lab:>10}: {r / 1e3:>7.1f}K matvec/s  "
                  f"({Ad.nbytes / 1024:.0f} KB operand)")
        print(f"  -> narrowing would buy {row['complex64'] / row['complex128']:.2f}x "
              f"on the matvec alone\n")
        lev1[K] = row
    res["matvec_width"] = {str(k): v for k, v in lev1.items()}

    # ---- lever 2: stage-B MI null by precomputed word-pair counts
    stats_path = os.environ.get("P42_SHAPES", "/tmp/phase42_p27_shapes.json")
    if not os.path.exists(stats_path):
        print("LEVER 2/3 skipped: no phase-27 shape file (run `shapes` first)")
        OUT["levers"] = res
        return
    with open(stats_path) as f:
        d = json.load(f)
    n_pairs, V_COV = d["n_pairs"], d["v_cov"]

    print("LEVER 2 -- stage-B MI null without touching the pair array.")
    print("  Each null re-labels WORDS, then counts category bigrams by "
          "gathering\n  through n_pairs indices -- O(n_pairs) per shuffle. But "
          "the corpus word-pair\n  counts W[i,j] never change across shuffles: "
          "with G the V x k one-hot of a\n  labelling, the category table is "
          "exactly G^T W G, which is O(V^2 k).\n  Same table, not an "
          "approximation of it.\n")
    k = 2
    rng = np.random.default_rng(7)
    pred = rng.integers(0, V_COV, size=n_pairs)
    succ = rng.integers(0, V_COV, size=n_pairs)
    labels = rng.integers(0, k, size=V_COV)

    def naive(lab):
        return np.bincount(lab[pred] * k + lab[succ],
                           minlength=k * k).reshape(k, k).astype(float)

    t0 = time.perf_counter()
    W = np.bincount(pred * V_COV + succ,
                    minlength=V_COV * V_COV).reshape(V_COV, V_COV).astype(float)
    t_pre = time.perf_counter() - t0

    def fast(lab):
        G = np.zeros((V_COV, k))
        G[np.arange(V_COV), lab] = 1.0
        return G.T @ W @ G

    print(f"  exactness: max |naive - W-matrix| over the contingency table = "
          f"{np.abs(naive(labels) - fast(labels)).max():.1f}")
    print(f"  one-off precompute of W ({V_COV} x {V_COV}): {t_pre * 1e3:.0f} ms")
    lev2 = {}
    for fn, lab, reps in ((naive, "naive (gather over n_pairs)", 20),
                          (fast, "W-matrix (V^2 k)", 2000)):
        for _ in range(3):
            fn(labels)
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(rng.permutation(labels))
        per = (time.perf_counter() - t0) / reps
        lev2[lab] = per
        print(f"  {lab:<30}: {per * 1e6:>9.1f} us/null -> 11 k x 500 = "
              f"{per * 5500:>7.2f}s")
    res["stage_b_lever"] = {k2: v for k2, v in lev2.items()}

    # ---- lever 3: stage-C per-word null in closed form
    print("\nLEVER 3 -- stage-C per-word null without shuffling the array.")
    print("  Permuting `pred` against a fixed `succ` re-partitions the succ "
          "multiset\n  into buckets whose SIZES are fixed by pred's marginal. "
          "The resulting\n  contingency table is therefore multivariate "
          "hypergeometric with those\n  margins -- so the null can be sampled "
          "in O(k^2), independent of n.\n  Verified below against the real "
          "shuffle at three phase-27 word sizes.\n")

    def _entropy(c):
        p = c[c > 0] / c.sum()
        return float(-(p * np.log2(p)).sum())

    def cond_gain(pr, su):
        H_u = _entropy(np.bincount(su, minlength=k))
        n = len(su)
        H_c = 0.0
        for pc in set(pr.tolist()):
            sub = su[pr == pc]
            H_c += len(sub) / n * _entropy(np.bincount(sub, minlength=k))
        return max(H_u - H_c, 0.0)

    def gain_from_table(T):
        n = T.sum()
        H_u = _entropy(T.sum(0))
        H_c = 0.0
        for r in range(T.shape[0]):
            if T[r].sum():
                H_c += T[r].sum() / n * _entropy(T[r])
        return max(H_u - H_c, 0.0)

    lev3 = {}
    rng = np.random.default_rng(3)
    for n in (3340, 37_495, 259_324):     # 'right', 'not', phase 27's largest
        pr = (rng.random(n) < 0.42).astype(int)
        su = (rng.random(n) < 0.61).astype(int)
        a = np.bincount(pr, minlength=k)
        b = np.bincount(su, minlength=k)
        NP = 4000
        t0 = time.perf_counter()
        perm = np.array([cond_gain(rng.permutation(pr), su) for _ in range(NP)])
        t_perm = (time.perf_counter() - t0) / NP
        t0 = time.perf_counter()
        hyp = np.empty(NP)
        for i in range(NP):
            rem = b.copy()
            T = np.empty((k, k))
            for r in range(k):
                draw = rng.multivariate_hypergeometric(rem.astype(int), int(a[r]))
                T[r] = draw
                rem = rem - draw
            hyp[i] = gain_from_table(T)
        t_hyp = (time.perf_counter() - t0) / NP
        print(f"  n={n:>6}: shuffle null p99={np.percentile(perm, 99):.6f} "
              f"mean={perm.mean():.6f} | hypergeom p99="
              f"{np.percentile(hyp, 99):.6f} mean={hyp.mean():.6f}")
        print(f"          per draw {t_perm * 1e6:>8.1f} us vs {t_hyp * 1e6:>6.1f} us "
              f"= {t_perm / t_hyp:>5.1f}x")
        lev3[n] = dict(perm_p99=float(np.percentile(perm, 99)),
                       hyp_p99=float(np.percentile(hyp, 99)),
                       speedup=t_perm / t_hyp, t_hyp_us=t_hyp * 1e6)
    res["stage_c_lever"] = {str(k2): v for k2, v in lev3.items()}
    OUT["levers"] = res


PARTS = dict(shapes=part_shapes, bench=part_bench, micro=part_micro,
             csr=part_csr, harness=part_harness, nulls=part_nulls,
             levers=part_levers)

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in PARTS] or list(PARTS)
    print("PHASE 42 -- performance re-baseline and profile (T6.5)")
    hostinfo()
    print(f"parts: {' '.join(want)}")
    t0 = time.time()
    for name in want:
        PARTS[name]()
    print(f"\ntotal phase-42 wall-clock: {time.time() - t0:.0f}s")
    dest = os.environ.get("P42_JSON")
    if dest:
        with open(dest, "w") as f:
            json.dump(OUT, f, indent=1, default=float)
        print(f"machine-readable results -> {dest}")
