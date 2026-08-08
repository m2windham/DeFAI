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
     parts: bench micro csr harness nulls   (default: all)
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
                  symbols=False, org=None):
    """Wall-clock one perceive load. Returns (organism, seconds, frames)."""
    K = K or shape["K_CAP"]
    if org is None:
        org = PolysemyOrganism(N=shape["DIM"], K=K, omega=0.15, beta=10.0,
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
        print("--- T3.3 / T1.8 tax check (phase-23 shape, 100K tokens, numba)")
        emb_c, seq = make_corpus(S23)
        _, t_off, f_off = time_perceive(S23, emb_c, seq, "numba", 100_000, 1)
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
        fr = OUT.get("bench", {}).get(shape["name"], {}).get("numba_rate")
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
        org_c = PolysemyOrganism(N=S23["DIM"], K=K, omega=0.15, beta=10.0,
                                 seed=0, backend=backend)
        cs.apply(org_c)
        print(f"  store after apply(): xi {org_c.xi.dtype}, P {org_c.P.dtype} "
              f"{'dense' if isinstance(org_c.P, np.ndarray) else type(org_c.P).__name__}")
        gc.collect()
        t0 = time.perf_counter()
        org_c.perceive(stream_of(emb_c, seq, S23["HOLD"], 40_000), g_in=5.0,
                       dt=0.05, eta=0.02, recruit=S23["recruit"])
        t_narrow = time.perf_counter() - t0
        org_w = PolysemyOrganism(N=S23["DIM"], K=K, omega=0.15, beta=10.0,
                                 seed=0, backend=backend)
        org_w.xi = org.xi.copy()
        org_w.used = org.used.copy()
        org_w.P = org.P.copy()
        gc.collect()
        t0 = time.perf_counter()
        org_w.perceive(stream_of(emb_c, seq, S23["HOLD"], 40_000), g_in=5.0,
                       dt=0.05, eta=0.02, recruit=S23["recruit"])
        t_wide = time.perf_counter() - t0
        f = 40_000 * S23["HOLD"]
        print(f"  perceive 160K frames from a complex128 store: {t_wide:.2f}s "
              f"({f / t_wide / 1e3:.1f}K frames/s)")
        print(f"  perceive 160K frames from a complex64  store: {t_narrow:.2f}s "
              f"({f / t_narrow / 1e3:.1f}K frames/s)  "
              f"tax {100 * (t_narrow / t_wide - 1):+.1f}%")
        res["perceive_wide_rate"] = f / t_wide
        res["perceive_narrow_rate"] = f / t_narrow

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
        n0 = len(rh.FAILURES)
        c0 = getattr(rh, "_N_CHECKS", None)
        buf = []
        real_print = __builtins__["print"] if isinstance(__builtins__, dict) \
            else __builtins__.print
        gc.collect()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        rows[name] = dt
        del n0, c0, buf, real_print
    total = time.perf_counter() - total0
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

    # ---- shapes from the phase-27 corpus (recomputed here, not assumed)
    n_pairs = int(os.environ.get("P42_NPAIRS", "0"))
    word_ns = None
    stats_path = "/tmp/phase42_p27_shapes.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            d = json.load(f)
        n_pairs, word_ns = d["n_pairs"], d["word_ns"]
        print(f"  using phase-27 shapes recorded at {stats_path}: "
              f"{n_pairs} category-bigram pairs, "
              f"{len(word_ns)} candidate words with n>=100\n")
    if not n_pairs:
        print("  (no phase-27 shape file; run tools/p42_shapes first)")
        return

    K_CATS = 2                     # phase 27's selected k

    def _entropy(c):
        p = c[c > 0] / c.sum()
        return float(-(p * np.log2(p)).sum())

    rng = np.random.default_rng(7)

    # ---- stage B null: one MI over the full pair-space array
    labels = rng.integers(0, K_CATS, size=n_pairs)
    pred_arr = rng.integers(0, len(labels), size=n_pairs)
    succ_arr = rng.integers(0, len(labels), size=n_pairs)

    def cat_bigram(lab, k):
        lp, ls = lab[pred_arr], lab[succ_arr]
        return np.bincount(lp * k + ls, minlength=k * k).reshape(k, k).astype(float)

    def mutual_information(M):
        p = M / M.sum()
        pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = p * np.log2(p / (pi @ pj + 1e-300) + 1e-300)
        return float(term[p > 0].sum())

    n_rep = 20
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        mutual_information(cat_bigram(rng.permutation(labels), K_CATS))
    per_b = (time.perf_counter() - t0) / n_rep
    b_total = per_b * 500 * 11
    print(f"  stage B: one MI null at {n_pairs} pairs = {per_b * 1e3:.1f} ms "
          f"-> 11 k x 500 shuffles = {b_total:.0f}s")

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

    OUT["nulls"] = dict(n_pairs=n_pairs, stage_b_s=b_total, stage_c_s=c_total,
                        per_b_ms=per_b * 1e3, n_words=len(word_ns),
                        sum_n=int(sum(word_ns)))
    print(f"\n  permutation-null total (B + C) = {b_total + c_total:.0f}s")
    print("  -> compared against the phase-27 run's total wall-clock in the "
          "verdict below.")


PARTS = dict(bench=part_bench, micro=part_micro, csr=part_csr,
             harness=part_harness, nulls=part_nulls)

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
