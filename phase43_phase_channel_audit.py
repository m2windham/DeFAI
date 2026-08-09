"""
PHASE 43 (T7.1) -- PHASE-CHANNEL AUDIT AND THE STORAGE FORK.

WHAT THIS ASKS. T1.9 (phase 33h) measured the RELEASE-HOLD cost floor at
76.1 KB/ACC-pt = 2.24x the supervised prototype bar and recorded "nothing
left to engineer", decomposing the floor as

    2.34x bytes x (0.865/0.903) accuracy = 2.24x
    of which  2.03 x (112/115) = 1.98x   complex-vs-real, at parity count
              + 10.5KB / 29.4KB = 0.36x  the CSR transition graph

That verdict is established for LOSSLESS WIDTH NARROWING (33g took
complex128 -> complex64 at measured zero accuracy cost). It is NOT
established for a REPRESENTATION CHANGE. The 2.03x term says a field memory
is a COMPLEX N-vector where a prototype is a REAL one. This phase asks
whether the stored complex vectors actually USE their imaginary channel,
and if they do not, what the store costs when it stops paying for one.

THE MEASUREMENT. For a slot vector v, the imaginary energy that survives
the best global de-rotation is

    R(v) = min_theta ||Im(e^{-i theta} v)||^2 / ||v||^2
         = (1 - |sum_i v_i^2| / sum_i |v_i|^2) / 2                (closed form)

R = 0 exactly when v is a global rotation of a real vector (the
(real N-vector, phase scalar) layout is then EXACT); R = 0.5 for an
isotropic complex vector (the layout throws half the energy away).
`organism_compress.real_phase_residual` computes it; the `real_phase`
CompressionSpec lever stores the layout.

PROVENANCE OF THE HYPOTHESES. An exploratory sandbox session (2026-08-08,
no branch, no repo changes, phase 19/20 recipe) reported median residual
1e-4 with 416/418 slots below 1e-3, word->slot assignment agreement 1.0000
and overlap-profile correlation 0.9996 under substitution. Those are
INDICATIVE ANCHORS, not results. Where a measurement here disagrees, the
measurement wins and the disagreement is reported as a finding.

======================================================================
PRE-REGISTRATION (committed before the run that tests it)
======================================================================

P1 -- RESIDUAL AT THE 33h CONFIGURATION. At K=112, N=64, omega=0.15,
g_in=4.0, dt=0.05, hold=8 (the arm the cost floor was measured on), the
median R over live slots is < 1e-3 and at least 95% of live slots are
below 1e-2, on seeds 0-4 and on held-out 5-9. The DISTRIBUTION is
reported, not only the median.
  MUNDANE ACCOUNT (M1): the residual is at floor only because the inputs
  are real. A pipeline with genuinely complex inputs would use the width,
  and the saving is an artifact of one embedding choice rather than a
  property of the mechanism. Section 2c is built to fire this.

P2 -- THE MECHANISTIC ACCOUNT, AND WE EXPECT TO FALSIFY THE STRONG FORM.
The work order's account is that `dz = 1j*omega*z + g_in*(x - z)` is a
content-blind global rotation, so for real inputs the reachable set is
EXACTLY {e^{i theta} . real} and the emptiness is STRUCTURAL. Deriving the
loop instead of describing it: with normalize() only rescaling by a real
factor, the direction obeys the linear recursion

    z_t ~ a z_{t-1} + b x_t ,   a = 1 - g_in*dt + i*omega*dt ,  b = g_in*dt

so z_t ~ sum_j b a^j x_{t-j} -- a PHASOR SUM whose per-lag coefficients
have DIFFERENT arguments. The reachable set is therefore NOT {rotation of
real}; it is the complex span of recent inputs, and it collapses to
{rotation of real} only in two limits: omega -> 0 (a real) and full
settling within a token (hold -> infinity drives z -> b/(1-a) . x, a global
scalar times one real vector). PREDICTION: R is a TRANSIENT, controlled by
both knobs. R(omega=0) is at machine floor; R rises monotonically with
omega and monotonically as `hold` falls; somewhere inside omega <= 8 at
hold=8 the residual leaves the floor (> 1e-2). A two-token closed form
  u = a^hold . x_prev + (1 - a^hold) . x
predicts the measured R across the whole (omega, hold) grid up to the
EMA averaging that slot refinement applies over visits.
  DECISION RULE: if R stays below 1e-3 everywhere in omega in [0, 8] the
  strong structural account SURVIVES and the emptiness is a property of the
  mechanism. If R exceeds 1e-2 anywhere in that range the strong account is
  FALSIFIED: the channel is empty because omega/g_in << 1 and settling is
  deep -- a PARAMETER REGIME, not a structure. Either way the storage fork
  is unaffected; what changes is what branch (A) actually forfeits.

P3 -- SUBSTITUTION IS BEHAVIOR-NEUTRAL AT THE 33h ARM. Replacing the store
with its de-rotated real part costs |dACC| <= 0.005 eval-only and <= 0.010
in store mode (quantize at every task boundary and carry it forward),
paired seeds 0-4, confirmed on held-out 5-9. 33e's pre-registered claim
threshold is 0.02; anything at or beyond that is a REAL cost and branch (A)
is not free.
  MUNDANE ACCOUNT (M3): `LabelEvidenceReadout` scores `np.abs(...)` of
  overlaps, so the readout is PHASE-BLIND by construction and a null dACC
  is guaranteed rather than earned. POSITIVE CONTROL: run the identical
  substitution on a store trained at high omega, where section 2b says the
  residual is large. If dACC is still ~0 there, the metric cannot see the
  imaginary channel and P3's null is uninformative -- that outcome must be
  reported as loudly as a pass.

P4 -- GAUGE. Rotating every slot by an independent random phase leaves
every reported metric unchanged (|dACC| = 0 exactly, both backends). If it
does, the per-slot phase is GAUGE, the 4-byte scalar buys E3 bitwise
round-trip rather than behavior, and the layout could in principle drop to
256 B/slot.

P5 -- THE LAYOUT ARITHMETIC, DERIVED BEFORE IT IS MEASURED (33h's process
rule). At N=64: complex64 slot 512 B + 8 B float32 count/age = 520 B vs a
prototype's 256 B = 2.031x. Real+phase: 256 + 4 + 8 = 268 B = 1.047x.
Carrying 33h's other two terms unchanged,
    memory term 1.047 x (112/115) = 1.020
    graph term  10.46KB / 29.44KB = 0.355
    byte ratio  1.375  x  (0.865/0.903) = 1.32x KB/ACC-pt
PREDICTION: the measured floor lands at 1.32x +- 0.02, the <=2x fallback is
MET, and parity (1.0x) is still NOT met -- parity needs a byte ratio <=
1.044 and the memory term alone is 1.020, leaving <= 0.7KB for a graph that
costs 10.5KB. This DISAGREES with the work order's sandbox arithmetic of
"~1.02x and the total -> ~1.13x": 1.02x is the memory term only, and the
sandbox total appears to omit or re-price the graph term that 33h's
decomposition keeps. If the measurement disagrees with THIS derivation,
the measurement wins and the derivation is reported as wrong.

WHAT THIS PHASE DOES NOT DO. It does not choose. The fork is mutually
exclusive -- (A) spend the width and forfeit the imaginary channel
permanently, (B) keep it and make it load-bearing -- and section 6 prices
both and hands the decision up.
"""

import copy
import time

import numpy as np

import organism_compress as oc
import phase33e_readout_geometry as p33e
import phase33h_cost_frontier as p33h
import text_recipe as tr
from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_compress import CompressionSpec, compress, real_phase_residual
from organism_numba import NumbaOrganism
from phase33c_gate_retest import EVICT, N, NORM, TASKS, X

SEEDS = (0, 1, 2, 3, 4)
SEEDS_HELD = (5, 6, 7, 8, 9)

K_FLOOR = 112                    # 33h's floor arm
DECODER = ("calib-b8", dict(decoder="calib", m=None, beta=8.0))
SPEC = p33h.SPEC                                        # 33g headline: c64+CSR
SPEC_RP = CompressionSpec(xi_dtype=np.complex64, p_floor=0.0,
                          meta_dtype=np.float32, real_phase=True)

# the 33h/33c perceive settings, named so the sweep can vary them
G_IN, DT, ETA, RECRUIT, OMEGA_0, HOLD_0 = 4.0, 0.05, 0.015, 0.6, 0.15, 8

CLAIM_THRESH = p33e.CLAIM_THRESH                        # 0.02


def q(a, name="R"):
    """A distribution, not a mean: the report P1 asks for."""
    a = np.asarray(a, float)
    if a.size == 0:
        return f"{name}: (no live slots)"
    p = np.percentile(a, [0, 25, 50, 75, 95, 100])
    return (f"{name}: n={a.size}  min {p[0]:.3e}  q25 {p[1]:.3e}  "
            f"median {p[2]:.3e}  q75 {p[3]:.3e}  p95 {p[4]:.3e}  max {p[5]:.3e}  "
            f"| frac<1e-3 {np.mean(a < 1e-3):.4f}  frac<1e-2 {np.mean(a < 1e-2):.4f}")


# ---------------------------------------------------------------------------
# training helpers -- 33h's stream construction, with omega/hold exposed
# ---------------------------------------------------------------------------

def train_digits(seed=0, K=K_FLOOR, omega=OMEGA_0, hold=HOLD_0, epochs=3,
                 evict=EVICT, imag=0.0, rng_imag=None):
    """33h's `run_arm` training loop verbatim, with `omega`/`hold` exposed and
    an optional GENUINELY COMPLEX input stream (section 2c).

    imag > 0 replaces each sample x by (x + i*imag*x') / |.| where x' is an
    independent random rotation of the sample space -- a stream whose frames
    are not global rotations of real vectors, which is the input class M1
    says would use the width.
    """
    Xtr, ytr, Xte, yte = p33h.build_split(seed)
    r = np.random.default_rng(seed)
    r.permutation(len(X))                    # burn the split draw (33b)
    org = NumbaOrganism(N=N, K=K, omega=omega, beta=10.0, seed=seed)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    Q = None
    if imag > 0:
        rg = np.random.default_rng(12345 if rng_imag is None else rng_imag)
        Q, _ = np.linalg.qr(rg.standard_normal((N, N)))
    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in r.permutation(idx):
                v = Xtr[i].astype(complex)
                if Q is not None:
                    v = v + 1j * imag * (Q @ Xtr[i])
                seq.extend([normalize(v, NORM)] * hold)
        org.perceive(seq, g_in=G_IN, eta=ETA, recruit=RECRUIT, evict=evict)
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        readout.invalidate(fresh)
        readout.observe(org, Xtr[idx], ytr[idx])
    return org, readout, (Xte, yte)


def score(org, readout, Xte, yte, kw):
    """Final-row accuracy under one decoder -- prediction is non-mutating."""
    row = p33e.eval_tasks(lambda Xe: readout.predict(org, Xe, **kw), Xte, yte)
    return float(np.mean(row))


def residual_of(org):
    return real_phase_residual(org.xi, org.used)


# ---------------------------------------------------------------------------
# section 2b: the two-token closed form the recursion predicts
# ---------------------------------------------------------------------------

def predicted_residual(omega, hold, g_in=G_IN, dt=DT, n_pairs=400, seed=0):
    """R of u = a^hold . x_prev + (1 - a^hold) . x over real sample pairs.

    Derivation: normalize() rescales by a REAL factor, so the DIRECTION of z
    obeys z_t ~ a z_{t-1} + b x_t with a = 1 - g_in*dt + i*omega*dt. Settled
    on one token, z ~ (b/(1-a)) x -- a global scalar times a real vector, so
    R = 0. `hold` frames after a token change, z ~ (b/(1-a)) [a^hold x_prev +
    (1-a^hold) x]; the leading scalar is global and drops out of R, leaving
    the bracket. This is the residual BEFORE slot refinement averages it.
    """
    a = (1.0 - g_in * dt) + 1j * (omega * dt)
    ah = a ** hold
    rg = np.random.default_rng(seed)
    i = rg.integers(0, len(X), n_pairs)
    j = rg.integers(0, len(X), n_pairs)
    u = ah * X[i].astype(complex) + (1.0 - ah) * X[j].astype(complex)
    return float(np.median(real_phase_residual(u)))


# ---------------------------------------------------------------------------
# static census: does ANY pipeline in this repo feed complex-valued frames?
# ---------------------------------------------------------------------------

def call_site_census():
    """Every `perceive(` call site and whether its stream is real-cast.

    Grep-level and deliberately crude -- the point is the denominator, not a
    parser. Reported so section 2c's synthetic stream is not mistaken for
    something a committed pipeline produces.
    """
    import glob
    import re
    rows = []
    for path in sorted(glob.glob("*.py")):
        src = open(path, encoding="utf-8").read()
        if ".perceive(" not in src:
            continue
        astype_complex = len(re.findall(r"astype\(complex\)|astype\(np\.complex", src))
        embed_complex = len(re.findall(r"embeddings?\s*=\s*\w+\.astype\(complex\)", src))
        genuinely = len(re.findall(r"1j\s*\*\s*(?!self\.omega|omega|0\.15)", src))
        rows.append((path, src.count(".perceive("), astype_complex + embed_complex,
                     genuinely))
    return rows


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    print(__doc__.split("PRE-REGISTRATION")[0].strip()[:0] or "", end="")
    print("PHASE 43 (T7.1): phase-channel audit and the storage fork\n")
    print(f"  protocol: 33h floor arm K={K_FLOOR}, N={N}, omega={OMEGA_0}, "
          f"g_in={G_IN}, dt={DT}, hold={HOLD_0}, evict={EVICT}, decoder={DECODER[0]}")

    # ---------------- 0. the derivation, BEFORE any measurement -----------
    print("\n" + "=" * 70)
    print("(0) LAYOUT ARITHMETIC, DERIVED FROM THE LAYOUT (P5)")
    proto_b = N * 4
    c64_b = N * 8 + 8
    rp_b = N * 4 + 4 + 8
    print(f"  prototype slot           {proto_b:4d} B  (real float32 N={N})")
    print(f"  c64 field memory         {c64_b:4d} B  (complex64 + 8B float32 count/age)"
          f"   = {c64_b / proto_b:.3f}x")
    print(f"  real+phase field memory  {rp_b:4d} B  (float32 N + float32 phase + 8B)"
          f"   = {rp_b / proto_b:.3f}x")
    print(f"  memory term  {rp_b / proto_b:.3f} x (112/115) = "
          f"{rp_b / proto_b * 112 / 115:.3f}")
    print("  graph and accuracy terms are carried from 33h and re-measured in (4)")

    # ---------------- 1. residual at the 33h configuration ----------------
    print("\n" + "=" * 70)
    print("(1) RESIDUAL AT THE 33h FLOOR ARM -- P1")
    res_by_seed, orgs = {}, {}
    for s in SEEDS + SEEDS_HELD:
        t0 = time.time()
        org, ro, (Xte, yte) = train_digits(seed=s)
        r = residual_of(org)
        res_by_seed[s] = r
        orgs[s] = (org, ro, Xte, yte)
        tag = "held-out" if s in SEEDS_HELD else "selection"
        print(f"  seed {s} ({tag:9s}) live {int(org.used.sum()):3d}  "
              f"{q(r)}   [{time.time() - t0:.1f}s]")
    pooled = np.concatenate([res_by_seed[s] for s in SEEDS])
    pooled_h = np.concatenate([res_by_seed[s] for s in SEEDS_HELD])
    print(f"\n  POOLED s=0-4  {q(pooled)}")
    print(f"  POOLED s=5-9  {q(pooled_h)}")
    p1 = (np.median(pooled) < 1e-3 and np.mean(pooled < 1e-2) >= 0.95
          and np.median(pooled_h) < 1e-3 and np.mean(pooled_h < 1e-2) >= 0.95)
    print(f"  P1 {'HELD' if p1 else 'MISSED'}")

    # ---------------- 2a. a different corpus ------------------------------
    print("\n" + "=" * 70)
    print("(2a) A DIFFERENT CORPUS -- the phase 19/20 text recipe")
    from organism import Organism
    for name, loader, rec in (("fables (phase 19)", tr.load_fables, tr.RECIPE_19),
                              ("gutenberg8 (phase 20)", tr.load_gutenberg8, tr.RECIPE_20)):
        t0 = time.time()
        R = tr.Recipe(loader(), **rec, verbose=False)
        K_txt = min(1200, len(R.vocab) * 4)
        o = Organism(N=R.N, K=K_txt, omega=OMEGA_0, beta=10.0, seed=0, backend="numba")
        o.perceive(R.stream(tr.HOLD_19), **tr.PERCEIVE_19)
        r = real_phase_residual(o.xi, o.used)
        print(f"  {name:22s} V={len(R.vocab):4d} N={R.N} K={K_txt} "
              f"live {int(o.used.sum()):4d}")
        print(f"      {q(r)}   [{time.time() - t0:.1f}s]")

    # ---------------- 2b. omega x hold sweep ------------------------------
    print("\n" + "=" * 70)
    print("(2b) OMEGA x HOLD SWEEP -- P2's decision rule (digits, seed 0, K=112)")
    OMEGAS = (0.0, 0.05, 0.15, 0.5, 1.0, 2.0, 4.0, 8.0)
    HOLDS = (8, 4, 2, 1)
    print(f"  {'omega':>6} | " + " | ".join(f"hold={h}: measured / predicted" for h in HOLDS))
    sweep = {}
    for om in OMEGAS:
        cells = []
        for h in HOLDS:
            o, _, _ = train_digits(seed=0, omega=om, hold=h)
            m = float(np.median(residual_of(o)))
            pr = predicted_residual(om, h)
            sweep[(om, h)] = (m, pr)
            cells.append(f"{m:.2e} / {pr:.2e}")
        print(f"  {om:6.2f} | " + " | ".join(cells))
    r0 = sweep[(0.0, HOLD_0)][0]
    rmax = max(v[0] for k, v in sweep.items() if k[1] == HOLD_0)
    print(f"\n  R(omega=0, hold=8) = {r0:.3e}   max over omega at hold=8 = {rmax:.3e}")
    print(f"  P2 strong structural account: "
          f"{'SURVIVES' if rmax < 1e-3 else 'FALSIFIED (R leaves the floor inside omega<=8)'}")

    # ---------------- 2c. genuinely complex inputs (M1) -------------------
    print("\n" + "=" * 70)
    print("(2c) GENUINELY COMPLEX INPUTS -- the mundane account M1")
    print("  static census of every perceive() call site in the repo:")
    for path, ncalls, ncast, ngen in call_site_census():
        print(f"    {path:42s} perceive x{ncalls}  real->complex casts {ncast}"
              f"  other 1j uses {ngen}")
    print("  (no committed pipeline builds a complex-valued STREAM; the arms below"
          "\n   construct one so M1 can be tested rather than assumed)")
    bar_c = {}
    for imag in (0.0, 0.25, 1.0):
        o, ro, (Xte, yte) = train_digits(seed=0, imag=imag)
        r = residual_of(o)
        acc = score(o, ro, Xte, yte, DECODER[1])
        o2 = copy.deepcopy(o)
        compress(o2, spec=SPEC_RP).apply(o2)
        acc_rp = score(o2, ro, Xte, yte, DECODER[1])
        bar_c[imag] = (float(np.median(r)), acc, acc_rp)
        print(f"  imag={imag:<5} live {int(o.used.sum()):3d}  median R {np.median(r):.3e}"
              f"  ACC {acc:.4f} -> real+phase {acc_rp:.4f}  (d {acc_rp - acc:+.4f})")

    # ---------------- 3. gauge --------------------------------------------
    print("\n" + "=" * 70)
    print("(3) IS THE PER-SLOT PHASE GAUGE? -- P4")
    org, ro, Xte, yte = orgs[0]
    base = score(org, ro, Xte, yte, DECODER[1])
    g = copy.deepcopy(org)
    rg = np.random.default_rng(7)
    g.xi = np.ascontiguousarray(g.xi * np.exp(1j * rg.uniform(0, 2 * np.pi, g.K))[:, None])
    rot = score(g, ro, Xte, yte, DECODER[1])
    print(f"  ACC {base:.6f} -> random per-slot rotation {rot:.6f}  "
          f"(d {rot - base:+.2e})")
    print(f"  P4 {'HELD -- the per-slot phase is gauge for the scored readout' if rot == base else 'MISSED -- the phase is read somewhere'}")

    # ---------------- 4. the floor, measured ------------------------------
    print("\n" + "=" * 70)
    print("(4) THE FLOOR RE-DERIVED WITH THE REAL+PHASE LAYOUT -- P5")
    rows = []
    for s in SEEDS + SEEDS_HELD:
        org, ro, Xte, yte = orgs[s]
        b_c64 = p33h.live_bytes(org, spec=SPEC)
        b_rp = p33h.live_bytes(org, spec=SPEC_RP)
        st_c64 = compress(p33h.StoreView(org), spec=SPEC)
        st_rp = compress(p33h.StoreView(org), spec=SPEC_RP)
        bar = p33h.run_prototypes_seed(s)
        acc = score(org, ro, Xte, yte, DECODER[1])
        rows.append(dict(seed=s, live=int(org.used.sum()), acc=acc,
                         b_c64=b_c64, b_rp=b_rp, bar_acc=bar["acc"],
                         bar_b=bar["bytes"], bar_n=bar["n"],
                         xi_c64=st_c64.xi_bytes, xi_rp=st_rp.xi_bytes,
                         graph=st_c64.p_bytes, meta=st_c64.meta_bytes))
    for r_ in rows:
        tag = "held" if r_["seed"] in SEEDS_HELD else "sel "
        print(f"  s{r_['seed']} {tag} live {r_['live']:3d} ACC {r_['acc']:.4f}  "
              f"c64 {r_['b_c64'] / 1e3:6.2f}KB ({p33h.kbpp(r_['b_c64'], r_['acc']):6.2f}) "
              f"| rp {r_['b_rp'] / 1e3:6.2f}KB ({p33h.kbpp(r_['b_rp'], r_['acc']):6.2f}) "
              f"| bar {r_['bar_n']:3d} protos {r_['bar_b'] / 1e3:5.2f}KB "
              f"ACC {r_['bar_acc']:.4f} ({p33h.kbpp(r_['bar_b'], r_['bar_acc']):6.2f})")

    def floor(sel, key):
        rr = [x for x in rows if (x["seed"] in SEEDS_HELD) == sel]
        org_k = np.mean([p33h.kbpp(x[key], x["acc"]) for x in rr])
        bar_k = np.mean([p33h.kbpp(x["bar_b"], x["bar_acc"]) for x in rr])
        return org_k, bar_k, org_k / bar_k

    for sel, name in ((False, "selection s=0-4"), (True, "HELD-OUT s=5-9")):
        ok, bk, ratio = floor(sel, "b_c64")
        ok2, bk2, ratio2 = floor(sel, "b_rp")
        print(f"\n  {name}: c64+CSR {ok:6.2f} vs bar {bk:5.2f} = {ratio:.3f}x   "
              f"| real+phase {ok2:6.2f} = {ratio2:.3f}x")
    print("\n  term decomposition on the held-out seeds (33h's form):")
    rr = [x for x in rows if x["seed"] in SEEDS_HELD]
    for key, lbl in (("b_c64", "c64+CSR"), ("b_rp", "real+phase")):
        xi_k = "xi_c64" if key == "b_c64" else "xi_rp"
        mem = np.mean([(x[xi_k] + x["meta"]) / x["bar_b"] for x in rr])
        gr = np.mean([x["graph"] / x["bar_b"] for x in rr])
        accf = np.mean([x["bar_acc"] / x["acc"] for x in rr])
        print(f"    {lbl:11s} memory {mem:.3f} + graph {gr:.3f} = byte ratio "
              f"{mem + gr:.3f}  x accuracy {accf:.3f}  = {(mem + gr) * accf:.3f}x")

    # ---------------- 5. behavior under substitution ----------------------
    print("\n" + "=" * 70)
    print("(5) BEHAVIOR UNDER SUBSTITUTION -- P3 (eval-only and store mode)")
    print("  eval-only: substitute the finished store, re-score (paired by construction)")
    ev = {}
    for s in SEEDS + SEEDS_HELD:
        org, ro, Xte, yte = orgs[s]
        a0 = score(org, ro, Xte, yte, DECODER[1])
        o2 = copy.deepcopy(org)
        compress(o2, spec=SPEC_RP).apply(o2)
        a1 = score(o2, ro, Xte, yte, DECODER[1])
        ev[s] = (a0, a1)
        print(f"    s{s} {'held' if s in SEEDS_HELD else 'sel '}  "
              f"{a0:.4f} -> {a1:.4f}   d {a1 - a0:+.4f}")
    d_sel = np.array([ev[s][1] - ev[s][0] for s in SEEDS])
    d_held = np.array([ev[s][1] - ev[s][0] for s in SEEDS_HELD])
    print(f"    eval-only delta: sel mean {d_sel.mean():+.4f} "
          f"[{d_sel.min():+.4f}, {d_sel.max():+.4f}]  | held mean {d_held.mean():+.4f} "
          f"[{d_held.min():+.4f}, {d_held.max():+.4f}]")

    print("\n  store mode: quantize at every task boundary and carry it forward")
    st = {}
    for s in SEEDS + SEEDS_HELD:
        a_c64 = p33h.run_arm(K_FLOOR, seed=s, spec=SPEC,
                             decoders=(DECODER,))["per_decoder"][DECODER[0]]["acc"]
        a_rp = p33h.run_arm(K_FLOOR, seed=s, spec=SPEC_RP,
                            decoders=(DECODER,))["per_decoder"][DECODER[0]]["acc"]
        st[s] = (a_c64, a_rp)
        print(f"    s{s} {'held' if s in SEEDS_HELD else 'sel '}  "
              f"{a_c64:.4f} -> {a_rp:.4f}   d {a_rp - a_c64:+.4f}")
    s_sel = np.array([st[s][1] - st[s][0] for s in SEEDS])
    s_held = np.array([st[s][1] - st[s][0] for s in SEEDS_HELD])
    print(f"    store-mode delta: sel mean {s_sel.mean():+.4f} "
          f"[{s_sel.min():+.4f}, {s_sel.max():+.4f}]  | held mean {s_held.mean():+.4f} "
          f"[{s_held.min():+.4f}, {s_held.max():+.4f}]")
    p3 = (np.abs(d_held).max() <= 0.005 and np.abs(s_held).max() <= 0.010)
    print(f"  P3 {'HELD' if p3 else 'MISSED'} (band: eval-only <=0.005, store <=0.010 "
          f"on held-out seeds; 33e claim threshold {CLAIM_THRESH})")
    print(f"  M3 positive control (from 2c): imag=1.0 median R "
          f"{bar_c[1.0][0]:.3e}, ACC {bar_c[1.0][1]:.4f} -> {bar_c[1.0][2]:.4f} "
          f"(d {bar_c[1.0][2] - bar_c[1.0][1]:+.4f})")
    print("    if that delta is ~0 too, the readout cannot see the imaginary channel"
          "\n    and P3's null is uninformative rather than evidence of emptiness.")

    # ---------------- 6. the fork -----------------------------------------
    print("\n" + "=" * 70)
    print("(6) THE FORK -- priced, not chosen")
    ok_c64, bk, ratio_c64 = floor(True, "b_c64")
    ok_rp, _, ratio_rp = floor(True, "b_rp")
    print(f"  (A) SPEND IT   store {ok_rp:.2f} KB/ACC-pt = {ratio_rp:.2f}x the bar "
          f"(<=2x fallback {'MET' if ratio_rp <= 2 else 'NOT met'}); "
          f"forfeits the imaginary channel")
    print(f"  (B) CASH IT IN store {ok_c64:.2f} KB/ACC-pt = {ratio_c64:.2f}x the bar; "
          f"33h's floor stands and the gate re-scope is forced")
    print(f"\nTOTAL {time.time() - t_all:.1f}s")
