"""
PHASE 62 -- CAPACITY OR KERNEL? THE INFINITE-CAPACITY CEILING OF THE PHASE FIELD.

What-if (SOP rule 23) from phase 61, REVISED BEFORE REGISTRATION.

The first what-if was: phase 61 failed for lack of BACKOFF (its product
kernel is exact-match-or-nothing), so give oscillators heterogeneous fades
and the readout becomes a sum of products -- backoff for free. The
development check (selection seed 0 of the control, before registration)
refuted the premise: a SINGLE fade with a moderate kick already is soft
backoff (lam = 0.2, amp = 3: a context matching only the last word keeps
~25% weight), that configuration was in phase 61's grid, and it still scored
WORSE than the pure bigram on real text (dev 5.854 vs 5.787). On the control
the two-channel mixture lost to the best single fade (3.115 vs 2.961 nats)
because splitting the oscillators halves each channel's capacity. My
phase 61 post-hoc diagnosis ('no backoff') was therefore at least partly
wrong -- the SOP rule 22 failure, committed by the rule's author the day it
was written. Recorded rather than quietly redesigned around.

THE LIVE QUESTION. Two explanations remain for phase 61's P2/P2b failure:
  CAPACITY -- a sharper (deeper) kernel lowers each stored context's signal
    against cross-talk, so depth cannot pay at N = 1024; or
  KERNEL   -- the field's kernel family cannot use depth on real text at any
    capacity.
They are separable exactly. With random kicks the EXPECTED overlap of two
states is
    E Re<z_t, z_s>/N = prod_k [ 1 if the words at position k agree
                                else rho_k ],   rho_k = sinc^2(amp lam^k)
(sinc(x) = sin(pi x)/(pi x); validated before registration against an
N = 16384 field: correlation 0.98-0.99 per pair, identical means). Expanding
the product, the infinite-N readout score is a FIXED-WEIGHT mixture of
skip-gram counts:
    S_w(ctx) = sum_M  W_M * c_M(ctx, w),
    W_M = prod_{k in M} (1 - rho_k) * prod_{k not in M} rho_k,
over subsets M of history positions, c_M the count of training contexts
agreeing with ctx on the positions in M and followed by w. At lam = 0 it is
exactly the Dirichlet bigram. Positions 0-3 are counted exactly; deeper
positions enter as the constant factor prod_{k>=4} rho_k (a scale absorbed
by alpha). A mixture of fades (the original what-if) is the fraction-
weighted sum of its channels' W -- also computable exactly, with no
capacity cost. So this phase tests the ORIGINAL what-if too, at infinite
capacity, where the development objection (halved channels) cannot apply.

GRID (exhaustive -- evaluation is a dot product): SINGLE lam in
{0, 0.2, 0.35, 0.5, 0.7} x amp in {1, 3, 10}; MIX every pair of distinct
SINGLE configurations with fraction f in {0.25, 0.5, 0.75}; alpha in
{1 ... 10000}. Selected on DEV, scored once on TEST. The infinite-N field
has no random seed, so there is nothing to reseed; the capacity measurement
below does use held-out seeds 5-9.

PREDICTIONS
  P1 (ceiling vs KN2, PREDICTED FAIL): the best infinite-N configuration
     beats KN2 on test. Reason in advance: fixed-weight interpolation
     typically trails Kneser-Ney, and this family's bigram limit (Dirichlet
     bigram) is 0.12 nats behind KN2 on test.
  P2 (does depth pay at infinite capacity -- decides CAPACITY vs KERNEL):
     the best infinite-N configuration with depth (anything but lam = 0)
     beats the infinite-N bigram on test by >= 0.02 nats.
     PASS -> phase 61's P2b failure was CAPACITY. FAIL -> KERNEL.
  P3 (the original what-if, capacity removed): the best infinite-N MIX beats
     the best infinite-N SINGLE on test by >= 0.01 nats.
  CAPACITY COST: the frozen best configuration realised at N = 1024
     (alpha/tau re-selected on dev at seed 0), test at seeds 5-9, reported
     as nats lost to finite capacity.

READOUT CONTROL (SOP rule 20) and VALIDATION, both on the mixed-order
synthetic source (V = 200; next = T2[a,b] p 0.4, T1[b] p 0.4, uniform 0.2;
200k training tokens over 40,000 pairs), seeds 5-9:
  (a) the best infinite-N depth configuration beats the infinite-N bigram
      by >= 0.30 nats on every seed (the source carries ~0.5 nats of
      second-order information: KN3 2.66 vs KN2 3.15 on selection seed 0);
  (b) on seed 5, the frozen control configuration realised at N = 16384
      scores within 0.05 nats of its infinite-N value, and not better.
  Either failing makes the run VOID.

KILL RULE. P2 fails -> the phase-field kernel family cannot use history
depth on real text at any capacity; the phase-kernel line (61-62) is closed.
P2 passes and P1 fails -> depth is real but the family's ceiling is below
KN2; the remaining gap is the smoothing form (context-adaptive discounting),
which is Kneser-Ney itself -- built before.

DEVELOPMENT RECORD (before registration; selection seed 0 and dev only):
  - the refuted first design is recorded above;
  - analytic kernel vs an N = 16384 field: per-pair correlation 0.98-0.99;
  - lam = 0 at infinite N reproduces the Dirichlet bigram on dev to 1.2e-14;
  - control seed 0: KN2 3.148, KN3 2.658, infinite-N bigram 3.167, best
    infinite-N depth configuration 2.697 (a MIX), gain +0.469.

RESULT (run 2026-09-24, after pre-registration a52103c). RUN VOID.
  VALIDATION (b) FAILED: the control's frozen configuration realised at
    N = 16384 scored 3.321 against its infinite-N 2.666 (+0.655; the bar was
    [0, 0.05]). Per the registration, the run is VOID and P1-P3 carry no
    verdict. Readout control (a) PASSED 5/5 (infinite-N depth gains
    +0.456 to +0.480 nats over the infinite-N bigram).
  MY DESIGN ERROR, recorded as such and not used to rescue the run: (b)
    required N = 16384 to sit within 0.05 of the infinite-N limit, which
    assumes capacity converges by 16k -- the very thing under test. It
    conflated 'is the formula right' (already answered by the per-pair
    kernel check, corr 0.98-0.99) with 'is capacity nearly solved at 16k'.
    The answer to the second is plainly no, and that is the finding.
  NUMBERS, reported without a verdict:
    real-text test, infinite-N: best 5.5423 (a MIX), best SINGLE 5.5462,
      bigram 5.6313; KN2 5.5086, KN3 5.4476. Depth over the family's own
      bigram +0.0890 (more than KN's own KN2->KN3 gain, 0.061); the best
      configuration still 0.034 behind KN2; mixing fades adds +0.0039.
    capacity cost of that configuration at N = 1024, seeds 5-9:
      +0.913 to +0.915 nats (6.456-6.457 vs 5.5423).
  WHAT-IF LOG (SOP rule 23). PROPERTY exposed: storing counts in
    superposition costs ~0.9 nats at N = 1024 and ~0.65 nats at N = 16384;
    cross-talk, not the kernel, dominates -- phase 61's 'no backoff'
    diagnosis was wrong. QUESTION raised: how does the capacity cost fall
    with N, and does any practical N reach the kernel's ceiling? Even that
    ceiling is below KN2 and is, in exact form, fixed-weight interpolated
    skip-gram counting (Jelinek-Mercer family) -- built before.

PRIOR ART: fixed-weight interpolation is Jelinek-Mercer smoothing; skip-gram
counts are standard; the kernel is fractional power encoding (Frady et al.
2021).
"""

import sys
import time
import numpy as np

from phase61_word_as_move import (load_books, split_books, build_vocab, encode, NGram,
                                  eval_positions, ngram_nll, Field, KN_DS, TAUS,
                                  SEL_SEED, HELD_OUT, N_OSC, DEV_EVAL, TEST_EVAL)

ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
SINGLE = [(lam, amp) for lam in (0.0, 0.2, 0.35, 0.5, 0.7) for amp in (1.0, 3.0, 10.0)]
FRACS = [0.25, 0.5, 0.75]
KPOS = 4                                   # history positions counted exactly
MASKS = list(range(1 << KPOS))             # subsets M of {0..3}; bit k = position k
HASH = np.uint64(0x9E3779B97F4A7C15)


# ======================================================= analytic kernel
def rho(lam, amp, kmax=24):
    return np.array([np.sinc(amp * lam ** k) ** 2 if k == 0 or lam > 0 else 1.0
                     for k in range(kmax)])


def weights(lam, amp):
    """W_M for every subset mask M of positions 0..3, with the deeper
    positions' expected factor folded in as a constant."""
    r = rho(lam, amp)
    tail = float(np.prod(r[KPOS:]))
    W = np.empty(len(MASKS))
    for M in MASKS:
        w = tail
        for k in range(KPOS):
            w *= (1 - r[k]) if (M >> k) & 1 else r[k]
        W[M] = w
    return W


# =================================================== skip-gram counting
def contexts(parts, V):
    """(ctx words at positions 0..3, next word) for every predicted token;
    position k = the word k+1 back; V is the padding id."""
    C, Y = [], []
    for s in parts:
        pad = np.concatenate([np.full(KPOS, V), s])
        n = len(s)
        C.append(np.stack([pad[KPOS - 1 - k: KPOS - 1 - k + n] for k in range(KPOS)], 1)[1:])
        Y.append(s[1:])
    return np.concatenate(C), np.concatenate(Y)


def keys(ctx, M):
    key = np.zeros(len(ctx), dtype=np.uint64)
    for k in range(KPOS):
        if (M >> k) & 1:
            key |= (ctx[:, k].astype(np.uint64) + np.uint64(1)) << np.uint64(15 * k)
    return key


class SkipCounts:
    def __init__(self, train, V):
        ctx, y = contexts(train, V)
        self.tot = len(y)
        self.uni = np.bincount(y, minlength=V + 1).astype(float)
        self.tabs = {}
        for M in MASKS[1:]:
            k = keys(ctx, M)
            j = k * HASH + (y.astype(np.uint64) + np.uint64(1))
            ku, kc = np.unique(k, return_counts=True)
            ju, jc = np.unique(j, return_counts=True)
            self.tabs[M] = (ku, kc, ju, jc)

    @staticmethod
    def _look(u, c, q):
        i = np.searchsorted(u, q); i = np.minimum(i, len(u) - 1)
        return np.where(u[i] == q, c[i], 0).astype(float)

    def matrices(self, parts, pos, V):
        """Per scored position: counts c_M(ctx, target) and c_M(ctx) for all M."""
        ctx_all, y_all = [], []
        for k, i in pos:
            s = parts[k]
            ctx_all.append([s[i - 1 - j] if i - 1 - j >= 0 else V for j in range(KPOS)])
            y_all.append(s[i])
        ctx = np.array(ctx_all, dtype=np.int64); y = np.array(y_all, dtype=np.int64)
        Ct = np.zeros((len(y), len(MASKS))); Cc = np.zeros((len(y), len(MASKS)))
        Ct[:, 0] = self.uni[y]; Cc[:, 0] = self.tot
        for M in MASKS[1:]:
            ku, kc, ju, jc = self.tabs[M]
            k = keys(ctx, M)
            Cc[:, M] = self._look(ku, kc, k)
            Ct[:, M] = self._look(ju, jc, k * HASH + (y.astype(np.uint64) + np.uint64(1)))
        return Ct, Cc, y


def inf_nll(Ct, Cc, y, p_uni, W, alpha):
    St, Sc = Ct @ W, Cc @ W
    return float(-np.mean(np.log((St + alpha * p_uni[y]) / (Sc + alpha))))


def configs():
    out = [("single", (c,), (1.0,)) for c in SINGLE]
    for i, a in enumerate(SINGLE):
        for b in SINGLE[i + 1:]:
            for f in FRACS:
                out.append(("mix", (a, b), (f, 1 - f)))
    return out


def W_of(cfg):
    _, parts, fr = cfg
    return sum(f * weights(*p) for f, p in zip(fr, parts))


def has_depth(cfg):
    return any(lam > 0 for lam, _ in cfg[1])


def select_inf(Ct, Cc, y, p_uni, pool):
    best = None
    for cfg in pool:
        W = W_of(cfg)
        for a in ALPHAS:
            d = inf_nll(Ct, Cc, y, p_uni, W, a)
            if best is None or d < best[0]:
                best = (d, cfg, a)
    return best


# ============================================== finite-N realisation
class ChannelField(Field):
    """A finite-N field realising a configuration: channel c holds fraction
    f_c of the oscillators with its own (lam, amp); history rings within
    each channel (block permutation)."""

    def __init__(self, V, N, cfg, seed):
        _, parts, fr = cfg
        rng = np.random.default_rng(10_000 + seed)
        sizes = [int(round(f * N)) for f in fr]; sizes[-1] = N - sum(sizes[:-1])
        lam = np.concatenate([np.full(n, p[0]) for n, p in zip(sizes, parts)])
        amp = np.concatenate([np.full(n, p[1]) for n, p in zip(sizes, parts)])
        perm, lo = [], 0
        for n in sizes:
            perm.append(lo + rng.permutation(n)); lo += n
        self.th = (rng.uniform(0, 2 * np.pi, (V, N)) * amp[None, :]).astype(np.float32)
        self.perm = np.concatenate(perm)
        self.N, self.V, self.lam = N, V, lam.astype(np.float32)
        self.C = np.zeros((V, N), dtype=np.complex64); self.n = np.zeros(V)


def finite(V, cfg, seed, train, dev, dpos, test, tpos, p_uni, N):
    f = ChannelField(V, N, cfg, seed); f.learn(train)
    if dev is not None:
        st, tg = f.stats(dev, dpos, TAUS)
        _, a, t = min((Field.nll(st, tg, p_uni, a, t), a, t) for a in ALPHAS for t in TAUS)
    else:
        a, t = None, None
    return f, a, t


# ============================================================ control
def control_stream(n, rng, T1, T2):
    V = len(T1)
    s = np.empty(n, dtype=np.int64); s[:2] = rng.integers(0, V, 2)
    for t in range(2, n):
        u = rng.random()
        s[t] = T2[s[t - 2], s[t - 1]] if u < 0.4 else (T1[s[t - 1]] if u < 0.8 else rng.integers(0, V))
    return s


def control(seed, validate=False, log=print):
    rng = np.random.default_rng(4242 + seed)
    V = 200
    T1 = rng.integers(0, V, V); T2 = rng.integers(0, V, (V, V))
    tr = [control_stream(200_000, rng, T1, T2)]
    dv = [control_stream(4_000, rng, T1, T2)]
    te = [control_stream(20_000, rng, T1, T2)]
    ng = NGram(tr, V)
    dpos, tpos = eval_positions(dv, 4_000), eval_positions(te, 20_000)
    sc = SkipCounts(tr, V)
    Dt, Dc, dy = sc.matrices(dv, dpos, V); Tt, Tc, ty = sc.matrices(te, tpos, V)
    pool = configs()
    bd = select_inf(Dt, Dc, dy, ng.p_uni, [c for c in pool if has_depth(c)])
    bb = select_inf(Dt, Dc, dy, ng.p_uni, [c for c in pool if not has_depth(c)])
    deep = inf_nll(Tt, Tc, ty, ng.p_uni, W_of(bd[1]), bd[2])
    flat = inf_nll(Tt, Tc, ty, ng.p_uni, W_of(bb[1]), bb[2])
    D = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn2", d))
    D3 = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn3", d))
    out = dict(deep=deep, flat=flat, kn2=ngram_nll(ng, te, tpos, "kn2", D),
               kn3=ngram_nll(ng, te, tpos, "kn3", D3), cfg=bd[1])
    log(f"  control seed {seed}: KN2={out['kn2']:.3f} KN3={out['kn3']:.3f}  inf-N bigram={flat:.3f}  "
        f"inf-N depth={deep:.3f} {bd[1]} alpha={bd[2]}  gain={flat - deep:+.3f}", flush=True)
    if validate:
        f = ChannelField(V, 16384, bd[1], seed); f.learn(tr)
        st, tg = f.stats(te, tpos, [0.0])
        fin = Field.nll(st, tg, ng.p_uni, bd[2], 0.0)
        out["finite"] = fin
        log(f"  VALIDATION seed {seed}: N=16384 realisation {fin:.3f} vs inf-N {deep:.3f} "
            f"(diff {fin - deep:+.3f}; must be in [0, 0.05])", flush=True)
    return out


# ================================================================ main
def main(stage):
    print("PHASE 62 -- capacity or kernel? the infinite-capacity ceiling\n", flush=True)
    if stage in ("control", "all"):
        print("READOUT CONTROL + VALIDATION -- mixed-order source")
        res = [control(s, validate=(s == HELD_OUT[0])) for s in HELD_OUT]
        ok_a = all(r["flat"] - r["deep"] >= 0.30 for r in res)
        v = res[0]["finite"] - res[0]["deep"]
        ok_b = -0.005 <= v <= 0.05
        print(f"  CONTROL (a) {'PASS' if ok_a else 'FAIL'}  VALIDATION (b) {'PASS' if ok_b else 'FAIL'}"
              f"  -> {'run valid' if ok_a and ok_b else 'RUN VOID'}\n", flush=True)

    if stage in ("text", "all"):
        t0 = time.time()
        books = load_books()
        tr, dv, te = split_books(books)
        w2i = build_vocab(tr); V = len(w2i)
        tr, dv, te = encode(tr, w2i), encode(dv, w2i), encode(te, w2i)
        ng = NGram(tr, V)
        dpos, tpos = eval_positions(dv, DEV_EVAL), eval_positions(te, TEST_EVAL)
        sc = SkipCounts(tr, V)
        Dt, Dc, dy = sc.matrices(dv, dpos, V); Tt, Tc, ty = sc.matrices(te, tpos, V)
        # sanity: lam = 0 at infinite N must equal the Dirichlet bigram exactly
        W0 = weights(0.0, 1.0)
        chk = abs(inf_nll(Dt, Dc, dy, ng.p_uni, W0, 300) - ngram_nll(ng, dv, dpos, "dir2", 300))
        print(f"REAL TEXT -- V={V}  [{time.time() - t0:.0f}s]  identity check (dev) lam=0 vs Dirichlet2: "
              f"|diff|={chk:.2e}", flush=True)
        D = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn2", d))
        D3 = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn3", d))
        kn2 = ngram_nll(ng, te, tpos, "kn2", D); kn3 = ngram_nll(ng, te, tpos, "kn3", D3)
        pool = configs()
        best = select_inf(Dt, Dc, dy, ng.p_uni, pool)
        bd = select_inf(Dt, Dc, dy, ng.p_uni, [c for c in pool if has_depth(c)])
        bb = select_inf(Dt, Dc, dy, ng.p_uni, [c for c in pool if not has_depth(c)])
        bs = select_inf(Dt, Dc, dy, ng.p_uni, [c for c in pool if c[0] == "single"])
        bm = select_inf(Dt, Dc, dy, ng.p_uni, [c for c in pool if c[0] == "mix"])
        T = lambda b: inf_nll(Tt, Tc, ty, ng.p_uni, W_of(b[1]), b[2])
        for tag, b in (("BEST", best), ("DEPTH", bd), ("BIGRAM", bb), ("SINGLE", bs), ("MIX", bm)):
            print(f"  {tag:<6} {b[1]} alpha={b[2]}  dev {b[0]:.4f}  test {T(b):.4f}", flush=True)
        tb, td, tf, ts, tm = T(best), T(bd), T(bb), T(bs), T(bm)
        print(f"\n  TEST ({len(tpos)} tokens) KN2={kn2:.4f} KN3(ref)={kn3:.4f}")
        print(f"  P1 inf-N best beats KN2:           {kn2 - tb:+.4f} -> {'PASS' if tb < kn2 else 'FAIL (predicted)'}")
        print(f"  P2 inf-N depth beats inf-N bigram: {tf - td:+.4f} -> "
              f"{'PASS (capacity)' if tf - td >= 0.02 else 'FAIL (kernel)'}")
        print(f"  P3 inf-N MIX beats inf-N SINGLE:   {ts - tm:+.4f} -> {'PASS' if ts - tm >= 0.01 else 'FAIL'}",
              flush=True)
        print("\n  CAPACITY COST -- frozen best realised at N=1024 (alpha/tau re-selected on dev, seed 0)")
        _, a, t = finite(V, best[1], SEL_SEED, tr, dv, dpos, None, None, ng.p_uni, N_OSC)
        for s in HELD_OUT:
            f = ChannelField(V, N_OSC, best[1], s); f.learn(tr)
            st, tg = f.stats(te, tpos, [t])
            fin = Field.nll(st, tg, ng.p_uni, a, t)
            print(f"    seed {s}: N=1024 {fin:.4f} vs inf-N {tb:.4f}  capacity cost {fin - tb:+.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
