"""
PHASE 63 -- SEPARATE THE STRONG MEMORIES, SUPERPOSE ONLY THE REST.

What-if (SOP rule 23) from phase 62. PROPERTY: storing every context in one
superposition costs ~0.9 nats at N = 1024 (phase 62), and cross-talk grows
with the SQUARE of each context type's count, so a few frequent contexts
should dominate it. Measured on the train split before registration: the
top 1,000 two-word context types hold 25% of positions but 94% of the
sum of squared counts (the cross-talk variance proxy); the top 10,000 hold
50% and 99%. QUESTION: keep the strong memories separately (exact counts)
and superpose only the long tail -- does the capacity wall fall, and does
the superposed tail then earn its keep?

MECHANISM. Explicit store: the top-B two-word context types (a, b) =
(word two back, last word) by training count, with exact next-word counts.
Superposed store: a phase field at N = 1024 whose configuration is SELECTED
AT FINITE N (see the development record) from {single fade lam in
{0, 0.2, 0.35, 0.5} x amp in {1, 3}} plus phase 62's infinite-N best mix,
on dev at B = 10,000, seed 0, then frozen for every B -- trained ONLY on
positions whose context is NOT in the explicit store (the field still runs
over the whole stream, so its state carries full history). Score for word w:
  S_w = S_explicit(w) + S_field(w)
  S_explicit(w) = sum over explicit types (a', b) sharing the last word b of
                  weight(a' == a) * n(a', b, w)
with the kernel's expected weight per channel c,
  weight(match)    = prod_{k>=2} rho_ck,
  weight(mismatch) = rho_c1 * prod_{k>=2} rho_ck,   rho_ck = sinc^2(amp_c lam_c^k),
mixed by channel fraction (types not sharing the last word weigh
sinc^2(amp) = 0 for integer amp). Positions >= 2 are treated as mismatched
for explicit types -- an approximation, stated. S_field is phase 61/62's
readout with its measured row-norm null. p(w) = (S_w^+ + alpha p_uni(w)) /
(sum + alpha).

ARMS, at each B in {1,000; 10,000; 100,000}:
  H   explicit top-B + superposed tail (the what-if).
  EB  explicit top-B + the tail stored as EXACT BIGRAM counts, weighted as
      mismatched contexts -- the cheap exact alternative for the tail.
  E0  explicit top-B only -- what the tail adds at all.
References: the infinite-N field (phase 62: 5.5423 test), KN2, KN3, and the
pure superposed field (phase 62: 6.456 test at N = 1024).
MEMORY LEDGER (floats; indexes not counted): explicit = 2 x distinct
(a, b, w) entries; field = 2 x V x N; bigram tail = 2 x distinct tail
bigrams; KN2 = 2 x distinct bigrams + V.

Selection: alpha (and tau for H) on DEV at seed 0 per arm and B; TEST once,
H at held-out seeds 5-9 (EB and E0 are deterministic).

PREDICTIONS
  P1 (mechanism): at B = 10,000, H's capacity cost -- H minus EB3, the
     same model with the same tail stored EXACTLY -- is < 0.10 nats on 5/5
     seeds, against phase 62's 0.91 for storing everything in superposition:
     the strong memories carried most of the cross-talk. (H minus the
     infinite-N 5.5423 is reported alongside.)
     This check ASSUMES (rule 24) that EB3 is the exact-storage counterpart
     of H -- same explicit store, same kernel weights, deep positions
     treated as mismatched in both; it does not assume the superposed tail
     behaves, which is what it measures.
  P2 (does superposition earn its keep; PREDICTED FAIL, reason in advance):
     H beats EB at the same B, 5/5 seeds, at B = 10,000. Reason: the tail is
     rare contexts (count 1-2), exactly where superposed storage is weakest
     relative to cross-talk, while an exact bigram tail costs ~40x fewer
     floats than the field. This check ASSUMES both arms share the explicit
     store and kernel weights exactly, so any difference is the tail's
     storage alone -- which is the claim, not an assumption about it.
  P3 (reference, predicted FAIL by construction): H beats KN2. H cannot
     exceed the infinite-N ceiling (5.5423), which is already behind KN2
     (5.5086).
  M1 (mundane account for any P2 pass): H's gain over EB comes from deeper
     (trigram-and-beyond) information in the tail, which EB lacks by design,
     not from superposition as such. Check: EB3 -- EB with the tail's exact
     TRIGRAM counts added at the match weight -- must NOT beat H; if it does,
     exact storage of the same information wins and M1 explains P2.

READOUT CONTROL (rule 20). On phase 62's mixed-order synthetic source,
seeds 5-9, B = 1,000, configuration selected on the control's dev the same
way: H must recover >= 40% of the EXACT tail's gain, (E0 - H) >= 0.4 x
(E0 - EB3), on every seed -- the superposed tail is readable where
cross-talk is low (V = 200). This check ASSUMES exact storage of the same
tail under the same kernel weights (EB3) bounds what the tail can deliver;
it does not assume superposition approaches that bound on real text, which
is P2's question.

DEVELOPMENT RECORD (before registration; control selection seed 0, dev only):
  - The first control bar (H beats E0 by >= 0.20) presumed a fact about the
    source that is false: on it even the exact bigram tail adds only ~0.07
    over E0. Rule 24, broken minutes after it was written; the bar is now
    relative to exact storage of the same tail.
  - PROPERTY found: phase 62's infinite-N best configuration is useless at
    finite N. Its kernel is so selective that an exact two-word match
    weighs ~0.024 (mismatch ~0.001) while cross-talk per row is O(1): the
    superposed tail added +0.002 over E0. Selectivity and signal-to-cross-
    talk trade against each other -- a configuration chosen where capacity
    is free is the worst one where it is not. Hence finite-N selection.
  - Control dev, seed 0 (E0 / EB / EB3 / H): lam 0.2 amp 3 -> 3.083 / 3.007
    / 2.946 / 3.002 (H recovers 59% of the exact tail's gain); lam 0.5 amp
    1 -> 3.133 / 3.064 / 3.026 / 3.061; phase 62 mix -> 3.001 / 2.922 /
    2.797 / 2.999 (0%).

KILL RULE. P2 fails -> superposition does not earn its keep even for the
long tail; the phase-field-as-memory line (61-63) closes, and the move/phase
kernel survives only as a similarity weighting over explicit counts, which
is known smoothing.
"""

import sys
import time
import numpy as np
from scipy import sparse

from phase61_word_as_move import (load_books, split_books, build_vocab, encode, NGram,
                                  eval_positions, ngram_nll, Field, KN_DS, TAUS,
                                  SEL_SEED, HELD_OUT, N_OSC, DEV_EVAL, TEST_EVAL)
from phase62_heterogeneous_fades import ChannelField, rho, control_stream

ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
CONFIGS = [("single", ((lam, amp),), (1.0,)) for lam in (0.0, 0.2, 0.35, 0.5) for amp in (1.0, 3.0)]
CONFIGS += [("mix", ((0.5, 3.0), (0.7, 3.0)), (0.5, 0.5))]      # phase 62's infinite-N best
BS = [1_000, 10_000, 100_000]
INF_N = 5.5423                                        # phase 62, test


def kernel_weights(cfg):
    """(match, mismatch) weight for an explicit two-word type sharing the
    last word, deeper positions treated as mismatched."""
    _, parts, fr = cfg
    wm = wx = 0.0
    for f, (lam, amp) in zip(fr, parts):
        r = rho(lam, amp)
        tail = float(np.prod(r[2:]))
        wm += f * tail; wx += f * r[1] * tail
    return wm, wx


def triples(parts, V):
    """(a, b, y) for every predicted position with >= 2 words of history,
    plus (part, index) so the field can be masked."""
    A, Bw, Y, P, I = [], [], [], [], []
    for k, s in enumerate(parts):
        A.append(s[:-2]); Bw.append(s[1:-1]); Y.append(s[2:])
        P.append(np.full(len(s) - 2, k)); I.append(np.arange(2, len(s)))
    return tuple(np.concatenate(x) for x in (A, Bw, Y, P, I))


class Store:
    """Explicit top-B two-word types + tail statistics."""

    def __init__(self, train, V, B):
        a, b, y, self.part, self.idx = triples(train, V)
        V1 = V + 1
        ctx = a * V1 + b
        u, c = np.unique(ctx, return_counts=True)
        top = u[np.argsort(-c, kind="stable")[:B]]
        self.is_exp = np.isin(ctx, top)
        e, t = self.is_exp, ~self.is_exp
        self.V = V
        # compact context ids (unseen contexts map to an all-zero row)
        self.cu = u
        cid = np.searchsorted(u, ctx)
        R = len(u) + 1
        # explicit: n(a,b,w), n(a,b), and per-last-word sums ce(b,w), ce(b)
        self.tri = sparse.csr_matrix((np.ones(e.sum()), (cid[e], y[e])), shape=(R, V1))
        self.tri_tot = np.asarray(self.tri.sum(1)).ravel()
        self.ce = sparse.csr_matrix((np.ones(e.sum()), (b[e], y[e])), shape=(V1, V1))
        self.ce_tot = np.asarray(self.ce.sum(1)).ravel()
        # tail as exact bigram (EB) and exact trigram (EB3, the M1 check)
        self.ct = sparse.csr_matrix((np.ones(t.sum()), (b[t], y[t])), shape=(V1, V1))
        self.ct_tot = np.asarray(self.ct.sum(1)).ravel()
        self.tt = sparse.csr_matrix((np.ones(t.sum()), (cid[t], y[t])), shape=(R, V1))
        self.tt_tot = np.asarray(self.tt.sum(1)).ravel()
        self.mem_exp = 2 * self.tri.nnz
        self.mem_tailbg = 2 * self.ct.nnz
        self.mem_tailtg = 2 * self.tt.nnz

    def row(self, ctx):
        i = np.minimum(np.searchsorted(self.cu, ctx), len(self.cu) - 1)
        return np.where(self.cu[i] == ctx, i, len(self.cu))

    def scores(self, parts, pos, wm, wx):
        """Per scored position: (target, total) score of explicit part, exact
        bigram tail, exact trigram-tail increment."""
        V1 = self.V + 1
        a = np.array([parts[k][i - 2] if i >= 2 else self.V for k, i in pos])
        b = np.array([parts[k][i - 1] for k, i in pos])
        y = np.array([parts[k][i] for k, i in pos])
        ctx = self.row(a * V1 + b)
        g = lambda M, r, c: np.asarray(M[r, c]).ravel()
        exp_t = wx * g(self.ce, b, y) + (wm - wx) * g(self.tri, ctx, y)
        exp_c = wx * self.ce_tot[b] + (wm - wx) * self.tri_tot[ctx]
        bg_t, bg_c = wx * g(self.ct, b, y), wx * self.ct_tot[b]
        tg_t, tg_c = (wm - wx) * g(self.tt, ctx, y), (wm - wx) * self.tt_tot[ctx]
        return dict(exp=(exp_t, exp_c), bg=(bg_t, bg_c), tg=(tg_t, tg_c)), y


def nll(num, den, y, p_uni, alpha):
    return float(-np.mean(np.log((num + alpha * p_uni[y]) / (den + alpha))))


def learn_tail(f, parts, store, chunk=20_000):
    """Phase 61's learn, but only positions whose context is NOT explicit
    are written; the field still reads the whole stream."""
    keep = [np.ones(len(s), bool) for s in parts]
    for k in range(len(parts)):
        sel = store.is_exp & (store.part == k)
        keep[k][store.idx[sel]] = False
    for k, s in enumerate(parts):
        th = None
        for lo in range(0, len(s) - 1, chunk):
            hi = min(lo + chunk, len(s) - 1)
            Z, th = f.states(s[lo:hi], th)
            nxt = s[lo + 1:hi + 1]
            m = keep[k][lo + 1:hi + 1]
            cols = np.flatnonzero(m)
            M = sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), (nxt[m], cols)),
                                  shape=(f.V, len(nxt)))
            f.C += (M @ Z.real).astype(np.float32) + 1j * (M @ Z.imag).astype(np.float32)
            np.add.at(f.n, nxt[m], 1)
    f.sig = np.linalg.norm(f.C, axis=1) / (f.N * np.sqrt(2))


def field_stats(f, parts, pos):
    return f.stats(parts, pos, TAUS)


def arms(store, f, parts, pos, cfg, p_uni, sel=None):
    """Returns {arm: nll} with alpha (and tau) either selected here (sel is
    None -> returns the selection) or taken from sel."""
    wm, wx = kernel_weights(cfg)
    sc, y = store.scores(parts, pos, wm, wx)
    et, ec = sc["exp"]; bt, bc = sc["bg"]; tt, tc = sc["tg"]
    cand = {"E0": (et, ec), "EB": (et + bt, ec + bc), "EB3": (et + bt + tt, ec + bc + tc)}
    out, choice = {}, {}
    for name, (n, d) in cand.items():
        if sel is None:
            v, a = min((nll(n, d, y, p_uni, a), a) for a in ALPHAS); choice[name] = (a,)
        else:
            a, = sel[name]; v = nll(n, d, y, p_uni, a)
        out[name] = v
    if f is not None:
        st, tg = f.stats(parts, pos, TAUS if sel is None else [sel["H"][1]])
        assert np.array_equal(tg, y)
        best = None
        for t in (TAUS if sel is None else [sel["H"][1]]):
            fn, fd = st[t]
            for a in (ALPHAS if sel is None else [sel["H"][0]]):
                v = nll(et + fn, ec + fd, y, p_uni, a)
                if best is None or v < best[0]:
                    best = (v, a, t)
        out["H"] = best[0]; choice["H"] = (best[1], best[2])
    return out, (choice if sel is None else sel)


def select_cfg(V, tr, dv, dpos, store, p_uni, seed, log=print):
    best = None
    for cfg in CONFIGS:
        t0 = time.time()
        f = ChannelField(V, N_OSC, cfg, seed); learn_tail(f, tr, store)
        out, sel = arms(store, f, dv, dpos, cfg, p_uni)
        log(f"    cfg {cfg[1]}: dev H={out['H']:.4f} EB={out['EB']:.4f} EB3={out['EB3']:.4f} "
            f"E0={out['E0']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if best is None or out["H"] < best[0]:
            best = (out["H"], cfg, sel)
    return best[1], best[2]


def control(seed, log=print):
    rng = np.random.default_rng(4242 + seed)
    V = 200
    T1 = rng.integers(0, V, V); T2 = rng.integers(0, V, (V, V))
    tr = [control_stream(200_000, rng, T1, T2)]
    dv = [control_stream(4_000, rng, T1, T2)]
    te = [control_stream(20_000, rng, T1, T2)]
    ng = NGram(tr, V)
    dpos, tpos = eval_positions(dv, 4_000), eval_positions(te, 20_000)
    store = Store(tr, V, 1_000)
    cfg, sel = select_cfg(V, tr, dv, dpos, store, ng.p_uni, seed, log=lambda *a, **k: None)
    f = ChannelField(V, N_OSC, cfg, seed); learn_tail(f, tr, store)
    out, _ = arms(store, f, te, tpos, cfg, ng.p_uni, sel)
    rec = (out["E0"] - out["H"]) / max(out["E0"] - out["EB3"], 1e-9)
    out["recovered"] = rec
    log(f"  control seed {seed}: cfg {cfg[1]}  E0={out['E0']:.3f} EB={out['EB']:.3f} "
        f"EB3={out['EB3']:.3f} H={out['H']:.3f}  H recovers {rec:.0%} of the exact tail's gain",
        flush=True)
    return out


def main(stage):
    print("PHASE 63 -- separate the strong memories, superpose only the rest\n", flush=True)
    if stage in ("control", "all"):
        print("READOUT CONTROL -- mixed-order source, B = 1,000")
        ok = all(o["recovered"] >= 0.40 for o in (control(s) for s in HELD_OUT))
        print(f"  CONTROL {'PASS' if ok else 'FAIL -> RUN VOID'}\n", flush=True)

    if stage in ("text", "all"):
        books = load_books()
        tr, dv, te = split_books(books)
        w2i = build_vocab(tr); V = len(w2i)
        tr, dv, te = encode(tr, w2i), encode(dv, w2i), encode(te, w2i)
        ng = NGram(tr, V)
        dpos, tpos = eval_positions(dv, DEV_EVAL), eval_positions(te, TEST_EVAL)
        D = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn2", d))
        D3 = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn3", d))
        kn2 = ngram_nll(ng, te, tpos, "kn2", D); kn3 = ngram_nll(ng, te, tpos, "kn3", D3)
        mem_kn2 = 2 * sum(len(r) for r in ng.big.values()) + V
        mem_field = 2 * V * N_OSC
        print(f"REAL TEXT V={V}  TEST KN2={kn2:.4f} KN3={kn3:.4f} infinite-N field={INF_N}  "
              f"memory: KN2 {mem_kn2:,} floats, field {mem_field:,} floats\n", flush=True)
        print("  configuration selection at finite N (B = 10,000, dev, seed 0):", flush=True)
        CFG, _ = select_cfg(V, tr, dv, dpos, Store(tr, V, 10_000), ng.p_uni, SEL_SEED)
        print(f"  FROZEN configuration {CFG[1]} fractions {CFG[2]}  "
              f"kernel weights (match, mismatch) {tuple(round(x, 4) for x in kernel_weights(CFG))}\n",
              flush=True)
        verdict = {}
        for B in BS:
            t0 = time.time()
            store = Store(tr, V, B)
            f = ChannelField(V, N_OSC, CFG, SEL_SEED); learn_tail(f, tr, store)
            _, sel = arms(store, f, dv, dpos, CFG, ng.p_uni)
            ref, _ = arms(store, None, te, tpos, CFG, ng.p_uni, sel)
            print(f"  B={B:>7,}: explicit {store.mem_exp:,} floats (+ bigram tail {store.mem_tailbg:,}; "
                  f"trigram tail {store.mem_tailtg:,})  selected {sel}  [{time.time() - t0:.0f}s]")
            print(f"    TEST E0={ref['E0']:.4f}  EB={ref['EB']:.4f}  EB3={ref['EB3']:.4f}", flush=True)
            Hs = []
            for s in HELD_OUT:
                f = ChannelField(V, N_OSC, CFG, s); learn_tail(f, tr, store)
                out, _ = arms(store, f, te, tpos, CFG, ng.p_uni, sel)
                Hs.append(out["H"])
                print(f"    seed {s}: H={out['H']:.4f}  capacity cost {out['H'] - INF_N:+.4f}  "
                      f"H vs EB {ref['EB'] - out['H']:+.4f}  H vs EB3 {ref['EB3'] - out['H']:+.4f}  "
                      f"H vs KN2 {kn2 - out['H']:+.4f}", flush=True)
            verdict[B] = (np.array(Hs), ref)
        Hs, ref = verdict[10_000]
        p1 = int((Hs - ref["EB3"] < 0.10).sum()); p2 = int((Hs < ref["EB"]).sum())
        m1 = int((ref["EB3"] < Hs).sum()); p3 = int((Hs < kn2).sum())
        print(f"\n  P1 H - EB3 < 0.10 at B=10k:       {p1}/5 -> {'PASS' if p1 == 5 else 'FAIL'}")
        print(f"  P2 H beats EB at B=10k:           {p2}/5 -> {'PASS' if p2 == 5 else 'FAIL (predicted)'}")
        print(f"  M1 EB3 beats H:                   {m1}/5 -> "
              f"{'M1 explains any P2 pass' if m1 > 0 else 'M1 rejected'}")
        print(f"  P3 H beats KN2:                   {p3}/5 -> {'PASS' if p3 == 5 else 'FAIL (predicted)'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
