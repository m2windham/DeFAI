"""
PHASE 61 -- A WORD IS A MOVE, NOT A PLACE.

Origin (owner challenge, 2026-09-24): apply Zahavy (2026, "LLMs can't jump")
to the architecture -- attempt an abductive 'jump' from a conceptual
inconsistency rather than a data anomaly, then test it like any other claim.

THE INCONSISTENCY. The substrate treats a unit as a PLACE the field settles
into, yet the failure-point scan (2026-09-23) found most learning happens
while the field is MOVING between places ('transit contamination'), and every
fix since phase 51 kept the place-assumption and patched around it.

THE THOUGHT EXPERIMENT. Ride the field. The same word heard after different
histories leaves you in different places, but the CHANGE it makes -- the
move -- is the same. Where you are is context; what happens to you is the word.

THE AXIOM. A word is the context-invariant part of the field's move. Context
is the field's state. Two consequences are tested here.

P1 -- IDENTITY FROM MOVES (phase 52's stream, where A0 'never got a slot').
  The substrate's own law is z' = normalize(z + dt(i w z + g(x - z))). It
  knows that law, so from (z_t, z_{t+1}) it can strip the context part and
  recover the move that does not depend on where it was. Units are formed by
  one fixed-bar online leader clusterer over (a) recovered moves, (b) settled
  states (the current reading), (c) raw frames.
  M1, NAMED IN ADVANCE: for a KNOWN linear-plus-renormalize law the invariant
  move is algebraically the drive x itself, so (a) must equal (c). A P1 pass
  therefore says the transit problem was self-inflicted -- the architecture
  read states when the invariant was in the drive -- and adds NO capability
  over a float clusterer on raw frames. It is recorded as a closure of the
  51-60 line, not as evidence for the axiom.
  PASS: (a) forms exactly 5 units, purity >= 0.95, on every held-out seed
  5-9 at HOLD in {2, 3, 6}; and recovered moves correlate >= 0.999 with the
  true frames (the M1 check).

P2 -- CONTEXT = WHERE YOU ARE (real text, the decisive test).
  If a word is a move, a history is a COMPOSITION of moves and the field's
  state carries more than the last word, with no transition table. Each word
  is a PHASE KICK on N oscillators; the phases accumulate the kicks along a
  ring with fade lam:
      theta_t = kick(w_t) + lam * Pi theta_{t-1},     z_t = exp(i theta_t)
  kick(w) = amp * R e_w ('emb': e_w the word's PPMI+SVD embedding from the
  TRAIN split, R a fixed random projection -- similar words kick alike) or
  amp * uniform(0, 2pi) ('rand'); Pi a fixed permutation, the ring the
  history travels along, which makes 'A then B' differ from 'B then A'.
  Because z is exp of a sum, the overlap of two states is a PRODUCT over
  history positions of how alike the words there are, with lam^k shrinking
  how much position k counts: lam and amp together set the depth cutoff.
  Readout, online, no gradient: C_w += z_t whenever w follows. Score
  S_w = Re<z, C_w>/N is a kernel count of past contexts like this one that
  w followed. p(w|z) = (max(S_w - tau sig_w, 0) + alpha p_uni(w)) /
  (sum + alpha), sig_w = |C_w| / (N sqrt 2) the MEASURED cross-talk null of
  row w (what an unrelated state would score against it).
  Baselines, same splits, same vocabulary: interpolated Kneser-Ney bigram
  (THE comparator), Dirichlet bigram (same smoothing family as the field --
  separates 'no depth' from 'weaker smoothing'), KN trigram (reference only).
  Corpus: 12 Gutenberg books, each split 80/10/10 contiguous train/dev/test,
  vocabulary = train words with count >= 3, the rest <unk>. N = 1024.
  Selection: lam, kick, amp, alpha, tau and the KN discounts are chosen on
  DEV (first 800 positions of each book's dev split) at selection seed 0
  from the grids in the code; TEST (first 2000 positions of each book's test
  split) is scored once per held-out seed 5-9 at the frozen choice.
  P2  PASS: the field (best lam > 0 arm) beats KN bigram on held-out test
      log-likelihood, 5/5 seeds.
  P2b PASS: that arm beats the same machine's best lam = 0 arm (a bigram by
      construction) on test, 5/5 seeds -- depth > 1 measured INSIDE the
      machine, not only against a differently-smoothed baseline.

DEVELOPMENT RECORD (before registration; selection seeds 0/1 and synthetic
data only, no test text touched):
  - v1 used an ADDITIVE state, z_t = d(w_t) * (lam Pi z_{t-1} + 1 - lam).
    Its overlap is a SUM over matching suffixes, so a two-word match weighs
    at most (1 + lam^2) times a one-word match; it FAILED the positive
    control (2.40 vs KN2 2.30 nats). Superposition holds depth; a linear
    readout of superposition cannot see it. Replaced by the phase field.
  - The first null, sqrt(n_w / 2N), assumed stored contexts are independent.
    They are not -- each repeat of a context type adds the same stray
    overlap -- and it under-estimated cross-talk ~10x on the control.
    Replaced by the row-norm null above.
  - With both fixes, control dev at seed 0: KN2 2.311, field 1.176 (N=1024).
    KN3 on the same data is 0.672: the field sees depth but is capacity-
    limited well short of an exact trigram.

POSITIVE CONTROL (the SOP gap the scan exposed). A synthetic second-order
source (V = 30) where the next symbol is fixed by the previous TWO (90%).
Same machine, same grids, same dev selection, random kicks, held-out seeds
5-9. The best lam > 0 arm must beat KN bigram on the control's test by
>= 1.0 nat/token, and the best lam = 0 arm must NOT (it is a bigram), on
every seed. If the control fails, P2 is VOID -- the probe cannot see depth.

KILL RULE. If the control passes and P2 fails, 'context = where the field
is' does not beat a bigram on real text at this scale and the axiom does not
yet carry language. If P2 fails but the field beats the Dirichlet bigram,
record 'depth present, smoothing insufficient' -- still a FAIL.

PRIOR ART, stated before the result. Words-as-operators: observable operator
models (Jaeger 2000), matrix-space compositional models. Order by
permutation + binding: HRR/FHRR, BEAGLE (Jones & Mewhort 2007), Sahlgren et
al. (2008). Linear fading-memory state + random-feature cosine readout is
reservoir computing with random Fourier features (Rahimi & Recht 2007); the
readout is Nadaraya-Watson-style kernel counting. A P2 pass would NOT establish novel mathematics; the only
candidate novelty is deriving the representation from the invariance
principle inside the oscillator substrate. A literature check precedes any
novelty claim.

Honest scope: P2's embeddings come from the train split (no leakage); the
field state is computed online; the readout's C accumulates online; test is
scored frozen, as the baselines are.

RESULT (run 2026-09-24, after pre-registration f8be3f8). KILL RULE FIRED.
  P1 PASS -- and explained entirely by M1. Recovered move ~ frame corr
     1.00000; moves form exactly 5 pure units at HOLD 2/3/6 on seeds 5-9.
     The settled-state reading collapses to 1-2 units at HOLD 2-3 and loses
     a unit at HOLD 6 on seed 7. No capability over raw-frame clustering;
     it closes the 51-60 line: the transit problem was self-inflicted.
  CONTROL PASS 5/5. KN2 2.27-2.42, field lam>0 1.10-1.23 (margin 1.10-1.31),
     field lam=0 2.31-2.46 (bigram level), KN3 0.67-0.71. The probe sees
     depth.
  P2 FAIL 0/5. Test (24,000 tokens): KN2 5.5086, Dirichlet2 5.6313, KN3
     5.4476; field 5.825-5.835 -- 0.32 nats worse than KN2 and ~0.20 worse
     than its own smoothing family, so not even 'depth present, smoothing
     insufficient'.
  P2b FAIL 0/5. Dev chose lam=0.2 with the softest random kick -- the
     machine selected to be a bigram -- and lam>0 is 0.002-0.004 worse than
     lam=0 on test. Every deeper setting was worse on dev.
  POST-HOC (dev only, not part of the registered verdict):
     capacity -- lam=0 at N=1024 5.787 vs its N->inf limit (Dirichlet2)
     5.594; N=4096 gives 5.722, closing a third of the gap.
     no backoff -- the phase kernel is a PRODUCT over history positions, so
     a context matching the last word but not the one before scores ~0.
     Dense trigrams (the control) reward that; sparse real text (KN3 beats
     KN2 by only 0.06 here) punishes it, because the bigram evidence is
     thrown away. Backoff needs several depths mixed in the readout, which
     is rebuilding smoothed n-grams in phase space.
  Prior art, checked: the kernel is fractional power encoding (Frady et al.
     2021, arXiv 2109.03429 -- the sinc^2 kernel the control shows);
     permute-then-bind n-grams are standard HDC.
"""

import os
import re
import sys
import glob
import time
import math
import numpy as np
from collections import Counter, defaultdict

CORPUS_DIR = "/tmp/gutenberg_p61"
BOOKS = [1342, 158, 105, 161, 1260, 345, 174, 120, 35, 36, 768, 219]
SEL_SEED = 0
HELD_OUT = [5, 6, 7, 8, 9]
MIN_COUNT = 3
EMB_DIM = 64
N_OSC = 1024
DEV_EVAL = 800      # dev tokens scored per book (selection)
TEST_EVAL = 2000    # test tokens scored per book (held-out)

LAMS = [0.0, 0.2, 0.35, 0.5, 0.7]
KICKS = ["emb", "rand"]          # move from the word's embedding, or random
AMPS = [1.0, 3.0, 10.0]          # kick size (sets the depth cutoff with lam)
ALPHAS = [1, 3, 10, 30, 100, 300, 1000]
TAUS = [0.0, 1.0, 2.0, 3.0, 4.0]
KN_DS = [0.5, 0.75, 0.9]


# ================================================================ corpus
def load_books():
    books = []
    for b in BOOKS:
        path = os.path.join(CORPUS_DIR, f"{b}.txt")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; fetch with:\n  mkdir -p {CORPUS_DIR} && "
                             f"curl -sS -o {path} https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt")
        txt = open(path, encoding="utf-8", errors="replace").read()
        s = re.search(r"\*\*\* ?START OF.*?\*\*\*", txt)
        e = re.search(r"\*\*\* ?END OF", txt)
        body = txt[s.end() if s else 0: e.start() if e else len(txt)]
        books.append(re.findall(r"[a-z']+", body.lower()))
    return books


def split_books(books):
    tr, dv, te = [], [], []
    for toks in books:
        n = len(toks)
        a, b = int(0.8 * n), int(0.9 * n)
        tr.append(toks[:a]); dv.append(toks[a:b]); te.append(toks[b:])
    return tr, dv, te


def build_vocab(train):
    c = Counter(w for toks in train for w in toks)
    vocab = ["<unk>"] + sorted(w for w, k in c.items() if k >= MIN_COUNT)
    return {w: i for i, w in enumerate(vocab)}


def encode(parts, w2i):
    return [np.array([w2i.get(w, 0) for w in toks], dtype=np.int64) for toks in parts]


# ============================================================ baselines
class NGram:
    """Interpolated Kneser-Ney bigram/trigram and a Dirichlet bigram, all
    trained on the same encoded train split."""

    def __init__(self, train, V):
        self.V = V
        self.uni = np.zeros(V)
        self.big = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        for s in train:
            np.add.at(self.uni, s, 1)
            for a, b in zip(s[:-1], s[1:]):
                self.big[int(a)][int(b)] += 1
            for a, b, c in zip(s[:-2], s[1:-1], s[2:]):
                self.tri[(int(a), int(b))][int(c)] += 1
        cont = np.zeros(V)
        for a, row in self.big.items():
            for b in row:
                cont[b] += 1
        self.p_cont = (cont + 1e-3) / (cont + 1e-3).sum()
        self.p_uni = (self.uni + 1) / (self.uni.sum() + V)
        self.big_tot = {a: sum(r.values()) for a, r in self.big.items()}
        self.tri_tot = {k: sum(r.values()) for k, r in self.tri.items()}

    def kn2(self, a, b, D):
        row = self.big.get(a)
        if not row:
            return self.p_cont[b]
        tot = self.big_tot[a]
        return max(row.get(b, 0) - D, 0) / tot + D * len(row) / tot * self.p_cont[b]

    def kn3(self, u, a, b, D):
        row = self.tri.get((u, a))
        low = self.kn2(a, b, D)
        if not row:
            return low
        tot = self.tri_tot[(u, a)]
        return max(row.get(b, 0) - D, 0) / tot + D * len(row) / tot * low

    def dir2(self, a, b, alpha):
        row = self.big.get(a, {})
        tot = self.big_tot.get(a, 0)
        return (row.get(b, 0) + alpha * self.p_uni[b]) / (tot + alpha)


def eval_positions(parts, n_per):
    """(part, index) pairs scored: the first n_per positions >= 2 of each
    part (context is always the part's own preceding tokens)."""
    return [(k, i) for k, s in enumerate(parts) for i in range(2, min(len(s), n_per + 2))]


def ngram_nll(ng, parts, pos, kind, param):
    tot = 0.0
    for k, i in pos:
        s = parts[k]
        a, b = int(s[i - 1]), int(s[i])
        if kind == "kn2":
            p = ng.kn2(a, b, param)
        elif kind == "kn3":
            p = ng.kn3(int(s[i - 2]), a, b, param)
        else:
            p = ng.dir2(a, b, param)
        tot -= math.log(p)
    return tot / len(pos)


# ================================================================ field
def ppmi_embeddings(train, V, dim, window=4):
    from scipy import sparse
    from sklearn.utils.extmath import randomized_svd
    rows, cols = [], []
    for s in train:
        n = len(s)
        for off in range(1, window + 1):
            rows.append(s[:n - off]); cols.append(s[off:])
    r = np.concatenate(rows); c = np.concatenate(cols)
    M = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(V, V)).tocsr()
    M = M + M.T
    tot = M.sum()
    rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel()
    M = M.tocoo()
    pmi = np.log(M.data * tot / (rs[M.row] * cs[M.col] + 1e-12) + 1e-12)
    keep = pmi > 0
    P = sparse.coo_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=(V, V)).tocsr()
    U, S, _ = randomized_svd(P, n_components=dim, random_state=0)
    E = U * np.sqrt(S)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


class Field:
    """Each word is a PHASE KICK theta_w on N oscillators. The field's phase
    accumulates the kicks along a ring (the permutation) with fade lam:
        theta_t = theta_{w_t} + lam * Pi theta_{t-1},   z_t = exp(i theta_t)
    Because z is exp of a SUM, the overlap of two states is a PRODUCT over
    history positions: Re<z_t, z_s>/N ~ prod_k phi(lam^k (theta_a - theta_b)),
    phi the characteristic function of the kick difference -- 1 where the
    words agree, small where they differ, and differences further back
    (lam^k small) matter less. A context agreeing on more recent words is
    MULTIPLICATIVELY closer, which is what a linear readout needs to see
    depth. (Development record: the first design used an additive state,
    z_t = d(w_t) * (lam Pi z_{t-1} + 1 - lam); its overlap is a SUM over
    matching suffixes, so a two-word match weighs at most (1 + lam^2) times
    a one-word match, and on development seeds 0/1 it failed the positive
    control -- 2.40 vs KN2 2.30 nats. Superposition can hold depth; a linear
    readout of superposition cannot see it.)"""

    def __init__(self, V, E, N, lam, kick, amp, seed):
        rng = np.random.default_rng(10_000 + seed)
        if kick == "rand" or E is None:
            theta = amp * rng.uniform(0, 2 * np.pi, (V, N))
        else:
            R = rng.standard_normal((E.shape[1], N))
            theta = amp * (E @ R)
        self.th = theta.astype(np.float32)
        self.perm = rng.permutation(N)
        self.N, self.lam, self.V = N, lam, V
        self.C = np.zeros((V, N), dtype=np.complex64)
        self.n = np.zeros(V)

    def states(self, s, th=None):
        """z_t after reading s[t], for every t; returns (Z, last phase) so
        long streams can be walked in chunks."""
        TH = np.empty((len(s), self.N), dtype=np.float32)
        th = np.zeros(self.N, dtype=np.float32) if th is None else th
        lam, kick, perm = np.float32(self.lam), self.th, self.perm
        for t, w in enumerate(s):
            th = kick[w] + lam * th[perm]
            TH[t] = th
        return np.exp(1j * TH).astype(np.complex64), th

    def learn(self, parts, chunk=20_000):
        from scipy import sparse
        for s in parts:
            th = None
            for lo in range(0, len(s) - 1, chunk):
                hi = min(lo + chunk, len(s) - 1)
                Z, th = self.states(s[lo:hi], th)     # states after s[lo..hi-1]
                nxt = s[lo + 1:hi + 1]
                M = sparse.csr_matrix((np.ones(len(nxt), dtype=np.float32),
                                       (nxt, np.arange(len(nxt)))), shape=(self.V, len(nxt)))
                self.C += (M @ Z.real).astype(np.float32) + 1j * (M @ Z.imag).astype(np.float32)
                np.add.at(self.n, nxt, 1)
        # measured cross-talk null of each readout row. Stored contexts are
        # NOT independent: every repeat of a context type adds the same stray
        # overlap, so cross-talk grows with the type's count, not sqrt(n).
        # The null is measured from the row itself: an unrelated random-phase
        # probe overlaps C_w by ~N(0, |C_w|^2 / 2N^2). (Development record:
        # the sqrt(n/2N) independence null under-estimated this ~10x on the
        # control, where a few cycle contexts carry thousands of counts.)
        self.sig = np.linalg.norm(self.C, axis=1) / (self.N * np.sqrt(2))

    def stats(self, parts, pos, taus, chunk=1000):
        """For each tau: (thresholded score of the target, row sum of the
        thresholded scores) per scored position, in count units. Chunked so
        the (positions x V) score matrix is never held whole."""
        by_part = defaultdict(list)
        for k, i in pos:
            by_part[k].append(i)
        Zs, tg = [], []
        for k, idx in by_part.items():
            s = parts[k]
            Z, _ = self.states(s[:max(idx)])  # Z[i-1] = state after s[i-1]
            idx = np.array(idx)
            Zs.append(Z[idx - 1]); tg.append(s[idx])
        Z = np.concatenate(Zs); tg = np.concatenate(tg)
        Cr, Ci = self.C.real.T.copy(), self.C.imag.T.copy()
        out = {t: (np.empty(len(tg)), np.empty(len(tg))) for t in taus}
        for lo in range(0, len(tg), chunk):
            hi = min(lo + chunk, len(tg))
            S = (Z[lo:hi].real @ Cr + Z[lo:hi].imag @ Ci) / self.N
            St = S[np.arange(hi - lo), tg[lo:hi]]
            for t in taus:
                thr = t * self.sig
                out[t][0][lo:hi] = np.maximum(St - thr[tg[lo:hi]], 0.0)
                out[t][1][lo:hi] = np.maximum(S - thr[None, :], 0.0).sum(1)
        return out, tg

    @staticmethod
    def nll(st, tg, p_uni, alpha, tau):
        num, den = st[tau]
        return float(-np.mean(np.log((num + alpha * p_uni[tg]) / (den + alpha))))

    @staticmethod
    def select(st, tg, p_uni):
        return min((Field.nll(st, tg, p_uni, a, t), a, t) for a in ALPHAS for t in TAUS)


def field_grid(train, dev, dpos, V, E, p_uni, seed, N, log=print):
    res = {}
    for lam in LAMS:
        for kick in KICKS:
            for amp in AMPS:
                t0 = time.time()
                f = Field(V, E, N, lam, kick, amp, seed)
                f.learn(train)
                st, tg = f.stats(dev, dpos, TAUS)
                best = Field.select(st, tg, p_uni)
                res[(lam, kick, amp)] = best
                log(f"    lam={lam:<4} kick={kick:<4} amp={amp:<4} dev nll={best[0]:.4f} "
                    f"(alpha={best[1]}, tau={best[2]})  [{time.time() - t0:.0f}s]", flush=True)
    return res


# ===================================================== positive control
def control_stream(n, rng, T):
    V = len(T)
    s = np.empty(n, dtype=np.int64); s[:2] = rng.integers(0, V, 2)
    for t in range(2, n):
        s[t] = T[s[t - 2], s[t - 1]] if rng.random() < 0.9 else rng.integers(0, V)
    return s


def positive_control(seed, log=print):
    """Same machine, same selection procedure, a source whose next symbol is
    fixed by the previous TWO. Returns test nats/token for KN2, KN3 and the
    field's best lam>0 and lam=0 arms (each selected on the control's dev)."""
    rng = np.random.default_rng(777 + seed)
    V = 30
    T = rng.integers(0, V, (V, V))        # ONE rule table for all splits
    tr = [control_stream(200_000, rng, T)]
    dv = [control_stream(4_000, rng, T)]
    te = [control_stream(20_000, rng, T)]
    ng = NGram(tr, V)
    dpos, tpos = eval_positions(dv, 4_000), eval_positions(te, 20_000)
    D = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn2", d))
    D3 = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn3", d))
    out = {"kn2": ngram_nll(ng, te, tpos, "kn2", D), "kn3": ngram_nll(ng, te, tpos, "kn3", D3)}
    for arm, lams in (("deep", [l for l in LAMS if l > 0]), ("lam0", [0.0])):
        best = None
        for lam in lams:
            for amp in AMPS:
                f = Field(V, None, N_OSC, lam, "rand", amp, seed); f.learn(tr)
                st, tg = f.stats(dv, dpos, TAUS)
                d, a, t = Field.select(st, tg, ng.p_uni)
                if best is None or d < best[0]:
                    best = (d, lam, amp, a, t)
        _, lam, amp, a, t = best
        f = Field(V, None, N_OSC, lam, "rand", amp, seed); f.learn(tr)
        st, tg = f.stats(te, tpos, [t])
        out[arm] = Field.nll(st, tg, ng.p_uni, a, t)
        out[arm + "_cfg"] = (lam, amp, a, t)
    log(f"  control seed {seed}: KN2={out['kn2']:.3f}  KN3(ref)={out['kn3']:.3f}  "
        f"field lam>0={out['deep']:.3f} {out['deep_cfg']}  field lam=0={out['lam0']:.3f}  "
        f"nats/token", flush=True)
    return out


# ================================================================== P1
def p1_seed(seed, hold, log=print):
    from phase51_causal_state_recruit import (hidden_walk, appearance_codebook,
                                              STATE_APPEARANCE, N_DIM, NOISE)
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    states = hidden_walk(3000, rng)
    app = np.array([STATE_APPEARANCE[s] for s in states])
    X, lab = [], []
    for a in app:
        for _ in range(hold):
            v = code[a] + NOISE * (rng.standard_normal(N_DIM) + 1j * rng.standard_normal(N_DIM))
            X.append(v / np.linalg.norm(v) * np.sqrt(N_DIM)); lab.append(a)
    X = np.array(X); lab = np.array(lab)
    omega, g, dt, norm = 0.25, 4.0, 0.05, np.sqrt(N_DIM)   # organism defaults
    A, B = 1 + dt * (1j * omega - g), dt * g
    Z = np.empty_like(X); z = np.zeros(N_DIM, complex) + 1.0
    z = z / np.linalg.norm(z) * norm
    for t, x in enumerate(X):
        z = A * z + B * x
        z = z / np.linalg.norm(z) * norm
        Z[t] = z
    # recover the context-invariant move: z_{t+1} = c (A z_t + B x), |x| = norm
    prevZ = np.vstack([np.ones(N_DIM) / np.sqrt(N_DIM) * norm, Z[:-1]])
    Xh = np.empty_like(X)
    for t in range(len(X)):
        u, v = Z[t], A * prevZ[t]           # x = (u/c - v)/B ; solve |u/c - v| = B norm
        a2 = np.vdot(u, u).real; b1 = -2 * np.vdot(v, u).real; c0 = np.vdot(v, v).real - (B * norm) ** 2
        disc = max(b1 * b1 - 4 * a2 * c0, 0.0)
        ic = max((-b1 + np.sqrt(disc)) / (2 * a2), (-b1 - np.sqrt(disc)) / (2 * a2))
        Xh[t] = (u * ic - v) / B
    corr = np.mean([abs(np.vdot(a_, b_)) / (np.linalg.norm(a_) * np.linalg.norm(b_))
                    for a_, b_ in zip(Xh, X)])
    settled = np.arange(hold - 1, len(X), hold)             # last frame of each token

    def leader(V_, bar=0.5):
        cents, cnt, out = [], [], np.empty(len(V_), int)
        for i, v in enumerate(V_):
            if cents:
                C = np.array(cents)
                o = np.abs(C.conj() @ v) / (np.linalg.norm(C, axis=1) * np.linalg.norm(v))
                k = int(np.argmax(o))
                if o[k] >= bar:
                    cnt[k] += 1; cents[k] = cents[k] + (v - cents[k]) / cnt[k]
                    out[i] = k; continue
            cents.append(v.copy()); cnt.append(1); out[i] = len(cents) - 1
        return out

    def judge(u, true):
        units = [k for k in np.unique(u) if (u == k).sum() >= 0.02 * len(u)]
        pur = [np.bincount(true[u == k]).max() / (u == k).sum() for k in units]
        covered = {int(np.bincount(true[u == k]).argmax()) for k in units}
        return len(units), float(np.min(pur)) if pur else 0.0, len(covered)

    r = {"move": judge(leader(Xh), lab), "settled": judge(leader(Z[settled]), lab[settled]),
         "raw": judge(leader(X), lab), "corr": corr}
    log(f"  seed {seed} HOLD={hold}: move units/purity/covered={r['move']}  "
        f"settled={r['settled']}  raw={r['raw']}  move~frame corr={corr:.5f}")
    return r


# ================================================================ main
def main(stage):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    print("PHASE 61 -- a word is a move, not a place\n")

    if stage in ("p1", "all"):
        print("P1 -- identity from moves (phase 52 stream)")
        ok = True
        for hold in (2, 3, 6):
            for s in HELD_OUT:
                r = p1_seed(s, hold)
                ok &= r["move"][0] == 5 and r["move"][1] >= 0.95 and r["move"][2] == 5 \
                    and r["corr"] >= 0.999
        print(f"  P1 {'PASS' if ok else 'FAIL'} (M1: move == drive; see docstring)\n")

    if stage in ("control", "all"):
        print("POSITIVE CONTROL -- second-order synthetic source")
        ok = True
        for s in HELD_OUT:
            o = positive_control(s)
            ok &= (o["kn2"] - o["deep"] >= 1.0) and (o["kn2"] - o["lam0"] < 1.0)
        print(f"  CONTROL {'PASS' if ok else 'FAIL -> P2 VOID'}\n")

    if stage in ("p2", "all", "timing"):
        t0 = time.time()
        books = load_books()
        tr, dv, te = split_books(books)
        w2i = build_vocab(tr)
        V = len(w2i)
        tr, dv, te = encode(tr, w2i), encode(dv, w2i), encode(te, w2i)
        print(f"P2 -- corpus: {sum(map(len, tr))} train / {sum(map(len, dv))} dev / "
              f"{sum(map(len, te))} test tokens, V={V}  [{time.time() - t0:.0f}s]")
        ng = NGram(tr, V)
        E = ppmi_embeddings(tr, V, EMB_DIM)
        dpos = eval_positions(dv, DEV_EVAL)
        if stage == "timing":
            f = Field(V, E, N_OSC, 0.35, "emb", 3.0, SEL_SEED)
            t1 = time.time(); f.learn(tr); print(f"  learn {time.time() - t1:.0f}s")
            t1 = time.time(); f.stats(dv, dpos, TAUS); print(f"  score dev {time.time() - t1:.0f}s")
            return
        D = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn2", d))
        D3 = min(KN_DS, key=lambda d: ngram_nll(ng, dv, dpos, "kn3", d))
        A2 = min(ALPHAS, key=lambda a: ngram_nll(ng, dv, dpos, "dir2", a))
        print(f"  selected on dev: KN2 D={D}, KN3 D={D3}, Dirichlet alpha={A2}")
        print(f"  dev: KN2={ngram_nll(ng, dv, dpos, 'kn2', D):.4f}  "
              f"KN3={ngram_nll(ng, dv, dpos, 'kn3', D3):.4f}")
        print("  field selection on dev (seed 0):", flush=True)
        res = field_grid(tr, dv, dpos, V, E, ng.p_uni, SEL_SEED, N_OSC)
        deep = min((k for k in res if k[0] > 0), key=lambda k: res[k][0])
        flat = min((k for k in res if k[0] == 0.0), key=lambda k: res[k][0])
        print(f"  FROZEN field: lam={deep[0]} kick={deep[1]} amp={deep[2]} "
              f"alpha={res[deep][1]} tau={res[deep][2]}  (dev {res[deep][0]:.4f})")
        print(f"  FROZEN lam=0: kick={flat[1]} amp={flat[2]} "
              f"alpha={res[flat][1]} tau={res[flat][2]}  (dev {res[flat][0]:.4f})")

        tpos = eval_positions(te, TEST_EVAL)
        kn2 = ngram_nll(ng, te, tpos, "kn2", D)
        kn3 = ngram_nll(ng, te, tpos, "kn3", D3)
        di2 = ngram_nll(ng, te, tpos, "dir2", A2)
        print(f"\n  TEST ({len(tpos)} tokens)  KN2={kn2:.4f}  Dirichlet2={di2:.4f}  "
              f"KN3(ref)={kn3:.4f}  nats/token", flush=True)
        p2 = p2b = 0
        for s in HELD_OUT:
            r = {}
            for name, cfg in (("field", deep), ("lam0", flat)):
                f = Field(V, E, N_OSC, cfg[0], cfg[1], cfg[2], s); f.learn(tr)
                _, a, t = res[cfg]
                st, tg = f.stats(te, tpos, [t])
                r[name] = Field.nll(st, tg, ng.p_uni, a, t)
            p2 += r["field"] < kn2; p2b += r["field"] < r["lam0"]
            print(f"  seed {s}: field={r['field']:.4f}  field(lam=0)={r['lam0']:.4f}  "
                  f"vs KN2 {kn2 - r['field']:+.4f}  vs lam0 {r['lam0'] - r['field']:+.4f}  "
                  f"vs Dir2 {di2 - r['field']:+.4f}  vs KN3 {kn3 - r['field']:+.4f}", flush=True)
        print(f"\n  P2  field beats KN bigram: {p2}/5 -> {'PASS' if p2 == 5 else 'FAIL'}")
        print(f"  P2b lam>0 beats lam=0:     {p2b}/5 -> {'PASS' if p2b == 5 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
