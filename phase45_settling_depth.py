"""
PHASE 45 (T7.3) -- SETTLING-DEPTH SWEEP.

WHAT THIS ASKS. Every committed text recipe holds each token for 8 frames
with `g_in*dt = 0.25`, i.e. roughly twice the field's own time constant
`1/(g_in*dt) = 4` frames. The sandbox reported that at that setting the
end-of-sequence field state is a PURE LAST-TOKEN ATTRACTOR -- same-sequence
ceiling exactly 1.0000 and suffix-sharing similarity 0.980 -- and that a
window opens near hold=3 (reproducibility 0.945, suffix-sharers 0.703).
Sweep the knob against the axes this project already measures, and price
the trade.

WHY THIS PHASE IS COUPLED TO T7.1. Phase 43 measured the imaginary channel
empty and localized the cause to TWO dials: `omega/g_in` and settling
depth. The residual PEAKS at hold 2-4 and is driven to floor by deep
settling. So settling depth is not one axis among several -- it is the same
dial that empties the phase channel, and any finding here transfers
directly to T7.6.

THE FIELD RECURSION IS INDEPENDENT OF THE MEMORY BANK. In `_perceive` the
z update is `dz = 1j*omega*z + g_in*(x - z)` followed by `normalize` --
nothing in it reads `xi`. So the end-of-sequence state can be computed
directly and vectorized over thousands of sequences, and section 0 CHECKS
that reimplementation against `Organism.perceive`'s own `org.z` rather than
assuming it.

======================================================================
PRE-REGISTRATION (committed before the run that tests it)
======================================================================

Definitions, all in the organism's own similarity `|<a,b>|/N`:
  CEILING  two runs of the SAME token sequence from DIFFERENT random
           initial z. High ceiling = the initial condition is forgotten and
           the state is a function of the token history.
  SUFFIX   two sequences that share the last token but have different
           prefixes.
  FLOOR    two sequences with different last tokens.
  PSI      (ceiling - suffix) / (ceiling - floor), the path-sensitivity
           index the work order names: 0 = the prefix is invisible, 1 = the
           prefix matters as much as the token identity does.

P1 -- THE LAST-TOKEN ATTRACTOR. At hold=8 (the committed setting):
ceiling >= 0.999, suffix >= 0.95, PSI <= 0.05. The sandbox anchor is
ceiling 1.0000 / suffix 0.980.

P2 -- THE WINDOW. PSI rises monotonically as hold falls, and PSI(hold=3) is
at least 5x PSI(hold=8). The same rise appears under the OTHER knob at
fixed frame count -- lowering `g_in*dt` at hold=8 -- which is the control
that separates "less settling" from "fewer frames".

P3 -- THE MUNDANE ACCOUNT, AND THE TEST THAT SEPARATES IT. M3: the time
constant `1/(g_in*dt) = 4` frames explains both numbers; there is no
window, only an EMA, and hold=3 "works" solely because settling is
INCOMPLETE -- the state is NOISY, not path-aware. The discriminator is
REPRODUCIBILITY ACROSS INITIAL CONDITIONS: un-converged transient is
initial-condition noise and does not repeat, prefix information does.
Two measurements decide it:
  (a) ceiling at hold=3 stays high (>= 0.90). Noise would drag it down
      along with suffix, and PSI would rise for the wrong reason.
  (b) PREFIX DECODING: with the final token FIXED, classify which of P
      distinct prefixes produced the state (1-NN over repeats from
      different initial z, leave-one-out), against a label-permutation
      null at the same sample size.
  DECISION RULE: decoding accuracy above the null's p99 AND ceiling >= 0.90
  => the retained information is PREFIX IDENTITY and the window is real.
  Decoding at chance, or a collapsed ceiling => M3 wins, the "window" is
  un-converged noise, and hold=3 must not be recommended for anything.

P4 -- THE COST, ON THE PROJECT'S OWN AXES. Shallower settling should cost
slot coverage (noisier routing fragments a word's evidence) and category
validity (phase 24's class-bigram MI against a permutation null).
PREDICTION: both degrade monotonically as hold falls, so the window is a
TRADE and the phase's output is a priced frontier, not a recommendation.
HONEST NEGATIVE: if neither degrades, hold=8 is simply over-settling and
there is a free lunch -- which would be a bigger finding than the window
and must be reported as such.

P5 -- THE CONSOLIDATION FLAG (conditional, not a claim). `consolidate()`'s
default `merge_thresh = 0.8` sits between ceiling and suffix at hold=3 and
BELOW suffix from hold=4 upward, where distinct suffix-sharing states would
be merged. Report the crossover hold at which suffix similarity passes 0.8.
Nothing in the repo consolidates sequence states today, so this is a flag
for whoever first does, not a defect claim -- and it is untested for that
use, which is the point.
"""

import time

import numpy as np

import text_recipe as tr
from organism import Organism, normalize
from polysemy_organism import PolysemyOrganism

HOLDS = (1, 2, 3, 4, 6, 8, 12)
GDT = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60)      # the other knob, at hold=8
OMEGA, BETA = 0.15, 10.0
G_IN, DT = 5.0, 0.05                            # phase 19's settings
SEEDS = (0, 1, 2, 3, 4)
SEEDS_HELD = (5, 6, 7, 8, 9)


# ---------------------------------------------------------------------------
# the field recursion, lifted verbatim from Organism._perceive and vectorized
# ---------------------------------------------------------------------------

def settle(seqs, emb, hold, g_in=G_IN, dt=DT, omega=OMEGA, z0=None, norm=None):
    """End-of-sequence field state for a batch of token sequences.

    `_perceive`'s z update, verbatim: x is normalized, then
    `z <- normalize(z + dt*(1j*omega*z + g_in*(x - z)))`, `hold` frames per
    token. Nothing here reads the memory bank because the update does not.
    seqs: (B, T) token ids. Returns (B, N) complex.
    """
    N = emb.shape[1]
    norm = np.sqrt(N) if norm is None else norm
    B = len(seqs)
    z = (np.asarray(z0, dtype=complex).copy() if z0 is not None
         else np.zeros((B, N), complex))
    E = np.asarray(emb, dtype=complex)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-300) * norm
    for t in range(np.shape(seqs)[1]):
        x = E[np.asarray(seqs)[:, t]]
        for _ in range(hold):
            z = z + dt * (1j * omega * z + g_in * (x - z))
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-300) * norm
    return z


def sim(a, b, N):
    """The organism's own similarity: |<a, b>| / N."""
    return np.abs(np.sum(np.conj(a) * b, axis=-1)) / N


def rand_z(rng, B, N, norm):
    z = rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-300) * norm


# ---------------------------------------------------------------------------
# phase 24's validity certificate, the two functions it needs
# ---------------------------------------------------------------------------

def mutual_information(M):
    """phase24_category_validity.mutual_information, verbatim."""
    p = M / M.sum()
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log2(p / (pi @ pj + 1e-300) + 1e-300)
    return float(term[p > 0].sum())


def cat_bigram(labels, k, pred_arr, succ_arr):
    """phase24_category_validity.cat_bigram, with the pair arrays passed in."""
    lp, ls = labels[pred_arr], labels[succ_arr]
    return np.bincount(lp * k + ls, minlength=k * k).reshape(k, k).astype(float)


def assign_slots(mem, states, N, chunk=20000):
    """argmax_k |<mem_k, state>| / N, chunked over states.

    The dense form `|mem.conj() @ states.T|` is (n_mem x n_tokens) complex,
    which at corpus scale (1200 memories x 400K tokens) is 6.5 GB. Chunking
    makes it O(n_mem x chunk) and changes nothing numerically -- argmax is
    per-column."""
    mem = np.ascontiguousarray(mem, dtype=complex)
    out = np.empty(len(states), dtype=int)
    for i in range(0, len(states), chunk):
        blk = np.ascontiguousarray(states[i:i + chunk], dtype=complex)
        out[i:i + chunk] = np.abs((mem.conj() @ blk.T) / N).argmax(0)
    return out


def slot_labels(org, n_mem):
    """Per-memory category label from `discover_categories_v2`'s own output
    (`word_slot_to_cat`), as an array indexed like `org.mem`'s rows."""
    return np.array([int(org.word_slot_to_cat.get(k, 0)) for k in range(n_mem)],
                    dtype=int)


def category_validity(org, seq_states, n_mem, n_null=200, seed=0):
    """MI(class bigram) vs a label-permutation null at the same sizes -- the
    phase 24 certificate, computed on the slot stream this organism assigns
    to the corpus. Returns (k, MI, null p99, z)."""
    labels = slot_labels(org, n_mem)
    k = int(labels.max()) + 1
    s = np.asarray(seq_states)
    pred_arr, succ_arr = s[:-1], s[1:]
    mi = mutual_information(cat_bigram(labels, k, pred_arr, succ_arr))
    rng = np.random.default_rng(seed)
    null = np.array([mutual_information(
        cat_bigram(rng.permutation(labels), k, pred_arr, succ_arr))
        for _ in range(n_null)])
    return k, mi, float(np.percentile(null, 99)), float(
        (mi - null.mean()) / (null.std() + 1e-12))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    print("PHASE 45 (T7.3): settling-depth sweep\n")
    R = tr.Recipe(tr.load_fables(), **tr.RECIPE_19, verbose=True)
    N, NORM, emb = R.N, R.NORM, R.embeddings
    V = len(R.vocab)

    # ---------------- 0. the reimplementation is checked, not assumed -----
    print("\n" + "=" * 70)
    print("(0) DOES `settle` REPRODUCE Organism.perceive's z? (checked, not assumed)")
    rng = np.random.default_rng(0)
    probe = rng.integers(0, V, (1, 12))
    o = Organism(N=N, K=64, omega=OMEGA, beta=BETA, seed=0, backend="numpy")
    z0 = np.asarray(o.z, complex)[None, :]
    o.perceive(R.stream(8, seq=list(probe[0])), **tr.PERCEIVE_19)
    mine = settle(probe, emb, 8, z0=z0, norm=NORM)[0]
    d = float(np.abs(mine - np.asarray(o.z, complex)).max())
    print(f"  max |d z| between `settle` and Organism.perceive: {d:.3e}  "
          f"{'MATCH' if d < 1e-9 else 'MISMATCH -- everything below is void'}")

    # ---------------- 1-2. the geometry sweep -----------------------------
    print("\n" + "=" * 70)
    print("(1/2) CEILING / SUFFIX / FLOOR / PSI -- P1, P2")
    B, T = 400, 10
    rg = np.random.default_rng(1)
    base = rg.integers(0, V, (B, T))
    diff_prefix = base.copy()
    diff_prefix[:, :-1] = rg.integers(0, V, (B, T - 1))     # same last token
    diff_last = base.copy()
    diff_last[:, -1] = rg.integers(0, V, B)                 # different last token

    def geometry(hold, g_in=G_IN, dt=DT):
        za = settle(base, emb, hold, g_in=g_in, dt=dt,
                    z0=rand_z(np.random.default_rng(11), B, N, NORM), norm=NORM)
        zb = settle(base, emb, hold, g_in=g_in, dt=dt,
                    z0=rand_z(np.random.default_rng(22), B, N, NORM), norm=NORM)
        zp = settle(diff_prefix, emb, hold, g_in=g_in, dt=dt,
                    z0=rand_z(np.random.default_rng(33), B, N, NORM), norm=NORM)
        zl = settle(diff_last, emb, hold, g_in=g_in, dt=dt,
                    z0=rand_z(np.random.default_rng(44), B, N, NORM), norm=NORM)
        c = float(np.mean(sim(za, zb, N)))
        s = float(np.mean(sim(za, zp, N)))
        f = float(np.mean(sim(za, zl, N)))
        psi = (c - s) / max(c - f, 1e-12)
        return c, s, f, psi

    print(f"  {'hold':>5} {'g_in*dt':>8} | {'ceiling':>8} {'suffix':>8} "
          f"{'floor':>8} {'PSI':>7} | merge_thresh=0.8 vs suffix")
    geo = {}
    for h in HOLDS:
        c, s, f, psi = geometry(h)
        geo[h] = (c, s, f, psi)
        flag = "suffix ABOVE 0.8 -- distinct states would MERGE" if s > 0.8 else \
               "suffix below 0.8 -- states stay distinct"
        print(f"  {h:>5} {G_IN * DT:>8.2f} | {c:>8.4f} {s:>8.4f} {f:>8.4f} "
              f"{psi:>7.4f} | {flag}")
    print(f"\n  the OTHER knob at fixed hold=8 and fixed frame count:")
    print(f"  {'hold':>5} {'g_in*dt':>8} | {'ceiling':>8} {'suffix':>8} "
          f"{'floor':>8} {'PSI':>7}")
    for gdt in GDT:
        c, s, f, psi = geometry(8, g_in=gdt / DT)
        print(f"  {8:>5} {gdt:>8.2f} | {c:>8.4f} {s:>8.4f} {f:>8.4f} {psi:>7.4f}")

    c8, s8, f8, psi8 = geo[8]
    c3, s3, f3, psi3 = geo[3]
    print(f"\n  P1 {'HELD' if (c8 >= 0.999 and s8 >= 0.95 and psi8 <= 0.05) else 'MISSED'}"
          f"  (hold=8: ceiling {c8:.4f}, suffix {s8:.4f}, PSI {psi8:.4f})")
    print(f"  P2 {'HELD' if psi3 >= 5 * psi8 else 'MISSED'}  "
          f"(PSI hold=3 / hold=8 = {psi3 / max(psi8, 1e-12):.1f}x)")

    # ---------------- 3. prefix identity vs un-converged noise ------------
    print("\n" + "=" * 70)
    print("(3) IS IT PREFIX IDENTITY OR UN-CONVERGED NOISE? -- P3's discriminator")
    n_pref, n_rep = 20, 12
    rg = np.random.default_rng(5)
    prefixes = rg.integers(0, V, (n_pref, T - 1))
    last = int(rg.integers(0, V))
    seqs = np.concatenate([np.repeat(prefixes, n_rep, axis=0),
                           np.full((n_pref * n_rep, 1), last)], axis=1)
    y = np.repeat(np.arange(n_pref), n_rep)
    NOISE = (0.0, 1e-3, 1e-2, 3e-2, 1e-1)
    print("  1-NN prefix-decoding accuracy at the final token, with the prefix")
    print("  signal's own magnitude in the organism's similarity units, and")
    print("  under added state noise -- exact arithmetic decodes what a")
    print("  THRESHOLDED consumer cannot necessarily use.")
    print(f"  {'hold':>5} | {'ceiling':>8} {'signal':>8} | "
          + " ".join(f"{'s=' + str(s):>9}" for s in NOISE)
          + f" | {'null p99':>9} | verdict")
    for h in HOLDS:
        Z = settle(seqs, emb, h, norm=NORM,
                   z0=rand_z(np.random.default_rng(77), len(seqs), N, NORM))
        accs = []
        for s in NOISE:
            rgn = np.random.default_rng(101)
            Zs = Z if s == 0 else Z + s * NORM / np.sqrt(N) * (
                rgn.standard_normal(Z.shape) + 1j * rgn.standard_normal(Z.shape))
            Zn = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-300)
            S = np.abs(Zn.conj() @ Zn.T)
            np.fill_diagonal(S, -1.0)
            nn = S.argmax(1)
            accs.append(float(np.mean(y[nn] == y)))
        # the null permutes the LABELS and re-scores the SAME neighbor
        # structure, so it kills exactly "any neighbor graph looks good"
        rgn = np.random.default_rng(9)
        null = np.array([(lambda p: np.mean(p[nn] == p))(rgn.permutation(y))
                         for _ in range(400)])
        p99 = float(np.percentile(null, 99))
        signal = geo[h][0] - geo[h][1]
        ok = accs[0] > p99 and geo[h][0] >= 0.90
        print(f"  {h:>5} | {geo[h][0]:>8.4f} {signal:>8.4f} | "
              + " ".join(f"{a:>9.3f}" for a in accs)
              + f" | {p99:>9.4f} | "
              + ("PREFIX IDENTITY" if ok else "not above null / ceiling collapsed"))

    # ---------------- 4. the cost on the project's own axes ---------------
    print("\n" + "=" * 70)
    print("(4) WHAT IT COSTS -- P4 (coverage and phase-24 category validity)")
    K_TXT = min(1200, V * 4)
    print(f"  {'hold':>5} | {'frames':>8} {'slots':>6} {'memories':>9} "
          f"{'coverage':>10} | {'k':>3} {'MI':>7} {'nullp99':>8} {'MIz':>7}")
    for h in HOLDS:
        org = PolysemyOrganism(N=N, K=K_TXT, omega=OMEGA, beta=BETA, seed=0)
        stream = R.stream(h)
        org.perceive(stream, **tr.PERCEIVE_19)
        org.consolidate(merge_thresh=0.84, prune_frac=0.001)
        n_mem = org.mem.shape[0]
        states = np.asarray([emb[w] for w in R.train_seq])
        assign = assign_slots(org.mem, states, N)
        slot_word = {}
        for k in range(n_mem):
            mem_of = np.asarray(R.train_seq)[assign == k]
            if len(mem_of):
                slot_word[k] = int(np.bincount(mem_of, minlength=V).argmax())
        cov = len(set(slot_word.values()))
        try:
            org.discover_categories_v2(verbose=False)
            kk, mi, p99, z = category_validity(org, assign, n_mem)
            cat = f"{kk:>3} {mi:>7.4f} {p99:>8.4f} {z:>7.1f}"
        except Exception as e:                      # a real outcome at low hold
            cat = f"  -       -        -        -   ({type(e).__name__})"
        print(f"  {h:>5} | {len(stream):>8} {int(org.used.sum()):>6} {n_mem:>9} "
              f"{cov:>4}/{V:<5} | {cat}")

    print(f"\nTOTAL {time.time() - t_all:.1f}s")
