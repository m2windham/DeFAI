"""Phase 57 -- does a PREDICATE-SET representation give the always-in-transit
symbol a stable identity, where a float vector could not?

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

BOOKKEEPING NOTE: this does NOT reopen Path A. Path A was retired on
2026-08-28 and this changes the representation rather than the oscillator
architecture -- it is the first test of the "port the ideas to a different
methodology" line. The oscillator field appears here only as the SOURCE OF
CONTAMINATION being tested against, never as the thing under test.

=============================================================================
THE CLAIM BEING TESTED
=============================================================================
Derivation, from the measured failures:

  Floats force identity to be a DISTANCE, distance needs a THRESHOLD, and the
  threshold is what fails for items whose signal is always polluted by their
  neighbours (phase 52: A0 occurs 1404 times and never gets a memory).

  If a token is instead a SET OF PREDICATES, identity is |A n B| -- an integer
  count, no threshold on a continuous quantity. And an item's identity can be
  accumulated as the RECURRENCE-FILTERED INTERSECTION of its occurrences:
  predicates driven by the item fire every time and survive; predicates driven
  by whichever neighbour happened to precede it differ each time and fall out
  by themselves. Contamination is filtered by the arithmetic rather than by an
  added mechanism.

FIRST FALSIFIABLE CONSEQUENCE: A0 should stabilize.

=============================================================================
FAIRNESS -- two things this had to get right or it would prove nothing
=============================================================================
(1) THE CONTAMINATION MUST BE REAL. Phase 52's raw emitted frames are a clean
    code plus iid noise, with NO cross-symbol pollution at the input; the
    contamination came from the FIELD DYNAMICS carrying a^hold = 0.168 of the
    previous symbol into every block. Encoding raw frames would be an easy
    test of nothing. So this replays the exact perceive recursion
    (dz = i*omega*z + g_in*(x - z), renormalized, at the committed
    g_in=4.0 / dt=0.05 / omega=0.15) and encodes the SETTLED STATE -- the
    same contaminated signal the organism recognizes on. Only the
    representation differs.

(2) THE ENCODER MUST BE LABEL-FREE AND GENERIC. It is a standard sparse LSH
    sketch: project onto R fixed random complex directions, keep the top-k by
    magnitude, emit (index, sign(Re), sign(Im)). It has no knowledge of the
    process, the appearances, or the number of items.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P0  PRECONDITION: the ORGANISM baseline on this same settled trace reproduces
    phase 52's failure (< 5 distinct slots for 5 appearances). If it does not,
    this is not testing the same phenomenon and no verdict may be read.

P1  DECISIVE: the predicate representation achieves INJECTIVITY -- all five
    appearances map to five distinct concepts -- on all five seeds.

P2  A0 in particular is acquired decisively, not by tipping a tie: its
    top-vs-second concept overlap margin >= 0.10 (the same bar phase 56 used,
    for comparability).

P3  The concept count does not explode: <= 15 concepts for 5 true items.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT (the one that would make a win meaningless)
=============================================================================
Sets simply beat floats at this noise level, and the RECURRENCE FILTER --
the specific mechanism the derivation rests on -- contributes nothing.

CONTROL, built in: an UNFILTERED arm using the same predicate sets and the
same online matching, but with identity = running UNION instead of the
recurrence-filtered intersection. If the unfiltered arm also reaches 5/5, M1
has FIRED: the result is about sets, not about the filter, and NEED 2's
mechanism is unsupported even though the outcome looks good.

M2 -- a second, quieter account: the LSH encoding is simply insensitive to a
perturbation as small as a^hold = 0.168, so nothing needed filtering.
DIAGNOSTIC: report A0's raw predicate-set stability across DIFFERENT
predecessors before any filtering. If those sets are already near-identical,
this protocol cannot speak to NEED 2 either way and says so.

=============================================================================
KILL RULE
=============================================================================
If P1 fails, changing the representation does NOT fix the transit problem,
the derivation's first claim is falsified, and the predicate-set line is
finished before it is built.
"""
import numpy as np
from phase51_causal_state_recruit import (N_DIM, HOLD, NOISE, hidden_walk,
                                          appearance_codebook, emit,
                                          N_APPEARANCE, STATE_APPEARANCE)

G_IN, DT, OMEGA = 4.0, 0.05, 0.15      # the committed regime
R_PROJ, TOP_K = 512, 32                # LSH sketch: 32 predicates per token
MATCH_BAR = 0.45                       # fraction of a concept's identity that
                                       # must be present to count as the same
RECUR = 0.60                           # a predicate is part of an identity if
                                       # it recurs in >= 60% of assignments
N_SYMBOLS = 6000


def settled_states(frames):
    """Replay perceive's field recursion and return the SETTLED state at the
    end of each symbol's frame block -- the contaminated signal the organism
    actually recognizes on."""
    z = np.zeros(N_DIM, dtype=complex)
    z[0] = 1.0
    z = z / np.linalg.norm(z) * np.sqrt(N_DIM)
    out = []
    for i, x in enumerate(frames):
        dz = 1j * OMEGA * z + G_IN * (x - z)
        z = z + DT * dz
        z = z / (np.linalg.norm(z) + 1e-9) * np.sqrt(N_DIM)
        if (i + 1) % HOLD == 0:
            out.append(z.copy())
    return np.asarray(out)


def sketch(Z, rng):
    """Label-free sparse LSH predicate sets. No knowledge of the process."""
    Rm = (rng.standard_normal((R_PROJ, N_DIM))
          + 1j * rng.standard_normal((R_PROJ, N_DIM)))
    Rm /= np.linalg.norm(Rm, axis=1, keepdims=True)
    P = Z @ Rm.conj().T                                  # (T, R_PROJ)
    sets = []
    for row in P:
        idx = np.argpartition(-np.abs(row), TOP_K)[:TOP_K]
        sets.append(frozenset(
            (int(i), int(np.sign(row[i].real)), int(np.sign(row[i].imag)))
            for i in idx))
    return sets


class Concepts:
    """Online, label-free, gradient-free identity formation over predicate
    sets. `filtered=True` is the derivation's mechanism: identity is the
    RECURRENCE-FILTERED intersection. `filtered=False` is M1's control:
    identity is the running union."""

    def __init__(self, filtered=True):
        self.filtered = filtered
        self.counts, self.n = [], []      # per concept: predicate counter, #assignments

    def identity(self, c):
        if not self.filtered:
            return set(self.counts[c].keys())            # union
        need = max(1.0, RECUR * self.n[c])
        return {p for p, v in self.counts[c].items() if v >= need}

    def match(self, S):
        best, score = -1, 0.0
        for c in range(len(self.counts)):
            ident = self.identity(c)
            if not ident:
                continue
            s = len(S & ident) / len(ident)
            if s > score:
                best, score = c, s
        return best, score

    def observe(self, S):
        c, s = self.match(S)
        if c < 0 or s < MATCH_BAR:
            self.counts.append({}); self.n.append(0); c = len(self.counts) - 1
        for p in S:
            self.counts[c][p] = self.counts[c].get(p, 0) + 1
        self.n[c] += 1
        return c


def run_seed(seed):
    from organism import Organism
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    st = hidden_walk(N_SYMBOLS, rng)
    frames, app = emit(st, code, rng)

    Z = settled_states(frames)
    app = app[:len(Z)]
    sets = sketch(Z, np.random.default_rng(90210 + 555 + seed))

    out = {}

    # --- P0: the ORGANISM on the same stream (phase 52's failure) ---------
    org = Organism(N=N_DIM, K=24, seed=seed, backend="numpy")
    org.perceive(frames)
    live = np.flatnonzero(org.used)
    o = np.abs(np.asarray([code[a] for a in range(N_APPEARANCE)])
               @ org.xi[live].conj().T) / N_DIM
    m_org = {a: int(live[np.argmax(o[a])]) for a in range(N_APPEARANCE)}
    row0 = np.sort(o[0])[::-1]
    out["ORGANISM (float vector)"] = dict(
        distinct=len(set(m_org.values())), n_units=len(live),
        a0_margin=float(row0[0] - row0[1]), mapping=m_org)

    # --- the predicate arms ----------------------------------------------
    for tag, filt in (("PREDICATE (filtered)", True), ("PREDICATE (unfiltered)", False)):
        C = Concepts(filtered=filt)
        assign = np.array([C.observe(S) for S in sets])
        # each appearance's dominant concept, and A0's margin
        mp, margins = {}, {}
        for a in range(N_APPEARANCE):
            mask = app == a
            if not mask.any():
                continue
            scores = []
            for c in range(len(C.counts)):
                ident = C.identity(c)
                if not ident:
                    scores.append(0.0); continue
                scores.append(float(np.mean([len(S & ident) / len(ident)
                                             for S in np.array(sets, dtype=object)[mask]])))
            scores = np.array(scores)
            mp[a] = int(np.argmax(scores))
            top = np.sort(scores)[::-1]
            margins[a] = float(top[0] - top[1]) if len(top) > 1 else 0.0
        out[tag] = dict(distinct=len(set(mp.values())), n_units=len(C.counts),
                        a0_margin=margins.get(0, 0.0), mapping=mp)

    # --- M2 diagnostic: is A0 even contaminated in this encoding? ---------
    A = np.array([STATE_APPEARANCE[s] for s in st])[:len(Z)]
    idx0 = [i for i in range(1, len(A)) if A[i] == 0]
    byprev = {}
    for i in idx0:
        byprev.setdefault(A[i - 1], []).append(sets[i])
    within, across = [], []
    keys = sorted(byprev)
    for k in keys:
        g = byprev[k][:60]
        for i in range(0, len(g) - 1, 2):
            within.append(len(g[i] & g[i + 1]) / TOP_K)
    if len(keys) >= 2:
        g1, g2 = byprev[keys[0]][:60], byprev[keys[1]][:60]
        for a_, b_ in zip(g1, g2):
            across.append(len(a_ & b_) / TOP_K)
    out["_m2"] = (float(np.mean(within)) if within else np.nan,
                  float(np.mean(across)) if across else np.nan)
    return out


def main():
    print("=" * 78)
    print("PHASE 57 -- predicate-set identity vs float-vector identity")
    print("Does the always-in-transit symbol stabilize when the REPRESENTATION")
    print("changes, on the SAME contaminated signal?")
    print(f"LSH sketch: {TOP_K} of {R_PROJ} projections; recurrence filter {RECUR}")
    print("=" * 78)
    seeds = range(5, 10)
    R = {s: run_seed(s) for s in seeds}
    arms = ("ORGANISM (float vector)", "PREDICATE (filtered)", "PREDICATE (unfiltered)")

    print(f"\n{'arm':<26}{'units':>7}{'distinct/5':>12}{'A0 margin':>12}   mapping (seed 5)")
    D = {}
    for a in arms:
        d = np.array([R[s][a]["distinct"] for s in seeds])
        u = np.mean([R[s][a]["n_units"] for s in seeds])
        g = np.array([R[s][a]["a0_margin"] for s in seeds])
        D[a] = (d, g)
        print(f"{a:<26}{u:>7.1f}{d.mean():>12.1f}{g.mean():>12.4f}   {R[5][a]['mapping']}")

    print("\n--- P0 PRECONDITION: does the organism reproduce phase 52? " + "-" * 14)
    d_org = D["ORGANISM (float vector)"][0]
    p0 = (d_org < N_APPEARANCE).all()
    print(f"  organism distinct {d_org.tolist()}  -> "
          f"{'reproduces the failure' if p0 else 'DOES NOT -- verdict withheld'}")

    print("\n--- P1 DECISIVE: does the predicate representation get injectivity? " + "-" * 5)
    d_f = D["PREDICATE (filtered)"][0]
    p1 = (d_f == N_APPEARANCE).all()
    print(f"  filtered distinct {d_f.tolist()}  ({int((d_f==5).sum())}/5 seeds at 5/5)")
    print(f"  P1: {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2: is A0 acquired decisively? " + "-" * 38)
    g_f = D["PREDICATE (filtered)"][1]
    p2 = (g_f >= 0.10).all()
    print(f"  A0 margin {g_f.mean():.4f}  (>=0.10 on {int((g_f>=0.10).sum())}/5)")
    print(f"  P2: {'HELD' if p2 else 'FAILED'}")

    print("\n--- P3: did the concept count explode? " + "-" * 34)
    u_f = np.mean([R[s]["PREDICATE (filtered)"]["n_units"] for s in seeds])
    print(f"  concepts {u_f:.1f} for 5 true items -> {'HELD' if u_f <= 15 else 'FAILED'}")

    print("\n--- M1: is it the SETS, or the RECURRENCE FILTER? " + "-" * 23)
    d_u = D["PREDICATE (unfiltered)"][0]
    print(f"  unfiltered (running union) distinct {d_u.tolist()}")
    if (d_u == N_APPEARANCE).all():
        print("  M1 FIRED -- the union arm also reaches 5/5, so the filter is NOT")
        print("  the mechanism. Any win here is about sets, not about NEED 2.")
    else:
        print("  M1 rejected -- sets alone do not do it; the filter is load-bearing.")

    print("\n--- M2: was A0 even contaminated in this encoding? " + "-" * 22)
    w = np.mean([R[s]["_m2"][0] for s in seeds])
    a_ = np.mean([R[s]["_m2"][1] for s in seeds])
    print(f"  A0 predicate overlap, SAME predecessor: {w:.4f}")
    print(f"  A0 predicate overlap, DIFFERENT predecessor: {a_:.4f}")
    if abs(w - a_) < 0.02:
        print("  Sets are already near-identical across predecessors -> this")
        print("  protocol CANNOT speak to NEED 2 either way. Report as such.")
    else:
        print(f"  Contamination is visible in the encoding (gap {w - a_:+.4f}),")
        print("  so the filter had something real to remove.")

    print("\n--- KILL RULE " + "-" * 58)
    if not p0:
        print("  P0 failed -> VOID, no verdict.")
    elif not p1:
        print("  P1 FAILED -> changing the representation does NOT fix the transit")
        print("  problem. The derivation's first claim is falsified and the")
        print("  predicate-set line is finished before it is built.")
    else:
        print("  P1 held: the transit problem is representational, not inherent.")
    print("=" * 78)


if __name__ == "__main__":
    main()
