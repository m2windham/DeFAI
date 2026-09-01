"""Phase 60 -- replace both tuned constants with measured nulls.

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

=============================================================================
WHAT THIS FIXES
=============================================================================
Phase 59 split the aliased appearance perfectly (purity 1.0000, +0.3739 bits,
matching phase 51's oracle partition) but FAILED P2: it over-split the
non-aliased appearances, worst unified coverage 0.4635 against a 0.70 bar.

And the arbitration rested on two tuned constants -- MATCH_BAR = 0.45 and
RECUR = 0.60 -- which undercut the derivation's own claim that a set
representation escapes thresholds. It does escape thresholds on continuous
MAGNITUDES; it did not escape tuned numbers.

Three nulls replace them. Each answers the question the constant was
guessing at:

  MATCH  "is this the same thing?"  ->  HYPERGEOMETRIC null. An observation
         of k predicates and an identity of m predicates, drawn from a
         universe of U, overlap by chance with mean km/U. Match on the
         concept whose overlap is most significant against that, and recruit
         only when even the best is not.

  IDENTITY  "which predicates belong to this concept?"  ->  BINOMIAL null
         against the predicate's own BASE RATE. A predicate driven by the
         item fires far more often inside its concept than it does at large;
         a predicate driven by whichever neighbour happened to precede the
         item fires at its base rate inside the concept too, and is dropped.
         The base rate is estimated online from the stream itself.

  SPLIT  "should this concept become two?"  ->  PREDICTIVE-GAIN null, which
         is phase 51's criterion and N1's: split only when conditioning on
         context measurably reduces successor entropy above a PERMUTATION
         null. This is the piece phase 59 did not have, and it is precisely
         the decision P2 failed.

The only number left is a significance level, which is a statistical
convention rather than an architectural guess -- and M1 below tests whether
it behaves like one.

HONEST SCOPE: concept formation is online; the split pass runs over
accumulated statistics, as phase 51's did. This is not a fully online
algorithm and does not claim to be.

IMPLEMENTATION NOTE (added after a first attempt timed out, recorded rather
than silently applied): each concept's identity set is CACHED and refreshed
every REFRESH assignments instead of being recomputed on every observation.
The criterion is unchanged -- this only changes how often it is re-evaluated,
so an identity can lag its exact value by at most REFRESH assignments. The
M1 sweep runs on three seeds rather than five, for cost.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  P2 FROM PHASE 59 IS FIXED: minimum unified coverage >= 0.70 on all five
    seeds, against the 0.4635 it failed at. This is the decisive clause.
P2  A1 STILL SPLITS: >= 2 dominant concepts aligned with the hidden states,
    purity >= 0.80. Fixing over-splitting by refusing to split anything is
    not a fix.
P3  COMPACT: <= 8 concepts (6 is ideal -- five appearances, one split).
P4  Held-out predictive information over the appearance partition >= 0.20
    bits, on the axis where phase 51's ideal causal partition scored +0.372.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT
=============================================================================
The nulls are just a better-tuned constant in disguise. If the result only
works at one significance level, nothing was gained.

CONTROL: sweep z_crit over 3, 4, 6, 8, 10. A principled criterion should be
STABLE across that range; a constant in disguise will be knife-edge. Reported
as a table, and if the result holds only at one value M1 has fired.

=============================================================================
KILL RULE
=============================================================================
If P1 fails, the arbitration between stabilizing an item and distinguishing
its senses is not solved by measured nulls, and this line needs a different
answer for it rather than a tuning pass.
"""
import numpy as np
from phase51_causal_state_recruit import (hidden_walk, appearance_codebook, emit,
                                          N_APPEARANCE, predictive_information,
                                          _entropy)
from phase57_predicate_identity import settled_states, sketch, N_SYMBOLS, R_PROJ
from phase59_lag_tagged_aliasing import lag_tag

UNIVERSE = R_PROJ * 4 * 2          # (index, sign_re, sign_im) x lag in {0,1}
Z_CRIT = 6.0
N_PERM = 300


class NullConcepts:
    """Identity formation with no tuned bars: matching by a hypergeometric
    null, identity by a per-predicate binomial null against its base rate."""

    def __init__(self, z_crit=Z_CRIT, universe=UNIVERSE, refresh=25):
        self.z, self.U = z_crit, universe
        self.counts, self.n = [], []
        self.base = {}          # predicate -> times seen anywhere
        self.total = 0
        self.refresh = refresh
        self._cache, self._stamp = [], []

    def identity(self, c):
        """Cached; refreshed every `refresh` assignments. Criterion unchanged."""
        if self._stamp[c] is not None and self.n[c] - self._stamp[c] < self.refresh:
            return self._cache[c]
        out = self._identity_exact(c)
        self._cache[c], self._stamp[c] = out, self.n[c]
        return out

    def _identity_exact(self, c):
        """Predicates whose within-concept rate beats their own base rate."""
        n = self.n[c]
        if n < 3:
            return set(self.counts[c].keys())
        out = set()
        for p, k in self.counts[c].items():
            q = self.base.get(p, 1) / max(self.total, 1)      # base rate
            mu, var = n * q, n * q * (1 - q)
            if var <= 0:
                continue
            if (k - mu) / np.sqrt(var) >= self.z:             # binomial null
                out.add(p)
        return out

    def _match_z(self, S, c):
        """Significance of overlap against a hypergeometric null."""
        I = self.identity(c)
        m, k = len(I), len(S)
        if m == 0 or k == 0:
            return -np.inf
        obs = len(S & I)
        mu = k * m / self.U
        var = mu * (1 - m / self.U) * (1 - k / self.U) * self.U / max(self.U - 1, 1)
        if var <= 0:
            return -np.inf
        return (obs - mu) / np.sqrt(var)

    def observe(self, S):
        best, bz = -1, -np.inf
        for c in range(len(self.counts)):
            z = self._match_z(S, c)
            if z > bz:
                best, bz = c, z
        if best < 0 or bz < self.z:                # not significantly anything
            self.counts.append({}); self.n.append(0)
            self._cache.append(set()); self._stamp.append(None)
            best = len(self.counts) - 1
        for p in S:
            self.counts[best][p] = self.counts[best].get(p, 0) + 1
            self.base[p] = self.base.get(p, 0) + 1
        self.n[best] += 1
        self.total += 1
        if self._stamp[best] is None:
            self._stamp[best] = -10**9      # force a compute on first query
        return best


def split_by_predictive_gain(assign, rng, z_crit=Z_CRIT, n_perm=N_PERM):
    """Phase 51's criterion as the SPLIT decision: a concept becomes two only
    when conditioning on the predecessor measurably reduces successor entropy
    above a permutation null. This is the piece phase 59 lacked."""
    a = np.asarray(assign)
    prev, cur, nxt = a[:-2], a[1:-1], a[2:]
    K = int(a.max()) + 1
    out = cur.copy()
    extra = {}
    for c in np.unique(cur):
        m = cur == c
        if m.sum() < 200:
            continue
        pv, nv = prev[m], nxt[m]
        H = _entropy(np.bincount(nv, minlength=K))

        def cond(pvals):
            h, n = 0.0, len(pvals)
            for p in np.unique(pvals):
                mm = pvals == p
                h += (mm.sum() / n) * _entropy(np.bincount(nv[mm], minlength=K))
            return h

        gain = H - cond(pv)
        null = np.array([H - cond(rng.permutation(pv)) for _ in range(n_perm)])
        sd = null.std()
        if sd <= 0 or (gain - null.mean()) / sd < z_crit:
            continue                                    # no evidence -> no split
        for p in np.unique(pv):                         # split by predecessor
            key = (int(c), int(p))
            if key not in extra:
                extra[key] = K + len(extra)
            out[np.flatnonzero(m)[pv == p]] = extra[key]
    return out


def run_seed(seed, z_crit=Z_CRIT):
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    st = hidden_walk(N_SYMBOLS, rng)
    frames, app = emit(st, code, rng)
    Z = settled_states(frames); st = st[:len(Z)]; app = app[:len(Z)]
    sets = lag_tag(sketch(Z, np.random.default_rng(90210 + 555 + seed)), lags=1)

    C = NullConcepts(z_crit=z_crit)
    base_assign = np.array([C.observe(S) for S in sets])
    assign = split_by_predictive_gain(base_assign,
                                      np.random.default_rng(90210 + 77 + seed),
                                      z_crit=z_crit)

    ap, sp = app[1:-1], st[1:-1]
    # unified coverage of the NON-ALIASED appearances
    unified = []
    for a_ in (0, 2, 3, 4):
        m = ap == a_
        if m.any():
            unified.append(np.bincount(assign[m]).max() / m.sum())
    # A1's split and its alignment with the hidden states
    m1 = ap == 1
    cnt = np.bincount(assign[m1])
    dom = [int(d) for d in np.argsort(-cnt)[:3] if cnt[d] > 0.05 * m1.sum()]
    pur, tot = 0.0, 0
    for d in dom:
        sub = sp[m1][assign[m1] == d]
        if len(sub):
            pur += np.bincount(sub, minlength=6).max(); tot += len(sub)
    pi_c, _ = predictive_information(assign, app[2:], N_APPEARANCE)
    pi_a, _ = predictive_information(ap, app[2:], N_APPEARANCE)
    return dict(unified=float(np.min(unified)) if unified else 0.0,
                n_dom=len(dom), purity=float(pur / max(tot, 1)),
                units=len(np.unique(assign)), pi_gain=pi_c - pi_a)


def main():
    print("=" * 78)
    print("PHASE 60 -- both tuned constants replaced by measured nulls")
    print("MATCH: hypergeometric | IDENTITY: binomial vs base rate | "
          "SPLIT: predictive gain")
    print("=" * 78)
    seeds = range(5, 10)
    R = {s: run_seed(s) for s in seeds}

    un = np.array([R[s]["unified"] for s in seeds])
    nd = np.array([R[s]["n_dom"] for s in seeds])
    pu = np.array([R[s]["purity"] for s in seeds])
    u = np.array([R[s]["units"] for s in seeds])
    gp = np.array([R[s]["pi_gain"] for s in seeds])

    print(f"\n{'units':>8}{'min unified':>13}{'A1 concepts':>13}{'purity':>9}{'PI gain':>10}")
    print(f"{u.mean():>8.1f}{un.mean():>13.4f}{nd.mean():>13.1f}"
          f"{pu.mean():>9.4f}{gp.mean():>+10.4f}")

    print("\n--- P1 DECISIVE: is phase 59's over-splitting fixed? " + "-" * 20)
    p1 = (un >= 0.70).all()
    print(f"  min unified coverage {un.mean():.4f} (phase 59: 0.4635), "
          f">=0.70 on {int((un>=0.70).sum())}/5  -> {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2: does A1 still split? " + "-" * 44)
    p2 = (nd >= 2).all() and (pu >= 0.80).all()
    print(f"  dominant concepts {nd.tolist()}, purity {pu.mean():.4f}  "
          f"-> {'HELD' if p2 else 'FAILED'}")

    print("\n--- P3: compactness " + "-" * 53)
    p3 = (u <= 8).all()
    print(f"  {u.mean():.1f} concepts (6 ideal)  -> {'HELD' if p3 else 'FAILED'}")

    print("\n--- P4: predictive information " + "-" * 42)
    p4 = (gp > 0).all() and gp.mean() >= 0.20
    print(f"  {gp.mean():+.4f} bits, positive {int((gp>0).sum())}/5 "
          f"(phase 51 oracle: +0.372)  -> {'HELD' if p4 else 'FAILED'}")

    print("\n--- M1: are the nulls a tuned constant in disguise? " + "-" * 21)
    print(f"{'z_crit':>8}{'units':>8}{'min unified':>13}{'purity':>9}{'PI gain':>10}")
    stable = []
    sweep_seeds = list(seeds)[:3]          # three seeds for cost; stated in docstring
    for zc in (3.0, 4.0, 6.0, 8.0, 10.0):
        rr = {s: run_seed(s, z_crit=zc) for s in sweep_seeds}
        uu = np.mean([rr[s]["units"] for s in sweep_seeds])
        un2 = np.mean([rr[s]["unified"] for s in sweep_seeds])
        pu2 = np.mean([rr[s]["purity"] for s in sweep_seeds])
        gp2 = np.mean([rr[s]["pi_gain"] for s in sweep_seeds])
        stable.append(un2 >= 0.70 and pu2 >= 0.80)
        print(f"{zc:>8.1f}{uu:>8.1f}{un2:>13.4f}{pu2:>9.4f}{gp2:>+10.4f}")
    if sum(stable) >= 4:
        print("  M1 rejected -- stable across the range; this behaves like a")
        print("  significance level, not like a tuned architectural constant.")
    else:
        print("  M1 FIRED -- the result is knife-edge in z_crit, so a constant")
        print("  was replaced by another constant wearing a null's clothes.")

    print("\n--- KILL RULE " + "-" * 58)
    if not p1:
        print("  P1 FAILED -> measured nulls do not solve the arbitration between")
        print("  stabilizing an item and distinguishing its senses. This line")
        print("  needs a different answer for it, not a tuning pass.")
    else:
        print("  P1 held: the arbitration is decidable from the data.")
    print("=" * 78)


if __name__ == "__main__":
    main()
