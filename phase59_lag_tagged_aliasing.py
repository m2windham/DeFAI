"""Phase 59 -- can ONE appearance carry TWO identities? Lag-tagged predicates
against the aliasing case.

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

=============================================================================
THE TENSION THIS PROBES, AND IT IS THE INTERESTING PART
=============================================================================
Phase 57 stabilized A0 by REMOVING context: predicates driven by whichever
neighbour preceded the item differ each time and fall out of the intersection.

But the aliasing case needs the opposite. Appearance A1 is emitted by two
hidden states with DIFFERENT futures, and the only thing distinguishing them
IS the predecessor. A mechanism that removes context to stabilize A0 should
MERGE A1's two senses -- which is the arity conflict (P1) reappearing one
level down, at the representation instead of at the dynamics.

The derivation's answer (NEED 3) is that order is REPRESENTATIONAL, not
dynamical: a predicate is a pair (feature, lag), so the predecessor's
predicates are IN the token at lag 1 rather than smeared into it. Then
removing contamination and preserving context are not the same operation --
lag-0 predicates stabilize the item, lag-1 predicates distinguish its senses,
and the recurrence filter keeps whichever of the two actually recurs.

This phase tests whether that is true or merely tidy.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  A1 SPLITS: its occurrences are dominated by >= 2 concepts, and those
    concepts align with the hidden states s1/s2 -- conditional purity >= 0.80,
    on all five seeds. (Alignment is scored eval-only against ground truth;
    no label enters the mechanism.)
P2  NON-ALIASED appearances do NOT over-split: A0/A2/A3/A4 each keep >= 0.70
    of their occurrences in a single dominant concept. A mechanism that
    splits everything has not discovered aliasing, it has just fragmented.
P3  The resulting partition beats the APPEARANCE partition on held-out
    predictive information, by >= 0.20 bits. Phase 51 measured +0.372 for the
    ideal causal partition; this is the same axis, so the numbers are
    comparable and a much smaller gain would mean the split is cosmetic.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT
=============================================================================
Adding lag-1 predicates doubles the token size, so any gain is extra
information generally rather than order specifically. CONTROL: a
DOUBLE-WIDTH lag-0 arm -- 2x the projections, all at lag 0, same token size,
no history. If it splits A1 too, M1 has fired.

=============================================================================
KILL RULE
=============================================================================
If P1 fails, the representation cannot hold two identities behind one
appearance, and it solves the transit problem at the cost of the polysemy
problem -- which would make it a worse substrate than the one it replaced,
since phase 51 showed the criterion needs both.
"""
import numpy as np
from phase51_causal_state_recruit import (hidden_walk, appearance_codebook, emit,
                                          N_APPEARANCE, STATE_APPEARANCE,
                                          predictive_information)
from phase57_predicate_identity import settled_states, sketch, Concepts, N_SYMBOLS
import phase57_predicate_identity as p57


def lag_tag(sets, lags=1):
    """Order as REPRESENTATION: a predicate is (feature, lag). The
    predecessor's predicates are IN the token, not smeared into it."""
    out = []
    for t in range(len(sets)):
        S = {(p, 0) for p in sets[t]}
        for L in range(1, lags + 1):
            if t - L >= 0:
                S |= {(p, L) for p in sets[t - L]}
        out.append(frozenset(S))
    return out


def run_seed(seed, arm):
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    st = hidden_walk(N_SYMBOLS, rng)
    frames, app = emit(st, code, rng)
    Z = settled_states(frames)
    st = st[:len(Z)]; app = app[:len(Z)]

    if arm == "LAG-TAGGED":
        base = sketch(Z, np.random.default_rng(90210 + 555 + seed))
        sets = lag_tag(base, lags=1)
    else:                                   # M1: double width, all lag 0
        old_k = p57.TOP_K
        p57.TOP_K = old_k * 2
        try:
            sets = sketch(Z, np.random.default_rng(90210 + 555 + seed))
        finally:
            p57.TOP_K = old_k

    C = Concepts(filtered=True)
    assign = np.array([C.observe(S) for S in sets])

    # --- P1: does A1 split, and do the pieces align with s1/s2? ----------
    m1 = app == 1
    a1_assign, a1_state = assign[m1], st[m1]
    counts = np.bincount(a1_assign)
    dom = np.argsort(-counts)[:2]
    dom = [int(d) for d in dom if counts[d] > 0.05 * m1.sum()]
    purity = 0.0
    if len(dom) >= 2:
        tot = 0
        for d in dom:
            sub = a1_state[a1_assign == d]
            if len(sub):
                purity += np.bincount(sub, minlength=6).max(); tot += len(sub)
        purity = purity / max(tot, 1)

    # --- P2: do the non-aliased appearances stay unified? ----------------
    unified = []
    for a in (0, 2, 3, 4):
        m = app == a
        if m.any():
            c = np.bincount(assign[m])
            unified.append(c.max() / m.sum())

    # --- P3: predictive information vs the appearance partition ----------
    lab_c, lab_a, nxt = assign[1:-1], app[1:-1], app[2:]
    pi_c, _ = predictive_information(lab_c, nxt, N_APPEARANCE)
    pi_a, _ = predictive_information(lab_a, nxt, N_APPEARANCE)
    return dict(n_dom=len(dom), purity=float(purity),
                unified=float(np.min(unified)) if unified else 0.0,
                pi_concept=pi_c, pi_appearance=pi_a, units=len(C.counts))


def main():
    print("=" * 78)
    print("PHASE 59 -- can one appearance carry TWO identities?")
    print("Removing context stabilizes A0 (phase 57); the aliasing case needs")
    print("context PRESERVED. Order as representation (lag tags) is the claim.")
    print("=" * 78)
    seeds = range(5, 10)
    R = {a: {s: run_seed(s, a) for s in seeds}
         for a in ("LAG-TAGGED", "DOUBLE-WIDTH lag-0 (M1)")}

    print(f"\n{'arm':<26}{'units':>7}{'A1 concepts':>13}{'purity':>9}"
          f"{'min unified':>13}{'PI gain':>10}")
    for a in R:
        r = R[a]
        nd = np.array([r[s]["n_dom"] for s in seeds])
        pu = np.array([r[s]["purity"] for s in seeds])
        un = np.array([r[s]["unified"] for s in seeds])
        gp = np.array([r[s]["pi_concept"] - r[s]["pi_appearance"] for s in seeds])
        u = np.mean([r[s]["units"] for s in seeds])
        print(f"{a:<26}{u:>7.1f}{nd.mean():>13.1f}{pu.mean():>9.4f}"
              f"{un.mean():>13.4f}{gp.mean():>+10.4f}")

    lt = R["LAG-TAGGED"]
    nd = np.array([lt[s]["n_dom"] for s in seeds])
    pu = np.array([lt[s]["purity"] for s in seeds])
    un = np.array([lt[s]["unified"] for s in seeds])
    gp = np.array([lt[s]["pi_concept"] - lt[s]["pi_appearance"] for s in seeds])

    print("\n--- P1: does A1 split into senses aligned with the hidden states? " + "-" * 6)
    p1 = (nd >= 2).all() and (pu >= 0.80).all()
    print(f"  dominant concepts {nd.tolist()}, purity {pu.mean():.4f} "
          f"(>=0.80 on {int((pu>=0.80).sum())}/5)  -> {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2: do the non-aliased appearances stay unified? " + "-" * 20)
    p2 = (un >= 0.70).all()
    print(f"  worst unified coverage {un.mean():.4f} "
          f"(>=0.70 on {int((un>=0.70).sum())}/5)  -> {'HELD' if p2 else 'FAILED'}")

    print("\n--- P3: predictive information vs the appearance partition " + "-" * 14)
    p3 = (gp > 0).all() and gp.mean() >= 0.20
    print(f"  gain {gp.mean():+.4f} bits, positive {int((gp>0).sum())}/5 "
          f"(phase 51's ideal causal partition got +0.372)  -> "
          f"{'HELD' if p3 else 'FAILED'}")

    print("\n--- M1: is it the LAG TAGS or just a bigger token? " + "-" * 23)
    dw = R["DOUBLE-WIDTH lag-0 (M1)"]
    nd2 = np.array([dw[s]["n_dom"] for s in seeds])
    pu2 = np.array([dw[s]["purity"] for s in seeds])
    print(f"  double-width lag-0: A1 concepts {nd2.mean():.1f}, purity {pu2.mean():.4f}")
    if (nd2 >= 2).all() and (pu2 >= 0.80).all():
        print("  M1 FIRED -- a same-size token with no history splits A1 too.")
    else:
        print("  M1 rejected -- token size alone does not split A1; the lag tags do.")

    print("\n--- KILL RULE " + "-" * 58)
    if not p1:
        print("  P1 FAILED -> the representation cannot hold two identities behind")
        print("  one appearance. It fixes transit at the cost of polysemy, which")
        print("  makes it a WORSE substrate than the one it replaced.")
    else:
        print("  P1 held: one appearance can carry two identities. Removing")
        print("  contamination and preserving context are separable operations.")
    print("=" * 78)


if __name__ == "__main__":
    main()
