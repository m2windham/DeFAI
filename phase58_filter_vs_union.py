"""Phase 58 -- is it the SETS or the RECURRENCE FILTER? M1, properly specified.

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

Phase 57 confirmed its decisive claim (P1: predicate sets reach 5/5
injectivity where the float vector reaches 4/5) but its M1 control FIRED --
the unfiltered running-union arm also reached 5/5, so phase 57 did not
establish that the recurrence filter is the mechanism.

That firing was real and is not reinterpreted here. What phase 57 also
recorded is that the union arm reached 5/5 with 131.8 concepts and an A0
margin of 0.0588, against the filtered arm's 6.6 and 0.5224 -- i.e. it may
have achieved injectivity by FRAGMENTING into ~132 units of which five
happened to dominate. Phase 57's pre-registered criterion (distinct == 5)
could not tell "formed five identities" from "fragmented into 132".

This phase specifies the control properly and re-runs it. The discriminating
measurement is COVERAGE: of all occurrences of an appearance, what fraction
land in that appearance's own dominant concept? Forming an identity means
high coverage; fragmenting means low coverage even at perfect injectivity.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  FILTERED coverage >= 0.80 on all five seeds. If identity formation is
    real, most occurrences of a thing land in that thing's concept.
P2  UNION coverage < 0.50, i.e. materially worse -- confirming that its 5/5
    injectivity in phase 57 came from fragmentation rather than from forming
    identities.
P3  Compactness separates them: FILTERED <= 15 concepts, UNION >= 50.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT
=============================================================================
Coverage differs only because the two arms recruit at different RATES, so
this measures the match bar rather than the filter. CONTROL: report both
arms' concept counts alongside coverage, and additionally run the union arm
at a RELAXED match bar tuned to produce a comparable concept count. If a
union arm with comparable compactness also reaches high coverage, the filter
is not the mechanism and M1 stands.

=============================================================================
KILL RULE
=============================================================================
If FILTERED and UNION reach comparable coverage at comparable compactness,
the recurrence filter contributes nothing, NEED 2's mechanism is dropped
from the derivation, and the phase 57 result stands as "sets beat floats"
with no explanation attached.
"""
import numpy as np
from phase51_causal_state_recruit import (hidden_walk, appearance_codebook, emit,
                                          N_APPEARANCE)
from phase57_predicate_identity import (settled_states, sketch, Concepts,
                                        N_SYMBOLS, MATCH_BAR)


def coverage_and_units(app, assign):
    """Of all occurrences of each appearance, what fraction land in that
    appearance's own dominant concept? Fragmentation shows up here even when
    injectivity is perfect."""
    covs, dom = [], {}
    for a in range(N_APPEARANCE):
        m = app == a
        if not m.any():
            continue
        c = np.bincount(assign[m])
        dom[a] = int(np.argmax(c))
        covs.append(c.max() / m.sum())
    return float(np.mean(covs)), len(set(dom.values())), dom


def run_seed(seed, bars):
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    st = hidden_walk(N_SYMBOLS, rng)
    frames, app = emit(st, code, rng)
    Z = settled_states(frames); app = app[:len(Z)]
    sets = sketch(Z, np.random.default_rng(90210 + 555 + seed))

    out = {}
    for tag, filt, bar in bars:
        C = Concepts(filtered=filt)
        global MATCH_BAR
        import phase57_predicate_identity as p57
        old = p57.MATCH_BAR
        p57.MATCH_BAR = bar
        try:
            assign = np.array([C.observe(S) for S in sets])
        finally:
            p57.MATCH_BAR = old
        cov, distinct, _ = coverage_and_units(app, assign)
        out[tag] = dict(coverage=cov, distinct=distinct, units=len(C.counts))
    return out


def main():
    print("=" * 78)
    print("PHASE 58 -- filter vs union, with the control specified properly")
    print("Discriminating measure: COVERAGE (did it FORM identities, or")
    print("fragment into many units of which a few happen to dominate?)")
    print("=" * 78)
    seeds = range(5, 10)
    bars = (("FILTERED", True, 0.45),
            ("UNION", False, 0.45),
            ("UNION (relaxed bar)", False, 0.12))   # M1's compactness-matched arm
    R = {s: run_seed(s, bars) for s in seeds}

    print(f"\n{'arm':<22}{'units':>8}{'distinct/5':>12}{'coverage':>11}")
    D = {}
    for tag, _, _ in bars:
        u = np.array([R[s][tag]["units"] for s in seeds])
        d = np.array([R[s][tag]["distinct"] for s in seeds])
        c = np.array([R[s][tag]["coverage"] for s in seeds])
        D[tag] = (u, d, c)
        print(f"{tag:<22}{u.mean():>8.1f}{d.mean():>12.1f}{c.mean():>11.4f}")

    uf, df, cf = D["FILTERED"]
    uu, du, cu = D["UNION"]
    ur, dr, cr = D["UNION (relaxed bar)"]

    print("\n--- P1: does the FILTERED arm actually form identities? " + "-" * 17)
    p1 = (cf >= 0.80).all()
    print(f"  coverage {cf.mean():.4f}, >=0.80 on {int((cf>=0.80).sum())}/5  "
          f"-> {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2: was the UNION arm's 5/5 fragmentation? " + "-" * 26)
    p2 = (cu < 0.50).all()
    print(f"  coverage {cu.mean():.4f}, <0.50 on {int((cu<0.50).sum())}/5  "
          f"-> {'HELD' if p2 else 'FAILED'}")

    print("\n--- P3: compactness " + "-" * 53)
    p3 = (uf <= 15).all() and (uu >= 50).all()
    print(f"  filtered {uf.mean():.1f} units, union {uu.mean():.1f}  "
          f"-> {'HELD' if p3 else 'FAILED'}")

    print("\n--- M1: a UNION arm matched for compactness " + "-" * 29)
    print(f"  union at a relaxed bar: {ur.mean():.1f} units, "
          f"coverage {cr.mean():.4f}, distinct {dr.mean():.1f}")
    m1_fires = (cr >= 0.80).all() and (ur <= 15).all()
    if m1_fires:
        print("  M1 STANDS -- a union arm with comparable compactness also forms")
        print("  identities. The recurrence filter is NOT the mechanism.")
    else:
        print("  M1 rejected -- the union arm cannot reach filtered-level coverage")
        print("  at comparable compactness. The filter is load-bearing.")

    print("\n--- KILL RULE " + "-" * 58)
    if m1_fires:
        print("  Filter contributes nothing -> DROP NEED 2's mechanism from the")
        print("  derivation. Phase 57 stands as 'sets beat floats', unexplained.")
    else:
        print("  Filter is load-bearing: identity formation is a real mechanism,")
        print("  not an artifact of fragmentation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
