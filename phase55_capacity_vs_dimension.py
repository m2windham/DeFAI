"""Phase 55 -- DECISION EXPERIMENT 2 of 3: is the binding ceiling a
MECHANISM limit or a DIMENSION limit?

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

=============================================================================
THE QUESTION
=============================================================================
N8 (phase-superposition binding) is recorded as "partial -- unsolved", on
phase 16's measurement that superposition holds about FIVE items cleanly.
That number has always been quoted as a property of the MECHANISM.

VSA capacity theory says superposition capacity scales LINEARLY with
representation dimension, at roughly 0.5 bits per component. Phase 16 ran at
N=64. Standard VSA implementations run D = 10^3 - 10^4.

So the five-item ceiling may be arithmetic, not architecture -- and this
project has never once measured capacity against N. That is a cheap thing to
have never checked before recording a capability as unsolved.

=============================================================================
THE DESIGN
=============================================================================
Superpose m quasi-orthogonal complex vectors, then ask how many can be
recovered by nearest-neighbour against the codebook. Sweep N and m. Report
the capacity m* at which recovery falls below 0.95, per N.

No organism is involved: this is a property of the REPRESENTATION, and
pinning it to the mechanism would confound the question.

=============================================================================
FIRST RUN VOID, 2026-08-28 -- design error, corrected below before the
committed run. Recorded rather than quietly fixed.
=============================================================================
The first design used an EXACTLY ORTHONORMAL codebook, on the reasoning that
it removes chance orthogonality (M1). That reasoning was wrong twice:

 (1) **It caps the answer by construction.** At most N mutually orthonormal
     vectors exist in N dimensions, so the codebook -- and therefore m* --
     could never exceed N. The measured m* then saturated against the grid
     ceiling (128) rather than against any capacity, and reported "capacity
     is FLAT in N" as a KILL. That verdict was an artifact of my own limits
     and is withdrawn.
 (2) **It removes the mechanism, not a confound.** Quasi-orthogonality of
     random high-dimensional vectors IS what VSA capacity rests on. Testing
     it with an orthonormal basis tests something else.

P2 caught this, which is what P2 was for: m*(64) came out at 50 against
phase 16's ~5, i.e. the protocol was not measuring phase 16's quantity.

CORRECTED DESIGN: random unit-norm complex codebook (quasi-orthogonal, as
VSA specifies), of a size held FIXED across N -- which is the right control
for "did capacity rise, or did the task get easier" -- and a grid extended
far enough to find the ceiling rather than hit mine.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  m* grows with N, and grows APPROXIMATELY LINEARLY. Concretely: m*(512)
    >= 4 x m*(64), on all five seeds. If capacity is flat in N, the ceiling
    IS the mechanism and N8 stays unsolved for the reason recorded.
P2  m*(64) lands in the neighbourhood of phase 16's five items (predicted
    3-10), which is the check that this protocol measures the same thing
    phase 16 measured rather than a different quantity that happens to
    scale.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT
=============================================================================
Capacity rises with N only because a larger codebook of RANDOM vectors is
more orthogonal by chance, so the gain is the codebook getting easier rather
than superposition getting better. CONTROL: use an EXACTLY orthonormal
codebook (QR) at every N, which removes chance orthogonality entirely. If
capacity still scales, M1 is rejected.

=============================================================================
KILL RULE
=============================================================================
If m* is flat in N (m*(512) < 2 x m*(64)), the binding ceiling is
architectural, dimension will not buy it, and N8 stays closed on the terms
already recorded.
"""
import numpy as np

DIMS = (32, 64, 128, 256, 512, 1024, 2048)
MS = (2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 320, 500)
N_TRIALS = 200
CODEBOOK = 512          # FIXED across N: the control for "did the task get
                        # easier". Quasi-orthogonal random vectors, which is
                        # what VSA capacity actually rests on -- an
                        # orthonormal basis would cap the codebook at N and
                        # cap the answer with it (see the VOID note above).


def codebook(N, rng, orthonormal=False):
    M = (rng.standard_normal((CODEBOOK, N))
         + 1j * rng.standard_normal((CODEBOOK, N)))
    if orthonormal:                      # kept only for the reported contrast
        Q, _ = np.linalg.qr(M[:min(CODEBOOK, N)].T)
        return Q.T * np.sqrt(N)
    return M / np.linalg.norm(M, axis=1, keepdims=True) * np.sqrt(N)


def recovery(N, m, seed, orthonormal=False):
    """Superpose m items; fraction recovered by nearest-neighbour."""
    rng = np.random.default_rng(70000 + seed * 97 + N)
    C = codebook(N, rng, orthonormal)
    if m > len(C):
        return np.nan
    hit = tot = 0
    for _ in range(N_TRIALS):
        idx = rng.choice(len(C), size=m, replace=False)
        s = C[idx].sum(axis=0)
        s = s / np.linalg.norm(s) * np.sqrt(N)
        sim = np.abs(C @ s.conj()) / N
        top = np.argsort(-sim)[:m]
        hit += len(set(top.tolist()) & set(idx.tolist())); tot += m
    return hit / tot


def capacity(N, seed, orthonormal=False, bar=0.95):
    """Largest m whose recovery is still >= bar."""
    best = 0
    for m in MS:
        r = recovery(N, m, seed, orthonormal)
        if np.isnan(r):
            break
        if r >= bar:
            best = m
        else:
            break
    return best


def main():
    print("=" * 78)
    print("PHASE 55 -- binding ceiling: mechanism limit or dimension limit?")
    print(f"DECISION EXPERIMENT 2 of 3   (bar 0.95, quasi-orthogonal codebook of {CODEBOOK}, fixed across N)")
    print("=" * 78)
    seeds = range(0, 5)

    print(f"\n{'N':>6}{'m* (mean)':>12}{'per-seed':>26}{'m*/N':>10}")
    caps = {}
    for N in DIMS:
        cs = np.array([capacity(N, s) for s in seeds])
        caps[N] = cs
        print(f"{N:>6}{cs.mean():>12.1f}{str(cs.tolist()):>26}{cs.mean()/N:>10.4f}")

    print("\n--- P1: does capacity scale with dimension? " + "-" * 32)
    lo, hi = caps[64], caps[512]
    ratio = hi / np.maximum(lo, 1)
    print(f"  m*(64) = {lo.mean():.1f}   m*(512) = {hi.mean():.1f}"
          f"   ratio {ratio.mean():.2f}x   >=4x on {int((ratio>=4).sum())}/5 seeds")
    p1 = (ratio >= 4).all()
    print(f"  P1 (m*(512) >= 4x m*(64), all seeds): {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2: does m*(64) match phase 16's ~5 items? " + "-" * 29)
    ok2 = 3 <= caps[64].mean() <= 10
    print(f"  m*(64) = {caps[64].mean():.1f}  (phase 16 measured ~5)")
    print(f"  P2 (3-10): {'HELD -- same quantity' if ok2 else 'FAILED -- this may not be what phase 16 measured'}")

    print("\n--- M1: did capacity rise, or did the task get easier? " + "-" * 21)
    print(f"  The codebook is held FIXED at {CODEBOOK} items across every N, so")
    print("  the retrieval problem is identical at each dimension and only the")
    print("  representation width changes. M1 is controlled by construction.")
    orth = np.array([capacity(512, s, orthonormal=True) for s in seeds])
    print(f"  reported contrast, orthonormal basis at N=512: m* = {orth.mean():.1f}")
    print("  (that arm is capped at N by construction -- see the VOID note)")

    print("\n--- KILL RULE " + "-" * 58)
    if not p1:
        print("  Capacity is FLAT in N -> the binding ceiling is architectural.")
        print("  Dimension will not buy it and N8 stays closed as recorded.")
    else:
        print("  Capacity scales with N -> phase 16's five items is an artifact")
        print("  of N=64, not a property of the mechanism. N8's recorded reason")
        print("  needs editing.")
    print("\nSCOPE: this measures the REPRESENTATION's static superposition")
    print("capacity. Phase 16 also measured LIFETIME (collapse in 9-56 steps")
    print("under attractor pull), which is a dynamics property and is NOT")
    print("addressed here. A capacity result does not rescue lifetime.")
    print("=" * 78)


if __name__ == "__main__":
    main()
