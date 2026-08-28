"""Phase 56 -- DECISION EXPERIMENT 3 of 3: does a slow companion state give
the always-in-transit symbol a memory?

PRE-REGISTERED 2026-08-28 before any run, per SOP rule 6.

=============================================================================
THE BLOCKER THIS ATTACKS
=============================================================================
Phase 52 went VOID because the organism gives NO SLOT AT ALL to appearance
A0. At K=24 on an exactly orthonormal codebook it is smeared across three
slots (0.602 / 0.560 / 0.521) while A1-A4 each own one cleanly, and the
collision persists at K=12, 16 and 24, so it is not capacity.

A0 is the one symbol that is always in transit -- three predecessors, a
deterministic successor -- so the 0.168 of the previous symbol the field
carries into every block lands it somewhere different every time.

=============================================================================
WHAT IS BEING TESTED, AND WHAT IS NOT NOVEL ABOUT IT
=============================================================================
The memory-nonlinearity trade-off (Inubushi & Yoshimura, Sci. Rep. 2017) is
a general result: nonlinear dynamics degrade stored memory regardless of the
form of the nonlinearity. DeFAI's settling IS its nonlinearity, so phase 52's
blocker is that theorem's prediction rather than a tuning failure. The remedy
that paper proposes is a MIXTURE reservoir carrying both linear and nonlinear
dynamics.

**This arm is therefore published engineering and must not be claimed as
novel.** T8.1's re-scope says so and this phase repeats it. What is being
tested is not the idea; it is whether the idea removes THIS blocker.

Protocol-level: no library file is touched. The companion is built as a
stream transformation -- each frame is presented as [nonlinear channel |
slow linear channel], the slow channel being an EMA of history at a longer
time constant. The organism then runs unmodified on the 2N-dimensional
stream, and its nonlinear settling operates on the first half while the
second half carries a linear trace the settling cannot erase.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  INJECTIVITY IS RESTORED: all five appearances land on distinct slots, on
    all five held-out seeds. This is the decisive clause -- it is exactly
    phase 52's failed P0.
P2  A0 in particular acquires a slot whose overlap with A0's clean code
    exceeds its overlap with every other slot by a clear margin (>= 0.10),
    rather than merely tipping a three-way tie.
P3  The baseline reproduces phase 52's failure on this host: without the
    companion, injectivity is 4/5 or worse. If the baseline does NOT fail,
    this phase is measuring a different setup than phase 52 and says so.

=============================================================================
M1 -- NAMED MUNDANE ACCOUNT
=============================================================================
Injectivity is restored merely because the input is now 2N-dimensional, so
every appearance is easier to separate for reasons having nothing to do with
memory. CONTROL: a WIDTH-MATCHED arm at the same 2N, where the second half
is the frame's own copy rather than a slow trace -- same dimensionality, no
history. If the width-matched control also restores injectivity, M1 has
fired and the companion's SLOWNESS bought nothing.

=============================================================================
KILL RULE
=============================================================================
If P1 fails, a linear companion does not fix the transit problem, phase 52's
blocker survives the published remedy for it, and Gate 0's cheapest arm is
spent. Report that plainly; it is a strong negative and it materially raises
the cost of any rebuild.
"""
import numpy as np
from phase51_causal_state_recruit import (N_DIM, hidden_walk, appearance_codebook,
                                          emit, N_APPEARANCE, HOLD, NOISE)

K = 24
N_TRAIN = 6000
TAU_SLOW = 40.0      # frames; ~8x the nonlinear channel's 1/(g_in*dt) = 5


def build_streams(seed):
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    st = hidden_walk(N_TRAIN, rng)
    frames, app = emit(st, code, rng)
    F = np.asarray(frames)
    N = F.shape[1]

    # slow linear channel: an EMA over history at a long time constant. This
    # is the LINEAR half of a mixture reservoir; it is not settled and not
    # renormalized per frame, so the nonlinearity cannot erase it.
    a = np.exp(-1.0 / TAU_SLOW)
    slow = np.empty_like(F)
    acc = np.zeros(N, dtype=complex)
    for i in range(len(F)):
        acc = a * acc + (1 - a) * F[i]
        slow[i] = acc

    def norm2(X):
        return X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(2 * N)

    mixture = norm2(np.concatenate([F, slow], axis=1))
    widthmatched = norm2(np.concatenate([F, F], axis=1))   # M1's control
    return code, app, list(F), list(mixture), list(widthmatched), N


def injectivity(org, code, N, doubled):
    live = np.flatnonzero(org.used)
    if len(live) == 0:
        return 0, {}, 0.0
    probe = np.asarray([code[a] for a in range(N_APPEARANCE)])
    if doubled:                       # probe the same way the stream was built
        probe = np.concatenate([probe, probe], axis=1)
        probe = probe / np.linalg.norm(probe, axis=1, keepdims=True) * np.sqrt(2 * N)
    o = np.abs(probe @ org.xi[live].conj().T) / org.xi.shape[1]
    m = {a: int(live[np.argmax(o[a])]) for a in range(N_APPEARANCE)}
    row0 = np.sort(o[0])[::-1]
    margin = float(row0[0] - row0[1]) if len(row0) > 1 else 0.0
    return len(set(m.values())), m, margin


def run_seed(seed):
    from organism import Organism
    code, app, base, mix, wm, N = build_streams(seed)
    out = {}
    for tag, stream, doubled in (("BASELINE", base, False),
                                 ("MIXTURE", mix, True),
                                 ("WIDTH-MATCHED", wm, True)):
        dim = 2 * N if doubled else N
        org = Organism(N=dim, K=K, seed=seed, backend="numpy")
        org.perceive(stream)
        n, m, margin = injectivity(org, code, N, doubled)
        out[tag] = dict(n_distinct=n, mapping=m, a0_margin=margin,
                        live=int(org.used.sum()))
    return out


def main():
    print("=" * 78)
    print("PHASE 56 -- does a slow companion state give the always-in-transit")
    print("symbol a memory?   DECISION EXPERIMENT 3 of 3")
    print(f"tau_slow = {TAU_SLOW} frames vs the nonlinear channel's ~5")
    print("NOT NOVEL: mixture reservoirs are Inubushi & Yoshimura 2017.")
    print("=" * 78)
    hout = range(5, 10)
    R = {s: run_seed(s) for s in hout}

    print(f"\n{'arm':<16}{'live':>6}{'distinct/5':>12}{'A0 margin':>12}   mapping (seed 5)")
    for tag in ("BASELINE", "MIXTURE", "WIDTH-MATCHED"):
        nd = np.array([R[s][tag]["n_distinct"] for s in hout])
        mg = np.array([R[s][tag]["a0_margin"] for s in hout])
        lv = np.mean([R[s][tag]["live"] for s in hout])
        print(f"{tag:<16}{lv:>6.1f}{nd.mean():>12.1f}{mg.mean():>12.4f}"
              f"   {R[5][tag]['mapping']}")

    nd_b = np.array([R[s]["BASELINE"]["n_distinct"] for s in hout])
    nd_m = np.array([R[s]["MIXTURE"]["n_distinct"] for s in hout])
    nd_w = np.array([R[s]["WIDTH-MATCHED"]["n_distinct"] for s in hout])
    mg_m = np.array([R[s]["MIXTURE"]["a0_margin"] for s in hout])

    print("\n--- P3: does the BASELINE reproduce phase 52's failure? " + "-" * 18)
    p3 = (nd_b < N_APPEARANCE).all()
    print(f"  distinct slots {nd_b.tolist()}  -> "
          f"{'reproduces the failure' if p3 else 'DOES NOT -- different setup than phase 52'}")

    print("\n--- P1 (DECISIVE): does the companion restore injectivity? " + "-" * 14)
    p1 = (nd_m == N_APPEARANCE).all()
    print(f"  MIXTURE distinct slots {nd_m.tolist()}  ({int((nd_m==5).sum())}/5 seeds at 5/5)")
    print(f"  P1: {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2: does A0 get a slot decisively, not by tipping a tie? " + "-" * 13)
    p2 = (mg_m >= 0.10).all()
    print(f"  A0 top-vs-second margin {mg_m.mean():.4f}  (>=0.10 on "
          f"{int((mg_m>=0.10).sum())}/5)")
    print(f"  P2: {'HELD' if p2 else 'FAILED'}")

    print("\n--- M1: is it the SLOWNESS or just the extra width? " + "-" * 22)
    print(f"  WIDTH-MATCHED distinct slots {nd_w.tolist()}")
    if (nd_w == N_APPEARANCE).all():
        print("  M1 FIRED -- doubling the width alone restores injectivity, so")
        print("  the companion's slowness bought nothing. NOT evidence for a")
        print("  mixture reservoir.")
    else:
        print("  M1 rejected -- extra width alone does not restore injectivity.")

    print("\n--- KILL RULE " + "-" * 58)
    if not p1:
        print("  P1 FAILED -> a linear companion does NOT fix the transit")
        print("  problem. Phase 52's blocker survives the published remedy for")
        print("  it, Gate 0's cheapest arm is spent, and any rebuild costs more")
        print("  than the literature suggested.")
    else:
        print("  P1 held: the transit problem is addressable, and phase 52 can")
        print("  be re-run on a substrate that represents its own stream.")
    print("=" * 78)


if __name__ == "__main__":
    main()
