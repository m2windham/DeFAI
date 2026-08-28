"""Phase 53 -- is the generative path's evidence-blindness measurable?

PRE-REGISTERED 2026-08-28 before the run, per SOP rule 6.

=============================================================================
THE FINDING THIS TESTS
=============================================================================
A compute-and-use audit (owner, 2026-08-28) traced every consumer of the
transition graph and found `TransitionGraph.confidence` -- the Wilson lower
bound written precisely because "a symbol visited once, whose one successor
was j, gets Pn = 1.0" -- reaches only `edge_quality(weights='lcb')` and
therefore only `plan_reliable` / `plan_visit` / `path_report`.

Every GENERATIVE consumer runs on raw MLE: `rollout`, `kstep`, `recall`,
`recall2`, `recall_directed`, `next_hops`/`plan`, `MacroGraph` ranking, and
`SparseTransitions`. In `recall` the MLE row is multiplied by gamma=2.5 and
added to the overlap score, so a single observed transition out of a
once-visited memory produces a FULL-STRENGTH pull on the field.

That is a code-reading, not a measurement. This phase measures it.

Protocol-level only: no library file is touched. `rollout` computes its own
`normalized(idx)`, and `edge_quality(idx, weights=...)` is public, so both
weightings are reachable without changing a default.

=============================================================================
THE STREAM
=============================================================================
A dominant cycle plus ONE spurious edge out of a rarely-visited node -- the
exact shape `confidence`'s docstring names. The spurious node is visited a
handful of times and every one of its observations goes the same way, so MLE
assigns it probability 1.0 while the evidence behind that 1.0 is trivial.

=============================================================================
PRE-REGISTERED PREDICTIONS
=============================================================================
P1  Under MLE, symbolic rollouts traverse the spurious edge materially more
    often than under LCB. Predicted relative reduction >= 30%, sign-consistent
    on all five held-out seeds. If it fails, evidence-blindness is a real
    property of the code with no measurable consequence on generation, and
    that is worth knowing and recording.

P2  LCB does NOT pay for this by degrading traversal of well-evidenced edges:
    the dominant-cycle traversal rate falls by < 10% relative. A method that
    fixes the spurious edge by suppressing everything has not fixed anything.

=============================================================================
M1 -- THE NAMED MUNDANE ACCOUNT
=============================================================================
LCB shrinks EVERY edge, so any drop in spurious traversal is just uniform
conservatism rather than evidence-sensitivity. P2 is the control that catches
this, and it is pre-registered rather than added afterwards. If P1 holds and
P2 fails, M1 has fired and the result is reported as such.

=============================================================================
KILL RULE
=============================================================================
If the spurious edge is traversed at indistinguishable rates under both
weightings, the audit's finding is a code-reading with no behavioral
consequence on this protocol. It stays recorded as a code fact and NOT as a
defect, and no default should change on its account.

Paired seeds: fit s=0-4, held out s=5-9. At n=5 the smallest attainable
two-sided sign-test p is 2/32 = 0.0625; sign census is the primary evidence.
"""
import numpy as np
from organism import TransitionGraph

K = 8
DOMINANT = [0, 1, 2, 3, 4, 0]      # the well-evidenced cycle
RARE_NODE = 6                       # visited a handful of times
SPURIOUS_TO = 7                     # its single observed successor
N_CYCLES = 400
N_RARE = 3                          # <-- the whole point: 3 observations, Pn = 1.0
STEPS = 60
N_ROLLOUTS = 400


def build_graph(seed):
    g = TransitionGraph(K)
    rng = np.random.default_rng(seed)
    for _ in range(N_CYCLES):
        for a, b in zip(DOMINANT[:-1], DOMINANT[1:]):
            g.observe(a, b)
        # the dominant cycle occasionally passes through the rare node
        if rng.random() < 0.02:
            g.observe(4, RARE_NODE)
    for _ in range(N_RARE):
        g.observe(RARE_NODE, SPURIOUS_TO)   # every observation agrees -> MLE 1.0
    g.observe(SPURIOUS_TO, 0)
    return g


def traversals(g, weights, seed):
    """Sample rollouts under a given edge weighting and count how often the
    spurious edge and the dominant cycle are traversed."""
    idx = list(range(K))
    W = g.edge_quality(idx, weights=weights)
    rng = np.random.default_rng(1000 + seed)
    spur = dom = 0
    for _ in range(N_ROLLOUTS):
        cur = 0
        for _ in range(STEPS):
            row = np.asarray(W[cur], dtype=float)
            s = row.sum()
            if s <= 0:
                break
            nxt = int(rng.choice(len(row), p=row / s))
            if cur == RARE_NODE and nxt == SPURIOUS_TO:
                spur += 1
            if cur in DOMINANT[:-1] and nxt == DOMINANT[DOMINANT.index(cur) + 1]:
                dom += 1
            cur = nxt
    return spur, dom


def main():
    print("=" * 78)
    print("PHASE 53 -- evidence-blind generation: MLE vs Wilson-LCB on rollouts")
    print(f"rare node {RARE_NODE} observed {N_RARE}x, all agreeing -> MLE gives it 1.0")
    print("=" * 78)

    rows = {}
    for tag, seeds in (("SELECTION", range(0, 5)), ("HELD-OUT", range(5, 10))):
        sp_m, sp_l, dm_m, dm_l = [], [], [], []
        for s in seeds:
            g = build_graph(s)
            a, b = traversals(g, "mle", s)
            c, d = traversals(g, "lcb", s)
            sp_m.append(a); dm_m.append(b); sp_l.append(c); dm_l.append(d)
        rows[tag] = tuple(np.array(x) for x in (sp_m, sp_l, dm_m, dm_l))
        sm, sl, dm, dl = rows[tag]
        print(f"\n--- {tag} (seeds {list(seeds)}) " + "-" * 30)
        print(f"{'weighting':<14}{'spurious traversals':>22}{'dominant traversals':>22}")
        print(f"{'MLE':<14}{sm.mean():>22.1f}{dm.mean():>22.1f}")
        print(f"{'LCB':<14}{sl.mean():>22.1f}{dl.mean():>22.1f}")

    sm, sl, dm, dl = rows["HELD-OUT"]
    print("\n--- P1: spurious-edge traversal, MLE vs LCB (held-out) " + "-" * 15)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(sm > 0, (sm - sl) / np.maximum(sm, 1e-9), 0.0)
    print(f"  MLE {sm.mean():.1f} -> LCB {sl.mean():.1f}   "
          f"relative reduction {100*rel.mean():+.1f}%   reduced {int((sl<sm).sum())}/5")
    p1 = (sl < sm).all() and rel.mean() >= 0.30
    print(f"  P1 (>=30% reduction, all 5): {'HELD' if p1 else 'FAILED'}")

    print("\n--- P2 / M1: does LCB pay by suppressing GOOD edges? " + "-" * 17)
    with np.errstate(divide='ignore', invalid='ignore'):
        drel = np.where(dm > 0, (dm - dl) / np.maximum(dm, 1e-9), 0.0)
    print(f"  dominant-cycle traversals MLE {dm.mean():.1f} -> LCB {dl.mean():.1f}"
          f"   relative change {100*drel.mean():+.1f}%")
    p2 = abs(drel.mean()) < 0.10
    print(f"  P2 (<10% relative loss on well-evidenced edges): "
          f"{'HELD' if p2 else 'FAILED -- M1 FIRED, this is uniform conservatism'}")

    print("\n--- KILL RULE " + "-" * 58)
    if (sl == sm).all():
        print("  Indistinguishable. The audit finding is a CODE FACT with no")
        print("  behavioral consequence here. No default should change on it.")
    else:
        print("  Not triggered: the weighting demonstrably changes generation.")

    print("\n" + "=" * 78)
    print("SCOPE: a constructed graph with one deliberately under-evidenced edge.")
    print("Measures that the weighting MATTERS where confidence says it should,")
    print("not how often real corpora contain such edges. Protocol-level: no")
    print("library default was changed, and none is recommended here.")
    print("=" * 78)


if __name__ == "__main__":
    main()
