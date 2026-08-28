"""Phase 52 (T8.1) -- the causal-state criterion INSIDE recruit, online.

PRE-REGISTERED 2026-08-28, BEFORE ANY RUN. Committed before the run that
tests it, per SOP rule 6.

=============================================================================
WHAT THIS ADDS OVER PHASE 51
=============================================================================
Phase 51 compared partitions POST HOC on a trained bank and found the causal
partition strictly better (+0.372 bits held-out, 5/5, loss localized exactly
to the aliased symbol). It explicitly did NOT show the criterion can be used
ONLINE, and said so.

This does. `perceive(recruit_mode='predictive')` -- additive, DEFAULT OFF,
numpy-only, verified bitwise identical to origin/main on the default path --
asks at the recruit site: appearance says refine slot k, but given this
context does k predict a different future than k does on average? If yes,
this is a different causal state wearing the same appearance, and it earns
its own slot.

Bookkeeping is SPARSE by construction -- dicts over (prev, slot) pairs that
actually occur, never a dense K^3 tensor. That is what keeps this off phase
40's rock: phase 40 priced a dense third-order tensor at ~1.4M entries at
K=112 and called it dead on arrival. It was right about the dense form.

=============================================================================
OUTCOME 2026-08-28: **VOID ON ITS OWN PRECONDITION.** Read before the
predictions below; they were never adjudicable.
=============================================================================
The phase does not report a result because its substrate precondition fails,
and the failure is informative.

P0 (added after the first runs, and it should have been here from the start):
the organism must form ONE SLOT PER APPEARANCE before a context-conditioned
split of those slots can mean anything. It does not.

Measured, K=24, orthogonal codebook (Gram matrix exactly the identity):
  A0 -> s3 0.602 | s2 0.560 | s1 0.521      <- no slot of its own
  A1 -> s1 0.828     A2 -> s2 0.697
  A3 -> s3 0.664     A4 -> s4 0.695
A0 is SMEARED across three slots. The collision persists at K=12, 16 and 24,
so it is not capacity.

WHY, and this is the finding: A0 is the one symbol that is always in transit.
It has three predecessors and a deterministic successor, so its frame blocks
are always entered from a different symbol. The field carries a^hold = 0.168
of the previous symbol into every block, so A0's settled state is a different
blend every time and never condenses into an attractor. **This is the settling
problem appearing as a missing memory** -- not an abstraction about phase, an
absent slot for a symbol that occurs 1404 times.

TWO ERRORS OF MINE, both recorded rather than quietly fixed:
 (1) The first run reported 0 splits because `_note` was called with the
     predecessor and slot arguments swapped, so the table was keyed backwards
     from the lookup and no split could fire. Fixed; it was not the only
     problem, and the run was void either way.
 (2) I set K_BASE=12 "deliberately tight" and dropped phase 51's P4 guard,
     which existed to catch exactly this class of failure.

**CORRECTION TO PHASE 51, which is the part that travels.** Phase 51's P4
reported "slot-argmax agreement with appearance labels: 1.0000" and I read it
as the organism having recovered the appearance partition. **It measures
PURITY, not INJECTIVITY.** If two appearances map to one slot, every probe of
each is still assigned consistently, so purity is 1.0 while the partition has
silently merged two symbols. P4 cannot detect a merge and did not.

Phase 51's headline SURVIVES, because its APPEARANCE arm used ground-truth
appearance ids rather than the organism's slots, so what it compared was the
idealized appearance partition against the causal one. But phase 51 therefore
never tested the organism's REAL partition, P4 was the check that was supposed
to bridge that gap, and it does not. Any future use of that agreement number
must check injectivity as well.

=============================================================================
PRE-REGISTERED PREDICTIONS (never adjudicated -- P0 failed first)
=============================================================================
P1  ONLINE beats APPEARANCE on held-out predictive information, on every
    paired seed, margin >= 0.20 bits. Weaker than phase 51's >= 0.30 because
    an online estimator sees less evidence per decision than a batch one and
    must pay for its own bootstrapping. If P1 fails, the criterion does not
    survive being made causal in time, and phase 51 stands as a batch-only
    result.

P2  The split it finds is the ALIASED appearance and essentially only that.
    Extra slots spent should be small: predicted <= 3 above the appearance
    arm's slot count. A method that splits everything buys predictive
    information by cardinality, which is the T1.5 confound, not the mechanism.

P3  The default path is untouched: bitwise identical to origin/main. Verified
    before this run and pinned in the harness, not asserted here.

=============================================================================
M1 -- THE NAMED MUNDANE ACCOUNT
=============================================================================
The online arm wins purely by SPENDING MORE SLOTS. Splitting always refines a
partition, and a finer partition trivially predicts better (T1.5 measured
exactly this; phase 41 was built around the same control).

CONTROL, built in and not added afterwards: APPEARANCE+K, an appearance arm
given the SAME number of slots the online arm ended up using. If the online
arm does not beat that, M1 has fired and the result is negative -- and will be
reported that way.

=============================================================================
KILL RULE (pre-named)
=============================================================================
If ONLINE does not beat APPEARANCE+K on held-out, sign-consistent across all
five held-out seeds, the mechanism buys nothing beyond capacity and
recruit_mode='predictive' should be REMOVED rather than kept as a default-off
option. Adjudicated out loud either way.

Seeds: fit s=0-4, held out s=5-9, paired. At n=5 the smallest attainable
two-sided sign-test p is 2/32 = 0.0625; sign census is the primary evidence.
"""
import os
import numpy as np
from phase51_causal_state_recruit import (
    N_DIM, HOLD, NOISE, STATE_APPEARANCE, N_APPEARANCE, T, TRUE_CAUSAL,
    hidden_walk, appearance_codebook, emit, predictive_information, _entropy)

N_SYMBOLS_TRAIN = 6000
N_SYMBOLS_TEST = 6000
K_BASE = 12          # deliberately tight: splitting must EARN its slots


def _partition_from_slots(org, frames_app, code, live):
    o = np.abs(np.asarray([code[a] for a in frames_app]) @ org.xi[live].conj().T) / N_DIM
    return live[np.argmax(o, axis=1)]


def run_seed(seed):
    from organism import Organism
    rng = np.random.default_rng(90210 + seed)
    code = appearance_codebook(np.random.default_rng(90210 + 7000 + seed))
    tr_states = hidden_walk(N_SYMBOLS_TRAIN, rng)
    te_states = hidden_walk(N_SYMBOLS_TEST, rng)
    tr_frames, tr_app = emit(tr_states, code, rng)
    te_frames, te_app = emit(te_states, code, rng)
    te_a, te_p, te_n = te_app[1:-1], te_app[:-2], te_app[2:]

    out = {}

    # --- APPEARANCE: the current recruit rule -----------------------------
    a_org = Organism(N=N_DIM, K=K_BASE, seed=seed, backend="numpy")
    a_org.perceive(tr_frames)
    a_live = np.flatnonzero(a_org.used)
    lab_a = _partition_from_slots(a_org, te_a, code, a_live)
    pi, hc = predictive_information(lab_a, te_n, N_APPEARANCE)
    out["APPEARANCE"] = dict(pi=pi, H_cond=hc, n_slots=len(a_live),
                             n_states=len(np.unique(lab_a)))

    # --- ONLINE: the criterion at the recruit site ------------------------
    o_org = Organism(N=N_DIM, K=K_BASE, seed=seed, backend="numpy")
    o_org.perceive(tr_frames, recruit_mode='predictive')
    o_live = np.flatnonzero(o_org.used)
    # route the query the way the mechanism routes: a split slot is reached
    # through its (prev, k) context, so score context-aware assignment
    base = _partition_from_slots(o_org, te_a, code, o_live)
    prev = np.concatenate([[-1], base[:-1]])
    lab_o = base.copy()
    for i, (p, b) in enumerate(zip(prev, base)):
        lab_o[i] = o_org.ctx_split.get((int(p), int(b)), b)
    pi, hc = predictive_information(lab_o, te_n, N_APPEARANCE)
    out["ONLINE"] = dict(pi=pi, H_cond=hc, n_slots=len(o_live),
                         n_states=len(np.unique(lab_o)),
                         n_splits=len(o_org.ctx_split))

    # --- APPEARANCE+K: M1's control, same slot budget ---------------------
    k_matched = max(K_BASE, len(o_live))
    m_org = Organism(N=N_DIM, K=k_matched, seed=seed, backend="numpy")
    m_org.perceive(tr_frames)
    m_live = np.flatnonzero(m_org.used)
    lab_m = _partition_from_slots(m_org, te_a, code, m_live)
    pi, hc = predictive_information(lab_m, te_n, N_APPEARANCE)
    out["APPEARANCE+K"] = dict(pi=pi, H_cond=hc, n_slots=len(m_live),
                               n_states=len(np.unique(lab_m)))

    # --- oracle ceiling ---------------------------------------------------
    lab_t = np.array([TRUE_CAUSAL[s] for s in te_states[1:-1]])
    pi, hc = predictive_information(lab_t, te_n, N_APPEARANCE)
    out["TRUE-CAUSAL (oracle)"] = dict(pi=pi, H_cond=hc, n_slots=0,
                                       n_states=len(np.unique(lab_t)))
    return out


def injectivity(org, code):
    """P0: distinct appearances must land on DISTINCT slots. Phase 51's P4
    measured purity, which is 1.0 even when two appearances share a slot."""
    live = np.flatnonzero(org.used)
    o = np.abs(np.asarray([code[a] for a in range(N_APPEARANCE)])
               @ org.xi[live].conj().T) / N_DIM
    m = {a: int(live[np.argmax(o[a])]) for a in range(N_APPEARANCE)}
    return len(set(m.values())), m


def main():
    print("=" * 78)
    print("PHASE 52 (T8.1) -- the causal-state criterion INSIDE recruit, online")
    print(f"backend=numpy (predictive mode is numpy-only by design)  "
          f"N={N_DIM} K_base={K_BASE}")
    print("=" * 78)
    # ---- P0 FIRST. If the substrate does not represent the stream, nothing
    # downstream is adjudicable and this phase must refuse to print a verdict.
    from organism import Organism
    rng0 = np.random.default_rng(90210 + 5)
    code0 = appearance_codebook(np.random.default_rng(90210 + 7005))
    st0 = hidden_walk(N_SYMBOLS_TRAIN, rng0)
    fr0, _ = emit(st0, code0, rng0)
    probe = Organism(N=N_DIM, K=24, seed=5, backend="numpy")
    probe.perceive(fr0)
    n_distinct, mapping = injectivity(probe, code0)
    print(f"\nP0 PRECONDITION -- one slot per appearance (K=24)")
    print(f"  appearance -> slot: {mapping}")
    print(f"  distinct slots: {n_distinct}/{N_APPEARANCE}")
    if n_distinct < N_APPEARANCE:
        print("  P0 FAILED -> PHASE VOID. The organism does not form one slot")
        print("  per appearance, so a context-conditioned split of its slots")
        print("  cannot be interpreted. See the OUTCOME block in this file's")
        print("  docstring: A0 is always in transit and never condenses.")
        print("  Reporting no verdict on P1/P2 or the kill rule -- a void is")
        print("  not a negative result and must not be recorded as one.")
        print("=" * 78)
        return
    sel, hout = range(0, 5), range(5, 10)
    R = {s: run_seed(s) for s in list(sel) + list(hout)}
    arms = ("APPEARANCE", "ONLINE", "APPEARANCE+K", "TRUE-CAUSAL (oracle)")

    def block(seeds, title):
        print(f"\n--- {title} (seeds {list(seeds)}) " + "-" * 26)
        print(f"{'arm':<24}{'slots':>7}{'states':>8}{'PI (bits)':>12}")
        rows = {}
        for a in arms:
            pis = np.array([R[s][a]["pi"] for s in seeds])
            print(f"{a:<24}{np.mean([R[s][a]['n_slots'] for s in seeds]):>7.1f}"
                  f"{np.mean([R[s][a]['n_states'] for s in seeds]):>8.1f}{pis.mean():>12.4f}")
            rows[a] = pis
        return rows

    rs, ro = block(sel, "SELECTION"), block(hout, "HELD-OUT")

    print("\n--- P1: ONLINE - APPEARANCE (held-out, paired) " + "-" * 24)
    d = ro["ONLINE"] - ro["APPEARANCE"]
    print(f"  mean {d.mean():+.4f} bits   positive {int((d>0).sum())}/5"
          f"   range [{d.min():+.4f}, {d.max():+.4f}]")
    print(f"  P1 (>=0.20 bits, all 5): {'HELD' if (d>0).all() and d.mean()>=0.20 else 'FAILED'}")

    print("\n--- KILL RULE / M1: ONLINE - APPEARANCE+K (same slot budget) " + "-" * 9)
    dk = ro["ONLINE"] - ro["APPEARANCE+K"]
    print(f"  mean {dk.mean():+.4f} bits   positive {int((dk>0).sum())}/5"
          f"   range [{dk.min():+.4f}, {dk.max():+.4f}]")
    if (dk > 0).all():
        print("  M1 REJECTED -- the gain is not capacity. Kill rule not triggered.")
    else:
        print("  M1 FIRED -- the gain is capacity alone. KILL RULE TRIGGERED:")
        print("  recruit_mode='predictive' should be REMOVED, not kept.")

    print("\n--- P2: what did it spend? " + "-" * 44)
    ns = np.mean([R[s]["ONLINE"]["n_splits"] for s in hout])
    extra = np.mean([R[s]["ONLINE"]["n_slots"] - R[s]["APPEARANCE"]["n_slots"] for s in hout])
    print(f"  context splits made: {ns:.1f}   extra slots vs APPEARANCE: {extra:+.1f}")
    print(f"  P2 (<= 3 extra slots): {'HELD' if extra <= 3 else 'FAILED'}")

    print("\n" + "=" * 78)
    print("SCOPE: same constructed stream as phase 51 -- the two equivalence")
    print("relations differ by construction. Shows the criterion SURVIVES being")
    print("made online and causal in time; not that real corpora need it.")
    print("=" * 78)


if __name__ == "__main__":
    main()
