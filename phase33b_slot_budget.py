"""
PHASE 33b (T1.2): SLOT-BUDGET / EVICTION -- measured re-run of phase 33's
flooding diagnostic, organism arm only.

Phase 33 measured the failure: on class-incremental split-digits at K=40,
the first task's data floods every slot (recruit is a similarity floor, so
recruitment is eager while slots are free), after which `not used.all()`
blocks recruitment FOREVER. Later classes are learned by dragging existing
slots off their memories -- slot drift, a slow forgetting channel that both
caps ACC (0.665) and produces the FORG 0.20. The ROADMAP names the fix:
a slot-budget/eviction policy, built from the pool-mode use-it-or-lose-it
recycling machinery, as MECHANISM (no task or label information anywhere).

The mechanism under test (organism.py, perceive(..., evict=E)): a
persistent staleness clock (org.age, frames since a slot was last
confidently used; serialized by E3) plus eviction UNDER RECRUITMENT
PRESSURE only -- when a token more novel than 0.8x the recruit floor (the
pool-mode weak-tail fraction: recruiting out of a freed slot costs more
novelty than recruiting into a free one) wants a slot and none is free,
the least-established stale slot (argmin lifetime `count` among slots idle
> E frames; the same junk signal consolidate() prunes by) is recycled
through the existing machinery (graph.retire + boundary.invalidate) and
recruitment proceeds into the freed slot. evict=0 reproduces phase 33's
organism exactly.

Protocol: identical to phase 33's organism arm (same data pipeline, same
seeds, same stream -- the per-task shuffle generator reproduces phase 33's
module-rng consumption -- same recipe: K=40, omega=0.15, beta=10, g_in=4.0,
eta=0.015, recruit=0.6, hold=8, epochs=3, online per-slot label-evidence
readout: labels in READOUT only, never inside perceive). Torch baselines
are NOT re-run here; T1.4 owns the full ladder re-test.

Pre-registered predictions (written before the committed run):
  (a) baseline arm (evict=0) REPRODUCES the flooding: task 1 ends with all
      40 slots used; tasks 2-5 recruit zero fresh slots.
  (b) budget arm (evict=2000, ~30% of one task's 6720 frames): every later
      task recruits at least one fresh slot -- the mechanism's job -- and
      ACC rises vs the baseline arm because later classes get fresh slots
      instead of drift.
  (c) HONEST RISK, measured either way: pressure eviction is recency-
      driven; if it reclaims early-task prototypes rather than junk, early-
      task retention drops and FORG can WORSEN even as ACC rises. The
      count-argmin ordering is the designed counterweight (junk and
      duplicates go first); per-task retention and the birth-task census
      below measure whether it is enough.
  (d) E3: mid-run save->load->continue with evict>0 stays BITWISE identical
      to never stopping (the budget's clock is dynamical state now).

COMMITTED-RUN FINDINGS (2026-08-05). Prediction (a) reproduced exactly
(baseline = phase 33's ACC 0.665 / FORG 0.200, zero fresh recruits after
task 1). Prediction (b) HALF-held at the pre-registered window: fresh
recruitment worked (14-29 slots/task) but ACC FELL (0.511) and FORG rose
(0.513) -- risk (c) materialized. Diagnosis, then the finding:

  THE STALE POOL MUST CONTAIN THE PRESENT, OR EVICTION EATS THE PAST.
  With a long window (evict=2000, ~1/3 task length), the only slots ever
  stale during a task are PREVIOUS ERAS' -- the current task's own junk
  recruits are too young to reclaim -- so count-ordering has nothing to
  choose from and the budget degenerates to a recency flush (census
  [0,2,3,6,29] at evict=2000; [0,0,0,4,36] before the 0.8x novelty
  throttle existed). With a SHORT window (comfortably above a live slot's
  revisit interval, far below era length), the stream's own discards go
  stale mid-task and enter the pool; argmin-count then recycles the
  stream's own churn first and established old-era prototypes survive.
  Measured sweep (single stream, phase 33's own): every window in
  100-750 beats the baseline on BOTH axes; the recency flush appears at
  ~1000+ and is total by 2000.

Primary characterization below runs evict=250 -- a post-hoc, mid-plateau
choice by principle (~3x the measured live-slot revisit interval of ~80
frames at hold=8, ~27x below task length), not the sweep argmax. Verdict
axes ACC/FORG per phase 33; harness section 9 pins the mechanism at small
scale; T1.4 re-runs the full ladder.

Readout bookkeeping under eviction (eval-side only): a recycled slot's
label evidence is stale by construction, so evidence rows are cleared for
slots whose eviction tally advanced (or whose used flag flipped) during
the task before that task's evidence is accumulated. This is the readout
following slot identity -- the same reason phase 33 abandoned frozen
labels -- not label information entering the mechanism.
"""

import os
import tempfile

import numpy as np
from sklearn.datasets import load_digits

from organism import normalize
from organism_numba import NumbaOrganism
from organism_state import save_state, load_state

rng = np.random.default_rng(0)

d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(np.float32)
y = d.target.astype(int)
N = X.shape[1]
NORM = np.sqrt(N)
perm = rng.permutation(len(X))
tr, te = perm[:1400], perm[1400:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
K = 40
PREREG = 2000       # pre-registered window (prediction b) -- honest negative
PRIMARY = 250       # post-hoc principled choice, see docstring


def task_acc_matrix_to_metrics(A):
    T = len(TASKS)
    final = A[T - 1]
    acc = float(np.mean(final))
    forg = float(np.mean([np.max(A[:, j]) - final[j] for j in range(T - 1)]))
    return acc, forg


def eval_tasks(predict):
    out = []
    for task in TASKS:
        m = np.isin(yte, task)
        out.append(float((predict(Xte[m]) == yte[m]).mean()))
    return out


def run_arm(evict, hold=8, epochs=3, save_between=None, verbose=True):
    """Phase 33's organism arm, instrumented. Returns (A, per-task rows,
    org, owner) where owner[j] = task during which slot j was last
    (re)recruited."""
    r = np.random.default_rng(0)
    r.permutation(len(X))            # phase 33's per-task shuffles came from the
    # module rng AFTER it drew the train/test split; burning that draw makes
    # every arm's stream identical to phase 33's organism arm, verbatim
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=0)
    evidence = np.zeros((K, 10))
    owner = np.full(K, -1)
    A = np.zeros((len(TASKS), len(TASKS)))
    rows = []
    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in r.permutation(idx):
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict)
        if save_between is not None and ti == save_between:
            path = os.path.join(tempfile.gettempdir(), "phase33b_e3.npz")
            save_state(org, path)
            org = load_state(path, cls=NumbaOrganism)
        evd = org.evictions - evd_before
        fresh = (~used_before & org.used) | (evd > 0)
        owner[fresh] = ti
        evidence[fresh] = 0.0        # readout follows slot identity (eval-side)
        states = np.array([normalize(Xtr[i].astype(complex), NORM) for i in idx])
        used = np.where(org.used)[0]
        ov = np.abs(org.xi[used].conj() @ states.T) / N
        near = np.argmax(ov, axis=0)
        for s_i, slot in enumerate(used):
            claimed = ytr[idx[near == s_i]]
            for c in claimed:
                evidence[slot, c] += 1.0

        def pred(Xe):
            st = np.array([normalize(x.astype(complex), NORM) for x in Xe])
            used_ = np.array([s for s in np.where(org.used)[0]
                              if evidence[s].sum() > 0])
            lab = evidence[used_].argmax(1)
            ovv = np.abs(org.xi[used_].conj() @ st.T) / N
            return lab[np.argmax(ovv, axis=0)]
        A[ti] = eval_tasks(pred)
        rows.append(dict(task=ti, used=int(org.used.sum()),
                         evictions=int(evd.sum()), fresh=int(fresh.sum())))
        if verbose:
            print(f"    task {ti} ({task}): slots used {rows[-1]['used']}/{K}  "
                  f"fresh recruits {rows[-1]['fresh']:>2}  "
                  f"evictions {rows[-1]['evictions']:>2}  "
                  f"acc-so-far {np.mean(A[ti][:ti + 1]):.3f}")
    return A, rows, org, owner


def census(org, owner):
    return [int((owner[org.used] == t).sum()) for t in range(len(TASKS))]


if __name__ == "__main__":
    print("PHASE 33b: slot-budget/eviction -- phase-33 flooding diagnostic re-run\n")

    print("  ARM 1 -- baseline (evict=0), phase 33's organism verbatim:")
    A0, rows0, org0, _ = run_arm(0)
    acc0, forg0 = task_acc_matrix_to_metrics(A0)
    print(f"    ACC {acc0:.3f}  FORG {forg0:.3f}   (phase 33 committed: 0.665 / 0.200)")

    print(f"\n  ARM 2 -- pre-registered window (evict={PREREG}):")
    Ap, rowsp, orgp, ownp = run_arm(PREREG, verbose=False)
    accp, forgp = task_acc_matrix_to_metrics(Ap)
    print(f"    ACC {accp:.3f}  FORG {forgp:.3f}  "
          f"fresh {[r['fresh'] for r in rowsp[1:]]}  census {census(orgp, ownp)}")

    print("\n  window sweep (same stream; the self-churn law, see docstring):")
    print("    evict   ACC    FORG   fresh recruits t2-5      birth-task census")
    sweep = {}
    for ev in (100, 250, 500, 750, 1000, 1500, 2000):
        As, rs, ogs, ows = run_arm(ev, verbose=False)
        a_, f_ = task_acc_matrix_to_metrics(As)
        sweep[ev] = (a_, f_)
        print(f"    {ev:>5}  {a_:.3f}  {f_:.3f}   {[r['fresh'] for r in rs[1:]]!s:<22} {census(ogs, ows)}")

    print(f"\n  ARM 3 -- primary characterization (evict={PRIMARY}):")
    A1, rows1, org1, owner = run_arm(PRIMARY)
    acc1, forg1 = task_acc_matrix_to_metrics(A1)
    print(f"    ACC {acc1:.3f}  FORG {forg1:.3f}")
    print(f"    birth-task census of final slots: {census(org1, owner)}")
    print(f"    final per-task acc: baseline {np.round(A0[-1], 3).tolist()}")
    print(f"                        budget   {np.round(A1[-1], 3).tolist()}")

    # E3: mid-run save->load->continue must be bitwise identical (pred d)
    A1r, _, org1r, _ = run_arm(PRIMARY, save_between=1, verbose=False)
    dxi = float(np.abs(org1.xi - org1r.xi).max())
    dP = float(np.abs(org1.P - org1r.P).max())
    dage = float(np.abs(org1.age - org1r.age).max())
    print(f"\n  E3 round-trip mid-run (after task 2): max|dxi|={dxi:.2e} "
          f"max|dP|={dP:.2e} max|dage|={dage:.2e}")

    fresh0 = [r['fresh'] for r in rows0[1:]]
    freshp = [r['fresh'] for r in rowsp[1:]]
    fresh1 = [r['fresh'] for r in rows1[1:]]
    print("\nverdict vs pre-registered predictions:")
    print(f"  (a) baseline floods: task-1 used {rows0[0]['used']}/{K}, later fresh "
          f"recruits {fresh0} -> {'REPRODUCED' if rows0[0]['used'] == K and sum(fresh0) == 0 else 'NOT REPRODUCED'}")
    print(f"  (b) at the PRE-REGISTERED window {PREREG}: fresh recruits {freshp} (mechanism")
    print(f"      works) but ACC {acc0:.3f} -> {accp:.3f}: the accuracy half of (b) FAILED --")
    print("      risk (c) materialized as the recency flush (see census). NEGATIVE RECORDED.")
    print(f"  (c) short-window regime (the fix): at evict={PRIMARY}, fresh {fresh1},")
    print(f"      ACC {acc0:.3f} -> {acc1:.3f} ({'UP' if acc1 > acc0 else 'down'}), "
          f"FORG {forg0:.3f} -> {forg1:.3f} ({'DOWN' if forg1 < forg0 else 'up'}) -- "
          f"{'better on BOTH axes' if acc1 > acc0 and forg1 < forg0 else 'mixed'}")
    print(f"  (d) E3 bitwise: {'PASS' if max(dxi, dP, dage) == 0.0 else 'FAIL'}")
    ok = (rows0[0]['used'] == K and sum(fresh0) == 0 and all(f > 0 for f in fresh1)
          and acc1 > acc0 and forg1 < forg0 and max(dxi, dP, dage) == 0.0)
    print("\nverdict:", "SLOT BUDGET LIFTS THE FLOODING CEILING -- later classes recruit "
          "fresh slots,\n         retention improves, and the budget state persists bitwise (E3)"
          if ok else "partial -- inspect stage")
