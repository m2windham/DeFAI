"""
T1.3 VALIDATION: `label_readout.py` (the eval-side online per-slot
label-evidence readout promoted out of phase 33) against three pins:

  1. EQUIVALENCE -- on the exact phase-33 protocol (split-digits, 5 tasks,
     K=40 organism), the module's evidence matrix and every per-task
     prediction are IDENTICAL to the original inline logic (reproduced
     verbatim below as the reference). Same math, same tie-breaking --
     bitwise, not tolerance-banded: this is a refactor, not a port.
  2. EVAL-SIDE PURITY -- observe/predict leave every organism attribute
     (xi, z, used, count, P, rng state) bitwise unchanged. Labels touch
     the readout only; the mechanism stays unsupervised by construction.
  3. LIFECYCLE -- remap folds votes drop->keep and clears drop;
     invalidate clears expired slots and prediction can no longer route
     through them (the EventBoundary notification contract, mirrored so
     pool-mode fusion/recycling and future eviction (T1.2) can forward
     straight through at the phase-script level).

Run: python test_label_readout.py  (nonzero exit on failure)
"""

import sys

import numpy as np
from sklearn.datasets import load_digits

from label_readout import LabelEvidenceReadout
from organism import Organism, normalize

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  (' + detail + ')') if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Reference: phase 33's ORIGINAL inline readout, verbatim (evidence
# accumulation + pred closure from run_organism as committed pre-T1.3).
# ---------------------------------------------------------------------------
def reference_observe(org, evidence, states, labels, N):
    used = np.where(org.used)[0]
    ov = np.abs(org.xi[used].conj() @ states.T) / N
    near = np.argmax(ov, axis=0)
    for s_i, slot in enumerate(used):
        claimed = labels[near == s_i]
        for c in claimed:
            evidence[slot, c] += 1.0


def reference_predict(org, evidence, st, N):
    used_ = np.array([s for s in np.where(org.used)[0]
                      if evidence[s].sum() > 0])
    lab = evidence[used_].argmax(1)
    ovv = np.abs(org.xi[used_].conj() @ st.T) / N
    return lab[np.argmax(ovv, axis=0)]


def org_snapshot(org):
    return (org.xi.copy(), org.z.copy(), org.used.copy(), org.count.copy(),
            org.P.copy(), org.rng.bit_generator.state)


def org_unchanged(org, snap):
    xi, z, used, count, P, rng_state = snap
    return (np.array_equal(org.xi, xi) and np.array_equal(org.z, z)
            and np.array_equal(org.used, used)
            and np.array_equal(org.count, count) and np.array_equal(org.P, P)
            and org.rng.bit_generator.state == rng_state)


if __name__ == "__main__":
    # Phase-33 data setup, verbatim (same seed, split, tasks).
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

    print("1. equivalence to the inline phase-33 readout (split-digits, K=40)")
    org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
    evidence_ref = np.zeros((40, 10))
    readout = LabelEvidenceReadout(K=40, n_classes=10)
    hold, epochs = 8, 3
    all_pred_equal = True
    for ti, task in enumerate(TASKS):
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in rng.permutation(idx):
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6)

        states = np.array([normalize(Xtr[i].astype(complex), NORM) for i in idx])
        reference_observe(org, evidence_ref, states, ytr[idx], N)
        snap = org_snapshot(org)
        readout.observe(org, Xtr[idx], ytr[idx])
        if ti == 0:
            check("observe leaves organism state bitwise unchanged",
                  org_unchanged(org, snap))

        for tj, teval in enumerate(TASKS[:ti + 1]):
            m = np.isin(yte, teval)
            st = np.array([normalize(x.astype(complex), NORM) for x in Xte[m]])
            p_ref = reference_predict(org, evidence_ref, st, N)
            p_mod = readout.predict(org, Xte[m])
            all_pred_equal &= np.array_equal(p_ref, p_mod)
    check("evidence matrices identical after all 5 tasks",
          np.array_equal(evidence_ref, readout.evidence),
          f"total votes {int(readout.evidence.sum())}")
    check("every per-task prediction identical (15 eval passes)", all_pred_equal)

    print("\n2. eval-side purity")
    snap = org_snapshot(org)
    readout.predict(org, Xte[:50])
    readout.observe(org, Xtr[:50], ytr[:50])
    check("predict+observe leave organism state bitwise unchanged",
          org_unchanged(org, snap))
    check("complex pre-encoded states accepted, same answer",
          np.array_equal(
              readout.predict(org, Xte[:50]),
              readout.predict(org, np.array([normalize(x.astype(complex), NORM)
                                             for x in Xte[:50]]))))

    print("\n3. slot-lifecycle notifications (EventBoundary contract)")
    r = LabelEvidenceReadout(K=4, n_classes=3)
    r.evidence[:] = [[5, 0, 0], [0, 3, 0], [2, 0, 1], [0, 0, 4]]
    r.remap(drop=2, keep=0)
    check("remap folds drop's votes into keep and clears drop",
          np.array_equal(r.evidence,
                         [[7, 0, 1], [0, 3, 0], [0, 0, 0], [0, 0, 4]]))
    r.invalidate(np.array([False, True, False, False]))
    check("invalidate clears expired slots' evidence",
          np.array_equal(r.evidence,
                         [[7, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 4]]))

    class FakeOrg:            # 4 orthogonal 2-hot memories, all slots used
        N, norm = 4, 2.0
        used = np.ones(4, bool)
        xi = np.array([[1, 1, 0, 0], [0, 1, 1, 0],
                       [0, 0, 1, 1], [1, 0, 0, 1]], complex) * np.sqrt(2)
    fake = FakeOrg()
    check("prediction never routes through invalidated/evidence-free slots",
          np.array_equal(sorted(set(r.evidenced(fake))), [0, 3]))
    probe = np.array([[0, 1, 1, 0]], float)   # slot 1's own pattern, now dead
    check("query nearest a dead slot falls to the best LIVE slot's label",
          r.predict(fake, probe)[0] in (r.evidence[0].argmax(),
                                        r.evidence[3].argmax()))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASS -- label_readout.py is the phase-33 readout, "
          "eval-side and lifecycle-safe.")
