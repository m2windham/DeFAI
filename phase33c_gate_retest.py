"""
PHASE 33c (T1.4): GATE RE-TEST -- the full phase-33 ladder, re-run after the
two product-critical mechanism fixes the RELEASE HOLD named (ROADMAP,
2026-07-15) landed as mechanism:

  T1.2 (phase 33b)  slot-budget/eviction under recruitment pressure
                    (`perceive(evict=E)`, organism.py; harness section 9)
  T1.3              online per-slot label evidence as a reusable eval-side
                    readout (`label_readout.py`; `test_label_readout.py`)

T1.1 (phase 36, un-oracled hierarchical recall) is sequencing-gate context
only: it is a GENERATION-side fix for the Attractor Crowding Collapse
(routing recall through discovered categories) and has no role in this
classification protocol at K=40 -- nothing here routes recall. Recorded so
nobody wonders why the ladder doesn't "use" phase 36.

Protocol: identical to phase 33 verbatim (class-incremental split-digits,
sklearn digits, 5 sequential tasks of 2 classes, every learner sees each
task once; same seeds, same data pipeline, same recipes). Standard metrics
ACC/FORG (definitions in `task_acc_matrix_to_metrics`), plus the COST axes:
wall-clock train time, state bytes, capability flags.

Ladder (phase 33's six arms, plus the T1.4 arm):
  mlp-seq       naive SGD fine-tuning          (catastrophic-forgetting floor)
  mlp-ewc       elastic weight consolidation   (canonical regularization fix)
  mlp-replay    experience replay, 20/class    (standard rehearsal fix)
  mlp-joint*    joint training on all data     (offline oracle CEILING, not CL)
  prototypes    growing class prototypes       (supervised memory-method bar:
                                                the 0.872 the gate is scored
                                                against)
  organism      evict=0, phase 33's arm        (REPRODUCTION ANCHOR)
  ORGANISM+B    evict=250 + LabelEvidenceReadout with eviction invalidation
                (T1.2 + T1.3 folded in -- the arm the gate re-test is about)

Readout bookkeeping under eviction (eval-side only, 33b's rule restated):
a recycled slot's label evidence is stale by construction, so slots whose
eviction tally advanced or whose used flag flipped fresh during a task are
`readout.invalidate()`d before that task's evidence is accumulated. This is
the readout following slot identity, not label information entering the
mechanism; `test_label_readout.py` pins the module's equivalence to 33b's
inline bookkeeping on this exact protocol.

Pre-registered predictions (written before the committed run):
  (a) Reproduction anchors, numpy-deterministic, expected EXACT: prototypes
      ACC 0.872; organism evict=0 ACC 0.665 / FORG 0.200 (phase 33's
      committed values). Torch arms are torch-version-dependent; expected
      near the committed SGD/EWC ~0.32 (tolerance +/-0.05), replay/joint
      wherever the committed run had them (not in the ROADMAP row; this run
      re-establishes them on record).
  (b) ORGANISM+budget reproduces phase 33b's primary characterization
      EXACTLY: ACC 0.712 / FORG 0.169, fresh recruits every later task --
      same stream verbatim, readout module pinned equivalent to the inline
      evidence.
  (c) Gate expectation, honest: 0.712 < 0.872 -- the organism remains NOT
      SOTA on raw class-incremental accuracy. What T1.2 changed is the gap
      (0.207 -> ~0.160) and the forgetting axis; FORG 0.169 should now beat
      every gradient arm by a wide margin AND the supervised prototype bar's
      own forgetting is measured here for the record (never committed in
      phase 33's row -- no prediction, just measurement).
  (d) Cost axes: the budget arm stays single-pass online, unsupervised,
      state within ~1 KB of the baseline organism (age clock = K floats);
      train wall-clock within ~2x of the evict=0 arm (eviction scans are
      rare, pressure-gated).

Owner decision point (verbatim from AGENT_TARGETS T1.4): the gate re-opens
only at >=SOTA or demonstrated cost-effectiveness at equal accuracy. This
script MEASURES; the decision is the owner's. The verdict section prints
the measured facts against that bar and takes no liberties with it.
"""

import time

import numpy as np

try:                    # baselines only -- organism arms run without torch
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
except ImportError:
    torch = nn = None
from sklearn.datasets import load_digits

from label_readout import LabelEvidenceReadout
from organism import normalize
from organism_numba import NumbaOrganism

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
EVICT = 250             # phase 33b's primary window (~3x live-slot revisit
                        # interval, ~27x below task length; see its docstring)

# committed anchors this run must reproduce (phase 33 / phase 33b rows)
ANCHOR_PROTO_ACC = 0.872
ANCHOR_ORG_ACC, ANCHOR_ORG_FORG = 0.665, 0.200
ANCHOR_BUDGET_ACC, ANCHOR_BUDGET_FORG = 0.712, 0.169


def task_acc_matrix_to_metrics(A):
    """A[i, j] = accuracy on task j's classes after training task i."""
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


if torch is not None:
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N, 128), nn.ReLU(), nn.Linear(128, 10))

        def forward(self, x):
            return self.net(x)


def mlp_predict(mlp):
    def f(Xe):
        with torch.no_grad():
            return mlp(torch.tensor(Xe)).argmax(1).numpy()
    return f


def run_mlp(mode, ewc_lam=1000.0, replay_per_class=20, epochs=300):
    """Phase 33's gradient arms, verbatim."""
    mlp = MLP()
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    A = np.zeros((len(TASKS), len(TASKS)))
    fishers, stars = [], []
    buf_x, buf_y = [], []
    t0 = time.time()
    if mode == 'joint':
        xb = torch.tensor(Xtr); yb = torch.tensor(ytr)
        for _ in range(epochs):
            opt.zero_grad(); lossf(mlp(xb), yb).backward(); opt.step()
        A[:] = eval_tasks(mlp_predict(mlp))
        return A, time.time() - t0, sum(p.numel() for p in mlp.parameters()) * 4
    for ti, task in enumerate(TASKS):
        mask = np.isin(ytr, task)
        cur_x, cur_y = Xtr[mask], ytr[mask]
        if mode == 'replay' and buf_x:
            cur_x = np.concatenate([cur_x] + buf_x)
            cur_y = np.concatenate([cur_y] + buf_y)
        xb = torch.tensor(cur_x); yb = torch.tensor(cur_y)
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(mlp(xb), yb)
            if mode == 'ewc':
                for F, st in zip(fishers, stars):
                    for (n_, p), f_, s_ in zip(mlp.named_parameters(), F, st):
                        loss = loss + (ewc_lam / 2) * (f_ * (p - s_) ** 2).sum()
            loss.backward(); opt.step()
        if mode == 'ewc':                     # Fisher diag on this task's data
            F = [torch.zeros_like(p) for p in mlp.parameters()]
            for i in range(0, len(cur_x), 8):
                mlp.zero_grad()
                out = mlp(torch.tensor(cur_x[i:i + 8]))
                nn.functional.nll_loss(
                    nn.functional.log_softmax(out, 1),
                    torch.tensor(cur_y[i:i + 8])).backward()
                for f_, p in zip(F, mlp.parameters()):
                    f_ += p.grad.detach() ** 2
            fishers.append([f_ / max(len(cur_x) // 8, 1) for f_ in F])
            stars.append([p.detach().clone() for p in mlp.parameters()])
        if mode == 'replay':
            for c in task:
                idx = np.where(ytr[mask] == c)[0][:replay_per_class]
                buf_x.append(Xtr[mask][idx]); buf_y.append(ytr[mask][idx])
        A[ti] = eval_tasks(mlp_predict(mlp))
    return A, time.time() - t0, sum(p.numel() for p in mlp.parameters()) * 4


def run_prototypes(thresh=0.55):
    """Phase 33's supervised memory-method bar, verbatim."""
    protos, labels = [], []
    A = np.zeros((len(TASKS), len(TASKS)))
    t0 = time.time()
    for ti, task in enumerate(TASKS):
        idx = np.where(np.isin(ytr, task))[0]
        for i in idx:
            v = Xtr[i] / (np.linalg.norm(Xtr[i]) + 1e-9)
            if protos:
                sims = np.array([v @ p for p in protos])
                j = int(np.argmax(sims))
                if sims[j] > thresh:
                    protos[j] = protos[j] + 0.1 * (v - protos[j])
                    protos[j] /= np.linalg.norm(protos[j]) + 1e-9
                    continue
            protos.append(v); labels.append(ytr[i])

        def pred(Xe, P=np.array(protos), L=np.array(labels)):
            Xn = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9)
            return L[np.argmax(Xn @ P.T, axis=1)]
        A[ti] = eval_tasks(pred)
    return A, time.time() - t0, len(protos) * N * 4, len(protos)


def run_organism(evict, hold=8, epochs=3, verbose=False):
    """Phase 33's organism arm with the T1.2/T1.3 mechanisms: evict=0 is the
    reproduction anchor (phase 33 verbatim), evict=EVICT is the T1.4 arm.
    Stream is phase 33's verbatim: a fresh rng burns the train/test split
    draw so per-task shuffles match the module-rng consumption exactly
    (phase 33b's technique). Returns per-task fresh-recruit/eviction rows
    and the birth-task census alongside the metrics."""
    r = np.random.default_rng(0)
    r.permutation(len(X))                    # burn the split draw (see above)
    org = NumbaOrganism(N=N, K=K, omega=0.15, beta=10.0, seed=0)
    readout = LabelEvidenceReadout(K=K, n_classes=10)
    owner = np.full(K, -1)
    A = np.zeros((len(TASKS), len(TASKS)))
    rows = []
    t0 = time.time()
    for ti, task in enumerate(TASKS):
        used_before = org.used.copy()
        evd_before = org.evictions.copy()
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in r.permutation(idx):
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6, evict=evict)
        fresh = (~used_before & org.used) | (org.evictions - evd_before > 0)
        owner[fresh] = ti
        readout.invalidate(fresh)            # readout follows slot identity
        readout.observe(org, Xtr[idx], ytr[idx])
        A[ti] = eval_tasks(lambda Xe: readout.predict(org, Xe))
        rows.append(dict(task=ti, used=int(org.used.sum()),
                         evictions=int((org.evictions - evd_before).sum()),
                         fresh=int(fresh.sum())))
        if verbose:
            print(f"      task {ti} ({task}): slots {rows[-1]['used']}/{K}  "
                  f"fresh {rows[-1]['fresh']:>2}  evict {rows[-1]['evictions']:>2}  "
                  f"acc-so-far {np.mean(A[ti][:ti + 1]):.3f}")
    census = [int((owner[org.used] == t).sum()) for t in range(len(TASKS))]
    state_bytes = (org.xi.nbytes + org.P.nbytes + org.count.nbytes
                   + (org.age.nbytes if evict > 0 else 0))
    return A, time.time() - t0, state_bytes, int(org.used.sum()), rows, census


if __name__ == "__main__":
    print("PHASE 33c (T1.4): gate re-test -- full ladder, T1.2+T1.3 folded in\n")
    rows = []

    mlp_arms = [
        ('mlp-seq', lambda: run_mlp('seq')),
        ('mlp-ewc', lambda: run_mlp('ewc')),
        ('mlp-replay', lambda: run_mlp('replay')),
        ('mlp-joint*', lambda: run_mlp('joint')),
    ]
    if torch is None:
        mlp_arms = []
        print("  torch not installed -- skipping the four MLP baseline arms")
        print("  (a gate re-test without them is NOT the deliverable: T1.4")
        print("  explicitly requires the torch baselines. Install torch.)\n")
    for name, fn in mlp_arms:
        A, t, bytes_ = fn()
        acc, forg = task_acc_matrix_to_metrics(A)
        rows.append((name, acc, forg, t, bytes_, '-'))
        print(f"  {name:<12} ACC {acc:.3f}  FORG {forg:.3f}  train {t:5.1f}s  state {bytes_/1e3:6.1f}KB")

    A, t, bytes_, np_ = run_prototypes()
    proto_acc, proto_forg = task_acc_matrix_to_metrics(A)
    rows.append(('prototypes', proto_acc, proto_forg, t, bytes_, np_))
    print(f"  {'prototypes':<12} ACC {proto_acc:.3f}  FORG {proto_forg:.3f}  train {t:5.1f}s  state {bytes_/1e3:6.1f}KB  ({np_} protos)")

    A0, t0_, bytes0, nm0, rows0, cen0 = run_organism(evict=0)
    org_acc, org_forg = task_acc_matrix_to_metrics(A0)
    rows.append(('organism', org_acc, org_forg, t0_, bytes0, nm0))
    print(f"  {'organism':<12} ACC {org_acc:.3f}  FORG {org_forg:.3f}  train {t0_:5.1f}s  state {bytes0/1e3:6.1f}KB  ({nm0} slots, evict=0 anchor)")

    print(f"  {'ORGANISM+B':<12} (evict={EVICT}, per-task trace):")
    A1, t1_, bytes1, nm1, rows1, cen1 = run_organism(evict=EVICT, verbose=True)
    bud_acc, bud_forg = task_acc_matrix_to_metrics(A1)
    rows.append(('ORGANISM+B', bud_acc, bud_forg, t1_, bytes1, nm1))
    print(f"  {'ORGANISM+B':<12} ACC {bud_acc:.3f}  FORG {bud_forg:.3f}  train {t1_:5.1f}s  state {bytes1/1e3:6.1f}KB  ({nm1} slots, census {cen1})")

    print("\n  * mlp-joint is the offline oracle ceiling, not a continual learner")

    print("\nreproduction anchors (pre-registered prediction a/b):")
    checks = [
        ("prototypes ACC", proto_acc, ANCHOR_PROTO_ACC),
        ("organism evict=0 ACC", org_acc, ANCHOR_ORG_ACC),
        ("organism evict=0 FORG", org_forg, ANCHOR_ORG_FORG),
        ("ORGANISM+B ACC", bud_acc, ANCHOR_BUDGET_ACC),
        ("ORGANISM+B FORG", bud_forg, ANCHOR_BUDGET_FORG),
    ]
    anchors_ok = True
    for label, got, want in checks:
        ok = abs(got - want) < 5e-4          # committed values are 3-decimal
        anchors_ok &= ok
        print(f"  {label:<24} {got:.3f} vs committed {want:.3f}  {'REPRODUCED' if ok else 'MISMATCH'}")
    flood_ok = rows0[0]['used'] == K and sum(r['fresh'] for r in rows0[1:]) == 0
    fresh_ok = all(r['fresh'] > 0 for r in rows1[1:])
    print(f"  evict=0 floods (40/40, zero later recruits): {'REPRODUCED' if flood_ok else 'MISMATCH'}")
    print(f"  evict={EVICT} recruits fresh slots every later task: {'YES' if fresh_ok else 'NO'}")

    print("\nper-task final accuracy (task 0 retention is the budget's work):")
    print(f"  organism evict=0   {np.round(A0[-1], 3).tolist()}")
    print(f"  ORGANISM+B         {np.round(A1[-1], 3).tolist()}")

    print("\ncapability flags (what the accuracy table cannot show):")
    print("  single-pass online:  organism arms, prototypes YES; all MLP arms NO (300 epochs/task)")
    print("  unsupervised repr.:  organism arms YES (labels only in readout); all others NO")
    print("  learns structure:    organism arms YES (transition graph -> prediction/planning);")
    print("                       nothing else on this ladder even has the concept")
    print("  persistence:         organism arms YES (E3, bitwise -- incl. the eviction clock);")
    print("                       MLP checkpoint YES; others ad hoc")

    print("\nverdict (against the owner's gate, verbatim: 'gate re-opens only at")
    print(">=SOTA or demonstrated cost-effectiveness at equal accuracy'):")
    gap0 = ANCHOR_PROTO_ACC - org_acc
    gap1 = proto_acc - bud_acc
    print(f"  raw accuracy: ORGANISM+B {bud_acc:.3f} vs supervised prototype bar {proto_acc:.3f}")
    if bud_acc >= proto_acc:
        print("    -- the organism MEETS the bar: >=SOTA on this ladder.")
    else:
        print(f"    -- still NOT SOTA. T1.2 narrowed the gap ({gap0:.3f} -> {gap1:.3f},")
        print(f"       {100 * (1 - gap1 / gap0):.0f}% closed) but did not close it.")
    grad_best = max((r[1] for r in rows if r[0] in ('mlp-seq', 'mlp-ewc')), default=None)
    if grad_best is not None:
        print(f"  vs gradient forgetting: ORGANISM+B ACC {bud_acc:.3f} vs best of SGD/EWC")
        print(f"    {grad_best:.3f} -- the continual-retention claim strengthens "
              f"({bud_acc / grad_best:.1f}x).")
    print(f"  forgetting axis: ORGANISM+B FORG {bud_forg:.3f} vs prototypes {proto_forg:.3f},")
    print(f"    organism evict=0 {org_forg:.3f}" + (
        f", replay {rows[2][2]:.3f}, EWC {rows[1][2]:.3f}, SGD {rows[0][2]:.3f}."
        if mlp_arms else "."))
    print("  equal-accuracy cost-effectiveness: accuracy is NOT equal to the bar")
    print(f"    ({bud_acc:.3f} vs {proto_acc:.3f}), so the cost-effectiveness branch of the")
    print("    gate cannot be claimed on this protocol. The surviving differentiated")
    print("    claims remain the capability flags above (unsupervised representation +")
    print("    structure learning + bitwise persistence in one single-pass online")
    print("    mechanism), now WITHOUT the flooding ceiling (fresh recruits every task).")
    print("  OWNER DECISION POINT: the measured facts are above; re-opening the gate")
    print("    is the owner's call, not this script's.")
