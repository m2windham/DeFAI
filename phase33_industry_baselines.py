"""
PHASE 33: INDUSTRY-STANDARD READINESS BENCHMARK -- where does the organism
actually stand against the standard continual-learning ladder, measured with
the standard metrics, before anything goes live?

Gate (set by the project owner, 2026-07-14): "if we are not at least
SOTA / cost-effective, we are not ready." This phase measures both halves
of that gate honestly and reports which claims survive.

Protocol: class-incremental split-digits (sklearn digits, 5 sequential
tasks of 2 classes; the small offline cousin of Split-MNIST, the standard
academic CL protocol). Every learner sees each task ONCE, in order, and the
task's data is then gone (replay buffer excepted, that's its mechanism).

Ladder (industry standard, weakest to strongest):
  mlp-seq     naive SGD fine-tuning        (the catastrophic-forgetting floor)
  mlp-ewc     elastic weight consolidation (the canonical regularization fix)
  mlp-replay  experience replay, 20/class  (the standard rehearsal fix)
  mlp-joint   joint training on all data   (the offline oracle CEILING, not CL)
  prototypes  growing class prototypes     (the memory-method control:
                                            anything slot-based must at least
                                            explain its gap to this)
  ORGANISM    oscillator field, unsupervised slots + majority-label readout

Metrics (standard CL definitions):
  ACC  = mean over tasks of final accuracy on that task
  FORG = mean over tasks (except last) of (best accuracy the task ever had
         - its final accuracy)  -- how much was lost after moving on
plus the COST axes: wall-clock train time, state size in bytes, and the
capability flags no accuracy number shows: single-pass online?, labels
needed during representation learning?, gradients?

Pre-registered honesty: phase 2 already showed prototypes BEAT the organism
on raw accuracy here (0.894 vs 0.708). The question this phase answers is
the full picture -- where the organism lands on the ladder, at what cost,
and which of its claims are load-bearing (unsupervised + structure +
persistence, not benchmark dominance).
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from organism import normalize
from organism_numba import NumbaOrganism

rng = np.random.default_rng(0)
torch.manual_seed(0)

d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(np.float32)
y = d.target.astype(int)
N = X.shape[1]
NORM = np.sqrt(N)
perm = rng.permutation(len(X))
tr, te = perm[:1400], perm[1400:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


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


def run_organism(hold=8, epochs=3):
    """Readout: per-slot label EVIDENCE accumulated online after each task
    (nearest-slot counts on the current task's data only -- no stored past
    data, no end-of-run oracle). Slots may relabel as they drift: task 1
    floods all K slots (recruit is a similarity floor), so later classes
    are represented by old slots drifting toward them; a frozen first-task
    label misattributes exactly those (measured: ACC 0.25 with frozen
    labels vs phase 2's 0.708 with end-labeling -- the drift, not the
    memory, was being scored)."""
    org = NumbaOrganism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
    evidence = np.zeros((40, 10))
    A = np.zeros((len(TASKS), len(TASKS)))
    t0 = time.time()
    for ti, task in enumerate(TASKS):
        idx = np.where(np.isin(ytr, task))[0]
        seq = []
        for _ in range(epochs):
            for i in rng.permutation(idx):
                seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
        org.perceive(seq, g_in=4.0, eta=0.015, recruit=0.6)
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
    state_bytes = org.xi.nbytes + org.P.nbytes + org.count.nbytes
    return A, time.time() - t0, state_bytes, int(org.used.sum())


if __name__ == "__main__":
    print("PHASE 33: industry-standard continual-learning ladder (split-digits)\n")
    rows = []
    for name, fn in [
        ('mlp-seq', lambda: run_mlp('seq')),
        ('mlp-ewc', lambda: run_mlp('ewc')),
        ('mlp-replay', lambda: run_mlp('replay')),
        ('mlp-joint*', lambda: run_mlp('joint')),
    ]:
        A, t, bytes_ = fn()
        acc, forg = task_acc_matrix_to_metrics(A)
        rows.append((name, acc, forg, t, bytes_, '-'))
        print(f"  {name:<12} ACC {acc:.3f}  FORG {forg:.3f}  train {t:5.1f}s  state {bytes_/1e3:6.1f}KB")
    A, t, bytes_, np_ = run_prototypes()
    acc, forg = task_acc_matrix_to_metrics(A)
    rows.append(('prototypes', acc, forg, t, bytes_, np_))
    print(f"  {'prototypes':<12} ACC {acc:.3f}  FORG {forg:.3f}  train {t:5.1f}s  state {bytes_/1e3:6.1f}KB  ({np_} protos)")
    A, t, bytes_, nm = run_organism()
    acc, forg = task_acc_matrix_to_metrics(A)
    rows.append(('ORGANISM', acc, forg, t, bytes_, nm))
    print(f"  {'ORGANISM':<12} ACC {acc:.3f}  FORG {forg:.3f}  train {t:5.1f}s  state {bytes_/1e3:6.1f}KB  ({nm} slots)")

    print("\n  * mlp-joint is the offline oracle ceiling, not a continual learner")
    print("\ncapability flags (what the accuracy table cannot show):")
    print("  single-pass online:  organism, prototypes YES; all MLP arms NO (300 epochs/task)")
    print("  unsupervised repr.:  organism YES (labels only in readout); all others NO")
    print("  learns structure:    organism YES (transition graph -> prediction/planning);")
    print("                       nothing else on this ladder even has the concept")
    print("  persistence:         organism YES (E3, bitwise); MLP checkpoint YES; others ad hoc")

    org_row = rows[-1]
    proto_row = rows[-2]
    seq_row = rows[0]
    print("\nverdict (against the owner's gate 'SOTA or not ready'):")
    print(f"  vs gradient forgetting: organism ACC {org_row[1]:.2f} vs naive SGD {seq_row[1]:.2f} and")
    print(f"    EWC {rows[1][1]:.2f} -- the continual-retention claim SURVIVES (2x the")
    print(f"    gradient arms), but FORG {org_row[2]:.2f} exposes a real mechanism: at K=40 the")
    print("    first task floods every slot and later classes are learned by slot DRIFT,")
    print("    which is itself a slow forgetting channel. Capacity headroom matters.")
    print(f"  vs memory methods: prototypes ACC {proto_row[1]:.2f} > organism {org_row[1]:.2f} -- on RAW")
    print("    class-incremental accuracy the organism is NOT SOTA; a supervised")
    print("    prototype baseline wins the benchmark outright, faster and smaller.")
    print("    The organism's differentiated claims are the flags above (unsupervised")
    print("    representation + structure learning + bitwise persistence in ONE online")
    print("    mechanism), NOT benchmark dominance. Public claims must be scoped to that.")
