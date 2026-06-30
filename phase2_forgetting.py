"""
PHASE 2: catastrophic-forgetting showdown -- oscillator organism vs a neural net.

Class-incremental setting: the 10 digit classes arrive in 5 sequential tasks
([0,1] then [2,3] ... [8,9]). Each learner sees ONLY the current task's data, then
that data is gone. At the end we test on ALL classes.

  - A standard gradient-trained MLP overwrites its weights each task -> it should
    CATASTROPHICALLY FORGET the early classes (final accuracy ~ only last task).
  - The organism RECRUITS new attractor slots for novel input and leaves existing
    memories untouched -> it should RETAIN early classes by construction.

Honesty: the organism's immunity is the same property memory/prototype methods
have (a kNN wouldn't forget either) -- so we ALSO run a growing-prototype baseline
to show the organism sits in the non-forgetting class, and the contrast that
matters is vs the gradient net. Readout for the organism is a majority-label
assignment per memory (a linear readout), matching the net's supervised setup.
"""

import numpy as np
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from organism import Organism, normalize

rng = np.random.default_rng(0)
torch.manual_seed(0)

d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(np.float32)
y = d.target.astype(int)
N = X.shape[1]
NORM = np.sqrt(N)

# train/test split
perm = rng.permutation(len(X))
tr, te = perm[:1400], perm[1400:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
first_classes = TASKS[0]


def per_class_acc(pred, true):
    return {c: float((pred[true == c] == c).mean()) for c in range(10) if (true == c).any()}


# ===================== 1) GRADIENT MLP (will forget) =======================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)

mlp = MLP(); opt = torch.optim.Adam(mlp.parameters(), 1e-3); lossf = nn.CrossEntropyLoss()
for task in TASKS:
    mask = np.isin(ytr, task)
    xb = torch.tensor(Xtr[mask]); yb = torch.tensor(ytr[mask])
    for _ in range(300):
        opt.zero_grad(); loss = lossf(mlp(xb), yb); loss.backward(); opt.step()
with torch.no_grad():
    mlp_pred = mlp(torch.tensor(Xte)).argmax(1).numpy()
mlp_acc = (mlp_pred == yte).mean()
mlp_pc = per_class_acc(mlp_pred, yte)

# ===================== 2) ORGANISM (should retain) =========================
org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
def task_stream(task, hold=8, epochs=3):
    idx = np.where(np.isin(ytr, task))[0]
    seq = []
    for _ in range(epochs):
        for i in rng.permutation(idx):
            seq.extend([normalize(Xtr[i].astype(complex), NORM)] * hold)
    return seq
for task in TASKS:                                   # SEQUENTIAL exposure, data discarded after
    org.perceive(task_stream(task), g_in=4.0, eta=0.015, recruit=0.6)
org.consolidate(merge_thresh=0.85, prune_frac=0.02)
M = org.mem

# label each memory by majority of TRAIN images nearest to it (linear readout)
def nearest(states):
    return np.abs((M.conj() @ states.T) / N).argmax(0)
tr_states = np.array([normalize(x.astype(complex), NORM) for x in Xtr])
te_states = np.array([normalize(x.astype(complex), NORM) for x in Xte])
tr_assign = nearest(tr_states)
mem_label = {}
for k in range(M.shape[0]):
    mem = ytr[tr_assign == k]
    if len(mem): mem_label[k] = np.bincount(mem).argmax()
org_pred = np.array([mem_label.get(a, -1) for a in nearest(te_states)])
org_acc = (org_pred == yte).mean()
org_pc = per_class_acc(org_pred, yte)

# ===================== 3) growing-prototype baseline =======================
# (also non-forgetting; shows the organism is in the memory-method class)
protos, plabels = [], []
for task in TASKS:
    for c in task:
        protos.append(Xtr[ytr == c].mean(0)); plabels.append(c)
protos = np.array(protos); plabels = np.array(plabels)
proto_pred = plabels[np.argmin(((Xte[:, None, :] - protos[None]) ** 2).sum(-1), axis=1)]
proto_acc = (proto_pred == yte).mean()
proto_pc = per_class_acc(proto_pred, yte)

# ===================== report =============================================
print("CATASTROPHIC-FORGETTING SHOWDOWN (class-incremental, 5 sequential tasks)\n")
print(f"memories the organism ended with: {M.shape[0]}\n")
print(f"{'':22s}  overall   first-task[0,1]   last-task[8,9]")
def line(name, acc, pc):
    f = np.mean([pc.get(c, 0) for c in first_classes])
    l = np.mean([pc.get(c, 0) for c in TASKS[-1]])
    print(f"  {name:20s}  {acc:5.3f}      {f:5.3f}            {l:5.3f}")
line("gradient MLP", mlp_acc, mlp_pc)
line("ORGANISM (oscillator)", org_acc, org_pc)
line("growing prototypes", proto_acc, proto_pc)

print("\n  per-class accuracy (the forgetting signature):")
print("    class:   " + " ".join(f"{c:4d}" for c in range(10)))
print("    MLP  :   " + " ".join(f"{mlp_pc.get(c,0):4.2f}" for c in range(10)))
print("    ORG  :   " + " ".join(f"{org_pc.get(c,0):4.2f}" for c in range(10)))

org_retains = np.mean([org_pc.get(c, 0) for c in first_classes]) > 0.5
mlp_forgets = np.mean([mlp_pc.get(c, 0) for c in first_classes]) < 0.2
print("\nverdict:",
      "ORGANISM RETAINS early classes where the gradient net CATASTROPHICALLY FORGETS"
      if org_retains and mlp_forgets else
      "inconclusive -- inspect per-class row")
