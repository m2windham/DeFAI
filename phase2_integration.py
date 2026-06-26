"""
PHASE 2 -- the decisive INTEGRATION test.

Three prior tasks showed the organism only TIES simple baselines per-metric. The
remaining justification is INTEGRATION: doing perception + pattern-completion +
memory + sequence-learning + prediction JOINTLY, online, unsupervised, in ONE
substrate -- where matching it requires hand-assembling several components.

Task: a structured stream of HALF-OCCLUDED digit images. The class follows a
transition cycle (0->1->2->3->4->0, mostly). At each step, BEFORE seeing the next
image, predict its class. To do well you must (a) recognize the occluded current
digit (perception/completion via memory) and (b) exploit learned temporal
structure. All ONLINE, no labels given to the learners during the stream.

We compare:
  ORGANISM            : one oscillator substrate (settle->identify->transition->predict),
                        learns memories + transitions online, unsupervised.
  ASSEMBLED BASELINE  : online prototypes (k-means-ish) + a transition table over
                        prototype ids + argmax successor. Same info, but 3 bolted parts.
  PERSISTENCE         : predict next = current class (no temporal model).

Honest question: does the unified substrate at least MATCH the assembled stack?
If yes -> integration value is real (one system replaces a pipeline). If no ->
integration costs accuracy.
"""

import numpy as np
from sklearn.datasets import load_digits
from organism import Organism, normalize

rng = np.random.default_rng(0)
d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(float)
y = d.target.astype(int)
N = X.shape[1]; NORM = np.sqrt(N)
byc = {c: X[y == c] for c in range(10)}
CLASSES = [0, 1, 2, 3, 4]
T = np.array([[0, .8, .07, .07, .06],
              [.06, 0, .8, .07, .07],
              [.07, .06, 0, .8, .07],
              [.07, .07, .06, 0, .8],
              [.8, .07, .07, .06, 0]])

def occlude(v):
    v = v.copy().reshape(8, 8); v[4:, :] = 0.0; return v.reshape(-1)   # hide bottom half

# build the sequence (class labels) and occluded observations
seq_cls, obs = [], []
c = 0
for t in range(4000):
    seq_cls.append(c)
    obs.append(occlude(byc[c][rng.integers(len(byc[c]))]))
    c = rng.choice(CLASSES, p=T[c])
seq_cls = np.array(seq_cls); obs = np.array(obs)
true_next = seq_cls[1:]

# ===================== ORGANISM (unified, online, unsupervised) ===========
org = Organism(N=N, K=20, omega=0.15, beta=10.0, seed=0)
P = np.zeros((20, 20)); z = org.z
org_pred_mem = []; cur_assign = []
prev = -1
for t, raw in enumerate(obs):
    x = normalize(raw.astype(complex), NORM)
    for _ in range(6):
        z = normalize(z + 0.05 * (1j * org.omega * z + 4.0 * (x - z)), org.norm)
    o2 = np.abs(org.overlaps(z, org.xi)); k = int(np.argmax(o2))
    nov = 1 - (np.abs(org.overlaps(z, org.xi[org.used])).max() if org.used.any() else 0)
    if nov > 0.4 and not org.used.all():
        f = int(np.argmin(org.used.astype(float))); org.xi[f] = normalize(z, org.norm); org.used[f] = True; k = f
    else:
        za = z * np.exp(-1j * np.angle(org.overlaps(z, org.xi)[k])); org.xi[k] = normalize(org.xi[k] + 0.02 * (za - org.xi[k]), org.norm)
    cur_assign.append(k)
    if prev >= 0: P[prev, k] += 1
    # PREDICT next memory from learned transitions
    org_pred_mem.append(int(np.argmax(P[k])) if P[k].sum() > 0 else k)
    prev = k
cur_assign = np.array(cur_assign); org_pred_mem = np.array(org_pred_mem)

# ===================== ASSEMBLED BASELINE (prototypes + table) ============
protos = []; Pb = np.zeros((20, 20)); pb_assign = []; pb_pred = []; prev = -1
for t, raw in enumerate(obs):
    if protos:
        ds = ((np.array(protos) - raw) ** 2).sum(1); k = int(np.argmin(ds)); dmin = ds[k]
    else:
        k, dmin = -1, 1e9
    if k < 0 or dmin > 30 and len(protos) < 20:        # recruit a new prototype
        protos.append(raw.copy()); k = len(protos) - 1
    else:
        protos[k] = 0.98 * protos[k] + 0.02 * raw       # online update
    pb_assign.append(k)
    if prev >= 0: Pb[prev, k] += 1
    pb_pred.append(int(np.argmax(Pb[k])) if Pb[k].sum() > 0 else k)
    prev = k
pb_assign = np.array(pb_assign); pb_pred = np.array(pb_pred)

# ===================== evaluate: next-CLASS prediction accuracy ============
def mem_to_class(assign):
    lab = {}
    for k in np.unique(assign):
        m = seq_cls[assign == k]
        if len(m): lab[k] = np.bincount(m).argmax()
    return lab

def acc_from(pred_mem, assign):
    lab = mem_to_class(assign)
    pred_cls = np.array([lab.get(p, -1) for p in pred_mem])[:-1]
    return (pred_cls == true_next).mean()

org_acc = acc_from(org_pred_mem, cur_assign)
pb_acc = acc_from(pb_pred, pb_assign)
persist_acc = (seq_cls[:-1] == true_next).mean()
# also: how well each RECOGNIZES the occluded current digit (perception)
org_recog = np.mean([mem_to_class(cur_assign).get(k, -1) == seq_cls[t] for t, k in enumerate(cur_assign)])
pb_recog = np.mean([mem_to_class(pb_assign).get(k, -1) == seq_cls[t] for t, k in enumerate(pb_assign)])

print("PHASE 2 -- INTEGRATION test: predict next class in an occluded digit sequence\n")
print(f"sequence length {len(obs)}, classes {CLASSES}, bottom-half occluded\n")
print(f"  occluded-digit recognition accuracy:")
print(f"    organism (unified)     : {org_recog:.3f}")
print(f"    assembled baseline     : {pb_recog:.3f}")
print(f"\n  NEXT-class prediction accuracy:")
print(f"    organism (one substrate)        : {org_acc:.3f}")
print(f"    assembled stack (3 components)  : {pb_acc:.3f}")
print(f"    persistence (no temporal model) : {persist_acc:.3f}")
print(f"\n  organism memories used: {org.used.sum()}, baseline prototypes: {len(protos)}")

beats_persist = org_acc > persist_acc + 0.05
matches_stack = org_acc > pb_acc - 0.05
print("\nverdict:",
      ("unified substrate MATCHES the assembled pipeline AND beats persistence "
       "-> integration value is real") if beats_persist and matches_stack else
      ("organism uses temporal structure (beats persistence) but trails the assembled stack"
       if beats_persist else "no clear temporal advantage -- inspect"))
