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

# ---- SMOOTHED baseline: same prototypes + transition table, but with temporal
# HYSTERESIS (stick with previous prototype unless a new one is clearly closer).
# This hands the baseline the temporal-continuity the oscillator field gets for
# free. If it now matches the organism, the organism's edge IS that smoothing.
protos_s = []; Ps = np.zeros((20, 20)); ps_assign = []; ps_pred = []; prev = -1; prev_k = -1
for t, raw in enumerate(obs):
    if protos_s:
        ds = ((np.array(protos_s) - raw) ** 2).sum(1); kn = int(np.argmin(ds))
        k = prev_k if (0 <= prev_k < len(protos_s) and ds[prev_k] < 1.3 * ds[kn]) else kn
        dmin = ds[k]
    else:
        k, dmin = -1, 1e9
    if k < 0 or dmin > 30 and len(protos_s) < 20:
        protos_s.append(raw.copy()); k = len(protos_s) - 1
    else:
        protos_s[k] = 0.98 * protos_s[k] + 0.02 * raw
    ps_assign.append(k)
    if prev >= 0: Ps[prev, k] += 1
    ps_pred.append(int(np.argmax(Ps[k])) if Ps[k].sum() > 0 else k)
    prev = k; prev_k = k
ps_assign = np.array(ps_assign); ps_pred = np.array(ps_pred)

# ---- FEATURE-EMA baseline: classify a temporally-smoothed FEATURE vector
# (continuous-state momentum, like the field carrying z across steps). If THIS
# catches the organism, the edge is continuous-state momentum; if not, the edge
# is something deeper in the oscillator representation itself.
protos_e = []; Pe = np.zeros((20, 20)); pe_assign = []; pe_pred = []; prev = -1
xs = np.zeros(N)
for t, raw in enumerate(obs):
    xs = 0.5 * xs + 0.5 * raw                       # continuous feature momentum
    if protos_e:
        ds = ((np.array(protos_e) - xs) ** 2).sum(1); k = int(np.argmin(ds)); dmin = ds[k]
    else:
        k, dmin = -1, 1e9
    if k < 0 or dmin > 30 and len(protos_e) < 20:
        protos_e.append(xs.copy()); k = len(protos_e) - 1
    else:
        protos_e[k] = 0.98 * protos_e[k] + 0.02 * xs
    pe_assign.append(k)
    if prev >= 0: Pe[prev, k] += 1
    pe_pred.append(int(np.argmax(Pe[k])) if Pe[k].sum() > 0 else k)
    prev = k
pe_assign = np.array(pe_assign); pe_pred = np.array(pe_pred)

# ---- COSINE baseline: prototypes matched by COSINE similarity (like the organism's
# normalized complex overlap) instead of Euclidean. If this catches the organism,
# the "integration win" is really just the metric -- NOT the oscillator dynamics.
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
protos_c = []; Pc = np.zeros((20, 20)); pc_assign = []; pc_pred = []; prev = -1
for t, raw in enumerate(obs):
    u = unit(raw)
    if protos_c:
        sims = np.array(protos_c) @ u; k = int(np.argmax(sims)); best = sims[k]
    else:
        k, best = -1, -1
    if k < 0 or best < 0.6 and len(protos_c) < 20:
        protos_c.append(u.copy()); k = len(protos_c) - 1
    else:
        protos_c[k] = unit(0.98 * protos_c[k] + 0.02 * u)
    pc_assign.append(k)
    if prev >= 0: Pc[prev, k] += 1
    pc_pred.append(int(np.argmax(Pc[k])) if Pc[k].sum() > 0 else k)
    prev = k
pc_assign = np.array(pc_assign); pc_pred = np.array(pc_pred)

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
ps_acc = acc_from(ps_pred, ps_assign)
pe_acc = acc_from(pe_pred, pe_assign)
pc_acc = acc_from(pc_pred, pc_assign)
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
print(f"    assembled stack (independent)   : {pb_acc:.3f}")
print(f"    assembled stack + HYSTERESIS    : {ps_acc:.3f}  <- discrete-label continuity")
print(f"    assembled stack + FEATURE-EMA   : {pe_acc:.3f}  <- continuous-state momentum")
print(f"    assembled stack + COSINE metric : {pc_acc:.3f}  <- same metric as organism")
print(f"    persistence (no temporal model) : {persist_acc:.3f}")
print(f"\n  organism memories used: {org.used.sum()}, baseline prototypes: {len(protos)}")

best_base = max(pb_acc, ps_acc, pe_acc, pc_acc)
print(f"\n  strongest baseline = {best_base:.3f} (cosine-metric prototypes+table)")
print("verdict:",
      "organism BEATS the best baseline -> integration value real"
      if org_acc > best_base + 0.02 else
      "INTEGRATION HYPOTHESIS NOT SUPPORTED: a matched-metric simple pipeline beats the "
      "organism. The earlier 'win' vs Euclidean baselines was a METRIC artifact, not the "
      "oscillator dynamics. The dynamics cost accuracy here.")
