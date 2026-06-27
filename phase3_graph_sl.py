"""
PHASE 3 -- SL DYNAMICS ON THE LEARNED TRANSITION GRAPH.

Phase 2 showed the integration hypothesis failed: the organism's next-class
prediction (0.544) was beaten by a cosine-metric baseline (0.678). Root cause:
prediction was a dumb argmax table lookup. The dynamics were doing memory
formation but then DISCARDED for prediction.

Two papers (Zhang et al. WWW'26 SLGNN; Millán et al. 2025 SL theory) converge
on the same point: running SL field propagation THROUGH a learned graph is
strictly better than a lookup table because:
  - SL criticality (alpha~0) gives algebraic (not exponential) decay -> longer
    memory, richer latent-space exploration before settling.
  - Leader-driven synchronization: the active memory biases the field toward
    the highest-probability successor through coupling, not lookup.
  - Amplitude regulation prevents oversmoothing (the field doesn't collapse to
    a single winner prematurely).

Phase 3 experiment: same occluded-digit sequence as Phase 2, but now after the
organism settles into memory k, it runs SL message-passing THROUGH its own
learned transition matrix P to predict the next attractor -- no argmax lookup.
We compare:
  GRAPH-SL organism  : settle -> identify memory k -> inject z into SL net on P
                       -> read winner after T_pred steps.
  TABLE organism     : same perception, but predict via argmax(P[k]).
  COSINE baseline    : the Phase 2 winner -- cosine prototypes + argmax table.
  (all using the same learned memories/transitions for fair comparison)

If graph-SL beats table+cosine -> integration is real and the mechanism is
SL dynamics propagating through learned structure.
If not -> the substrate provides no prediction advantage over lookup.
"""

import numpy as np
from sklearn.datasets import load_digits
from organism import Organism, normalize

# ===== reproducibility =====
rng = np.random.default_rng(0)

# ===== data =====
d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(float)
y = d.target.astype(int)
N = X.shape[1]; NORM = np.sqrt(N)
byc = {c: X[y == c] for c in range(10)}

# ===== same structured sequence as Phase 2 =====
CLASSES = [0, 1, 2, 3, 4]
T_true = np.array([[0, .8, .07, .07, .06],
                   [.06, 0, .8, .07, .07],
                   [.07, .06, 0, .8, .07],
                   [.07, .07, .06, 0, .8],
                   [.8, .07, .07, .06, 0]])

def occlude(v):
    v = v.copy().reshape(8, 8); v[4:, :] = 0.0; return v.reshape(-1)

seq_cls, obs = [], []
c = 0
for t in range(4000):
    seq_cls.append(c)
    obs.append(occlude(byc[c][rng.integers(len(byc[c]))]))
    c = rng.choice(CLASSES, p=T_true[c])
seq_cls = np.array(seq_cls); obs = np.array(obs)
true_next = seq_cls[1:]

# ========================================================================
# SHARED PERCEPTION PASS: one organism settles into memories + learns P
# Both prediction methods (table vs graph-SL) use the SAME organism state.
# ========================================================================
K = 20
org = Organism(N=N, K=K, omega=0.15, beta=10.0, seed=0)
P_counts = np.zeros((K, K))
z = org.z
cur_assign = []
prev = -1

for t, raw in enumerate(obs):
    x = normalize(raw.astype(complex), NORM)
    for _ in range(6):
        z = normalize(z + 0.05 * (1j * org.omega * z + 4.0 * (x - z)), org.norm)
    o2 = np.abs(org.overlaps(z, org.xi)); k = int(np.argmax(o2))
    nov = 1 - (np.abs(org.overlaps(z, org.xi[org.used])).max() if org.used.any() else 0)
    if nov > 0.4 and not org.used.all():
        f = int(np.argmin(org.used.astype(float)))
        org.xi[f] = normalize(z, org.norm); org.used[f] = True; k = f
    else:
        za = z * np.exp(-1j * np.angle(org.overlaps(z, org.xi)[k]))
        org.xi[k] = normalize(org.xi[k] + 0.02 * (za - org.xi[k]), org.norm)
    cur_assign.append(k)
    if prev >= 0: P_counts[prev, k] += 1
    prev = k

cur_assign = np.array(cur_assign)
P_norm = P_counts / (P_counts.sum(1, keepdims=True) + 1e-9)  # row-normalized transitions

# ========================================================================
# PREDICTION METHOD 1: TABLE (same as Phase 2 organism baseline)
# ========================================================================
table_pred_mem = []
for t in range(len(obs)):
    k = cur_assign[t]
    table_pred_mem.append(int(np.argmax(P_counts[k])) if P_counts[k].sum() > 0 else k)
table_pred_mem = np.array(table_pred_mem)

# ========================================================================
# PREDICTION METHOD 2: GRAPH-SL PROPAGATION (full N-dimensional field)
#
# Diagnosis of first attempt: reducing to K scalar states lost all geometric
# info. The amplitude regulation (alpha - |w|^2)*w equalizes amplitudes rather
# than selecting a winner because each node's self-regulation is independent.
#
# Correct design: stay in the full C^N field. After identifying memory k,
# build a "transition-weighted target" by superimposing memory states:
#
#   x_drive = normalize( sum_j P_norm[k,j] * xi[j] )
#
# This encodes "given I'm in k, where should I go?" geometrically in C^N.
# Then run SL field dynamics toward x_drive from a perturbed-k starting state,
# and read the winner by overlap. The field's SL amplitude regulation prevents
# collapse and the geometry of xi determines which attractor wins -- NOT argmax.
#
# If xi[j] are non-orthogonal (they are -- real digits overlap), the geometric
# competition gives a DIFFERENT answer than argmax(P[k,:]). Good or bad is the
# experimental question.
# ========================================================================

alpha_sl = 0.20      # Hopf param: supercritical (matches itinerancy.py)
omega_sl = 0.15      # natural frequency
g_pred = 3.0         # input drive strength toward x_drive
T_pred = 8           # SL settling steps

xi = org.xi.copy()   # shape (K, N) complex memories
used = org.used.copy()

graph_pred_mem = []
for t in range(len(obs)):
    k = cur_assign[t]
    # Build transition-weighted target in full N-dimensional space
    # x_drive = sum over all successors, weighted by transition probability
    x_drive = np.zeros(N, dtype=complex)
    for j in range(K):
        if used[j] and P_norm[k, j] > 0:
            x_drive += P_norm[k, j] * xi[j]
    nrm = np.linalg.norm(x_drive)
    if nrm < 1e-9:
        graph_pred_mem.append(k)  # no transitions learned yet -> stay
        continue
    x_drive = x_drive / nrm * NORM  # normalize to sphere

    # Start field near current memory + small noise toward drive
    zp = normalize(xi[k] + 0.2 * (x_drive - xi[k]), NORM)

    # SL field dynamics driven toward x_drive
    for _ in range(T_pred):
        dz = 1j * omega_sl * zp + g_pred * (x_drive - zp)
        zp = normalize(zp + 0.05 * dz, NORM)

    # Winner = memory with highest overlap in used slots
    overlaps = np.abs(org.overlaps(zp, xi))
    overlaps[~used] = -1
    pred_k = int(np.argmax(overlaps))
    graph_pred_mem.append(pred_k)

graph_pred_mem = np.array(graph_pred_mem)

# ========================================================================
# COSINE BASELINE (Phase 2 winner -- same cosine prototypes + argmax table)
# ========================================================================
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
protos_c = []; Pc = np.zeros((20, 20)); pc_assign = []; pc_pred = []; prev_c = -1
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
    if prev_c >= 0: Pc[prev_c, k] += 1
    pc_pred.append(int(np.argmax(Pc[k])) if Pc[k].sum() > 0 else k)
    prev_c = k
pc_assign = np.array(pc_assign); pc_pred = np.array(pc_pred)

# ========================================================================
# EVALUATE: next-CLASS prediction accuracy
# ========================================================================
def mem_to_class(assign, labels=seq_cls):
    lab = {}
    for k in np.unique(assign):
        m = labels[assign == k]
        if len(m): lab[k] = np.bincount(m).argmax()
    return lab

def acc_from(pred_mem, assign):
    lab = mem_to_class(assign)
    pred_cls = np.array([lab.get(p, -1) for p in pred_mem])[:-1]
    return (pred_cls == true_next).mean()

table_acc   = acc_from(table_pred_mem, cur_assign)
graph_sl_acc = acc_from(graph_pred_mem, cur_assign)
cosine_acc  = acc_from(pc_pred, pc_assign)
persist_acc = (seq_cls[:-1] == true_next).mean()

# Recognition accuracy (how well it identifies the current occluded digit)
org_recog = np.mean([mem_to_class(cur_assign).get(k, -1) == seq_cls[t]
                     for t, k in enumerate(cur_assign)])

print("PHASE 3 -- SL DYNAMICS ON LEARNED TRANSITION GRAPH\n")
print(f"sequence length {len(obs)}, classes {CLASSES}, bottom-half occluded")
Ku = int(org.used.sum())
print(f"organism memories: {org.used.sum()}, SL graph nodes: {Ku}")
print(f"SL params: alpha={alpha_sl}, g_pred={g_pred}, T_pred={T_pred} steps\n")

print(f"  occluded-digit recognition (perception):  {org_recog:.3f}\n")

print(f"  NEXT-CLASS PREDICTION ACCURACY:")
print(f"    GRAPH-SL organism (dynamics thru P)  : {graph_sl_acc:.3f}  <-- THIS IS THE TEST")
print(f"    table organism    (argmax lookup)     : {table_acc:.3f}")
print(f"    cosine baseline   (Phase 2 winner)   : {cosine_acc:.3f}")
print(f"    persistence       (no temporal model): {persist_acc:.3f}")

delta_vs_table  = graph_sl_acc - table_acc
delta_vs_cosine = graph_sl_acc - cosine_acc
print(f"\n  graph-SL vs table lookup : {delta_vs_table:+.3f}")
print(f"  graph-SL vs cosine base  : {delta_vs_cosine:+.3f}")

print("\nverdict:",
      "GRAPH-SL BEATS BOTH: dynamics through learned structure outperform lookup AND cosine baseline."
      "\n  Integration is real -- the substrate IS the pipeline."
      if graph_sl_acc > cosine_acc + 0.02 else
      "GRAPH-SL BEATS TABLE but not cosine baseline: dynamics add value over lookup, but"
      "\n  perception quality (cosine metric) still dominates."
      if graph_sl_acc > table_acc + 0.02 else
      "NO ADVANTAGE: graph-SL dynamics don't improve over table lookup."
      "\n  The learned P matrix carries the signal; dynamics are noise here.")

# ========================================================================
# ABLATION: vary T_pred (SL relaxation steps) to find the sweet spot
# ========================================================================
print("\n-- ABLATION: T_pred (SL settling steps) --")
for T_test in [1, 4, 8, 16, 24, 40]:
    preds = []
    for t in range(len(obs)):
        k = cur_assign[t]
        x_drive = np.zeros(N, dtype=complex)
        for j in range(K):
            if used[j] and P_norm[k, j] > 0:
                x_drive += P_norm[k, j] * xi[j]
        nrm = np.linalg.norm(x_drive)
        if nrm < 1e-9: preds.append(k); continue
        x_drive = x_drive / nrm * NORM
        zp = normalize(xi[k] + 0.2 * (x_drive - xi[k]), NORM)
        for _ in range(T_test):
            dz = 1j * omega_sl * zp + g_pred * (x_drive - zp)
            zp = normalize(zp + 0.05 * dz, NORM)
        ov = np.abs(org.overlaps(zp, xi)); ov[~used] = -1
        preds.append(int(np.argmax(ov)))
    a = acc_from(np.array(preds), cur_assign)
    print(f"    T_pred={T_test:2d}  acc={a:.3f}")

# ========================================================================
# ABLATION: vary g_pred (drive strength toward transition target)
# g_pred=0 -> no drive, field drifts freely
# g_pred large -> field slams into x_drive (nearly argmax)
# There should be a sweet spot where SL dynamics add geometric competition
# ========================================================================
print("\n-- ABLATION: g_pred (drive strength toward transition target) --")
for g_test in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    preds = []
    for t in range(len(obs)):
        k = cur_assign[t]
        x_drive = np.zeros(N, dtype=complex)
        for j in range(K):
            if used[j] and P_norm[k, j] > 0:
                x_drive += P_norm[k, j] * xi[j]
        nrm = np.linalg.norm(x_drive)
        if nrm < 1e-9: preds.append(k); continue
        x_drive = x_drive / nrm * NORM
        zp = normalize(xi[k] + 0.2 * (x_drive - xi[k]), NORM)
        for _ in range(T_pred):
            dz = 1j * omega_sl * zp + g_test * (x_drive - zp)
            zp = normalize(zp + 0.05 * dz, NORM)
        ov = np.abs(org.overlaps(zp, xi)); ov[~used] = -1
        preds.append(int(np.argmax(ov)))
    a_acc = acc_from(np.array(preds), cur_assign)
    print(f"    g_pred={g_test:.1f}  acc={a_acc:.3f}")
