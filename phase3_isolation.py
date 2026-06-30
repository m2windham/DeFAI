"""
PHASE 3b -- ISOLATING THE PREDICTION MECHANISM.

Phase 3a showed: graph-SL prediction (0.577) > table prediction (0.551) > persist (0.0)
but still < cosine baseline (0.678). Question: is the gap between 0.577 and 0.678
due to prediction mechanism quality, or perception quality?

This experiment gives BOTH methods the same (cosine) perception, then tests:
  A. cosine-perceive + argmax-table   (Phase 2 cosine winner)
  B. cosine-perceive + graph-SL       (Phase 3 mechanism on better perception)

If B > A: the graph-SL dynamics add genuine prediction value on top of good perception.
If B ≈ A: the table is optimal given perception; dynamics add no value.
If B < A: dynamics HURT -- the geometric computation is noisier than argmax.

Also tests: organism-perceive + graph-SL (Phase 3a) vs cosine-perceive + graph-SL
to directly measure how much the perception gap costs.
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
T_true = np.array([[0, .8, .07, .07, .06],
                   [.06, 0, .8, .07, .07],
                   [.07, .06, 0, .8, .07],
                   [.07, .07, .06, 0, .8],
                   [.8, .07, .07, .06, 0]])

def occlude(v):
    v = v.copy().reshape(8, 8); v[4:, :] = 0.0; return v.reshape(-1)

def unit(v): return v / (np.linalg.norm(v) + 1e-9)

seq_cls, obs = [], []
c = 0
for t in range(4000):
    seq_cls.append(c)
    obs.append(occlude(byc[c][rng.integers(len(byc[c]))]))
    c = rng.choice(CLASSES, p=T_true[c])
seq_cls = np.array(seq_cls); obs = np.array(obs)
true_next = seq_cls[1:]

# ====== SHARED: organism perception + memories ==============================
K = 20
org = Organism(N=N, K=K, omega=0.15, beta=10.0, seed=0)
P_org = np.zeros((K, K))
z = org.z
org_assign = []
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
    org_assign.append(k)
    if prev >= 0: P_org[prev, k] += 1
    prev = k
org_assign = np.array(org_assign)
P_org_norm = P_org / (P_org.sum(1, keepdims=True) + 1e-9)
xi = org.xi.copy(); used = org.used.copy()

# ====== SHARED: cosine perception + transitions =============================
protos_c = []; Pc = np.zeros((20, 20)); cos_assign = []; prev_c = -1
for t, raw in enumerate(obs):
    u = unit(raw)
    if protos_c:
        sims = np.array(protos_c) @ u; k = int(np.argmax(sims)); best = sims[k]
    else:
        k, best = -1, -1
    if k < 0 or (best < 0.6 and len(protos_c) < 20):
        protos_c.append(u.copy()); k = len(protos_c) - 1
    else:
        protos_c[k] = unit(0.98 * protos_c[k] + 0.02 * u)
    cos_assign.append(k)
    if prev_c >= 0: Pc[prev_c, k] += 1
    prev_c = k
cos_assign = np.array(cos_assign)
Pc_norm = Pc / (Pc.sum(1, keepdims=True) + 1e-9)
protos_c_arr = np.array(protos_c)  # shape (n_protos, N)

# ====== SL PREDICTION helper (best params from Phase 3a ablation) ===========
omega_sl = 0.15
g_pred = 8.0
T_pred = 30
dt_sl = 0.05

def sl_predict_from(k, P_norm_local, xi_local, used_local):
    """Predict next memory via SL dynamics toward transition-weighted target."""
    x_drive = np.zeros(N, dtype=complex)
    for j in range(len(used_local)):
        if used_local[j] and P_norm_local[k, j] > 0:
            x_drive += P_norm_local[k, j] * xi_local[j]
    nrm = np.linalg.norm(x_drive)
    if nrm < 1e-9:
        return k
    x_drive = x_drive / nrm * NORM
    zp = normalize(xi_local[k] + 0.3 * (x_drive - xi_local[k]), NORM)
    for _ in range(T_pred):
        dz = 1j * omega_sl * zp + g_pred * (x_drive - zp)
        zp = normalize(zp + dt_sl * dz, NORM)
    ov = np.abs((xi_local.conj() @ zp) / N)
    ov[~used_local] = -1
    return int(np.argmax(ov))

# ====== A: cosine-perceive + argmax table (Phase 2 cosine winner) ===========
cos_table_pred = []
for t in range(len(obs)):
    k = cos_assign[t]
    cos_table_pred.append(int(np.argmax(Pc[k])) if Pc[k].sum() > 0 else k)
cos_table_pred = np.array(cos_table_pred)

# ====== B: cosine-perceive + graph-SL =======================================
# Use cosine prototypes as "memories" (real-valued) in the SL target computation.
# We need complex xi for SL: embed real cosine protos as complex (imag=0).
n_cp = len(protos_c)
xi_cos = np.zeros((n_cp, N), dtype=complex)
for j, p in enumerate(protos_c):
    xi_cos[j] = p.astype(complex) * NORM   # scale to sphere
used_cos = np.ones(n_cp, bool)
P_cos_pad = np.zeros((n_cp, n_cp))
P_cos_pad[:n_cp, :n_cp] = Pc[:n_cp, :n_cp]
P_cos_norm = P_cos_pad / (P_cos_pad.sum(1, keepdims=True) + 1e-9)

cos_sl_pred = []
for t in range(len(obs)):
    k = cos_assign[t]
    cos_sl_pred.append(sl_predict_from(k, P_cos_norm, xi_cos, used_cos))
cos_sl_pred = np.array(cos_sl_pred)

# ====== C: organism-perceive + graph-SL (Phase 3a best) =====================
org_sl_pred = []
for t in range(len(obs)):
    k = org_assign[t]
    org_sl_pred.append(sl_predict_from(k, P_org_norm, xi, used))
org_sl_pred = np.array(org_sl_pred)

# ====== D: organism-perceive + argmax table =================================
org_table_pred = []
for t in range(len(obs)):
    k = org_assign[t]
    org_table_pred.append(int(np.argmax(P_org[k])) if P_org[k].sum() > 0 else k)
org_table_pred = np.array(org_table_pred)

# ====== EVALUATE ============================================================
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

a_acc = acc_from(cos_table_pred, cos_assign)   # A
b_acc = acc_from(cos_sl_pred,    cos_assign)   # B
c_acc = acc_from(org_sl_pred,    org_assign)   # C
d_acc = acc_from(org_table_pred, org_assign)   # D
persist_acc = (seq_cls[:-1] == true_next).mean()

print("PHASE 3b -- ISOLATING PREDICTION MECHANISM FROM PERCEPTION\n")
print(f"  {'method':<42s}  accuracy")
print(f"  {'A. cosine-perceive + argmax-table':<42s}  {a_acc:.3f}  (Phase 2 winner)")
print(f"  {'B. cosine-perceive + graph-SL dynamics':<42s}  {b_acc:.3f}  <- does SL help?")
print(f"  {'C. organism-perceive + graph-SL dynamics':<42s}  {c_acc:.3f}")
print(f"  {'D. organism-perceive + argmax-table':<42s}  {d_acc:.3f}")
print(f"  {'persistence (no model)':<42s}  {persist_acc:.3f}")

print(f"\n  graph-SL vs argmax table (cosine perception): {b_acc - a_acc:+.3f}")
print(f"  graph-SL vs argmax table (org perception)  : {c_acc - d_acc:+.3f}")
print(f"  perception gap (org vs cosine, same SL)    : {c_acc - b_acc:+.3f}")
print(f"  perception gap (org vs cosine, same table) : {d_acc - a_acc:+.3f}")

print("\nverdict:",
      "GRAPH-SL ADDS GENUINE PREDICTION VALUE: B > A with same cosine perception."
      if b_acc > a_acc + 0.02 else
      "TABLE SUFFICIENT: graph-SL dynamics don't improve over argmax given same perception."
      f"\n  The prediction bottleneck is perception (gap {a_acc - d_acc:+.3f}), not the prediction step."
      if abs(b_acc - a_acc) < 0.02 else
      f"GRAPH-SL HURTS: dynamics noisier than table (B - A = {b_acc - a_acc:+.3f}).")
