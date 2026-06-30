"""
PHASE 4a -- CONTINUOUS EMBEDDING PIPELINE (v2)

The organism stops speaking in class labels and starts speaking in geometry.

Key insight from diagnosis:
  - Memory alignment is near-perfect (0.986-0.995) — memories ARE the embeddings.
  - The iω rotation during prediction steps corrupts the real-part decode.
  - Fix: overlap-weighted decode — project field state against all memories,
    then use those overlaps as weights to reconstruct the embedding-space vector.

Key split: entropy of transition distribution
  - LOW entropy  : one clear successor → table should win (pick the winner)
  - HIGH entropy : many possible successors → field should win (stay in geometric
                   mean of successors rather than committing to one wrong one)

Transition structure:
  - Concepts 0-4 ('cluster A') cycle: 0→1→2→3→4→0 with p=0.85  [low entropy]
  - Concepts 5-19 ('open field'): uniform 1/15 transitions         [high entropy]
  - The organism's geometry should shine in the open-field region.
"""

import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(42)

# ---- concepts: 20 'words', 32-dim embeddings --------------------------------
N_CONCEPTS = 20
DIM = 32
N = DIM
NORM = np.sqrt(N)

# cluster A (0-4): tight cycle; cluster B (5-19): open field
cluster_A_center = rng.standard_normal(DIM)
cluster_A_center /= np.linalg.norm(cluster_A_center)
cluster_B_centers = rng.standard_normal((3, DIM))
cluster_B_centers /= np.linalg.norm(cluster_B_centers, axis=1, keepdims=True)

embeddings = np.zeros((N_CONCEPTS, DIM))
for i in range(5):   # cluster A: tight
    embeddings[i] = cluster_A_center + 0.15 * rng.standard_normal(DIM)
for i in range(5, 20):  # cluster B: spread across 3 sub-clusters
    sc = (i - 5) // 5
    embeddings[i] = cluster_B_centers[sc] + 0.4 * rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

# ---- transition structure ---------------------------------------------------
T_matrix = np.zeros((N_CONCEPTS, N_CONCEPTS))
# cluster A: tight cycle 0→1→2→3→4→0 (low entropy)
for i in range(5):
    nxt = (i + 1) % 5
    T_matrix[i, nxt] = 0.85
    # small probability to open field
    for j in range(5, 20):
        T_matrix[i, j] = 0.15 / 15
# cluster B: nearly uniform over all (high entropy)
for i in range(5, 20):
    for j in range(N_CONCEPTS):
        if i != j:
            T_matrix[i, j] = 1.0 / 19
T_matrix /= T_matrix.sum(1, keepdims=True)

# ---- sequence --------------------------------------------------------------
SEQ_LEN = 4000
seq = [0]
for _ in range(SEQ_LEN - 1):
    seq.append(rng.choice(N_CONCEPTS, p=T_matrix[seq[-1]]))
seq = np.array(seq)
obs = embeddings[seq]

# ---- ORGANISM --------------------------------------------------------------
org = Organism(N=N, K=30, omega=0.15, beta=10.0, seed=0)
z = org.z
P = np.zeros((30, 30))
cur_assign = []
predicted_z = []
prev_k = -1

for t, raw in enumerate(obs):
    x = normalize(raw.astype(complex), NORM)

    for _ in range(8):
        z = normalize(z + 0.05 * (1j * org.omega * z + 4.0 * (x - z)), org.norm)

    o2 = np.abs(org.overlaps(z, org.xi))
    k = int(np.argmax(o2))
    nov = 1 - (np.abs(org.overlaps(z, org.xi[org.used])).max() if org.used.any() else 0)
    if nov > 0.4 and not org.used.all():
        f = int(np.argmin(org.used.astype(float)))
        org.xi[f] = normalize(z, org.norm); org.used[f] = True; k = f
    else:
        za = z * np.exp(-1j * np.angle(org.overlaps(z, org.xi)[k]))
        org.xi[k] = normalize(org.xi[k] + 0.02 * (za - org.xi[k]), org.norm)

    cur_assign.append(k)
    if prev_k >= 0:
        P[prev_k, k] += 1

    # prediction: drive field toward P-weighted target in field space
    if P[k].sum() > 0:
        p_row = P[k] / P[k].sum()
        used_idx = np.where(org.used)[0]
        x_pred = np.sum([p_row[j] * org.xi[j] for j in used_idx], axis=0)
        x_pred = normalize(x_pred, org.norm)
        z_pred = z.copy()
        for _ in range(20):                        # more steps: better convergence
            z_pred = normalize(
                z_pred + 0.05 * (1j * org.omega * z_pred + 6.0 * (x_pred - z_pred)),
                org.norm
            )
        predicted_z.append(z_pred)
    else:
        predicted_z.append(z.copy())

    prev_k = k

cur_assign = np.array(cur_assign)

# ---- decode: overlap-weighted embedding reconstruction ---------------------
def decode_to_embedding(z_field):
    """
    Project field state to embedding space via overlap weights.
    Avoids iω rotation artifact of the real-part decode.
    Uses |overlap| with each memory as the weight, then reconstructs
    the embedding-space position as the weighted mean.
    """
    used_idx = np.where(org.used)[0]
    if len(used_idx) == 0:
        return z_field.real / (np.linalg.norm(z_field.real) + 1e-9)
    overlaps = np.abs(org.overlaps(z_field, org.xi[used_idx]))  # (n_used,)
    overlaps = np.maximum(overlaps, 0)
    if overlaps.sum() < 1e-9:
        return z_field.real / (np.linalg.norm(z_field.real) + 1e-9)
    # reconstruct: overlap-weighted mean of memory real parts (≈ concept embeddings)
    mem_reals = np.array([org.xi[j].real for j in used_idx])
    recon = (overlaps[:, None] * mem_reals).sum(0) / overlaps.sum()
    return recon / (np.linalg.norm(recon) + 1e-9)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ---- evaluate --------------------------------------------------------------
sim_org, sim_wtd, sim_tbl, sim_per = [], [], [], []

for t in range(len(predicted_z) - 1):
    actual = embeddings[seq[t + 1]]
    k = cur_assign[t]

    # organism: overlap-weighted decode
    sim_org.append(cos(decode_to_embedding(predicted_z[t]), actual))

    if P[k].sum() > 0:
        p_row = P[k] / P[k].sum()
        used_idx = np.where(org.used)[0]

        # weighted (soft table, no dynamics): P-weighted mean of memory embeddings
        mem_reals = np.array([org.xi[j].real for j in used_idx])
        mem_reals_n = mem_reals / (np.linalg.norm(mem_reals, axis=1, keepdims=True) + 1e-9)
        w = np.array([p_row[j] for j in used_idx])
        wtd_emb = (w[:, None] * mem_reals_n).sum(0)
        wtd_emb /= np.linalg.norm(wtd_emb) + 1e-9
        sim_wtd.append(cos(wtd_emb, actual))

        # table: argmax P[k] → discrete label
        j_best = int(np.argmax(p_row))
        tbl_emb = org.xi[j_best].real
        tbl_emb = tbl_emb / (np.linalg.norm(tbl_emb) + 1e-9)
        sim_tbl.append(cos(tbl_emb, actual))
    else:
        sim_wtd.append(0.0); sim_tbl.append(0.0)

    sim_per.append(cos(embeddings[seq[t]], actual))

# ---- entropy split ---------------------------------------------------------
def entropy(p):
    p = p[p > 0]; return float(-np.sum(p * np.log(p + 1e-9)))

hi_idx, lo_idx = [], []
for t in range(len(predicted_z) - 1):
    k = cur_assign[t]
    if P[k].sum() > 0:
        H = entropy(P[k] / P[k].sum())
        (hi_idx if H > 1.5 else lo_idx).append(t)

def mean_at(sims, idx):
    return float(np.mean([sims[i] for i in idx])) if idx else float('nan')

# ---- cluster-origin split --------------------------------------------------
# cluster A origin: steps where current concept is in 0-4 (cycle, low entropy)
# cluster B origin: steps where current concept is in 5-19 (open, high entropy)
clA_idx = [t for t in range(len(predicted_z) - 1) if seq[t] < 5]
clB_idx = [t for t in range(len(predicted_z) - 1) if seq[t] >= 5]

# ---- report ----------------------------------------------------------------
print("PHASE 4a -- CONTINUOUS EMBEDDING PIPELINE (v2)\n")
print(f"Concepts: {N_CONCEPTS}  |  DIM: {DIM}  |  N oscillators: {N}")
print(f"Cluster A (0-4): cycle 0→1→2→3→4→0 p=0.85  [LOW entropy]")
print(f"Cluster B (5-19): uniform transitions        [HIGH entropy]")
print(f"Sequence: {SEQ_LEN} steps\n")

print("Overall prediction quality (cosine sim to actual next embedding):")
print(f"  Organism  (field → overlap-weighted decode) : {np.mean(sim_org):.4f}")
print(f"  Weighted  (soft P table, no dynamics)       : {np.mean(sim_wtd):.4f}")
print(f"  Table     (argmax P[k] → discrete)          : {np.mean(sim_tbl):.4f}")
print(f"  Persist   (current = next)                  : {np.mean(sim_per):.4f}")

print(f"\nBy entropy of P[k] distribution:")
print(f"  LOW entropy (n={len(lo_idx)})  — organism: {mean_at(sim_org, lo_idx):.4f}  "
      f"weighted: {mean_at(sim_wtd, lo_idx):.4f}  table: {mean_at(sim_tbl, lo_idx):.4f}")
print(f"  HIGH entropy (n={len(hi_idx)}) — organism: {mean_at(sim_org, hi_idx):.4f}  "
      f"weighted: {mean_at(sim_wtd, hi_idx):.4f}  table: {mean_at(sim_tbl, hi_idx):.4f}")

print(f"\nBy origin cluster:")
print(f"  Cluster A / cycle (n={len(clA_idx)}) — organism: {mean_at(sim_org, clA_idx):.4f}  "
      f"weighted: {mean_at(sim_wtd, clA_idx):.4f}  table: {mean_at(sim_tbl, clA_idx):.4f}")
print(f"  Cluster B / open  (n={len(clB_idx)}) — organism: {mean_at(sim_org, clB_idx):.4f}  "
      f"weighted: {mean_at(sim_wtd, clB_idx):.4f}  table: {mean_at(sim_tbl, clB_idx):.4f}")

print(f"\nMemory slots used: {org.used.sum()}/30")

# ---- generation quality: does organism land on attractor or drift? ----------
print("\nGeneration quality: does field land on a real attractor?")
for label, idx in [("LOW entropy (cycle)", lo_idx[:50]), ("HIGH entropy (open)", hi_idx[:50])]:
    if not idx:
        continue
    overlap_at_pred = []
    for t in idx:
        best_ov = np.abs(org.overlaps(predicted_z[t], org.xi[org.used])).max()
        overlap_at_pred.append(float(best_ov))
    print(f"  {label}: mean max-overlap of predicted field = {np.mean(overlap_at_pred):.3f}")
    print(f"    (1.0 = field IS an attractor; 0.5 = halfway between)")
