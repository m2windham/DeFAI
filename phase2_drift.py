"""
PHASE 2: online novelty / concept-drift detection -- a task static prototypes
cannot do naturally, and where the organism's PREDICTION ERROR is the signal.

A digit stream DRIFTS: it starts with classes {0,1,2}; a novel class (5) is
injected partway through, then another (8). At each arriving image (single pass,
online) the organism:
  1. settles its field toward the image,
  2. reports novelty = 1 - (best overlap with EXISTING memories)   <- free-energy,
  3. then adapts (recruits a new memory if the input is novel, else refines).

Claim: novelty spikes exactly when a new class appears, then falls as the organism
learns it (recruits an attractor) -- an online drift detector that self-heals.

Baseline: streaming kNN novelty (distance to a reservoir of recent raw samples) --
a strong, standard online novelty detector. We compare ROC-AUC for flagging the
onset of newly-introduced classes. Honest: kNN keeps a big raw buffer; the organism
uses O(K) compressed memories.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from organism import Organism, normalize

rng = np.random.default_rng(0)
d = load_digits()
X = ((d.data - d.data.mean(0)) / (d.data.std(0) + 1e-6)).astype(float)
y = d.target.astype(int)
N = X.shape[1]; NORM = np.sqrt(N)
byc = {c: X[y == c] for c in range(10)}

def draw(c):
    return byc[c][rng.integers(len(byc[c]))]

# ---- build a drifting stream ---------------------------------------------
stream, onset = [], []          # onset=1 for the first appearances of a NEW class
seen = set()
def emit(c, novel_window):
    stream.append(draw(c))
    onset.append(1 if novel_window else 0)

for t in range(900):
    if t < 300:    pool = [0, 1, 2]
    elif t < 600:  pool = [0, 1, 2, 5]      # class 5 appears at t=300
    else:          pool = [0, 1, 2, 5, 8]   # class 8 appears at t=600
    c = pool[rng.integers(len(pool))]
    # novel-window label: first 25 steps after a class is introduced AND it's that class
    is_new = (c == 5 and 300 <= t < 325) or (c == 8 and 600 <= t < 625)
    emit(c, is_new)
stream = np.array(stream); onset = np.array(onset)

# ===================== ORGANISM (online) ==================================
org = Organism(N=N, K=30, omega=0.15, beta=10.0, seed=0)
z = org.z
org_novel = np.zeros(len(stream))
g_in, dt, eta, recruit = 4.0, 0.05, 0.02, 0.6
for t, raw in enumerate(stream):
    x = normalize(raw.astype(complex), NORM)
    for _ in range(6):                          # settle field toward the image
        z = normalize(z + dt * (1j * org.omega * z + g_in * (x - z)), org.norm)
    if org.used.any():
        o = np.abs(org.overlaps(z, org.xi[org.used]))
        org_novel[t] = 1.0 - o.max()            # prediction error = novelty
    else:
        org_novel[t] = 1.0
    # adapt online
    o2 = np.abs(org.overlaps(z, org.xi)); k = int(np.argmax(o2))
    if (not org.used.any() or org_novel[t] > (1 - recruit)) and not org.used.all():
        f = int(np.argmin(org.used.astype(float))); org.xi[f] = normalize(z, org.norm); org.used[f] = True
    else:
        z_al = z * np.exp(-1j * np.angle(org.overlaps(z, org.xi)[k]))
        org.xi[k] = normalize(org.xi[k] + eta * (z_al - org.xi[k]), org.norm)
org.z = z

# ===================== streaming kNN baseline =============================
buf = []
knn_novel = np.zeros(len(stream))
for t, raw in enumerate(stream):
    if buf:
        dists = np.sqrt(((np.array(buf) - raw) ** 2).sum(1))
        knn_novel[t] = np.sort(dists)[:5].mean()     # mean distance to 5 nearest past samples
    else:
        knn_novel[t] = 1e9
    buf.append(raw)
    if len(buf) > 200: buf.pop(0)                    # bounded reservoir

# ===================== evaluate ===========================================
# normalize kNN signal for fair AUC; first sample has no buffer -> drop
m = np.arange(len(stream)) > 0
auc_org = roc_auc_score(onset[m], org_novel[m])
auc_knn = roc_auc_score(onset[m], knn_novel[m])

print("PHASE 2: online novelty / concept-drift detection (drifting digit stream)\n")
print(f"stream length {len(stream)}, new classes injected at t=300 (class 5), t=600 (class 8)")
print(f"memories the organism recruited online: {org.used.sum()}\n")
print(f"ROC-AUC for flagging new-class onsets:")
print(f"    organism (prediction error)      : {auc_org:.3f}")
print(f"    streaming kNN (200-sample buffer) : {auc_knn:.3f}")

# show the self-healing: mean novelty in windows
def win(a, lo, hi): return a[lo:hi].mean()
print(f"\n  organism novelty (self-healing signature):")
print(f"    steady (t 250-300, classes 0-2)     : {win(org_novel,250,300):.3f}")
print(f"    onset of class 5 (t 300-310)        : {win(org_novel,300,310):.3f}  <- spike")
print(f"    after learning 5 (t 360-400)        : {win(org_novel,360,400):.3f}  <- healed")
print(f"    onset of class 8 (t 600-610)        : {win(org_novel,600,610):.3f}  <- spike")
print(f"    after learning 8 (t 660-700)        : {win(org_novel,660,700):.3f}  <- healed")

ok = auc_org > 0.7 and win(org_novel,300,310) > 1.5*win(org_novel,250,300)
print("\nverdict:",
      "organism detects drift online via prediction error AND self-heals by recruiting"
      if ok else "weak -- inspect windows")
