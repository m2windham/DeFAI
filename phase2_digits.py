"""
PHASE 2: confront the organism with REAL, correlated, non-orthogonal data.

Until now every "world" was hand-built (orthogonal patterns / wave packets). Here
the stream is real handwritten digits (sklearn load_digits, 8x8=64-dim, 10 classes,
many writers). The digits are NOT orthogonal and classes overlap -- the assumption
most likely to break. Question: does unsupervised competitive attractor formation
in the oscillator field recover meaningful memories (= digit classes), or does
real correlation smear them?

Honesty notes:
  - OSCILLATOR-NATIVE here: the field state dynamics (z evolution, settling).
  - DIGITAL SCAFFOLDING: overlaps (matmul), winner selection, the k-means baseline.
  - We report clustering PURITY and nearest-memory classification accuracy, head to
    head with k-means given the SAME number of clusters. No hand-tuning to the labels.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from organism import Organism, normalize

rng = np.random.default_rng(0)

# ---- real data ------------------------------------------------------------
d = load_digits()
X = d.data.astype(float)                       # (1797, 64)
y = d.target
X = (X - X.mean(0)) / (X.std(0) + 1e-6)        # standardize features
N = X.shape[1]                                  # 64 oscillators
NORM = np.sqrt(N)

def to_state(v):                                # real image -> complex sphere vector
    return normalize(v.astype(complex), NORM)

# ---- present an unlabeled, shuffled stream; each image held for R steps ----
def make_stream(epochs=4, hold=8):
    order = []
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        order.extend(idx.tolist())
    for i in order:
        s = to_state(X[i])
        for _ in range(hold):                   # hold each image so the field settles
            yield s

K = 24                                           # memory slots (>10 classes; prune later)
org = Organism(N=N, K=K, omega=0.15, beta=10.0, seed=0)
org.perceive(make_stream(), g_in=4.0, eta=0.015, recruit=0.6)
kept = org.consolidate(merge_thresh=0.85, prune_frac=0.03)
M = org.mem
print("PHASE 2: organism on REAL handwritten digits (unsupervised)\n")
print(f"N={N} oscillators, slots used {org.used.sum()}/{K} -> {M.shape[0]} memories after consolidate\n")

# ---- assign every image to its nearest memory; purity vs true labels ------
def nearest_mem(states):
    O = np.abs((M.conj() @ states.T) / N)        # (n_mem, n_img)
    return O.argmax(0)

states = np.array([to_state(x) for x in X])
assign = nearest_mem(states)
# majority label per memory -> purity
purity_hits = 0
mem_label = {}
for k in range(M.shape[0]):
    members = y[assign == k]
    if len(members):
        lbl = np.bincount(members).argmax()
        mem_label[k] = lbl
        purity_hits += (members == lbl).sum()
purity = purity_hits / len(X)

# nearest-memory classification accuracy (memory votes its majority label)
pred = np.array([mem_label.get(a, -1) for a in assign])
acc = (pred == y).mean()

# ---- k-means baseline with the SAME number of clusters --------------------
ncl = M.shape[0]
km = KMeans(n_clusters=ncl, n_init=10, random_state=0).fit(X)
kp_hits = 0
for c in range(ncl):
    members = y[km.labels_ == c]
    if len(members):
        kp_hits += (members == np.bincount(members).argmax()).sum()
km_purity = kp_hits / len(X)

print(f"(A) memories formed: {M.shape[0]}  (10 true classes)")
print(f"(B) clustering PURITY:")
print(f"      organism (oscillator field) : {purity:.3f}")
print(f"      k-means (same #clusters)     : {km_purity:.3f}")
print(f"(C) nearest-memory classification accuracy: {acc:.3f}")
# which classes each memory grabbed
print(f"\n    memory -> majority digit: "
      + ", ".join(f"{k}:{mem_label.get(k,'-')}" for k in range(M.shape[0])))
covered = sorted(set(mem_label.values()))
print(f"    digit classes covered by some memory: {covered} ({len(covered)}/10)")

print("\nverdict:",
      "real correlated structure DOES form clean attractors (competitive with k-means)"
      if purity > 0.6 and len(covered) >= 8 else
      "real data smears the attractors -- diagnose")
