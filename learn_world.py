"""
THE ORGANISM: form memories from an un-designed stream, then itinerate among them.

Nothing here is hand-fed as a clean pattern. A hidden world cycles irregularly
through H latent "regimes"; each emits NOISY high-dim observations. The field sees
only the noisy stream -- never the regime labels, never the clean generators. It
must (1) DISCOVER the recurring structure and write it into its own memory slots,
then (2) with the input switched OFF, AUTONOMOUSLY wander among the memories IT
formed -- recalling experiences nobody planted.

Fuses the validated pieces:
  - competitive Hebbian learning  -> forms attractors from experience (unsupervised)
  - modern-Hopfield retrieval      -> those attractors are recallable memories
  - sphere normalization           -> winner-take-all competition (AKOrN-style)
  - slow fatigue                   -> paces autonomous itinerancy in recall

Tests: (A) did the self-formed memories capture the hidden regimes it never saw
cleanly? (B) in recall (no input), does it itinerate among its OWN memories, and
do those correspond to real regimes (genuine recollection)?
"""

import numpy as np
rng = np.random.default_rng(0)

N, H, K = 128, 4, 6              # N units, H hidden regimes, K memory slots (>H)
NORM = np.sqrt(N)
obs_noise = 0.6                  # heavy observation noise on the stream
dt = 0.05

# hidden world generators (orthogonal), UNKNOWN to the field -- eval only
raw = rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H))
G, _ = np.linalg.qr(raw)
G = (G.T) * NORM                 # (H, N) true regimes

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-9) * NORM

def overlaps(z, M):
    return (M.conj() @ z) / N

# ---- memory slots: start RANDOM (not the true patterns) -------------------
xi = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
xi = np.array([normalize(x) for x in xi])
used = np.zeros(K, bool)

# ============================ PHASE 1: PERCEIVE + LEARN ====================
g_in, g_rec, omega, beta = 4.0, 2.0, 0.25, 12.0
eta = 0.02                       # slow competitive learning rate
recruit_thresh = 0.55            # below this match = novel experience -> new slot

z = normalize(rng.standard_normal(N) + 1j * rng.standard_normal(N))
regime, T_LEARN = 0, 60000
captured_log = []
for s in range(T_LEARN):
    if rng.random() < 0.004:                      # occasional regime switch
        regime = rng.integers(H)
    x = normalize(G[regime] + obs_noise * NORM / np.sqrt(N) *
                  (rng.standard_normal(N) + 1j * rng.standard_normal(N)))

    # FORMATION: let the input drive the field freely (no sharp retrieval, which
    # would collapse all frames onto the first-formed memory). Memory helps
    # perception later, in recall -- not while the clusters are still forming.
    dz = 1j * omega * z + g_in * (x - z)
    z = normalize(z + dt * dz)

    # competitive learning with recruitment
    o2 = overlaps(z, xi); m2 = np.abs(o2)
    kbest = int(np.argmax(m2))
    if m2[kbest] < recruit_thresh and not used.all():
        f = int(np.argmin(used.astype(float) + m2 * 1e-3))   # a free slot
        xi[f] = normalize(z); used[f] = True
    else:
        z_al = z * np.exp(-1j * np.angle(o2[kbest]))         # phase-align to slot
        xi[kbest] = normalize(xi[kbest] + eta * (z_al - xi[kbest]))
        used[kbest] = True

    if s % 2000 == 0:
        cap = [max(np.abs(overlaps(G[h], xi))) for h in range(H)]
        captured_log.append(np.mean(cap))

print("THE ORGANISM -- form memories from a noisy stream, then recall them\n")
print(f"N={N}, hidden regimes H={H}, memory slots K={K}, obs noise={obs_noise}")
print(f"slots recruited/used: {used.sum()}/{K}\n")

# ---- (A) did self-formed memories capture the hidden regimes? -------------
print("(A) regime capture (field never saw clean generators):")
for h in range(H):
    ov = np.abs(overlaps(G[h], xi))
    kbest = int(np.argmax(ov))
    print(f"    regime {h}: best matching slot {kbest} at overlap {ov[kbest]:.3f}")
cap_mean = np.mean([max(np.abs(overlaps(G[h], xi))) for h in range(H)])
print(f"    mean capture = {cap_mean:.3f}  ({'regimes learned' if cap_mean > 0.7 else 'weak'})")

# ============================ PHASE 2: RECALL / ITINERATE ==================
# input OFF; autonomous itinerancy among SELF-FORMED memories (used slots only).
print("\n(B) RECALL: input OFF -- autonomous itinerancy among self-formed memories")
Mu = xi[used]                                     # only memories it actually formed
Ku = Mu.shape[0]
tau_h, lam, Dnoise, g_rec2 = 25.0, 2.0, 0.005, 5.0
z = normalize(rng.standard_normal(N) + 1j * rng.standard_normal(N))
h_fat = np.zeros(Ku)
visit_seq = []
for s in range(40000):
    o = overlaps(z, Mu); m = np.abs(o)
    fat = np.maximum(1 - lam * h_fat, 0.0)
    score = m * fat
    w = np.exp(beta * (score - score.max())); w /= w.sum()
    phase = o / (m + 1e-9)
    T = (w * phase) @ Mu
    dz = 1j * omega * z + g_rec2 * (T - z)
    noise = np.sqrt(2 * Dnoise * dt) * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    z = normalize(z + dt * dz + noise)
    h_fat = h_fat + dt / tau_h * (m - h_fat)
    if s % 20 == 0:
        a = int(np.argmax(m))
        visit_seq.append(a if m[a] > 0.4 else -1)

visit_seq = np.array(visit_seq)
clear = visit_seq[visit_seq >= 0]
trans = int(np.sum(clear[1:] != clear[:-1])) if len(clear) > 1 else 0
visited = sorted(np.unique(clear).tolist())
# do the recalled memories correspond to REAL regimes?
recalled_regime = {}
for k in visited:
    ov = np.abs(overlaps(Mu[k], G))               # this memory's overlap with each true regime
    recalled_regime[k] = (int(np.argmax(ov)), float(ov.max()))

print(f"    self-formed memories visited: {visited}  ({len(visited)}/{Ku}), transitions={trans}")
print(f"    fraction of time in a clear memory: {np.mean(visit_seq >= 0):.2f}")
print("    each recalled memory -> real regime it corresponds to:")
for k in visited:
    h, ov = recalled_regime[k]
    print(f"      memory {k}  <->  regime {h}  (overlap {ov:.3f})")
distinct_regimes = len(set(v[0] for v in recalled_regime.values()))
print(f"\n    -> recall spans {distinct_regimes} distinct real regimes")
ok = cap_mean > 0.7 and trans >= 3 and distinct_regimes >= 2
print("\nverdict:", "THE ORGANISM forms its own memories and autonomously recalls them"
      if ok else "partial -- inspect which stage is weak")
