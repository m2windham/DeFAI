"""
Multi-attractor cognition: autonomous itinerancy among stored patterns.

No sensory input, ever. The field stores K complex patterns as attractors and
should AUTONOMOUSLY wander among them -- settle into one "thought", then leave it
for another -- driven purely by intrinsic dynamics. This is the jump from
tracking an external world to exploring its own state space.

Substrate: Stuart-Landau field z (complex). Stored patterns xi^mu (K,N).
  overlap     o_mu  = <xi^mu, z>                  (which thought is active)
  fatigue     h_mu  : slow variable, rises while mu is active, decays otherwise
  gain        g_mu  = relu(1 - lam * h_mu)        (fatigued thoughts are suppressed)
  recon       R     = sum_mu g_mu o_mu xi^mu / N  (pull toward non-fatigued matches)
  dz = (a0 + i w - |z|^2) z + g_rec (R - z) + noise

Winnerless competition: the best-matching, non-fatigued pattern wins and is
reinforced (an attractor); its fatigue then builds, releasing the field to the
next pattern. Noise breaks ties so it keeps exploring instead of freezing.

Tests: (1) is it metastable -- settles into patterns AND leaves them (not frozen,
not chaotic)? (2) coverage -- does it visit many/all stored patterns? (3) dwell
statistics -- clean attractor dwells punctuated by fast transitions?
"""

import numpy as np
rng = np.random.default_rng(0)

N, K = 128, 6
dt = 0.05
STEPS = 60000

a0, omega, g_rec = 0.3, 0.25, 5.0
tau_h, lam = 25.0, 2.0       # fatigue MUCH slower than locking: lock hard, dwell, exit
Dnoise = 0.005               # low: lock hard + dwell; fatigue (not noise) paces exits
beta = 15.0                  # modern-Hopfield retrieval sharpness (winner-take-all)

# stored patterns: ORTHOGONAL complex memories (QR) so each is a clean attractor
# with no cross-talk contamination -> the field can lock deeply (m -> ~1).
raw = rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))
Qm, _ = np.linalg.qr(raw)                       # N x K orthonormal columns
xi = (Qm.T) * np.sqrt(N)                         # K x N, orthonormal, norm sqrt(N)

# AKOrN-style: state lives on a sphere of fixed norm; normalization IS the
# competition (removes the amplitude escape route, forcing the field to pick one
# pattern). Dynamics = gradient ascent on alignment energy to FRESH patterns,
# plus a global phase rotation (the oscillation) and noise (symmetry breaking).
TARGET_NORM = np.sqrt(N)              # so perfect alignment to a pattern gives m=1
z = rng.standard_normal(N) + 1j * rng.standard_normal(N)
z = z / np.linalg.norm(z) * TARGET_NORM
h = np.zeros(K)

def overlaps(z):
    return (xi.conj() @ z) / N        # (K,) complex; |.|=1 when fully aligned

active_seq, m_log, h_log = [], [], []
for s in range(STEPS):
    o = overlaps(z)
    m = np.abs(o)
    fat = np.maximum(1 - lam * h, 0.0)            # fatigue gain (suppress recent thoughts)
    # modern-Hopfield retrieval: sharp softmax over fatigue-discounted overlaps
    # selects ONE pattern (winner-take-all, independent of subspace degeneracy).
    score = m * fat
    w = np.exp(beta * (score - score.max())); w = w / w.sum()
    phase = o / (m + 1e-9)                          # unit phase per pattern
    T = (w * phase) @ xi                            # target = selected pattern, full norm
    dz = 1j * omega * z + g_rec * (T - z)          # oscillation + pull toward the thought
    noise = np.sqrt(2 * Dnoise * dt) * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    z = z + dt * dz + noise
    z = z / np.linalg.norm(z) * TARGET_NORM        # project back to the sphere
    # slow fatigue on ABSOLUTE activity: a thought that was strongly active stays
    # suppressed for ~tau_h even after the field leaves it (prevents stickiness).
    h = h + dt / tau_h * (m - h)
    if s % 20 == 0:
        active = int(np.argmax(m))
        active_seq.append(active if m[active] > 0.4 else -1)    # -1 = no clear thought
        m_log.append(m.copy()); h_log.append(h.copy())

active_seq = np.array(active_seq)
m_log = np.array(m_log)

# ---- analysis -------------------------------------------------------------
print(f"Multi-attractor cognition: {K} stored patterns, N={N}, autonomous (no input)\n")
print("NaN check:", "FAILED" if np.any(np.isnan(z)) else "clean")
print(f"DIAG: peak overlap ever reached = {m_log.max():.3f}  "
      f"(needs >0.4 for a 'clear thought'); mean per-step max overlap = "
      f"{m_log.max(axis=1).mean():.3f}")

# transitions: collapse consecutive identical labels
clear = active_seq[active_seq >= 0]
trans = np.sum(clear[1:] != clear[:-1]) if len(clear) > 1 else 0
visited = np.unique(clear)
frac_locked = np.mean(active_seq >= 0)            # fraction of time in a clear thought

# dwell lengths
dwells = []
if len(clear):
    cur, run = clear[0], 1
    for x in clear[1:]:
        if x == cur: run += 1
        else: dwells.append(run); cur, run = x, 1
    dwells.append(run)
dwells = np.array(dwells)

print(f"\n(1) regime:")
print(f"    fraction of time in a clear attractor : {frac_locked:.2f}")
print(f"    transitions between attractors        : {trans}")
regime = ("FROZEN (one thought)" if trans <= 1 else
          "no stable thoughts" if frac_locked < 0.3 else
          "ITINERANT (settles then moves on)")
print(f"    -> {regime}")
print(f"\n(2) coverage: visited {len(visited)} of {K} stored patterns  {sorted(visited.tolist())}")
print(f"\n(3) dwell stats (in 20-step units): mean={dwells.mean():.1f}  "
      f"min={dwells.min() if len(dwells) else 0}  max={dwells.max() if len(dwells) else 0}")

# crude trajectory print: who is active over time
print("\n    thought trajectory (each char = ~400 steps, '.'=no clear thought):")
sym = "".join(str(x) if x >= 0 else '.' for x in active_seq[::20])
for i in range(0, len(sym), 80):
    print("    " + sym[i:i+80])

ok = trans >= 3 and len(visited) >= 3 and frac_locked > 0.3
print("\nverdict:", "AUTONOMOUS ITINERANCY -- the field thinks its way across memories"
      if ok else "not yet itinerant -- diagnose below")
