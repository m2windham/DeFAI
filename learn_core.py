"""
Learning the identity core K from a data stream (no hand-built prior).

Builds on itinerancy.py. Previously K_pred was DESIGNED to expect a travelling
wave. Here K starts as small random noise and must LEARN the world's dynamics by
slow predictive plasticity while the fast Stuart-Landau field itinerates.

Two timescales (the plastic/elastic split, made literal):
  FAST  z_i : the working field, integrated every dt.
  SLOW  K_ij: the identity core, nudged by a delta/predictive-coding rule so that
              it predicts z(t) from z(t-dt). Diagonal forced to 0 so K cannot
              cheat with the trivial "predict yourself" (K=I) collapse -- the
              learning analogue of the dark-room problem.

Active-inference field dynamics (prediction + correction):
  zhat(t) = K @ z(t-dt)                       # generative prediction
  eps     = pi * (z - zhat)                   # prediction error (drives pump)
  dz      = SL_intrinsic(z)                   # restless oscillation
          + g_gen*(zhat - z)                  # enact the generative model
          + drive*mask*(target - z)           # sensory correction (when observed)
  a_i     = a0 + beta*tanh(|eps|^2 + homeo)   # error heats the field (tanh: safe)

Learning rule (slow):
  dK_ij  = eta * eps_i * conj(z_j(t-dt)) - decay*K_ij ,   K_ii := 0

PAYOFF TEST -- occlusion / imagination:
  Periodically blank ALL sensory input for a window. A core that has internalized
  "the world is a wave moving at omega" will keep propagating the object across
  the blackout; a random/shuffled core will not. We compare the learned K against
  a phase-shuffled control on how well the field's representation tracks the
  (now-unobserved) true object during occlusion.
"""

import numpy as np

rng = np.random.default_rng(3)

# ----- field & world -------------------------------------------------------
N, dt = 64, 0.01
a0, r0, lam, beta = 0.20, 0.30, 0.4, 2.2
omega, KC, drive, pi_s = 0.6, 0.05, 2.5, 2.0   # weak KC: don't pin the bump in place
g_gen = 1.6                     # stronger enactment of the generative model
bump_w, bump_v = 5.0, 0.50      # faster object so occlusion truly tests prediction
idx = np.arange(N)
left, right = (idx - 1) % N, (idx + 1) % N

# learning hyperparameters (slow)
eta, decay = 4.0e-3, 5.0e-4      # sharper, more persistent learned structure

def world_state(t):
    centre = (8.0 + bump_v * t) % N
    d = ((idx - centre + N / 2) % N) - N / 2
    env = np.exp(-(d ** 2) / (2 * bump_w ** 2))
    target = env * np.exp(1j * (omega * t - 0.5 * d))
    mask = env > 0.25
    return centre, target, mask, np.where(mask, pi_s, 1.0)

def laplacian(z):
    return KC * (z[left] + z[right] - 2.0 * z)

def step_field(z, z_prev, K, t, observe=True, g=g_gen):
    centre, target, mask, pi = world_state(t)
    if not observe:
        mask = np.zeros(N, dtype=bool)        # OCCLUSION: nothing is observed
    zhat = K @ z_prev
    amp = np.abs(z)
    homeo = np.maximum(r0 - amp, 0.0)
    eps = pi * (z - zhat)
    e2 = np.abs(eps) ** 2 + lam * homeo ** 2
    a = a0 + beta * np.tanh(e2)
    dz = (a + 1j * omega - amp ** 2) * z + laplacian(z)
    dz = dz + g * (zhat - z) + drive * mask * (target - z)
    z_new = z + dt * dz
    return z_new, eps, centre, target

def rep_centre(z):
    """amplitude-weighted circular position of the field's representation."""
    w = np.maximum(np.abs(z) - r0, 0.0)
    ang = np.exp(1j * 2 * np.pi * idx / N)
    return (np.angle(np.sum(w * ang)) / (2 * np.pi)) % 1.0 * N

# ----- TRAIN ---------------------------------------------------------------
K = 0.05 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
np.fill_diagonal(K, 0.0)
K_init = K.copy()                    # true control: the untrained random core
z = 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
z_prev = z.copy()

TRAIN = 150000
track_curve = []
t = 0.0
for s in range(TRAIN):
    z_new, eps, centre, target = step_field(z, z_prev, K, t, observe=True)
    # slow predictive plasticity
    K += eta * np.outer(eps, np.conj(z_prev)) - decay * K
    np.fill_diagonal(K, 0.0)
    z_prev, z = z, z_new
    t += dt
    if s % 1000 == 0 and s > 5000:
        live = np.abs(target) > 0.1
        coh = np.abs(np.mean(np.exp(1j * (np.angle(z[live]) - np.angle(target[live])))))
        track_curve.append(coh)

track_curve = np.array(track_curve)
n = len(track_curve)
print("Learning the identity core K from a data stream")
print(f"N={N}, train steps={TRAIN}, eta={eta}, decay={decay}")
print("NaN check:", "FAILED" if np.any(np.isnan(K)) else "clean")
print(f"\n(1) did tracking improve as K learned?")
print(f"    phase coherence  first 20%: {track_curve[:n//5].mean():.3f}"
      f"   last 20%: {track_curve[-n//5:].mean():.3f}")
print(f"    -> K {'LEARNED to predict the world' if track_curve[-n//5:].mean() > track_curve[:n//5].mean() + 0.1 else 'did not learn'}")

# inspect what K learned: average phase advance on the dominant off-diagonal band
band = np.array([np.angle(K[i, (i+1) % N]) for i in range(N)])
print(f"    learned mean phase on i<-i+1 coupling: {np.mean(band):+.3f} rad "
      f"(omega*dt={omega*dt:+.3f}: {'matches anticipatory wave prior' if abs(np.mean(band)-omega*dt) < 0.3 else 'other structure'})")

# ----- OCCLUSION / IMAGINATION TEST ---------------------------------------
def occlusion_trial(Kuse, label, g=g_gen, blackout=1500):
    # warm up WITH observation so the field locks onto the (fast) object
    z = 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N)); zp = z.copy(); tt = 0.0
    for _ in range(8000):
        zn, _, _, _ = step_field(z, zp, Kuse, tt, observe=True, g=g); zp, z = z, zn; tt += dt
    obj0 = world_state(tt)[0]
    # OCCLUDE: no sensory; can the field carry the (now fast-moving) object?
    errs = []
    for _ in range(blackout):
        zn, _, centre, _ = step_field(z, zp, Kuse, tt, observe=False, g=g); zp, z = z, zn; tt += dt
        errs.append(abs(((rep_centre(z) - centre + N/2) % N) - N/2))
    moved = abs(((world_state(tt)[0] - obj0 + N/2) % N) - N/2)
    errs = np.array(errs)
    print(f"    {label:24s} blackout track err: {errs.mean():5.2f} nodes "
          f"(end {errs[-150:].mean():5.2f})   [object moved {moved:.0f} nodes]")
    return errs.mean()

print(f"\n(2) OCCLUSION test -- object moves ~{bump_v*15:.0f} nodes with sensory CUT OFF:")
e_learned = occlusion_trial(K,      "learned K (generative)")
e_random  = occlusion_trial(K_init, "untrained random K")
e_persist = occlusion_trial(K,      "persistence (g_gen=0)", g=0.0)
print(f"\n    null (no info / static) ~ object's travel distance.")
won = e_learned < 0.6 * e_random and e_learned < 0.6 * e_persist
print("verdict:",
      "learned core IMAGINES the moving object through the blackout "
      "(beats both random-core and persistence)"
      if won else
      "learned core does NOT yet out-predict random/persistence -- see numbers")
