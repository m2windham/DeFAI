"""
Chaotic itinerancy in a Stuart-Landau field driven by its own prediction error.

No external judge. A localized stimulus (an "object") travels around a ring of
complex Stuart-Landau oscillators z_i = r_i e^{i theta_i}. Wherever the object
currently is, those nodes receive a weak sensory drive toward the world value.
Everything else the field does is driven by ONE internal quantity:

    prediction residual   eps_i = pi_i * (z_i - sum_j K_pred_ij z_j)
    homeostatic surprise  (r0 - |z_i|)_+      (silence is surprising -> no dark room)
    local free energy      e2_i = |eps_i|^2 + lam*(r0-|z_i|)_+^2
    amplitude pump         a_i  = a0 + beta * tanh(e2_i)      (tanh: no detonation)
    stochastic temperature D_i  = D0 * tanh(e2_i)  gated by  (settled AND wrong)

K_pred is the IDENTITY CORE: a fixed generative model that "believes" the world
is a smooth wave travelling at angular velocity omega (each node is predicted as
the omega*dt phase-advanced average of its neighbours -> anticipatory).

Claims under test:
  1. not FROZEN (dark room) and not CHAOTIC -> metastable (edge of criticality)
  2. the field's active representation TRACKS the moving object
  3. the SURPRISE hotspot sits at the LEADING edge (anticipation from omega-advance)
  4. error is INTERMITTENT (re-heat as object moves, re-lock) -> itinerancy
"""

import numpy as np

rng = np.random.default_rng(1)

# ----- field --------------------------------------------------------------
N      = 64
dt     = 0.01
STEPS  = 80000

a0     = 0.20      # ALIVE baseline (origin unstable -> no global dark room)
r0     = 0.30      # homeostatic amplitude floor: only fights true silence
lam    = 0.4       # weight of homeostatic surprise
beta   = 2.2       # pump gain
D0     = 0.6       # max stochastic temperature
omega  = 0.6       # expected angular velocity of the world
KC     = 0.18      # WEAK diffusive coupling (preserves localized structure)
drive  = 2.5       # sensory drive strength (soft, not a hard clamp)
pi_s   = 2.0       # sensory precision
bump_w = 5.0       # half-width of the moving stimulus patch
bump_v = 0.10      # stimulus speed (nodes per time unit)
vel_thresh = 0.05  # |dz/dt| below this == "settled"
err_gate   = 0.30  # e2 above this == "still wrong"

# Identity core: anticipatory travelling-wave prior on the ring.
Kpred = np.zeros((N, N), dtype=complex)
rot = np.exp(1j * omega * dt)
for i in range(N):
    Kpred[i, (i - 1) % N] = 0.5 * rot
    Kpred[i, (i + 1) % N] = 0.5 * rot
Kpred *= 0.97

# weak diffusive working coupling (nearest neighbour)
idx = np.arange(N)
left, right = (idx - 1) % N, (idx + 1) % N


def laplacian(z):
    return KC * (z[left] + z[right] - 2.0 * z)


# ----- the moving world ----------------------------------------------------
def world_state(t):
    """Localized travelling wave-packet: Gaussian envelope around a moving
    centre, carrying phase that advances at omega. Returns (centre, target z,
    sensory mask, precision)."""
    centre = (8.0 + bump_v * t) % N
    d = ((idx - centre + N / 2) % N) - N / 2          # signed circular distance
    env = np.exp(-(d ** 2) / (2 * bump_w ** 2))        # amplitude envelope
    phase = omega * t - 0.5 * d                         # wave phase across packet
    target = env * np.exp(1j * phase)
    mask = env > 0.25                                   # nodes currently "observed"
    pi = np.where(mask, pi_s, 1.0)
    return centre, target, mask, pi


# ----- integrate -----------------------------------------------------------
z = 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

L_err, L_coh, L_boil, L_hot, L_bump, L_lead = [], [], [], [], [], []
t = 0.0
for step in range(STEPS):
    centre, target, mask, pi = world_state(t)

    # prediction residual + homeostatic surprise
    zhat = Kpred @ z
    amp = np.abs(z)
    homeo = np.maximum(r0 - amp, 0.0)
    eps = pi * (z - zhat)
    e2 = np.abs(eps) ** 2 + lam * homeo ** 2

    # Stuart-Landau drift with weak coupling + soft sensory drive
    a = a0 + beta * np.tanh(e2)
    dz = (a + 1j * omega - np.abs(z) ** 2) * z + laplacian(z)
    dz = dz + drive * mask * (target - z)              # only where observed

    # "stuck and wrong" gate for stochastic escape
    speed = np.abs(dz)
    stuck = (speed < vel_thresh) & (e2 > err_gate)
    D = np.where(stuck, D0 * np.tanh(e2), 0.0)
    noise = np.sqrt(D * dt) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    z = z + dt * dz + noise
    t += dt

    if step % 100 == 0 and step > 2000:   # skip transient
        # (1) phase coherence of the field with the world wave-packet (where it lives)
        live = np.abs(target) > 0.1
        dphase = np.angle(z[live]) - np.angle(target[live])
        coh = np.abs(np.mean(np.exp(1j * dphase)))
        # (2) representation centroid (amplitude-weighted position) vs object centre
        w = np.maximum(amp - r0, 0.0)
        ang = np.exp(1j * 2 * np.pi * idx / N)
        rep = (np.angle(np.sum(w * ang)) / (2 * np.pi)) % 1.0 * N
        # (3) where is the surprise hotspot relative to object centre? (leading edge?)
        hot = idx[np.argmax(e2)]
        lead = ((hot - centre + N / 2) % N) - N / 2     # >0 = ahead of object
        L_err.append(e2.mean())
        L_coh.append(coh)
        L_boil.append(np.mean(amp > 0.6))
        L_hot.append(hot)
        L_bump.append(centre)
        L_lead.append(lead)

L_err, L_coh, L_boil = map(np.array, (L_err, L_coh, L_boil))
L_hot, L_bump, L_lead = map(np.array, (L_hot, L_bump, L_lead))

# ----- report --------------------------------------------------------------
print("Stuart-Landau chaotic itinerancy  (no external judge)")
print(f"N={N}, steps={STEPS}, dt={dt}, a0={a0}, KC={KC}")
print("NaN check:", "FAILED" if np.any(np.isnan(z)) else "clean (tanh bound held)")

print(f"\n(1) regime:")
bf = L_boil.mean()
regime = "FROZEN (dark room)" if bf < 0.05 else "CHAOTIC" if bf > 0.9 else "METASTABLE"
print(f"    boiling frac = {bf:.2f}  -> {regime}")
print(f"    error intermittency (std of mean e2) = {L_err.std():.4f}  "
      f"({'intermittent = itinerancy' if L_err.std() > 0.003 else 'static'})")

print(f"\n(2) tracking: does the field's representation follow the object?")
d_rep = np.abs(((L_hot - L_bump + N/2) % N) - N/2)
print(f"    surprise hotspot vs object centre: mean dist = {d_rep.mean():.2f} nodes "
      f"(null ~{N/4:.0f})")
print(f"    phase coherence with world wave  : {L_coh.mean():.3f} "
      f"({'field infers the moving object' if L_coh.mean() > 0.5 else 'incoherent'})")

print(f"\n(3) anticipation: is surprise at the LEADING edge of the object?")
print(f"    mean signed lead of hotspot = {L_lead.mean():+.2f} nodes "
      f"({'AHEAD (anticipatory)' if L_lead.mean() > 0.3 else 'behind/centred (reactive)'})")

ok_regime = 0.05 <= bf <= 0.9
ok_track = d_rep.mean() < 0.6 * (N/4)
ok_coh = L_coh.mean() > 0.5
print("\nverdict:", "the field tracks the world via intrinsic frustration"
      if (ok_regime and (ok_track or ok_coh))
      else "still failing -- see which test(s) above are red")
