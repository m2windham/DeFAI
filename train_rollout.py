"""
Rollout-aware learning of a generative core -- and a hunt for its failure modes.

Isolates the teacher-forcing gap found in learn_core.py. The "world" is a complex
wave-packet that translates around a ring at velocity V while its phase advances
at PHI. The one-step transition (shift by V, multiply by e^{iPHI}) is LINEAR, so a
complex matrix M CAN represent it exactly -- meaning any failure to free-run is a
LEARNING/optimization failure, not a representational one. That makes the failures
diagnostic.

We compare three predictors on long free-running rollout (the "occlusion" regime):
  - persistence            : p_{k+1} = p_k            (the strong do-nothing baseline)
  - one-step / teacher-forced M : trained on |M w_k - w_{k+1}|^2
  - rollout-trained M      : trained on its OWN H-step free rollout (BPTT)

Failure probes (the opportunities):
  (A) spectral radius of M  -> does the operator decay to zero or blow up on long
                               rollout? (teacher forcing has no incentive to fix this)
  (B) rollout error vs horizon -> where does each predictor break?
  (C) generalization -> train on speed V, test on an UNSEEN speed.
"""

import torch

torch.manual_seed(0)
cf = torch.complex64

N      = 48
SW     = 4.0      # packet half-width
Q      = 0.5      # spatial wavenumber
PHI    = 0.20     # phase advance per step
V      = 0.30     # object velocity (nodes / step)
idx = torch.arange(N)


def world_seq(k0, length, v=V, phi=PHI, c0=None):
    """A sequence of complex wave-packet states, length+0..length-1."""
    if c0 is None:
        c0 = torch.rand(1).item() * N
    ks = torch.arange(length).float()
    centre = (c0 + v * (k0 + ks)) % N                       # (length,)
    d = ((idx[None, :] - centre[:, None] + N / 2) % N) - N / 2
    env = torch.exp(-(d ** 2) / (2 * SW ** 2))
    phase = phi * (k0 + ks)[:, None] - Q * d
    return (env * torch.exp(1j * phase)).to(cf)             # (length, N)


def batch(nseq, length, v=V, phi=PHI):
    return torch.stack([world_seq(0, length, v, phi) for _ in range(nseq)])  # (B,L,N)


def apply(M, w):                # w: (...,N) -> M w
    return torch.einsum('ij,...j->...i', M, w)


def free_rollout(M, w0, H):
    """Free-running: feed the model its own output H times."""
    p = w0
    outs = []
    for _ in range(H):
        p = apply(M, p)
        outs.append(p)
    return torch.stack(outs, dim=-2)        # (...,H,N)


def train(mode, H_train, steps=4000, lr=5e-3):
    M = (0.01 * torch.randn(N, N, dtype=cf)).clone().requires_grad_(True)
    opt = torch.optim.Adam([M], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        seqs = batch(32, H_train + 1)                       # (B, H+1, N)
        if mode == "teacher":                               # one-step, all k in parallel
            pred = apply(M, seqs[:, :-1, :])                # predict each next from truth
            loss = ((pred - seqs[:, 1:, :]).abs() ** 2).mean()
        else:                                               # rollout / BPTT
            roll = free_rollout(M, seqs[:, 0, :], H_train)  # (B,H,N) from its own outputs
            loss = ((roll - seqs[:, 1:, :]).abs() ** 2).mean()
        loss.backward()
        opt.step()
    return M.detach()


def spectral_radius(M):
    return torch.linalg.eigvals(M).abs().max().item()


def eval_rollout(M, H, v=V, phi=PHI, ntrials=64):
    """Mean per-step rollout error vs persistence, over the horizon."""
    seqs = batch(ntrials, H + 1, v, phi)
    w0 = seqs[:, 0, :]
    truth = seqs[:, 1:, :]
    if M is None:                              # persistence
        roll = w0[:, None, :].expand(-1, H, -1)
    else:
        roll = free_rollout(M, w0, H)
    err = (roll - truth).abs() ** 2            # (B,H,N)
    per_step = err.mean(dim=(0, 2))            # (H,)
    return per_step


# ---------------------------------------------------------------------------
print("Rollout-aware learning of a generative core  (torch BPTT)")
print(f"N={N}, V={V}, PHI={PHI}; one-step map is LINEAR -> exactly representable\n")

H_EVAL = 60
M_tf   = train("teacher", H_train=1,  steps=4000)
M_roll = train("rollout", H_train=20, steps=4000)

print("(A) spectral radius |lambda|_max of learned operator (1.0 = energy-preserving):")
print(f"    teacher-forced M : {spectral_radius(M_tf):.3f}")
print(f"    rollout M        : {spectral_radius(M_roll):.3f}")
print("    (far from 1.0 => long rollouts decay to zero or blow up)\n")

print(f"(B) free-running rollout error vs persistence over H={H_EVAL} steps:")
e_pers = eval_rollout(None,   H_EVAL)
e_tf   = eval_rollout(M_tf,   H_EVAL)
e_roll = eval_rollout(M_roll, H_EVAL)
for name, e in [("persistence", e_pers), ("teacher-forced", e_tf), ("rollout", e_roll)]:
    print(f"    {name:15s} step5={e[4]:.4f}  step20={e[19]:.4f}  step40={e[39]:.4f}  step60={e[59]:.4f}")
beat_pers = e_roll.mean() < e_pers.mean()
beat_tf   = e_roll.mean() < e_tf.mean()
print(f"\n    rollout beats persistence? {'YES' if beat_pers else 'NO'} "
      f"(mean {e_roll.mean():.4f} vs {e_pers.mean():.4f})")
print(f"    rollout beats teacher-forced? {'YES' if beat_tf else 'NO'} "
      f"(mean {e_roll.mean():.4f} vs {e_tf.mean():.4f})")

print(f"\n(C) generalization -- train on V={V}, test on UNSEEN V=0.6:")
g_pers = eval_rollout(None,   H_EVAL, v=0.6).mean().item()
g_roll = eval_rollout(M_roll, H_EVAL, v=0.6).mean().item()
print(f"    persistence {g_pers:.4f}   rollout {g_roll:.4f}   "
      f"-> {'generalizes' if g_roll < g_pers else 'FAILS to generalize (memorized one velocity)'}")
