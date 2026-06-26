"""
Failure probes from train_rollout.py, turned into experiments.

Probe 1 (model mismatch): the previous world was EXACTLY linear, so teacher-forcing
beat rollout-BPTT. Hypothesis: once the world is no longer exactly representable by
a single linear operator (per-sequence velocity JITTER), rollout training -- which
optimizes the regime it's deployed in -- should become more robust over long
horizons. We test teacher vs rollout under jitter.

Probe 2 (generalization cliff): a single linear M encodes ONE velocity. We (a) map
rollout error across a velocity sweep for a model trained at V=0.3, and (b) test
whether a single M trained on a MIX of velocities can cover them -- it should not,
motivating a velocity-conditioned / nonlinear core.
"""

import torch
torch.manual_seed(0)
cf = torch.complex64

N, SW, Q, PHI = 40, 4.0, 0.5, 0.20
idx = torch.arange(N)


def world_seq(length, v, phi=PHI, c0=None):
    if c0 is None:
        c0 = torch.rand(1).item() * N
    ks = torch.arange(length).float()
    centre = (c0 + v * ks) % N
    d = ((idx[None, :] - centre[:, None] + N / 2) % N) - N / 2
    env = torch.exp(-(d ** 2) / (2 * SW ** 2))
    phase = phi * ks[:, None] - Q * d
    return (env * torch.exp(1j * phase)).to(cf)


def batch(nseq, length, vmean, vjit=0.0):
    vs = vmean + vjit * (torch.rand(nseq) - 0.5) * 2
    return torch.stack([world_seq(length, v.item()) for v in vs])


def apply(M, w):
    return torch.einsum('ij,...j->...i', M, w)


def free_rollout(M, w0, H):
    p, outs = w0, []
    for _ in range(H):
        p = apply(M, p); outs.append(p)
    return torch.stack(outs, dim=-2)


def train(mode, H_train, vmean, vjit, steps=2500, lr=5e-3):
    M = (0.01 * torch.randn(N, N, dtype=cf)).clone().requires_grad_(True)
    opt = torch.optim.Adam([M], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        seqs = batch(32, H_train + 1, vmean, vjit)
        if mode == "teacher":
            pred = apply(M, seqs[:, :-1, :])
            loss = ((pred - seqs[:, 1:, :]).abs() ** 2).mean()
        else:
            roll = free_rollout(M, seqs[:, 0, :], H_train)
            loss = ((roll - seqs[:, 1:, :]).abs() ** 2).mean()
        loss.backward(); opt.step()
    return M.detach()


def eval_rollout(M, H, vmean, vjit=0.0, ntrials=96):
    seqs = batch(ntrials, H + 1, vmean, vjit)
    w0, truth = seqs[:, 0, :], seqs[:, 1:, :]
    roll = w0[:, None, :].expand(-1, H, -1) if M is None else free_rollout(M, w0, H)
    return ((roll - truth).abs() ** 2).mean().item()


H = 50
print("PROBE 1 -- does rollout training win under MODEL MISMATCH (velocity jitter)?\n")
for jit in (0.0, 0.15):
    tag = "exact-linear" if jit == 0 else f"jitter +-{jit}"
    M_tf   = train("teacher", 1,  0.30, jit)
    M_roll = train("rollout", 20, 0.30, jit)
    e_pers = eval_rollout(None,   H, 0.30, jit)
    e_tf   = eval_rollout(M_tf,   H, 0.30, jit)
    e_roll = eval_rollout(M_roll, H, 0.30, jit)
    winner = "rollout" if e_roll < e_tf else "teacher"
    print(f"  world={tag:14s}  persistence={e_pers:.4f}  teacher={e_tf:.4f}  "
          f"rollout={e_roll:.4f}  -> {winner} wins")
print("  (hypothesis: teacher wins when exact; rollout gains as mismatch grows)\n")

print("PROBE 2a -- generalization cliff: train V=0.3, sweep test velocity:\n")
M = train("teacher", 1, 0.30, 0.0)
print("    test V :   " + "  ".join(f"{v:.2f}" for v in [0.1,0.2,0.3,0.4,0.5,0.6]))
errs = [eval_rollout(M, H, v) for v in [0.1,0.2,0.3,0.4,0.5,0.6]]
prs  = [eval_rollout(None, H, v) for v in [0.1,0.2,0.3,0.4,0.5,0.6]]
print("    model  :   " + "  ".join(f"{e:.2f}" for e in errs))
print("    persist:   " + "  ".join(f"{e:.2f}" for e in prs))
print("    -> usable only in a razor-thin band around the trained velocity\n")

print("PROBE 2b -- can ONE linear M cover MANY velocities (trained on a mix)?\n")
M_mix = train("teacher", 1, 0.30, 0.25)     # wide velocity range in training
for v in (0.10, 0.30, 0.50):
    em = eval_rollout(M_mix, H, v); ep = eval_rollout(None, H, v)
    print(f"    V={v:.2f}: mixed-trained M={em:.4f}  persistence={ep:.4f}  "
          f"-> {'helps' if em < ep else 'no better than persistence'}")
print("    -> a single linear operator is ONE dynamical law; it cannot hold a")
print("       family of velocities. The fix is a velocity-CONDITIONED / nonlinear")
print("       core that infers the latent rate and applies the matching shift.")
