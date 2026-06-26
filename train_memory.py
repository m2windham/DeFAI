"""
Capstone: the generalization cliff is a MEMORY problem, not a capacity problem.

Probe 2 showed a one-step operator w_k -> w_{k+1} memorizes a single velocity.
Diagnosis: in this world every velocity has the SAME single-frame appearance
(velocity is independent of the spatial wavenumber Q), so velocity is a HIDDEN
state invisible to one frame. A Markov-in-one-frame predictor literally cannot
know which velocity it is in -> it must memorize one.

Cure: give the core MEMORY. A predictor that sees the last TWO frames can infer
motion (displacement between frames) and extrapolate it -- for ANY velocity --
without ever having been trained on that velocity.

We compare, across a velocity sweep none of the trained band covers:
  - persistence                 p = w_k
  - naive extrapolation         p = 2 w_k - w_{k-1}     (hardcoded, no training)
  - one-frame learned  M        (memoryless, from probe 2)
  - two-frame learned  A w_k + B w_{k-1}  (memory, rollout-trained on a velocity MIX)
"""

import torch
torch.manual_seed(0)
cf = torch.complex64

N, SW, Q, PHI = 40, 4.0, 0.5, 0.20
idx = torch.arange(N)


def world_seq(length, v, c0=None):
    if c0 is None:
        c0 = torch.rand(1).item() * N
    ks = torch.arange(length).float()
    centre = (c0 + v * ks) % N
    d = ((idx[None, :] - centre[:, None] + N / 2) % N) - N / 2
    env = torch.exp(-(d ** 2) / (2 * SW ** 2))
    phase = PHI * ks[:, None] - Q * d
    return (env * torch.exp(1j * phase)).to(cf)


def batch(nseq, length, vmean, vjit):
    vs = vmean + vjit * (torch.rand(nseq) - 0.5) * 2
    return torch.stack([world_seq(length, v.item()) for v in vs])


def ap(M, w):
    return torch.einsum('ij,...j->...i', M, w)


# ---- one-frame model (memoryless) ----
def train_one(vmean, vjit, H=15, steps=2200, lr=5e-3):
    M = (0.01 * torch.randn(N, N, dtype=cf)).requires_grad_(True)
    opt = torch.optim.Adam([M], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        s = batch(32, H + 1, vmean, vjit)
        p = s[:, 0, :]; loss = 0.0
        for k in range(H):
            p = ap(M, p); loss = loss + ((p - s[:, k + 1, :]).abs() ** 2).mean()
        (loss / H).backward(); opt.step()
    return M.detach()


def roll_one(M, s, H):
    p = s[:, 0, :]; outs = []
    for _ in range(H):
        p = ap(M, p); outs.append(p)
    return torch.stack(outs, 1)


# ---- two-frame model (memory) ----
def train_two(vmean, vjit, H=15, steps=2200, lr=5e-3):
    A = (0.01 * torch.randn(N, N, dtype=cf)).requires_grad_(True)
    B = (0.01 * torch.randn(N, N, dtype=cf)).requires_grad_(True)
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        s = batch(32, H + 2, vmean, vjit)
        wm1, w0 = s[:, 0, :], s[:, 1, :]; loss = 0.0
        for k in range(H):
            nxt = ap(A, w0) + ap(B, wm1)
            loss = loss + ((nxt - s[:, k + 2, :]).abs() ** 2).mean()
            wm1, w0 = w0, nxt
        (loss / H).backward(); opt.step()
    return A.detach(), B.detach()


def roll_two(AB, s, H):
    A, B = AB
    wm1, w0 = s[:, 0, :], s[:, 1, :]; outs = []
    for _ in range(H):
        nxt = ap(A, w0) + ap(B, wm1); outs.append(nxt); wm1, w0 = w0, nxt
    return torch.stack(outs, 1)


def ev(kind, model, v, H=40, nt=96):
    s = batch(nt, H + 2, v, 0.0)
    if kind == "pers":
        roll = s[:, 1, :][:, None, :].expand(-1, H, -1); truth = s[:, 2:2 + H, :]
    elif kind == "extrap":
        wm1, w0 = s[:, 0, :], s[:, 1, :]; outs = []
        for _ in range(H):
            nxt = 2 * w0 - wm1; outs.append(nxt); wm1, w0 = w0, nxt
        roll = torch.stack(outs, 1); truth = s[:, 2:2 + H, :]
    elif kind == "one":
        roll = roll_one(model, s[:, 1:, :], H); truth = s[:, 2:2 + H, :]
    else:
        roll = roll_two(model, s, H); truth = s[:, 2:2 + H, :]
    return ((roll - truth).abs() ** 2).mean().item()


print("Capstone: memory cures the velocity generalization cliff\n")
print("Training one-frame and two-frame cores on a velocity MIX (V in 0.05..0.55)...\n")
M1 = train_one(0.30, 0.25)
M2 = train_two(0.30, 0.25)

vs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
rows = {
    "persistence ": [ev("pers",   None, v) for v in vs],
    "extrapolate ": [ev("extrap", None, v) for v in vs],
    "one-frame M ": [ev("one",    M1,   v) for v in vs],
    "two-frame AB": [ev("two",    M2,   v) for v in vs],
}
print("    test V :   " + "  ".join(f"{v:.2f}" for v in vs))
for name, r in rows.items():
    print(f"    {name}:   " + "  ".join(f"{e:.2f}" for e in r))

print("\n  note V=0.60 is OUTSIDE the training band (0.05..0.55) -- the real test.")
oob_one = rows["one-frame M "][-1]; oob_two = rows["two-frame AB"][-1]; oob_p = rows["persistence "][-1]
print(f"  out-of-band V=0.60: persistence={oob_p:.2f}  one-frame={oob_one:.2f}  two-frame={oob_two:.2f}")
print("  verdict:", "MEMORY generalizes past the trained band; memoryless does not"
      if oob_two < 0.6 * oob_one and oob_two < 0.6 * oob_p else
      "two-frame did not clearly win out-of-band -- inspect numbers")
