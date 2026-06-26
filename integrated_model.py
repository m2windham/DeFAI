"""
THE INTEGRATED OSCILLATION MODEL.

Puts the learned generative core back INSIDE the nonlinear Stuart-Landau field and
tests the hypothesis the earlier failures handed us: the SL amplitude saturation
(|z|^2 limit cycle) should keep a free-running representation SHARP, where a bare
linear operator blurs it into mush.

Pipeline:
  1. World: a localized complex wave-packet translating around a ring.
  2. Differentiable SL field with a learnable complex core K. Field step:
         zhat = K @ z_prev
         eps  = z - zhat                      (prediction error)
         a    = a0 + beta*tanh(|eps|^2)       (error heats the field)
         dz   = (a + i*omega - |z|^2) z        (SL: nonlinear amplitude saturation)
              + g*(zhat - z)                   (enact the generative model)
              + drive*mask*(target - z)        (sensory correction, when observed)
  3. HYBRID training:
       (grad) learn K by BPTT: perceive the packet (warmup, observed), then OCCLUDE
              and free-run; loss = match the (unobserved) world over the horizon.
       (local) then show a local predictive-plasticity rule, seeded at the gradient
              solution, MAINTAINS it online without backprop.
  4. HEADLINE TEST: during occlusion, roll out the SAME learned K two ways --
       (a) bare linear:  z_{k+1} = K z_k
       (b) SL field   :  full nonlinear step
     and compare SHARPNESS (inverse participation ratio) and tracking error.
"""

import torch
torch.manual_seed(0)
cf = torch.complex64

N, SW, Q, PHI, V = 40, 3.5, 0.5, 0.20, 0.30
idx = torch.arange(N)

# fixed field params
a0, omega, beta, g_gen, drive = 0.25, 0.20, 1.5, 1.2, 3.0


def world_seq(length, v=V, c0=None):
    if c0 is None:
        c0 = torch.rand(1).item() * N
    ks = torch.arange(length).float()
    centre = (c0 + v * ks) % N
    d = ((idx[None, :] - centre[:, None] + N / 2) % N) - N / 2
    env = torch.exp(-(d ** 2) / (2 * SW ** 2))
    phase = PHI * ks[:, None] - Q * d
    return (env * torch.exp(1j * phase)).to(cf), centre


def batch(B, length):
    seqs, cs = zip(*[world_seq(length) for _ in range(B)])
    return torch.stack(seqs), torch.stack(cs)        # (B,L,N), (B,L)


def ap(K, w):
    return torch.einsum('ij,...j->...i', K, w)


def field_step(z, z_prev, K, target, mask):
    zhat = ap(K, z_prev)
    eps = z - zhat
    e2 = (eps.abs() ** 2)
    a = a0 + beta * torch.tanh(e2)
    dz = (a + 1j * omega - z.abs() ** 2) * z + g_gen * (zhat - z) + drive * mask * (target - z)
    return z + 0.1 * dz, eps


def run_field(K, seqs, warmup, horizon, learn_eps=None):
    """Perceive (warmup, observed) then occlude (horizon, free-run).
    Returns the occlusion rollout (B,horizon,N). If learn_eps given, apply a LOCAL
    predictive-plasticity update to a detached copy of K (returns updated K too)."""
    B = seqs.shape[0]
    z = 0.05 * torch.randn(B, N, dtype=cf)
    z_prev = z.clone()
    Kloc = K.clone()
    full = torch.ones(B, N)
    none = torch.zeros(B, N)
    for k in range(warmup):
        z_new, eps = field_step(z, z_prev, K, seqs[:, k, :], full)
        if learn_eps is not None:
            Kloc = Kloc + learn_eps * (torch.einsum('bi,bj->ij', eps, z_prev.conj()) / B) - 1e-3 * Kloc
            Kloc.fill_diagonal_(0)
        z_prev, z = z, z_new
    outs = []
    Kroll = Kloc if learn_eps is not None else K
    for k in range(horizon):
        z_new, _ = field_step(z, z_prev, Kroll, seqs[:, warmup + k, :], none)
        outs.append(z_new); z_prev, z = z, z_new
    return torch.stack(outs, 1), Kloc


def ipr(z):
    """inverse participation ratio: higher = sharper/more localized."""
    p = z.abs() ** 2
    return (p.pow(2).sum(-1) / p.sum(-1).clamp_min(1e-8) ** 2)


# ---------------------------------------------------------------------------
WARM, HOR = 18, 22
print("Integrated oscillation model -- training core by BPTT through the SL field...\n")
K = (0.02 * torch.randn(N, N, dtype=cf)).requires_grad_(True)
opt = torch.optim.Adam([K], lr=4e-3)
for it in range(1200):
    opt.zero_grad()
    seqs, _ = batch(16, WARM + HOR)
    roll, _ = run_field(K, seqs, WARM, HOR)
    loss = (roll - seqs[:, WARM:WARM + HOR, :]).abs().pow(2).mean()
    # encourage K to not blow up
    loss = loss + 1e-4 * K.abs().pow(2).mean()
    loss.backward(); opt.step()
    with torch.no_grad():
        K.fill_diagonal_(0)
Kg = K.detach()
print(f"  done. final occlusion loss ~ {loss.item():.4f}, |lambda|_max(K)={torch.linalg.eigvals(Kg).abs().max():.3f}\n")

# ---- HEADLINE TEST: same K, bare-linear vs SL field, during occlusion ----
seqs, cs = batch(64, WARM + HOR)
true = seqs[:, WARM:WARM + HOR, :]
true_sharp = ipr(true).mean(0)

# (b) SL field rollout
sl_roll, _ = run_field(Kg, seqs, WARM, HOR)
# (a) bare linear rollout from the same perceived state
with torch.no_grad():
    z = 0.05 * torch.randn(64, N, dtype=cf); zp = z.clone(); full = torch.ones(64, N)
    for k in range(WARM):
        z, _ = field_step(z, zp, Kg, seqs[:, k, :], full); zp = z
    lin = []
    p = z
    for k in range(HOR):
        p = ap(Kg, p); lin.append(p)
    lin = torch.stack(lin, 1)

def track_err(roll):
    w = (roll.abs() - a0).clamp_min(0)
    ang = torch.exp(1j * 2 * torch.pi * idx / N)
    rep = (torch.angle((w.to(cf) * ang).sum(-1)) / (2 * torch.pi)) % 1.0 * N
    return (((rep - cs[:, WARM:WARM + HOR]) + N / 2) % N - N / 2).abs().mean(0)

print("HEADLINE: free-running rollout during occlusion (same learned K)\n")
print(f"  sharpness (IPR; higher=sharper)   step1   step10   step22 (end)")
print(f"    true world          : {true_sharp[0]:.3f}   {true_sharp[9]:.3f}    {true_sharp[-1]:.3f}")
print(f"    bare linear  K@z     : {ipr(lin).mean(0)[0]:.3f}   {ipr(lin).mean(0)[9]:.3f}    {ipr(lin).mean(0)[-1]:.3f}")
print(f"    SL field (nonlinear) : {ipr(sl_roll).mean(0)[0]:.3f}   {ipr(sl_roll).mean(0)[9]:.3f}    {ipr(sl_roll).mean(0)[-1]:.3f}")
te_lin, te_sl = track_err(lin), track_err(sl_roll)
print(f"\n  tracking error (nodes)            step1   step10   step22 (end)")
print(f"    bare linear          : {te_lin[0]:.2f}    {te_lin[9]:.2f}     {te_lin[-1]:.2f}")
print(f"    SL field             : {te_sl[0]:.2f}    {te_sl[9]:.2f}     {te_sl[-1]:.2f}")
sl_end, lin_end = ipr(sl_roll).mean(0)[-1].item(), ipr(lin).mean(0)[-1].item()
import math
sl_keeps = math.isnan(lin_end) or sl_end > 1.3 * lin_end
print(f"\n  => SL field keeps the packet {'SHARPER / STABLE' if sl_keeps else 'NOT sharper'} than bare linear "
      f"(SL {sl_end:.3f} vs linear {lin_end:.3f} at end; linear NaN = it detonated)")

# ---- HYBRID: can a LOCAL plasticity rule maintain the gradient core? ----
print("\nHYBRID: does a local predictive-plasticity rule MAINTAIN the gradient core?")
seqs, _ = batch(64, WARM + HOR)
base_loss = (run_field(Kg, seqs, WARM, HOR)[0] - seqs[:, WARM:WARM+HOR, :]).abs().pow(2).mean().item()
print(f"    gradient core occlusion loss     : {base_loss:.4f}")
for rate in (3e-3, 5e-4, 1e-4):
    roll_loc, Kloc = run_field(Kg, seqs, WARM, HOR, learn_eps=rate)
    loc_loss = (roll_loc - seqs[:, WARM:WARM+HOR, :]).abs().pow(2).mean().item()
    drift = (Kloc - Kg).abs().mean().item() / Kg.abs().mean().item()
    tag = "MAINTAINS" if loc_loss < 1.5 * base_loss else "degrades"
    print(f"    local rate {rate:.0e}: loss {loc_loss:.4f}  (drift {drift*100:4.1f}%)  -> {tag}")
