"""
E2 VALIDATION (tier 0): direct numpy-vs-numba state equivalence on SHORT
streams, where float-reduction reordering has no room to compound into a
different discrete routing decision. The E1 regression harness remains the
authority at scale (run it with DEFAI_BACKEND=numpy and =numba); this file
exists because a line-level porting bug (a flipped branch, a wrong update
order) shows up here as a large state difference immediately, while the
harness's tolerance bands might absorb it.

Checks, one per perceive mode + recall + consolidate:
  1. plain perceive          -- phase-1 demo world, 3000 frames
  2. confirm-gated perceive  -- phase-14 stream, sigma=0.2, 400 tokens
  3. pool+amb perceive       -- phase-17/18 stack, sigma=0.2, 400 tokens
  4. recall + recall2        -- identical RNG stream by construction; the
                                committed hop sequences must agree
  5. consolidate             -- identical kept-slot list on the same state

Run: python test_fastpath_equivalence.py  (nonzero exit on failure)
"""

import copy
import sys

import numpy as np

import fastpath
from organism import Organism, normalize

if not fastpath.HAVE_NUMBA:
    print("numba not installed -- nothing to compare, skipping (exit 0)")
    sys.exit(0)

FAILURES = []


def check(name, value, tol):
    ok = value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.3e}  (tol {tol:.0e})")
    if not ok:
        FAILURES.append(name)


def pair(seed=0, **kw):
    a = Organism(backend="numpy", seed=seed, **kw)
    b = Organism(backend="numba", seed=seed, **kw)
    return a, b


def state_diff(a, b):
    """max abs difference across all perceive-visible state."""
    d = max(
        np.abs(a.xi - b.xi).max(),
        np.abs(a.z - b.z).max(),
        np.abs(a.count - b.count).max(),
        np.abs(a.P - b.P).max(),
        float((a.used != b.used).sum()),
    )
    return float(d)


# --------------------------------------------------------- shared worlds
def demo_stream(n, N=128, H=4, seed=0):
    rng = np.random.default_rng(1)
    NORM = np.sqrt(N)
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    T = np.array([[0.0, 0.8, 0.1, 0.1], [0.1, 0.0, 0.8, 0.1],
                  [0.1, 0.1, 0.0, 0.8], [0.8, 0.1, 0.1, 0.0]])
    r = np.random.default_rng(seed)
    h = 0; out = []
    for i in range(n):
        if i % 60 == 0 and i > 0:
            h = r.choice(H, p=T[h])
        out.append(G[h] + 0.5 * NORM / np.sqrt(N) *
                   (r.standard_normal(N) + 1j * r.standard_normal(N)))
    return out


print("E2 equivalence: numpy vs numba backends, short streams\n")
fastpath.warmup()

# 1. plain perceive ------------------------------------------------------
print("(1) plain perceive, 3000 frames")
a, b = pair(N=128, K=8, seed=0)
fr = demo_stream(3000)
a.perceive(fr); b.perceive(fr)
check("plain perceive state diff", state_diff(a, b), 1e-5)

# 2 & 3. confirm / pool+amb on the phase-14 world -----------------------
from phase14_noise_robust_perception import N as N14, sample_stream, frames

seq = sample_stream(400, seed=99)
fr14 = list(frames(seq, 0.2))

print("(2) confirm-gated perceive, sigma=0.2, 400 tokens")
a, b = pair(N=N14, K=60, omega=0.15, beta=10.0, seed=0)
a.perceive(fr14, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, confirm=3)
b.perceive(fr14, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, confirm=3)
check("confirm perceive state diff", state_diff(a, b), 1e-5)

print("(3) pool+amb perceive, sigma=0.2, 400 tokens")
a, b = pair(N=N14, K=60, omega=0.15, beta=10.0, seed=0)
kw = dict(g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
          active_bar=0.35, s_hat=0.2**2 * N14, probation=12000, amb=0.3)
a.perceive(fr14, **kw)
b.perceive(fr14, **kw)
check("pool+amb perceive state diff", state_diff(a, b), 1e-5)

# 4. recall / recall2 ----------------------------------------------------
print("(4) recall + recall2 on identical consolidated state")
base = Organism(backend="numpy", N=128, K=8, seed=0)
base.perceive(demo_stream(20000))
base.consolidate()
a = copy.deepcopy(base); a.backend = "numpy"
b = copy.deepcopy(base); b.backend = "numba"
sa = a.recall(8000); sb = b.recall(8000)
same = (len(sa) == len(sb)) and bool(np.all(sa == sb))
check("recall committed-sequence mismatch", 0.0 if same else 1.0, 0.0)
a = copy.deepcopy(base); a.backend = "numpy"
b = copy.deepcopy(base); b.backend = "numba"
sa = a.recall2(8000, topk=3); sb = b.recall2(8000, topk=3)
same = (len(sa) == len(sb)) and bool(np.all(sa == sb))
check("recall2 committed-sequence mismatch", 0.0 if same else 1.0, 0.0)

# 5. consolidate ---------------------------------------------------------
print("(5) consolidate kept-slot list")
a = copy.deepcopy(base)
kept_new = a.consolidate()
# reference: the pre-E2 per-pair scan, reproduced verbatim
c = copy.deepcopy(base)
keepable = c.count > 0.05 * c.count.max()
idx = list(np.where(c.used & keepable)[0])
merged = []
for k in idx:
    dup = next((j for j in merged
                if abs(c.overlaps(c.xi[k], c.xi[j:j+1])[0]) > 0.8), None)
    if dup is None:
        merged.append(k)
check("consolidate kept-list mismatch", 0.0 if merged == kept_new else 1.0, 0.0)

print()
if FAILURES:
    print(f"FAIL: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("ALL EQUIVALENCE CHECKS PASS -- backends agree on short streams.")
sys.exit(0)
