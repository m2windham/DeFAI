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
  6. slot-budget eviction    -- T1.2 evict>0 in plain (two-world flooding)
                                and pool+amb modes; the persistent budget
                                clock and eviction tallies must agree too
  7. narrowed store (T1.8)   -- perceiving with a COMPRESSED store
                                (complex64 xi). The numba kernel promotes
                                its inputs to complex128; the numpy loops
                                use self.xi directly, so without the dtype
                                guard in Organism.perceive the two backends
                                would compute the same stream at different
                                widths. This check is the guard's pin:
                                identical results, store dtype preserved,
                                and a c64 store must track a c128 one to
                                float32 precision (a tolerance, not zero --
                                the store really is quantized)

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
        np.abs(a.age - b.age).max(),
        np.abs(a.evictions - b.evictions).max(),
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

# 6. slot-budget eviction (T1.2) ----------------------------------------
print("(6) slot-budget eviction: plain two-world flooding + pool+amb")


def two_world_stream(G, lo, hi, n, seed, N):
    r = np.random.default_rng(seed)
    NORM = np.sqrt(N)
    h = lo; out = []
    for i in range(n):
        if i % 60 == 0 and i > 0:
            h = int(r.integers(lo, hi))
        out.append(G[h] + 0.5 * NORM / np.sqrt(N) *
                   (r.standard_normal(N) + 1j * r.standard_normal(N)))
    return out


Nw, Hw = 64, 4
wrng = np.random.default_rng(7)
Gw, _ = np.linalg.qr(wrng.standard_normal((Nw, 2 * Hw)) + 1j * wrng.standard_normal((Nw, 2 * Hw)))
Gw = Gw.T * np.sqrt(Nw)
sA = two_world_stream(Gw, 0, Hw, 12000, 1, Nw)
sB = two_world_stream(Gw, Hw, 2 * Hw, 12000, 2, Nw)
a, b = pair(N=Nw, K=6, seed=0)
a.perceive(sA); a.perceive(sB, evict=800)
b.perceive(sA); b.perceive(sB, evict=800)
check("plain+evict perceive state diff", state_diff(a, b), 1e-5)
check("plain+evict eviction-count mismatch",
      float(abs(a.evictions.sum() - b.evictions.sum())), 0.0)

a, b = pair(N=N14, K=12, omega=0.15, beta=10.0, seed=0)
seq6 = sample_stream(600, seed=99)
fr6 = list(frames(seq6, 0.2))
kw6 = dict(g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
           active_bar=0.35, s_hat=0.2**2 * N14, probation=12000, amb=0.3,
           evict=600)
a.perceive(fr6, **kw6)
b.perceive(fr6, **kw6)
check("pool+amb+evict perceive state diff", state_diff(a, b), 1e-5)
check("pool+amb+evict eviction-count mismatch",
      float(abs(a.evictions.sum() - b.evictions.sum())), 0.0)

# (7) T1.8: narrowed store -- compute width must stay complex128 on BOTH
# backends, and the store dtype must survive the round trip
print("(7) narrowed store (T1.8 complex64 xi): cross-backend + vs full width")
a, b = pair(N=Nw, K=6, seed=0)
a.xi = a.xi.astype(np.complex64)
b.xi = b.xi.astype(np.complex64)
a.perceive(sA); a.perceive(sB, evict=800)
b.perceive(sA); b.perceive(sB, evict=800)
check("c64-store perceive state diff (numpy vs numba)", state_diff(a, b), 1e-5)
check("c64 store dtype preserved (numpy)",
      float(a.xi.dtype != np.complex64), 0.0)
check("c64 store dtype preserved (numba)",
      float(b.xi.dtype != np.complex64), 0.0)

# same stream at full width: the quantized store must not move the routing,
# only the stored digits (float32 eps ~1.2e-7 against |xi| ~ 1)
w, _ = pair(N=Nw, K=6, seed=0)
w.perceive(sA); w.perceive(sB, evict=800)
check("c64 vs c128 store, xi drift", float(np.abs(a.xi - w.xi).max()), 1e-4)
check("c64 vs c128 store, transition-graph mismatch",
      float(np.abs(a.P - w.P).max()), 0.0)
check("c64 vs c128 store, slot-occupancy mismatch",
      float((a.used != w.used).sum()), 0.0)

# consolidation products are computed at compute width regardless of store
a.consolidate(); w.consolidate()
check("c64 store consolidate kept-list mismatch",
      float(a.kept_idx != w.kept_idx), 0.0)
check("c64 store mem at compute width",
      float(a.mem.dtype != np.complex128), 0.0)

print()
if FAILURES:
    print(f"FAIL: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("ALL EQUIVALENCE CHECKS PASS -- backends agree on short streams.")
sys.exit(0)
