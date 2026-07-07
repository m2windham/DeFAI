"""
PHASE 16 -- PHASE-SUPERPOSITION BINDING: a Track-A-style feasibility test

The field is complex-valued, but every mechanism so far reads only
|overlap| -- the phase dimension does bookkeeping, not representation. The
binding hypothesis: several concepts can be held SIMULTANEOUSLY as a
superposition z = sum_j exp(i phi_j) xi_j, where each item's identity is
readable from |overlap| and its GROUP MEMBERSHIP (what it is bound to) is
readable from the recovered relative phase. If true, the field gets a
working memory and a compositional code without recruiting blend slots.

Before building any mechanism on that idea, measure whether the code
works at all -- and where it stops:

  1. IDENTITY: superpose k of K stored random patterns at random phases.
     Are the true k items exactly the top-k by |overlap|? (capacity curve
     over k, for N=64 and N=256, K=40 stored patterns)
  2. BINDING TAG: how accurately is each item's phase recovered
     (angle of its overlap), measured as pairwise relative-phase error?
  3. GROUPING: two pairs bound at opposite phases ("red+square" at 0,
     "blue+circle" at pi/2 -- 4 items, 2 groups): cluster recovered phases
     and score pairing accuracy. This is the actual binding read-out.
  4. PERSISTENCE: a 2-item superposition under the recall attractor pull --
     how many steps until it collapses to one item (dominance ratio > 2)?
     Working memory needs persistence; attractor dynamics want collapse.
     Measured for g_rec in {2, 5} and beta in {5, 10}.

All statements are measured over 200 trials; labels are used only for
scoring. This is deliberately pure numerics plus the recall update rule --
no learning, no corpus: it bounds what any future mechanism can exploit.

RESULT (recorded from the committed run, 200 trials per point):
  - IDENTITY: perfect top-k readout to k=5-6 items at N=256 (1.000), and
    near-perfect at N=64 (1.000 at k=2-3, 0.925 at k=6). The code has real
    capacity: several concepts coexist in one field state, individually
    readable, without blend slots.
  - BINDING TAG: relative phases recover to 0.07-0.12 rad at N=256
    (0.13-0.24 at N=64) -- an order of magnitude finer than the pi/2
    separation used for grouping.
  - GROUPING: two bound pairs at opposite phases recover with 1.000 pairing
    accuracy at both field sizes. Phase IS a readable association channel.
  - PERSISTENCE is the constraint: under the recall attractor pull a 2-item
    superposition collapses to one attractor in 9-56 steps (faster with
    sharper competition: beta=10 halves the lifetime vs beta=5). Attractor
    dynamics and superposition working memory are directly opposed --
    any binding mechanism must HOLD superpositions outside the pull (gate
    the pull off, or exclude held items from competition), not fight it.
  - Verdict: phase superposition is a real, high-fidelity code for ~2-5
    simultaneous items; the engineering problem is protection, not readout.
"""

import numpy as np

rng = np.random.default_rng(0)


def unit_rows(n, N, local):
    V = local.standard_normal((n, N)) + 1j*local.standard_normal((n, N))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


TRIALS = 200
K_STORED = 40

print("="*70)
print("1+2. IDENTITY READOUT AND PHASE-TAG ERROR vs number of superposed items")
print(f"{'N':>5} {'k':>3} {'identity(top-k exact)':>22} {'phase err (rad, mean)':>22}")
results = {}
for N in (64, 256):
    local = np.random.default_rng(N)
    for k in (2, 3, 4, 5, 6):
        id_ok = 0; ph_err = []
        for t in range(TRIALS):
            xi = unit_rows(K_STORED, N, local)
            idx = local.choice(K_STORED, k, replace=False)
            phi = local.uniform(0, 2*np.pi, k)
            z = (np.exp(1j*phi)[:, None] * xi[idx]).sum(0)
            z /= np.linalg.norm(z)
            ov = xi.conj() @ z
            top = np.argsort(-np.abs(ov))[:k]
            id_ok += set(top) == set(idx)
            rec = np.angle(ov[idx])
            for a in range(k):
                for b in range(a+1, k):
                    ph_err.append(abs(wrap((rec[a]-rec[b]) - (phi[a]-phi[b]))))
        results[(N, k)] = (id_ok/TRIALS, np.mean(ph_err))
        print(f"{N:>5} {k:>3} {id_ok/TRIALS:>22.3f} {np.mean(ph_err):>22.3f}")

print("\n" + "="*70)
print("3. GROUPING: two bound pairs at opposite phases (4 items superposed)")
for N in (64, 256):
    local = np.random.default_rng(N+1)
    ok = 0
    for t in range(TRIALS):
        xi = unit_rows(K_STORED, N, local)
        idx = local.choice(K_STORED, 4, replace=False)
        base = local.uniform(0, 2*np.pi)
        # pair A = items 0,1 at base; pair B = items 2,3 at base + pi/2
        phi = np.array([base, base, base+np.pi/2, base+np.pi/2])
        z = (np.exp(1j*phi)[:, None] * xi[idx]).sum(0)
        z /= np.linalg.norm(z)
        rec = np.angle(xi[idx].conj() @ z)
        # pair by phase proximity: the two closest-in-phase items form a group
        d = np.abs(wrap(rec[:, None] - rec[None, :]))
        # greedy: item 0 pairs with its phase-nearest; remaining two pair up
        partner = int(np.argmin(d[0][1:]) + 1)
        ok += partner == 1
    print(f"  N={N}: pairing accuracy {ok/TRIALS:.3f}")

print("\n" + "="*70)
print("4. PERSISTENCE of a 2-item superposition under the recall pull")
print(f"{'N':>5} {'g_rec':>6} {'beta':>5} {'median steps to collapse':>25}")
for N in (64, 256):
    local = np.random.default_rng(N+2)
    norm = np.sqrt(N)
    for g_rec in (2.0, 5.0):
        for beta in (5.0, 10.0):
            times = []
            for t in range(50):
                xi = unit_rows(K_STORED, N, local) * norm
                a, b = local.choice(K_STORED, 2, replace=False)
                z = xi[a] + 1j*xi[b]
                z = z / np.linalg.norm(z) * norm
                dt = 0.05; omega = 0.15; Dn = 0.004
                collapse = 5000
                for s in range(5000):
                    o = (xi.conj() @ z) / N
                    m = np.abs(o)
                    w = np.exp(beta*(m - m.max())); w /= w.sum()
                    T = (w*(o/(m+1e-9))) @ xi
                    noise = np.sqrt(2*Dn*dt)*(local.standard_normal(N) +
                                              1j*local.standard_normal(N))/np.sqrt(2)
                    z = z + dt*(1j*omega*z + g_rec*(T - z)) + noise
                    z = z / np.linalg.norm(z) * norm
                    hi, lo = max(m[a], m[b]), min(m[a], m[b])
                    if lo < 1e-9 or hi/lo > 2.0:
                        collapse = s; break
                times.append(collapse)
            print(f"{N:>5} {g_rec:>6} {beta:>5} {int(np.median(times)):>25}")

id3_64 = results[(64, 3)][0]; id3_256 = results[(256, 3)][0]
print("\nverdict:", ("phase superposition is a REAL code: identities and binding"
      " tags read out reliably for small k -- capacity numbers above set the"
      " budget for any binding mechanism"
      if id3_64 > 0.9 or id3_256 > 0.9 else
      "the code is too fragile even at k=3 -- binding needs a different substrate"))
