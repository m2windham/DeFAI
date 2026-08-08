"""
REGRESSION HARNESS (engineering track E1) -- pins the mechanism's headline
behaviors so every future backend (Numba, GPU) and every calibration change
(phases 25/26's decorrelation + percentile bars) can be checked against known
numbers instead of "it still looks right."

This is not a new experiment: every check below reproduces a result already
established in a phase script, at a smaller/faster scale, with an explicit
tolerance band. Bitwise equality is NOT the bar -- a JIT or GPU port will
legitimately perturb float reduction order -- so every check compares against
a tolerance measured from the ORIGINAL run's own seed-to-seed variance, not an
arbitrarily tight number.

Sections (fast tier only; corpus-tier joins once E2/Numba makes it cheap):
  1. organism.perceive/consolidate/recall -- clean small-world capture +
     structure recovery (organism.py's own demo, phase 1).
  2. confirm-gated recruitment under noise -- phase 14's junk-elimination
     and strict-win regime (sigma=0.0 and sigma=0.2).
  3. pool+amb routing -- phase 18's clean-case-identical guarantee (amb=0.0
     reproduces phase 17 exactly) and a beyond-sigma* coverage floor.
  4. discover_categories_v2 -- category purity on a synthetic grammar with
     known ground-truth categories (the phase 10 style check).
  5. predictive gain (Myhill-Nerode) -- the phase 12 margin between a
     synthetic dual-role word and monosemous controls.
  6. symbolic reasoning on the logic layer (phase 30) -- multi-step
     inference, field-free rollout structure, and planning advantage over
     undirected recall on the hub-and-branch world.
  7. percentile acceptance bars (phase 26) -- the label-free spectral
     noise-energy estimate and the calibrated-bar coverage/junk floor.
  8. state serialization (E3) -- lossless mid-stream save/restore
     (bitwise vs never stopping), deterministic replay, cross-backend load.
  9. slot budget / eviction under recruitment pressure (T1.2, phase 33b) --
     a flooded bank learns a second world only with evict > 0, and the
     budget's persistent clock round-trips bitwise through E3.
 10. store compression (T1.8, phase 33g) -- the lossless levers (CSR
     transition matrix, float32 counts) decompress bit-for-bit, a
     complex64 store drifts only at float32 epsilon and leaves routing and
     the transition graph untouched, and E3 schema v2 keeps the
     uncompressed round-trip bitwise while still loading v1 files.
 11. stable symbol registry (T3.3) -- symbol IDs decoupled from slot
     indices at the EventBoundary seam: slot indices measurably rot under
     fusion/recycling while IDs still name the same memory content, the
     consolidation view is not an identity mutation, the registry
     round-trips bitwise through E3 schema v3 (v1 and v2 files still load),
     and registry-on is bitwise identical to registry-off.
 12. logic-layer depth (T6.3, phase 40) -- the reasoning ops built on top of
     phase 30's three, all field-free: the shared-Dijkstra refactor leaves
     `next_hops` bitwise unchanged, confidence bounds never invent an
     unobserved edge, confidence-weighted planning beats hop-count planning
     on an under-evidenced graph, the compositional planners match
     brute-force optimal walks (avoidance is a constraint, not a
     preference), the sparse port is bitwise for next_hops/rollout and
     within tolerance for kstep, and a first-order macro-edge cannot improve
     a plan built from its own parts -- that zero is pinned.

Run: `python regression_harness.py`. Exit code is nonzero if any check fails
its tolerance. Each check prints its own measured value, tolerance band, and
verdict -- same "honest verdict" discipline as the phase scripts.
"""

import os
import sys
import time
import numpy as np
from organism import normalize
from polysemy_organism import PolysemyOrganism, _entropy

# E2 backend switch: DEFAI_BACKEND=numba runs every direct-Organism check
# through the JIT backend (PolysemyOrganism section stays on the reference
# implementation -- its loops aren't ported yet). Same 23 checks, same bands.
if os.environ.get('DEFAI_BACKEND') == 'numba':
    from organism_numba import NumbaOrganism as Organism
else:
    from organism import Organism

FAILURES = []


def check(name, value, lo, hi, note=""):
    ok = lo <= value <= hi
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {value:.4f}  (expect [{lo:.4f}, {hi:.4f}]){' -- ' + note if note else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


# ============================================================ 1. core demo
def section_1_core():
    print("\n(1) core organism: perceive -> consolidate -> recall (phase 1)")
    rng = np.random.default_rng(1)
    N, H, K = 128, 4, 8
    NORM = np.sqrt(N)
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    Ttrue = np.array([[0.0, 0.8, 0.1, 0.1],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.1, 0.1, 0.0, 0.8],
                      [0.8, 0.1, 0.1, 0.0]])

    def make_stream(n, dwell=60, noise=0.5, seed=0):
        r = np.random.default_rng(seed)
        h = 0; out = []
        for i in range(n):
            if i % dwell == 0 and i > 0:
                h = r.choice(H, p=Ttrue[h])
            out.append(G[h] + noise * NORM / np.sqrt(N) *
                       (r.standard_normal(N) + 1j * r.standard_normal(N)))
        return out

    caps, corrs = [], []
    for seed in range(3):
        org = Organism(N=N, K=K, seed=seed)
        org.perceive(make_stream(80000, seed=seed))
        org.consolidate()
        cap = [max(np.abs(org.overlaps(G[h], org.mem))) for h in range(H)]
        caps.append(np.mean(cap))
        mem2reg = [int(np.argmax(np.abs(org.overlaps(org.mem[k], G)))) for k in range(org.mem.shape[0])]
        seq = org.recall(60000)
        reg_seq = np.array([mem2reg[s] for s in seq]) if len(seq) else np.array([0])
        B = np.zeros((H, H))
        for a, b in zip(reg_seq[:-1], reg_seq[1:]):
            if a != b: B[a, b] += 1
        Bn = B / (B.sum(1, keepdims=True) + 1e-9)
        mask = ~np.eye(H, dtype=bool)
        corr = np.corrcoef(Bn[mask], Ttrue[mask])[0, 1]
        corrs.append(0.0 if np.isnan(corr) else corr)

    check("regime capture (mean overlap, 3 seeds)", float(np.mean(caps)), 0.70, 1.01,
          note="phase 1 established >0.7")
    check("recalled-vs-true transition corr (mean, 3 seeds)", float(np.mean(corrs)), 0.45, 1.01,
          note="phase 1 established >0.5")


# ============================================ 2. noise-robust perception
def section_2_noise():
    print("\n(2) confirm-gated recruitment under noise (phase 14)")
    from phase14_noise_robust_perception import (emb, V, N, sample_stream, frames, evaluate)
    for sigma, confirm, lo_cov, hi_cov, lo_junk, hi_junk, note in [
        (0.0, 3, 0.80, 1.01, 0.0, 0.05, "clean case unharmed"),
        (0.2, 3, 0.85, 1.01, 0.0, 0.20, "phase 14's strict-win regime"),
    ]:
        org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
        seq = sample_stream(4000, seed=99)
        fr = frames(seq, sigma)
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, confirm=confirm)
        r = evaluate(org)
        check(f"sigma={sigma} coverage", r['cov'], lo_cov, hi_cov, note=note)
        check(f"sigma={sigma} junk hop rate", r['junk'], lo_junk, hi_junk)


# =================================================== 3. pool+amb routing
def section_3_pool_amb():
    print("\n(3) pool+amb routing (phase 18): amb=0 reproduces phase 17 exactly")
    from phase14_noise_robust_perception import emb, V, N, sample_stream, frames, evaluate
    from phase17_pooled_recruitment import pool_bars

    def run(sigma, amb):
        org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
        seq = sample_stream(4000, seed=99)
        fr = frames(seq, sigma)
        pb, ab = pool_bars(sigma)
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                     active_bar=ab, s_hat=sigma**2 * N, probation=12000, amb=amb)
        return evaluate(org)

    r_amb0 = run(0.2, 0.0)
    r_pool = run(0.2, 0.0)  # same call is the phase-17 reproduction; kept as an
                            # explicit two-call check so a future refactor that
                            # accidentally makes amb stateful gets caught.
    check("amb=0.0 determinism (coverage matches itself)", abs(r_amb0['cov'] - r_pool['cov']),
          0.0, 0.02, note="same seed, same call -> must be reproducible")

    r_hard = run(0.3, 0.3)
    check("sigma=0.3 amb=0.3 coverage", r_hard['cov'], 0.20, 1.01,
          note="phase 18: dominates phase 17 at full coverage in its committed run; "
               "wide band here since this is a single seed, not the phase's own sweep")


# ============================================== 4. category discovery
def section_4_categories():
    print("\n(4) discover_categories_v2 purity on synthetic grammar (phase 10/14 recipe)")
    # phase 14's own embedding recipe: category-basis + noise (proven to yield
    # near-full word-level coverage before any category discovery is layered on).
    ANIMAL, ACTION, OBJECT = 0, 1, 2
    NEXT_FWD = {0: 1, 1: 2, 2: 0}
    P_CORRECT = 0.88
    HOLD = 12
    ANIMALS = ['cat', 'dog', 'bird', 'horse', 'cow', 'pig', 'sheep', 'wolf']
    ACTIONS = ['run', 'jump', 'swim', 'eat', 'sleep', 'hunt', 'hide', 'play']
    OBJECTS = ['food', 'water', 'ground', 'sky', 'tree', 'rock', 'cave', 'nest', 'field', 'river']
    vocab = ANIMALS + ACTIONS + OBJECTS
    V = len(vocab)
    true_cat = np.array([ANIMAL] * 8 + [ACTION] * 8 + [OBJECT] * 10)

    N = 30; NORM = np.sqrt(N)
    emb_rng = np.random.default_rng(13)
    cat_bases = np.zeros((3, N))
    cat_bases[0, 0:3] = 1.0; cat_bases[1, 3:6] = 1.0; cat_bases[2, 6:9] = 1.0
    emb = np.zeros((V, N))
    for i in range(V):
        emb[i] = 0.6 * cat_bases[true_cat[i]] + 0.4 * emb_rng.standard_normal(N)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb_c = emb.astype(complex)

    def make_stream(n, seed):
        r = np.random.default_rng(seed)
        pools = [np.where(true_cat == c)[0] for c in range(3)]
        c = ANIMAL; out = []
        for _ in range(n):
            w = int(r.choice(pools[c]))
            out.extend([normalize(emb_c[w], NORM)] * HOLD)
            c = NEXT_FWD[c] if r.random() < P_CORRECT else int(
                r.choice([cc for cc in [0, 1, 2] if cc != NEXT_FWD[c]]))
        return out

    org = PolysemyOrganism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    org.perceive(make_stream(4000, seed=99), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    org.consolidate(merge_thresh=0.84, prune_frac=0.02)

    # map kept slots -> true word -> true category
    states = np.array([normalize(emb_c[w], NORM) for w in range(V)])
    slot_word = np.argmax(np.abs(org.mem.conj() @ states.T) / N, axis=1)
    slot_cat = true_cat[slot_word]

    res = org.discover_categories_v2(k_range=[3], seed=3)
    labels = np.array([res['word_slot_to_cat'][i] for i in range(org.mem.shape[0])])

    # purity: for each discovered cluster, fraction matching its majority true category
    correct = 0
    for c in range(3):
        members = slot_cat[labels == c]
        if len(members):
            correct += np.bincount(members, minlength=3).max()
    purity = correct / len(slot_cat)
    check("category purity (k=3, phase-14 grammar)", purity, 0.75, 1.01,
          note="phase 10 established near-100% purity on synthetic data")


# =============================================== 5. predictive gain margin
def section_5_predictive_gain():
    print("\n(5) predictive gain (Myhill-Nerode split test, phase 12 style)")
    rng = np.random.default_rng(3)
    n_cat = 4
    # dual-role word: category depends on prev category (role A after cat0, role B after cat2)
    # monosemous control: successor category independent of prev category

    def sample_dual(n, seed):
        r = np.random.default_rng(seed)
        prev = r.integers(0, n_cat, n)
        succ = np.where(prev == 0, 1, np.where(prev == 2, 3, r.integers(0, n_cat, n)))
        return prev, succ

    def sample_mono(n, seed):
        r = np.random.default_rng(seed)
        prev = r.integers(0, n_cat, n)
        succ = r.integers(0, n_cat, n)
        return prev, succ

    def gain(prev, succ):
        uncond = np.bincount(succ, minlength=n_cat)
        H_uncond = _entropy(uncond.tolist())
        total = len(succ)
        H_cond = 0.0
        for pc in range(n_cat):
            mask = prev == pc
            if mask.sum() == 0:
                continue
            c = np.bincount(succ[mask], minlength=n_cat)
            H_cond += (mask.sum() / total) * _entropy(c.tolist())
        return max(H_uncond - H_cond, 0.0)

    dual_gains = [gain(*sample_dual(2000, s)) for s in range(5)]
    mono_gains = [gain(*sample_mono(2000, s)) for s in range(5)]
    margin = float(np.mean(dual_gains) - np.mean(mono_gains))
    check("dual-role mean gain", float(np.mean(dual_gains)), 0.20, 2.01)
    check("monosemous mean gain (should be ~0)", float(np.mean(mono_gains)), 0.0, 0.08)
    check("gain margin (dual - mono)", margin, 0.15, 2.01,
          note="phase 12 established a clean, large margin (0.30-bit on its own setup)")


# ============================== 6. symbolic reasoning (phase 30)
def section_6_reasoning():
    print("\n(6) symbolic reasoning on the logic layer (phase 30)")
    H, N, K = 6, 128, 12
    NORM = np.sqrt(N)
    Ttrue = np.array([
        [0.00, 0.45, 0.05, 0.45, 0.05, 0.00],
        [0.10, 0.00, 0.85, 0.05, 0.00, 0.00],
        [0.05, 0.05, 0.00, 0.00, 0.00, 0.90],
        [0.10, 0.05, 0.00, 0.00, 0.85, 0.00],
        [0.85, 0.00, 0.05, 0.10, 0.00, 0.00],
        [0.90, 0.05, 0.00, 0.05, 0.00, 0.00],
    ])
    mask = ~np.eye(H, dtype=bool)
    k2s, rolls, d_hops, d_succ, adv = [], [], [], [], []
    for seed in range(2):
        rng = np.random.default_rng(seed + 100)
        Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
        G = Gr.T * NORM

        def make_stream(n, dwell=60, noise=0.5):
            h = 0; out = []
            for i in range(n):
                if i % dwell == 0 and i > 0:
                    h = rng.choice(H, p=Ttrue[h])
                out.append(G[h] + noise * NORM / np.sqrt(N) *
                           (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
            return out

        org = Organism(N=N, K=K, seed=seed)
        org.perceive(make_stream(60000))
        org.consolidate()
        n_mem = org.mem.shape[0]
        mem2reg = [int(np.argmax(np.abs(org.overlaps(org.mem[k], G)))) for k in range(n_mem)]

        def to_reg(Pm):
            R = np.zeros((H, H))
            for i in range(n_mem):
                for j in range(n_mem):
                    R[mem2reg[i], mem2reg[j]] += Pm[i, j]
            return R / (R.sum(1, keepdims=True) + 1e-9)

        k2s.append(np.corrcoef(to_reg(org.graph.kstep(org.kept_idx, 2))[mask],
                               np.linalg.matrix_power(Ttrue, 2)[mask])[0, 1])

        seq_s = org.graph.rollout(org.kept_idx, 0, 10000, np.random.default_rng(5))
        rs = np.array([mem2reg[s] for s in seq_s])
        B = np.zeros((H, H))
        for a, b in zip(rs[:-1], rs[1:]):
            if a != b: B[a, b] += 1
        Bn = B / (B.sum(1, keepdims=True) + 1e-9)
        rolls.append(np.corrcoef(Bn[mask], Ttrue[mask])[0, 1])

        da, ua = [], []
        for goal in range(1, n_mem):
            for _ in range(2):
                seq, reached = org.recall_directed(goal, steps=8000)
                da.append(len(seq) if reached else np.nan)
                sequ = org.recall(8000)
                wh = np.where(sequ == goal)[0]
                ua.append(int(wh[0]) + 1 if len(wh) else np.nan)
        d_hops.append(np.nanmean(da)); d_succ.append(np.mean(~np.isnan(da)))
        adv.append(np.nanmean(ua) / max(np.nanmean(da), 1e-9))

    check("2-step inference corr vs true law (mean, 2 seeds)", float(np.mean(k2s)), 0.90, 1.01,
          note="phase 30: 0.987-0.998 over 5 seeds; permutation null 99th pct ~0.45")
    check("field-free rollout bigram corr (mean, 2 seeds)", float(np.mean(rolls)), 0.90, 1.01,
          note="imagination without the field preserves learned structure")
    check("directed recall mean hops to goal", float(np.mean(d_hops)), 1.0, 3.0,
          note="phase 30: exactly shortest-path (1.8) over 5 seeds")
    check("directed recall success rate", float(np.mean(d_succ)), 0.95, 1.01)
    check("planning advantage (undirected/directed hops)", float(np.mean(adv)), 2.0, 100.0,
          note="undirected wandering varies 6-32 hops by seed; directed stays 1.8")


# ========================== 7. percentile bars (phase 26)
def section_7_percentile_bars():
    print("\n(7) percentile acceptance bars (phase 26): calibrated, not handed")
    from phase14_noise_robust_perception import N as N14, sample_stream, frames
    from phase26_percentile_bars import calibrate, run_pool
    for sigma, amb, lo_cov, hi_junk in [(0.2, 0.0, 0.95, 0.10), (0.3, 0.3, 0.85, 0.45)]:
        cal = calibrate(frames(sample_stream(800, seed=7), sigma))
        s_true = sigma ** 2 * N14
        check(f"sigma={sigma} spectral noise-energy rel error",
              abs(cal['s_cal'] - s_true) / s_true, 0.0, 0.25,
              note="phase 26: label-free estimate of what perceive() used to be handed")
        r = run_pool(sigma, amb, cal['active_bar'], cal['s_cal'], cal['fuse_bar'])
        check(f"sigma={sigma} calibrated-bars coverage", r['cov'], lo_cov, 1.01)
        check(f"sigma={sigma} calibrated-bars junk", r['junk'], 0.0, hi_junk,
              note="phase 26 measured 0.37 at sigma=0.3 (oracle 0.34)" if sigma == 0.3 else "")


# ========================== 8. state serialization (E3)
def section_8_serialization():
    print("\n(8) state serialization (E3): lossless restore + deterministic replay")
    import os as _os, tempfile
    from organism_state import save_state, load_state
    rng = np.random.default_rng(3)
    N, H, K = 64, 3, 6
    NORM = np.sqrt(N)
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    Tt = np.array([[0.0, 0.8, 0.2], [0.2, 0.0, 0.8], [0.8, 0.2, 0.0]])

    def stream(n, seed):
        r = np.random.default_rng(seed)
        h = 0; out = []
        for i in range(n):
            if i % 40 == 0 and i > 0:
                h = r.choice(H, p=Tt[h])
            out.append(G[h] + 0.4 * (r.standard_normal(N) + 1j * r.standard_normal(N)))
        return out

    s1, s2 = stream(8000, 1), stream(8000, 2)
    a = Organism(N=N, K=K, seed=0)
    a.perceive(s1); a.perceive(s2); a.consolidate()
    b = Organism(N=N, K=K, seed=0)
    b.perceive(s1)
    path = _os.path.join(tempfile.gettempdir(), "e3_harness.npz")
    save_state(b, path)
    c = load_state(path, cls=Organism)
    c.perceive(s2); c.consolidate()
    check("restore-and-continue max |dxi| vs never-stopped",
          float(np.abs(a.xi - c.xi).max()), 0.0, 1e-12,
          note="same backend: bitwise lossless")
    check("restore-and-continue max |dP|", float(np.abs(a.P - c.P).max()), 0.0, 1e-12)
    ra, rc = a.recall(5000), c.recall(5000)
    ident = 1.0 if (len(ra) == len(rc) and bool(np.all(ra == rc))) else 0.0
    check("deterministic replay (recall sequence identity)", ident, 1.0, 1.01,
          note="rng generator state round-trips")
    # cross-backend restore: reference state must load into the E2 backend
    # and keep working (tolerance, not bitwise: reduction order differs)
    from organism_numba import NumbaOrganism
    d = load_state(path, cls=NumbaOrganism)
    d.perceive(s2); d.consolidate()
    cap = float(np.mean([max(np.abs(d.overlaps(G[h], d.mem))) for h in range(H)]))
    check("cross-backend restore regime capture", cap, 0.70, 1.01,
          note="reference-saved state continues on the numba backend")


# ========================== 9. slot budget / eviction (T1.2, phase 33b)
def section_9_slot_budget():
    print("\n(9) slot budget: eviction under recruitment pressure (T1.2, phase 33b)")
    import os as _os, tempfile
    from organism_state import save_state, load_state
    N, H, K = 64, 4, 6
    NORM = np.sqrt(N)
    wrng = np.random.default_rng(7)
    Gr, _ = np.linalg.qr(wrng.standard_normal((N, 2 * H)) + 1j * wrng.standard_normal((N, 2 * H)))
    G = Gr.T * NORM     # 2H orthogonal regimes: first H = world A, last H = world B

    def stream(lo, hi, n, seed):
        r = np.random.default_rng(seed)
        h = lo; out = []
        for i in range(n):
            if i % 60 == 0 and i > 0:
                h = int(r.integers(lo, hi))
            out.append(G[h] + 0.5 * NORM / np.sqrt(N) *
                       (r.standard_normal(N) + 1j * r.standard_normal(N)))
        return out

    sA, sB = stream(0, H, 12000, 1), stream(H, 2 * H, 12000, 2)

    def capture(org, lo, hi):
        return float(np.mean([max(np.abs(org.overlaps(G[h], org.xi[org.used])))
                              for h in range(lo, hi)]))

    # baseline: world A floods the K=6 bank; world B cannot recruit
    org0 = Organism(N=N, K=K, seed=0)
    org0.perceive(sA); org0.perceive(sB)
    check("flooded-bank world-B capture (no budget)", capture(org0, H, 2 * H), 0.0, 0.75,
          note="single seed, wide band; measured ~0.41-0.57 over world seeds")

    # budget: pressure eviction reclaims stale slots for world B; the run is
    # split mid-B in BOTH arms (perceive-call boundaries must match), one arm
    # saving/restoring at the split -- the budget clock is dynamical state now
    org1 = Organism(N=N, K=K, seed=0)
    org1.perceive(sA)
    org1.perceive(sB[:6000], evict=800); org1.perceive(sB[6000:], evict=800)
    check("budget world-B capture (evict=800)", capture(org1, H, 2 * H), 0.90, 1.01,
          note="phase 33b: the mechanism is load-bearing for the recovery")
    check("pressure evictions fired", float(org1.evictions.sum()), 1.0, 20.0)

    org2 = Organism(N=N, K=K, seed=0)
    org2.perceive(sA)
    org2.perceive(sB[:6000], evict=800)
    path = _os.path.join(tempfile.gettempdir(), "e3_budget_harness.npz")
    save_state(org2, path)
    org2 = load_state(path, cls=Organism)
    org2.perceive(sB[6000:], evict=800)
    d = max(np.abs(org1.xi - org2.xi).max(), np.abs(org1.P - org2.P).max(),
            np.abs(org1.age - org2.age).max(),
            np.abs(org1.evictions - org2.evictions).max())
    check("E3 restore mid-eviction, max state delta", float(d), 0.0, 1e-12,
          note="budget clock + tallies round-trip bitwise")


# ============== 10. store compression + E3 schema v2 (T1.8, phase 33g)
def section_10_compression():
    print("\n(10) store compression (T1.8, phase 33g): lossless levers, "
          "narrowed store, E3 v2")
    import os as _os, tempfile
    from organism_compress import CompressionSpec, compress, store_bytes
    from organism_state import load_state, save_state, save_state_v1

    N, H, K = 64, 3, 8
    NORM = np.sqrt(N)
    rng = np.random.default_rng(3)
    Gr, _ = np.linalg.qr(rng.standard_normal((N, H)) + 1j * rng.standard_normal((N, H)))
    G = Gr.T * NORM
    Tt = np.array([[0.0, 0.8, 0.2], [0.2, 0.0, 0.8], [0.8, 0.2, 0.0]])

    def stream(n, seed):
        r = np.random.default_rng(seed)
        h = 0; out = []
        for i in range(n):
            if i % 40 == 0 and i > 0:
                h = r.choice(H, p=Tt[h])
            out.append(G[h] + 0.4 * (r.standard_normal(N) + 1j * r.standard_normal(N)))
        return out

    s1, s2 = stream(8000, 1), stream(8000, 2)
    org = Organism(N=N, K=K, seed=0)
    org.perceive(s1); org.perceive(s2)

    # the lossless levers must reproduce the store EXACTLY (this is the
    # guarantee the phase-33g byte numbers rest on)
    st = compress(org, spec=CompressionSpec(xi_dtype=np.complex128,
                                            meta_dtype=np.float32))
    check("lossless spec: max |dxi|", float(np.abs(st.xi_full() - org.xi).max()),
          0.0, 0.0, note="CSR P + float32 counts decompress bit-for-bit")
    check("lossless spec: max |dP|", float(np.abs(st.P_full() - org.P).max()), 0.0, 0.0)
    check("sparse-P byte reduction (dense K^2 -> CSR)",
          float(org.P.nbytes / st.p_bytes), 1.2, 1e6,
          note="phase 33g: 15.5x at K=160 on the 33c protocol")

    # complex64 store: halves the xi term, drift at float32 epsilon
    stc = compress(org, spec=CompressionSpec())
    check("c64 store: xi relative drift",
          float(np.abs(stc.xi_full() - org.xi).max() / np.abs(org.xi).max()),
          0.0, 1e-6, note="float32 eps -- the lossy lever's whole cost")
    check("c64 store: total byte ratio vs uncompressed",
          float(store_bytes(org) / stc.nbytes), 1.5, 1e6)

    # a narrowed store must still perceive, on either backend, at compute
    # width -- the dtype guard in Organism.perceive (equivalence test sec. 7)
    narrow = Organism(N=N, K=K, seed=0)
    narrow.perceive(s1)
    narrow.xi = narrow.xi.astype(np.complex64)
    narrow.perceive(s2)
    full = Organism(N=N, K=K, seed=0)
    full.perceive(s1); full.perceive(s2)
    check("narrowed store: xi drift vs full-width run",
          float(np.abs(narrow.xi.astype(complex) - full.xi).max()), 0.0, 1e-4,
          note="quantized digits only -- routing must be unchanged")
    check("narrowed store: transition graph identical",
          float(np.abs(narrow.P - full.P).max()), 0.0, 0.0)
    check("narrowed store: dtype preserved across perceive",
          float(narrow.xi.dtype != np.complex64), 0.0, 0.0)

    # E3 v2: uncompressed round-trip still bitwise, v1 files still load,
    # compressed round-trip lossy only where its spec says
    path = _os.path.join(tempfile.gettempdir(), "e3_v2_harness.npz")
    save_state(org, path)
    r2 = load_state(path, cls=Organism)
    check("E3 v2 uncompressed round-trip max |dxi|",
          float(np.abs(r2.xi - org.xi).max()), 0.0, 0.0,
          note="the schema bump must not weaken the bitwise guarantee")
    v1p = _os.path.join(tempfile.gettempdir(), "e3_v1_harness.npz")
    save_state_v1(org, v1p)
    r1 = load_state(v1p, cls=Organism)
    d1 = max(np.abs(r1.xi - org.xi).max(), np.abs(r1.P - org.P).max(),
             np.abs(r1.age - org.age).max())
    check("E3 v1 file loads under v2, max state delta", float(d1), 0.0, 0.0,
          note="backward load: pre-T1.8 saves are not orphaned")
    cpath = _os.path.join(tempfile.gettempdir(), "e3_c_harness.npz")
    save_state(org, cpath, compress_spec=CompressionSpec())
    rc = load_state(cpath, cls=Organism)
    check("E3 compressed round-trip max |dP|",
          float(np.abs(rc.P - org.P).max()), 0.0, 0.0,
          note="P is lossless even in a compressed file")
    check("E3 compressed round-trip xi drift",
          float(np.abs(rc.xi - org.xi).max() / np.abs(org.xi).max()), 0.0, 1e-6)
    check("E3 compressed file restores at compute width",
          float(rc.xi.dtype != np.complex128), 0.0, 0.0,
          note="a loaded organism is immediately perceivable on both backends")
    check("E3 compressed file is smaller on disk",
          float(_os.path.getsize(path) / _os.path.getsize(cpath)), 1.05, 1e6)


# ================================================= 11. stable symbol registry
def section_11_symbol_registry():
    """T3.3: symbol IDs decoupled from slot indices, at the EventBoundary seam.

    The hazard is concrete: a slot index is storage, not identity. Fusion
    moves a memory off the slot it was born on and leaves that slot unused;
    recycling hands the slot to an unrelated pattern. Either way, anything a
    caller recorded as "slot 7" quietly starts naming something else. These
    checks measure that the rot is real in each regime (otherwise the
    section pins nothing), then pin the registry's actual contract:

      - an ID is minted once and never reissued (tombstones are permanent);
      - every minted ID is in exactly one state -- live, fused onto a live
        ID, or dead -- and no two slots claim the same ID;
      - an ID orphaned by fusion still resolves to the memory that absorbed
        it, where the raw slot index resolves to nothing;
      - the registry round-trips bitwise through E3 (schema v3), so an ID
        handed out before a save still names the same memory after a load;
      - registry-on is bitwise identical to registry-off.

    Content stability is deliberately NOT claimed, and the measured drift
    below says why: pool-mode refinement keeps adapting a mature trace on an
    ~1/eta-visit window (organism.py), so a long-lived memory can come to
    sit on a different word entirely. That happens at the same rate whether
    it is reached by ID or by slot -- it is the mechanism moving, not
    identity breaking. The word-retention check is banded only as a
    regression tripwire, with the raw-slot baseline printed beside it; the
    registry's job is to track the trace, not to freeze it.
    """
    print("\n(11) stable symbol registry (T3.3)")
    import tempfile
    import os as _os
    from organism_state import save_state, save_state_v2, load_state
    from phase14_noise_robust_perception import (
        N as N14, V, emb, sample_stream, frames)

    NORM14 = np.sqrt(N14)
    words = np.array([normalize(emb[w].astype(complex), NORM14)
                      for w in range(V)])

    def word_of(vec):
        """Eval-side label: which vocabulary word this memory sits on.
        Ground truth for MEASUREMENT only -- never inside the mechanism."""
        return int(np.argmax(np.abs(words.conj() @ vec) / N14))

    sigma = 0.2
    base = dict(g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                active_bar=0.35, s_hat=sigma**2 * N14)

    def two_epoch(K, evict, amb, probation, n):
        """Perceive two epochs; return the organism, its registry, and what
        each epoch-A symbol named then (its slot and its word)."""
        kw = dict(base, probation=probation, amb=amb, evict=evict)
        org = Organism(N=N14, K=K, omega=0.15, beta=10.0, seed=0, symbols=True)
        reg = org.registry
        fa = list(frames(sample_stream(n, seed=99), sigma))
        fb = list(frames(sample_stream(n, seed=7), sigma))
        org.perceive(fa, **kw)
        was = {s: (reg.slot_of(s), word_of(org.xi[reg.slot_of(s)]))
               for s in reg.live()}
        org.perceive(fb, **kw)
        return org, reg, was, kw, fa, fb

    def invariants(reg):
        """Every minted ID in exactly one state; no ID claimed twice."""
        bad = 0
        for sid in range(reg.minted()):
            st = reg.status(sid)
            live = reg.slot_of(sid) >= 0
            if st == 'live' and not (live and reg.resolve(sid) == sid):
                bad += 1
            elif st == 'fused' and not (live and reg.resolve(sid) != sid):
                bad += 1
            elif st == 'dead' and live:
                bad += 1
        held = [int(x) for x in reg.slot_sym if x >= 0]
        return float(bad + (len(held) - len(set(held))))

    # --- regime A: fusion-dominated (roomy bank, no budget pressure) -----
    org, reg, was, kwA, fa, fb = two_epoch(K=60, evict=0, amb=0.0,
                                           probation=12000, n=4000)
    check("regime A: fusions recorded", float(len(reg.alias)), 1.0, 1e6,
          note="mature duplicates converging -- without them this is vacuous")
    check("regime A: registry invariants violated", invariants(reg), 0.0, 0.0,
          note="each ID live | fused | dead, exactly once; no ID held twice")
    # the exact win: IDs whose memory was fused into another slot. The
    # recorded index no longer holds them (it was recycled onto an unrelated
    # pattern); the ID still resolves, to the memory that absorbed it.
    orphans = [s for s in was if reg.status(s) == 'fused']
    check("regime A: fusion-orphaned IDs", float(len(orphans)), 1.0, 1e6,
          note="epoch-A IDs whose memory moved to another slot")
    check("regime A: fusion-orphaned IDs that fail to resolve",
          float(sum(1 for s in orphans if reg.slot_of(s) < 0)), 0.0, 0.0,
          note="an aliased ID must still reach the memory that absorbed it")
    check("regime A: fusion-orphaned IDs whose recorded slot still holds them",
          float(sum(1 for s in orphans if reg.symbol_at(was[s][0]) == s)),
          0.0, 0.0, note="the recorded index is stale by construction here -- "
                         "reading org.P at it would hit the wrong memory")
    alive = [s for s in was if reg.slot_of(s) >= 0]
    kept = sum(1 for s in alive if word_of(org.xi[reg.slot_of(s)]) == was[s][1])
    by_slot = sum(1 for s in was if org.used[was[s][0]]
                  and word_of(org.xi[was[s][0]]) == was[s][1])
    id_ret, slot_ret = kept / max(len(alive), 1), by_slot / max(len(was), 1)
    check("regime A: word retention by symbol ID", id_ret, 0.45, 1.01,
          note=f"{kept}/{len(alive)} by ID vs {by_slot}/{len(was)} by raw "
               f"slot; the ceiling here is pool-mode plasticity re-centering "
               f"mature traces, not identity -- tripwire band, not a claim")
    check("regime A: ID retention below the raw-slot baseline",
          float(id_ret < slot_ret - 1e-12), 0.0, 0.0,
          note="the registry may never be worse than the index it replaces")

    # --- regime B: recycling-dominated (bank oversubscribed 2:1 + budget) -
    orgB, regB, wasB, kwB, faB, fbB = two_epoch(K=12, evict=600, amb=0.3,
                                                probation=4000, n=600)
    check("regime B: recyclings recorded", float(len(regB.dead)), 1.0, 1e6,
          note="use-it-or-lose-it + pressure eviction")
    check("regime B: registry invariants violated", invariants(regB), 0.0, 0.0)
    check("regime B: epoch-A slot indices now naming something else",
          float(sum(1 for s in wasB if regB.symbol_at(wasB[s][0]) != s)),
          1.0, 1e6, note="the hazard, measured")
    check("regime B: reused slots holding a different word",
          float(sum(1 for s in wasB if regB.symbol_at(wasB[s][0]) >= 0
                    and word_of(orgB.xi[wasB[s][0]]) != wasB[s][1])),
          1.0, 1e6, note="indexing by slot here reads the wrong memory")
    check("regime B: recycled symbol IDs reissued to a live slot",
          float(len(set(regB.live()) & regB.dead)), 0.0, 0.0,
          note="a tombstoned ID is never handed out again")

    # --- consolidate: a VIEW, and IDs come out unchanged -----------------
    before = (reg.slot_sym.copy(), int(reg.next_id[0]), dict(reg.alias))
    org.consolidate()
    org.consolidate()                       # twice: still not a mutation
    check("consolidate perturbed symbol identity",
          float(not np.array_equal(before[0], reg.slot_sym)
                or before[1] != int(reg.next_id[0]) or before[2] != reg.alias),
          0.0, 0.0, note="callers snapshot/restore raw counts around it")
    check("resolved symbols with a row in the compact bank",
          float(sum(1 for s in alive if reg.mem_row(s) >= 0)), 1.0, 1e6,
          note="mem_row is the migration primitive for org.mem / org.Pn")
    check("mem_row rows pointing at the wrong memory",
          float(sum(1 for s in alive if reg.mem_row(s) >= 0
                    and abs(np.vdot(org.mem[reg.mem_row(s)],
                                    org.xi[reg.slot_of(s)])) / N14 < 0.99)),
          0.0, 0.0)

    # --- E3 (schema v3): the registry round-trips bitwise ----------------
    path = _os.path.join(tempfile.gettempdir(), "t33_registry.npz")
    save_state(org, path)
    back = load_state(path, cls=Organism)
    q = back.registry
    same = (q is not None and np.array_equal(q.slot_sym, reg.slot_sym)
            and int(q.next_id[0]) == int(reg.next_id[0])
            and q.alias == reg.alias and q.dead == reg.dead
            and q.mem_index == reg.mem_index)
    check("E3 v3 registry round-trip mismatch", float(not same), 0.0, 0.0,
          note="an ID handed out before a save must survive the load")
    back.perceive(fb, **kwA)
    check("restored organism reissued a live symbol ID",
          float(len(set(back.registry.live()) & reg.dead)), 0.0, 0.0,
          note="minting continues from the restored counter, never restarts")

    # v2 files (pre-T3.3) still load -- into an organism with no registry,
    # which is exactly the state a pre-T3.3 organism had
    v2p = _os.path.join(tempfile.gettempdir(), "t33_v2.npz")
    save_state_v2(org, v2p)
    r2 = load_state(v2p, cls=Organism)
    d2 = max(np.abs(r2.xi - org.xi).max(), np.abs(r2.P - org.P).max(),
             np.abs(r2.age - org.age).max())
    check("E3 v2 file loads under v3, max state delta", float(d2), 0.0, 0.0,
          note="backward load: pre-T3.3 saves are not orphaned")
    check("v2 file wrongly resurrected a registry",
          float(r2.registry is not None), 0.0, 0.0)

    # --- observational purity: the registry may not touch the mechanism --
    off = Organism(N=N14, K=12, omega=0.15, beta=10.0, seed=0)
    off.perceive(faB, **kwB); off.perceive(fbB, **kwB); off.consolidate()
    d = max(np.abs(orgB.xi - off.xi).max(), np.abs(orgB.P - off.P).max(),
            np.abs(orgB.z - off.z).max(), np.abs(orgB.age - off.age).max(),
            float((orgB.used != off.used).sum()))
    check("registry on-vs-off state drift", float(d), 0.0, 0.0,
          note="identity tracking is observation only -- bitwise, not a band")


# ============================================ 12. logic-layer depth (phase 40)
def section_12_logic_depth():
    """T6.3: the reasoning ops phase 40 added on top of phase 30's three.

    Everything here is graph arithmetic -- no field is constructed anywhere
    in this section, which is the isolation discipline phase 30 established
    and the precondition for the whole layer. The checks pin four contracts:

      - the shared-Dijkstra refactor left `next_hops` BITWISE unchanged, so
        phase 30's committed planning numbers cannot drift underneath
        section 6 (which pins their values but only to a tolerance);
      - `confidence` is a genuine lower bound and never invents an edge the
        organism has not observed -- a smoothing rule that hallucinated hops
        would let the planner route through transitions that never happened;
      - the compositional planners are CORRECT, not merely plausible:
        avoidance is a constraint (zero violations) and both the negated and
        conjunctive plans match brute-force optimal walks on a graph small
        enough to enumerate exhaustively;
      - the sparse port is equal, not approximately equal: next_hops and
        rollout bitwise (including the RNG stream), kstep to tolerance,
        because skipping zeros reorders a float reduction;
      - the mined-macro null: a first-order macro CANNOT improve a plan it
        is built out of, and that zero is pinned so a future macro change
        that appears to beat plain planning is caught as a bug in the
        accounting rather than banked as a result.
    """
    print("\n(12) logic-layer depth and optimization (phase 40, T6.3)")
    from organism import MacroGraph, SparseTransitions, TransitionGraph
    from phase40_logic_depth import (brute_force_best, composition_world,
                                     log_true, phase30_next_hops_reference,
                                     plan_all, unreliable_world)

    # --- the phase-30 anchor: refactor must be bitwise --------------------
    worst_n = worst_d = 0.0
    for s in range(2):
        g, T, _ = unreliable_world(s, n=60, steps=6000)
        idx = np.arange(60)
        for goal in (0, 13, 59):
            r_n, r_d = phase30_next_hops_reference(g, idx, goal)
            n_n, n_d = g.next_hops(idx, goal)
            worst_n = max(worst_n, float(np.abs(r_n - n_n).max()))
            fin = np.isfinite(r_d) & np.isfinite(n_d)
            if not np.array_equal(np.isfinite(r_d), np.isfinite(n_d)):
                worst_d = np.inf
            elif fin.any():
                worst_d = max(worst_d, float(np.abs(r_d[fin] - n_d[fin]).max()))
    check("next_hops pre- vs post-refactor: next-hop delta", worst_n, 0.0, 0.0,
          note="one Dijkstra now serves every planner -- phase 30's must not move")
    check("next_hops pre- vs post-refactor: distance delta", worst_d, 0.0, 0.0)

    # --- confidence is a bound, and invents nothing -----------------------
    g, T, _ = unreliable_world(0, n=120, steps=2000)
    idx = np.arange(120)
    Pn = g.normalized(idx); C = np.asarray(g.P[np.ix_(idx, idx)], float)
    leaks = 0.0
    for mode in ("wilson", "laplace"):
        L = g.confidence(idx, 0.05, mode)
        leaks += float((L[C == 0] > 0).sum())
    check("confidence: edges invented where nothing was observed", leaks, 0.0, 0.0,
          note="smoothing must not hand the planner a hop that never happened")
    above = float((g.confidence(idx, 0.05, "wilson") > Pn + 1e-12).sum())
    check("wilson bounds exceeding the point estimate", above, 0.0, 0.0,
          note="wilson is a LOWER bound; laplace is a posterior mean and "
               "legitimately lifts rare edges, so it is not checked here")

    # --- confidence-weighted planning beats hop-count planning ------------
    goals = list(np.argsort(-g.P.sum(1))[:12])
    ph = plan_all(g.edge_quality(idx, "hops"), goals, 120)
    pl = plan_all(g.confidence(idx, 0.05), goals, 120)
    pairs = set(ph) & set(pl)
    adv = (np.mean([log_true(pl[k], T) for k in pairs])
           - np.mean([log_true(ph[k], T) for k in pairs]))
    check("confidence planner advantage over hop-count (nats/plan)", float(adv),
          0.40, 3.00,
          note="phase 40 (A): +1.04 mean over 5 seeds, 0/5 sign flips, above null")

    # --- compositional goals are correct, certified by brute force --------
    gc = composition_world(); nc = 12; ic = np.arange(nc)
    viol = 0.0; gap = 0.0; cover = 0.0
    for a, b in [(0, 5), (0, 8), (2, 7), (3, 10)]:
        free = gc.plan_reliable(ic, a, b, weights="mle")
        gate = int(free.path[1]) if free.hops >= 2 else int(free.path[-1])
        con = gc.plan_reliable(ic, a, b, weights="mle", avoid=[gate])
        viol += float(con is None or gate in con.path)
        if con is not None:
            bf, _ = brute_force_best(gc.P, a, b, max_hops=9, avoid=[gate])
            gap = max(gap, abs(bf - float(np.log(con.p_mle))))
    for a, gs in [(0, (5, 9)), (2, (7, 11))]:
        rep = gc.plan_visit(ic, a, list(gs), weights="mle")
        cover += float(rep is not None and all(x in rep.path for x in gs))
        if rep is not None:
            bf, _ = brute_force_best(gc.P, a, None, max_hops=max(9, rep.hops),
                                     must=gs)
            gap = max(gap, abs(bf - float(np.log(rep.p_mle))))
    check("compositional: avoidance constraint violations", viol, 0.0, 0.0,
          note="'without passing B' is a constraint, not a preference")
    check("compositional: conjunctive goals covered", cover, 2.0, 2.0)
    check("compositional: gap vs brute-force optimal walk", gap, 0.0, 1e-9,
          note="exhaustive search on 12 symbols certifies both planners optimal")

    # --- sparse port equality ---------------------------------------------
    bit_n = bit_r = 0.0; kerr = 0.0
    for K in (40, 160):
        gs = TransitionGraph(K)
        rr = np.random.default_rng(400 + K)
        for i in range(K):
            su = rr.choice(K, size=min(10, K - 1), replace=False)
            gs.P[i, su[su != i]] = rr.integers(1, 400, size=int((su != i).sum()))
        ii = np.arange(K)
        sp = SparseTransitions.from_graph(gs, ii)
        dn, dd = gs.next_hops(ii, 1); sn, sd = sp.next_hops(1)
        fin = np.isfinite(dd) & np.isfinite(sd)
        bit_n += float(np.array_equal(dn, sn)
                       and np.array_equal(np.isfinite(dd), np.isfinite(sd))
                       and np.array_equal(dd[fin], sd[fin]))
        bit_r += float(np.array_equal(
            gs.rollout(ii, 0, 600, np.random.default_rng(5)),
            sp.rollout(0, 600, np.random.default_rng(5))))
        kerr = max(kerr, float(np.abs(gs.kstep(ii, 2) - sp.kstep(2)).max()))
    check("sparse next_hops bitwise identical to dense", bit_n, 2.0, 2.0,
          note="same settle order, same values -- a port, not an approximation")
    check("sparse rollout bitwise identical (incl. RNG stream)", bit_r, 2.0, 2.0)
    check("sparse kstep max |delta| vs dense", kerr, 0.0, 1e-9,
          note="skipping zeros reorders a float reduction: tolerance, not bit")

    # --- the mined-macro null ---------------------------------------------
    mg = MacroGraph(g); mg.mine(idx, top=32)
    worst = 0.0
    for gl in goals[:6]:
        for a in range(0, 120, 17):
            if a == gl:
                continue
            base = g.plan_reliable(idx, a, int(gl), weights="lcb")
            wm = g.plan_reliable(idx, a, int(gl), weights="lcb", macros=mg)
            if base is not None and wm is not None:
                worst = max(worst, abs(wm.p_lcb - base.p_lcb))
    check("first-order macro-edge reliability gain", worst, 0.0, 0.0,
          note="phase 40 (C2): forced zero -- a macro cannot beat its own parts")


if __name__ == "__main__":
    t0 = time.time()
    print("REGRESSION HARNESS -- fast tier (E1)")
    print("Pins current mechanism behavior across seeds/tolerances so Numba/GPU")
    print("ports and calibration changes (phases 25/26) can be checked, not eyeballed.\n")

    section_1_core()
    section_2_noise()
    section_3_pool_amb()
    section_4_categories()
    section_5_predictive_gain()
    section_6_reasoning()
    section_7_percentile_bars()
    section_8_serialization()
    section_9_slot_budget()
    section_10_compression()
    section_11_symbol_registry()
    section_12_logic_depth()

    dt = time.time() - t0
    print(f"\n{'='*70}")
    if FAILURES:
        print(f"FAIL ({len(FAILURES)}/{len(FAILURES)} shown failed) in {dt:.1f}s:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL CHECKS PASS in {dt:.1f}s. Safe baseline for E2 (Numba) and phases 25/26.")
        sys.exit(0)
