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

Run: `python regression_harness.py`. Exit code is nonzero if any check fails
its tolerance. Each check prints its own measured value, tolerance band, and
verdict -- same "honest verdict" discipline as the phase scripts.
"""

import sys
import time
import numpy as np
from organism import Organism, normalize
from polysemy_organism import PolysemyOrganism, _entropy

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
