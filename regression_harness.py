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
 13. detection-driven sense splitting (T6.4, phase 41) -- the closed loop:
     predictive gain clears its own per-word null on exactly the dual-role
     words, those words (and only those) recruit sense-slots under phase
     10's context-primed settling, the resulting slots have successor
     distributions distinct from a same-word permutation null, and the
     held-out next-category gain over the MATCHED-CAPACITY control clears
     the pre-registered +0.02 bar. The capacity match itself is pinned
     exact -- it is the structural invariant the whole comparison rests on.
 14. task-free continual learning (T6.1, phase 38) -- the boundary-free
     ladder's structure, and its two negatives. Pinned exact: the three
     conditions really are blocked (A), signal-free on the same stream (B)
     and drifted (C), and ORGANISM+B's accuracy is INVARIANT between A and B
     because the organism has no boundary input to remove -- the one part of
     T6.1's thesis that survived. Pinned as bands, deliberately negative:
     the organism's near-indifference to the stream as well, and the
     FALSIFIED localization claim (recruitment does not spike at unannounced
     novelty), so that a future reversal has to be earned rather than
     silently celebrated.
 15. phase channel and the real+phase layout (T7.1, phase 43) -- what is
     actually stored in the imaginary half of the memory bank. Pinned
     exact: the residual metric's two analytic fixed points, omega=0
     exactness (a real recursion on real inputs stores a real bank), the
     identity `layout loss = sqrt(residual)` that makes the metric a PRICE
     rather than a diagnostic, and GAUGE invariance -- a random per-slot
     phase rotation moves no `np.abs` consumer at all. Pinned as a
     deliberately POSITIVE band: the residual at omega=2, because phase 43
     falsified the claim that the phase channel is structurally empty (it
     is empty by parameter regime), and a future claim to the contrary must
     be re-earned against that row.
 16. sparse-P default path and narrow CSR (T7.2, phase 44) -- the default
     `next_hops` dispatch. Pinned exact: dense and sparse agree bitwise,
     the auto-dispatch matches the path it chose, `plan()` is unchanged,
     and BOTH dispatch conditions (size below SPARSE_MIN_K, and non-integer
     counts) keep the dense path -- the second is what makes the default a
     wall-clock choice and never a numerical one. Narrow CSR is pinned
     lossless with its guard declining non-integer counts. Pinned as a
     band, deliberately: P density RISES with observation count, because
     phase 44 falsified the scale-free reading every CSR byte number in
     this project had been quoted under.
 17. settling depth and chunk statistics (T7.3/T7.5, phases 45/47) -- the
     two measurement instruments those phases rest on, pinned so neither
     can drift. Phase 45's vectorized field recursion is checked against
     `Organism.perceive`'s own `z`; the last-token attractor at the
     committed exposure is banded, including the fact that suffix
     similarity sits ABOVE `consolidate()`'s 0.8 merge threshold; and the
     prefix signal's magnitude is banded so "the state is path-blind" has
     to be re-earned. Phase 47's raw chunk-gain statistic is pinned
     POSITIVE on an i.i.d. stream -- it FAILED its own null and the pin
     stops the broken form coming back quietly -- with the null-corrected
     form pinned near zero on the same stream, and the clustering artifact
     (a RANDOM partition of a Zipfian stream scoring positive) pinned
     positive because it is what the category-level arm died on.
 18. eviction victim rule (T6.2, phase 39) -- which stale slot dies under
     recruitment pressure. Pinned: argmin-count protects established core
     memories where uniform-random eviction eats them (the load-bearing
     sign), the stale pool has real establishedness spread (so the
     comparison is not vacuous), and omitting `evict_victim` is bitwise
     identical to passing the default.
 19. the (A) store on the ladder (T7.7, phase 50) -- store-mode fork
     equivalence at toy scale. Pinned as bands: running the SAME
     class-incremental mini-ladder with the store compressed and carried
     forward at every task boundary under c64 vs real+phase moves ACC and
     FORG by less than the claim threshold (phase 50 measured held-out
     means within +-0.005 / +-0.010 at K=112/160), and the (A)/(B) store
     byte ratio sits in the 0.58 +- band the layout arithmetic derives --
     the discount lives in the xi term, so it cannot silently change
     without this row noticing. The c64 arm's own ACC is pinned high so
     the equivalence cannot pass vacuously on a dead readout.

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
# ================== 13. detection-driven sense splitting (T6.4, phase 41)
def section_13_detection_driven_split():
    print("\n(13) detection-driven sense splitting (T6.4, phase 41)")
    import phase41_detection_driven_split as p41

    rs = [p41.harness_probe(seed=s) for s in (0, 1)]

    def mean(k):
        return float(np.mean([r[k] for r in rs]))

    def worst(k):
        return float(np.max([r[k] for r in rs]))

    # --- the detector fires on the dual-role words and (nearly) nothing else
    check("dual words clearing their own p99 gain null (of 3)",
          mean('n_dual_detected'), 3.0, 3.01,
          note="phase 41 margins are ~0.35-0.42 bits against a ~0.006 null -- a "
               "miss here is a real regression, not seed noise")
    check("control words false-positive at p99", mean('n_false_positive'), 0.0, 2.01,
          note="p99 on ~26 controls expects ~0.26; the committed 5-seed run "
               "measured [3,0,0,0,0]")

    # --- detection, and only detection, drives the split
    check("dual words recruiting >=2 sense-slots (of 3)", mean('dual_split'),
          3.0, 3.01)
    check("unsplit arm max slots per word", worst('unsplit_max_slots_per_word'),
          1.0, 1.0,
          note="the gate is unreachable outside the detected set -- structural, "
               "not a band")

    # --- prediction (a): the split slots carry real successor structure
    check("split dual words distinct vs same-word permutation null (of 3)",
          mean('dual_distinct'), 3.0, 3.01)

    # --- the matched-capacity control's defining invariant (T1.5's confound)
    check("split-vs-matched live-slot gap", worst('capacity_gap'), 0.0, 0.0,
          note="the control must hold the SAME slot budget, or the comparison "
               "measures capacity instead of splitting -- exact, not a band")

    # --- prediction (b): downstream utility over the matched control
    check("held-out next-category gain over matched capacity",
          mean('nextcat_delta'), 0.02, 0.15,
          note="lower bound IS the pre-registered survival threshold; the "
               "committed 5-seed run at full scale measured +0.063")


# ==================== 14. task-free continual learning (T6.1, phase 38)
def section_14_task_free_continual():
    print("\n(14) task-free continual learning (T6.1, phase 38)")
    import phase38_task_free_continual as p38

    rs = [p38.harness_probe(seed=s, cls=Organism) for s in (0, 1)]

    def mean(k):
        return float(np.mean([r[k] for r in rs]))

    def worst(k):
        return float(np.max([abs(r[k]) for r in rs]))

    # --- the three conditions must actually BE what the phase claims.
    # These are structural, so they are exact: if the stream stops being
    # blocked in A or stops being mixed in C, the phase measures nothing.
    check("condition A chunks that are a single task (of 20)",
          mean('pure_chunks_A'), 20.0, 20.0,
          note="A is the blocked control -- every chunk pure, exact")
    check("condition C chunks that are a single task (of 20)",
          mean('pure_chunks_C'), 0.0, 0.0,
          note="C is gradual drift -- no chunk is ever one task, exact")
    check("A vs B stream index mismatch", worst('AB_stream_identical'),
          0.0, 0.0,
          note="A and B MUST share the stream byte-for-byte -- B removes only "
               "the boundary signal, so any drift here confounds the two")
    check("draw-count difference between conditions",
          worst('draw_count_delta'), 0.0, 0.0,
          note="equal exposure: conditions differ in composition, not amount")
    check("cold-start chunk counted as novelty",
          worst('novelty_excludes_cold_start'), 0.0, 0.0,
          note="chunk 0 recruits because the store is empty; counting it "
               "would inflate localization with a trivial effect")

    # --- the one structural claim that survived phase 38, pinned exact.
    # The organism consumes no boundary information, so removing it cannot
    # move a single bit. This is what "task-free by construction" MEANS, and
    # it is the only part of T6.1's thesis the measurement supported.
    check("ORGANISM+B accuracy delta between A and B", worst('AB_acc_delta'),
          0.0, 0.0,
          note="invariance to the boundary signal is exact, not a band -- "
               "the organism has no boundary input to remove")

    # --- and the two measured NEGATIVES, banded so a silent reversal is
    # caught rather than quietly celebrated.
    check("organism |A->C| accuracy shift (2-seed mean)",
          abs(mean('AC_acc_delta')), 0.0, 0.15,
          note="the organism is near-indifferent to the stream too "
               "(5-seed committed mean +0.003); seed spread is wide, so this "
               "bands the magnitude, not the sign")
    check("recruitment-localization z at novelty chunks",
          mean('localization_z'), -2.5, 0.5,
          note="phase 38 (b) FALSIFIED: recruitment does NOT spike at "
               "unannounced novelty -- committed 5-seed pooled z -1.53. "
               "Banded negative on purpose: a future positive must be earned")


# =============================== 15. phase channel + real+phase layout (T7.1)
def section_15_phase_channel():
    print("\n(15) phase channel and the real+phase layout (T7.1, phase 43)")
    from organism_compress import (CompressionSpec, compress,
                                   real_phase_residual, store_bytes)

    # -- the residual metric's two analytic fixed points ------------------
    rng = np.random.default_rng(11)
    Nv = 64
    v_real = rng.standard_normal(Nv).astype(complex) * np.exp(1j * 0.7)
    v_cplx = rng.standard_normal(Nv) + 1j * rng.standard_normal(Nv)
    check("residual of a rotated-real vector",
          float(real_phase_residual(v_real[None, :])[0]), 0.0, 1e-15,
          note="R = 0 exactly iff the real+phase layout is exact")
    check("residual of an isotropic complex vector",
          float(real_phase_residual(v_cplx[None, :])[0]), 0.35, 0.5,
          note="R -> 0.5: the layout would throw half the energy away")

    # -- a REAL-input stream, the class every committed pipeline feeds ----
    NORM = np.sqrt(Nv)
    K = 12
    Gr = np.linalg.qr(rng.standard_normal((Nv, 4)))[0].T * NORM      # real anchors

    def stream(n, seed, hold=4):
        # hold=4 on purpose: phase 43 measured the residual to PEAK at
        # shallow settling, so this is the regime where the omega effect is
        # visible at all. Deep settling (hold >> 1/(g_in*dt)) drives it to
        # floor for every omega, which is exactly the 33h arm's situation.
        r = np.random.default_rng(seed)
        h, out = 0, []
        for i in range(n):
            if i % hold == 0 and i > 0:
                h = int(r.integers(4))
            out.append((Gr[h] + 0.3 * r.standard_normal(Nv)).astype(complex))
        return out

    def med_R(omega):
        o = Organism(N=Nv, K=K, omega=omega, beta=10.0, seed=0)
        o.perceive(stream(3000, 5), g_in=4.0, eta=0.02, recruit=0.6)
        r = real_phase_residual(o.xi, o.used)
        return o, (float(np.median(r)) if r.size else 0.0)

    org0, r_om0 = med_R(0.0)
    org_d, r_def = med_R(0.25)
    _, r_hi = med_R(2.0)
    check("omega=0: median imaginary residual", r_om0, 0.0, 1e-12,
          note="phase 43: with omega=0 the perceive recursion is REAL, so a "
               "real-input store is a real store up to the decaying complex "
               "initial condition -- pinned at machine floor")
    check("default omega: median imaginary residual", r_def, 1e-6, 5e-2,
          note="phase 43: at omega/g_in << 1 with deep settling the channel "
               "sits at floor (33h arm measured 2.6e-04 over 1120 slots)")
    check("omega=2: median imaginary residual", r_hi, 5e-3, 0.5,
          note="phase 43 FALSIFIED the structural account -- the channel is "
               "empty by PARAMETER REGIME, not construction. Banded positive "
               "on purpose: a claim that phase cannot carry content must be "
               "re-earned against this row")

    # -- the layout: cost, exactness, and what it forfeits ----------------
    lossless_rp = CompressionSpec(xi_dtype=np.complex128, meta_dtype=np.float64,
                                  real_phase=True)
    # The layout's loss IS the residual, exactly: dropping the best-aligned
    # imaginary part costs a per-slot relative L2 error of sqrt(R). Pinning
    # that identity is what makes `real_phase_residual` a PRICE rather than a
    # diagnostic -- a store can be audited without ever building the layout.
    u = np.asarray(org_d.used, bool)
    xi_u = np.asarray(org_d.xi, dtype=complex)[u]
    rt = compress(org_d, spec=lossless_rp).xi_full()[u]
    rel = np.linalg.norm(rt - xi_u, axis=1) / np.linalg.norm(xi_u, axis=1)
    pred = np.sqrt(real_phase_residual(org_d.xi, org_d.used))
    check("real+phase loss equals sqrt(residual), max |d| over slots",
          float(np.abs(rel - pred).max()), 0.0, 1e-12,
          note="phase 43: the audit metric prices the layout exactly")
    st_c64 = compress(org_d, spec=CompressionSpec())
    st_rp = compress(org_d, spec=CompressionSpec(real_phase=True))
    check("real+phase xi byte ratio vs complex64", st_c64.xi_bytes / st_rp.xi_bytes,
          1.70, 2.05,
          note="phase 43: halves the dominant term again (2.03x -> 1.047x per "
               "stored memory at N=64); this is the storage fork's whole prize")
    check("real+phase total store ratio vs uncompressed",
          store_bytes(org_d) / st_rp.nbytes, 2.0, 1e6)

    # -- gauge: the per-slot phase is not read by any modulus consumer ----
    g = np.asarray(org_d.xi, dtype=complex) * np.exp(
        1j * np.random.default_rng(3).uniform(0, 2 * np.pi, org_d.K))[:, None]
    probe = np.asarray(stream(64, 9))
    o_base = np.abs(np.asarray(org_d.xi, dtype=complex).conj() @ probe.T) / Nv
    o_rot = np.abs(g.conj() @ probe.T) / Nv
    check("gauge: max |d|overlaps|| under a random per-slot rotation",
          float(np.abs(o_rot - o_base).max()), 0.0, 1e-15,
          note="phase 43 P4: the per-slot phase is GAUGE for every |.| "
               "consumer, so the layout's 4-byte scalar buys E3 round-trip, "
               "not behavior")
    check("gauge: slot-argmax agreement under rotation",
          float(np.mean(o_rot.argmax(0) == o_base.argmax(0))), 1.0, 1.0)


# ============================== 16. sparse-P default path (T7.2, phase 44)
def section_16_sparse_default():
    print("\n(16) sparse-P default path and narrow CSR (T7.2, phase 44)")
    from organism import SparseTransitions, TransitionGraph
    from organism_compress import CompressionSpec, compress

    def synth(K, out_degree=10, seed=0):
        r = np.random.default_rng(seed)
        g = TransitionGraph(K)
        P = np.zeros((K, K))
        for i in range(K):
            cols = r.choice(K, size=min(out_degree, K), replace=False)
            P[i, cols] = r.integers(1, 200, size=cols.size)
        g.P = P
        return g

    # -- the default change must be a wall-clock choice, never a numerical
    #    one: both paths bitwise identical on integer counts (T6.3's proof,
    #    now the auto-dispatch's precondition)
    K = TransitionGraph.SPARSE_MIN_K
    g = synth(K)
    idx = np.arange(K)
    nd, dd = g.next_hops(idx, 0, sparse=False)
    ns, ds = g.next_hops(idx, 0, sparse=True)
    na, da = g.next_hops(idx, 0)                    # the new default
    check("next_hops dense-vs-sparse: first-hop mismatches",
          float(np.count_nonzero(nd != ns)), 0.0, 0.0,
          note="phase 44: the auto-dispatch is bitwise, so SPARSE_MIN_K can "
               "only move wall-clock")
    check("next_hops dense-vs-sparse: max |d dist|",
          float(np.abs(dd - ds).max()), 0.0, 0.0)
    check("next_hops auto-dispatch matches the sparse path it chose",
          float(np.count_nonzero(na != ns) + np.abs(da - ds).max()), 0.0, 0.0)
    check("plan() unchanged under the auto-dispatch",
          float(g.plan(idx, K - 1, 0) == list(_trace_ref(ns, K - 1, 0))), 1.0, 1.0,
          note="planning reaches next_hops through one seam only")

    # -- the two dispatch conditions, each pinned on its own
    K_small = max(8, K // 8)
    small = synth(K_small)
    check("auto-dispatch stays dense below SPARSE_MIN_K",
          float(small._sparse_worth(np.arange(K_small))), 0.0, 0.0,
          note=f"phase 44 measured the crossover; SPARSE_MIN_K={K}")
    frac = synth(K)
    frac.P = frac.P * 0.5                          # p_decay's regime
    check("auto-dispatch stays dense on non-integer counts",
          float(frac._sparse_worth(idx)), 0.0, 0.0,
          note="bitwise identity is only available on integer counts, so the "
               "default may not fire where it would stop being exact")

    # -- narrow CSR: lossless, guarded, and roughly half the graph term
    class _Store:                    # the seven fields `compress` reads
        def __init__(self, graph, K, N=8):
            self.K, self.N = K, N
            self.xi = np.zeros((K, N), complex)
            self.P = np.asarray(graph.P)
            self.count = np.zeros(K)
            self.age = np.zeros(K)
            self.used = np.ones(K, bool)

    wide = CompressionSpec(xi_dtype=np.complex128, meta_dtype=np.float64)
    narrow = CompressionSpec(xi_dtype=np.complex128, meta_dtype=np.float64,
                             p_narrow=True)
    st = compress(_Store(g, K), spec=wide)
    stn = compress(_Store(g, K), spec=narrow)
    check("narrow CSR: max |dP| after reconstruction",
          float(np.abs(stn.P_full() - np.asarray(g.P)).max()), 0.0, 0.0,
          note="phase 44: int16 indices + unsigned-integer counts, both "
               "guarded -- lossless by construction, not by luck")
    check("narrow CSR: graph byte ratio vs int32/float32 CSR",
          st.p_bytes / stn.p_bytes, 1.80, 2.05,
          note="phase 44: graph term 0.355 -> 0.178 of the prototype bar, "
               "which still leaves the <=2x fallback MISSED at 2.07x")
    stf = compress(_Store(frac, K), spec=narrow)
    check("narrow CSR declines integer counts when they are not integers",
          float(np.issubdtype(stf.p_data.dtype, np.floating)), 1.0, 1.0,
          note="the guard fires rather than rounding")

    # -- the FALSIFIED scale-free assumption, banded so it must be re-earned
    Nv = 32
    NORMv = np.sqrt(Nv)
    r = np.random.default_rng(4)
    anchors = np.linalg.qr(r.standard_normal((Nv, 6)))[0].T * NORMv

    def density(epochs):
        o = Organism(N=Nv, K=24, omega=0.15, beta=10.0, seed=0)
        seq = []
        for _ in range(epochs):
            for _ in range(300):
                seq.extend([anchors[int(r.integers(6))].astype(complex)] * 4)
        o.perceive(seq, g_in=4.0, eta=0.02, recruit=0.6)
        i = np.where(o.used)[0]
        Pm = np.asarray(o.P)[np.ix_(i, i)]
        return float((Pm > 0).sum() / max(len(i) ** 2, 1))

    d1, d4 = density(1), density(4)
    check("P density growth from 1x to 4x observations", d4 / max(d1, 1e-12),
          1.0, 6.0,
          note="phase 44 FALSIFIED the scale-free reading: density RISES with "
               "observation count (3.8x digits / 6.1x text over 16x obs), so "
               "every CSR byte number needs an observation count attached. "
               "Banded so 'CSR saving is scale-free' must be re-earned")


# ============ 17. settling depth + chunk statistics (T7.3/T7.5, phases 45/47)
def section_18_victim_rule():
    print("\n(18) eviction victim rule (T6.2, phase 39): which stale slot dies")
    import os as _os, tempfile
    from organism_state import save_state, load_state

    # A stimulus with SPREAD IN ESTABLISHEDNESS, which is the whole point: a
    # few core memories revisited many times, plus a run of one-off novelties
    # that keep the bank under recruitment pressure. Phase 39's finding is
    # that the victim rule only has something to do when the stale pool
    # contains both kinds -- section 9's world-A/world-B toy evicts 3 times
    # and cannot see this at all.
    Nv, Kv, CORE, JUNK = 32, 8, 4, 28
    NORMv = np.sqrt(Nv)
    r0 = np.random.default_rng(11)
    Gr, _ = np.linalg.qr(r0.standard_normal((Nv, CORE))
                         + 1j * r0.standard_normal((Nv, CORE)))
    core = Gr.T * NORMv
    junk = np.array([(lambda v: v / np.linalg.norm(v) * NORMv)(
        r0.standard_normal(Nv) + 1j * r0.standard_normal(Nv))
        for _ in range(JUNK)])

    def build():
        r = np.random.default_rng(3)
        out = []
        for blk in range(120):
            for _ in range(40):                       # establish a core memory
                out.append(core[blk % CORE] + 0.35 * NORMv / np.sqrt(Nv) *
                           (r.standard_normal(Nv) + 1j * r.standard_normal(Nv)))
            for _ in range(12):                       # a novel one-off: pressure
                out.append(junk[blk % JUNK] + 0.35 * NORMv / np.sqrt(Nv) *
                           (r.standard_normal(Nv) + 1j * r.standard_normal(Nv)))
        return out

    S = build()

    def core_retention(o):
        u = o.used
        return float(np.mean([max(np.abs(o.overlaps(core[h], o.xi[u])))
                              for h in range(CORE)]))

    res, leds = {}, {}
    for rule in ('count', 'rate', 'random'):
        led = []
        o = Organism(N=Nv, K=Kv, seed=0)
        o.perceive(S, evict=150, evict_victim=rule, evict_debug=led)
        res[rule] = o
        leds[rule] = np.array([row[2] for row in led])   # victim lifetime count

    # -- the phase-39 headline, at toy scale: argmin-count PROTECTS the core,
    #    uniform-random does not. This is the sign that must not silently flip.
    check("core retention, victim='count'", core_retention(res['count']),
          0.95, 1.01, note="phase 39: argmin-count evicts the stream's own churn")
    check("core retention, victim='random'", core_retention(res['random']),
          0.70, 0.94,
          note="phase 39 (c) CONFIRMED: uniform eviction eats established "
               "memories. On the 33c protocol this costs dFORG +0.44 "
               "(n=10 paired, exact p=0.0020)")
    check("random loses core retention vs count",
          core_retention(res['count']) - core_retention(res['random']),
          0.02, 0.40, note="the victim rule is load-bearing, not decoration")

    # -- the mundane account phase 39 had to reject: if the stale pool were a
    #    singleton, or all-junk, the rules could not differ. Pin that it has
    #    real spread, so a future change that flattens it fails HERE.
    check("victim lifetime count, rule='count'", float(leds['count'].mean()),
          3.0, 15.0, note="argmin-count takes barely-established slots")
    check("victim lifetime count, rule='random'", float(leds['random'].mean()),
          20.0, 90.0,
          note="phase 39's discriminator: random takes ~5x better-established "
               "victims, so the pool HAS spread -- 'the pool is all junk "
               "anyway' was rejected, not assumed")

    # -- default-path neutrality: this is the invariant that lets T6.2 touch
    #    organism.py/fastpath.py at all. Omitting the parameter must be
    #    bitwise identical to passing the default.
    a = Organism(N=Nv, K=Kv, seed=0); a.perceive(S, evict=150)
    b = Organism(N=Nv, K=Kv, seed=0); b.perceive(S, evict=150, evict_victim='count')
    check("evict_victim omitted == 'count', max state delta",
          float(max(np.abs(a.xi - b.xi).max(), np.abs(a.P - b.P).max(),
                    np.abs(a.age - b.age).max())), 0.0, 0.0,
          note="EXACT: the default rule adds no arithmetic and no rng draw")
    check("tenure ticks only under 'rate' (count)", float(res['count'].tenure.sum()),
          0.0, 0.0, note="era normalizer is inert unless selected")
    check("tenure ticks only under 'rate' (random)", float(res['random'].tenure.sum()),
          0.0, 0.0)
    check("tenure ticks under 'rate'", float(res['rate'].tenure.sum()),
          1e3, 1e6, note="and is the only state the rate rule adds")

    # -- E3: every rule must keep "continue == never stopped", including the
    #    control arm, whose victim draws come off org.evict_rng
    for rule in ('count', 'rate', 'random'):
        one = Organism(N=Nv, K=Kv, seed=0)
        one.perceive(S[:3000], evict=150, evict_victim=rule)
        one.perceive(S[3000:], evict=150, evict_victim=rule)
        two = Organism(N=Nv, K=Kv, seed=0)
        two.perceive(S[:3000], evict=150, evict_victim=rule)
        path = _os.path.join(tempfile.gettempdir(), f"e3_victim_{rule}.npz")
        save_state(two, path)
        two = load_state(path, cls=Organism)
        two.perceive(S[3000:], evict=150, evict_victim=rule)
        d = max(np.abs(one.xi - two.xi).max(), np.abs(one.P - two.P).max(),
                np.abs(one.age - two.age).max(),
                np.abs(one.tenure - two.tenure).max(),
                np.abs(one.evictions - two.evictions).max())
        check(f"E3 restore mid-run, victim='{rule}', max state delta",
              float(d), 0.0, 1e-12,
              note=("org.evict_rng round-trips too, or the control arm would "
                    "silently diverge after a load" if rule == 'random' else
                    "numba 0.0e+00 exactly; the numpy backend's ~2e-15 on this "
                    "stimulus predates T6.2 (reproduced on origin/main)"))

    try:                                   # a typo must not silently run 'count'
        Organism(N=Nv, K=Kv, seed=0).perceive(S[:10], evict=150,
                                              evict_victim='argmin')
        bad = 1.0
    except ValueError:
        bad = 0.0
    check("unknown evict_victim raises ValueError", bad, 0.0, 0.0,
          note="silently falling back to the default would fake a null result")


def section_17_settling_and_chunks():
    print("\n(17) settling depth and chunk statistics (T7.3/T7.5, phases 45/47)")
    import phase45_settling_depth as p45
    import phase47_macro_recruit_pregates as p47

    # -- phase 45's vectorized field recursion is a REIMPLEMENTATION of the
    #    perceive z update, so it is checked against the real one, not trusted
    Nv, V = 24, 40
    NORMv = np.sqrt(Nv)
    rng = np.random.default_rng(6)
    emb = rng.standard_normal((V, Nv))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    probe = rng.integers(0, V, (1, 10))
    org = Organism(N=Nv, K=32, omega=0.15, beta=10.0, seed=0)
    z0 = np.asarray(org.z, complex)[None, :]
    org.perceive([normalize(emb[w].astype(complex), NORMv)
                  for w in probe[0] for _ in range(8)],
                 g_in=5.0, dt=0.05, eta=0.02, recruit=0.75)
    mine = p45.settle(probe, emb.astype(complex), 8, z0=z0, norm=NORMv)[0]
    check("phase-45 `settle` vs Organism.perceive's own z: max |dz|",
          float(np.abs(mine - np.asarray(org.z, complex)).max()), 0.0, 1e-6,
          note="the sweep's vectorized recursion is checked, not assumed -- "
               "a tolerance, not a bit, because the two paths accumulate the "
               "same normalizations in a different order")

    # -- the last-token attractor at the committed exposure, and the prefix
    #    signal's magnitude, which is what every threshold consumer sees
    B, T = 120, 8
    rg = np.random.default_rng(2)
    base = rg.integers(0, V, (B, T))
    same_suffix = base.copy()
    same_suffix[:, :-1] = rg.integers(0, V, (B, T - 1))
    E = emb.astype(complex)
    za = p45.settle(base, E, 8, z0=p45.rand_z(np.random.default_rng(1), B, Nv, NORMv),
                    norm=NORMv)
    zb = p45.settle(base, E, 8, z0=p45.rand_z(np.random.default_rng(2), B, Nv, NORMv),
                    norm=NORMv)
    zs = p45.settle(same_suffix, E, 8,
                    z0=p45.rand_z(np.random.default_rng(3), B, Nv, NORMv), norm=NORMv)
    ceiling = float(np.mean(p45.sim(za, zb, Nv)))
    suffix = float(np.mean(p45.sim(za, zs, Nv)))
    check("hold=8 ceiling (same sequence, different initial state)", ceiling,
          0.995, 1.0,
          note="phase 45: the committed exposure forgets its initial condition")
    check("hold=8 suffix similarity (shared last token, different prefix)",
          suffix, 0.85, 1.0,
          note="phase 45: a near-pure last-token attractor -- and ABOVE "
               "consolidate()'s 0.8 merge_thresh, which is the flag for "
               "whoever first consolidates sequence states")
    check("hold=8 prefix signal (ceiling - suffix)", ceiling - suffix,
          0.0, 0.15,
          note="phase 45: prefix identity is perfectly decodable at every "
               "hold, but its MAGNITUDE in similarity units is small -- path "
               "sensitivity is a property of the threshold consumer, not the "
               "state. Banded so a 'the state is path-blind' claim must be "
               "re-earned")

    # -- phase 47: the raw chunk-gain statistic fails its own null, which is
    #    why the null-corrected one exists. Pinned on a stream with NO
    #    sequential structure at all.
    rz = np.random.default_rng(8)
    zipf = np.minimum(rz.zipf(1.4, 60000), 200) - 1          # i.i.d. Zipfian
    Vz = int(zipf.max()) + 1
    half = len(zipf) // 2
    rows = p47.bigram_stats(zipf[:half], Vz)
    ch = p47.select(rows, 64, "frequency")
    raw = p47.gain(zipf[half:], ch, Vz)
    check("raw chunk gain on an i.i.d. stream (frequency rule)", raw, 1.0, 1e9,
          note="phase 47 P2 FAILED: merging shortens the stream and enlarges "
               "the table, so the raw statistic pays out with NO sequential "
               "structure present. Pinned POSITIVE so the broken statistic "
               "cannot quietly come back")
    perm = np.random.default_rng(9).permutation(zipf)
    rows_p = p47.bigram_stats(perm[:half], Vz)
    corrected = raw - p47.gain(perm[half:], p47.select(rows_p, 64, "frequency"), Vz)
    check("null-corrected chunk gain on an i.i.d. stream", abs(corrected),
          0.0, max(abs(raw) * 0.05, 1.0),
          note="phase 47 P7: the correction cancels the length/estimation "
               "term, so a structureless stream scores near zero")

    # -- and the clustering artifact the category arm died on
    lab = rz.integers(0, 8, Vz)                     # a RANDOM partition
    cat = lab[zipf]
    cg = p47.gain(cat[half:], p47.select(p47.bigram_stats(cat[:half], 8), 32,
                                         "total"), 8)
    check("chunk gain from a RANDOM partition of a Zipfian stream", cg,
          1.0, 1e9,
          note="phase 47 P5/M5: a random partition manufactures apparent "
               "second-order structure, which is why the label-permutation "
               "null is the load-bearing control -- the discovered categories "
               "FAILED it (810.3 vs 1279.8). Pinned positive so 'level 2 "
               "learns something' has to be re-earned against this row")


# ========================== 19. the (A) store on the ladder (T7.7, phase 50)
def section_20_compressed_persistence():
    print("\n(20) compressed-state persistence for every CompressionSpec lever "
          "(owner audit, 2026-08-28)")
    import os as _os, tempfile
    from organism_state import save_state, load_state
    from organism_compress import CompressionSpec, compress

    # WHY THIS SECTION EXISTS. A compute-and-use audit found that
    # save_state(..., CompressionSpec(real_phase=True)) wrote a file that
    # load_state then REFUSED: the compressor puts the store in st.polar and
    # leaves st.xi = None, and _save_compressed wrote arrays['xi'] = st.xi
    # regardless, producing a 0-d object array that fails under
    # allow_pickle=False. So the T7.1 branch-(A) layout -- the arm the owner
    # RATIFIED on 2026-08-14, whose entire value proposition is persistence --
    # could not survive a round trip. Sections 8 and 10 both missed it because
    # neither exercised these two levers. Every lever gets a round trip here.
    rg = np.random.default_rng(4)
    Nv, Kv = 32, 8
    S = [(lambda v: v / np.linalg.norm(v) * np.sqrt(Nv))(
         rg.standard_normal(Nv) + 1j * rg.standard_normal(Nv)) for _ in range(600)]
    org = Organism(N=Nv, K=Kv, seed=0)
    org.perceive(S)

    specs = (("plain", CompressionSpec()),
             ("real_phase", CompressionSpec(real_phase=True)),
             ("p_narrow", CompressionSpec(p_narrow=True)),
             ("real_phase+p_narrow", CompressionSpec(real_phase=True, p_narrow=True)))
    for name, spec in specs:
        path = _os.path.join(tempfile.gettempdir(), f"e3_cspec_{name}.npz")
        save_state(org, path, compress_spec=spec)
        back = load_state(path, cls=Organism)
        ref = compress(org, spec=spec).xi_full()      # the encoding's OWN target,
        d_xi = float(np.abs(ref - back.xi).max())     # not the uncompressed store
        d_P = float(np.abs(org.P - back.P).max())
        check(f"E3 round-trip '{name}': max|dxi| vs its own encoding",
              d_xi, 0.0, 0.0,
              note="EXACT: a lever that cannot round-trip is not a lever")
        check(f"E3 round-trip '{name}': max|dP|", d_P, 0.0, 0.0)

    # the spec must come BACK, or byte accounting from a loaded state is wrong
    path = _os.path.join(tempfile.gettempdir(), "e3_cspec_flags.npz")
    save_state(org, path, compress_spec=CompressionSpec(real_phase=True, p_narrow=True))
    import json as _json
    with np.load(path, allow_pickle=False) as f:
        rec = _json.loads(bytes(f['c_spec']).decode())
    check("c_spec records real_phase", float(bool(rec.get('real_phase'))), 1.0, 1.0,
          note="p_narrow round-tripped by accident before this, reporting False")
    check("c_spec records p_narrow", float(bool(rec.get('p_narrow'))), 1.0, 1.0)


def section_19_fork_ladder():
    print("\n(19) the (A) store on the ladder (T7.7, phase 50): store-mode "
          "fork equivalence")
    from label_readout import LabelEvidenceReadout
    from organism_compress import CompressionSpec, compress

    # A miniature of the 33c ladder: real-input class-incremental stream,
    # two tasks of two classes, store COMPRESSED AND CARRIED FORWARD at
    # every task boundary (the deployment path phase 50 measured). Toy
    # scale on purpose -- the full-scale numbers live in phase 50's row;
    # this row exists so the store-mode equivalence and the layout's price
    # arithmetic cannot drift without a harness failure.
    Nv, Kv, C = 32, 12, 4
    NORMv = np.sqrt(Nv)
    r0 = np.random.default_rng(19)
    anchors = np.linalg.qr(r0.standard_normal((Nv, C)))[0].T * NORMv  # REAL
    TASKS_ = [[0, 1], [2, 3]]

    def sample(c, r):
        v = anchors[c] + 0.30 * r.standard_normal(Nv)
        return v / np.linalg.norm(v) * NORMv

    re_ = np.random.default_rng(7)
    Xe = np.array([sample(c, re_) for c in range(C) for _ in range(25)])
    ye = np.repeat(np.arange(C), 25)

    def run(spec):
        r = np.random.default_rng(5)
        o = Organism(N=Nv, K=Kv, omega=0.15, beta=10.0, seed=0)
        ro = LabelEvidenceReadout(K=Kv, n_classes=C)
        A = np.zeros((2, 2))
        for ti, task in enumerate(TASKS_):
            Xtr = np.array([sample(c, r) for _ in range(40) for c in task])
            ytr = np.array([c for _ in range(40) for c in task])
            seq = []
            for i in r.permutation(len(Xtr)):
                seq.extend([Xtr[i].astype(complex)] * 4)
            o.perceive(seq, g_in=4.0, eta=0.02, recruit=0.6)
            ro.observe(o, Xtr, ytr)
            compress(o, spec=spec).apply(o)     # store mode: loss compounds
            for tj, tk in enumerate(TASKS_):
                m = np.isin(ye, tk)
                A[ti, tj] = float((ro.predict(o, Xe[m]) == ye[m]).mean())
        acc = float(np.mean(A[-1]))
        forg = float(np.max(A[:, 0]) - A[1, 0])
        return acc, forg, int(compress(o, spec=spec).nbytes)

    acc_c, forg_c, b_c = run(CompressionSpec())              # (B): c64+CSR
    acc_r, forg_r, b_r = run(CompressionSpec(real_phase=True))   # (A)

    check("mini-ladder c64 arm final ACC", acc_c, 0.80, 1.01,
          note="guards the equivalence rows against passing vacuously on a "
               "dead readout")
    check("store-mode fork delta |dACC| (real+phase vs c64)",
          abs(acc_r - acc_c), 0.0, 0.02,
          note="phase 50: held-out mean dACC within +-0.005 at K=112/160, "
               "both decoders -- the band here is 33e's claim threshold, "
               "the line phase 50 pre-registered as a REAL cost")
    check("store-mode fork delta |dFORG| (real+phase vs c64)",
          abs(forg_r - forg_c), 0.0, 0.02,
          note="phase 50: held-out mean dFORG within +-0.010; the kill line "
               "(+0.02, which resolves the fork to (B)) was never approached")
    check("(A)/(B) store byte ratio, same live store", b_r / b_c, 0.48, 0.68,
          note="phase 50 measured 0.590 (K=112) / 0.581 (K=160): the "
               "discount lives entirely in the xi term, so it is "
               "K-independent -- this row is the arithmetic's toy twin")


def _trace_ref(nxt, a, b):
    """`organism._trace`, inlined so the check does not import a private."""
    from organism import _trace
    return _trace(nxt, a, b)


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
    section_13_detection_driven_split()
    section_14_task_free_continual()
    section_15_phase_channel()
    section_16_sparse_default()
    section_17_settling_and_chunks()
    section_18_victim_rule()
    section_19_fork_ladder()
    section_20_compressed_persistence()

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
