"""
PHASE 18 -- MIXTURE HYGIENE AT THE SOURCE: soft ambiguity-gated routing

Phase 17 crossed sigma* for STORAGE but left generation behind: frozen
mixture slots, born while greedy routing among ~40 young loose-barred pools
is fluctuation-dominated, pollute recall -- and five post-hoc cures all
failed, because no per-slot statistic can undo bootstrap-era mixing after
the fact. This phase acts BEFORE the mixing: evidence routing itself is
weighted by attribution confidence (Organism.perceive(amb=...)):

  - while the winning pool is PROVISIONAL, its absorption is scaled by
    conf = clip(margin/amb, 0, 1), margin = the winner's lead over the
    runner-up across all live slots. Confidence scales everything --
    evidence mass, the running-mean step, the visit-quality EMA, and
    staying alive (age *= 1-conf): a pool subsisting on contested scraps
    expires, one with regular confident wins stays young;
  - mature winners absorb at full weight (their annealed bar already
    rejects wrong-word tokens); at sigma=0 margins are huge, conf
    saturates, and the clean case is untouched by construction;
  - amb=0 reproduces phase 17 exactly.

Why SOFT and not a threshold: Part A measures the routing-chaos margins
directly. Correct and wrong greedy assignments have OVERLAPPING margin
distributions at sigma=0.3 (wrong best-of-many RIVALS the right match),
so any hard margin gate must trade coverage for purity along a frontier
-- and prototypes confirmed every hard variant (drop contested tokens,
recruit on deep ambiguity, remove the weak-tail dead zone, raise confirm,
shrink probation) just slides ALONG that frontier. Down-weighting is the
only move that shifted it outward: mixing events barely move any pool,
while a word's ~150 recurrences arrive mostly at decent margins.

Protocol: same world and scorer as phases 14/17 (26 words, 4000-word
stream, recall2 generation), sigma in {0, 0.2, 0.25, 0.3, 0.35, 0.4} x
{gate (phase 14), pooled (phase 17, amb=0), amb=0.3, amb=1.0}, plus the
amb frontier at sigma=0.3. Success criteria: clean case unharmed (within
0.03); at sigma=0.2 the phase-17 strict win must be preserved or
extended; at sigma=0.3 amb=0.3 must dominate phase 17 (higher generation
AND prediction at coverage >= 0.99) and amb=1.0 must dominate the
phase-14 gate (higher generation AND coverage).

RESULT (recorded from the committed run):
  - CLEAN CASE IDENTICAL: amb=0.3 reproduces phase 17's sigma=0 line
    exactly (26 slots, coverage 1.00, generation 0.985).
  - STRICT WIN EXTENDED AT sigma=0.2: generation 0.810 -> 0.862 with
    exactly 26 slots (phase 17 carried 28); every metric at ceiling.
  - sigma=0.25: the largest gains, right at sigma*: amb=0.3 lifts
    generation 0.509 -> 0.659 (coverage 0.96, junk 0.198 -> 0.070) and
    amb=1.0 reaches 0.672 junk-free -- both beat every phase-17 arm.
  - AT sigma=0.3 THE GATE DOMINATES BOTH PREDECESSORS: amb=0.3 keeps
    coverage 1.00 and beats phase 17 on generation (0.406 vs 0.271)
    and prediction (0.579 vs 0.504); amb=1.0 beats the phase-14 gate
    on BOTH axes at once (generation 0.584 vs 0.489, coverage 0.69 vs
    0.50, junk 0.000) -- the "abandon vocabulary for purity" corner is
    no longer the gate's to claim.
  - BEYOND THE STORAGE BOUNDARY the gate still helps but cannot save
    generation: at sigma=0.4 amb=0.3 improves every phase-17 metric
    (coverage 0.62 -> 0.73, generation 0.243 -> 0.286, prediction
    0.285 -> 0.351); at 0.35 it trades coverage for generation like
    everything else. Part A says why: pool maturation no longer
    outruns the error rate there.
  - THE FRONTIER IS THE RESIDUAL: amb tunes a purity-coverage frontier
    (0.406/1.00 at 0.3 through 0.584/0.69 at 1.0) that Part A shows is
    information-limited at the single-token level: at sigma=0.3 with
    size-1 pools, 77% of greedy assignments are WRONG, and the margin
    barely discriminates (error still 70% above the median margin;
    correct 0.097+-0.075 vs wrong 0.062+-0.055) -- no routing statistic
    of one settled state can reach both ends at once. Maturation is
    what kills the error (30% at pool size 16), which is why gating
    only provisional winners works at all. Open: a richer evidence unit -- attribution informed by
    temporal context (the transition prior already knows what should
    come next), so that ambiguous tokens inherit confidence from
    sequence structure instead of being down-weighted.
"""

import numpy as np
from organism import Organism, normalize
from phase14_noise_robust_perception import (
    emb, V, N, NORM, sample_stream, frames, evaluate)
from phase17_pooled_recruitment import pool_bars


# ---------------------------------------------------------------- Part A
def routing_margins(sigma, n_pool, trials=3000, seed=5):
    """The bootstrap routing chaos, measured directly: greedy assignment of
    noisy tokens to one pool per word, each pool the centroid of n_pool
    noisy tokens. Returns error rate and the margin (winner minus
    runner-up) distributions for correct vs wrong assignments -- the
    overlap of those distributions is what any routing gate must live
    with. (One pool per word understates the live-pool count during
    bootstrap, so the true chaos is worse.)"""
    rng = np.random.default_rng(seed)
    ok_m, bad_m = [], []
    for rep in range(6):
        pools = np.array([normalize(
            np.mean([emb[w] + sigma * rng.standard_normal(N) for _ in range(n_pool)],
                    axis=0).astype(complex), NORM) for w in range(V)])
        for _ in range(trials // 6):
            w = int(rng.integers(V))
            tok = normalize((emb[w] + sigma * rng.standard_normal(N)).astype(complex), NORM)
            ov = np.abs(pools.conj() @ tok) / N
            k = int(np.argmax(ov))
            top = ov[k]; ov[k] = -1.0
            margin = top - ov.max()
            (ok_m if k == w else bad_m).append(margin)
    ok_m, bad_m = np.array(ok_m), np.array(bad_m)
    err = len(bad_m) / (len(ok_m) + len(bad_m))
    med = float(np.median(np.concatenate([ok_m, bad_m])))
    hi = np.concatenate([ok_m[ok_m > med], bad_m[bad_m > med]])
    err_hi = float(np.mean(np.concatenate([np.zeros(len(ok_m[ok_m > med])),
                                           np.ones(len(bad_m[bad_m > med]))]))) if len(hi) else 0.0
    return dict(err=err, ok=(float(ok_m.mean()), float(ok_m.std())),
                bad=(float(bad_m.mean()), float(bad_m.std())) if len(bad_m) else (0.0, 0.0),
                err_hi=err_hi)


# ---------------------------------------------------------------- Part B/C
def run(sigma, mode, amb=0.0):
    org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    seq = sample_stream(4000, seed=99)
    fr = frames(seq, sigma)
    pb, ab = pool_bars(sigma)
    if mode == 'gate':                                    # phase 14
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, confirm=3)
    else:                                                 # pooled (phase 17) + amb
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                     active_bar=ab, s_hat=sigma**2 * N, probation=12000, amb=amb)
    r = evaluate(org)
    label = mode if mode == 'gate' else (f"amb={amb}" if amb > 0 else "pooled")
    print(f"  {label:<9} slots={r['slots']:>3}  coverage={r['cov']:.2f}  "
          f"prediction={r['pred']:.3f}  generation={r['gram']:.3f}  "
          f"junk hops={r['junk']:.3f}  hops={r['hops']}")
    return r


if __name__ == '__main__':
    print("PHASE 18: soft ambiguity-gated evidence routing\n")

    print("(A) bootstrap routing chaos: greedy-assignment error and margins")
    print(f"{'sigma':>6} {'n_pool':>7} {'error':>7} {'correct margin':>16} "
          f"{'wrong margin':>15} {'err above median':>17}")
    for sigma in (0.2, 0.3, 0.4):
        for n_pool in (1, 4, 16):
            m = routing_margins(sigma, n_pool)
            print(f"{sigma:>6} {n_pool:>7} {m['err']:>6.1%} "
                  f"{m['ok'][0]:>9.3f} +-{m['ok'][1]:.3f} "
                  f"{m['bad'][0]:>8.3f} +-{m['bad'][1]:.3f} {m['err_hi']:>16.1%}")
    print("    correct and wrong margins OVERLAP at n_pool=1 (the bootstrap era):\n"
          "    a hard threshold cannot separate them -- purity must be bought\n"
          "    with coverage. Down-weighting (soft conf) is the frontier-shifting\n"
          "    move; maturation (n_pool up) is what actually kills the error.\n")

    print("(B) the amb frontier at sigma=0.3")
    frontier = {}
    for amb in (0.0, 0.1, 0.3, 0.5, 1.0, 2.0):
        frontier[amb] = run(0.3, 'pooled', amb=amb)

    print("\n(C) deployment sweep (26 words, 4000-word stream, recall2 generation)")
    results = {}
    for sigma in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4):
        print(f"\n=== sigma={sigma} ===")
        for mode, amb in (('gate', 0.0), ('pooled', 0.0), ('amb', 0.3), ('amb', 1.0)):
            results[(sigma, mode, amb)] = (frontier[amb] if sigma == 0.3 and mode != 'gate'
                                           and amb in frontier else run(sigma, mode, amb=amb))

    print("\n" + "=" * 74)
    print("PHASE 18 SUMMARY -- generation grammaticality (chance ~0.33) / coverage")
    print(f"{'sigma':>6} {'gate (p14)':>13} {'pooled (p17)':>13} {'amb=0.3':>13} {'amb=1.0':>13}")
    for sigma in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4):
        row = " ".join(f"{results[(sigma, m, a)]['gram']:>8.3f}/{results[(sigma, m, a)]['cov']:.2f}"
                       for m, a in (('gate', 0.0), ('pooled', 0.0), ('amb', 0.3), ('amb', 1.0)))
        print(f"{sigma:>6} {row}")

    r0, r03 = results[(0.0, 'pooled', 0.0)], results[(0.0, 'amb', 0.3)]
    clean_ok = r03['gram'] >= r0['gram'] - 0.03 and r03['cov'] >= r0['cov']
    ext02 = results[(0.2, 'amb', 0.3)]['gram'] >= results[(0.2, 'pooled', 0.0)]['gram']
    p3a, p3p = results[(0.3, 'amb', 0.3)], results[(0.3, 'pooled', 0.0)]
    dom_p17 = (p3a['cov'] >= 0.99 and p3a['gram'] > p3p['gram'] and p3a['pred'] > p3p['pred'])
    p3g, p3h = results[(0.3, 'gate', 0.0)], results[(0.3, 'amb', 1.0)]
    dom_gate = p3h['gram'] > p3g['gram'] and p3h['cov'] > p3g['cov']
    if clean_ok and ext02 and dom_p17 and dom_gate:
        print("\nverdict: SOFT AMBIGUITY GATE SHIFTS THE FRONTIER -- clean case "
              "untouched, sigma=0.2 win extended "
              f"({results[(0.2,'pooled',0.0)]['gram']:.3f} -> {results[(0.2,'amb',0.3)]['gram']:.3f}), "
              f"and at sigma=0.3 amb=0.3 dominates phase 17 at full coverage "
              f"({p3p['gram']:.3f} -> {p3a['gram']:.3f}) while amb=1.0 dominates the "
              f"phase-14 gate on both axes ({p3h['gram']:.3f}/{p3h['cov']:.2f} vs "
              f"{p3g['gram']:.3f}/{p3g['cov']:.2f}); the frontier itself is the open "
              "problem -- single-token attribution is information-limited (Part A)")
    elif clean_ok:
        print("\nverdict: clean-safe but the gate does not dominate its predecessors "
              "everywhere -- report the frontier honestly and re-diagnose")
    else:
        print("\nverdict: the gate taxes clean learning -- not accepted as-is")
