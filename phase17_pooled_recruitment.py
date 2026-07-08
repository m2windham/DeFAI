"""
PHASE 17 -- RECRUITMENT BEYOND SIGMA*: pooled probationary evidence

Phase 14 ended on an analytic wall: probationary recruitment is a strict win
up to sigma=0.2, but at sigma* ~= 0.24 a genuine revisit's overlap with its
own clean memory, 1/sqrt(1 + sigma^2 N), falls below the 0.6 confirmation
bar -- so ANY single-shot scheme must recycle real words (coverage 0.50-0.62
at sigma=0.3). Its own closing conjecture: denoising must pool across
occurrences BEFORE the keep/discard decision.

This phase implements that conjecture (Organism.perceive(pool=True)):

  - provisional slots become fast-averaging EVIDENCE POOLS: they update by
    running mean over their matched frames, so the trace's noise shrinks
    ~1/n while the decision is still open (the phase-14 slow EMA, eta=0.02,
    keeps essentially the first token's noise through probation);
  - matching, confirmation visits, and recruitment all use ONE lower bar
    set from the noise level -- pool_bar = min(recruit, 0.8/(1+sigma^2 N)),
    80% of the expected SAME-WORD TOKEN-TOKEN overlap 1/(1+sigma^2 N),
    which is what the first revisit actually sees (a 1-token pool is just
    another noisy token, NOT a clean memory);
  - the 0.6 activity bar for counting/transition learning is exposed as
    active_bar = min(0.6, 0.85/sqrt(1+sigma^2 N)) -- beyond sigma* even a
    perfectly denoised memory is matched by its own tokens at only
    1/sqrt(1+sigma^2 N), so a fixed 0.6 silently stops P from learning.

Both formulas reduce to the phase-14 defaults at sigma=0: the mechanism is
strictly a generalization, and the clean case must be UNHARMED.

Predicted new boundary. Pooling moves the binding constraint from
token-vs-memory overlap (dies at sigma*=0.24) to token-vs-token
DISCRIMINATION: the same-word pairwise overlap 1/(1+sigma^2 N) must stay
separable from the cross-word overlap (~= word-word overlap /(1+sigma^2 N)
plus a noise-noise fluctuation of std sigma^2 sqrt(N)/(1+sigma^2 N), which
does NOT shrink relative to the signal). Part A measures these margins;
the sweep should find a new collapse where they cross, around sigma ~= 0.4.

The mechanism that survived iteration (16 designs; each simpler scheme's
failure is measured and documented in organism.py's docstring): saccade-
gated evidence, annealed per-slot acceptance bars, running-mean pooling
with a plasticity floor, held-out visit-quality graduation, online fusion
of converged duplicates, a weak-tail dead zone, and use-it-or-lose-it slot
recycling.

Protocol: same world and scorer as phase 14 (26 words, 4000-word stream,
recall2 generation), sigma in {0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5} x
{confirm=0 baseline, confirm=3 (phase 14), pooled (confirm=3, pool=True)}.
Success criteria: at sigma=0.3 (beyond sigma*, where phase 14's gate
recycles real words and coverage collapses to 0.50-0.62) pooled coverage
>= 0.9 with prediction above both arms; the clean case must be UNHARMED
(within 0.03 of baseline). Generation is reported but not claimed: the
smoke runs showed the storage win does not yet transfer to generation
(frozen mixture pools from the early routing chaos survive as slots), and
that residual is characterized honestly below.
"""

import numpy as np
from organism import Organism, normalize
from phase14_noise_robust_perception import (
    emb, cats, V, N, NORM, NEXT_FWD, sample_stream, frames, evaluate)


# ---------------------------------------------------------------- Part A
def margins(sigma, n_tok=300, seed=5):
    """Measured overlap margins that govern pooling at noise level sigma:
    same-word token-token (what the FIRST revisit sees), same-word
    token-vs-pooled-centroid for growing pools, and cross-word token-token
    (the false-match channel). Predictions from the analytic forms."""
    rng = np.random.default_rng(seed)
    words = rng.choice(V, size=8, replace=False)
    toks = {w: np.array([normalize((emb[w] + sigma * rng.standard_normal(N)).astype(complex), NORM)
                         for _ in range(n_tok)]) for w in words}
    same, cross = [], []
    for i, w in enumerate(words):
        ov = np.abs(toks[w][:100].conj() @ toks[w][100:200].T) / N
        same.extend(np.diag(ov))
        for v in words[i + 1:]:
            ov = np.abs(toks[w][:50].conj() @ toks[v][:50].T) / N
            cross.extend(np.diag(ov))
    pooled = {}
    for n in (1, 2, 4, 8, 32):
        vals = []
        for w in words:
            c = normalize(toks[w][:n].mean(0), NORM)
            vals.extend(np.abs(toks[w][200:].conj() @ c) / N)
        pooled[n] = float(np.mean(vals))
    pred_pair = 1 / (1 + sigma**2 * N)
    pred_asym = 1 / np.sqrt(1 + sigma**2 * N)
    return dict(same=(float(np.mean(same)), float(np.std(same))),
                cross=(float(np.mean(cross)), float(np.std(cross))),
                pooled=pooled, pred_pair=pred_pair, pred_asym=pred_asym)


# ---------------------------------------------------------------- Part B
def pool_bars(sigma):
    """Both reduce to the phase-14 defaults at sigma=0."""
    pool_bar = min(0.5, 0.8 / (1 + sigma**2 * N))
    active_bar = min(0.6, 0.85 / np.sqrt(1 + sigma**2 * N))
    return pool_bar, active_bar


def run(sigma, mode):
    org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    seq = sample_stream(4000, seed=99)
    fr = frames(seq, sigma)
    if mode == 'base':                                    # phase-13 status quo
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    elif mode == 'gate':                                  # phase 14
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, confirm=3)
    elif mode == 'bars':                                  # ablation: bars without pooling
        pb, ab = pool_bars(sigma)
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.02, recruit=pb, confirm=3,
                     active_bar=ab)
    else:                                                 # phase 17: pooled
        pb, ab = pool_bars(sigma)
        # probation doubles: pooling defers the keep/discard decision across
        # more occurrences by construction (denoise first, THEN confirm).
        # eta=0.05: the per-VISIT plasticity floor -- a maturing pool must
        # re-center on its recent (bar-filtered, nearly pure) evidence or
        # early contamination stays baked into the running mean forever.
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                     active_bar=ab, s_hat=sigma**2 * N, probation=12000)
    r = evaluate(org)
    print(f"  {mode:<7} slots={r['slots']:>3}  coverage={r['cov']:.2f}  "
          f"prediction={r['pred']:.3f}  generation={r['gram']:.3f}  "
          f"junk hops={r['junk']:.3f}  hops={r['hops']}")
    return r


if __name__ == '__main__':
    print("PHASE 17: pooled probationary recruitment beyond sigma* ~= 0.24\n")

    print("(A) overlap margins, measured vs predicted (N=%d)" % N)
    print(f"{'sigma':>6} {'same-word pair':>17} {'pred':>6} {'cross-word':>14} "
          f"{'pool->inf':>10} {'pred':>6}")
    for sigma in (0.2, 0.3, 0.4, 0.5):
        mg = margins(sigma)
        print(f"{sigma:>6} {mg['same'][0]:>9.3f} +-{mg['same'][1]:.3f} "
              f"{mg['pred_pair']:>6.3f} {mg['cross'][0]:>7.3f} +-{mg['cross'][1]:.3f} "
              f"{mg['pooled'][32]:>10.3f} {mg['pred_asym']:>6.3f}")
    print("    boundary logic: pooling works while same-word pair overlap clears the\n"
          "    cross-word mean by ~1 std; the ratio is fixed but the fluctuation floor\n"
          "    (sigma^2 sqrt(N)/(1+sigma^2 N)) does not shrink -- expect collapse ~0.4\n")

    print("(B) deployment sweep (26 words, 4000-word stream, recall2 generation)")
    results = {}
    for sigma in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        pb, ab = pool_bars(sigma)
        print(f"\n=== sigma={sigma}  (pool_bar={pb:.3f}, active_bar={ab:.3f}) ===")
        for mode in ('base', 'gate', 'pooled'):
            results[(sigma, mode)] = run(sigma, mode)

    print("\n=== mechanism ablation at sigma=0.3: pooled bars, slow EMA (no pooling) ===")
    results['bars'] = run(0.3, 'bars')

    print("\n" + "=" * 74)
    print("PHASE 17 SUMMARY -- generation grammaticality (chance ~0.33) / coverage")
    print(f"{'sigma':>6} {'base':>13} {'gate (p14)':>13} {'pooled (p17)':>13}")
    for sigma in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        row = " ".join(f"{results[(sigma, m)]['gram']:>8.3f}/{results[(sigma, m)]['cov']:.2f}"
                       for m in ('base', 'gate', 'pooled'))
        print(f"{sigma:>6} {row}")
    print(f"\nablation (bars without pooling) at sigma=0.3: "
          f"generation {results['bars']['gram']:.3f} coverage {results['bars']['cov']:.2f}")

    clean_ok = results[(0.0, 'pooled')]['gram'] >= results[(0.0, 'base')]['gram'] - 0.03
    star = results[(0.3, 'pooled')]
    beat_pred = star['pred'] - max(results[(0.3, 'base')]['pred'], results[(0.3, 'gate')]['pred'])
    gen_ok = star['gram'] > max(results[(0.3, 'base')]['gram'], results[(0.3, 'gate')]['gram'])
    if clean_ok and star['cov'] >= 0.9 and beat_pred > 0:
        cols = [s for s in (0.35, 0.4, 0.5) if results[(s, 'pooled')]['cov'] < 0.9]
        edge = f"; storage boundary moves to sigma ~= {cols[0]}" if cols else \
            "; no storage collapse seen up to 0.5"
        gen_note = "and generation follows" if gen_ok else \
            "but generation does NOT follow -- frozen mixture slots pollute recall"
        print(f"\nverdict: POOLED RECRUITMENT CROSSES SIGMA* FOR STORAGE -- sigma=0.3 "
              f"coverage {star['cov']:.2f} (gate {results[(0.3,'gate')]['cov']:.2f}), "
              f"prediction +{beat_pred:.2f}, clean case unharmed{edge}; {gen_note}")
    elif clean_ok:
        print("\nverdict: clean-safe but sigma=0.3 not conquered -- pooling as implemented "
              "does not extract the pooled information; re-diagnose the matching step")
    else:
        print("\nverdict: pooling taxes clean learning -- not accepted as-is")
