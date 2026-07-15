"""
PHASE 32: SELF-CONSISTENT CALIBRATION BEYOND RANK SATURATION (V >= N).

The gap (pre-registered in phase 26): the spectral noise-energy estimator
needs noise-only eigendirections (vocabulary rank < N). Real text is the
opposite regime -- phase 23's corpus has V=395 words in N=30 dims -- so the
covariance floor is signal-contaminated and the phase-26 arm cannot be
trusted there. Without a V >= N estimator, self-calibration (a Path B
requirement and the phase-26 result) only works on toy vocabularies.

The method, in the project's own spirit (pool evidence when singles are
information-insufficient -- phases 17/26 both rediscovered this):

  SELF-CONSISTENT FIXED POINT. Guess s. Run the organism's own pooling
  perception on a calibration prefix with bars placed from that guess.
  Pooling produces DENOISED slots (n visits shrink a slot's noise ~1/n),
  and token-vs-denoised-slot matching is far better conditioned than the
  token-vs-token matching that failed in phase 26 (overlap 1/sqrt(1+s)
  instead of 1/(1+s), against cross fluctuations that shrink as the slot
  denoises). Measure the median best-slot overlap of settled tokens against
  mature slots, invert the phase-17 overlap law
        ov(n) = 1/sqrt((1+s)(1+s/n))
  for s, and feed the estimate back into the bars. Iterate. A wrong guess
  mis-places the bars, but pooling still denoises whatever it accepts, and
  the measurement pulls the estimate toward the truth; the claim under test
  is that this converges to the true s -- ON RANK-SATURATED DATA where the
  spectral method cannot run at all.

Setup: V=60 words in N=30 dims (rank-saturated, 2x oversubscribed),
phase-14 embedding recipe (3 category bases + noise), category-cyclic
grammar stream, HOLD=12. Ground truth (true s, word identities) used ONLY
for evaluation -- standing rules.

Pre-registered success criteria:
  (a) converged s_cal within 30% of true s at sigma = 0.1, 0.2, 0.3
      (the spectral method's own bar was 25% on the EASIER V < N case);
  (b) pool perception with bars placed from s_cal reproduces the
      oracle-bars coverage within 0.05 at each sigma;
  (c) ablation (redundancy check): is the fixed point NEEDED, or does
      iteration 1 (a single measure-and-correct pass from s0=0) already
      land? If one pass suffices, the loop is over-engineering -- report
      honestly either way.

Uses the E2 numba backend for the inner perception runs (dogfooding: this
is exactly the fast-iteration workload E2 was built for).
"""

import numpy as np
from organism import Organism, normalize
from organism_numba import NumbaOrganism
from phase26_percentile_bars import settle_tokens as _settle_generic

# ------------------------------------------------------------- world (V>N)
N = 30
NORM = np.sqrt(N)
V = 60
CATS = 3
HOLD = 12
P_CORRECT = 0.88
true_cat = np.arange(V) % CATS

emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((CATS, N))
cat_bases[0, 0:3] = 1.0; cat_bases[1, 3:6] = 1.0; cat_bases[2, 6:9] = 1.0
emb = np.zeros((V, N))
for i in range(V):
    emb[i] = 0.6 * cat_bases[true_cat[i]] + 0.4 * emb_rng.standard_normal(N)
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
emb_c = emb.astype(complex)
NEXT = {0: 1, 1: 2, 2: 0}


def sample_stream(n, seed):
    r = np.random.default_rng(seed)
    pools = [np.where(true_cat == c)[0] for c in range(CATS)]
    c = 0; seq = []
    for _ in range(n):
        seq.append(int(r.choice(pools[c])))
        c = NEXT[c] if r.random() < P_CORRECT else int(
            r.choice([cc for cc in range(CATS) if cc != NEXT[c]]))
    return seq


def frames(seq, sigma, seed=1):
    r = np.random.default_rng(seed)
    out = []
    for w in seq:
        e = emb[w] + (sigma * r.standard_normal(N) if sigma > 0 else 0)
        out.extend([normalize(e.astype(complex), NORM)] * HOLD)
    return out


def settle_tokens(fr):
    return _settle_generic(fr)          # same field settling; N matches


# --------------------------------------------------- bars from an s guess
def bars_from_s(s):
    """Phase-17 bar map with s measured instead of handed (phase 26 showed
    s_hat is the load-bearing quantity; the bar shapes stay)."""
    pool_bar = min(0.5, 0.8 / (1 + s))
    active_bar = min(0.6, 0.85 / np.sqrt(1 + s))
    return pool_bar, active_bar


def run_pool(fr, s, probation=12000, amb=0.3, cls=NumbaOrganism, K=120):
    pb, ab = bars_from_s(s)
    org = cls(N=N, K=K, omega=0.15, beta=10.0, seed=0)
    org.perceive(fr, g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                 active_bar=ab, s_hat=s, probation=probation, amb=amb)
    return org


# ------------------------------------------------ the self-consistent loop
def measure_s(org, toks, min_visits=8):
    """Median best-slot overlap of settled tokens against mature slots,
    inverted through ov(n) = 1/sqrt((1+s)(1+s/n)). nvis isn't retained
    after perceive, so slot maturity is proxied by count (confident-active
    frames): a slot alive for >= min_visits*HOLD confident frames has
    pooled at least ~min_visits tokens; its residual 1/sqrt(1+s/n) factor
    is folded in with n = count/HOLD."""
    used = np.where(org.used)[0]
    if len(used) == 0:
        return None
    xi = org.xi[used]
    nv = np.maximum(org.count[used] / HOLD, 1.0)
    ov = np.abs(toks.conj() @ xi.T) / N          # tokens x slots
    best = np.argmax(ov, axis=1)
    o = ov[np.arange(len(toks)), best]
    n = nv[best]
    mature = n >= min_visits
    if mature.sum() < 20:
        return None
    # per-slot medians, then the upper quartile across slots: contaminated
    # or mixed slots depress their own median (biasing s up); the cleanest
    # slots are the best estimate of the true same-word overlap law
    slots = np.unique(best[mature])
    per_slot_o, per_slot_n = [], []
    for s_ in slots:
        sel = mature & (best == s_)
        if sel.sum() >= 3:
            per_slot_o.append(np.median(o[sel]))
            per_slot_n.append(np.median(n[sel]))
    if len(per_slot_o) < 5:
        return None
    order = np.argsort(per_slot_o)
    top = order[int(0.75 * len(order)):]
    med_o = float(np.median(np.array(per_slot_o)[top]))
    med_n = float(np.median(np.array(per_slot_n)[top]))
    # invert 1/sqrt((1+s)(1+s/n)) = med_o with n = med_n: quadratic in s
    # (1+s)(1+s/n) = 1/med_o^2  ->  s^2/n + s(1+1/n) + 1 - 1/med_o^2 = 0
    a = 1.0 / med_n
    b = 1.0 + 1.0 / med_n
    c = 1.0 - 1.0 / med_o ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    return max((-b + np.sqrt(disc)) / (2 * a), 0.0)


def self_consistent_s(fr_cal, s0=0.0, iters=10, tol=0.05):
    """Failure to measure is itself a signal: if pooling can't engage at the
    current guess (no mature slots -> measure_s None), the bars are too
    tight for the actual noise, so ESCALATE s and retry rather than give up.
    Found necessary at sigma >= 0.2: the s0=0 bars admit nothing, and the
    original break-on-None died at iteration zero (first run's honest
    partial; kept here per the negative-results rule)."""
    toks = settle_tokens(fr_cal)
    s = s0
    hist = [s]
    for _ in range(iters):
        org = run_pool(fr_cal, s)
        s_new = measure_s(org, toks)
        if s_new is None:
            s = s * 2.0 + 0.5          # too-tight bars: escalate and retry
            hist.append(s)
            continue
        hist.append(s_new)
        if abs(s_new - s) <= tol * max(s_new, 1e-9):
            s = s_new
            break
        s = s_new
    return s, hist


# ------------------------------------------------------------- evaluation
def coverage(org):
    states = np.array([normalize(emb_c[w], NORM) for w in range(V)])
    used = np.where(org.used)[0]
    if len(used) == 0:
        return 0.0, 0
    ovr = np.abs(org.xi[used].conj() @ states.T) / N
    w2s = {w: int(used[ovr[:, w].argmax()]) for w in range(V)}
    s2w = {int(used[i]): int(ovr[i].argmax()) for i in range(len(used))}
    cov = float(np.mean([s2w[w2s[w]] == w for w in range(V)]))
    return cov, len(used)


if __name__ == "__main__":
    print("PHASE 32: self-consistent calibration at V=60 > N=30 (rank-saturated)\n")
    fr_cal = frames(sample_stream(800, seed=7), 0.0)  # placeholder; rebuilt per sigma

    results_ok = []
    for sigma in [0.1, 0.2, 0.3]:
        s_true = sigma ** 2 * N
        fr_cal = frames(sample_stream(800, seed=7), sigma)
        fr_eval = frames(sample_stream(4000, seed=99), sigma)

        s_cal, hist = self_consistent_s(fr_cal)
        one_pass = hist[1] if len(hist) > 1 else float('nan')
        rel = abs(s_cal - s_true) / max(s_true, 1e-9)
        print(f"sigma={sigma}: s_true={s_true:.2f}  s_cal={s_cal:.2f} "
              f"(rel err {rel:.0%})  trajectory {[f'{h:.2f}' for h in hist]}")

        org_or = run_pool(fr_eval, s_true)
        org_ca = run_pool(fr_eval, s_cal)
        cov_or, n_or = coverage(org_or)
        cov_ca, n_ca = coverage(org_ca)
        print(f"          coverage: oracle-bars {cov_or:.2f} ({n_or} slots)  "
              f"calibrated-bars {cov_ca:.2f} ({n_ca} slots)")
        print(f"          ablation: one-pass s={one_pass:.2f} vs converged {s_cal:.2f}")
        results_ok.append(rel <= 0.30 and cov_ca >= cov_or - 0.05)

    print("\nverdict:",
          "SELF-CONSISTENT CALIBRATION WORKS at V >= N: the organism measures "
          "its own noise level by trying to remember -- the phase-26 caveat is closed"
          if all(results_ok) else
          "partial -- see failing sigmas above; the V >= N arm stays open")
