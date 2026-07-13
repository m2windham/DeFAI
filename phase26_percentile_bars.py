"""
PHASE 26: PERCENTILE ACCEPTANCE BARS -- make perception's absolute constants
measured quantities.

The problem (from phase 25's diagnosis): the pool-mode perception constants
are absolute numbers tuned on near-orthogonal synthetic embeddings -- 0.7
fusion, 0.6 active_bar, the 0.85/sqrt(1+sigma^2 N) bar map, and the s_hat
noise energy handed to perceive() a priori. Phase 25 measured that even
fully decorrelated real-text embeddings leave coverage at 216/395, and
pre-registered these constants as the likely second cause. A Path B product
cannot be handed sigma or hand-tuned bars -- it doesn't control its input
embeddings.

The mechanism (this phase): CALIBRATE THE BARS FROM THE MEASURED SIMILARITY
DISTRIBUTION, label-free, on a short stream prefix:

  - settle a field over the prefix exactly as perceive() does; at each input
    saccade the field state is a settled token. Collect a buffer of them.
  - CROSS-word similarity distribution, for free and label-free: consecutive
    settled tokens are different words BY CONSTRUCTION -- a saccade means
    the word changed. |overlap(tok_t, tok_{t+1})| samples the cross mode
    with zero labels.
  - SAME-word mode: each token's best match among non-adjacent buffer
    entries. Real words recur; the median best-match overlap m1 estimates
    the same-word token-token overlap 1/(1+s), which inverts to a measured
    noise energy s_cal = 1/m1 - 1 (the quantity perceive()'s annealed bar
    formula needs as s_hat -- the roadmap's "generalize the s_hat hook").
  - bars from the two measured modes:
      active_bar = midpoint(q95(cross), 1/sqrt(1+s_cal))
                   -- between the cross ceiling and the expected token-vs-
                      denoised-memory overlap;
      fuse_bar   = midpoint(1, q95(cross)/m1)
                   -- both cross and same token overlaps are attenuated by
                      the same (1+s), so their RATIO estimates the CLEAN
                      cross-similarity ceiling that denoised duplicate
                      memories must exceed to be called the same pattern
                      (the old hardcoded 0.7); ratio clamped to 0.9 as a
                      sanity bound (a fuse bar above 0.95 would block all
                      fusion -- documented constant, pathology guard only);
      s_hat      = s_cal.

Pre-registered success criterion: the calibrated bars must reproduce the
phase 14/17/18 headline behaviors (coverage/junk at sigma = 0, 0.2, 0.3)
within the regression-harness tolerance of the ORACLE bars (pool_bars(sigma)
+ true s_hat = sigma^2 N). Not "better" -- the claim is that the constants
can be MEASURED rather than KNOWN, at no behavioral cost.

Ablation matrix (redundancy check, pre-registered): calibrating three
quantities (active_bar, s_hat, fuse_bar) might be over-redundant -- maybe
one measured quantity carries all the load. Arms:
  oracle    : pool_bars(sigma), s_hat=sigma^2 N, fuse 0.7   (the incumbent)
  cal-full  : all three calibrated
  cal-bar   : calibrated active_bar only (oracle s_hat, fuse 0.7)
  cal-shat  : calibrated s_hat only (oracle active_bar, fuse 0.7)
Gate mode (phase 14, no pooling) is checked separately with a calibrated
recruit/active_bar vs the 0.5/0.6 defaults.
"""

import numpy as np
from organism import Organism, normalize
from phase14_noise_robust_perception import emb, V, N, NORM, sample_stream, frames, evaluate
from phase17_pooled_recruitment import pool_bars


# ---------------------------------------------------------------- calibrate
def settle_tokens(fr, omega=0.15, g_in=5.0, dt=0.05, max_tokens=600, seed=0):
    """Run the same input-driven field dynamics perceive() uses over a frame
    stream and return the settled token states captured at input saccades."""
    r = np.random.default_rng(seed)
    z = normalize(r.standard_normal(N) + 1j * r.standard_normal(N), NORM)
    toks = []
    prev_x = None
    for x in fr:
        x = normalize(x, NORM)
        if prev_x is not None and np.linalg.norm(x - prev_x) > 0.5 * NORM:
            toks.append(z.copy())
            if len(toks) >= max_tokens:
                break
        z = normalize(z + dt * (1j * omega * z + g_in * (x - z)), NORM)
        prev_x = x
    return np.array(toks)


def _noise_floor_factor(n, dim, k, reps=20, seed=11):
    """Self-calibration by simulation: where does THIS estimator (mean of the
    bottom-k covariance eigenvalues, at aspect dim/n) place a KNOWN pure-noise
    floor? Pure isotropic Gaussian tokens have floor exactly 1/dim; the
    measured position divided by 1/dim is the finite-sample shrinkage factor
    to divide out of the data measurement. Uses no knowledge of the data."""
    r = np.random.default_rng(seed)
    fs = []
    for _ in range(reps):
        X = r.standard_normal((n, dim)) + 1j * r.standard_normal((n, dim))
        X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(dim)
        lam = np.sort(np.linalg.eigvalsh((X.conj().T @ X).real / (n * dim)))
        fs.append(np.mean(lam[:k]) * dim)
    return float(np.mean(fs))


def calibrate(fr, q=0.95, k_floor=2):
    """Measure the similarity structure of settled tokens and place every
    absolute perception constant from it. Label-free.

    Estimator history (documented per standing rules): the obvious same-word
    estimator -- median best-match overlap in the token buffer -- FAILS
    beyond sigma*: best-of-n matching is fluctuation-dominated exactly when
    the bars matter most (measured: m1=0.585 at sigma=0.3 where truth is
    0.27; the max over ~600 cross pairs with std 1/sqrt(N) beats the true
    same-word overlap). Single-pair statistics are information-insufficient
    at low SNR -- the same reason phase 17 had to pool evidence. The working
    estimator pools across the WHOLE buffer spectrally: with V < N the word
    covariance is rank-limited, so the bottom eigenvalues of the token
    covariance are pure noise floor; floor*N = s/(1+s) inverts to the noise
    energy. The estimator's own finite-sample bias is removed by simulating
    pure noise at the same (n, N) -- self-calibration, no data oracle.
    Documented limitation: needs vocabulary rank < N; on real text with
    V >> N the floor is signal-contaminated and this arm of the calibration
    must be re-derived (pre-registered as the open item for the real-text
    port)."""
    T = settle_tokens(fr)
    n = len(T)
    O = np.abs((T.conj() @ T.T) / N)
    cross = np.array([O[i, i + 1] for i in range(n - 1)])       # saccade => different word
    qc = float(np.quantile(cross, q))
    # spectral noise energy, self-calibrated
    C = (T.conj().T @ T).real / (n * N)
    lam = np.sort(np.linalg.eigvalsh(C)); lam = lam / lam.sum()
    c_fac = _noise_floor_factor(n, N, k_floor)
    noise_frac = float(np.clip(np.mean(lam[:k_floor]) * N / c_fac, 0.0, 0.95))
    s_cal = noise_frac / (1.0 - noise_frac)
    mem_ov = 1.0 / np.sqrt(1.0 + s_cal)                         # token vs denoised memory
    active_bar = 0.5 * (qc + mem_ov)
    # clean cross ceiling: subtract the (analytic) noise-induced spread from
    # the measured cross quantile, then undo the (1+s) attenuation
    noise_std = np.sqrt(s_cal * (2.0 + s_cal)) / ((1.0 + s_cal) * np.sqrt(N))
    c_hi = float(np.clip((qc - 1.645 * noise_std) * (1.0 + s_cal), 0.0, 0.85))
    fuse_bar = 0.5 * (1.0 + c_hi)
    return dict(q_cross=qc, s_cal=float(s_cal), mem_ov=mem_ov,
                active_bar=float(active_bar), fuse_bar=float(fuse_bar))


# ---------------------------------------------------------------- run matrix
def run_pool(sigma, amb, active_bar, s_hat, fuse_bar):
    org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    fr = frames(sample_stream(4000, seed=99), sigma)
    org.perceive(fr, g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                 active_bar=active_bar, s_hat=s_hat, probation=12000, amb=amb,
                 fuse_bar=fuse_bar)
    return evaluate(org)


def run_gate(sigma, recruit, active_bar):
    org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    fr = frames(sample_stream(4000, seed=99), sigma)
    org.perceive(fr, g_in=5.0, dt=0.05, eta=0.02, recruit=recruit, confirm=3,
                 active_bar=active_bar)
    return evaluate(org)


if __name__ == "__main__":
    print("PHASE 26: percentile acceptance bars from the measured similarity distribution\n")

    print(f"{'sigma':>6} {'mode':<10} {'active_bar':>10} {'s_hat':>7} {'fuse':>6}"
          f"  {'cov':>5} {'junk':>5} {'pred':>5} {'slots':>5}")
    results = {}
    for sigma, amb in [(0.0, 0.0), (0.2, 0.0), (0.3, 0.3)]:
        # calibration prefix: a SEPARATE stream (seed 7) -- the bars must not
        # be fit on the evaluation stream
        cal = calibrate(frames(sample_stream(800, seed=7), sigma))
        pb, ab_or = pool_bars(sigma)
        s_or = sigma ** 2 * N
        print(f"  [calibration at sigma={sigma}: s_cal={cal['s_cal']:.2f} "
              f"(true {s_or:.2f}), cross q95={cal['q_cross']:.3f}]")
        arms = {
            'oracle':   (ab_or,             s_or,          0.7),
            'cal-full': (cal['active_bar'], cal['s_cal'],  cal['fuse_bar']),
            'cal-bar':  (cal['active_bar'], s_or,          0.7),
            'cal-shat': (ab_or,             cal['s_cal'],  0.7),
        }
        for name, (ab, sh, fb) in arms.items():
            r = run_pool(sigma, amb, ab, sh, fb)
            results[(sigma, name)] = r
            print(f"{sigma:>6} {name:<10} {ab:>10.3f} {sh:>7.2f} {fb:>6.3f}"
                  f"  {r['cov']:>5.2f} {r['junk']:>5.2f} {r['pred']:>5.2f} {r['slots']:>5}")

    print("\ngate mode (phase 14, no pooling):")
    print(f"{'sigma':>6} {'mode':<10} {'recruit':>8} {'active_bar':>10}"
          f"  {'cov':>5} {'junk':>5} {'pred':>5} {'slots':>5}")
    gate_results = {}
    for sigma in [0.0, 0.2]:
        cal = calibrate(frames(sample_stream(800, seed=7), sigma))
        for name, (rc, ab) in {'oracle': (0.5, 0.6),
                               'cal': (cal['active_bar'], cal['active_bar'])}.items():
            r = run_gate(sigma, rc, ab)
            gate_results[(sigma, name)] = r
            print(f"{sigma:>6} {name:<10} {rc:>8.3f} {ab:>10.3f}"
                  f"  {r['cov']:>5.2f} {r['junk']:>5.2f} {r['pred']:>5.2f} {r['slots']:>5}")

    # ------------------------------------------------------------- verdict
    # pre-registered: cal-full within harness tolerance of oracle
    tol_cov, tol_junk = 0.05, 0.10
    ok = all(
        results[(s, 'cal-full')]['cov'] >= results[(s, 'oracle')]['cov'] - tol_cov and
        results[(s, 'cal-full')]['junk'] <= results[(s, 'oracle')]['junk'] + tol_junk
        for s in [0.0, 0.2, 0.3]
    )
    gate_ok = all(
        gate_results[(s, 'cal')]['cov'] >= gate_results[(s, 'oracle')]['cov'] - tol_cov
        for s in [0.0, 0.2]
    )
    print("\nablation reading: compare cal-bar / cal-shat rows to cal-full --")
    print("if a single calibrated quantity matches cal-full, the others are redundant.")
    print("\nverdict:",
          "PERCENTILE BARS WORK: measured bars reproduce oracle behavior at all "
          "sigmas -- perception no longer needs to be HANDED its constants"
          if ok and gate_ok else
          "partial -- calibrated arm(s) fall outside tolerance; see rows above")
