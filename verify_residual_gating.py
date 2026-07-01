"""
PHASE 8 THEOREM VERIFICATION -- why additive context blending cannot split a
same-word slot, and why word-conditional residual gating can.

This preserves (in-repo) the analysis behind the phase-8 negative result.
Phase 8 found that context-biased perceive (z_store = normalize(z + alpha*ctx))
never recruits a second slot for a dual-role word at any usable alpha. That is
not a tuning failure -- it is analytic:

THEOREM (additive gate ceiling).
  Let w be a word vector and c1, c2 two context vectors orthogonal to w and to
  each other, all with equal norm. For composites v_i = normalize(w + alpha*c_i),
      |<v1, v2>| / |v|^2  =  1 / (1 + alpha^2).
  The recruitment gate fires when this overlap drops below the recruit
  threshold 0.5, which requires alpha > 1 -- exactly where the word component
  is no longer the majority of the vector and word identity collapses
  (phase 8 confirmed empirically: alpha=1.0 destroys the memories).
  Corollary: two same-word/different-context slots overlap at ~0.89 for
  alpha=0.35, above the 0.84 merge threshold, so consolidate() folds any
  split back together.

FIX (residual gating).
  Compare same-word patterns in the word-conditional context residual:
      r = v - vdot(w_hat, v) * w_hat        (np.vdot conjugates its first
                                             argument, matching the overlaps()
                                             convention), then L2-normalize.
  Distinct contexts give residual overlap ~0 (novelty ~1 -> recruits); the
  same context gives residual overlap ~1 (novelty ~0 -> updates) -- and both
  are INDEPENDENT of alpha, so the word component can stay dominant.

This script verifies both claims numerically, with clean orthogonal contexts
and with noisy ones, and exits nonzero on failure so it can serve as a check.
"""

import sys
import numpy as np

N = 30
rng = np.random.default_rng(0)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def orthogonalize(v, others):
    for o in others:
        v = v - np.vdot(o, v) / np.vdot(o, o) * o
    return unit(v)


def residual(v, w_hat):
    r = v - np.vdot(w_hat, v) * w_hat
    return unit(r)


def run_case(noise, label):
    """noise: std of complex perturbation added to each composite's context."""
    w = unit(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    c1 = orthogonalize(rng.standard_normal(N) + 1j * rng.standard_normal(N), [w])
    c2 = orthogonalize(rng.standard_normal(N) + 1j * rng.standard_normal(N), [w, c1])

    print(f"\n--- {label} ---")
    print(f"{'alpha':>6} {'overlap':>8} {'theory':>7} {'resid distinct':>15} {'resid same':>11}")
    ok = True
    for a in [0.35, 0.5, 0.7, 1.0]:
        def composite(c):
            eps = noise * unit(rng.standard_normal(N) + 1j * rng.standard_normal(N))
            return unit(w + a * (unit(c + eps)))

        v1, v2, v1b = composite(c1), composite(c2), composite(c1)
        ov = abs(np.vdot(v1, v2))
        theory = 1 / (1 + a ** 2)
        r1, r2, r1b = (residual(v, w) for v in (v1, v2, v1b))
        rd = abs(np.vdot(r1, r2))     # distinct contexts -> want ~0
        rs = abs(np.vdot(r1, r1b))    # same context      -> want ~1
        print(f"{a:>6} {ov:>8.3f} {theory:>7.3f} {rd:>15.3f} {rs:>11.3f}")
        ok &= abs(ov - theory) < 0.05      # composite overlap matches theorem
        ok &= ov > 0.5 or a >= 1           # additive gate cannot fire below alpha=1
        ok &= rd < 0.35 and rs > 0.85      # residual gate separates, any alpha
    return ok


ok_clean = run_case(noise=0.0, label="clean orthogonal contexts (theorem exact)")
ok_noisy = run_case(noise=0.3, label="noisy contexts (realistic attractors)")

print("\nverdict:", "PASS -- additive gate provably stuck; residual gate separates, alpha-independent"
      if (ok_clean and ok_noisy) else "FAIL -- claims do not reproduce, re-derive before Track B")
sys.exit(0 if (ok_clean and ok_noisy) else 1)
