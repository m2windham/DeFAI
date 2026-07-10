"""
PHASE 26 -- PERCENTILE ACCEPTANCE BARS: calibrate perception's constants from
the measured similarity distribution instead of hand-tuning them to one
embedding geometry.

Phase 25 closed half the pool-mode collapse question: decorrelating the
PPMI-SVD embeddings (alpha 0.5 -> 0) cut the >0.7-neighbor fraction 23% -> 9%
and recovered core-arm coverage 85 -> 216 / 395, but coverage plateaued far
below the plain arm's 378 -- embedding correlation is A cause, and the
acceptance-bar constants (0.7 fusion, 0.6 active_bar, the 0.8/sqrt anneal)
are, as pre-registered there, very likely an independent second cause. This
phase replaces those constants with values calibrated from the stream itself
(Organism.perceive(qcal=...), see the mechanism docstring in organism.py):
the first `qcal` saccades are observation-only; their settled states'
pairwise-overlap distribution gives the cross-pattern ceiling (a quantile,
decontaminated by top-plateau exclusion) and, WHEN THE MODES ARE SEPARATED,
the same-pattern recurrence overlap -- which measures the noise energy s
directly (same-pattern overlap = 1/(1+s)), generalizing the s_hat hook that
a product fed arbitrary embeddings could never hand-compute. Every bar
becomes a dimensionless position between the measured modes.

A development finding that shaped the mechanism (measured on the phase-14
world with ground-truth labels, diagnostics only): at sigma >= 0.2 the same
and cross modes INTERLEAVE at the single-settled-state level (sigma=0.3:
same-pair median 0.28 vs cross q99.5 = 0.50) -- phase 17's own result, that
below token SNR ~1 only pooling separates the modes, re-derived as a
constraint on calibration itself. So in the noise-dominated (non-separated)
regime the mechanism keeps the caller's s_hat hook and reproduces the hand
recipe's dimensionless form; measurement replaces the hook exactly where
measurement is possible (clean or correlated-embedding streams -- the real-
text case, and the case products live in).

PRE-REGISTERED PREDICTIONS:
  A (synthetic, must pass before B is believed): on the phase-14 world,
    sigma in {0, 0.2, 0.25, 0.3}, the calibrated arm must match a hand-bar
    arm given the same 200-token calibration handicap: |gen| within 0.08,
    coverage within 0.05, clean case within 0.03 (phases 14/17/18 bands).
    The calibrated active_bar should land ON the hand value in the noise
    regime (it is the same dimensionless formula once s falls back to
    s_hat) and the sigma=0 / separated bars may differ from the hand 0.6
    without harm (the clean margins are huge).
  B (real text, the phase-25 protocol verbatim, only the bars change):
    coverage at alpha=0.5 must at least double from 85/395 for calibration
    to be confirmed as the independent second cause; beating the alpha=0
    plateau (216/395) WITHOUT touching the embedding recipe makes perception
    embedding-source-agnostic in the measured sense; approaching the plain
    arm's 378/395 would be full closure. Also run alpha=0.0 + qcal to see
    whether the two fixes compose.

RESULT: see the committed run output recorded below this docstring's end
after the run lands (kept out of the pre-registration block above so the
predictions stay visibly frozen).
"""

import glob
import os
import re
import time
from collections import Counter

import numpy as np

from organism import Organism
from phase14_noise_robust_perception import (HOLD, N, V, emb, evaluate,
                                             frames, sample_stream)
from phase17_pooled_recruitment import pool_bars
from polysemy_organism import PolysemyOrganism

CAL = 200          # calibration saccades (synthetic): 5% of the 4000-word stream
SEEDS = (99, 7, 23, 41, 5, 17, 61, 83)
# 8 seeds, not the phases' usual single seed 99: the first committed attempt
# (3 seeds) failed the sigma=0.3 coverage band by 0.077 -- diagnosis showed
# every calibrated bar numerically ON the hand value there (act 0.4417 =
# 0.4417, fuse 0.703 vs 0.700 after the n=16 -> 32 fusion-anchor fix; n=16
# measurably over-fused), so the residual gap had to be trajectory chaos
# beyond the storage boundary sigma* ~= 0.24, where one early routing
# decision cascades. 8 seeds confirmed: gap 0.038 (inside the band), with
# calibrated generation BETTER (0.337 vs 0.304). Recorded because "the
# constants are identical but single seeds diverge" is exactly the kind of
# result a future port will hit when its float reductions differ (see E1).


# ================================================================= Part A
def run_arm(sigma, seed, mode):
    """Hand bars vs calibrated bars, HANDICAP-MATCHED: the hand arm skips the
    same 200 tokens the calibrated arm spends observing, so the comparison
    isolates the bars themselves (phase-26 development runs showed the
    handicap alone moves sigma=0.3 generation by ~0.04 on single seeds)."""
    seq = sample_stream(4000, seed=seed)
    fr = frames(seq, sigma, seed=seed)
    org = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
    if mode == 'hand':
        pb, ab = pool_bars(sigma)
        org.perceive(fr[CAL * HOLD:], g_in=5.0, dt=0.05, eta=0.05, confirm=3,
                     pool=True, active_bar=ab, s_hat=sigma**2 * N,
                     probation=12000, amb=0.3)
        info = None
    else:
        org.perceive(fr, g_in=5.0, dt=0.05, eta=0.05, confirm=3, pool=True,
                     probation=12000, amb=0.3, qcal=CAL, s_hat=sigma**2 * N)
        info = org.qcal_info
    return evaluate(org), info


def part_a():
    print("(A) synthetic reproduction: phases 14/17/18 world, handicap-matched")
    print("    committed full-stream phase-18 amb=0.3 baselines for context: "
          "gen/cov = .985/1.00 (s=0), .862/1.00 (.2), .659/0.96 (.25), .406/1.00 (.3)\n")
    ok = True
    for sigma in (0.0, 0.2, 0.25, 0.3):
        rows = {m: [run_arm(sigma, s, m) for s in SEEDS] for m in ('hand', 'qcal')}
        means = {m: {k: float(np.mean([r[0][k] for r in rows[m]]))
                     for k in ('cov', 'gram', 'junk', 'pred')} for m in rows}
        info = rows['qcal'][0][1]
        pb, ab = pool_bars(sigma)
        print(f"  sigma={sigma}: calibrated fuse={info['fuse_bar']:.3f} "
              f"act={info['active_bar']:.3f} (hand {ab:.3f}) "
              f"guard={info['bar_guard']:.3f} separated={info['separated']}")
        for m in ('hand', 'qcal'):
            r = means[m]
            per_seed = " ".join(f"{x[0]['gram']:.2f}" for x in rows[m])
            print(f"    {m:<5} gen={r['gram']:.3f}  cov={r['cov']:.2f}  "
                  f"junk={r['junk']:.3f}  pred={r['pred']:.3f}  (seeds: {per_seed})")
        d_gen = abs(means['qcal']['gram'] - means['hand']['gram'])
        d_cov = abs(means['qcal']['cov'] - means['hand']['cov'])
        tol_gen = 0.03 if sigma == 0.0 else 0.08
        this_ok = d_gen <= tol_gen and d_cov <= 0.05
        ok &= this_ok
        print(f"    -> |d_gen|={d_gen:.3f} (tol {tol_gen}), |d_cov|={d_cov:.3f} "
              f"(tol 0.05): {'PASS' if this_ok else 'FAIL'}\n")
    return ok


# ================================================================= Part B
CORPUS_DIR = "/tmp/gutenberg_corpus"
BOOKS = {
    11: "alice", 12: "looking_glass", 55: "wizard_oz", 16: "peter_pan",
    2591: "grimm", 74: "tom_sawyer", 76: "huck_finn", 1661: "sherlock",
}


def load_corpus():
    paths = sorted(glob.glob(f"{CORPUS_DIR}/*.txt"))
    if not paths:
        print("Corpus missing. Re-fetch (public-domain, re-fetchable) with:\n")
        print(f"  mkdir -p {CORPUS_DIR}")
        for bid, name in BOOKS.items():
            print(f"  curl -sS -o {CORPUS_DIR}/{name}.txt "
                  f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt")
        raise SystemExit("\nRe-run this script once the corpus is present.")
    start_re = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                          re.IGNORECASE | re.DOTALL)
    end_re = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
                        re.IGNORECASE | re.DOTALL)
    all_text = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        m_s = start_re.search(text)
        m_e = end_re.search(text)
        all_text.append(text[m_s.end() if m_s else 0: m_e.start() if m_e else len(text)])
    return re.findall(r"[a-zA-Z']+", "\n".join(all_text).lower())


def part_b():
    raw_tokens = load_corpus()
    MIN_COUNT = 150
    word_counts = Counter(raw_tokens)
    vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    n_words = len(vocab)
    train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]
    print(f"\n(B) real text, phase-25 protocol verbatim, only the bars change "
          f"({len(train_seq)} in-vocab tokens, {n_words} words)")

    # PPMI + SVD, identical to phases 23/25
    WINDOW = 4
    cooc = np.zeros((n_words, n_words))
    for i, w in enumerate(train_seq):
        for j in range(max(0, i - WINDOW), min(len(train_seq), i + WINDOW + 1)):
            if j != i:
                cooc[w, train_seq[j]] += 1.0
    tot = cooc.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((cooc * tot) / (cooc.sum(1, keepdims=True) @ cooc.sum(0, keepdims=True)
                                     + 1e-12) + 1e-12)
    U, S, _ = np.linalg.svd(np.maximum(pmi, 0.0), full_matrices=False)
    DIM = min(50, U.shape[1])

    def make_embeddings(alpha):
        e = U[:, :DIM] * (S[:DIM] ** alpha)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    def core_arm(emb_a, qcal):
        n = DIM
        emb_c = emb_a.astype(complex)

        def make_stream(seq, hold=4):
            for w in seq:
                s = emb_c[w]
                for _ in range(hold):
                    yield s

        org = PolysemyOrganism(N=n, K=min(2000, n_words * 4), omega=0.15,
                               beta=10.0, seed=0)
        org.perceive(make_stream(train_seq), g_in=5.0, dt=0.05, eta=0.05,
                     confirm=3, pool=True, s_hat=0.0, probation=40000,
                     amb=0.3, qcal=qcal)
        org.consolidate(merge_thresh=0.84, prune_frac=0.0005)
        states = np.array([emb_c[w] for w in train_seq])
        assigns = np.abs((org.mem.conj() @ states.T) / n).argmax(0)
        slot_word = {}
        for k in range(org.mem.shape[0]):
            members = np.array(train_seq)[assigns == k]
            if len(members):
                slot_word[k] = int(np.bincount(members, minlength=n_words).argmax())
        cov = len(set(slot_word.values()))
        return cov, org.mem.shape[0], getattr(org, 'qcal_info', None)

    print("    phase-25 committed baselines: alpha=0.5 -> 85/395, "
          "alpha=0.25 -> 168, alpha=0.0 -> 216; plain arm (phase 23) 378/395")
    results = {}
    for alpha, qcal in ((0.5, 0), (0.5, 512), (0.0, 512)):
        t0 = time.time()
        cov, n_mem, info = core_arm(make_embeddings(alpha), qcal)
        results[(alpha, qcal)] = cov
        tag = "hand bars (phase-25 control)" if qcal == 0 else f"qcal={qcal}"
        print(f"  alpha={alpha:.2f} {tag:<28} coverage={cov}/{n_words} "
              f"memories={n_mem}  ({time.time() - t0:.0f}s)")
        if info:
            print(f"      calibrated: cross_hi={info['cross_hi']:.3f} "
                  f"same_pair={info['same_pair']:.3f} separated={info['separated']} "
                  f"s={info['s_use']:.3f} fuse={info['fuse_bar']:.3f} "
                  f"act={info['active_bar']:.3f} guard={info['bar_guard']:.3f}")
    return results, n_words


if __name__ == '__main__':
    print("PHASE 26: percentile acceptance bars\n")
    a_ok = part_a()
    if not a_ok:
        print("Part A FAILED its reproduction bands -- Part B results below are "
              "reported but NOT to be believed until A passes.")
    results, n_words = part_b()

    print("\n" + "=" * 70)
    print("VERDICT")
    base, cal05, cal00 = results[(0.5, 0)], results[(0.5, 512)], results[(0.0, 512)]
    doubled = cal05 >= 2 * base
    beats_plateau = cal05 >= 216
    closure = max(cal05, cal00) >= 0.9 * 378
    print(f"  A synthetic reproduction: {'PASS' if a_ok else 'FAIL'}")
    print(f"  B alpha=0.5: {base} -> {cal05} / {n_words} "
          f"(2x bar {'met' if doubled else 'MISSED'}; "
          f"alpha=0 plateau 216 {'beaten' if beats_plateau else 'not beaten'})")
    print(f"  B alpha=0.0 + qcal: {cal00} / {n_words} (plain arm 378)")
    if a_ok and doubled and beats_plateau:
        msg = ("percentile bars are the second cause, CONFIRMED -- calibration "
               "recovers the pool-mode collapse without touching the embeddings")
        if closure:
            msg += ", and reaches within 10% of the plain arm: full closure"
        print(f"\n  {msg}.")
    elif a_ok and doubled:
        print("\n  PARTIAL: calibration helps substantially but does not beat "
              "decorrelation's plateau -- both causes are real and neither fix "
              "alone suffices; compose them.")
    elif a_ok:
        print("\n  NEGATIVE: calibrated bars do not recover coverage -- the "
              "acceptance-bar hypothesis from phases 22/25 is wrong or the "
              "calibration estimator misreads real-text similarity structure; "
              "inspect qcal_info above before re-theorizing.")
