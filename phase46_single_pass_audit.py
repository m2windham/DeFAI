"""
PHASE 46 (T7.4) -- SINGLE-PASS AUDIT.

WHAT THIS ASKS, AND WHY IT IS AN AUDIT. "Single-pass online learning" is one
of the capability axes any re-scope of the RELEASE HOLD's cost branch would
stand on (33h's owner-decision paragraph names it first). But the recipe
that produced the language-track headline numbers runs FIFTEEN epochs over
the same corpus (phase 19, inherited by 20/21/22/23). A re-scoped gate must
not rest on an axis the production recipe violates. So: re-measure the
headline results at ONE epoch against the multi-epoch numbers, on paired
seeds, and report which survive.

THIS IS NOT AN IMPROVEMENT TARGET. A large drop is a legitimate and
important finding -- arguably the more important one, because it would
mean the wording in NOVELTY has to change rather than the mechanism.

Three axes, each measured the way the project already measures it:
  COVERAGE            distinct words that own a consolidated memory
                      (phase 19's own metric, its own consolidate settings)
  CATEGORY VALIDITY   phase 24's certificate -- class-bigram mutual
                      information against a label-permutation null at the
                      same partition sizes
  POLYSEMY DETECTION  phase 12/21's predictive split gain over emergent
                      categories, against phase 21's directly measured
                      noise floor (the 99th percentile of a matched-n null)

======================================================================
PRE-REGISTRATION (committed before the run that tests it)
======================================================================

P1 -- COVERAGE IS THE AXIS THAT PAYS. One epoch loses at least 10
percentage points of word coverage against 15 on the fables recipe.
Mechanism, stated so the prediction is falsifiable rather than vague:
`recruit` is a SIMILARITY FLOOR, so a rare word is absorbed into a
near-enough existing slot unless it recurs often enough to be seen while
no slot is near enough -- which is exactly what repeated exposure buys.
  MUNDANE ACCOUNT (M1): multi-epoch is doing ordinary optimization work
  that ANY online learner would need, and nothing about the mechanism is
  implicated. The honest claim would then be "single-pass CAPABLE,
  multi-pass TUNED", and the fix is a wording change in NOVELTY rather
  than an experiment. This phase cannot distinguish "ordinary optimization"
  from "a mechanism that needs revisiting" on its own -- what it CAN do is
  say which axes need the epochs, which is the input that wording change
  requires.

P2 -- CATEGORY VALIDITY SURVIVES ONE PASS. At 1 epoch the class-bigram MI
still clears its permutation null with z > 3 on both corpora. Reason to
expect it: categories are read off TRANSITION COUNTS, which accumulate
within a single pass at full rate, whereas coverage depends on the
recruitment race.

P3 -- POLYSEMY DETECTION IS CORPUS-BOUND, NOT EPOCH-BOUND. Phase 19
measured that the predictive-gain statistic needs n = 200-500 occurrences
per candidate. Epochs do not manufacture occurrences of DISTINCT contexts
-- re-reading the same corpus multiplies n while leaving the context
distribution identical -- so the count of words clearing the noise floor
should be roughly EPOCH-INVARIANT once the vocabulary bar is met, and the
small corpus should fail at every epoch count for lack of tokens rather
than for lack of passes. Baseline check before predicting: this is the
same distinction the n-gram literature draws between token count and type
coverage, and it is why re-reading a corpus does not improve a bigram
model's perplexity on held-out text.
  DECISION RULE: if the count of words above the noise floor at 1 epoch is
  within +-20% of the 15-epoch count on gutenberg8, detection is
  single-pass and P3 holds. If it rises materially with epochs, then
  epochs ARE manufacturing signal on a fixed corpus, which would mean the
  statistic is reading the organism's convergence rather than the
  language's structure -- a much more serious finding, and it must be
  reported as the headline if it fires.

P4 -- WHAT THE AUDIT DECIDES. An axis is "single-pass" iff its 1-epoch
value either sits inside the paired-seed spread of the 15-epoch value, or
still clears its own measured null. Every axis that fails is re-worded in
NOVELTY as multi-pass tuned, in the same commit. Nothing here is allowed
to conclude "single-pass online" as a blanket capability unless all three
axes pass.
"""

import time

import numpy as np

import text_recipe as tr
from phase45_settling_depth import (assign_slots, cat_bigram,
                                    mutual_information, slot_labels)
from polysemy_organism import PolysemyOrganism, _entropy

EPOCHS = (1, 3, 15)
SEEDS = (0, 1, 2)
K_CATS = 3                       # phase 21's convention for the gain analysis
OMEGA, BETA = 0.15, 10.0


def entropy_np(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


def predictive_gains(train_seq, word_cat, V, min_occ=100):
    """Phase 12/21's predictive split gain per word, over emergent
    categories. `word_cat[w]` is w's category or None."""
    seq = np.asarray(train_seq)
    cat = np.array([word_cat.get(int(w), -1) for w in range(V)], int)
    cats = cat[seq]
    out = []
    for w in range(V):
        occ = np.where(seq == w)[0]
        occ = occ[(occ > 0) & (occ < len(seq) - 1)]
        if occ.size == 0:
            continue
        s, p = cats[occ + 1], cats[occ - 1]
        keep = (s >= 0) & (p >= 0)
        s, p = s[keep], p[keep]
        n = s.size
        if n < min_occ:
            continue
        mx = int(cat.max()) + 1
        uncond = _entropy(np.bincount(s, minlength=mx))
        cond = 0.0
        for pc in np.unique(p):
            i = p == pc
            cond += i.sum() / n * _entropy(np.bincount(s[i], minlength=mx))
        out.append((w, n, max(uncond - cond, 0.0)))
    return out


def noise_floor(ns, k_cats, draws=3000, seed=5):
    """Phase 21's directly measured floor: matched-n random category
    assignments, 99th percentile of the resulting gains."""
    rng = np.random.default_rng(seed)
    if not ns:
        return float("inf")
    vals = []
    for _ in range(draws):
        n = int(rng.choice(ns))
        s = rng.integers(0, k_cats, size=n)
        p = rng.integers(0, k_cats, size=n)
        uncond = entropy_np(np.bincount(s, minlength=k_cats))
        cond = 0.0
        for pc in np.unique(p):
            i = p == pc
            cond += i.sum() / n * entropy_np(np.bincount(s[i], minlength=k_cats))
        vals.append(max(uncond - cond, 0.0))
    return float(np.percentile(vals, 99))


def run_pipeline(R, epochs, seed, K, n_null=200):
    """Phase 19's recipe at `epochs` passes: perceive -> consolidate ->
    coverage -> categories -> predictive gain. Returns the three axes."""
    V = len(R.vocab)
    org = PolysemyOrganism(N=R.N, K=K, omega=OMEGA, beta=BETA, seed=seed,
                           backend="numba")
    stream = R.stream(tr.HOLD_19)
    for _ in range(epochs):
        org.perceive(stream, **tr.PERCEIVE_19)
    used = int(org.used.sum())
    org.consolidate(merge_thresh=0.84, prune_frac=0.001)
    n_mem = org.mem.shape[0]
    states = R.embeddings[np.asarray(R.train_seq)]
    assign = assign_slots(org.mem, states, R.N)
    slot_word = {}
    for k in range(n_mem):
        members = np.asarray(R.train_seq)[assign == k]
        if len(members):
            slot_word[k] = int(np.bincount(members, minlength=V).argmax())
    coverage = len(set(slot_word.values()))

    # -- category validity (phase 24's certificate), free k selection ------
    org.discover_categories_v2(verbose=False)
    labels = slot_labels(org, n_mem)
    k = int(labels.max()) + 1
    pred_arr, succ_arr = assign[:-1], assign[1:]
    mi = mutual_information(cat_bigram(labels, k, pred_arr, succ_arr))
    rng = np.random.default_rng(seed)
    null = np.array([mutual_information(
        cat_bigram(rng.permutation(labels), k, pred_arr, succ_arr))
        for _ in range(n_null)])
    mi_z = float((mi - null.mean()) / (null.std() + 1e-12))
    mi_p99 = float(np.percentile(null, 99))

    # -- polysemy detection (phase 21's convention: k = 3) -----------------
    org.discover_categories_v2(k_range=[K_CATS], verbose=False)
    word_cat = {}
    for slot, w in slot_word.items():
        c = org.word_slot_to_cat.get(slot)
        if c is not None:
            word_cat[w] = int(c)
    gains = predictive_gains(R.train_seq, word_cat, V)
    p99 = noise_floor([n for _, n, _ in gains], K_CATS)
    above = [g for g in gains if g[2] > p99]
    return dict(used=used, mem=n_mem, coverage=coverage, V=V, k=k, mi=mi,
                mi_p99=mi_p99, mi_z=mi_z, n_cand=len(gains),
                n_above=len(above), floor=p99,
                best=max([g[2] for g in gains], default=0.0))


if __name__ == "__main__":
    t_all = time.time()
    print("PHASE 46 (T7.4): single-pass audit -- 1 epoch vs the 15-epoch recipe\n")

    corpora = [("fables (phase 19)", tr.load_fables, tr.RECIPE_19),
               ("gutenberg8 (phase 20)", tr.load_gutenberg8, tr.RECIPE_20)]
    results = {}
    for name, loader, rec in corpora:
        print("=" * 70)
        print(f"CORPUS: {name}")
        R = tr.Recipe(loader(verbose=False) if loader is tr.load_gutenberg8
                      else loader(), **rec, verbose=True)
        K = min(1200, len(R.vocab) * 4)
        eps = EPOCHS if name.startswith("fables") else (1, 3)
        print(f"  K={K}  seeds={SEEDS}  epochs={eps}"
              + ("   (15 epochs omitted on this corpus: 3.3M frames/epoch)"
                 if not name.startswith("fables") else ""))
        print(f"  {'epochs':>6} {'seed':>4} | {'slots':>6} {'mem':>5} "
              f"{'coverage':>12} | {'k':>3} {'MI':>7} {'p99':>7} {'MIz':>7} | "
              f"{'cands':>6} {'above':>6} {'floor':>7} {'best':>7}")
        for e in eps:
            for s in SEEDS:
                r = run_pipeline(R, e, s, K)
                results[(name, e, s)] = r
                print(f"  {e:>6} {s:>4} | {r['used']:>6} {r['mem']:>5} "
                      f"{r['coverage']:>5}/{r['V']:<6} | {r['k']:>3} "
                      f"{r['mi']:>7.4f} {r['mi_p99']:>7.4f} {r['mi_z']:>7.1f} | "
                      f"{r['n_cand']:>6} {r['n_above']:>6} {r['floor']:>7.4f} "
                      f"{r['best']:>7.4f}")

        def agg(e, key):
            v = [results[(name, e, s)][key] for s in SEEDS]
            return float(np.mean(v)), float(np.min(v)), float(np.max(v))

        lo, hi = eps[0], eps[-1]
        print(f"\n  paired summary, {lo} epoch vs {hi} epochs (seeds {SEEDS}):")
        for key, label in (("coverage", "word coverage"),
                           ("mi_z", "category MI z-score"),
                           ("n_above", "words above the noise floor")):
            a, amin, amax = agg(lo, key)
            b, bmin, bmax = agg(hi, key)
            print(f"    {label:<28} {a:>8.2f} [{amin:.2f}, {amax:.2f}]  ->  "
                  f"{b:>8.2f} [{bmin:.2f}, {bmax:.2f}]   "
                  f"delta {b - a:+.2f} ({(b - a) / max(a, 1e-9) * 100:+.1f}%)")
        cov1, _, _ = agg(lo, "coverage")
        covN, _, _ = agg(hi, "coverage")
        pts = (covN - cov1) / results[(name, lo, SEEDS[0])]["V"] * 100
        print(f"    P1 (coverage loses >=10 points at 1 epoch): "
              f"{'HELD' if pts >= 10 else 'MISSED'}  ({pts:+.1f} points)")
        z1, _, _ = agg(lo, "mi_z")
        print(f"    P2 (category validity survives one pass, z>3): "
              f"{'HELD' if z1 > 3 else 'MISSED'}  (z={z1:.1f})")
        a1, _, _ = agg(lo, "n_above")
        aN, _, _ = agg(hi, "n_above")
        rel = abs(aN - a1) / max(a1, 1e-9)
        print(f"    P3 (detection epoch-invariant to +-20%): "
              f"{'HELD' if rel <= 0.20 else 'MISSED'}  "
              f"({a1:.1f} -> {aN:.1f}, {rel * 100:.0f}% change)")

    print(f"\nTOTAL {time.time() - t_all:.1f}s")
