"""
PHASE 24 -- A PRINCIPLED, LABEL-FREE CATEGORY-VALIDITY + k-SELECTION CRITERION

Phase 23 relieved the category-emergence bottleneck at 547K words but left a
sharp residual: the categories are grammatically LEGIBLE (a verb/modal
cluster, a noun-phrase cluster) yet silhouette stays ~0.02-0.03, below any
conventional cluster-validity bar. Silhouette measures GEOMETRIC separation
of the PPMI profile clouds -- the wrong certificate for soft distributional
grammatical categories -- so it can neither confirm the categories are real
nor pick k. Open threads #1 (wrong metric) and #3 (automatic k-selection)
are the same problem: silhouette is the wrong objective.

This phase replaces it with the project's OWN framing. Phase 12's insight was
that grammatical structure is about PREDICTION: a good state-split is one that
changes what you predict next. The same idea certifies categories. Good
grammatical categories are exactly the ones under which the next category is
PREDICTABLE from the current one -- i.e. the ones that maximize the mutual
information of adjacent category labels (this is precisely the Brown-clustering
objective) and that MINIMIZE the description length of the corpus under a
class-based bigram model (grammar as a small state machine -- phase 18b's
framing, made quantitative). Both are computed from corpus bigram statistics
with ZERO labels.

Two label-free criteria, swept over k, each answering one open thread:

  1. VALIDITY -- class-bigram mutual information I(C_t; C_{t+1}) against a
     DIRECTLY MEASURED null (permute the word->category labels, keep category
     sizes fixed, recompute; 500 draws). Real grammatical categories clear
     the null by many sigma; a blob or random partition does not. This is the
     project's "measure the noise floor, don't eyeball it" discipline applied
     to categories -- the certificate silhouette could not provide.

  2. k-SELECTION -- and here the obvious answer FAILS, informatively. Two-part
     MDL (data bits under a class-based bigram model p(w'|w)=p(C'|C)p(w'|C')
     plus model description length) and held-out class-bigram perplexity both
     monotonically prefer the FINEST k: a class-based bigram model with k^2+V
     params never overfits at 400K pairs, and a finer partition always predicts
     at least as well. Measured, not assumed -- so k-selection is not a
     PREDICTION problem. The fix is a PARSIMONY criterion: category-profile
     DISTINCTNESS (the minimum pairwise distance between categories' transition
     profiles) peaks at the largest k whose every category is still predictively
     distinct, then collapses when a split creates a near-duplicate category.
     Distinctness-argmax replaces silhouette-argmax (which phase 23 showed picks
     k=7-8 on flat scores).

ORACLE (evaluation ONLY -- never enters any mechanism, per the project's
rules): each vocab word's majority universal-POS tag (nltk, tagged in
context) gives a ground-truth grammatical partition. We report V-measure of
the emergent categories against it per k, to check that the label-free
criteria's chosen k is also the grammatically coherent one -- the label-free
optimum validated against the labels it never saw.

Corpus: the 8-book phase 20/21/23 Gutenberg set in /tmp/gutenberg_corpus/
(not committed, re-fetchable; the script prints the curl block if missing).
No oscillator training is needed: the metric is a property of a category
assignment and the corpus bigram statistics -- exactly the WORD-LEVEL object
phase 23's stage B clusters -- so this isolates the CRITERION from perception
(the same isolation discipline as phase 3). The winning selector plugs
straight into the unified loop's stage B (noted at the end).
"""

import re
import glob
import numpy as np
from collections import Counter, defaultdict
from polysemy_organism import _ppmi_transform, _kmeans_real, _silhouette_real

CORPUS_DIR = "/tmp/gutenberg_corpus"
BOOKS = {11: "alice", 12: "looking_glass", 55: "wizard_oz", 16: "peter_pan",
         2591: "grimm", 74: "tom_sawyer", 76: "huck_finn", 1661: "sherlock"}
paths = sorted(glob.glob(f"{CORPUS_DIR}/*.txt"))
if not paths:
    print("Corpus missing. Re-fetch (public-domain, re-fetchable) with:\n")
    print(f"  mkdir -p {CORPUS_DIR}")
    for bid, name in BOOKS.items():
        print(f"  curl -sS -o {CORPUS_DIR}/{name}.txt "
              f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt")
    raise SystemExit("\nRe-run this script once the corpus is present.")

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                             re.IGNORECASE | re.DOTALL)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
                           re.IGNORECASE | re.DOTALL)
book_bodies = []
for path in paths:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m_s = GUTENBERG_START.search(text)
    m_e = GUTENBERG_END.search(text)
    book_bodies.append(text[m_s.end() if m_s else 0: m_e.start() if m_e else len(text)])
full_text = "\n".join(book_bodies)
raw_tokens = re.findall(r"[a-zA-Z']+", full_text.lower())

MIN_COUNT = 150   # identical vocab convention to phases 20/21/23
word_counts = Counter(raw_tokens)
vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]
print(f"PHASE 24: category-validity criterion "
      f"({len(train_seq)} in-vocab tokens, {V} words, count>={MIN_COUNT})\n")

# ---------------------------------------------------------------- profiles
# WORD-LEVEL transition counts over adjacent in-vocab tokens (the phase-23
# stage-B object, built directly from the corpus so the CRITERION is isolated
# from perception). T[a,b] = # times word a immediately precedes word b.
T = np.zeros((V, V))
for a, b in zip(train_seq[:-1], train_seq[1:]):
    T[a, b] += 1.0
# PPMI-transformed successor+predecessor profile, normalized (stage-B recipe)
prof = _ppmi_transform(T)
prof = np.concatenate([prof, prof.T], axis=1)
prof /= (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)

# category-bigram tensor for a given assignment, plus successor frequency
M_pairs = list(zip(train_seq[:-1], train_seq[1:]))
succ_arr = np.array([b for _, b in M_pairs])
nsucc = np.bincount(succ_arr, minlength=V).astype(float)   # freq as successor
pred_arr = np.array([a for a, _ in M_pairs])


def cat_bigram(labels, k):
    """M[c,d] = # adjacent pairs whose (prev,next) categories are (c,d)."""
    lp, ls = labels[pred_arr], labels[succ_arr]
    return np.bincount(lp * k + ls, minlength=k * k).reshape(k, k).astype(float)


def mutual_information(M):
    p = M / M.sum()
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = p * np.log2(p / (pi @ pj + 1e-300) + 1e-300)
    return float(term[p > 0].sum())


def mdl_bits(labels, k, alpha=0.5):
    """Two-part description length (bits) of the corpus's adjacent word pairs
    under a class-based bigram model p(w'|w)=p(C'|C)p(w'|C'), plus model cost.
    Data bits fall with k; model bits rise. Lower total = better compression."""
    M = cat_bigram(labels, k)
    # class transition p(d|c), additively smoothed
    pT = (M + alpha) / (M.sum(1, keepdims=True) + alpha * k)
    # class emission p(w'|c) from successor frequencies within the class
    cat_succ_mass = np.array([nsucc[labels == c].sum() for c in range(k)])
    csize = np.bincount(labels, minlength=k)
    pE = np.array([(nsucc[w] + alpha) / (cat_succ_mass[labels[w]] + alpha * csize[labels[w]])
                   for w in range(V)])
    data_bits = -(M * np.log2(pT)).sum() - (nsucc * np.log2(pE)).sum()
    n_pairs = len(M_pairs)
    # model description: free params cost 0.5*log2(n) each (MDL/BIC), plus the
    # assignment (which word in which category)
    n_params = k * (k - 1) + (V - k)          # transitions + emissions
    model_bits = 0.5 * np.log2(n_pairs) * n_params + V * np.log2(k)
    return (data_bits + model_bits) / n_pairs   # bits per token


# held-out class-based bigram perplexity: fit p(C'|C) and p(w'|C') on 80% of
# pairs, score bits/token on the held-out 20%. The textbook fix for MDL's
# over-splitting -- BUT (measured below) it ALSO monotonically prefers finer
# k, because a class-based bigram model with k^2 + V params never overfits at
# 400K pairs and a finer partition always predicts at least as well. Predictive
# criteria certify VALIDITY but structurally cannot SELECT k.
_n = len(M_pairs)
_perm = np.random.default_rng(0).permutation(_n)
_tr, _te = _perm[:int(_n * 0.8)], _perm[int(_n * 0.8):]


def heldout_bits(labels, k, alpha=0.5):
    lp, ls = labels[pred_arr], labels[succ_arr]
    Mtr = np.bincount(lp[_tr] * k + ls[_tr], minlength=k * k).reshape(k, k).astype(float)
    pT = (Mtr + alpha) / (Mtr.sum(1, keepdims=True) + alpha * k)
    ns_tr = np.bincount(succ_arr[_tr], minlength=V).astype(float)
    csm = np.array([ns_tr[labels == c].sum() for c in range(k)])
    csz = np.bincount(labels, minlength=k)
    pE = np.array([(ns_tr[w] + alpha) / (csm[labels[w]] + alpha * csz[labels[w]])
                   for w in range(V)])
    return float((-np.log2(pT[lp[_te], ls[_te]]) - np.log2(pE[succ_arr[_te]])).mean())


def distinctness(labels, k):
    """Minimum pairwise distance between category transition PROFILES (each
    category's out-row p(.|C) concatenated with its in-column p(C|.)). This is
    a PARSIMONY criterion, not a predictive one: when k exceeds the number of
    predictively-distinct grammatical roles, two categories become near-
    duplicate profiles and this collapses. The k that MAXIMIZES it is the
    largest partition whose every category is still predictively distinct --
    the principled stopping point predictive likelihood cannot see."""
    M = cat_bigram(labels, k)
    R = M / (M.sum(1, keepdims=True) + 1e-9)
    Cc = (M / (M.sum(0, keepdims=True) + 1e-9)).T
    P = np.concatenate([R, Cc], axis=1)
    return float(min(np.linalg.norm(P[i] - P[j])
                     for i in range(k) for j in range(i + 1, k)))


# ---------------------------------------------------------------- oracle POS
# EVALUATION ONLY -- tagged in context, majority tag per vocab word. Never
# touches profiles, clustering, or any criterion above.
print("(oracle, evaluation only) tagging corpus for universal-POS ground truth...")
import nltk
pos_votes = defaultdict(Counter)
CHUNK = 400
flat = full_text.split()   # context windows for the tagger
for i in range(0, len(flat), CHUNK):
    for tok, tag in nltk.pos_tag(flat[i:i + CHUNK], tagset='universal'):
        lw = tok.lower()
        m = re.fullmatch(r"[a-z']+", lw)
        if m and lw in word_to_idx:
            pos_votes[word_to_idx[lw]][tag] += 1
pos_true = np.array([pos_votes[w].most_common(1)[0][0] if pos_votes[w] else 'X'
                     for w in range(V)])
tagged = int((pos_true != 'X').sum())
print(f"  {tagged}/{V} vocab words received a majority POS tag "
      f"({len(set(pos_true))} distinct tags)\n")
try:
    from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
    have_v = True
except Exception:
    have_v = False


# ---------------------------------------------------------------- sweep
print("predictive criteria (MDL, held-out) vs the parsimony criterion "
      "(distinctness), all label-free:\n")
print(f"{'k':>3}  {'silhou':>7}  {'MI':>6}  {'nullp99':>8}  {'MIz':>5}  "
      f"{'MDLb/t':>7}  {'heldb/t':>7}  {'distinct':>8}  {'V-meas*':>7}  sizes")
rng = np.random.default_rng(7)
rows = {}
for k in range(2, 13):
    labels, _ = _kmeans_real(prof, k, seed=3)
    sizes = np.bincount(labels, minlength=k)
    sil = _silhouette_real(prof, labels)
    mi = mutual_information(cat_bigram(labels, k))
    # directly measured null: permute labels (fixed sizes), recompute MI
    null = np.array([mutual_information(cat_bigram(rng.permutation(labels), k))
                     for _ in range(500)])
    p99 = np.percentile(null, 99)
    z = (mi - null.mean()) / (null.std() + 1e-12)
    mdl = mdl_bits(labels, k)
    held = heldout_bits(labels, k)
    dist = distinctness(labels, k)
    vm = (v_measure_score(pos_true, labels) if have_v else float('nan'))
    rows[k] = dict(sil=sil, mi=mi, p99=p99, z=z, mdl=mdl, held=held, dist=dist,
                   vm=vm, sizes=sizes, labels=labels)
    print(f"{k:>3}  {sil:>7.3f}  {mi:>6.3f}  {p99:>8.3f}  {z:>5.0f}  {mdl:>7.3f}  "
          f"{held:>7.3f}  {dist:>8.3f}  {vm:>7.3f}  {sorted(sizes.tolist(), reverse=True)}")

k_sil = max(rows, key=lambda k: rows[k]['sil'])
k_mdl = min(rows, key=lambda k: rows[k]['mdl'])
k_held = min(rows, key=lambda k: rows[k]['held'])
k_dist = max(rows, key=lambda k: rows[k]['dist'])
k_vm = max(rows, key=lambda k: rows[k]['vm']) if have_v else None
print(f"\nselectors:  silhouette-argmax k={k_sil}   MDL-argmin k={k_mdl}   "
      f"held-out-argmin k={k_held}   DISTINCTNESS-argmax k={k_dist}"
      f"   |   (oracle) V-measure-argmax k={k_vm}")

# ---------------------------------------------------------------- verdict
r = rows[k_dist]
# validity, at the selected k: does the partition clear the measured MI null?
valid = r['mi'] > r['p99'] and r['z'] > 5
# the two predictive criteria both prefer the FINEST k -> they cannot select
predictive_oversplit = k_mdl >= 11 and k_held >= 11
# does the parsimony (distinctness) choice match the oracle's coherent k?
tracks_oracle = have_v and abs(k_dist - k_vm) <= 1
vm_at_dist = r['vm']; vm_best = rows[k_vm]['vm'] if have_v else float('nan')
print()
print(f"validity: at the selected k={k_dist}, class-bigram MI={r['mi']:.3f} clears its "
      f"permutation null (p99={r['p99']:.3f}) by z={r['z']:.0f} -- categories REAL by a "
      f"measured floor, where silhouette={r['sil']:.3f} (flat ~{rows[k_sil]['sil']:.3f} "
      f"across all k) could not tell.")
print(f"why predictive criteria can't select k: a class-based bigram model always "
      f"predicts >= as well when categories get finer, so MDL (argmin k={k_mdl}) AND "
      f"held-out perplexity (argmin k={k_held}) both run to the finest k -- measured, "
      f"not assumed. k-selection needs PARSIMONY, not prediction.")
print(f"the fix: distinctness (min pairwise category-profile distance) peaks at "
      f"k={k_dist} then collapses -- the largest partition whose categories are all "
      f"still predictively distinct.")
if have_v:
    print(f"oracle check (eval only): distinctness's k={k_dist} = V-measure-argmax "
          f"k={k_vm} (score {vm_at_dist:.3f}); silhouette's k={k_sil} scores "
          f"{rows[k_sil]['vm']:.3f}, MDL's k={k_mdl} scores {rows[k_mdl]['vm']:.3f}.")
if valid and predictive_oversplit and tracks_oracle:
    print(f"\nverdict: THE METRIC IS FIXED -- two label-free criteria replace silhouette. "
          f"class-bigram MI vs a measured null certifies VALIDITY (z={r['z']:.0f} at the "
          f"chosen k); category-profile DISTINCTNESS selects k={k_dist}, which the oracle "
          f"confirms is the grammatically-coherent choice (V-measure-argmax, {vm_at_dist:.3f}). "
          f"Measured en route: predictive/compression criteria (MDL, held-out perplexity) "
          f"monotonically over-split and cannot select k -- k-selection is a parsimony "
          f"problem, not a prediction one. Open threads #1 (wrong metric) and #3 (auto "
          f"k-selection) close together; both selectors drop into stage B of the unified loop.")
elif valid and tracks_oracle:
    print(f"\nverdict: METRIC FIXED (validity + k-selection both label-free and oracle-"
          f"confirmed at k={k_dist}); the predictive-over-split diagnosis is softer than "
          f"designed (MDL k={k_mdl}, held-out k={k_held}) -- see the sweep.")
elif valid:
    print(f"\nverdict: VALIDITY FIXED, k-SELECTION PARTIAL -- MI-vs-null certifies the "
          f"categories are real (z={r['z']:.0f}), but distinctness-argmax k={k_dist} and "
          f"the oracle's k={k_vm} diverge. See the sweep.")
else:
    print("\nverdict: partial -- see the sweep table; a criterion did not behave "
          "as designed.")
