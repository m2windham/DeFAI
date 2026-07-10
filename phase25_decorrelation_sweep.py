"""
PHASE 25 -- DECORRELATION SWEEP: is embedding correlation the whole cause of
the pool-mode perception collapse?

Phase 22 measured it first at 2.4K tokens, phase 23 confirmed it persists at
547K words: the pooled/ambiguity-gated perception stack (`perceive(pool=True,
confirm=3, amb=0.3)`), whose acceptance-bar constants (0.7 fusion, 0.6
active_bar, the 0.8/sqrt(...) anneal in organism.py) were calibrated where
random cross-word overlap is ~1/sqrt(N) (near-orthogonal synthetic patterns),
collapses on real text: only 85/395 words covered, because 68%+ of real
PPMI-SVD embeddings have a neighbor above the 0.7 fusion bar and transitive
linkage merges distinct words into a handful of components.

This phase asks a narrower, pre-registered question: how much of that
correlation is an artifact of the embedding recipe's own singular-value
weighting, not an intrinsic property of the vocabulary? Phase 23's recipe
(unchanged here) is `emb = U[:, :50] * sqrt(S[:50])`, i.e. alpha=0.5 in the
standard `U . S^alpha` family (Levy & Goldberg's SVD-weighting knob). Larger
alpha injects more of the top singular directions -- which encode corpus-wide
frequency/context structure shared by every word -- into every vector, which
is a specific, nameable mechanism for the measured correlation.

PRE-REGISTERED PREDICTION: sweeping alpha in {0.5, 0.25, 0.0} (0.0 = pure
left singular vectors, renormalized -- full decorrelation in this family)
should MONOTONICALLY reduce the >0.7-neighbor fraction and the transitive
component count. If core-arm coverage recovers toward the recipe arm's
378/395 as alpha drops, correlation is confirmed as the (or a) primary cause.
If the neighbor fraction collapses but coverage does NOT recover, the
pool-mode bar constants are a second, independent cause -- an informative
negative that hands the problem to phase 26 (percentile-calibrated bars)
either way; this script does not need to resolve which.

Caveat pre-registered: at V~=395 words, dims 30-50 of the SVD are mostly
noise; alpha=0.0 (full whitening) can amplify that noise rather than reveal
signal. The sweep is reported in full so the tradeoff is visible, not assumed.

Corpus: same 8-book Gutenberg corpus as phase 20/21/23
(/tmp/gutenberg_corpus/*.txt, not committed -- re-fetch with the curl block
below if missing).
"""

import os
import re
import glob
import time
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import PolysemyOrganism

CORPUS_DIR = "/tmp/gutenberg_corpus"
BOOKS = {
    11: "alice", 12: "looking_glass", 55: "wizard_oz", 16: "peter_pan",
    2591: "grimm", 74: "tom_sawyer", 76: "huck_finn", 1661: "sherlock",
}
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
all_text = []
for path in paths:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m_s = GUTENBERG_START.search(text)
    m_e = GUTENBERG_END.search(text)
    all_text.append(text[m_s.end() if m_s else 0: m_e.start() if m_e else len(text)])
raw_tokens = re.findall(r"[a-zA-Z']+", "\n".join(all_text).lower())

MIN_COUNT = 150
word_counts = Counter(raw_tokens)
vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]

print(f"PHASE 25: decorrelation sweep "
      f"({len(raw_tokens)} raw tokens, {len(train_seq)} in-vocab, "
      f"{N_WORDS} words, count>={MIN_COUNT})\n")

# ---------------------------------------------------------- PPMI (once)
WINDOW = 4
cooc = np.zeros((N_WORDS, N_WORDS))
for i, w in enumerate(train_seq):
    for j in range(max(0, i - WINDOW), min(len(train_seq), i + WINDOW + 1)):
        if j != i:
            cooc[w, train_seq[j]] += 1.0
tot = cooc.sum()
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log((cooc * tot) / (cooc.sum(1, keepdims=True) @ cooc.sum(0, keepdims=True)
                                 + 1e-12) + 1e-12)
ppmi_emb = np.maximum(pmi, 0.0)
U, S, _ = np.linalg.svd(ppmi_emb, full_matrices=False)
DIM = min(50, U.shape[1])
print(f"PPMI + SVD computed once (DIM={DIM}); sweeping the singular-value "
      f"weighting alpha on top of this fixed U, S.\n")


def make_embeddings(alpha):
    e = U[:, :DIM] * (S[:DIM] ** alpha)
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return e


def neighbor_diagnostic(emb):
    """phase 22/23's root-cause measurement, verbatim: fraction of words with
    a same-arm neighbor above the 0.7 fusion bar, and transitive component
    count under that adjacency."""
    ov_nn = np.abs(emb @ emb.T); np.fill_diagonal(ov_nn, 0)
    nn = ov_nn.max(1)
    adj = ov_nn > 0.7
    seen = np.zeros(N_WORDS, bool); comps = 0
    sizes = []
    for i in range(N_WORDS):
        if not seen[i]:
            comps += 1; stack = [i]; size = 0
            while stack:
                j = stack.pop()
                if not seen[j]:
                    seen[j] = True; size += 1
                    stack.extend(np.where(adj[j] & ~seen)[0].tolist())
            sizes.append(size)
    return dict(neighbor_frac=float((nn > 0.7).mean()), n_components=comps,
                largest_component=max(sizes) if sizes else 0)


def coverage_map(org, emb_c, N):
    org.consolidate(merge_thresh=0.84, prune_frac=0.0005)
    n_mem = org.mem.shape[0]
    states = np.array([emb_c[w] for w in train_seq])
    assigns = np.abs((org.mem.conj() @ states.T) / N).argmax(0)
    slot_word = {}
    for k in range(n_mem):
        members = np.array(train_seq)[assigns == k]
        if len(members):
            slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
    return len(set(slot_word.values())), n_mem


def core_arm_coverage(emb):
    """The pooled/ambiguity-gated perception stack, single pass -- the arm
    that measurably collapses (85/395 at alpha=0.5, phase 23). Same recipe,
    varying only the embeddings fed into it."""
    N = DIM; NORM = np.sqrt(N)
    emb_c = emb.astype(complex)

    def make_stream(seq, hold=4):
        for w in seq:
            s = emb_c[w]
            for _ in range(hold):
                yield s

    K_CAP = min(2000, N_WORDS * 4)
    org = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
    org.perceive(make_stream(train_seq), g_in=5.0, dt=0.05, eta=0.05, confirm=3,
                 pool=True, s_hat=0.0, probation=40000, amb=0.3)
    cov, n_mem = coverage_map(org, emb_c, N)
    return cov, n_mem


results = []
for alpha in (0.5, 0.25, 0.0):
    t0 = time.time()
    emb = make_embeddings(alpha)
    diag = neighbor_diagnostic(emb)
    cov, n_mem = core_arm_coverage(emb)
    dt = time.time() - t0
    results.append(dict(alpha=alpha, coverage=cov, n_mem=n_mem, **diag, time_s=dt))
    print(f"alpha={alpha:.2f}  neighbor_frac={diag['neighbor_frac']:.0%}  "
          f"components={diag['n_components']:>3}  largest_component={diag['largest_component']:>3}  "
          f"core-arm coverage={cov}/{N_WORDS}  memories={n_mem}  ({dt:.0f}s)")

print("\n" + "=" * 70)
print("VERDICT")
frac_trend = [r['neighbor_frac'] for r in results]
cov_trend = [r['coverage'] for r in results]
frac_monotone = all(frac_trend[i] >= frac_trend[i + 1] - 1e-9 for i in range(len(frac_trend) - 1))
cov_recovers = cov_trend[-1] > cov_trend[0] * 1.5

print(f"  neighbor fraction across alpha=0.5->0.25->0.0: {[f'{f:.0%}' for f in frac_trend]}"
      f"  (monotone decrease: {frac_monotone})")
print(f"  core-arm coverage across alpha=0.5->0.25->0.0: {cov_trend} / {N_WORDS}"
      f"  (recovers >=1.5x: {cov_recovers})")

if frac_monotone and cov_recovers:
    print("\n  CONFIRMED: embedding correlation (the SVD weighting exponent) is a "
          "primary cause of the pool-mode collapse. Recommend alpha<=0.25 as the "
          "default embedding recipe, and phase 26's percentile bars as the "
          "durable fix (this sweep is a single fixed recipe, not calibration-robust).")
elif frac_monotone and not cov_recovers:
    print("\n  PARTIAL: decorrelation reduces embedding-neighbor correlation as "
          "predicted, but core-arm coverage does not recover proportionally -- "
          "the pool-mode acceptance-bar constants are an independent, second "
          "cause. This motivates phase 26 (percentile-calibrated bars) directly, "
          "rather than as a fallback.")
else:
    print("\n  UNEXPECTED: neighbor fraction did not decrease monotonically with "
          "alpha -- re-examine before trusting either conclusion above.")
