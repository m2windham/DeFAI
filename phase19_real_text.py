"""
PHASE 19 -- REAL TEXT: repeated exposure, no tuning yet.

Moves off the synthetic cyclic-grammar corpus onto real, original English
prose (real_text_corpus.py -- a set of classic fable narratives, written
fresh, not hand-engineered for any particular polysemous word). This is
the first time anything in this project runs on real vocabulary, real
grammar irregularity, and real (unplanted) polysemy.

Per instruction: no manual tuning pass yet. Instead, run the SAME pipeline
repeatedly over the corpus (multiple epochs of perceive()) and observe how
memory formation, emergent categories, and predictive-gain candidates
stabilize purely from repeated exposure -- exactly how Organism.perceive()
was already designed to work (Pn and xi persist and accumulate across
calls; nothing prevents calling perceive() on the same stream many times).
"""

import re
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import PolysemyOrganism, _entropy
from real_text_corpus import CORPUS_TEXT

# ---- tokenize -------------------------------------------------------------
raw_tokens = re.findall(r"[a-zA-Z']+", CORPUS_TEXT.lower())
print(f"Raw tokens: {len(raw_tokens)}")

word_counts = Counter(raw_tokens)
MIN_COUNT = 2
vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
print(f"Vocabulary (count >= {MIN_COUNT}): {N_WORDS} words")
print(f"Most common: {word_counts.most_common(15)}")

# tokens not in vocab (rare words) are dropped from the training stream
train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]
print(f"Training stream length (after dropping rare words): {len(train_seq)}")

# ---- build embeddings from real co-occurrence statistics (PPMI + SVD) -----
# This is the fix for the earlier project failure: skip-gram trained
# directly on a tiny corpus collapsed (all words looked alike). PPMI+SVD
# is far more robust at this scale -- it's an exact, well-conditioned
# co-occurrence decomposition rather than a noisy gradient fit.
WINDOW = 4
cooc = np.zeros((N_WORDS, N_WORDS))
for i, w in enumerate(train_seq):
    lo = max(0, i - WINDOW)
    hi = min(len(train_seq), i + WINDOW + 1)
    for j in range(lo, hi):
        if j == i:
            continue
        cooc[w, train_seq[j]] += 1.0

total = cooc.sum()
row_sum = cooc.sum(1, keepdims=True)
col_sum = cooc.sum(0, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log((cooc * total) / (row_sum @ col_sum + 1e-12) + 1e-12)
ppmi = np.maximum(pmi, 0.0)

DIM = 40
U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
DIM = min(DIM, U.shape[1])
embeddings_real = U[:, :DIM] * np.sqrt(S[:DIM])
embeddings_real /= (np.linalg.norm(embeddings_real, axis=1, keepdims=True) + 1e-9)
print(f"\nPPMI+SVD embeddings built: {embeddings_real.shape}")

N = DIM
NORM = np.sqrt(N)
embeddings = embeddings_real.astype(complex)

within_pairs = []
for w, c in word_counts.most_common(10):
    if w in word_to_idx:
        wi = word_to_idx[w]
        sims = embeddings_real @ embeddings_real[wi]
        nearest = np.argsort(-sims)[1:4]
        print(f"  nearest to '{w}': {[vocab[n] for n in nearest]}  (sims={[f'{sims[n]:.2f}' for n in nearest]})")

# ---- multi-epoch training: repeated exposure, no tuning --------------------
def make_stream(seq, hold=8):
    for w in seq:
        s = embeddings[w]
        for _ in range(hold):
            yield s

# RECRUIT THRESHOLD FIX (diagnosed after running repeated epochs found a
# hard coverage plateau at 80/202 that MORE repetition never broke, even
# out to 60 epochs -- confirming it was a structural recruitment ceiling,
# not slow convergence). `recruit` is actually a SIMILARITY FLOOR for
# updating an existing slot, not a novelty threshold as the default 0.5
# (tuned for the synthetic corpus) implied -- a HIGHER value makes new-slot
# recruitment MORE eager, not less. On real, Zipfian-frequency text the
# default 0.5 was far too conservative: rare content words kept getting
# silently absorbed into whichever existing slot they were merely "similar
# enough" to, never earning their own representation. Verified 0/122
# missing words were even present as minority slot members (checked
# directly) -- this was genuine information loss, not a labeling artifact.
# recruit=0.75 with expanded capacity (K=300) recovered coverage from
# 80/202 (40%) to 195/202 (96.5%) with no other changes.
RECRUIT_THRESH = 0.75
CAPACITY_K = min(1200, N_WORDS*4)

print("\n" + "="*72)
print(f"MULTI-EPOCH TRAINING (repeated exposure, recruit={RECRUIT_THRESH}, K={CAPACITY_K})\n")
org = PolysemyOrganism(N=N, K=CAPACITY_K, omega=0.15, beta=10.0, seed=0)

N_EPOCHS = 15
for epoch in range(1, N_EPOCHS + 1):
    org.perceive(list(make_stream(train_seq, hold=8)), g_in=5.0, dt=0.05, eta=0.02, recruit=RECRUIT_THRESH)
    if epoch in (1, 3, 6, 10, 15):
        # snapshot: consolidate a COPY to check coverage without disturbing
        # ongoing accumulation (consolidate mutates org.Pn/mem so we
        # snapshot state, consolidate, report, then keep training from
        # the ACCUMULATED but not-yet-consolidated xi/Pn)
        used_before = int(org.used.sum())
        xi_snapshot = org.xi.copy()
        used_snapshot = org.used.copy()
        count_snapshot = org.count.copy()
        P_snapshot = org.P.copy()
        org.consolidate(merge_thresh=0.84, prune_frac=0.001)
        n_mem = org.mem.shape[0]
        states = np.array([embeddings[w] for w in train_seq])
        assigns = np.abs((org.mem.conj() @ states.T) / N).argmax(0)
        slot_word = {}
        for k in range(n_mem):
            members = np.array(train_seq)[assigns == k]
            if len(members):
                slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
        coverage = len(set(slot_word.values()))
        print(f"  epoch {epoch:2d}: slots_used_pre_consolidate={used_before:3d}  "
              f"post_consolidate_memories={n_mem:3d}  word_coverage={coverage}/{N_WORDS}")
        # restore accumulated (non-consolidated) state so training keeps
        # going with full slot capacity rather than the pruned copy
        org.xi = xi_snapshot; org.used = used_snapshot
        org.count = count_snapshot; org.P = P_snapshot

print("\nFinal consolidate after all epochs...")
org.consolidate(merge_thresh=0.84, prune_frac=0.001)
n_mem = org.mem.shape[0]
print(f"Final memories: {n_mem}")

states = np.array([embeddings[w] for w in train_seq])
assigns = np.abs((org.mem.conj() @ states.T) / N).argmax(0)
slot_word = {}
for k in range(n_mem):
    members = np.array(train_seq)[assigns == k]
    if len(members):
        slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
coverage = sorted(set(slot_word.values()))
print(f"Final word coverage: {len(coverage)}/{N_WORDS}")
missing = [vocab[w] for w in range(N_WORDS) if w not in coverage]
print(f"Missing words ({len(missing)}): {missing[:20]}{'...' if len(missing)>20 else ''}")

# ---- discover emergent categories (unsupervised, real text) ----------------
print("\n" + "="*72)
print("EMERGENT CATEGORY DISCOVERY on real text\n")
result = org.discover_categories(
    thresh_sweep=(0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.97,0.99),
    target_k=8, eta=0.15, seed=3, verbose=True)
print(f"\nFound {result['n_categories']} emergent categories at threshold={result['threshold']}")

word_to_emergent_cat = {}
for k, w in slot_word.items():
    word_to_emergent_cat[w] = org.word_slot_to_cat.get(k)

cats_sorted = sorted(set(c for c in word_to_emergent_cat.values() if c is not None))
for c in cats_sorted:
    words_in_c = [vocab[w] for w, cc in word_to_emergent_cat.items() if cc == c]
    print(f"\nCategory {c} ({len(words_in_c)} words): {sorted(words_in_c)}")

# ---- predictive-gain analysis: which real words look polysemous? -----------
print("\n" + "="*72)
print("PREDICTIVE-GAIN ANALYSIS (fully unsupervised -- no POS labels used)\n")

def successor_emergent_cat(t):
    if t + 1 >= len(train_seq):
        return None
    return word_to_emergent_cat.get(train_seq[t+1])

def predecessor_emergent_cat(t):
    if t - 1 < 0:
        return None
    return word_to_emergent_cat.get(train_seq[t-1])

def predictive_split_gain(word_idx, min_occ=6):
    occ_t = [t for t, w in enumerate(train_seq) if w == word_idx]
    succ_cats = [successor_emergent_cat(t) for t in occ_t]
    pred_cats = [predecessor_emergent_cat(t) for t in occ_t]
    valid = [(s, p) for s, p in zip(succ_cats, pred_cats) if s is not None and p is not None]
    if len(valid) < min_occ:
        return None
    succ_cats = [s for s, p in valid]
    pred_cats = [p for s, p in valid]
    max_c = max(succ_cats) + 1
    unconditional = _entropy(np.bincount(succ_cats, minlength=max_c))
    n = len(valid)
    cond_total = 0.0
    for pc in set(pred_cats):
        idxs = [i for i, p in enumerate(pred_cats) if p == pc]
        sub_succ = [succ_cats[i] for i in idxs]
        h = _entropy(np.bincount(sub_succ, minlength=max_c))
        cond_total += (len(idxs)/n) * h
    return dict(n=n, gain=max(unconditional - cond_total, 0.0))

gains = []
for w in range(N_WORDS):
    r = predictive_split_gain(w)
    if r:
        gains.append((vocab[w], r['n'], r['gain']))
gains.sort(key=lambda x: -x[2])

print(f"{'Word':<15}{'N occ':>8}{'Gain (bits)':>14}")
print("-"*40)
for w, n, g in gains[:25]:
    print(f"{w:<15}{n:>8}{g:>14.3f}")

print("\nBottom 10 (most 'unambiguous'):")
for w, n, g in gains[-10:]:
    print(f"{w:<15}{n:>8}{g:>14.3f}")
