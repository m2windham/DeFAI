"""
PHASE 20 -- GENUINELY LARGE REAL CORPUS

Phase 19 concluded, with a directly measured noise floor (not a guess),
that the predictive-gain polysemy mechanism needs roughly n=200-500+
occurrences per candidate word to be statistically trustworthy -- a bar
the hand-written fable corpus (1,400-2,800 tokens) could never clear for
most content words.

This phase tests the mechanism on ~537,000 words of real public-domain
prose (8 classic books: Alice in Wonderland, Through the Looking Glass,
Wizard of Oz, Peter Pan, Grimm's Fairy Tales, Tom Sawyer, Huckleberry
Finn, Sherlock Holmes -- fetched from Project Gutenberg), restricted to a
frequent-word vocabulary specifically so common content words clear the
n=200-500 bar this project already proved is required.
"""

import re
import glob
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import PolysemyOrganism, _entropy

# ---- load and clean the corpus -----------------------------------------------
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

all_text = []
for path in sorted(glob.glob("/tmp/gutenberg_corpus/*.txt")):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m_start = GUTENBERG_START.search(text)
    m_end = GUTENBERG_END.search(text)
    start = m_start.end() if m_start else 0
    end = m_end.start() if m_end else len(text)
    body = text[start:end]
    all_text.append(body)
    print(f"  {path.split('/')[-1]}: {len(body.split())} words (after header/footer strip)")

full_text = "\n".join(all_text)
raw_tokens = re.findall(r"[a-zA-Z']+", full_text.lower())
print(f"\nTotal raw tokens across corpus: {len(raw_tokens)}")

# ---- vocabulary: restrict to frequent words so common content words --------
# clear the n=200-500 bar Phase 19 proved is required. Rare words (the long
# tail of any natural corpus) are dropped -- not because they're
# uninteresting, but because no amount of data makes THEM individually
# reach that bar; this is an explicit, honest scope restriction, not
# swept under the rug.
word_counts = Counter(raw_tokens)
MIN_COUNT = 150
vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
print(f"Vocabulary (count >= {MIN_COUNT}): {N_WORDS} words")
print(f"Occurrence range: min={word_counts[vocab[np.argmin([word_counts[w] for w in vocab])]]}  "
      f"max={word_counts.most_common(1)[0]}")

train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]
print(f"Training stream length (restricted to frequent vocab): {len(train_seq)}")

# ---- candidate real polysemous words worth watching for -----------------------
CANDIDATE_POLYSEMOUS = ['watch', 'light', 'bank', 'spring', 'duck', 'bear', 'run',
                         'well', 'fair', 'kind', 'still', 'right', 'left', 'match',
                         'train', 'fly', 'rock', 'park', 'book', 'fine', 'saw', 'letter']
present = [w for w in CANDIDATE_POLYSEMOUS if w in word_to_idx]
print(f"\nCandidate polysemous words present in vocab: {present}")
for w in present:
    print(f"  '{w}': {word_counts[w]} occurrences")

# ---- build embeddings (PPMI + SVD) -------------------------------------------
print("\nBuilding PPMI+SVD embeddings...")
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

DIM = 50
U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
embeddings_real = U[:, :DIM] * np.sqrt(S[:DIM])
embeddings_real /= (np.linalg.norm(embeddings_real, axis=1, keepdims=True) + 1e-9)
N = DIM
NORM = np.sqrt(N)
embeddings = embeddings_real.astype(complex)
print(f"Embeddings built: {embeddings_real.shape}")

for w in present:
    wi = word_to_idx[w]
    sims = embeddings_real @ embeddings_real[wi]
    nearest = np.argsort(-sims)[1:6]
    print(f"  nearest to '{w}': {[vocab[n] for n in nearest]}")

# ---- train PolysemyOrganism (recruit=0.75, fixed per Phase 19) --------------
def make_stream(seq, hold=4):
    for w in seq:
        s = embeddings[w]
        for _ in range(hold):
            yield s

RECRUIT_THRESH = 0.75
CAPACITY_K = min(1000, N_WORDS * 3)
N_EPOCHS = 3

print(f"\n{'='*72}\nTRAINING ({N_EPOCHS} epochs, recruit={RECRUIT_THRESH}, K={CAPACITY_K})\n")
org = PolysemyOrganism(N=N, K=CAPACITY_K, omega=0.15, beta=10.0, seed=0)
for epoch in range(1, N_EPOCHS + 1):
    t0 = __import__('time').time()
    org.perceive(list(make_stream(train_seq, hold=4)), g_in=5.0, dt=0.05, eta=0.02, recruit=RECRUIT_THRESH)
    print(f"  epoch {epoch}: {__import__('time').time()-t0:.1f}s, slots_used={int(org.used.sum())}")

print("\nConsolidating...")
org.consolidate(merge_thresh=0.84, prune_frac=0.0005)
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
print(f"Word coverage: {len(coverage)}/{N_WORDS}")
missing = [vocab[w] for w in range(N_WORDS) if w not in coverage]
print(f"Missing ({len(missing)}): {missing}")

np.save('/tmp/phase20_org_mem.npy', org.mem)
np.save('/tmp/phase20_assigns.npy', assigns)
# compact raw transition counts restricted to kept memories, in the SAME
# order as org.mem -- needed by discover_categories_v2's PPMI transform,
# which requires unnormalized counts (org.Pn alone is already row-
# normalized and has lost the magnitude information PPMI needs).
raw_counts_compact = org.P[np.ix_(org.kept_idx, org.kept_idx)]
np.save('/tmp/phase20_raw_counts.npy', raw_counts_compact)
import pickle
with open('/tmp/phase20_state.pkl', 'wb') as f:
    pickle.dump(dict(slot_word=slot_word, org_used=org.used, org_P=org.P,
                      org_Pn=org.Pn, n_mem=n_mem, kept_idx=org.kept_idx), f)
print("Saved intermediate state to /tmp/phase20_*.")
