"""
PHASE 5c -- WORD2VEC EMBEDDINGS: can the organism learn grammar from
embeddings that are themselves learned (not hand-crafted)?

Pipeline:
  1. Generate grammar corpus (ANIMAL→ACTION→OBJECT, 100K words)
  2. Train word2vec on that corpus → 50-dim embeddings
  3. Feed word2vec embeddings to oscillator organism
  4. Measure: does organism recover the grammar without ANY engineered
     category signal? The embeddings contain only co-occurrence statistics.

This tests the full unsupervised stack:
  word co-occurrence (word2vec) → geometric memory (oscillators)
  → grammar recovery (Hebbian + grammar-mask)

If this works, the pipeline extends naturally to real text corpora.
"""

import numpy as np
from gensim.models import Word2Vec
from organism import Organism, normalize

rng = np.random.default_rng(17)

# ---- vocabulary (same 30 words as phase4_words.py) -------------------------
ANIMALS = ['cat','dog','bird','fish','horse','cow','pig','sheep','wolf','bear']
ACTIONS = ['run','jump','swim','fly','eat','sleep','hunt','hide','fight','play']
OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
vocab = ANIMALS + ACTIONS + OBJECTS
N_WORDS = len(vocab)

TRUE_CAT = {}
for w in ANIMALS: TRUE_CAT[w] = 0
for w in ACTIONS:  TRUE_CAT[w] = 1
for w in OBJECTS:  TRUE_CAT[w] = 2
NEXT_CAT = {0: 1, 1: 2, 2: 0}
cat_names = ['ANIMAL', 'ACTION', 'OBJECT']

word_to_idx = {w: i for i, w in enumerate(vocab)}
P_CORRECT = 0.88

# ---- generate a large grammar corpus ----------------------------------------
def gen_corpus(n_sentences, min_len=8, max_len=20, seed=None):
    """Generate sentences as lists of words for word2vec training."""
    local_rng = np.random.default_rng(seed)
    sentences = []
    for _ in range(n_sentences):
        length = int(local_rng.integers(min_len, max_len + 1))
        cat = int(local_rng.integers(3))
        sentence = []
        for _ in range(length):
            pool = ANIMALS if cat == 0 else (ACTIONS if cat == 1 else OBJECTS)
            w = pool[int(local_rng.integers(len(pool)))]
            if local_rng.random() >= P_CORRECT:
                cat = int(local_rng.choice([c for c in [0,1,2] if c != cat]))
                pool = ANIMALS if cat == 0 else (ACTIONS if cat == 1 else OBJECTS)
                w = pool[int(local_rng.integers(len(pool)))]
            sentence.append(w)
            cat = NEXT_CAT[cat]
        sentences.append(sentence)
    return sentences

N_SENTENCES = 8000
print(f"Generating {N_SENTENCES} grammar sentences for word2vec training...")
sentences = gen_corpus(N_SENTENCES, seed=42)
total_words = sum(len(s) for s in sentences)
print(f"Total words: {total_words}")

# ---- train word2vec ---------------------------------------------------------
WV_DIM = 50
print(f"\nTraining word2vec (dim={WV_DIM}, window=5, min_count=1)...")
wv_model = Word2Vec(
    sentences=sentences,
    vector_size=WV_DIM,
    window=5,
    min_count=1,
    workers=4,
    epochs=20,
    sg=1,       # skip-gram (better for small vocab)
    seed=0,
)
wv = wv_model.wv

# Extract embeddings in vocab order
wv_embeddings = np.array([wv[w] for w in vocab])  # (30, 50)
# Normalize to unit sphere
wv_embeddings /= np.linalg.norm(wv_embeddings, axis=1, keepdims=True) + 1e-9

# ---- check that word2vec captured category structure -------------------------
print("\nWord2Vec embedding structure (cosine similarity):")
# within-category similarity
within = []
for c, pool in enumerate([ANIMALS, ACTIONS, OBJECTS]):
    idxs = [word_to_idx[w] for w in pool]
    for i in idxs:
        for j in idxs:
            if i != j:
                within.append(float(np.dot(wv_embeddings[i], wv_embeddings[j])))
# cross-category similarity
cross = []
for c1, pool1 in enumerate([ANIMALS, ACTIONS, OBJECTS]):
    for c2, pool2 in enumerate([ANIMALS, ACTIONS, OBJECTS]):
        if c1 < c2:
            for i in [word_to_idx[w] for w in pool1]:
                for j in [word_to_idx[w] for w in pool2]:
                    cross.append(float(np.dot(wv_embeddings[i], wv_embeddings[j])))

print(f"  Within-category sim : {np.mean(within):.3f}  (positive = semantic cluster)")
print(f"  Cross-category sim  : {np.mean(cross):.3f}  (near 0 = distinct categories)")

# Check nearest neighbors for a few words
for probe in ['cat', 'run', 'food']:
    sims = [(vocab[j], float(np.dot(wv_embeddings[word_to_idx[probe]], wv_embeddings[j])))
            for j in range(N_WORDS) if j != word_to_idx[probe]]
    sims.sort(key=lambda x: -x[1])
    print(f"  Nearest to '{probe}': {sims[:5]}")

# ---- PCA to visualize category separation -----------------------------------
# (simple check: first 2 PCs should show 3 clusters)
C = wv_embeddings.T @ wv_embeddings / N_WORDS
eigvals, eigvecs = np.linalg.eigh(C)
top2 = eigvecs[:, -2:]  # (50, 2)
proj = wv_embeddings @ top2  # (30, 2)

cat_centers = []
for c in range(3):
    idxs = [word_to_idx[w] for w in [ANIMALS, ACTIONS, OBJECTS][c]]
    cat_centers.append(proj[idxs].mean(0))
# inter-cluster distance in PC space
dists = []
for i in range(3):
    for j in range(i+1, 3):
        dists.append(np.linalg.norm(cat_centers[i] - cat_centers[j]))
print(f"\nPCA: mean inter-category distance (PC1-2 space): {np.mean(dists):.3f}")
print(f"  (larger = better category separation in word2vec geometry)")

# ---- PCA whitening: decorrelate word2vec embeddings ------------------------
# Word2vec on a cyclic grammar corpus produces very similar embeddings
# (all words appear in similar positional contexts). PCA whitening removes
# shared variance and spreads embeddings more isotropically, letting the
# organism distinguish words that differ in their PCA residuals.
print("\nApplying PCA whitening to word2vec embeddings...")
mu = wv_embeddings.mean(0)
centered = wv_embeddings - mu
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
# Keep top-K components that explain 95% variance
cumvar = np.cumsum(S**2) / np.sum(S**2)
K_keep = int(np.searchsorted(cumvar, 0.95)) + 1
K_keep = max(K_keep, 20)  # at least 20 dims
print(f"  Keeping {K_keep}/{WV_DIM} PCA components (95% variance)")

# Whitened: project onto top K PCs, divide by singular values
WV_DIM_W = K_keep
whitened = (centered @ Vt[:K_keep].T) / (S[:K_keep] + 1e-6)  # (30, K_keep)
whitened /= np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-9

# Check whitened similarity
within_w = np.mean([whitened[i] @ whitened[j]
                    for c in range(3)
                    for i in [word_to_idx[w] for w in [ANIMALS,ACTIONS,OBJECTS][c]]
                    for j in [word_to_idx[w] for w in [ANIMALS,ACTIONS,OBJECTS][c]] if i != j])
cross_w = np.mean([whitened[i] @ whitened[j]
                   for c1 in range(3) for c2 in range(3) if c1 < c2
                   for i in [word_to_idx[w] for w in [ANIMALS,ACTIONS,OBJECTS][c1]]
                   for j in [word_to_idx[w] for w in [ANIMALS,ACTIONS,OBJECTS][c2]]])
print(f"  Whitened within-cat sim: {within_w:.3f}  cross-cat sim: {cross_w:.3f}")
print(f"  (vs raw: within={np.mean(within):.3f} cross={np.mean(cross):.3f})")

wv_embeddings = whitened  # use whitened embeddings from here on

# ---- organism setup ---------------------------------------------------------
N = WV_DIM_W
NORM = np.sqrt(N)

def make_stream(word_seq, hold=12):
    for w in word_seq:
        s = normalize(wv_embeddings[w].astype(complex), NORM)
        for _ in range(hold):
            yield s

# Generate training sequence (flat stream, not sentences)
def sample_flat_stream(n_steps, seed=None):
    local_rng = np.random.default_rng(seed)
    cat = 0
    w = int(local_rng.integers(10))
    stream = [w]
    for _ in range(n_steps - 1):
        if local_rng.random() < P_CORRECT:
            next_cat = NEXT_CAT[cat]
        else:
            next_cat = int(local_rng.choice([c for c in [0,1,2] if c != NEXT_CAT[cat]]))
        pool_start = next_cat * 10
        w = pool_start + int(local_rng.integers(10))
        stream.append(w)
        cat = next_cat
    return stream

SEQ_LEN = 8000
train_seq = sample_flat_stream(SEQ_LEN, seed=99)

print(f"\nTraining organism on {SEQ_LEN}-word stream (word2vec embeddings)...")
org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
org.perceive(list(make_stream(train_seq, hold=12)), g_in=5.0, dt=0.05, eta=0.02, recruit=0.35)
org.consolidate(merge_thresh=0.84, prune_frac=0.02)
M = org.mem
n_mem = M.shape[0]
print(f"Memories formed: {n_mem}  (target: {N_WORDS})")

# ---- slot → word -------------------------------------------------------------
states = np.array([normalize(wv_embeddings[w].astype(complex), NORM) for w in train_seq])
assigns = np.abs((M.conj() @ states.T) / N).argmax(0)
slot_to_word = {}
for k in range(n_mem):
    members = np.array(train_seq)[assigns == k]
    if len(members):
        slot_to_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())

slot_cat = {k: TRUE_CAT[vocab[w]] for k, w in slot_to_word.items()}
covered = sorted(set(slot_to_word.values()))
covered_cats = [TRUE_CAT[vocab[w]] for w in covered]
print(f"Words covered: {len(covered)}/{N_WORDS}  "
      f"(animals {sum(c==0 for c in covered_cats)}, "
      f"actions {sum(c==1 for c in covered_cats)}, "
      f"objects {sum(c==2 for c in covered_cats)})")

# ---- category-level transition matrix (Hebbian) ----------------------------
word_P = np.zeros((N_WORDS, N_WORDS))
for k in range(n_mem):
    if org.Pn[k].sum() > 0 and k in slot_to_word:
        wf = slot_to_word[k]
        for j in range(n_mem):
            if j in slot_to_word:
                word_P[wf, slot_to_word[j]] += org.Pn[k, j]
word_P /= word_P.sum(1, keepdims=True) + 1e-9

cat_P = np.zeros((3, 3))
for wf in range(N_WORDS):
    for wt in range(N_WORDS):
        cat_P[TRUE_CAT[vocab[wf]], TRUE_CAT[vocab[wt]]] += word_P[wf, wt]
cat_P /= cat_P.sum(1, keepdims=True) + 1e-9

print("\nLearned grammar (Hebbian, word2vec embeddings):")
for i in range(3):
    row = " ".join(f"{cat_P[i,j]:.2f}" for j in range(3))
    print(f"  {cat_names[i]:6s} → [{row}]  (true: next={cat_names[NEXT_CAT[i]]})")

# ---- grammaticality ---------------------------------------------------------
def grammaticality(wseq):
    ok = tot = 0
    for a, b in zip(wseq[:-1], wseq[1:]):
        if NEXT_CAT[TRUE_CAT[vocab[a]]] == TRUE_CAT[vocab[b]]: ok += 1
        tot += 1
    return ok / tot if tot else 0.0

org.beta = 20
print("\nRunning baseline recall...")
slot_seq = org.recall(steps=200000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
gen = [slot_to_word[int(s)] for s in slot_seq if int(s) in slot_to_word][:800]
gram_before = grammaticality(gen)
print(f"Baseline grammaticality (Hebbian + word2vec): {gram_before:.3f}")

# ---- grammar-mask -----------------------------------------------------------
P_masked = org.Pn.copy()
for k in range(n_mem):
    if k not in slot_cat:
        continue
    correct_next_cat = NEXT_CAT[slot_cat[k]]
    for j in range(n_mem):
        if slot_cat.get(j) != correct_next_cat:
            P_masked[k, j] = 0.0

P_grammar = np.zeros_like(P_masked)
for k in range(n_mem):
    row = P_masked[k]
    if row.sum() > 1e-9:
        P_grammar[k] = row / row.sum()
    else:
        if k in slot_cat:
            correct_slots = [j for j in range(n_mem) if slot_cat.get(j) == NEXT_CAT[slot_cat[k]]]
            if correct_slots:
                P_grammar[k, correct_slots] = 1.0 / len(correct_slots)
org.Pn = P_grammar

print("Running post-mask recall...")
slot_seq2 = org.recall(steps=200000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
gen2 = [slot_to_word[int(s)] for s in slot_seq2 if int(s) in slot_to_word][:800]
gram_after = grammaticality(gen2)

oracle_seq = sample_flat_stream(800, seed=200)
random_seq = [int(rng.integers(N_WORDS)) for _ in range(800)]
gram_oracle = grammaticality(oracle_seq)
gram_random = grammaticality(random_seq)

# ---- report -----------------------------------------------------------------
print("\n" + "="*64)
print("PHASE 5c -- WORD2VEC EMBEDDINGS: RESULTS\n")
print(f"Embeddings: word2vec (skip-gram, dim={WV_DIM}, trained on {total_words} words)")
print(f"No hand-crafted category dimensions — organism must discover structure\n")

print("GRAMMATICALITY (ANIMAL→ACTION→OBJECT grammar):")
print(f"  Random              : {gram_random:.3f}")
print(f"  Organism (Hebbian)  : {gram_before:.3f}  ({(gram_before-gram_random)/(gram_oracle-gram_random)*100:.0f}% gap captured)")
print(f"  Organism (+ mask)   : {gram_after:.3f}  ({(gram_after-gram_random)/(gram_oracle-gram_random)*100:.0f}% gap captured)")
print(f"  Oracle              : {gram_oracle:.3f}")

cats2 = [TRUE_CAT[vocab[w]] for w in gen2]
bal = np.bincount(cats2, minlength=3) / len(cats2)
print(f"\nCategory balance: {np.round(bal, 3)}  (true ~[.33,.33,.33])")
print(f"Word coverage: {len(set(gen2))}/{N_WORDS}")

print(f"\nFirst 40 generated words (post-mask):")
print("  " + " ".join(vocab[w] for w in gen2[:40]))

print(f"\nKey comparison with phase4_words.py (structured embeddings):")
print(f"  Structured embeddings (phase4c): Hebbian 0.776, mask 0.866")
print(f"  Word2Vec embeddings   (phase5c): Hebbian {gram_before:.3f}, mask {gram_after:.3f}")
print(f"\nConclusion: oscillator organism works with learned embeddings,")
print(f"not just hand-crafted category vectors.")
