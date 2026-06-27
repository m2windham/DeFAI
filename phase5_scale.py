"""
PHASE 5 -- SCALE TEST: 100 words, 5 categories, 64-dim oscillators

Test hypothesis: the grammar-mask pipeline (Hebbian perception +
grammar-aware P correction) scales to larger vocabularies and more
complex grammar structures.

Grammar (5-category cyclic):
  SUBJECT → VERB → OBJECT → PLACE → TIME → SUBJECT → ...
  p(correct_category) = 0.85

Vocabulary: 100 words, 20 per category
Embeddings: 64-dim structured vectors
  Dims 0-14:  category signal (5 categories × 3 bits)
  Dims 15-63: random within-category variation
"""

import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(42)

# ---- vocabulary ---------------------------------------------------------------
SUBJECTS = [f"subj_{i}" for i in range(20)]
VERBS    = [f"verb_{i}" for i in range(20)]
OBJECTS  = [f"obj_{i}"  for i in range(20)]
PLACES   = [f"place_{i}" for i in range(20)]
TIMES    = [f"time_{i}"  for i in range(20)]

vocab = SUBJECTS + VERBS + OBJECTS + PLACES + TIMES
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)   # 100
N_CATS = 5
DIM = 64
N = DIM
NORM = np.sqrt(N)

CAT = {}
for w in SUBJECTS: CAT[w] = 0
for w in VERBS:    CAT[w] = 1
for w in OBJECTS:  CAT[w] = 2
for w in PLACES:   CAT[w] = 3
for w in TIMES:    CAT[w] = 4
NEXT_CAT = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}

# ---- structured embeddings ---------------------------------------------------
cat_bases = np.zeros((N_CATS, DIM))
for c in range(N_CATS):
    cat_bases[c, c*3:(c+1)*3] = 1.0   # dims 0-14: 3 bits per category

embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(vocab):
    c = CAT[w]
    embeddings[i] = 0.55 * cat_bases[c] + 0.45 * rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

within = np.mean([embeddings[i] @ embeddings[j]
                  for i in range(20) for j in range(20) if i != j])
across = np.mean([embeddings[i] @ embeddings[j]
                  for i in range(20) for j in range(20, 40)])
print(f"Embedding structure — within-category sim: {within:.3f}  cross-category: {across:.3f}")

# ---- grammar ------------------------------------------------------------------
P_CORRECT = 0.85

def sample_sentence_stream(n_steps, seed=None):
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    cat = 0
    w = int(local_rng.integers(20))
    stream = [w]
    for _ in range(n_steps - 1):
        if local_rng.random() < P_CORRECT:
            next_cat = NEXT_CAT[cat]
        else:
            next_cat = int(local_rng.choice([c for c in range(N_CATS) if c != NEXT_CAT[cat]]))
        base = next_cat * 20
        w = base + int(local_rng.integers(20))
        stream.append(w)
        cat = next_cat
    return stream

def make_stream(word_seq, hold=10):
    for w in word_seq:
        s = normalize(embeddings[w].astype(complex), NORM)
        for _ in range(hold):
            yield s

# ---- train organism -----------------------------------------------------------
SEQ_LEN = 20000
train_seq = sample_sentence_stream(SEQ_LEN, seed=99)

print(f"\nTraining organism on {SEQ_LEN}-word stream...")
org = Organism(N=N, K=120, omega=0.15, beta=10.0, seed=0)
org.perceive(list(make_stream(train_seq, hold=10)), g_in=5.0, dt=0.05, eta=0.02, recruit=0.45)
org.consolidate(merge_thresh=0.86, prune_frac=0.015)
M = org.mem
n_mem = M.shape[0]
print(f"Memories formed: {n_mem}  (target: {N_WORDS})")

# ---- slot → word -------------------------------------------------------------
states = np.array([normalize(embeddings[w].astype(complex), NORM) for w in train_seq])
assigns = np.abs((M.conj() @ states.T) / N).argmax(0)
slot_to_word = {}
for k in range(n_mem):
    members = np.array(train_seq)[assigns == k]
    if len(members):
        slot_to_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())

slot_cat = {k: CAT[vocab[w]] for k, w in slot_to_word.items()}
covered = sorted(set(slot_to_word.values()))
print(f"Words covered: {len(covered)}/{N_WORDS}")
for c in range(N_CATS):
    cat_name = ['SUBJ','VERB','OBJ','PLACE','TIME'][c]
    print(f"  {cat_name}: {sum(CAT[vocab[w]]==c for w in covered)}/20")

# ---- category-level transition matrix ----------------------------------------
word_P = np.zeros((N_WORDS, N_WORDS))
for k in range(n_mem):
    if org.Pn[k].sum() > 0 and k in slot_to_word:
        wf = slot_to_word[k]
        for j in range(n_mem):
            if j in slot_to_word:
                word_P[wf, slot_to_word[j]] += org.Pn[k, j]
word_P /= word_P.sum(1, keepdims=True) + 1e-9

cat_P = np.zeros((N_CATS, N_CATS))
for wf in range(N_WORDS):
    for wt in range(N_WORDS):
        cat_P[CAT[vocab[wf]], CAT[vocab[wt]]] += word_P[wf, wt]
cat_P /= cat_P.sum(1, keepdims=True) + 1e-9

print("\nLearned grammar (category transitions, Hebbian only):")
cat_names = ['SUBJ', 'VERB', 'OBJ', 'PLACE', 'TIME']
for i in range(N_CATS):
    row = " ".join(f"{cat_P[i,j]:.2f}" for j in range(N_CATS))
    print(f"  {cat_names[i]:5s} → [{row}]  (true: next={cat_names[NEXT_CAT[i]]})")

# ---- baseline grammaticality -------------------------------------------------
def grammaticality(wseq):
    ok = tot = 0
    for a, b in zip(wseq[:-1], wseq[1:]):
        if NEXT_CAT[CAT[vocab[a]]] == CAT[vocab[b]]: ok += 1
        tot += 1
    return ok / tot if tot else 0.0

org.beta = 20
print("\nRunning baseline recall...")
slot_seq = org.recall(steps=200000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
gen = [slot_to_word[int(s)] for s in slot_seq if int(s) in slot_to_word][:1000]
gram_before = grammaticality(gen)
print(f"Baseline grammaticality: {gram_before:.3f}")

# ---- grammar-mask P ----------------------------------------------------------
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
            correct_slots = [j for j in range(n_mem)
                             if slot_cat.get(j) == NEXT_CAT[slot_cat[k]]]
            if correct_slots:
                P_grammar[k, correct_slots] = 1.0 / len(correct_slots)

org.Pn = P_grammar

# ---- post-mask grammaticality ------------------------------------------------
print("Running post-mask recall...")
slot_seq2 = org.recall(steps=200000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
gen2 = [slot_to_word[int(s)] for s in slot_seq2 if int(s) in slot_to_word][:1000]
gram_after = grammaticality(gen2)

oracle_seq = sample_sentence_stream(1000, seed=200)
random_seq = [int(rng.integers(N_WORDS)) for _ in range(1000)]
gram_oracle = grammaticality(oracle_seq)
gram_random = grammaticality(random_seq)

# ---- report ------------------------------------------------------------------
print("\n" + "="*62)
print("PHASE 5 -- SCALE TEST: 100 WORDS, 5 CATEGORIES\n")
print(f"N_WORDS={N_WORDS}  N_CATS={N_CATS}  DIM={DIM}  Memories={n_mem}")
print(f"Grammar: SUBJ→VERB→OBJ→PLACE→TIME→SUBJ  (p={P_CORRECT})\n")

print("GRAMMATICALITY:")
print(f"  Random (floor)          : {gram_random:.3f}")
print(f"  Organism (Hebbian only) : {gram_before:.3f}  ({(gram_before-gram_random)/(gram_oracle-gram_random)*100:.0f}% gap)")
print(f"  Organism (grammar-mask) : {gram_after:.3f}  ({(gram_after-gram_random)/(gram_oracle-gram_random)*100:.0f}% gap)")
print(f"  Oracle (ceiling)        : {gram_oracle:.3f}")

cats2 = [CAT[vocab[w]] for w in gen2]
bal = np.bincount(cats2, minlength=N_CATS) / len(cats2)
print(f"\nCategory balance: {np.round(bal, 3)}  (true ~[.2,.2,.2,.2,.2])")
print(f"Word coverage: {len(set(gen2))}/{N_WORDS}")

print(f"\nFirst 40 generated words (post-mask):")
print("  " + " ".join(vocab[w] for w in gen2[:40]))

print(f"\nKey result: does grammar-mask scale to 5-category grammar?")
print(f"  Hebbian → {gram_before:.3f},  mask → {gram_after:.3f},  oracle → {gram_oracle:.3f}")
