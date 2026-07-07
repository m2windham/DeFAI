"""
DEPLOYMENT-SCENARIO SIMULATION: how would this organism behave as a local AI?

The question this script answers with measurements instead of speculation:
if the current architecture (organism.py, unchanged) were deployed as an
always-on local learner, what would it actually do under realistic usage?

Four scenarios, all using the phase-4-style word world (26 pure words,
cyclic ANIMAL->ACTION->OBJECT grammar at P_CORRECT=0.88 -- so 0.88 is the
oracle ceiling for grammaticality, and ~0.33 is chance):

  A. COLD START -- learning curve. Fresh organism, streams of 500 / 1500 /
     4000 / 8000 words. How much experience until the vocabulary is covered
     and generated sequences approach the grammar ceiling? A local AI's
     "install-to-useful" time.
  B. NOISE ROBUSTNESS -- same as A at 4000 words but every token corrupted
     with Gaussian noise (sigma=0.3 relative). Sensor-grade input rather
     than clean symbols.
  C. CONCEPT DRIFT -- after 4000 words the world's grammar REVERSES
     (ANIMAL->OBJECT->ACTION). Windowed next-category prediction accuracy
     before and after the switch: does the organism adapt online, and how
     fast? No retraining, no reset -- the stream just changes.
  D. CONTINUAL LEARNING -- vocabulary arrives in two disjoint halves,
     3000 words each. After learning half B, does the organism still cover
     and generate half A (the catastrophic-forgetting question a
     local, always-learning AI lives or dies by)?

Everything runs on the stock Organism (perceive -> consolidate -> recall);
category labels are used only for scoring. Single seed, small field
(N=30, K=40): the point is the behavioral envelope, not records.
"""

import numpy as np
from organism import Organism, normalize

# ---- world ---------------------------------------------------------------------
ANIMALS = ['cat','dog','bird','horse','cow','pig','sheep','wolf']
ACTIONS = ['run','jump','swim','eat','sleep','hunt','hide','play']
OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
vocab = ANIMALS + ACTIONS + OBJECTS
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
ANIMAL, ACTION, OBJECT = 0, 1, 2
CAT = {}
for w in ANIMALS: CAT[word_to_idx[w]] = ANIMAL
for w in ACTIONS: CAT[word_to_idx[w]] = ACTION
for w in OBJECTS: CAT[word_to_idx[w]] = OBJECT
NEXT_FWD = {0: 1, 1: 2, 2: 0}
NEXT_REV = {0: 2, 2: 1, 1: 0}

DIM = 30; N = DIM; NORM = np.sqrt(N)
P_CORRECT = 0.88
HOLD = 12

emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, DIM))
cat_bases[0, 0:3] = 1.0; cat_bases[1, 3:6] = 1.0; cat_bases[2, 6:9] = 1.0
embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(vocab):
    embeddings[i] = 0.6*cat_bases[CAT[i]] + 0.4*emb_rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9


def sample_stream(n, next_cat, words_by_cat, seed):
    local = np.random.default_rng(seed)
    cat = ANIMAL
    seq = []
    for _ in range(n):
        pool = words_by_cat[cat]
        seq.append(word_to_idx[local.choice(pool)])
        cat = next_cat[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != next_cat[cat]]))
    return seq


FULL = {ANIMAL: ANIMALS, ACTION: ACTIONS, OBJECT: OBJECTS}


def frames(seq, noise=0.0, seed=0):
    local = np.random.default_rng(seed)
    out = []
    for w in seq:
        e = embeddings[w].copy()
        if noise > 0:
            e = e + noise * local.standard_normal(DIM)
        s = normalize(e.astype(complex), NORM)
        out.extend([s] * HOLD)
    return out


def word_slot_maps(org):
    """word -> best slot, slot -> best word (by memory overlap); eval only."""
    M = org.xi[org.used]
    used_idx = np.where(org.used)[0]
    states = np.array([normalize(embeddings[w].astype(complex), NORM) for w in range(N_WORDS)])
    ov = np.abs((M.conj() @ states.T) / N)          # slots x words
    w2s = {w: int(used_idx[np.argmax(ov[:, w])]) for w in range(N_WORDS)}
    s2w = {int(used_idx[i]): int(np.argmax(ov[i])) for i in range(len(used_idx))}
    return w2s, s2w, ov


def coverage(org, words):
    """Words that own a dedicated slot (their best slot maps back to them)."""
    w2s, s2w, _ = word_slot_maps(org)
    return sum(1 for w in words if s2w.get(w2s[word_to_idx[w]]) == word_to_idx[w]) / len(words)


def recall_grammaticality(org, next_cat, n_gen=400, steps=60000):
    kept = org.consolidate(merge_thresh=0.84, prune_frac=0.02)
    # slot -> word via compacted memory order
    M = org.mem
    states = np.array([normalize(embeddings[w].astype(complex), NORM) for w in range(N_WORDS)])
    ov = np.abs((M.conj() @ states.T) / N)
    s2w = {i: int(np.argmax(ov[i])) for i in range(M.shape[0])}
    seq = org.recall(steps=steps)
    gen = [s2w[int(s)] for s in seq][:n_gen]
    ok = tot = 0
    for a, b in zip(gen[:-1], gen[1:]):
        if next_cat[CAT[a]] == CAT[b]: ok += 1
        tot += 1
    gram = ok / max(tot, 1)
    shuf = gen.copy(); np.random.default_rng(0).shuffle(shuf)
    ok_s = sum(1 for a, b in zip(shuf[:-1], shuf[1:]) if next_cat[CAT[a]] == CAT[b])
    return gram, ok_s / max(len(shuf)-1, 1), len(gen)


def predict_accuracy(org, seq_eval):
    """Next-category prediction on held-out transitions using learned P."""
    w2s, s2w, _ = word_slot_maps(org)
    ok = tot = 0
    for a, b in zip(seq_eval[:-1], seq_eval[1:]):
        k = w2s[a]
        row = org.P[k]
        if row.sum() <= 0: continue
        j = int(np.argmax(row))
        if j in s2w:
            ok += CAT[s2w[j]] == CAT[b]; tot += 1
    return ok / max(tot, 1)


print("="*70)
print("A. COLD START -- learning curve (clean input)")
print(f"{'stream len':>11} {'coverage':>9} {'grammaticality':>15} {'shuffled':>9}  (oracle 0.88, chance ~0.33)")
for L in (500, 1500, 4000, 8000):
    org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
    seq = sample_stream(L, NEXT_FWD, FULL, seed=99)
    org.perceive(frames(seq), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    cov = coverage(org, vocab)
    gram, base, n_gen = recall_grammaticality(org, NEXT_FWD)
    print(f"{L:>11} {cov:>9.2f} {gram:>15.3f} {base:>9.3f}")

print("\n" + "="*70)
print("B. NOISE ROBUSTNESS -- 4000 words, per-token Gaussian corruption")
for sigma in (0.0, 0.3, 0.6):
    org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
    seq = sample_stream(4000, NEXT_FWD, FULL, seed=99)
    org.perceive(frames(seq, noise=sigma, seed=1), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    cov = coverage(org, vocab)
    gram, base, _ = recall_grammaticality(org, NEXT_FWD)
    print(f"  sigma={sigma}: coverage={cov:.2f}  grammaticality={gram:.3f}  (shuffled {base:.3f})")

print("\n" + "="*70)
print("C. CONCEPT DRIFT -- grammar reverses at word 4000 (no reset, no retraining)")
org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
chunks = []
for i in range(4):
    chunks.append(sample_stream(1000, NEXT_FWD, FULL, seed=100+i))
for i in range(4):
    chunks.append(sample_stream(1000, NEXT_REV, FULL, seed=200+i))
probe_fwd = sample_stream(800, NEXT_FWD, FULL, seed=7)
probe_rev = sample_stream(800, NEXT_REV, FULL, seed=8)
print(f"{'after chunk':>12} {'regime':>8} {'acc on FWD':>11} {'acc on REV':>11}")
for i, ch in enumerate(chunks):
    org.perceive(frames(ch), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
    regime = 'FWD' if i < 4 else 'REV'
    print(f"{i+1:>12} {regime:>8} {predict_accuracy(org, probe_fwd):>11.3f} "
          f"{predict_accuracy(org, probe_rev):>11.3f}")
print("  (adaptation = REV column rising after chunk 4; retention of the old")
print("   world = FWD column. Raw Hebbian counts never decay -- expect inertia.)")

print("\n" + "="*70)
print("D. CONTINUAL LEARNING -- vocabulary in two disjoint halves, 3000 words each")
half_a = {c: FULL[c][:len(FULL[c])//2] for c in FULL}
half_b = {c: FULL[c][len(FULL[c])//2:] for c in FULL}
words_a = sum(half_a.values(), []); words_b = sum(half_b.values(), [])
org = Organism(N=N, K=40, omega=0.15, beta=10.0, seed=0)
org.perceive(frames(sample_stream(3000, NEXT_FWD, half_a, seed=99)),
             g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
cov_a_before = coverage(org, words_a)
org.perceive(frames(sample_stream(3000, NEXT_FWD, half_b, seed=98)),
             g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
cov_a_after = coverage(org, words_a)
cov_b = coverage(org, words_b)
gram, base, n_gen = recall_grammaticality(org, NEXT_FWD)
both = sum(1 for w in set(sum(FULL.values(), []))
           if coverage(org, [w]) == 1.0)
print(f"  half-A coverage: before B = {cov_a_before:.2f}   after B = {cov_a_after:.2f}"
      f"   (forgetting = the drop)")
print(f"  half-B coverage after its 3000 words: {cov_b:.2f}")
print(f"  combined recall grammaticality: {gram:.3f}  (shuffled {base:.3f})")

print("\ndone -- see README 'capability envelope' for the interpretation")
