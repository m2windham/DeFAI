"""
PHASE 20B -- analysis on the large-corpus trained organism (loads state
saved by phase20_large_corpus.py, avoiding re-running the ~12-minute
training step).
"""
import re
import glob
import pickle
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import PolysemyOrganism, _entropy

# ---- rebuild the CHEAP part (tokenize/vocab, ~1s) independently, so this
# script never imports phase20_large_corpus.py directly -- that module's
# top level also runs the ~14-minute embeddings+training pipeline, which
# we've already run once (saved to /tmp/phase20_*) and don't want to
# silently re-trigger just by importing it for its vocab objects. -----------
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
    all_text.append(text[start:end])
full_text = "\n".join(all_text)
raw_tokens = re.findall(r"[a-zA-Z']+", full_text.lower())

word_counts = Counter(raw_tokens)
MIN_COUNT = 150
vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]

CANDIDATE_POLYSEMOUS = ['watch', 'light', 'bank', 'spring', 'duck', 'bear', 'run',
                        'well', 'fair', 'kind', 'still', 'right', 'left', 'match',
                        'train', 'fly', 'rock', 'park', 'book', 'fine', 'saw', 'letter']
present = [w for w in CANDIDATE_POLYSEMOUS if w in word_to_idx]
print(f"Rebuilt vocab: {N_WORDS} words, {len(train_seq)} training tokens (matches prior run)")

mem = np.load('/tmp/phase20_org_mem.npy')
assigns = np.load('/tmp/phase20_assigns.npy')
with open('/tmp/phase20_state.pkl', 'rb') as f:
    state = pickle.load(f)
slot_word = state['slot_word']
n_mem = state['n_mem']
N = mem.shape[1]

# rebuild a lightweight PolysemyOrganism shell just to reuse discover_categories()
org = PolysemyOrganism(N=N, K=n_mem, omega=0.15, beta=10.0, seed=0)
org.xi[:n_mem] = mem
org.used[:n_mem] = True
org.Pn = state['org_Pn']
org.mem = mem

print(f"Loaded: {n_mem} memories, {len(set(slot_word.values()))}/{N_WORDS} word coverage")

# ---- category discovery ------------------------------------------------------
print("\nDiscovering emergent categories...")
result = org.discover_categories(
    thresh_sweep=(0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99,0.995,0.999,0.9999),
    target_k=10, eta=0.15, seed=3, verbose=True)
print(f"\nFound {result['n_categories']} emergent categories at threshold={result['threshold']}")

word_to_emergent_cat = {}
for k, w in slot_word.items():
    word_to_emergent_cat[w] = org.word_slot_to_cat.get(k)

cats_sorted = sorted(set(c for c in word_to_emergent_cat.values() if c is not None))
for c in cats_sorted:
    words_in_c = [vocab[w] for w, cc in word_to_emergent_cat.items() if cc == c]
    print(f"\nCategory {c} ({len(words_in_c)} words): {sorted(words_in_c)[:40]}"
          f"{'...' if len(words_in_c) > 40 else ''}")

# ---- predictive-gain analysis -------------------------------------------------
print("\n" + "="*72)
print("PREDICTIVE-GAIN ANALYSIS on large real corpus\n")

def successor_emergent_cat(t):
    if t + 1 >= len(train_seq):
        return None
    return word_to_emergent_cat.get(train_seq[t+1])

def predecessor_emergent_cat(t):
    if t - 1 < 0:
        return None
    return word_to_emergent_cat.get(train_seq[t-1])

def predictive_split_gain(word_idx, min_occ=100):
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
    r = predictive_split_gain(w, min_occ=100)
    if r:
        gains.append((vocab[w], r['n'], r['gain']))
gains.sort(key=lambda x: -x[2])

print(f"{'Word':<15}{'N occ':>8}{'Gain (bits)':>14}")
print("-"*40)
for w, n, g in gains[:30]:
    marker = "  <-- CANDIDATE" if w in present else ""
    print(f"{w:<15}{n:>8}{g:>14.3f}{marker}")

print("\nBottom 15 (most 'unambiguous'):")
for w, n, g in gains[-15:]:
    marker = "  <-- CANDIDATE" if w in present else ""
    print(f"{w:<15}{n:>8}{g:>14.3f}{marker}")

print(f"\nCandidate polysemous words -- where did they rank?")
gain_lookup = {w: (n, g) for w, n, g in gains}
for w in present:
    if w in gain_lookup:
        n, g = gain_lookup[w]
        rank = [x[0] for x in gains].index(w) + 1
        print(f"  '{w}': n={n}  gain={g:.3f}  rank={rank}/{len(gains)}")
    else:
        print(f"  '{w}': insufficient occurrences (<100) for gain calc")

# ---- noise floor calibration at THIS run's actual n and category count -------
print("\n" + "="*72)
print("NOISE FLOOR CALIBRATION at this run's actual scale\n")
n_cats_actual = result['n_categories']
ns = [n for w, n, g in gains]
rng = np.random.default_rng(5)
def entropy_np(counts):
    counts = np.asarray(counts, dtype=float); p = counts/counts.sum()
    p = p[p>0]; return -np.sum(p*np.log2(p))

noise_gains = []
for trial in range(3000):
    n = int(rng.choice(ns))
    succ = rng.integers(0, n_cats_actual, size=n)
    pred = rng.integers(0, n_cats_actual, size=n)
    uncond = entropy_np(np.bincount(succ, minlength=n_cats_actual))
    cond = 0.0
    for pc in set(pred.tolist()):
        idx = pred == pc
        cond += idx.sum()/n * entropy_np(np.bincount(succ[idx], minlength=n_cats_actual))
    noise_gains.append(max(uncond - cond, 0))
noise_gains = np.array(noise_gains)
print(f"n_categories={n_cats_actual}, sample sizes matched to this run's actual word occurrence counts")
print(f"  noise floor: mean={noise_gains.mean():.4f}  90th={np.percentile(noise_gains,90):.4f}  "
      f"99th={np.percentile(noise_gains,99):.4f}  max={noise_gains.max():.4f}")

n_above_99th = sum(1 for w, n, g in gains if g > np.percentile(noise_gains, 99))
print(f"\nReal words with gain above 99th-percentile noise floor: {n_above_99th}/{len(gains)}")
print("These are the words with STATISTICALLY DEFENSIBLE polysemy signal:")
for w, n, g in gains:
    if g > np.percentile(noise_gains, 99):
        marker = "  <-- matches our candidate list!" if w in present else ""
        print(f"  {w:<15} n={n:<6} gain={g:.3f}{marker}")
