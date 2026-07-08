"""
PHASE 22 -- UNIFYING THE TRACKS: the full loop on real text
(perceive -> categories -> polysemy -> generate, zero labels anywhere)

The two research lines each solved half of this. The core track built the
perception stack (pooled probationary recruitment + soft ambiguity gating,
phases 14/17/18) and layered recall, all validated on synthetic worlds.
The language track (its phases 19-21) got category discovery and polysemy
detection working on real English, and layered generation working on the
synthetic polysemy world (its phase 18b) -- but never closed perceive ->
disambiguate -> generate on real text. This phase closes it, on the
committed fables corpus (real_text_corpus.py: ~2.7K-token stream, ~200
word vocabulary, real grammar, unplanted polysemy).

Stages (every one unsupervised; ground truth is never consulted):
  A. PERCEIVE -- three arms compared on word coverage:
     - lang-track recipe (its phase 19): plain perceive, recruit=0.75,
       15 epochs (the multi-epoch workaround for the recruit-floor bug);
     - core stack, 1 epoch: pool=True, confirm=3, amb=0.3 (phases 17+18;
       s_hat=0 since real tokens are noiseless -- the question is whether
       evidence pooling + the ambiguity gate handle Zipfian recurrence
       and correlated real embeddings without the recruit hack);
     - core stack, 3 epochs.
  B. CATEGORIES -- discover_categories_v2 (PPMI + k-means) on the best
     arm. Contributes to the open k-selection problem: alongside
     silhouette-argmax we report a BALANCE-argmax rule (pick k in 3..8
     maximizing min/max category size) -- phase 21 chose k by hand this
     way; here it is automated and the two rules are compared.
  C. POLYSEMY -- predictive split gain per word (Myhill-Nerode test, its
     phase 12) against a DIRECTLY MEASURED noise floor (its phase 21):
     3000 null draws with matched occurrence counts, 99th percentile.
  D. GENERATE -- layered generation (its phase 18b) rebuilt on the
     real-text organism: an emergent-category FSM estimated from
     LOW-GAIN (certified unambiguous) words only, within-category
     kernel-weighted word selection at context-drifted positions, word
     habituation. Scored without labels against corpus statistics:
     mean log2-likelihood of generated category bigrams under the
     corpus's category transitions, and the fraction of generated word
     bigrams that occur verbatim in the corpus -- both vs a
     random-word baseline.

RESULT: recorded in the follow-up commit, from the committed run.
"""

import re
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import PolysemyOrganism, _entropy
from real_text_corpus import CORPUS_TEXT

# ---------------------------------------------------------------- corpus
raw_tokens = re.findall(r"[a-zA-Z']+", CORPUS_TEXT.lower())
word_counts = Counter(raw_tokens)
vocab = sorted([w for w, c in word_counts.items() if c >= 2])
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]

# PPMI + SVD embeddings (its phase 19's recipe -- robust at tiny scale)
WINDOW = 4
cooc = np.zeros((N_WORDS, N_WORDS))
for i, w in enumerate(train_seq):
    for j in range(max(0, i - WINDOW), min(len(train_seq), i + WINDOW + 1)):
        if j != i:
            cooc[w, train_seq[j]] += 1.0
tot = cooc.sum()
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log((cooc * tot) / (cooc.sum(1, keepdims=True) @ cooc.sum(0, keepdims=True) + 1e-12) + 1e-12)
ppmi = np.maximum(pmi, 0.0)
U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
DIM = min(40, U.shape[1])
emb = U[:, :DIM] * np.sqrt(S[:DIM])
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
N = DIM; NORM = np.sqrt(N)
emb_c = emb.astype(complex)

def stream(seq, hold=8):
    for w in seq:
        for _ in range(hold):
            yield emb_c[w]

def coverage_map(org):
    """slot->word attribution by corpus-token assignment (its phase 19's
    scorer), plus word coverage."""
    org.consolidate(merge_thresh=0.84, prune_frac=0.001)
    n_mem = org.mem.shape[0]
    states = np.array([emb_c[w] for w in train_seq])
    assigns = np.abs((org.mem.conj() @ states.T) / N).argmax(0)
    slot_word = {}
    for k in range(n_mem):
        members = np.array(train_seq)[assigns == k]
        if len(members):
            slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
    return slot_word, len(set(slot_word.values())), n_mem


# ---------------------------------------------------------------- Stage A
print("PHASE 22: the unified loop on real text "
      f"({len(train_seq)} tokens, {N_WORDS} words, N={N})\n")
print("(A) perception arms -- word coverage")
K_CAP = min(1200, N_WORDS * 4)

orgs = {}
o = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
for _ in range(15):
    o.perceive(list(stream(train_seq)), g_in=5.0, dt=0.05, eta=0.02, recruit=0.75)
orgs['recipe (p19: recruit=0.75, 15 epochs)'] = o

o = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
o.perceive(list(stream(train_seq)), g_in=5.0, dt=0.05, eta=0.05, confirm=3,
           pool=True, s_hat=0.0, probation=12000, amb=0.3)
orgs['core x1 (pool+amb, single pass)'] = o

o = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
for _ in range(3):
    o.perceive(list(stream(train_seq)), g_in=5.0, dt=0.05, eta=0.05, confirm=3,
               pool=True, s_hat=0.0, probation=12000, amb=0.3)
orgs['core x3'] = o

best = None
for label, org in orgs.items():
    slot_word, cov, n_mem = coverage_map(org)
    print(f"  {label:<40} memories={n_mem:>3}  coverage={cov}/{N_WORDS}")
    if best is None or cov > best[1]:
        best = (label, cov, org, slot_word)
label, cov, org, slot_word = best
print(f"  -> continuing with: {label}")

# root cause of the core arms' collapse, measured: real embeddings are
# heavily correlated, so pool mode's constants (online fusion at 0.7,
# acceptance bars on a 0.8 scale) -- calibrated for near-orthogonal
# synthetic patterns -- merge DISTINCT words
ov_nn = np.abs(emb @ emb.T); np.fill_diagonal(ov_nn, 0)
nn = ov_nn.max(1)
adj = ov_nn > 0.7
seen = np.zeros(N_WORDS, bool); comps = 0
for i in range(N_WORDS):
    if not seen[i]:
        comps += 1; stack = [i]
        while stack:
            j = stack.pop()
            if not seen[j]:
                seen[j] = True
                stack.extend(np.where(adj[j] & ~seen)[0].tolist())
print(f"  root cause of the core arms' collapse: {int((nn > 0.7).sum())}/{N_WORDS} words "
      f"({(nn > 0.7).mean():.0%}) have an embedding neighbor above the 0.7 online-fusion "
      f"bar;\n  transitive fusion linkage alone collapses the vocabulary to {comps} "
      "components -- the pooled\n  stack's thresholds assume near-orthogonal patterns "
      "and real PPMI/SVD embeddings are not")

# ---------------------------------------------------------------- Stage B
print("\n(B) emergent categories: silhouette-argmax vs balance-argmax (k in 3..8)")
from polysemy_organism import _ppmi_transform, _kmeans_real, _silhouette_real
raw_counts = org.P[np.ix_(org.kept_idx, org.kept_idx)]
prof = _ppmi_transform(raw_counts)
prof = np.concatenate([prof, prof.T], axis=1)
prof /= (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)
stats = {}
for k in range(3, 9):
    labels_k, _ = _kmeans_real(prof, k, seed=3)
    sizes = np.bincount(labels_k, minlength=k)
    stats[k] = (float(_silhouette_real(prof, labels_k)),
                float(sizes.min() / max(sizes.max(), 1)), labels_k)
    print(f"  k={k}: silhouette={stats[k][0]:.3f}  balance={stats[k][1]:.3f}  "
          f"sizes={sorted(sizes.tolist(), reverse=True)}")
k_sil = max(stats, key=lambda k: stats[k][0])
k_bal = max(stats, key=lambda k: stats[k][1])
print(f"  silhouette-argmax: k={k_sil}   balance-argmax: k={k_bal}"
      f"   (phase 21 chose k=3 by hand at 547K words)")
# Neither selector is trusted for the GAIN statistic: its phase 20/21
# proved the noise-floor ceiling scales with category count (more
# conditioning bins -> higher resolvable-gain ceiling under pure noise),
# so the polysemy test runs at the smallest usable k -- the same
# balance-plus-floor criterion phase 21 applied by hand. Automatic
# k-selection remains OPEN: on this corpus both selectors prefer larger k.
K_CATS = 3
labels_cat = stats[K_CATS][2]
print(f"  -> stages C/D use k={K_CATS} (its phase-21 criterion: smallest k keeps "
      "the gain noise floor low)")
org.word_slot_to_cat = {i: int(labels_cat[i]) for i in range(len(labels_cat))}
org.cat_attractor = {c: normalize(org.mem[[i for i in range(len(labels_cat))
                                           if labels_cat[i] == c]].mean(0), NORM)
                     for c in sorted(set(labels_cat.tolist()))}
word_to_cat = {}
for s, w in slot_word.items():
    word_to_cat.setdefault(w, org.word_slot_to_cat.get(s))
for c in sorted(set(word_to_cat.values())):
    ws = sorted(vocab[w] for w, cc in word_to_cat.items() if cc == c)
    print(f"  category {c} ({len(ws)} words): {ws[:14]}{' ...' if len(ws) > 14 else ''}")

# ---------------------------------------------------------------- Stage C
print("\n(C) predictive-gain polysemy vs measured noise floor")
def split_gain(word_idx, min_occ=6):
    occ = [t for t, w in enumerate(train_seq) if w == word_idx]
    pairs = [(word_to_cat.get(train_seq[t - 1]), word_to_cat.get(train_seq[t + 1]))
             for t in occ if 0 < t < len(train_seq) - 1]
    pairs = [(p, s) for p, s in pairs if p is not None and s is not None]
    if len(pairs) < min_occ:
        return None
    succ = [s for _, s in pairs]; pred = [p for p, _ in pairs]
    H_u = _entropy(np.bincount(succ, minlength=K_CATS))
    n = len(pairs); H_c = 0.0
    for pc in set(pred):
        sub = [succ[i] for i in range(n) if pred[i] == pc]
        H_c += len(sub) / n * _entropy(np.bincount(sub, minlength=K_CATS))
    return dict(n=n, gain=max(H_u - H_c, 0.0))

gains = {w: r for w in range(N_WORDS) if (r := split_gain(w))}
rng = np.random.default_rng(5)
ns = [r['n'] for r in gains.values()]
null = []
for _ in range(3000):
    n = int(rng.choice(ns))
    succ = rng.integers(0, K_CATS, size=n); pred = rng.integers(0, K_CATS, size=n)
    H_u = _entropy(np.bincount(succ, minlength=K_CATS))
    H_c = sum((pred == pc).sum() / n *
              _entropy(np.bincount(succ[pred == pc], minlength=K_CATS))
              for pc in set(pred.tolist()))
    null.append(max(H_u - H_c, 0.0))
p99 = float(np.percentile(null, 99))
ranked = sorted(gains.items(), key=lambda kv: -kv[1]['gain'])
above = [(vocab[w], r['n'], r['gain']) for w, r in ranked if r['gain'] > p99]
print(f"  noise floor (99th pct of 3000 matched null draws): {p99:.4f}")
print(f"  words above floor ({len(above)}):")
for w, n, g in above[:12]:
    print(f"    {w:<12} n={n:>4}  gain={g:.3f}")
LOW_GAIN = {w for w, r in gains.items() if r['gain'] <= p99}
beta_gain = {w: (gains[w]['gain'] / ranked[0][1]['gain'] if w in gains else 0.0)
             for w in range(N_WORDS)}

# ---------------------------------------------------------------- Stage D
print("\n(D) layered generation on real text (its phase 18b, rebuilt here)")
xi_core = {w: org.mem[s] for s, w in
           sorted(slot_word.items(), key=lambda kv: org.count[org.kept_idx[kv[0]]])}
for w in range(N_WORDS):
    xi_core.setdefault(w, normalize(emb_c[w], NORM))

cat_ids = sorted(org.cat_attractor)
cat_idx = {c: i for i, c in enumerate(cat_ids)}
# category FSM from certified-unambiguous words only (18b's counting fix)
C = np.zeros((len(cat_ids), len(cat_ids)))
for a, b in zip(train_seq[:-1], train_seq[1:]):
    ca, cb = word_to_cat.get(a), word_to_cat.get(b)
    if a in LOW_GAIN and b in LOW_GAIN and ca is not None and cb is not None:
        C[cat_idx[ca], cat_idx[cb]] += 1
Cn = C / (C.sum(1, keepdims=True) + 1e-9)
# corpus-wide category bigram stats (ALL words) -- the scoring reference
C_all = np.zeros_like(C)
for a, b in zip(train_seq[:-1], train_seq[1:]):
    ca, cb = word_to_cat.get(a), word_to_cat.get(b)
    if ca is not None and cb is not None:
        C_all[cat_idx[ca], cat_idx[cb]] += 1
C_all_n = C_all / (C_all.sum(1, keepdims=True) + 1e-9)

words_in_cat = {c: [w for w, cc in word_to_cat.items()
                    if cc == c and w in LOW_GAIN] for c in cat_ids}
high_gain = [w for w in range(N_WORDS) if w not in LOW_GAIN and w in gains]
for c in cat_ids:
    words_in_cat[c] = words_in_cat[c] + high_gain

def drifted(w, prev_w, alpha=0.35):
    core = xi_core[w]
    pc = word_to_cat.get(prev_w) if prev_w is not None else None
    b = beta_gain.get(w, 0.0)
    if pc is not None and pc in org.cat_attractor and b > 0:
        return normalize(core + b * alpha * org.cat_attractor[pc], NORM)
    return core

def generate(n_words=400, seed=1, beta_sim=20.0, lam_h=6.0, tau_h=3.0,
             g_rec=6.0, dt=0.05, Dn=0.003, settle=6):
    rng = np.random.default_rng(seed)
    h = np.zeros(N_WORDS)
    cur_cat = cat_ids[int(rng.integers(len(cat_ids)))]
    cur_w = int(rng.integers(N_WORDS))
    z = drifted(cur_w, None)
    out = [cur_w]
    for _ in range(n_words - 1):
        nxt_cat = cat_ids[int(rng.choice(len(cat_ids), p=Cn[cat_idx[cur_cat]]))]
        cand = words_in_cat[nxt_cat] or list(range(N_WORDS))
        pos = np.array([drifted(c, cur_w) for c in cand])
        sims = np.abs((pos.conj() @ z) / N)
        fat = np.maximum(1 - lam_h * h[cand], 0.0)
        sc = sims * fat
        if sc.sum() <= 1e-9:
            sc = sims
        p = np.exp(beta_sim * (sc - sc.max())); p /= p.sum()
        nxt_w = cand[int(rng.choice(len(cand), p=p))]
        target = drifted(nxt_w, cur_w)
        for _ in range(settle):
            noise = np.sqrt(2 * Dn * dt) * (rng.standard_normal(N) +
                                            1j * rng.standard_normal(N)) / np.sqrt(2)
            z = normalize(z + dt * (1j * 0.15 * z + g_rec * (target - z)) + noise, NORM)
            h += dt / tau_h * (-h)
        h[nxt_w] = min(h[nxt_w] + 1.0, 1.0)
        out.append(nxt_w)
        cur_w, cur_cat = nxt_w, nxt_cat
    return out

def cat_loglik(seq):
    lps = []
    for a, b in zip(seq[:-1], seq[1:]):
        ca, cb = word_to_cat.get(a), word_to_cat.get(b)
        if ca is not None and cb is not None:
            lps.append(np.log2(C_all_n[cat_idx[ca], cat_idx[cb]] + 1e-9))
    return float(np.mean(lps)) if lps else float('nan')

corpus_bigrams = set(zip(train_seq[:-1], train_seq[1:]))
def bigram_hit(seq):
    return float(np.mean([(a, b) in corpus_bigrams for a, b in zip(seq[:-1], seq[1:])]))

gen = generate(400, seed=1)
rng_b = np.random.default_rng(42)
rand = [int(rng_b.integers(N_WORDS)) for _ in range(800)]
shuf = gen.copy(); rng_b.shuffle(shuf)
print(f"  category log2-likelihood/hop: generated={cat_loglik(gen):.3f}  "
      f"shuffled={cat_loglik(shuf):.3f}  random={cat_loglik(rand):.3f}  "
      f"(uniform chance={np.log2(1/len(cat_ids)):.3f})")
print(f"  corpus-bigram hit rate: generated={bigram_hit(gen):.3f}  "
      f"random={bigram_hit(rand):.3f}")
print(f"  word coverage in generation: {len(set(gen))}/{N_WORDS}")
print(f"  sample: {' '.join(vocab[w] for w in gen[:30])}")

ok_cov = cov >= 0.8 * N_WORDS
ok_gen = (cat_loglik(gen) > cat_loglik(rand) + 0.3 and
          bigram_hit(gen) > 3 * bigram_hit(rand))
print("\nverdict:", ("THE LOOP CLOSES ON REAL TEXT -- core perception stack covers "
      f"{cov}/{N_WORDS} words, categories + polysemy + layered generation run "
      "end-to-end unsupervised with structure well above chance"
      if ok_cov and ok_gen else
      "partial -- the loop runs but a stage is below bar; see the stage tables"))
