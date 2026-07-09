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

RESULT (recorded from the committed run):
  - PERCEIVE: the language-track recipe wins; re-sweeping its recruit
    floor on the extended corpus lifts coverage 320 -> 335/376 at
    recruit=0.85 (0.9 over-fragments, 316). The core pooled/ambiguity
    stack COLLAPSES on real text (100/376 in one pass, and more epochs
    do not help) -- root cause measured: 68% of PPMI/SVD embeddings
    have a neighbor above the 0.7 online-fusion bar, and transitive
    fusion linkage alone collapses the vocabulary to 162 components.
    Pool-mode constants assume near-orthogonal patterns; real word
    embeddings are not. Open: embedding decorrelation or
    correlation-aware calibration of the fusion/acceptance bars.
  - CATEGORIES: no real structure at this corpus scale -- best
    silhouette 0.011 across k=3..8 even on word-level aggregated
    PPMI profiles, with a 291/335-word blob at k=3. Selector
    comparison (open thread): silhouette-argmax picks k=8,
    balance-argmax k=5; neither reproduces phase 21's k=3, and with
    silhouettes this flat neither should be trusted. This is the
    language track's own scale wall (its phases 19-20): category
    statistics need ~100x more text than 2.4K tokens.
  - POLYSEMY: the per-word permutation null (methodological upgrade
    over phase 21's pooled uniform-draw floor -- it preserves both
    marginals) leaves 12 testable candidates (n>=20), of which one
    ('for', gain 0.628 vs own p99 0.405) clears its null. With a
    degenerate category blob the conditioning variable is nearly
    constant, so detection power here is near zero by construction;
    the statistic, not the mechanism, is starved.
  - GENERATE: word-level structure transfers cleanly -- 20.1% of
    generated bigrams occur verbatim in the corpus vs 1.8% random
    (11x), with 221/376 words used. The category-likelihood score is
    UNINFORMATIVE under a blob (random -0.799 outscores generated
    -1.023 by living inside the dominant category) -- recorded as a
    scoring lesson, not a generation failure.
  - VERDICT: the loop is WIRED -- all four stages run end-to-end
    unsupervised on real text -- and the binding constraint is now a
    single measured quantity: corpus scale for category emergence.
    Next lever: run this exact loop at the language track's 547K-word
    scale, where categories are known to form (its phase 21).
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

# the recruit floor re-swept on THIS corpus (phase 19 tuned 0.75 on an
# earlier, smaller version of the text): 0.85 is the new peak, 0.9 over-
# fragments and coverage falls back
o = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
for _ in range(15):
    o.perceive(list(stream(train_seq)), g_in=5.0, dt=0.05, eta=0.02, recruit=0.85)
orgs['recipe, recruit=0.85'] = o

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
# WORD-level transition profiles: aggregate slot-level Hebbian counts by
# each slot's attributed word. Clustering raw slots (its phases 19/21) is
# hostage to slot duplication -- the recruit=0.85 arm carries ~2 slots per
# word, and duplicate slots' sparse split profiles smear into one giant
# blob. Aggregation makes stage B invariant to how perception divides a
# word across slots.
covered = sorted(set(slot_word.values()))
w_of = {w: i for i, w in enumerate(covered)}
raw_slot = org.P[np.ix_(org.kept_idx, org.kept_idx)]
word_counts_P = np.zeros((len(covered), len(covered)))
for si, wa in slot_word.items():
    for sj, wb in slot_word.items():
        word_counts_P[w_of[wa], w_of[wb]] += raw_slot[si, sj]
prof = _ppmi_transform(word_counts_P)
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
word_to_cat = {covered[i]: int(labels_cat[i]) for i in range(len(covered))}
org.cat_attractor = {}
for c in sorted(set(labels_cat.tolist())):
    slots_c = [s for s, w in slot_word.items() if word_to_cat.get(w) == c]
    if slots_c:
        org.cat_attractor[c] = normalize(org.mem[slots_c].mean(0), NORM)
for c in sorted(set(word_to_cat.values())):
    ws = sorted(vocab[w] for w, cc in word_to_cat.items() if cc == c)
    print(f"  category {c} ({len(ws)} words): {ws[:14]}{' ...' if len(ws) > 14 else ''}")

# ---------------------------------------------------------------- Stage C
print("\n(C) predictive-gain polysemy vs per-word permutation null")
def cond_gain(pred, succ):
    H_u = _entropy(np.bincount(succ, minlength=K_CATS))
    n = len(succ); H_c = 0.0
    for pc in set(pred.tolist()):
        sub = succ[pred == pc]
        H_c += len(sub) / n * _entropy(np.bincount(sub, minlength=K_CATS))
    return max(H_u - H_c, 0.0)

def split_gain(word_idx, min_occ=20, n_perm=500, rng=None):
    """Gain plus the word's OWN permutation null: shuffling the predecessor
    labels against the successor labels preserves both marginals exactly,
    which the phase-21 uniform-draw null did not -- with a skewed category
    marginal the uniform null overstates achievable spurious gain and
    buries real signal. min_occ=20: below that even the permutation null's
    p99 exceeds the entropy ceiling and no word could ever clear it."""
    occ = [t for t, w in enumerate(train_seq) if w == word_idx]
    pairs = [(word_to_cat.get(train_seq[t - 1]), word_to_cat.get(train_seq[t + 1]))
             for t in occ if 0 < t < len(train_seq) - 1]
    pairs = [(p, s) for p, s in pairs if p is not None and s is not None]
    if len(pairs) < min_occ:
        return None
    pred = np.array([p for p, _ in pairs]); succ = np.array([s for _, s in pairs])
    g = cond_gain(pred, succ)
    null = []
    for _ in range(n_perm):
        null.append(cond_gain(rng.permutation(pred), succ))
    return dict(n=len(pairs), gain=g, p99=float(np.percentile(null, 99)))

rng = np.random.default_rng(5)
gains = {w: r for w in range(N_WORDS) if (r := split_gain(w, rng=rng))}
ranked = sorted(gains.items(), key=lambda kv: -(kv[1]['gain'] - kv[1]['p99']))
above = [(vocab[w], r['n'], r['gain'], r['p99']) for w, r in ranked
         if r['gain'] > r['p99']]
print(f"  candidates with n>=20 occurrences: {len(gains)}")
print(f"  words above their own permutation-null p99 ({len(above)}):")
for w, n, g, p in above[:12]:
    print(f"    {w:<12} n={n:>4}  gain={g:.3f}  null p99={p:.3f}")
# untested rare words (n<20) count as unambiguous: no evidence of polysemy
LOW_GAIN = {w for w in range(N_WORDS)
            if w not in gains or gains[w]['gain'] <= gains[w]['p99']}
max_gain = max((r['gain'] for r in gains.values()), default=1.0) or 1.0
beta_gain = {w: (gains[w]['gain'] / max_gain if w in gains else 0.0)
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
ok_words = bigram_hit(gen) > 5 * bigram_hit(rand)
ok_cats = (cat_loglik(gen) > cat_loglik(rand) + 0.3 and
           max(stats[k][0] for k in stats) > 0.05)
sil_max = max(stats[k][0] for k in stats)
blob = max(np.bincount(labels_cat))
if ok_cov and ok_words and ok_cats:
    print(f"\nverdict: THE LOOP CLOSES ON REAL TEXT -- coverage {cov}/{N_WORDS}, "
          "category flow and word structure both well above chance, end-to-end "
          "unsupervised")
elif ok_cov and ok_words:
    print(f"\nverdict: THE LOOP IS WIRED AND WORD-LEVEL STRUCTURE TRANSFERS -- "
          f"coverage {cov}/{N_WORDS}, generated word bigrams "
          f"{bigram_hit(gen)/max(bigram_hit(rand),1e-9):.0f}x random -- but "
          f"CATEGORY EMERGENCE IS THE MEASURED BOTTLENECK at this corpus scale "
          f"(best silhouette {sil_max:.3f}, largest category {blob}/{len(covered)} "
          "words): the same scale wall its phases 19-20 measured. The category-"
          "likelihood score is uninformative under a degenerate blob (random can "
          "outscore the generator by living inside it). Next lever: corpus scale, "
          "not mechanism")
else:
    print("\nverdict: partial -- the loop runs but a stage is below bar; "
          "see the stage tables")
