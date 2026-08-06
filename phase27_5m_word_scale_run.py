"""
PHASE 27 -- THE UNIFIED LOOP AT 5M WORDS, WITH PHASE 24's LABEL-FREE
CATEGORY-VALIDITY CRITERION WIRED INTO STAGE B (T2.1)

Phase 23 ran the unified loop (perceive -> categories -> polysemy ->
generation) at 547K words and relieved the category-emergence bottleneck,
but its own stage-B criterion (silhouette-argmax) was later shown (phase 24)
to be the wrong objective: silhouette stayed flat (~0.02-0.04) regardless of
category quality, and the two textbook predictive fixes (MDL, held-out
class-bigram perplexity) both provably over-split to the finest k. Phase 24
replaced silhouette with two label-free criteria built directly on the
project's own framing (phase 12's "does knowing X change the prediction"):
class-bigram mutual information I(C_t; C_{t+1}) against a DIRECTLY MEASURED
permutation null (VALIDITY), and category-profile DISTINCTNESS-argmax
(k-SELECTION, a parsimony criterion, since predictive criteria structurally
cannot select k). At 547K words / 395 words / 408K in-vocab pairs, phase 24
measured: distinctness selected k=6, MI-vs-null z=65 at that k, and k=6 also
matched the oracle POS V-measure-argmax (0.547) -- unseen by the criterion.

This phase is the scale replication phase 24 named as its own follow-on and
that FABLE_HANDOFF/AGENT_TARGETS/ROADMAP list as T2.1: 50-100 Gutenberg
books (this run: 42 verified books, ~5.22M raw words, chosen to hit the
5M-word target primarily; the 8-book 547K corpus averaged ~68K words/book,
so ~50 books of similar length would undershoot 5M -- longer classics were
mixed in instead), phase 24's VALIDITY + k-SELECTION criteria wired directly
into stage B (replacing phase 23's fixed k=3 / silhouette sweep), stage C's
per-word permutation-null polysemy test carried over unchanged (its null
gets DIRECTLY TIGHTER here: same procedure, ~9x the tokens per word ->
lower-variance permutation null, a harder bar to clear -- exactly what
"under tighter nulls" in the pre-registration means; no code change needed
to produce this effect).

PRE-REGISTERED PREDICTIONS (written before the committed run; standing rule
in FABLE_HANDOFF/AGENT_TARGETS -- record outcomes below whether they hit or
miss, partials included):

  (P1) CATEGORY STRUCTURE SHARPENS. At the distinctness-selected k, the
       MI-vs-permutation-null z-score at 5M words exceeds phase 24's z=65 at
       547K words (more tokens -> a tighter, lower-variance null -> a
       measured gain in z is the operational meaning of "sharpens", not an
       eyeballed one). The selected k itself should be STABLE or GRACEFULLY
       LARGER than phase 24's k=6 (more text can resolve finer-grained but
       still-distinct grammatical roles; a collapse to k<=3 or an unbounded
       run to the sweep ceiling would both falsify "sharpens").

  (P2) POLYSEMY DETECTIONS HOLD UNDER TIGHTER NULLS. The word "right"
       (phase 21/23's headline lexical item) continues to clear its own
       per-word permutation null at 5M words, and the honest, falsifiable
       part: its margin (gain - null p99) should NOT need to shrink for the
       detection to survive -- with ~9x the occurrences the null's variance
       drops on its own, so survival despite a tighter null is the
       interesting result, not a foregone one (a corpus-scale confound --
       e.g. more genres diluting "right"'s conditioning categories -- could
       still sink it). Same measured, not assumed, standard for the wider
       word list: report the honest full ranking, not cherry-picked.

  Scope decision, pre-registered: the oracle POS V-measure cross-check
  (phase 24's evaluation-only arm) is DROPPED for this run -- nltk POS
  tagging over 5M raw tokens is evaluation-only scaffolding, not one of the
  two pre-registered questions, and its runtime (untagged at 547K it was
  already the single slowest stage) would consume the runtime budget this
  row is supposed to spend on the mechanism itself. If time remains after
  the committed run, it is a natural follow-up, not a silent omission.

  Runtime budget (pre-registered, from e2_benchmark.py on this host): 3-epoch
  perceive at phase-23 shape (547K-word corpus) measured 105.8s at 46.3K
  frames/s on this 4-core host (vs the pinned-hardware 67.3K frames/s in
  FABLE_HANDOFF -- this host is slower, so its own number is the honest
  planning basis). At MIN_COUNT scaled proportionally to corpus growth (see
  below), in-vocab tokens grow ~8.8x -> perceive stage budgeted at
  ~8.8x105.8s =~ 15-16 min for ONE arm. Stage-A DEVIATION FROM PHASE 23,
  pre-registered: phase 23 swept 3 perception arms and phase 27 runs only
  the arm phase 23 already measured as the winner at this exact corpus
  family (recipe, recruit=0.75, 3 epochs) -- re-sweeping arms is not one of
  this row's two pre-registered questions, and 3 arms would triple the
  single most expensive stage for no new answer. This is scope control, not
  hiding a result: if the winning arm turns out NOT to be the best choice at
  5M words, that would itself be a finding, reported honestly below.

  Vocabulary-size deviation, pre-registered: phase 23/24 used MIN_COUNT=150
  at 547K raw words (395-word vocab). Held constant at 5.22M raw words that
  bar produces a 2863-word vocabulary -- K_CAP would hit its 2000-slot
  ceiling and, more importantly, the PPMI-SVD embedding step is O(V^3)
  (LAPACK SVD of a dense VxV matrix): V=2863 is computationally a different
  regime than V=395, not a proportional scale-up. MIN_COUNT is instead
  scaled proportionally to corpus growth (150 * 5.22M/547K =~ 1430, rounded
  to 1500) to hold vocabulary size in the same computational and
  statistical-power regime as phase 23/24 (354 words here vs 395 there)
  while giving each surviving word ~8.8x more occurrences -- the change that
  actually sharpens both stage-B statistics and stage-C nulls. This is the
  intended scale lever, not a workaround.
"""

import os
import re
import glob
import time
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import (PolysemyOrganism, _entropy, _ppmi_transform,
                               _kmeans_real, _silhouette_real)

T_START = time.time()


def elapsed():
    return f"[{time.time() - T_START:6.0f}s]"


# ---------------------------------------------------------------- corpus
CORPUS_DIR = "/tmp/gutenberg_corpus_5m"
# 42 books verified by title match at fetch time (title substring from the
# Gutenberg header checked against the expected title before being kept --
# guards against a wrong/stale ebook id silently injecting mismatched text).
BOOKS = {
    11: "alice", 12: "looking_glass", 55: "wizard_oz", 16: "peter_pan",
    74: "tom_sawyer", 76: "huck_finn", 1661: "sherlock_adventures",
    1400: "great_expectations", 98: "tale_of_two_cities", 1342: "pride_prejudice",
    84: "frankenstein", 345: "dracula", 174: "dorian_gray", 43: "jekyll_hyde",
    2600: "war_and_peace", 1260: "jane_eyre", 768: "wuthering_heights",
    46: "christmas_carol", 219: "heart_of_darkness", 36: "war_of_worlds",
    35: "time_machine", 164: "20k_leagues", 103: "around_world_80days",
    120: "treasure_island", 158: "emma", 105: "persuasion",
    121: "northanger_abbey", 141: "mansfield_park", 161: "sense_sensibility",
    205: "walden", 844: "importance_of_being_earnest", 996: "don_quixote",
    2554: "crime_and_punishment", 600: "notes_from_underground",
    135: "les_miserables", 203: "uncle_toms_cabin", 244: "study_in_scarlet",
    108: "return_of_sherlock_holmes", 2852: "hound_of_baskervilles",
    829: "gullivers_travels", 1998: "thus_spake_zarathustra", 1232: "the_prince",
}
paths = sorted(glob.glob(f"{CORPUS_DIR}/*.txt"))
if not paths:
    print("Corpus missing. Re-fetch (public-domain, re-fetchable; each file "
          "was verified by Gutenberg header Title: match at fetch time -- "
          "keep that check when re-fetching) with:\n")
    print(f"  mkdir -p {CORPUS_DIR}")
    for bid, name in BOOKS.items():
        print(f"  curl -sS -o {CORPUS_DIR}/{bid}.txt "
              f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt")
    raise SystemExit("\nRe-run this script once the corpus is present.")

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                             re.IGNORECASE | re.DOTALL)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
                           re.IGNORECASE | re.DOTALL)
all_text = []
for path in paths:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m_s = GUTENBERG_START.search(text)
    m_e = GUTENBERG_END.search(text)
    all_text.append(text[m_s.end() if m_s else 0: m_e.start() if m_e else len(text)])
raw_tokens = re.findall(r"[a-zA-Z']+", "\n".join(all_text).lower())

MIN_COUNT = 1500   # scaled proportionally from phase 23/24's 150 @ 547K raw
word_counts = Counter(raw_tokens)
vocab = sorted([w for w, c in word_counts.items() if c >= MIN_COUNT])
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
train_seq = [word_to_idx[w] for w in raw_tokens if w in word_to_idx]

occ_by_word = {w: [] for w in range(N_WORDS)}
for t, w in enumerate(train_seq):
    occ_by_word[w].append(t)

print(f"PHASE 27: the unified loop at 5M-word scale ({len(paths)} books, "
      f"{len(raw_tokens)} raw tokens, {len(train_seq)} in-vocab, "
      f"{N_WORDS} words, count>={MIN_COUNT})  {elapsed()}\n")

# PPMI + SVD embeddings (phase 20/23 recipe, unchanged)
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
emb = U[:, :DIM] * np.sqrt(S[:DIM])
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
N = DIM; NORM = np.sqrt(N)
emb_c = emb.astype(complex)
print(f"  embeddings built (V={N_WORDS}, DIM={DIM})  {elapsed()}\n")


def make_stream(seq, hold=4):
    for w in seq:
        s = emb_c[w]
        for _ in range(hold):
            yield s


def coverage_map(org, chunk=50000):
    """slot->word attribution by corpus-token assignment. Chunked over
    tokens (bounded memory): the naive one-shot (n_mem x N_TOKENS) complex
    matmul is ~32 GB at this corpus's token count -- the same one-shot-
    broadcast memory blowup phase 36 hit and fixed in _silhouette_real."""
    org.consolidate(merge_thresh=0.84, prune_frac=0.0005)
    n_mem = org.mem.shape[0]
    train_arr = np.array(train_seq)
    assigns = np.empty(len(train_seq), dtype=np.int64)
    for i in range(0, len(train_seq), chunk):
        states = emb_c[train_arr[i:i + chunk]]
        assigns[i:i + chunk] = np.abs((org.mem.conj() @ states.T) / N).argmax(0)
    slot_word = {}
    for k in range(n_mem):
        members = np.array(train_seq)[assigns == k]
        if len(members):
            slot_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
    return slot_word, len(set(slot_word.values())), n_mem


# ---------------------------------------------------------------- Stage A
print("(A) perception -- single arm (phase 23's measured winner at this "
      "corpus family; not re-swept, see pre-registration)")
K_CAP = min(2000, N_WORDS * 4)


def train_recipe(recruit, epochs):
    o = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
    for ep in range(epochs):
        o.perceive(make_stream(train_seq), g_in=5.0, dt=0.05, eta=0.02, recruit=recruit)
        print(f"    epoch {ep+1}/{epochs} done  {elapsed()}")
    return o


CHECKPOINT = "/tmp/phase27_stageA_checkpoint.npz"
t0 = time.time()
if os.path.exists(CHECKPOINT):
    import organism_state
    org = organism_state.load_state(CHECKPOINT, cls=PolysemyOrganism)
    print(f"  loaded stage-A checkpoint from {CHECKPOINT} (skipping perceive)  "
          f"{elapsed()}")
else:
    org = train_recipe(0.75, 3)
    import organism_state
    organism_state.save_state(org, CHECKPOINT)
    print(f"  stage-A checkpoint saved to {CHECKPOINT}  {elapsed()}")
slot_word, cov, n_mem = coverage_map(org)
print(f"  recipe (recruit=0.75, 3 epochs): memories={n_mem}  "
      f"coverage={cov}/{N_WORDS}  ({time.time()-t0:.0f}s)  {elapsed()}\n")

# ---------------------------------------------------------------- Stage B
print("(B) emergent categories -- phase 24's VALIDITY (MI-vs-null) + "
      "k-SELECTION (distinctness-argmax), replacing phase 23's fixed k=3 / "
      "silhouette sweep")
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

# word-level bigram pairs over the FULL corpus, restricted to covered words
# (the object phase 24's criteria are computed on -- corpus statistics, not
# oscillator counts, isolating the criterion from perception noise exactly
# as phase 24 did)
cw = sorted(covered)
cidx = {w: i for i, w in enumerate(cw)}
pair_seq = [cidx[w] for w in train_seq if w in cidx]
pred_arr = np.array(pair_seq[:-1]); succ_arr = np.array(pair_seq[1:])
V_COV = len(cw)


def cat_bigram(labels, k):
    lp, ls = labels[pred_arr], labels[succ_arr]
    return np.bincount(lp * k + ls, minlength=k * k).reshape(k, k).astype(float)


def mutual_information(M):
    p = M / M.sum()
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = p * np.log2(p / (pi @ pj + 1e-300) + 1e-300)
    return float(term[p > 0].sum())


def distinctness(labels, k):
    M = cat_bigram(labels, k)
    R = M / (M.sum(1, keepdims=True) + 1e-9)
    Cc = (M / (M.sum(0, keepdims=True) + 1e-9)).T
    P = np.concatenate([R, Cc], axis=1)
    return float(min(np.linalg.norm(P[i] - P[j])
                     for i in range(k) for j in range(i + 1, k)))


print(f"  {'k':>3}  {'silhou':>7}  {'MI':>6}  {'nullp99':>8}  {'MIz':>5}  "
      f"{'distinct':>8}  sizes  {elapsed()}")
rng_b = np.random.default_rng(7)
rows = {}
for k in range(2, 13):
    labels_full = np.zeros(N_WORDS, dtype=int)
    labels_k, _ = _kmeans_real(prof, k, seed=3)
    for i, w in enumerate(covered):
        labels_full[w] = labels_k[i]
    labels_pairspace = np.array([labels_full[w] for w in cw])
    sizes = np.bincount(labels_k, minlength=k)
    sil = _silhouette_real(prof, labels_k)
    mi = mutual_information(cat_bigram(labels_pairspace, k))
    null = np.array([mutual_information(cat_bigram(rng_b.permutation(labels_pairspace), k))
                     for _ in range(500)])
    p99 = np.percentile(null, 99)
    z = (mi - null.mean()) / (null.std() + 1e-12)
    dist = distinctness(labels_pairspace, k)
    rows[k] = dict(sil=sil, mi=mi, p99=p99, z=z, dist=dist, sizes=sizes,
                   labels_k=labels_k, labels_pairspace=labels_pairspace)
    print(f"  {k:>3}  {sil:>7.3f}  {mi:>6.3f}  {p99:>8.3f}  {z:>5.0f}  "
          f"{dist:>8.3f}  {sorted(sizes.tolist(), reverse=True)}  {elapsed()}")

k_dist = max(rows, key=lambda k: rows[k]['dist'])
r_sel = rows[k_dist]
print(f"\n  distinctness-argmax k={k_dist}  (phase 24 @547K words: k=6, z=65)")
print(f"  at k={k_dist}: MI={r_sel['mi']:.3f}  null p99={r_sel['p99']:.3f}  "
      f"z={r_sel['z']:.0f}  silhouette={r_sel['sil']:.3f}")

K_CATS = k_dist
labels_cat = r_sel['labels_k']
blob = int(max(np.bincount(labels_cat)))
blob_frac = blob / len(covered)
print(f"  largest category {blob}/{len(covered)} words ({blob_frac:.0%})  {elapsed()}")
word_to_cat = {covered[i]: int(labels_cat[i]) for i in range(len(covered))}
org.cat_attractor = {}
for c in sorted(set(labels_cat.tolist())):
    slots_c = [s for s, w in slot_word.items() if word_to_cat.get(w) == c]
    if slots_c:
        org.cat_attractor[c] = normalize(org.mem[slots_c].mean(0), NORM)
for c in sorted(set(word_to_cat.values())):
    ws = sorted(vocab[w] for w, cc in word_to_cat.items() if cc == c)
    print(f"  category {c} ({len(ws)} words): {ws[:16]}{' ...' if len(ws) > 16 else ''}")

# ---------------------------------------------------------------- P1 check
sharpens_z = r_sel['z'] > 65
sharpens_k = 4 <= k_dist <= 10   # "stable or gracefully larger" than k=6, not
                                 # a collapse (<=3) or a run to the sweep ceiling
print(f"\n  (P1) category structure sharpens: z {r_sel['z']:.0f} > phase-24's "
      f"65 -> {sharpens_z}; k={k_dist} stable/graceful vs phase-24's k=6 -> "
      f"{sharpens_k}  {elapsed()}")

# ---------------------------------------------------------------- Stage C
print("\n(C) predictive-gain polysemy vs per-word permutation null "
      "(tighter at 5M words: ~8.8x the occurrences per surviving word)")


def cond_gain(pred, succ):
    H_u = _entropy(np.bincount(succ, minlength=K_CATS))
    n = len(succ); H_c = 0.0
    for pc in set(pred.tolist()):
        sub = succ[pred == pc]
        H_c += len(sub) / n * _entropy(np.bincount(sub, minlength=K_CATS))
    return max(H_u - H_c, 0.0)


def split_gain(word_idx, min_occ=100, n_perm=500, rng=None):
    pairs = [(word_to_cat.get(train_seq[t - 1]), word_to_cat.get(train_seq[t + 1]))
             for t in occ_by_word[word_idx] if 0 < t < len(train_seq) - 1]
    pairs = [(p, s) for p, s in pairs if p is not None and s is not None]
    if len(pairs) < min_occ:
        return None
    pred = np.array([p for p, _ in pairs]); succ = np.array([s for _, s in pairs])
    g = cond_gain(pred, succ)
    null = [cond_gain(rng.permutation(pred), succ) for _ in range(n_perm)]
    return dict(n=len(pairs), gain=g, p99=float(np.percentile(null, 99)))


rng_c = np.random.default_rng(5)
gains = {}
for wi, w in enumerate(range(N_WORDS)):
    r = split_gain(w, rng=rng_c)
    if r is not None:
        gains[w] = r
    if (wi + 1) % 100 == 0:
        print(f"    polysemy test: {wi+1}/{N_WORDS} words scanned, "
              f"{len(gains)} with n>=100  {elapsed()}")
ranked = sorted(gains.items(), key=lambda kv: -(kv[1]['gain'] - kv[1]['p99']))
above = [(vocab[w], r['n'], r['gain'], r['p99']) for w, r in ranked
         if r['gain'] > r['p99']]
print(f"  candidates with n>=100 occurrences: {len(gains)}  {elapsed()}")
print(f"  words above their own permutation-null p99 ({len(above)}):")
for w, n, g, p in above[:30]:
    print(f"    {w:<14} n={n:>6}  gain={g:.3f}  null p99={p:.3f}  margin={g-p:.3f}")
if len(above) > 30:
    print(f"    ... and {len(above) - 30} more")

# ---------------------------------------------------------------- P2 check
right_status = "not in vocab at this MIN_COUNT"
right_survives = None
if 'right' in word_to_idx and word_to_idx['right'] in gains:
    r = gains[word_to_idx['right']]
    right_survives = r['gain'] > r['p99']
    right_status = (f"n={r['n']}  gain={r['gain']:.3f}  null p99={r['p99']:.3f}  "
                    f"margin={r['gain']-r['p99']:.3f}  "
                    f"({'ABOVE' if right_survives else 'below'} its null)")
print(f"\n  (P2) 'right' at 5M words: {right_status}  {elapsed()}")
print(f"  (P2) phase 21 reference @547K: n=622  gain=0.070  null p99=0.043  "
      f"margin=0.027 (ABOVE)")

# ---------------------------------------------------------------- Stage D
print("\n(D) layered generation on real text (phase 18b/23, unchanged)")
xi_core = {w: org.mem[s] for s, w in
           sorted(slot_word.items(), key=lambda kv: org.count[org.kept_idx[kv[0]]])}
for w in range(N_WORDS):
    xi_core.setdefault(w, normalize(emb_c[w], NORM))

LOW_GAIN = {w for w in range(N_WORDS)
            if w not in gains or gains[w]['gain'] <= gains[w]['p99']}
max_gain = max((r['gain'] for r in gains.values()), default=1.0) or 1.0
beta_gain = {w: (gains[w]['gain'] / max_gain if w in gains else 0.0)
             for w in range(N_WORDS)}

cat_ids = sorted(org.cat_attractor)
cat_idx = {c: i for i, c in enumerate(cat_ids)}
C = np.zeros((len(cat_ids), len(cat_ids)))
C_all = np.zeros_like(C)
for a, b in zip(train_seq[:-1], train_seq[1:]):
    ca, cb = word_to_cat.get(a), word_to_cat.get(b)
    if ca is None or cb is None:
        continue
    C_all[cat_idx[ca], cat_idx[cb]] += 1
    if a in LOW_GAIN and b in LOW_GAIN:
        C[cat_idx[ca], cat_idx[cb]] += 1
Cn = C / (C.sum(1, keepdims=True) + 1e-9)
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
rng_g = np.random.default_rng(42)
rand = [int(rng_g.integers(N_WORDS)) for _ in range(800)]
print(f"  category log2-likelihood/hop: generated={cat_loglik(gen):.3f}  "
      f"random={cat_loglik(rand):.3f}  (uniform chance={np.log2(1/len(cat_ids)):.3f})")
print(f"  corpus-bigram hit rate: generated={bigram_hit(gen):.3f}  "
      f"random={bigram_hit(rand):.3f}")
print(f"  word coverage in generation: {len(set(gen))}/{N_WORDS}")
print(f"  sample: {' '.join(vocab[w] for w in gen[:30])}  {elapsed()}")

# ---------------------------------------------------------------- verdict
print(f"\n{'='*70}")
print(f"total wall-clock: {elapsed()}")
print(f"\nPRE-REGISTERED OUTCOME:")
print(f"  P1 (sharpens): z-up={sharpens_z} (z={r_sel['z']:.0f} vs 65), "
      f"k-graceful={sharpens_k} (k={k_dist} vs 6)  "
      f"-> {'CONFIRMED' if sharpens_z and sharpens_k else 'PARTIAL/NOT CONFIRMED'}")
print(f"  P2 (polysemy holds): 'right' survives={right_survives}  "
      f"total words above null={len(above)}/{len(gains)} candidates  "
      f"-> {'CONFIRMED' if right_survives else ('N/A' if right_survives is None else 'NOT CONFIRMED')}")
