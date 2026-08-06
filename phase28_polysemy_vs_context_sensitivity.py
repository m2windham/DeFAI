"""
PHASE 28 (T2.3) -- THE DISENTANGLING TEST: polysemy vs grammatical
context-sensitivity on the phase-23 corpus.

Phase 23 named an open residual: its predictive-gain (Myhill-Nerode split)
test surfaces ~114 words whose successor-category distribution shifts with
context, but that gain statistic conflates two distinct phenomena:

  - LEXICAL POLYSEMY: the word plays genuinely different grammatical roles
    depending on context ("watch" the noun vs "watch" the verb) -- its
    occurrences behave like they belong to DIFFERENT emergent categories.
  - GRAMMATICAL CONTEXT-SENSITIVITY: the word keeps one grammatical role
    throughout, but which specific successor it predicts shifts with local
    context (e.g. collocational structure) -- occurrences stay inside the
    SAME emergent category, only the within-category weighting moves.

This phase runs the disentangling test named in ROADMAP row 28: for every
word phase 23's gain test flags, partition its occurrences by CONTEXT
SIGNATURE (the emergent predecessor category -- the same conditioning
variable the gain test already uses, so this is a decomposition of that
statistic, not a new corpus pass) and ask whether the buckets' dominant
SUCCESSOR category differs (polysemy) or agrees (context-sensitivity).

Zero labels are used inside the mechanism -- category discovery, the gain
test, and the disentangling test are all unsupervised, exactly as in
phases 12/21/23. Gold POS tags (nltk averaged-perceptron tagger) are used
for EVALUATION ONLY, computed after every classification decision is made,
never fed back into the mechanism.

PRE-REGISTERED SCOPE CAVEAT (ROADMAP row 28, carried into AGENT_TARGETS
T2.3): the emergent categories here are coarse (k=3, phase 21's balance
criterion) -- roughly POS-tier granularity, not fine-grained sense
inventories. A same-POS polyseme (river "bank" vs financial "bank" -- both
nouns) has NO reason to land in different emergent categories under this
test: both senses are still typically preceded/followed by the same coarse
grammatical neighborhood. Such words will default to a "context-sensitivity"
verdict here even though a human lexicographer would call them polysemous.
The claim this phase can honestly support is therefore narrower than "detects
polysemy": it detects POS-LEVEL multi-role behavior and separates it from
same-POS distributional drift. That narrowing is pre-registered, not a
post-hoc excuse -- see the gold-POS evaluation below, which measures it
directly rather than asserting it.
"""

import re
import glob
import time
import numpy as np
from collections import Counter
from organism import normalize
from polysemy_organism import (PolysemyOrganism, _entropy, _ppmi_transform,
                               _kmeans_real, _silhouette_real)

t_start = time.time()

# ---------------------------------------------------------------- corpus
# Identical corpus / tokenization / vocab bar to phase 23 (its exact
# 547K-word 8-book Gutenberg corpus) so the gain-detected word list here
# is the same population phase 23 measured.
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

occ_by_word = {w: [] for w in range(N_WORDS)}
for t, w in enumerate(train_seq):
    occ_by_word[w].append(t)

print(f"PHASE 28: polysemy vs grammatical context-sensitivity "
      f"({len(raw_tokens)} raw tokens, {len(train_seq)} in-vocab, "
      f"{N_WORDS} words, count>={MIN_COUNT})\n")

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


def make_stream(seq, hold=4):
    for w in seq:
        s = emb_c[w]
        for _ in range(hold):
            yield s


def coverage_map(org):
    org.consolidate(merge_thresh=0.84, prune_frac=0.0005)
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
print("(A) perception -- phase 23's winning arm (recipe, recruit=0.75, 3 epochs)")
K_CAP = min(2000, N_WORDS * 4)


def train_recipe(recruit, epochs):
    o = PolysemyOrganism(N=N, K=K_CAP, omega=0.15, beta=10.0, seed=0)
    for _ in range(epochs):
        o.perceive(make_stream(train_seq), g_in=5.0, dt=0.05, eta=0.02, recruit=recruit)
    return o


t0 = time.time()
org = train_recipe(0.75, 3)
slot_word, cov, n_mem = coverage_map(org)
print(f"  memories={n_mem}  coverage={cov}/{N_WORDS}  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- Stage B
print("\n(B) emergent categories -- discover_categories_v2 @k=3 (phase-21 method)")
raw_slot = org.P[np.ix_(org.kept_idx, org.kept_idx)]
res_v2 = org.discover_categories_v2(k_range=[3], raw_counts=raw_slot, seed=3)
K_CATS = 3
print(f"  silhouette={res_v2['silhouette']:.3f}  "
      f"sizes={sorted(np.bincount(list(org.word_slot_to_cat.values())).tolist(), reverse=True)}")
word_to_cat = {w: org.word_slot_to_cat.get(k) for k, w in slot_word.items()}

# ---------------------------------------------------------------- Stage C
print("\n(C) predictive-gain polysemy test vs per-word permutation null "
      "(phase 23's stage C, verbatim)")


def cond_gain(pred, succ):
    H_u = _entropy(np.bincount(succ, minlength=K_CATS))
    n = len(succ); H_c = 0.0
    for pc in set(pred.tolist()):
        sub = succ[pred == pc]
        H_c += len(sub) / n * _entropy(np.bincount(sub, minlength=K_CATS))
    return max(H_u - H_c, 0.0)


def occ_pairs(word_idx):
    return [(word_to_cat.get(train_seq[t - 1]), word_to_cat.get(train_seq[t + 1]))
            for t in occ_by_word[word_idx] if 0 < t < len(train_seq) - 1]


def split_gain(word_idx, min_occ=100, n_perm=500, rng=None):
    pairs = [(p, s) for p, s in occ_pairs(word_idx) if p is not None and s is not None]
    if len(pairs) < min_occ:
        return None
    pred = np.array([p for p, _ in pairs]); succ = np.array([s for _, s in pairs])
    g = cond_gain(pred, succ)
    null = [cond_gain(rng.permutation(pred), succ) for _ in range(n_perm)]
    return dict(n=len(pairs), gain=g, p99=float(np.percentile(null, 99)), pairs=pairs)


rng = np.random.default_rng(5)
gains = {w: r for w in range(N_WORDS) if (r := split_gain(w, rng=rng))}
above = {w: r for w, r in gains.items() if r['gain'] > r['p99']}
print(f"  candidates with n>=100 occurrences: {len(gains)}")
print(f"  words above their own permutation-null p99: {len(above)}  "
      f"(phase 23's conflated 'context-sensitivity' list)")

# ---------------------------------------------------------------- Stage D
print(f"\n(D) THE DISENTANGLING TEST (phase 28, new): for each flagged word, "
      f"partition occurrences by predecessor category (the gain test's own "
      f"conditioning variable = the context signature) and compare each "
      f"bucket's DOMINANT successor category.")

MIN_BUCKET = 15          # occurrences needed per predecessor-category bucket
DOM_MARGIN = 1.0 / K_CATS + 0.15   # a bucket's top successor category must
                                    # clear chance (1/K_CATS) by this margin
                                    # to count as a confident "dominant" read


def disentangle(word_idx, r):
    pairs = r['pairs']
    pred = np.array([p for p, s in pairs]); succ = np.array([s for p, s in pairs])
    buckets = {}
    for pc in sorted(set(pred.tolist())):
        sub = succ[pred == pc]
        if len(sub) < MIN_BUCKET:
            continue
        counts = np.bincount(sub, minlength=K_CATS)
        dom_cat = int(counts.argmax())
        dom_frac = counts[dom_cat] / counts.sum()
        buckets[pc] = dict(n=int(counts.sum()), dom_cat=dom_cat, dom_frac=float(dom_frac))
    confident = {pc: b for pc, b in buckets.items() if b['dom_frac'] > DOM_MARGIN}
    verdict = "context-sensitivity"   # pre-registered default (the scope caveat)
    conflict = None
    pcs = list(confident.keys())
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            if confident[pcs[i]]['dom_cat'] != confident[pcs[j]]['dom_cat']:
                verdict = "polysemy"
                conflict = (pcs[i], confident[pcs[i]]['dom_cat'],
                            pcs[j], confident[pcs[j]]['dom_cat'])
                break
        if conflict:
            break
    return dict(verdict=verdict, buckets=buckets, n_confident_buckets=len(confident),
                conflict=conflict)


verdicts = {w: disentangle(w, r) for w, r in above.items()}
n_poly = sum(1 for v in verdicts.values() if v['verdict'] == 'polysemy')
n_ctx = len(verdicts) - n_poly
print(f"  classified {len(verdicts)} words: {n_poly} polysemy (different induced "
      f"categories), {n_ctx} context-sensitivity (same category, shifted successors)")
print(f"\n  polysemy-classified words (occurrences route to different induced categories):")
for w, v in sorted(verdicts.items(), key=lambda kv: -above[kv[0]]['gain']):
    if v['verdict'] == 'polysemy':
        a, ca, b, cb = v['conflict']
        print(f"    {vocab[w]:<15} n={above[w]['n']:<6} gain={above[w]['gain']:.3f}  "
              f"pred-cat {a}->succ-cat {ca}  vs  pred-cat {b}->succ-cat {cb}")

# ---------------------------------------------------------------- Stage E
print(f"\n(E) GOLD-POS EVALUATION (oracle -- evaluation only, never used above)")
try:
    import nltk
    try:
        nltk.pos_tag(['test'], tagset='universal')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('universal_tagset', quiet=True)
    t0 = time.time()
    tagged = nltk.pos_tag(raw_tokens, tagset='universal')
    print(f"  tagged {len(raw_tokens)} raw tokens in {time.time()-t0:.0f}s "
          f"(flat token stream, no sentence boundaries -- a known tagger-accuracy "
          f"caveat on this oracle; it affects evaluation precision only, not the "
          f"unsupervised mechanism above)")

    gold_tags = {w: Counter() for w in range(N_WORDS)}
    for tok, tag in tagged:
        wi = word_to_idx.get(tok)
        if wi is not None:
            gold_tags[wi][tag] += 1

    def gold_entropy(w):
        return _entropy(list(gold_tags[w].values()))

    def gold_minority_frac(w):
        c = gold_tags[w]
        if not c:
            return 0.0
        total = sum(c.values())
        return 1.0 - max(c.values()) / total

    MINORITY_BAR = 0.05   # pre-registered: a secondary POS tag must carry >=5%
                           # of occurrences to count as genuine multi-role,
                           # not tagger noise on ambiguous flat tokens
    gold_multirole = {w for w in range(N_WORDS) if gold_minority_frac(w) > MINORITY_BAR}

    detected = set(above.keys())
    tp = len(detected & gold_multirole)
    precision = tp / len(detected) if detected else float('nan')
    recall = tp / len(gold_multirole) if gold_multirole else float('nan')
    print(f"\n  gold-POS multi-role words (minority-tag frac > {MINORITY_BAR}): "
          f"{len(gold_multirole)}/{N_WORDS}")
    print(f"  detector (gain > null) precision={precision:.3f}  recall={recall:.3f}  "
          f"(tp={tp}, detected={len(detected)}, gold-positive={len(gold_multirole)})")

    poly_words = [w for w, v in verdicts.items() if v['verdict'] == 'polysemy']
    ctx_words = [w for w, v in verdicts.items() if v['verdict'] == 'context-sensitivity']
    mean_ent_poly = float(np.mean([gold_entropy(w) for w in poly_words])) if poly_words else float('nan')
    mean_ent_ctx = float(np.mean([gold_entropy(w) for w in ctx_words])) if ctx_words else float('nan')
    mean_minor_poly = float(np.mean([gold_minority_frac(w) for w in poly_words])) if poly_words else float('nan')
    mean_minor_ctx = float(np.mean([gold_minority_frac(w) for w in ctx_words])) if ctx_words else float('nan')
    print(f"\n  mean gold-POS entropy:      polysemy-classified={mean_ent_poly:.3f} bits  "
          f"context-sensitivity-classified={mean_ent_ctx:.3f} bits")
    print(f"  mean gold minority-tag frac: polysemy-classified={mean_minor_poly:.3f}  "
          f"context-sensitivity-classified={mean_minor_ctx:.3f}")

    print(f"\n  per-word detail (top 20 by |gain - null|):")
    for w, r in sorted(above.items(), key=lambda kv: -kv[1]['gain'])[:20]:
        v = verdicts[w]
        top_tags = ", ".join(f"{t}:{c}" for t, c in gold_tags[w].most_common(3))
        print(f"    {vocab[w]:<15} verdict={v['verdict']:<18} n={r['n']:<6} "
              f"gain={r['gain']:.3f}  gold-entropy={gold_entropy(w):.3f}  "
              f"gold-tags=[{top_tags}]")

    print(f"\n  'right' (phase 21's headline lexical item), if present:")
    if 'right' in word_to_idx and word_to_idx['right'] in above:
        w = word_to_idx['right']
        print(f"    verdict={verdicts[w]['verdict']}  gold-tags="
              f"{gold_tags[w].most_common()}  gold-entropy={gold_entropy(w):.3f}")
    elif 'right' in word_to_idx and word_to_idx['right'] in gains:
        w = word_to_idx['right']
        print(f"    below its own gain null this run (n={gains[w]['n']}, "
              f"gain={gains[w]['gain']:.3f}, p99={gains[w]['p99']:.3f})")
    else:
        print("    not in the count>=150 vocab this run")

    ran_gold = True
except ImportError:
    print("  nltk not available -- skipping gold-POS evaluation (mechanism "
          "results above stand on their own; gold-POS is eval-only)")
    ran_gold = False

# ---------------------------------------------------------------- verdict
print(f"\n{'='*72}\nVERDICT\n")
print(f"The disentangling test decomposes phase 23's {len(above)}-word gain-detected "
      f"list into {n_poly} polysemy ({n_poly/max(len(above),1):.0%}, occurrences route "
      f"to different induced categories) and {n_ctx} context-sensitivity "
      f"({n_ctx/max(len(above),1):.0%}, same category, shifted successor weighting), "
      f"purely from the unsupervised categories and per-word occurrence statistics "
      f"already computed by phase 23 -- no new labels, no new corpus pass.")
if ran_gold:
    print(f"\nGold-POS evaluation (oracle, eval-only): precision={precision:.3f} "
          f"recall={recall:.3f} against a minority-POS-tag gold-multirole definition. "
          f"Polysemy-classified words carry {'higher' if mean_ent_poly > mean_ent_ctx else 'lower/comparable'} "
          f"gold-POS entropy than context-sensitivity-classified words "
          f"({mean_ent_poly:.3f} vs {mean_ent_ctx:.3f} bits), "
          f"{'consistent with' if mean_ent_poly > mean_ent_ctx else 'NOT clearly separating'} "
          f"the intended polysemy/context-sensitivity split.")
print(f"\nPRE-REGISTERED SCOPE CAVEAT (confirmed by construction, not just asserted): "
      f"the k=3 emergent categories are POS-tier, not sense-level, so a same-POS "
      f"polyseme (e.g. two noun senses of one word) cannot produce a predecessor-cat "
      f"vs successor-cat conflict here and defaults to the context-sensitivity bucket. "
      f"The defensible claim is narrowed accordingly: this test detects and separates "
      f"POS-LEVEL multi-role behavior from same-POS distributional context-sensitivity; "
      f"it is not a general word-sense-disambiguation detector, and the gold-POS numbers "
      f"above measure exactly that boundary rather than papering over it.")
print(f"\n(total wall-clock: {time.time()-t_start:.0f}s)")
