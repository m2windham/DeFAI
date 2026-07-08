"""
PHASE 18B -- LAYERED GENERATION: category-routed backbone + continuous
within-category word selection

Phase 18's pure kernel-regression flow failed generation cleanly and
informatively: role information is (by design, Phase 17) a SMALL drift
(~3% of the similarity scale) relative to word identity (~97-100%). That's
exactly what you want for a graceful, non-discrete REPRESENTATION -- but it
means raw nearest-neighbor retrieval over the full exemplar pool cannot
reliably decode it, because the retrieval pool for a polysemous word mixes
BOTH roles' occurrences with no separation mechanism. That is the
first-mover/context-blindness problem returning in continuous clothing:
you can't fix it by amplifying the drift either (tested up to alpha=3.0,
still fails) -- the pool itself has no separation to exploit.

The fix mirrors what was already validated for the discrete case (Phase
13): use TWO LEVELS.
  Level 1 (coarse, small discrete state -- the CORRECT level for grammar):
    an emergent CATEGORY transition model. Categories themselves are
    unsupervised (Phase 10). The transition matrix over K_cat~3 categories
    is learned directly from the corpus, exactly like Phase 5's
    grammar-mask approach (which independently got 0.885 grammaticality,
    the strongest number in this whole project) -- because grammar IS a
    small discrete state machine, and pretending otherwise cost us nothing
    but generation quality.
  Level 2 (fine, continuous): WITHIN the chosen category, select the
    specific word via kernel-weighted similarity, and represent it at its
    context-drifted continuous position (Phase 17) -- so word identity,
    polysemous role, and fine meaning stay continuous and graded.

This keeps everything that was actually validated (unsupervised category
discovery, unsupervised predictive-gain role sensitivity, continuous
positional representation) while being honest that "no discrete state
anywhere in the system" was not achievable without sacrificing generation
quality -- and explains precisely why, mechanistically.
"""

import numpy as np
from organism import normalize
from phase8_true_polysemy import (
    embeddings, N, NORM, word_to_idx, vocab, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, ALPHA_CTX, N_WORDS,
    org_base, slot_word_b, PURE_ANIMALS, PURE_ACTIONS, OBJECTS, NEXT_CAT,
)
from phase10_emergent_category import emergent_cat_attractor, word_to_emergent_cat, cat_slots
from phase12_predictive_split_test import predictive_split_gain

idx_to_word = {v: k for k, v in word_to_idx.items()}
word_to_slot = {w: k for k, w in slot_word_b.items()}
xi_core = {}
for w in range(N_WORDS):
    xi_core[w] = org_base.mem[word_to_slot[w]] if w in word_to_slot else normalize(embeddings[w].astype(complex), NORM)

all_words = PURE_ANIMALS + PURE_ACTIONS + OBJECTS + DUAL_WORDS
raw_gain = {w: (predictive_split_gain(word_to_idx[w]) or {}).get('gain', 0.0) for w in all_words}
max_gain = max(raw_gain.values())
beta = {word_to_idx[w]: raw_gain[w] / max_gain for w in raw_gain}

def continuous_position(w, prev_word, alpha_ctx=ALPHA_CTX):
    core = xi_core[w]
    b = beta.get(w, 0.0)
    prev_cat = word_to_emergent_cat.get(prev_word) if prev_word is not None else None
    if prev_cat is not None and prev_cat in emergent_cat_attractor and b > 0:
        return normalize(core + b*alpha_ctx*emergent_cat_attractor[prev_cat], NORM)
    return core

# ---- Level 1: emergent category transition matrix -- built from LOW-GAIN
#      (unambiguous, single-role) words ONLY. Using every word's single
#      modal category (the naive approach) silently discards exactly the
#      polysemy information the rest of this project recovered: 'fish'
#      would be counted as ANIMAL always, corrupting the count even for
#      its ACTION occurrences. Restricting to words below the gain
#      threshold sidesteps the problem entirely -- their category IS a
#      reliable single label by construction (Phase 12's predictive-gain
#      test is precisely what certifies this). ------------------------------
n_cats = len(cat_slots)
cat_id_list = sorted(set(word_to_emergent_cat.values()))
cat_idx_of = {c: i for i, c in enumerate(cat_id_list)}
LOW_GAIN_THRESH = 0.15
unambiguous_words = set(w for w in range(N_WORDS) if beta.get(w, 0.0) * max_gain <= LOW_GAIN_THRESH)

Ccat = np.zeros((len(cat_id_list), len(cat_id_list)))
for a, b_ in zip(train_seq[:-1], train_seq[1:]):
    if a not in unambiguous_words or b_ not in unambiguous_words:
        continue
    ca = word_to_emergent_cat.get(a); cb = word_to_emergent_cat.get(b_)
    if ca is None or cb is None:
        continue
    Ccat[cat_idx_of[ca], cat_idx_of[cb]] += 1
Ccat_norm = Ccat / (Ccat.sum(1, keepdims=True) + 1e-9)

print("Level 1 -- emergent category transition matrix (built from unambiguous words only):")
for i, c in enumerate(cat_id_list):
    row = " ".join(f"{Ccat_norm[i,j]:.2f}" for j in range(len(cat_id_list)))
    print(f"  cat {c}: [{row}]")

# ---- Level 2: words grouped by emergent category, for within-category vote ---
# Unambiguous (low-gain) words go ONLY in their single modal category's pool
# -- their role is fixed, restricting them prevents grammar errors.
# High-gain (dual/polysemous) words are made eligible candidates in EVERY
# category's pool: their single modal label (from discover_categories,
# which only sees an aggregate/blended profile -- Phase 15/16's finding)
# is not a reliable restriction. Instead, let the continuous kernel
# similarity (their context-drifted position vs current z) do the fine
# discrimination -- this is exactly what Phase 17 validated works: a
# dual word's position naturally lands near the RIGHT category's
# neighborhood once given the right context, and far from the wrong one.
high_gain_words = set(w for w in range(N_WORDS) if w not in unambiguous_words)
words_in_cat = {c: [] for c in cat_id_list}
for w, c in word_to_emergent_cat.items():
    if w in unambiguous_words:
        words_in_cat[c].append(w)
for c in cat_id_list:
    words_in_cat[c].extend(high_gain_words)

CAT_OF = {}
for w in PURE_ANIMALS: CAT_OF[word_to_idx[w]] = ANIMAL
for w in PURE_ACTIONS: CAT_OF[word_to_idx[w]] = ACTION
for w in OBJECTS: CAT_OF[word_to_idx[w]] = OBJECT

def grammaticality(seq):
    ok = tot = 0
    for a, b in zip(seq[:-1], seq[1:]):
        ca = CAT_OF.get(a); cb = CAT_OF.get(b)
        if ca is None or cb is None:
            continue
        if NEXT_CAT[ca] == cb:
            ok += 1
        tot += 1
    return ok / max(tot, 1)

def generate_layered(n_words=400, dt=0.05, g_rec=6.0, omega=0.15, Dn=0.003, seed=0,
                      settle_steps=6, beta_sim=20.0, tau_h=3.0, lam_h=6.0):
    rng = np.random.default_rng(seed)
    # Word-level habituation -- the missing piece. Without it, a word that
    # is a strong kernel-similarity match keeps winning every time it's
    # eligible (no discrete recruitment/threshold exists here to break the
    # tie, and the kernel-flow version already proved this exact failure
    # mode in Phase 18's first attempt). Organism.recall() always had this
    # (h[k]/fatigue); the layered generator needs the direct analogue,
    # trivial here since the state space is just N_WORDS.
    h_word = np.zeros(N_WORDS)
    # FUNCTIONAL CATEGORY is tracked as explicit FSM state, decoupled from
    # word identity -- a dual word's single modal label (from
    # discover_categories, an aggregate/blended profile) must NOT drive the
    # next transition, or the state machine silently reverts to using the
    # word's fixed default role, exactly the bug just fixed on the counting
    # side. The state advances via the unambiguous-word-only cycle
    # regardless of which specific (possibly dual) word occupies the
    # current position.
    cur_cat = cat_id_list[int(rng.integers(len(cat_id_list)))]
    cur_word = int(rng.integers(N_WORDS))
    prev_word = None
    z = continuous_position(cur_word, prev_word)
    generated = [cur_word]

    for _ in range(n_words - 1):
        probs = Ccat_norm[cat_idx_of[cur_cat]]
        nxt_cat = cat_id_list[int(rng.choice(len(cat_id_list), p=probs))]

        candidates = words_in_cat[nxt_cat]
        # continuous within-category selection: kernel-weighted similarity of
        # each candidate's CONTEXT-DRIFTED position (given cur_word as context)
        # to the current flow state z -- this is where role/position nuance
        # from Phase 17 actually gets used, at the scale where it's decidable
        # (choosing among ~8-12 candidates, not ~12000 exemplars).
        cand_positions = np.array([continuous_position(c, cur_word) for c in candidates])
        sims = np.abs((cand_positions.conj() @ z) / N)
        fat = np.maximum(1 - lam_h*h_word[candidates], 0.0)
        scored = sims * fat
        if scored.sum() <= 1e-9:
            scored = sims   # everyone fatigued -- fall back to raw similarity
        w = np.exp(beta_sim*(scored - scored.max())); w /= w.sum()
        nxt_word = candidates[int(rng.choice(len(candidates), p=w))]

        target = continuous_position(nxt_word, cur_word)
        # settle the field toward the chosen target (keeps continuous dynamics
        # for the ACTUAL field state, even though category+word choice used
        # explicit distributions above)
        for _ in range(settle_steps):
            noise = np.sqrt(2*Dn*dt)*(rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(2)
            z = normalize(z + dt*(1j*omega*z + g_rec*(target - z)) + noise, NORM)
            h_word = h_word + dt/tau_h*(-h_word)
        h_word[nxt_word] = min(h_word[nxt_word] + 1.0, 1.0)

        generated.append(nxt_word)
        prev_word = cur_word
        cur_word = nxt_word
        cur_cat = nxt_cat
    return generated

print("\nGenerating layered sequence (400 words)...")
gen = generate_layered(n_words=400, seed=1)
gram = grammaticality(gen)
cov = len(set(gen))
print(f"  Grammaticality: {gram:.3f}")
print(f"  Word coverage: {cov}/{N_WORDS}")
print(f"  First 40 words: {' '.join(vocab[w] for w in gen[:40])}")

rng_g = np.random.default_rng(42)
random_seq = [int(rng_g.integers(N_WORDS)) for _ in range(800)]
gram_random = grammaticality(random_seq)
print(f"\n  Random baseline: {gram_random:.3f}")
print(f"  Phase 8 discrete baseline (grammar-masked): 0.839")
print(f"  Phase 8 discrete context (grammar-masked):  0.956")

print("\n--- Context sensitivity of generated dual-role transitions ---")
for dw in DUAL_WORDS:
    dwi = word_to_idx[dw]
    occ = [t for t in range(1, len(gen)-1) if gen[t] == dwi]
    after = {ANIMAL: [], ACTION: [], OBJECT: []}
    for t in occ:
        prev_cat = CAT_OF.get(gen[t-1])
        next_cat = CAT_OF.get(gen[t+1])
        if prev_cat is not None and next_cat is not None:
            after[prev_cat].append(next_cat)
    def summarize(lst):
        if not lst: return "n/a"
        c = np.bincount(lst, minlength=3)
        return f"A={c[0]} V={c[1]} O={c[2]}"
    print(f"  '{dw}' (n={len(occ)}): after_ANIMAL:[{summarize(after[ANIMAL])}]  "
          f"after_ACTION:[{summarize(after[ACTION])}]  after_OBJECT:[{summarize(after[OBJECT])}]")
