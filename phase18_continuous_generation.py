"""
PHASE 18 -- CONTINUOUS TRANSITION FIELD: generation without a discrete P matrix

Phase 17 gave every word a continuous, context-drifted POSITION. But nothing
could read it back out into a sequence yet -- Organism.recall() still works
by picking among K discrete attractors via a discrete Pn[cur,:] lookup row.

This phase replaces that lookup with a continuous flow field, built as a
Nadaraya-Watson kernel regression directly over the corpus's own sequential
(position_t -> position_t+1) pairs -- no discretization step anywhere:

  target(z) = normalize( sum_i softmax_i(overlap(z, pos_i)) * pos_{i+1} )
  dz/dt     = i*omega*z + g_rec*(target - z)  [+ noise]

pos_i are the CONTINUOUS, context-drifted positions from Phase 17 (so the
same word occupies different manifold neighborhoods depending on context,
exactly as validated). The kernel weighting means "what comes next" is a
smooth function of WHERE you currently are on the manifold, not a lookup
keyed by which of K discrete slots fired.

This is the generative half of "continuous self-organizing meaning
dynamics" -- test whether it actually produces grammatical, context-
sensitive sequences end to end, closing the loop opened in Phase 6/7.
"""

import numpy as np
from organism import normalize
from phase8_true_polysemy import (
    embeddings, N, NORM, word_to_idx, vocab, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, ALPHA_CTX, N_WORDS,
    org_base, slot_word_b, PURE_ANIMALS, PURE_ACTIONS, OBJECTS, NEXT_CAT,
)
from phase10_emergent_category import emergent_cat_attractor, word_to_emergent_cat
from phase12_predictive_split_test import predictive_split_gain

idx_to_word = {v: k for k, v in word_to_idx.items()}
word_to_slot = {w: k for k, w in slot_word_b.items()}
xi_core = {}
for w in range(N_WORDS):
    if w in word_to_slot:
        xi_core[w] = org_base.mem[word_to_slot[w]]
    else:
        xi_core[w] = normalize(embeddings[w].astype(complex), NORM)

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

print("Building sequential (position_t, position_{t+1}) training pairs from the corpus...")
positions_seq = []
prev_word = None
for w in train_seq:
    positions_seq.append(continuous_position(w, prev_word))
    prev_word = w
positions_seq = np.array(positions_seq)   # (T, N) complex
words_seq = np.array(train_seq)
print(f"  {len(positions_seq)} sequential positions built.")

# ---- continuous kernel-weighted flow field ------------------------------------
POS_T = positions_seq[:-1]     # "from" positions
POS_T1 = positions_seq[1:]     # "to" positions (targets)
WORD_T1 = words_seq[1:]        # word identity of the target, for decoding/eval

def kernel_vote_target(z, h, prev_word, beta_sim=30.0, top_k=400, lam=3.0):
    """Nadaraya-Watson kernel WEIGHTED VOTE over successor WORD IDENTITY
    (not a raw vector average). Averaging complex position vectors directly
    fails here: many different successor words within the same true
    category have near-orthogonal identity components (each word's
    embedding is 60% shared category basis + 40% independent random noise
    -- Phase 8's construction), so blending several different words'
    positions cancels their identity and lands nowhere near any real word.
    Voting picks a coherent WORD, then uses that word's own CONTINUOUS,
    context-drifted position (Phase 17) as the flow target -- so matching
    and disambiguation stay continuous and kernel-weighted, but the actual
    target is always a real point on a real word's manifold neighborhood,
    never an averaged non-word.

    Habituation (h, indexed over corpus exemplars) is unchanged from the
    first attempt -- it's what keeps the flow from re-locking onto the
    same dense word repeatedly."""
    sims = np.abs((POS_T.conj() @ z) / N)
    fat = np.maximum(1 - lam*h, 0.0)
    scored = sims * fat
    if top_k is not None and top_k < len(scored):
        idx = np.argpartition(-scored, top_k)[:top_k]
    else:
        idx = np.arange(len(scored))
    s = scored[idx]
    w = np.exp(beta_sim * (s - s.max()))
    w /= w.sum() + 1e-12
    votes = {}
    for wt, wd in zip(w, WORD_T1[idx]):
        votes[wd] = votes.get(wd, 0.0) + wt
    winner = max(votes, key=votes.get)
    target = continuous_position(winner, prev_word)
    return target, winner, idx, sims

def generate_continuous(n_words=400, dt=0.05, g_rec=6.0, omega=0.15, Dn=0.003,
                         beta_sim=30.0, top_k=400, seed=0, settle_steps=6,
                         tau_h=3.0, lam=5.0):
    rng = np.random.default_rng(seed)
    z = normalize(rng.standard_normal(N).astype(complex) +
                  1j*rng.standard_normal(N), NORM)
    h = np.zeros(len(POS_T))
    generated = []
    last_decoded_word = None
    for step in range(n_words * settle_steps):
        target, winner, idx, sims = kernel_vote_target(z, h, last_decoded_word,
                                                         beta_sim=beta_sim, top_k=top_k, lam=lam)
        noise = np.sqrt(2*Dn*dt) * (rng.standard_normal(N) + 1j*rng.standard_normal(N)) / np.sqrt(2)
        z = normalize(z + dt*(1j*omega*z + g_rec*(target - z)) + noise, NORM)
        h = h + dt/tau_h*(sims - h)
        if step % settle_steps == settle_steps - 1:
            core_sims = np.abs(np.array([xi_core[wi] for wi in range(N_WORDS)]).conj() @ z) / N
            wi = int(np.argmax(core_sims))
            generated.append(wi)
            last_decoded_word = wi
    return generated

# ---- evaluation: grammaticality (same metric as all previous phases) ---------
CAT_OF = {}
for w in PURE_ANIMALS: CAT_OF[word_to_idx[w]] = ANIMAL
for w in PURE_ACTIONS: CAT_OF[word_to_idx[w]] = ACTION
for w in OBJECTS: CAT_OF[word_to_idx[w]] = OBJECT
# dual words: no fixed category -- evaluated separately via context-conditioned check

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

print("\nGenerating continuous-flow sequence (400 words)...")
gen = generate_continuous(n_words=400, seed=1)
gram = grammaticality(gen)
coverage = len(set(gen))
print(f"  Grammaticality (single-role words only): {gram:.3f}")
print(f"  Word coverage: {coverage}/{N_WORDS}")
print(f"  First 40 words: {' '.join(vocab[w] for w in gen[:40])}")

# ---- oracle / random baselines for comparison ---------------------------------
rng_g = np.random.default_rng(42)
random_seq = [int(rng_g.integers(N_WORDS)) for _ in range(800)]
gram_random = grammaticality(random_seq)
print(f"\n  Random baseline grammaticality: {gram_random:.3f}")
print(f"  For reference -- Phase 8 baseline (discrete, grammar-masked): 0.839")
print(f"                    Phase 8 context (discrete, grammar-masked): 0.956")

# ---- polysemy behavior IN GENERATION: does the continuous flow route fish's
#      successor differently depending on what preceded fish in the GENERATED
#      sequence itself (not just training statistics)? -------------------------
print("\n--- Context sensitivity of generated dual-role word transitions ---")
for dw in DUAL_WORDS:
    dwi = word_to_idx[dw]
    occ = [t for t, w in enumerate(gen[:-1]) if w == dwi]
    after_animal_prev, after_action_prev, after_object_prev = [], [], []
    for t in occ:
        if t == 0:
            continue
        prev_w = gen[t-1]
        prev_cat = CAT_OF.get(prev_w)
        next_w = gen[t+1]
        next_cat = CAT_OF.get(next_w)
        if prev_cat is None or next_cat is None:
            continue
        if prev_cat == ANIMAL: after_animal_prev.append(next_cat)
        elif prev_cat == ACTION: after_action_prev.append(next_cat)
        elif prev_cat == OBJECT: after_object_prev.append(next_cat)
    print(f"  '{dw}': n_occurrences_in_gen={len(occ)}  "
          f"after_ANIMAL->{after_animal_prev}  after_ACTION->{after_action_prev}  after_OBJECT->{after_object_prev}")
