"""
PHASE 5b -- POLYSEMY CHALLENGE: context-sensitive word sense disambiguation

Challenge: words that play MULTIPLE grammatical roles (polysemous words).
Example: "fish" is both an ANIMAL (catch fish) and a VERB (to fish for).
         "bear" is ANIMAL and VERB (bear the burden).
         "fly" is ANIMAL (insect) and ACTION (to fly).

The question: can a single oscillator memory slot handle polysemy,
or does the organism naturally form separate slots for the same word
in different contexts?

Hypothesis: the oscillator's attractor structure allows context-to-context
transitions to create DIFFERENT effective states for the same input word,
even if only one slot nominally represents it. The CONTEXT (what came
before) biases which trajectory the field takes from that attractor.

Grammar: ANIMAL → ACTION → OBJECT (baseline)
Polysemous extension:
  - "fly" appears as both ANIMAL (→ACTION) and ACTION (→OBJECT)
  - "fish" appears as both ANIMAL (→ACTION) and OBJECT (after ACTION)
  - "bear" appears as both ANIMAL (→ACTION) and ACTION (→OBJECT)

Metric: does the organism's P matrix show context-sensitive routing
for polysemous words (different successors depending on prior)?
"""

import numpy as np
from organism import Organism, normalize

rng = np.random.default_rng(55)

# ---- vocabulary: 3 clear + 3 polysemous ------------------------------------
ANIMALS_CLEAR  = ['cat','dog','bird','horse','cow','wolf','sheep']   # 7 clear animals
ACTIONS_CLEAR  = ['run','jump','swim','hunt','hide','fight','play']  # 7 clear actions
OBJECTS_CLEAR  = ['food','water','ground','sky','tree','rock','cave','nest','field','river']  # 10 clear objects

# Polysemous words: each appears in 2 roles
POLY_WORDS = ['fly', 'fish', 'bear']  # ANIMAL and ACTION roles
# When preceded by animal → they act as ACTION (→OBJECT)
# When preceded by action → they act as OBJECT (→ANIMAL) -- stretch
# Simplified: fly/fish/bear function as ANIMAL in some sentences, ACTION in others

# For clarity: assign a "primary" role for embedding, "secondary" for grammar generation
# fly:  ANIMAL (primary embedding), can follow animals as "fly over sky"
# fish: OBJECT (primary embedding), can also start sentences as ANIMAL
# bear: ANIMAL (primary embedding), can also act as ACTION

vocab_clear = ANIMALS_CLEAR + ACTIONS_CLEAR + OBJECTS_CLEAR  # 24 unambiguous
vocab_poly  = POLY_WORDS                                       # 3 ambiguous
vocab = vocab_clear + vocab_poly
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)

DIM = 40
N = DIM
NORM = np.sqrt(N)

# Primary category assignment
PRIMARY_CAT = {}
for w in ANIMALS_CLEAR: PRIMARY_CAT[w] = 0
for w in ACTIONS_CLEAR:  PRIMARY_CAT[w] = 1
for w in OBJECTS_CLEAR:  PRIMARY_CAT[w] = 2
PRIMARY_CAT['fly']  = 1  # primarily action (bird flies) — embedding near ACTION
PRIMARY_CAT['fish'] = 0  # primarily animal — embedding near ANIMAL
PRIMARY_CAT['bear'] = 0  # primarily animal — embedding near ANIMAL

CAT = PRIMARY_CAT.copy()  # used for embedding construction
NEXT_CAT = {0: 1, 1: 2, 2: 0}
cat_names = ['ANIMAL', 'ACTION', 'OBJECT']

# ---- structured embeddings --------------------------------------------------
cat_bases = np.zeros((3, DIM))
cat_bases[0, 0:4]  = 1.0
cat_bases[1, 4:8]  = 1.0
cat_bases[2, 8:12] = 1.0

embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(vocab):
    c = CAT[w]
    embeddings[i] = 0.6 * cat_bases[c] + 0.4 * rng.standard_normal(DIM)
    # polysemous words: blend two category signals
    if w == 'fly':    # action-leaning but some animal signal
        embeddings[i] += 0.2 * cat_bases[0]
    if w == 'fish':   # animal-leaning but some object signal
        embeddings[i] += 0.2 * cat_bases[2]
    if w == 'bear':   # animal-leaning but some action signal
        embeddings[i] += 0.2 * cat_bases[1]
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

# ---- grammar: mixed roles for polysemous words ------------------------------
# Base grammar: ANIMAL → ACTION → OBJECT → ANIMAL
# Polysemous grammar additions:
#   fly can follow ANIMAL (as action: "cat fly sky") or start as animal: "fly swim"
#   fish can follow ACTION (as object: "hunt fish") or start as animal: "fish run"
#   bear can follow ANIMAL (as action: "cat bear") or start as animal: "bear run"
# This creates genuine ambiguity: when we see "fly" in position t,
# the NEXT word depends on whether fly was used as ANIMAL or ACTION context.

P_CORRECT = 0.85
fly_idx  = word_to_idx['fly']
fish_idx = word_to_idx['fish']
bear_idx = word_to_idx['bear']

def sample_stream_with_polysemy(n_steps, seed=None):
    """Generate stream where polysemous words appear in both roles."""
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    # Roles: each word at each position is assigned a role
    # Polysemous words alternate between their two roles
    stream = []
    roles  = []  # actual role at each position (0=ANIMAL,1=ACTION,2=OBJECT)
    cat = 0
    # start: pick from animals or polysemous-as-animal
    animal_pool = list(range(7))  # ANIMALS_CLEAR indices
    poly_animal = [word_to_idx['fish'], word_to_idx['bear']]
    all_animal = animal_pool + poly_animal

    action_pool = list(range(7, 14))  # ACTIONS_CLEAR indices
    poly_action = [word_to_idx['fly']]
    all_action = action_pool + poly_action

    object_pool = list(range(14, 24))  # OBJECTS_CLEAR indices
    poly_object = [word_to_idx['fish']]  # fish can be object too
    all_object = object_pool + poly_object

    cat_to_pool = {0: all_animal, 1: all_action, 2: all_object}

    # start in random animal
    w = int(local_rng.choice(all_animal))
    stream.append(w)
    roles.append(0)

    for _ in range(n_steps - 1):
        if local_rng.random() < P_CORRECT:
            next_cat = NEXT_CAT[cat]
        else:
            next_cat = int(local_rng.choice([c for c in [0,1,2] if c != NEXT_CAT[cat]]))
        pool = cat_to_pool[next_cat]
        w = int(local_rng.choice(pool))
        stream.append(w)
        roles.append(next_cat)
        cat = next_cat
    return stream, roles

def make_stream(word_seq, hold=12):
    for w in word_seq:
        s = normalize(embeddings[w].astype(complex), NORM)
        for _ in range(hold):
            yield s

# ---- generate training data --------------------------------------------------
SEQ_LEN = 10000
train_seq, train_roles = sample_stream_with_polysemy(SEQ_LEN, seed=99)

# Show polysemous word context statistics in training data
for w_name, w_idx in [('fly', fly_idx), ('fish', fish_idx), ('bear', bear_idx)]:
    positions = [t for t, w in enumerate(train_seq) if w == w_idx]
    role_counts = [0, 0, 0]
    for t in positions:
        role_counts[train_roles[t]] += 1
    print(f"'{w_name}' appears {len(positions)}x: as ANIMAL={role_counts[0]}, ACTION={role_counts[1]}, OBJECT={role_counts[2]}")

# ---- train organism ---------------------------------------------------------
print(f"\nTraining organism on {SEQ_LEN}-word polysemous stream...")
org = Organism(N=N, K=50, omega=0.15, beta=10.0, seed=0)
org.perceive(list(make_stream(train_seq, hold=12)), g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
org.consolidate(merge_thresh=0.84, prune_frac=0.02)
M = org.mem
n_mem = M.shape[0]
print(f"Memories formed: {n_mem}  (target: {N_WORDS}={len(vocab)})")

# ---- slot analysis ----------------------------------------------------------
states = np.array([normalize(embeddings[w].astype(complex), NORM) for w in train_seq])
assigns = np.abs((M.conj() @ states.T) / N).argmax(0)

slot_to_word = {}
for k in range(n_mem):
    members = np.array(train_seq)[assigns == k]
    if len(members):
        slot_to_word[k] = int(np.bincount(members, minlength=N_WORDS).argmax())

slot_cat = {}
for k, w_idx in slot_to_word.items():
    slot_cat[k] = CAT[vocab[w_idx]]

covered = sorted(set(slot_to_word.values()))
print(f"Words covered: {len(covered)}/{N_WORDS}")

# ---- key question: does organism split polysemous words? --------------------
print("\n--- POLYSEMY ANALYSIS ---")
for w_name, w_idx in [('fly', fly_idx), ('fish', fish_idx), ('bear', bear_idx)]:
    # Which slots map to this word?
    slots_for_word = [k for k, wi in slot_to_word.items() if wi == w_idx]
    # Where does this word get assigned in the stream?
    positions = [t for t, w in enumerate(train_seq) if w == w_idx]
    slot_at_pos = [int(assigns[t]) for t in positions]
    role_at_pos = [train_roles[t] for t in positions]

    # Does different role → different slot?
    role_to_slots = {0: set(), 1: set(), 2: set()}
    for slot, role in zip(slot_at_pos, role_at_pos):
        role_to_slots[role].add(slot)

    print(f"\n'{w_name}' (primary cat={cat_names[CAT[w_name]]}):")
    print(f"  Primary embedding slot(s): {slots_for_word}")
    print(f"  Role→slot mapping: ANIMAL→{role_to_slots[0]}, ACTION→{role_to_slots[1]}, OBJECT→{role_to_slots[2]}")

    # P matrix: what does the organism predict AFTER fly/fish/bear in each role?
    if slots_for_word:
        k = slots_for_word[0]  # primary slot
        if org.Pn[k].sum() > 0:
            top3 = np.argsort(org.Pn[k])[::-1][:5]
            top3_words = [(vocab[slot_to_word[j]], f"{org.Pn[k,j]:.2f}") for j in top3 if j in slot_to_word]
            print(f"  Top successors from primary slot: {top3_words}")

    # Context-conditional transition: what actually follows in the stream?
    animal_context_next = []   # what follows when fly/fish/bear used as ANIMAL
    action_context_next = []   # what follows when fly/fish/bear used as ACTION
    object_context_next = []

    for t, pos in enumerate(positions[:-1]):
        if pos + 1 < len(train_seq):
            role = role_at_pos[t]
            nxt = vocab[train_seq[pos + 1]]
            if role == 0: animal_context_next.append(nxt)
            elif role == 1: action_context_next.append(nxt)
            else: object_context_next.append(nxt)

    if animal_context_next:
        from collections import Counter
        top_a = Counter(animal_context_next).most_common(3)
        print(f"  In stream, after as ANIMAL → top3: {top_a}")
    if action_context_next:
        from collections import Counter
        top_ac = Counter(action_context_next).most_common(3)
        print(f"  In stream, after as ACTION → top3: {top_ac}")

# ---- grammaticality with role-aware metric ----------------------------------
def grammaticality_role(wseq, roles_seq):
    """Grammar where polysemous words are evaluated by actual role, not primary cat."""
    ok = tot = 0
    for t in range(len(wseq) - 1):
        a_role = roles_seq[t]
        b_role = roles_seq[t + 1]
        if NEXT_CAT[a_role] == b_role:
            ok += 1
        tot += 1
    return ok / tot if tot else 0.0

def grammaticality_simple(wseq):
    """Simple grammar using primary category only."""
    ok = tot = 0
    for a, b in zip(wseq[:-1], wseq[1:]):
        if NEXT_CAT[CAT[vocab[a]]] == CAT[vocab[b]]: ok += 1
        tot += 1
    return ok / tot if tot else 0.0

# ---- baseline recall --------------------------------------------------------
def field_to_slot(word_idx):
    z = normalize(embeddings[word_idx].astype(complex), NORM)
    ovs = np.abs(org.overlaps(z, M))
    return int(np.argmax(ovs))

org.beta = 20
print("\n\nRunning baseline recall...")
slot_seq = org.recall(steps=200000, tau_h=15.0, lam=2.5, gamma=2.0, g_rec=7.0, Dn=0.004)
gen = [slot_to_word[int(s)] for s in slot_seq if int(s) in slot_to_word][:800]
gram_base = grammaticality_simple(gen)

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
gram_after = grammaticality_simple(gen2)

oracle_seq, oracle_roles = sample_stream_with_polysemy(800, seed=200)
random_seq = [int(rng.integers(N_WORDS)) for _ in range(800)]
gram_oracle = grammaticality_simple(oracle_seq)
gram_random = grammaticality_simple(random_seq)

# ---- report -----------------------------------------------------------------
print("\n" + "="*64)
print("PHASE 5b -- POLYSEMY CHALLENGE\n")
print(f"Vocab: {N_WORDS} words (24 clear + 3 polysemous: fly/fish/bear)")
print(f"Grammar: ANIMAL→ACTION→OBJECT (polysemous words serve dual roles)\n")

print("GRAMMATICALITY (primary-category evaluation):")
print(f"  Random            : {gram_random:.3f}")
print(f"  Hebbian only      : {gram_base:.3f}  ({(gram_base-gram_random)/(gram_oracle-gram_random)*100:.0f}% gap)")
print(f"  + grammar-mask    : {gram_after:.3f}  ({(gram_after-gram_random)/(gram_oracle-gram_random)*100:.0f}% gap)")
print(f"  Oracle            : {gram_oracle:.3f}")

cats2 = [CAT[vocab[w]] for w in gen2]
bal = np.bincount(cats2, minlength=3) / len(cats2)
print(f"\nCategory balance: {np.round(bal, 3)}  (true ~[.33,.33,.33])")
print(f"Word coverage: {len(set(gen2))}/{N_WORDS}")

poly_in_gen = {w: gen2.count(word_to_idx[w]) for w in ['fly','fish','bear']}
print(f"\nPolysemous word usage in generated sequence: {poly_in_gen}")

print(f"\nFirst 40 words (post-mask):")
print("  " + " ".join(vocab[w] for w in gen2[:40]))

print(f"\nKey question answered: does organism naturally split polysemous words?")
print(f"  → Check 'Role→slot mapping' above.")
print(f"  If ANIMAL role and ACTION role map to DIFFERENT slots: organism disambiguates!")
print(f"  If same slot for both roles: organism doesn't split — polysemy unresolved.")
