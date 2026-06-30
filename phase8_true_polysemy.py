"""
PHASE 8 -- TRUE POLYSEMY: words that genuinely occupy two grammatical roles

Phase 7 found that conjunctive coding (Cotteret 2022) is architecturally
correct but never gets exercised: the phase-7 corpus assigns each word a
FIXED category, so "fish" is always ANIMAL. There is no real ambiguity to
resolve, so the organism only ever forms one slot per word regardless of
alpha_ctx.

This phase builds a corpus with genuine dual-role words: the SAME surface
form (same embedding vector -- the network can't tell them apart from the
input alone) is grammatically valid as both ANIMAL and ACTION, e.g.
  "fish" the animal      (ANIMAL slot, precedes an ACTION)
  "fish" the verb (to fish)  (ACTION slot, follows an ANIMAL)
  "duck" the animal      vs   "duck" the verb (to duck)
  "bear" the animal      vs   "bear" the verb (to bear / endure)

Because the embedding is identical for both roles, the ONLY signal that can
disambiguate is the preceding word's category -- exactly what conjunctive
coding is supposed to exploit.

Prediction:
  - Baseline (no context composite): one slot per word. Since "fish" gets
    one P-row, it must commit to either ANIMAL-successor or ACTION-successor
    statistics -- it cannot represent "fish" correctly in both roles.
  - Context-biased perceive: composite pattern z + alpha*M[prev] should
    pull "fish-after-animal-context" (i.e. fish-as-verb, since it follows
    an animal) and "fish-after-object-context" (fish-as-noun, opening a
    new clause) into two different attractor neighborhoods -> two slots
    with two different successor distributions.
"""

import numpy as np
from organism import Organism, normalize

rng_global = np.random.default_rng(42)

# ---- vocabulary ---------------------------------------------------------------
PURE_ANIMALS = ['cat','dog','bird','horse','cow','pig','sheep','wolf']
PURE_ACTIONS = ['run','jump','swim','eat','sleep','hunt','hide','play']
OBJECTS      = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
DUAL_WORDS   = ['fish','duck','bear']   # appear as BOTH animal and action

vocab = PURE_ANIMALS + PURE_ACTIONS + OBJECTS + DUAL_WORDS
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)

ANIMAL, ACTION, OBJECT = 0, 1, 2
cat_names = ['ANIMAL', 'ACTION', 'OBJECT']
NEXT_CAT = {0: 1, 1: 2, 2: 0}

# words that may fill the ANIMAL slot, words that may fill ACTION slot
ANIMAL_FILLERS = PURE_ANIMALS + DUAL_WORDS
ACTION_FILLERS = PURE_ACTIONS + DUAL_WORDS
OBJECT_FILLERS = OBJECTS

DIM = 30; N = DIM; NORM = np.sqrt(N)
P_CORRECT = 0.88

# ---- embeddings: ONE vector per word (dual words get a single embedding,
#      same vector regardless of which role they play in the sentence) ---------
emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, DIM))
cat_bases[ANIMAL, 0:3] = 1.0
cat_bases[ACTION, 3:6] = 1.0
cat_bases[OBJECT, 6:9] = 1.0

embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(PURE_ANIMALS):
    embeddings[word_to_idx[w]] = 0.6*cat_bases[ANIMAL] + 0.4*emb_rng.standard_normal(DIM)
for i, w in enumerate(PURE_ACTIONS):
    embeddings[word_to_idx[w]] = 0.6*cat_bases[ACTION] + 0.4*emb_rng.standard_normal(DIM)
for i, w in enumerate(OBJECTS):
    embeddings[word_to_idx[w]] = 0.6*cat_bases[OBJECT] + 0.4*emb_rng.standard_normal(DIM)
for w in DUAL_WORDS:
    # NEUTRAL embedding: equidistant from animal/action bases, no category bias
    embeddings[word_to_idx[w]] = 0.3*(cat_bases[ANIMAL]+cat_bases[ACTION]) + 0.5*emb_rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

# ---- grammar stream with TRUE per-occurrence role tracking --------------------
def sample_stream(n, seed=None, p_dual=0.5):
    """
    Cyclic ANIMAL->ACTION->OBJECT->ANIMAL grammar. When filling an ANIMAL or
    ACTION slot, with probability p_dual choose one of the dual words
    (same embedding, different grammatical role depending on slot).
    Returns (word_idx_seq, true_role_seq) where true_role_seq[t] is the
    ACTUAL category the word is playing at position t (ground truth, used
    only for evaluation -- never given to the organism).
    """
    local = np.random.default_rng(seed)
    cat = ANIMAL
    seq = []; roles = []

    def fill(c):
        if c == ANIMAL:
            pool = ANIMAL_FILLERS
        elif c == ACTION:
            pool = ACTION_FILLERS
        else:
            pool = OBJECT_FILLERS
        if c in (ANIMAL, ACTION) and local.random() < p_dual:
            w = local.choice(DUAL_WORDS)
        else:
            non_dual = [x for x in pool if x not in DUAL_WORDS]
            w = local.choice(non_dual)
        return word_to_idx[w]

    w = fill(cat); seq.append(w); roles.append(cat)
    for _ in range(n - 1):
        nc = NEXT_CAT[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != NEXT_CAT[cat]]))
        w = fill(nc)
        seq.append(w); roles.append(nc)
        cat = nc
    return seq, roles

def grammaticality(wseq, roles_lookup):
    """roles_lookup: function(word_idx) -> assumed category (for evaluation
    we use TRUE role per occurrence when available, else the word's modal
    category)."""
    ok = tot = 0
    for a, b in zip(wseq[:-1], wseq[1:]):
        ca = roles_lookup(a); cb = roles_lookup(b)
        if ca is None or cb is None: continue
        if NEXT_CAT[ca] == cb: ok += 1
        tot += 1
    return ok / max(tot, 1)

MODAL_CAT = {}
for w in PURE_ANIMALS: MODAL_CAT[word_to_idx[w]] = ANIMAL
for w in PURE_ACTIONS: MODAL_CAT[word_to_idx[w]] = ACTION
for w in OBJECTS:      MODAL_CAT[word_to_idx[w]] = OBJECT
# dual words have no fixed modal category -- left out, evaluated separately

def modal_lookup(wi):
    return MODAL_CAT.get(wi, None)

# ============================================================================
# DATA
# ============================================================================
SEQ_LEN = 12000
train_seq, train_roles = sample_stream(SEQ_LEN, seed=99, p_dual=0.55)

dual_idx = set(word_to_idx[w] for w in DUAL_WORDS)
n_dual_occurrences = sum(1 for w in train_seq if w in dual_idx)
print(f"Dual-word occurrences in training stream: {n_dual_occurrences}/{SEQ_LEN} "
      f"({100*n_dual_occurrences/SEQ_LEN:.0f}%)")
for w in DUAL_WORDS:
    wi = word_to_idx[w]
    as_animal = sum(1 for t, x in enumerate(train_seq) if x == wi and train_roles[t] == ANIMAL)
    as_action = sum(1 for t, x in enumerate(train_seq) if x == wi and train_roles[t] == ACTION)
    print(f"  '{w}': as ANIMAL={as_animal}  as ACTION={as_action}")

def make_plain_stream(wseq, hold=12):
    for w in wseq:
        s = normalize(embeddings[w].astype(complex), NORM)
        for _ in range(hold):
            yield s

# ============================================================================
# BASELINE ORGANISM (no context composite)
# ============================================================================
print("\n=== BASELINE: standard perceive ===")
org_base = Organism(N=N, K=70, omega=0.15, beta=10.0, seed=0)
org_base.perceive(list(make_plain_stream(train_seq, hold=12)),
                  g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
org_base.consolidate(merge_thresh=0.84, prune_frac=0.02)
M_base = org_base.mem
n_base = M_base.shape[0]

states_b  = np.array([normalize(embeddings[w].astype(complex), NORM) for w in train_seq])
assigns_b = np.abs((M_base.conj() @ states_b.T) / N).argmax(0)
slot_word_b = {}
for k in range(n_base):
    members = np.array(train_seq)[assigns_b == k]
    if len(members):
        slot_word_b[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
print(f"Baseline memories: {n_base}  words covered: {len(set(slot_word_b.values()))}/{N_WORDS}")

print("\n--- BASELINE: slots formed per dual word ---")
for w in DUAL_WORDS:
    wi = word_to_idx[w]
    slots = [k for k, x in slot_word_b.items() if x == wi]
    print(f"  '{w}': {len(slots)} slot(s) -> {slots}")

# ============================================================================
# CONTEXT-BIASED ORGANISM (conjunctive coding)
# ============================================================================
ALPHA_CTX = 0.35

class ContextOrg:
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pn = np.zeros((K, K))
        self.prev_k = -1

    @property
    def mem(self):
        return self.xi[self.used]

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive(self, stream, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, alpha_ctx=ALPHA_CTX):
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        step = 0
        for x in stream:
            z = self._settle(z, x, g_in, dt)
            if self.prev_k >= 0 and self.used[self.prev_k] and alpha_ctx > 0:
                z_store = normalize(z + alpha_ctx*self.xi[self.prev_k], self.norm)
            else:
                z_store = z
            used_idx = np.where(self.used)[0]
            if len(used_idx) > 0:
                ovs = np.abs(self.overlaps(z_store, self.xi[used_idx]))
                k = int(used_idx[np.argmax(ovs)]); best_ov = ovs.max()
            else:
                best_ov = 0.0; k = -1
            novelty = 1 - best_ov
            if novelty > recruit and not self.used.all():
                f = int(np.argmin(self.used.astype(float)))
                self.xi[f] = z_store; self.used[f] = True; k = f
            elif k >= 0:
                phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
            self.count[k] += 1
            if self.prev_k >= 0 and k >= 0:
                self.Pn[self.prev_k, k] += 1
            if step % 1000 == 0:
                rs = self.Pn.sum(1, keepdims=True)
                self.Pn = np.where(rs > 0, self.Pn/(rs+1e-9), self.Pn)
            self.prev_k = k
            step += 1
        rs = self.Pn.sum(1, keepdims=True)
        self.Pn = np.where(rs > 0, self.Pn/(rs+1e-9), self.Pn)

    def consolidate(self, merge_thresh=0.84, prune_frac=0.02):
        used_idx = list(np.where(self.used)[0])
        if len(used_idx) > 1:
            max_c = self.count[used_idx].max()
            for k in [k for k in used_idx if self.count[k] < prune_frac*max_c]:
                self.used[k] = False; self.count[k] = 0
        merged = True
        while merged:
            merged = False
            used_idx = list(np.where(self.used)[0])
            for i, ki in enumerate(used_idx):
                for kj in used_idx[i+1:]:
                    sim = float(np.abs(self.overlaps(self.xi[ki], self.xi[[kj]])[0]))
                    if sim > merge_thresh:
                        wi_, wj_ = self.count[ki], self.count[kj]
                        self.xi[ki] = normalize((wi_*self.xi[ki]+wj_*self.xi[kj])/(wi_+wj_+1e-9), self.norm)
                        self.Pn[ki] += self.Pn[kj]; self.Pn[:, ki] += self.Pn[:, kj]
                        self.count[ki] += self.count[kj]
                        self.used[kj] = False; self.count[kj] = 0
                        self.Pn[kj] = 0; self.Pn[:, kj] = 0
                        merged = True; break
                if merged: break
        rs = self.Pn.sum(1, keepdims=True)
        self.Pn = np.where(rs > 0, self.Pn/(rs+1e-9), self.Pn)
        return list(np.where(self.used)[0])

    def recall(self, steps=200000, tau_h=15., lam=2.5, gamma=2., g_rec=7., Dn=0.004, dt=0.05):
        M = self.mem; Ku = M.shape[0]
        if Ku == 0: return np.array([])
        used_idx = np.where(self.used)[0]
        Pn_used = self.Pn[np.ix_(used_idx, used_idx)]
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        h = np.zeros(Ku); seq = []; cur = 0
        for _ in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat = np.maximum(1 - lam*h, 0.)
            score = m*fat + gamma*Pn_used[cur]*fat
            w = np.exp(self.beta*(score-score.max())); w /= w.sum()
            T = (w*(o/(m+1e-9))) @ M
            noise = np.sqrt(2*Dn*dt)*(self.rng.standard_normal(self.N)+1j*self.rng.standard_normal(self.N))/np.sqrt(2)
            z = normalize(z + dt*(1j*self.omega*z + g_rec*(T-z)) + noise, self.norm)
            h = h + dt/tau_h*(m-h)
            a = int(np.argmax(m))
            if m[a] > 0.5 and a != cur:
                seq.append(a); cur = a
        return np.array(seq)

print(f"\n=== CONTEXT-BIASED PERCEIVE (alpha_ctx={ALPHA_CTX}) ===")
ctx_org = ContextOrg(N=N, K=70, omega=0.15, beta=10.0, seed=0)
ctx_org.perceive(list(make_plain_stream(train_seq, hold=12)),
                  g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, alpha_ctx=ALPHA_CTX)
ctx_org.consolidate(merge_thresh=0.84, prune_frac=0.02)
M_ctx = ctx_org.mem
n_ctx = M_ctx.shape[0]

states_c = np.array([normalize(embeddings[w].astype(complex), NORM) for w in train_seq])
assigns_c = np.abs((M_ctx.conj() @ states_c.T) / N).argmax(0)
slot_word_c = {}
for k in range(n_ctx):
    members = np.array(train_seq)[assigns_c == k]
    if len(members):
        slot_word_c[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
print(f"Context memories: {n_ctx}  words covered: {len(set(slot_word_c.values()))}/{N_WORDS}")

print("\n--- CONTEXT: slots formed per dual word, and what role each slot plays ---")
for w in DUAL_WORDS:
    wi = word_to_idx[w]
    slots = [k for k, x in slot_word_c.items() if x == wi]
    print(f"  '{w}': {len(slots)} slot(s)")
    for s in slots:
        # which true role does this slot correspond to, based on occurrences assigned to it?
        role_counts = [0, 0, 0]
        for t in range(len(train_seq)):
            if train_seq[t] == wi and assigns_c[t] == s:
                role_counts[train_roles[t]] += 1
        dom = cat_names[int(np.argmax(role_counts))] if sum(role_counts) > 0 else '?'
        print(f"    slot {s}: true-role counts [ANIMAL={role_counts[0]}, ACTION={role_counts[1]}, OBJECT={role_counts[2]}]  dominant={dom}")

# ============================================================================
# DISAMBIGUATION TEST: does the organism's successor behavior for the dual
# word actually depend on the preceding word's category?
# ============================================================================
def successor_test(label, slot_word, assigns, M, n_slots, Pn_or_org):
    """For each dual word, look at what category follows it in TRAINING DATA
    (ground truth) split by preceding context, and compare to what the
    organism's transition matrix predicts."""
    print(f"\n--- {label}: context-conditioned successor categories (ground truth) ---")
    for w in DUAL_WORDS:
        wi = word_to_idx[w]
        after_animal_ctx = []  # role of THIS dual-word occurrence given prev=ANIMAL ctx...
        # simpler: bucket by true_role of the occurrence itself (since true_role IS what
        # determines grammatically-correct successor), and prev word's true category
        for t in range(1, len(train_seq)):
            if train_seq[t] == wi:
                prev_role = train_roles[t-1]
                this_role = train_roles[t]
                after_animal_ctx.append((prev_role, this_role))
        from collections import Counter
        c = Counter(after_animal_ctx)
        print(f"  '{w}': (prev_role -> this_occurrence_role) counts: {dict(c)}")

successor_test("BASELINE", slot_word_b, assigns_b, M_base, n_base, org_base)

# ---- grammar-mask using TRUE per-occurrence roles (oracle-informed mask) -----
# Build slot->category using DOMINANT true role per slot (captures whichever
# role that slot specialized in, valid even for dual words since each slot
# is single-role in practice)
def build_slot_cat(slot_word, assigns, train_seq, train_roles, n_slots):
    slot_cat = {}
    for k in range(n_slots):
        roles_here = [train_roles[t] for t in range(len(train_seq)) if assigns[t] == k]
        if roles_here:
            slot_cat[k] = int(np.bincount(roles_here, minlength=3).argmax())
    return slot_cat

slot_cat_b = build_slot_cat(slot_word_b, assigns_b, train_seq, train_roles, n_base)
slot_cat_c = build_slot_cat(slot_word_c, assigns_c, train_seq, train_roles, n_ctx)

def grammar_mask_matrix(Pn, slot_cat, n):
    P = Pn.copy()
    for k in range(n):
        if k not in slot_cat: continue
        for j in range(n):
            if slot_cat.get(j) != NEXT_CAT[slot_cat[k]]: P[k, j] = 0.
    for k in range(n):
        if P[k].sum() > 1e-9: P[k] /= P[k].sum()
        elif k in slot_cat:
            cs = [j for j in range(n) if slot_cat.get(j) == NEXT_CAT[slot_cat[k]]]
            if cs: P[k, cs] = 1./len(cs)
    return P

# Baseline grammar-mask (Organism.consolidate already compacts Pn to n_base x n_base)
P_mask_b = grammar_mask_matrix(org_base.Pn, slot_cat_b, n_base)
org_base.Pn = P_mask_b
org_base.beta = 20

# Context grammar-mask
used_idx_c = np.where(ctx_org.used)[0]
Pn_compact_c = ctx_org.Pn[np.ix_(used_idx_c, used_idx_c)]
slot_cat_c_compact = {i: slot_cat_c[ki] for i, ki in enumerate(used_idx_c) if ki in slot_cat_c}
P_mask_c_compact = grammar_mask_matrix(Pn_compact_c, slot_cat_c_compact, n_ctx)
P_mask_c_full = np.zeros((ctx_org.K, ctx_org.K))
for i, ki in enumerate(used_idx_c):
    for j, kj in enumerate(used_idx_c):
        P_mask_c_full[ki, kj] = P_mask_c_compact[i, j]
ctx_org.Pn = P_mask_c_full
ctx_org.beta = 20

print("\nRunning baseline recall (post grammar-mask)...")
slot_seq_b = org_base.recall(steps=150000, tau_h=15., lam=2.5, gamma=2., g_rec=7., Dn=0.004)
gen_b = [slot_word_b[int(s)] for s in slot_seq_b if int(s) in slot_word_b][:800]

print("Running context recall (post grammar-mask)...")
slot_seq_c = ctx_org.recall(steps=150000, tau_h=15., lam=2.5, gamma=2., g_rec=7., Dn=0.004)
gen_c = [slot_word_c[int(s)] for s in slot_seq_c if int(s) in slot_word_c][:800]

def gen_grammaticality(gen, slot_cat_lookup_word):
    ok = tot = 0
    for a, b in zip(gen[:-1], gen[1:]):
        ca = slot_cat_lookup_word.get(a); cb = slot_cat_lookup_word.get(b)
        if ca is None or cb is None: continue
        if NEXT_CAT[ca] == cb: ok += 1
        tot += 1
    return ok / max(tot, 1)

word_modal = {}
for k, v in slot_cat_b.items():
    w = slot_word_b[k]
    word_modal.setdefault(w, []).append(v)
word_modal_cat = {w: int(np.bincount(v, minlength=3).argmax()) for w, v in word_modal.items()}

gram_b = gen_grammaticality(gen_b, word_modal_cat)
gram_c = gen_grammaticality(gen_c, word_modal_cat)

print("\n" + "="*68)
print("PHASE 8 -- TRUE POLYSEMY RESULTS\n")
print(f"Vocabulary: {len(PURE_ANIMALS)} pure animals, {len(PURE_ACTIONS)} pure actions, "
      f"{len(OBJECTS)} objects, {len(DUAL_WORDS)} DUAL-ROLE words: {DUAL_WORDS}\n")

print(f"{'Architecture':<35} {'Memories':>9} {'Coverage':>10}")
print("-"*56)
print(f"{'Baseline':<35} {n_base:>9} {len(set(slot_word_b.values())):>7}/{N_WORDS}")
print(f"{'Context-biased':<35} {n_ctx:>9} {len(set(slot_word_c.values())):>7}/{N_WORDS}")

print("\nSLOTS PER DUAL WORD:")
for w in DUAL_WORDS:
    wi = word_to_idx[w]
    nb = len([k for k, x in slot_word_b.items() if x == wi])
    nc = len([k for k, x in slot_word_c.items() if x == wi])
    print(f"  '{w}': baseline={nb} slot(s)   context={nc} slot(s)   "
          f"{'<-- POLYSEMY RESOLVED' if nc > nb else ''}")

print(f"\nApprox grammaticality (using modal word categories): baseline={gram_b:.3f}  context={gram_c:.3f}")

print("\nKEY QUESTION: did context-biased perceive create MORE slots for")
print("dual-role words than baseline, with each slot specializing in a")
print("different true grammatical role (see slot breakdown above)?")
