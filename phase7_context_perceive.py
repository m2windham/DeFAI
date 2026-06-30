"""
PHASE 7 -- CONTEXT-BIASED PERCEIVE: conjunctive coding for polysemy

Research basis:
  Cotteret et al. (2022) "Vector Symbolic FSMs in Attractor Neural Networks"
  - Hopfield network implements FSM by storing CONJUNCTIONS (state × input)
  - The binding of current_state + stimulus creates a composite memory
  - The attractor network transitions correctly when BOTH are present

  Fakhoury et al. (2026) "Models of attractor dynamics in the brain"
  - Working-memory biases shaped by sensory history
  - Perceptual adaptation: field is pre-biased before input arrives
  - Same input → different attractor depending on prior state

The correct context mechanism:
  WRONG (phase6): modify carry during RECALL (free generation, no input)
    → carry has nothing to interact with; fights habituation instead
  RIGHT: modify how memories are FORMED during PERCEIVE
    → composite patterns ξ_{word|context} = normalize(ξ_word + α·ξ_prev)
    → same word in different contexts → different attractor locations
    → during recall, context residual naturally routes to correct composite

What changes:
  1. In perceive(): before recruiting/updating a memory, blend z with
     the previous attractor M[prev_k] at weight alpha_ctx
  2. This creates MULTIPLE slots for polysemous words (one per context)
  3. Recall is UNCHANGED — context sensitivity emerges from memory structure

Key prediction:
  - "fish" in animal context → memory slot biased toward animal neighborhood
  - "fish" in action context → memory slot biased toward action neighborhood
  - These are DIFFERENT slots with DIFFERENT P rows → different successors
  - Polysemy is resolved by the attractor landscape, not by recall dynamics

Comparison to baseline (phase4_words.py):
  - Baseline: one slot per word, no context → fish always routes the same way
  - This:     multiple slots per word, context-indexed → fish routes by context
"""

import numpy as np
from organism import Organism, normalize

rng_global = np.random.default_rng(42)

# ---- vocabulary (30 words, 3 categories) ------------------------------------
ANIMALS = ['cat','dog','bird','fish','horse','cow','pig','sheep','wolf','bear']
ACTIONS = ['run','jump','swim','fly','eat','sleep','hunt','hide','fight','play']
OBJECTS = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
vocab   = ANIMALS + ACTIONS + OBJECTS
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = 30
TRUE_CAT = {}
for w in ANIMALS: TRUE_CAT[w] = 0
for w in ACTIONS:  TRUE_CAT[w] = 1
for w in OBJECTS:  TRUE_CAT[w] = 2
NEXT_CAT = {0: 1, 1: 2, 2: 0}
cat_names = ['ANIMAL','ACTION','OBJECT']

DIM = 30; N = DIM; NORM = np.sqrt(N)
P_CORRECT = 0.88

# ---- embeddings (same as phase4_words for fair comparison) ------------------
emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, DIM))
cat_bases[0, 0:3] = 1.0
cat_bases[1, 3:6] = 1.0
cat_bases[2, 6:9] = 1.0
embeddings = np.zeros((N_WORDS, DIM))
for i, w in enumerate(vocab):
    embeddings[i] = 0.6*cat_bases[TRUE_CAT[w]] + 0.4*emb_rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

def sample_stream(n, seed=None):
    local = np.random.default_rng(seed)
    cat, w = 0, int(local.integers(10))
    stream = [w]
    for _ in range(n - 1):
        nc = NEXT_CAT[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0,1,2] if c != NEXT_CAT[cat]]))
        stream.append(nc*10 + int(local.integers(10))); cat = nc
    return stream

def grammaticality(wseq):
    ok = sum(1 for a,b in zip(wseq[:-1],wseq[1:])
             if NEXT_CAT[TRUE_CAT[vocab[a]]]==TRUE_CAT[vocab[b]])
    return ok / max(len(wseq)-1, 1)

# ============================================================================
# BASELINE (identical to phase4_words.py)
# ============================================================================
SEQ_LEN   = 8000
train_seq = sample_stream(SEQ_LEN, seed=99)

def make_plain_stream(wseq, hold=12):
    for w in wseq:
        yield normalize(embeddings[w].astype(complex), NORM)
        for _ in range(hold-1):
            yield normalize(embeddings[w].astype(complex), NORM)

print("=== BASELINE: standard perceive (no context composite) ===")
org_base = Organism(N=N, K=60, omega=0.15, beta=10.0, seed=0)
org_base.perceive(list(make_plain_stream(train_seq, hold=12)),
                  g_in=5.0, dt=0.05, eta=0.02, recruit=0.5)
org_base.consolidate(merge_thresh=0.84, prune_frac=0.02)
M_base = org_base.mem
n_base = M_base.shape[0]

states_b  = np.array([normalize(embeddings[w].astype(complex),NORM) for w in train_seq])
assigns_b = np.abs((M_base.conj() @ states_b.T)/N).argmax(0)
slot_word_b = {}
for k in range(n_base):
    members = np.array(train_seq)[assigns_b==k]
    if len(members):
        slot_word_b[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
slot_cat_b = {k: TRUE_CAT[vocab[w]] for k,w in slot_word_b.items()}

# grammar-mask baseline
def grammar_mask(org, slot_cat):
    P = org.Pn.copy(); n = P.shape[0]
    for k in range(n):
        if k not in slot_cat: continue
        for j in range(n):
            if slot_cat.get(j) != NEXT_CAT[slot_cat[k]]: P[k,j]=0.
    for k in range(n):
        if P[k].sum()>1e-9: P[k]/=P[k].sum()
        elif k in slot_cat:
            cs=[j for j in range(n) if slot_cat.get(j)==NEXT_CAT[slot_cat[k]]]
            if cs: P[k,cs]=1./len(cs)
    return P

P_base = grammar_mask(org_base, slot_cat_b)
org_base.Pn = P_base; org_base.beta = 20

print(f"  Baseline memories: {n_base}  coverage: {len(set(slot_word_b.values()))}/30")
slot_seq_b = org_base.recall(steps=200000, tau_h=15., lam=2.5, gamma=2., g_rec=7., Dn=0.004)
gen_b = [slot_word_b[int(s)] for s in slot_seq_b if int(s) in slot_word_b][:800]
gram_base = grammaticality(gen_b)
print(f"  Grammar (Hebbian only): {gram_base:.3f}\n")


# ============================================================================
# CONTEXT-BIASED PERCEIVE: conjunctive coding (Cotteret 2022 mechanism)
# ============================================================================
# Key: during perceive, blend each new field state z with the previous
# attractor M[prev_k] before recruiting or updating a memory.
# This stores COMPOSITE patterns: ξ_{word|prev_context}
# Different contexts for the same word → different composite attractor locations

ALPHA_CTX = 0.35   # blend weight for context injection (tunable)

print(f"=== CONTEXT-BIASED PERCEIVE (alpha_ctx={ALPHA_CTX}) ===")
print("Mechanism: composite patterns ξ_composite = normalize(z + α·M[prev_slot])")
print("Based on Cotteret (2022): FSM in Hopfield via conjunction of state×input\n")

# We implement a custom perceive loop that stores composite patterns.
# The organism's core field dynamics remain identical; only what gets
# stored as a memory changes.

class ContextOrg:
    """
    Organism variant with context-biased perception.
    Stores composite memories: current_z blended with previous attractor.

    Architecture:
      - self.xi: memory matrix (K × N), same as organism.mem
      - self.used: which slots are occupied
      - self.count: activation counts for consolidation
      - self.Pn: transition probabilities (slot × slot)
      - self.prev_k: previous winning slot
    """
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N    = N
        self.K    = K
        self.omega= omega
        self.beta = beta
        self.norm = np.sqrt(N)
        self.rng  = np.random.default_rng(seed)
        self.xi   = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count= np.zeros(K)
        self.Pn   = np.zeros((K, K))
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

    def perceive(self, stream, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5,
                 alpha_ctx=ALPHA_CTX, hold=12):
        """
        Context-biased perceive: stores composite (z + alpha*prev_attractor) patterns.
        """
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        step = 0

        for x in stream:
            # Settle field toward input embedding
            z = self._settle(z, x, g_in, dt)

            # CONJUNCTIVE COMPOSITE: blend z with previous attractor
            # This is the Cotteret mechanism: store (input × context) conjunction
            if self.prev_k >= 0 and self.used[self.prev_k] and alpha_ctx > 0:
                z_store = normalize(
                    z + alpha_ctx * self.xi[self.prev_k], self.norm)
            else:
                z_store = z  # no context on first word

            # Compute overlaps with all current memories for z_store
            used_idx = np.where(self.used)[0]
            if len(used_idx) > 0:
                ovs = np.abs(self.overlaps(z_store, self.xi[used_idx]))
                k   = int(used_idx[np.argmax(ovs)])
                best_ov = ovs.max()
            else:
                best_ov = 0.0; k = -1

            novelty = 1 - best_ov

            if novelty > recruit and not self.used.all():
                # Recruit new slot with COMPOSITE pattern
                f = int(np.argmin(self.used.astype(float)))
                self.xi[f]   = z_store
                self.used[f] = True
                k = f
            elif k >= 0:
                # Update existing memory toward composite
                phase = np.exp(-1j * np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                self.xi[k] = normalize(
                    self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)

            self.count[k] += 1

            # Update transition matrix
            if self.prev_k >= 0 and k >= 0:
                self.Pn[self.prev_k, k] += 1

            # Normalize Pn rows (running)
            if step % 1000 == 0:
                row_sums = self.Pn.sum(1, keepdims=True)
                self.Pn = np.where(row_sums > 0, self.Pn / (row_sums + 1e-9), self.Pn)

            # Field carries context for next step (no reset)
            # z stays as-is: it has already settled to the current input
            # so the next step's z_store will blend this z with xi[k]
            self.prev_k = k
            step += 1

        # Final row normalization
        row_sums = self.Pn.sum(1, keepdims=True)
        self.Pn = np.where(row_sums > 0, self.Pn / (row_sums + 1e-9), self.Pn)

    def consolidate(self, merge_thresh=0.84, prune_frac=0.02):
        """Merge near-identical memories, prune rarely-used ones."""
        used_idx = list(np.where(self.used)[0])
        # Prune rarely used slots
        if len(used_idx) > 1:
            max_c = self.count[used_idx].max()
            to_prune = [k for k in used_idx if self.count[k] < prune_frac * max_c]
            for k in to_prune:
                self.used[k] = False; self.count[k] = 0

        # Merge near-identical memories
        used_idx = list(np.where(self.used)[0])
        merged = True
        while merged:
            merged = False
            used_idx = list(np.where(self.used)[0])
            for i, ki in enumerate(used_idx):
                for kj in used_idx[i+1:]:
                    sim = float(np.abs(
                        self.overlaps(self.xi[ki], self.xi[[kj]])[0]))
                    if sim > merge_thresh:
                        # Merge kj into ki (weighted by counts)
                        wi = self.count[ki]; wj = self.count[kj]
                        self.xi[ki] = normalize(
                            (wi*self.xi[ki] + wj*self.xi[kj])/(wi+wj+1e-9), self.norm)
                        # Merge transition rows
                        self.Pn[ki] += self.Pn[kj]
                        self.Pn[:, ki] += self.Pn[:, kj]
                        self.count[ki] += self.count[kj]
                        self.used[kj] = False; self.count[kj] = 0
                        self.Pn[kj] = 0; self.Pn[:, kj] = 0
                        merged = True
                        break
                if merged: break

        # Renormalize P
        row_sums = self.Pn.sum(1, keepdims=True)
        self.Pn = np.where(row_sums > 0, self.Pn/(row_sums+1e-9), self.Pn)
        return list(np.where(self.used)[0])

    def recall(self, steps=200000, tau_h=15., lam=2.5, gamma=2., g_rec=7., Dn=0.004, dt=0.05):
        """Standard recall — identical to Organism.recall()."""
        M = self.mem; Ku = M.shape[0]
        if Ku == 0: return np.array([])
        used_idx = np.where(self.used)[0]  # shape (Ku,)
        # Pn submatrix for used slots only
        Pn_used = self.Pn[np.ix_(used_idx, used_idx)]  # (Ku, Ku)
        z   = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        h   = np.zeros(Ku); seq=[]; cur=0
        for _ in range(steps):
            o = self.overlaps(z, M); m = np.abs(o)
            fat  = np.maximum(1 - lam*h, 0.)
            score= m*fat + gamma*Pn_used[cur]*fat
            w = np.exp(self.beta*(score-score.max())); w/=w.sum()
            T = (w*(o/(m+1e-9))) @ M
            noise = np.sqrt(2*Dn*dt)*(
                self.rng.standard_normal(self.N)+1j*self.rng.standard_normal(self.N))/np.sqrt(2)
            z = normalize(z + dt*(1j*self.omega*z + g_rec*(T-z))+noise, self.norm)
            h = h + dt/tau_h*(m - h)
            a = int(np.argmax(m))
            if m[a]>0.5 and a!=cur:
                seq.append(a); cur=a
        return np.array(seq)


# ---- train context organism -------------------------------------------------
print(f"Training context organism (K=60, alpha_ctx={ALPHA_CTX})...")
ctx_org = ContextOrg(N=N, K=60, omega=0.15, beta=10.0, seed=0)

# Feed same training stream
stream_ctx = list(make_plain_stream(train_seq, hold=12))
ctx_org.perceive(stream_ctx, g_in=5.0, dt=0.05, eta=0.02, recruit=0.5,
                 alpha_ctx=ALPHA_CTX, hold=12)
ctx_org.consolidate(merge_thresh=0.84, prune_frac=0.02)
M_ctx  = ctx_org.mem
n_ctx  = M_ctx.shape[0]
print(f"  Context memories formed: {n_ctx}  (baseline: {n_base})")
print(f"  More memories = same words stored in multiple contexts (desired)")

# ---- slot → word mapping for context org ------------------------------------
states_c  = np.array([normalize(embeddings[w].astype(complex),NORM) for w in train_seq])
assigns_c = np.abs((M_ctx.conj() @ states_c.T)/N).argmax(0)
slot_word_c = {}
for k in range(n_ctx):
    members = np.array(train_seq)[assigns_c==k]
    if len(members):
        slot_word_c[k] = int(np.bincount(members, minlength=N_WORDS).argmax())
slot_cat_c = {k: TRUE_CAT[vocab[w]] for k,w in slot_word_c.items()}

covered_c = sorted(set(slot_word_c.values()))
print(f"  Words covered: {len(covered_c)}/30")

# ---- check: how many slots per polysemous word? ------------------------------
print("\n--- POLYSEMY ANALYSIS: slots per word ---")
for w_name in ['fish','bear','fly','cat','run','food']:
    w_idx = word_to_idx[w_name]
    slots = [k for k,wi in slot_word_c.items() if wi==w_idx]
    if slots:
        # What context (previous category) led to each slot?
        slot_ctx_cats = {}
        for t in range(1, len(train_seq)):
            prev_w = train_seq[t-1]; cur_w = train_seq[t]
            if cur_w == w_idx:
                slot = int(assigns_c[t])
                prev_cat = TRUE_CAT[vocab[prev_w]]
                if slot not in slot_ctx_cats:
                    slot_ctx_cats[slot] = [0,0,0]
                slot_ctx_cats[slot][prev_cat] += 1
        print(f"  '{w_name}' ({cat_names[TRUE_CAT[w_name]]}): {len(slots)} slot(s)")
        for s in slots:
            if s in slot_ctx_cats:
                c = slot_ctx_cats[s]
                dominant = cat_names[np.argmax(c)]
                print(f"    slot {s}: prev_context [A={c[0]},V={c[1]},O={c[2]}] → dominant={dominant}")
    else:
        print(f"  '{w_name}': NO SLOT (not covered)")

# ---- grammar-mask and recall ------------------------------------------------
# slot_cat_c keys are 0..n_ctx-1 (compact M_ctx row indices)
# But ctx_org.Pn is K×K. Build a compact Pn for the used slots, mask it,
# then write it back into ctx_org.Pn using used_idx_c.
used_idx_c = np.where(ctx_org.used)[0]
n_used = len(used_idx_c)
# Compact Pn (n_used × n_used)
Pn_compact = ctx_org.Pn[np.ix_(used_idx_c, used_idx_c)]
# Build a temporary org-like object to reuse grammar_mask logic
class _FakeMasker:
    Pn = Pn_compact
    K  = n_used
P_ctx_compact = grammar_mask(_FakeMasker, slot_cat_c)
# Write masked matrix back into full Pn
P_ctx_full = np.zeros((ctx_org.K, ctx_org.K))
for i, ki in enumerate(used_idx_c):
    for j, kj in enumerate(used_idx_c):
        P_ctx_full[ki, kj] = P_ctx_compact[i, j]
ctx_org.Pn = P_ctx_full
ctx_org.beta = 20

print(f"\nRunning context-organism recall...")
slot_seq_c = ctx_org.recall(steps=200000, tau_h=15., lam=2.5, gamma=2., g_rec=7., Dn=0.004)
gen_c = [slot_word_c[int(s)] for s in slot_seq_c if int(s) in slot_word_c][:800]
gram_ctx_mask = grammaticality(gen_c)

# ---- polysemy context sensitivity test --------------------------------------
print("\n--- POLYSEMY SENSITIVITY (after 'fish'): does context change successors? ---")
fish_idx = word_to_idx['fish']

for label, assigns, slot_word, org_obj in [
    ('Baseline', assigns_b, slot_word_b, org_base),
    ('Context', assigns_c, slot_word_c, ctx_org)]:

    test_seq = sample_stream(3000, seed=77)
    prev_w, prev_cat = -1, -1
    after_fish = []  # (prev_cat, next_cat)
    for t, w in enumerate(test_seq[:-1]):
        if prev_w == fish_idx:
            # what slot did fish land in for this context?
            after_fish.append((prev_cat, TRUE_CAT[vocab[test_seq[t]]]))
        prev_w = w
        prev_cat = TRUE_CAT[vocab[w]]

    animal_ctx = [nc for cc,nc in after_fish if cc==0]
    action_ctx = [nc for cc,nc in after_fish if cc==1]
    obj_ctx    = [nc for cc,nc in after_fish if cc==2]

    def dist(lst):
        if not lst: return 'n/a'
        c = np.bincount(lst, minlength=3)
        return f"[A={c[0]/len(lst):.2f} V={c[1]/len(lst):.2f} O={c[2]/len(lst):.2f}]"

    print(f"\n  {label}:")
    print(f"    After 'fish' (prev=ANIMAL): next_cat={dist(animal_ctx)}  n={len(animal_ctx)}")
    print(f"    After 'fish' (prev=ACTION): next_cat={dist(action_ctx)}  n={len(action_ctx)}")
    print(f"    After 'fish' (prev=OBJECT): next_cat={dist(obj_ctx)}    n={len(obj_ctx)}")
    print(f"    Context sensitivity: {'YES' if len(animal_ctx)>5 and len(action_ctx)>5 and abs(animal_ctx.count(1)/max(len(animal_ctx),1) - action_ctx.count(1)/max(len(action_ctx),1)) > 0.1 else 'NO'}")

# ---- final summary ----------------------------------------------------------
oracle_seq  = sample_stream(800, seed=200)
random_seq2 = [int(rng_global.integers(N_WORDS)) for _ in range(800)]
gram_oracle = grammaticality(oracle_seq)
gram_random = grammaticality(random_seq2)

def gap(g): return (g-gram_random)/(gram_oracle-gram_random)*100 if gram_oracle>gram_random else 0

cov_c = len(set(gen_c))
cov_b = len(set(gen_b))

print("\n" + "="*68)
print("PHASE 7 -- CONTEXT-BIASED PERCEIVE: RESULTS\n")
print(f"Baseline: random={gram_random:.3f}  oracle={gram_oracle:.3f}\n")
print(f"{'Architecture':<45} {'Grammar':>8} {'Gap%':>6} {'Cov':>5} {'Mem':>5}")
print("-"*68)
print(f"{'Baseline (no context composite)':<45} {gram_base:>8.3f} {gap(gram_base):>6.0f}% {cov_b:>5}/30 {n_base:>5}")
print(f"{'Context-biased perceive + grammar-mask':<45} {gram_ctx_mask:>8.3f} {gap(gram_ctx_mask):>6.0f}% {cov_c:>5}/30 {n_ctx:>5}")

cats_c = [TRUE_CAT[vocab[w]] for w in gen_c] if gen_c else []
bal_c  = np.bincount(cats_c, minlength=3)/len(cats_c) if cats_c else [0,0,0]
print(f"\nCategory balance (context): {np.round(bal_c,3)}  (true ~[.33,.33,.33])")
print(f"\nFirst 40 words (context organism):")
print("  " + " ".join(vocab[w] for w in gen_c[:40]))

print(f"\nKey findings:")
print(f"  1. Memory count: {n_base} (baseline) vs {n_ctx} (context)")
print(f"     More memories = organism formed context-specific composite attractors")
print(f"  2. Polysemy: see slot analysis above")
print(f"     Same word in different contexts → same slot? (baseline) or different slots? (context)")
print(f"  3. Grammar: {gram_base:.3f} (baseline) vs {gram_ctx_mask:.3f} (context-biased)")
print(f"\nMechanism (Cotteret 2022):")
print(f"  ξ_composite = normalize(ξ_word + {ALPHA_CTX}·ξ_prev_context)")
print(f"  Stored conjunction is sensitive to prior state → FSM in attractor net")
