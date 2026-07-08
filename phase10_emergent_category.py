"""
PHASE 10 -- UNSUPERVISED CATEGORY EMERGENCE via second-order recruitment

Phase 9B proved the mechanism is capable: residual-gated recruitment cleanly
splits fish/duck/bear by role when context is a CATEGORY-averaged attractor
(purity 0.907) instead of a single specific previous word's attractor
(purity 0.591). But 9B used an ORACLE -- the true ANIMAL/ACTION/OBJECT label
-- to build that average. This phase removes the oracle.

Idea: categories are not given, they must emerge the same way word-slots
emerged in Phase 1 -- via greedy recruit/update on a similarity threshold.
The only change is WHAT gets clustered: instead of raw embeddings, we
cluster each word-slot's TRANSITION PROFILE (its row in the Hebbian P
matrix -- which other slots it tends to be followed by). Words in the same
grammatical category have similar successor distributions purely because
of corpus statistics (e.g. all ANIMAL words are usually followed by an
ACTION word) -- no label is used, only observed transition frequencies.

Pipeline:
  1. Take org_base (Phase 8's baseline organism: standard WTA, one slot per
     word, already has a learned Pn transition matrix from the real corpus).
  2. For each word-slot k, build a transition-profile feature vector from
     Pn[k] (successor distribution) and Pn[:,k] (predecessor distribution),
     concatenated and L2-normalized.
  3. Run greedy recruit/EMA-update (same control flow as Organism.perceive)
     over these profile vectors to recruit "category slots" -- no count of
     categories is specified in advance, no labels are used.
  4. Evaluate: do emergent category-slot assignments line up with the true
     ANIMAL/ACTION/OBJECT labels for the single-role (non-dual) words?
  5. If yes: build category context attractors (mean xi of slots assigned
     to each emergent category) and re-run the Phase 9 residual-clustering
     test using EMERGENT (not oracle) category as context for fish/duck/bear.
     Compare purity to the oracle ceiling (0.907) and the raw xi[prev_k]
     floor (0.591).
"""

import numpy as np
from organism import normalize
from phase8_true_polysemy import (
    embeddings, N, NORM, word_to_idx, vocab, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, ALPHA_CTX, N_WORDS,
    org_base, slot_word_b, slot_cat_b, n_base, PURE_ANIMALS, PURE_ACTIONS, OBJECTS,
)

rng = np.random.default_rng(11)
idx_to_word = {v: k for k, v in word_to_idx.items()}
def vocab_word(wi): return idx_to_word[wi]

# ============================================================================
# STEP 1-3: emergent category recruitment over transition profiles
# ============================================================================
Pn = org_base.Pn  # (n_base, n_base), already row-normalized
print(f"Baseline organism: {n_base} word-slots, Pn shape {Pn.shape}")

# transition-profile feature per slot: [outgoing row | incoming column], L2-norm
profiles = np.concatenate([Pn, Pn.T], axis=1)  # (n_base, 2*n_base)
profiles = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-9)

def greedy_recruit_cluster(X, recruit_thresh, eta=0.1, seed=0):
    """Same control flow as Organism.perceive's WTA recruitment, applied to
    arbitrary feature vectors. No k specified -- categories emerge from
    the recruit threshold alone, exactly like word-slots emerged from
    embedding novelty in Phase 1."""
    n, d = X.shape
    proto = np.zeros((n, d))   # at most n category prototypes (upper bound)
    used = np.zeros(n, dtype=bool)
    counts = np.zeros(n)
    assign = np.full(n, -1)
    order = np.arange(n)
    local_rng = np.random.default_rng(seed)
    local_rng.shuffle(order)
    for i in order:
        x = X[i]
        used_idx = np.where(used)[0]
        if len(used_idx) > 0:
            sims = used_idx[np.newaxis, :]  # placeholder
            overlaps = X[i] @ proto[used_idx].T  # cosine since both unit-norm
            best = used_idx[np.argmax(overlaps)]
            best_ov = overlaps.max()
        else:
            best_ov = -1.0; best = -1
        novelty = 1 - best_ov
        if novelty > recruit_thresh and (~used).any():
            f = int(np.argmin(used.astype(float)))
            proto[f] = x; used[f] = True
            assign[i] = f; counts[f] += 1
        elif best >= 0:
            proto[best] = proto[best] + eta*(x - proto[best])
            nrm = np.linalg.norm(proto[best])
            if nrm > 1e-9: proto[best] /= nrm
            assign[i] = best; counts[best] += 1
        else:
            f = 0; proto[f] = x; used[f] = True; assign[i] = f; counts[f] += 1
    cat_slots = sorted(set(assign.tolist()))
    return assign, proto, cat_slots, counts

# sweep recruit threshold to find where ~3 emergent categories form
print("\n--- Sweeping recruit threshold for emergent category count ---")
true_cat_lookup = {}
for w in PURE_ANIMALS: true_cat_lookup[w] = ANIMAL
for w in PURE_ACTIONS: true_cat_lookup[w] = ACTION
for w in OBJECTS: true_cat_lookup[w] = OBJECT

def purity_against_true(assign, cat_slots):
    """For single-role words only (dual words excluded -- no single true cat)."""
    correct = 0; total = 0
    per_cluster_majority = {}
    for cs in cat_slots:
        members = [k for k in range(n_base) if assign[k] == cs]
        true_cats = []
        for k in members:
            w = vocab[slot_word_b[k]]
            if w in true_cat_lookup:
                true_cats.append(true_cat_lookup[w])
        if true_cats:
            maj = int(np.bincount(true_cats, minlength=3).argmax())
            per_cluster_majority[cs] = maj
            correct += sum(1 for tc in true_cats if tc == maj)
            total += len(true_cats)
    return (correct/total if total else 0.0), per_cluster_majority

best_result = None
for thresh in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    assign, proto, cat_slots, counts = greedy_recruit_cluster(profiles, thresh, eta=0.15, seed=3)
    pur, maj = purity_against_true(assign, cat_slots)
    print(f"  thresh={thresh:.2f}: n_emergent_categories={len(cat_slots):3d}  purity={pur:.3f}")
    if 2 <= len(cat_slots) <= 6 and (best_result is None or
        (abs(len(cat_slots)-3) < abs(best_result[0]-3))):
        best_result = (len(cat_slots), thresh, assign, proto, cat_slots, pur, maj)

n_cat_found, best_thresh, assign, proto, cat_slots, purity, cluster_majority = best_result
print(f"\nSelected operating point: threshold={best_thresh}, "
      f"{n_cat_found} emergent categories, purity={purity:.3f}")

print("\nEmergent category composition:")
for cs in cat_slots:
    members = [k for k in range(n_base) if assign[k] == cs]
    words = [vocab[slot_word_b[k]] for k in members]
    true_counts = [0, 0, 0]
    for w in words:
        if w in true_cat_lookup:
            true_counts[true_cat_lookup[w]] += 1
    maj_label = cat_names[int(np.argmax(true_counts))] if sum(true_counts) else '?'
    print(f"  Emergent-cat {cs} (n={len(members)}, true-majority={maj_label}): {words[:12]}")

# ============================================================================
# STEP 4-5: build emergent category context attractors, redo residual test
# ============================================================================
emergent_cat_attractor = {}
for cs in cat_slots:
    members = [k for k in range(n_base) if assign[k] == cs]
    if not members: continue
    vecs = org_base.mem[members]
    emergent_cat_attractor[cs] = normalize(vecs.mean(0), NORM)

# map: word -> emergent category slot (via its baseline word-slot's assignment)
word_to_emergent_cat = {}
for k in range(n_base):
    w_idx = slot_word_b[k]
    word_to_emergent_cat[w_idx] = assign[k]

print(f"\nBuilt {len(emergent_cat_attractor)} emergent category attractors.")

class EmergentContextOrg:
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive_emergent(self, wseq, roles, word_to_cat, cat_attractor, hold=12,
                           g_in=5.0, dt=0.05, eta=0.02, recruit=0.5, alpha_ctx=ALPHA_CTX,
                           record_words=None):
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        record_words = record_words or set()
        records = []
        prev_word = None
        for w, role in zip(wseq, roles):
            x = normalize(embeddings[w].astype(complex), NORM)
            for _ in range(hold):
                z = self._settle(z, x, g_in, dt)
                wdir = z
                prev_cat = word_to_cat.get(prev_word) if prev_word is not None else None
                if prev_cat is not None and prev_cat in cat_attractor and alpha_ctx > 0:
                    z_store = normalize(z + alpha_ctx*cat_attractor[prev_cat], self.norm)
                else:
                    z_store = z
                if vocab_word(w) in record_words:
                    records.append((vocab_word(w), role, z_store.copy(), wdir.copy()))
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
            prev_word = w
        return records

print(f"\nReplaying perceive with EMERGENT (unsupervised) category context...")
org2 = EmergentContextOrg(N=N, K=70, omega=0.15, beta=10.0, seed=0)
records = org2.perceive_emergent(train_seq, train_roles, word_to_emergent_cat,
                                  emergent_cat_attractor, hold=12, g_in=5.0, dt=0.05,
                                  eta=0.02, recruit=0.5, alpha_ctx=ALPHA_CTX,
                                  record_words=set(DUAL_WORDS))
HOLD = 12
occurrence_records = records[HOLD-1::HOLD]
print(f"Occurrence-level samples: {len(occurrence_records)}")

def spherical_kmeans_k2(X, n_iter=50, seed=0):
    rng_ = np.random.default_rng(seed)
    n = X.shape[0]
    centers_idx = rng_.choice(n, size=2, replace=False)
    centers = X[centers_idx].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(n_iter):
        sims = np.abs(X.conj() @ centers.T)
        new_labels = sims.argmax(1)
        if np.array_equal(new_labels, labels) and it > 0:
            labels = new_labels; break
        labels = new_labels
        for c in range(2):
            members = X[labels == c]
            if len(members) > 0:
                mean = members.mean(0); nrm = np.linalg.norm(mean)
                if nrm > 1e-9: centers[c] = mean / nrm
    return labels, centers

def silhouette_complex(X, labels):
    Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    n = Xu.shape[0]
    sims = np.abs(Xu.conj() @ Xu.T)
    dist = 1 - sims
    sil = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i]); same[i] = False
        a = dist[i, same].mean() if same.sum() > 0 else 0.0
        b_candidates = [dist[i, labels == c].mean() for c in set(labels) if c != labels[i]]
        b = min(b_candidates) if b_candidates else 0.0
        sil[i] = (b - a) / max(a, b, 1e-9)
    return sil.mean()

def confusion(labels, true_roles):
    roles = np.array(true_roles)
    c0_animal = ((labels==0)&(roles==ANIMAL)).sum(); c0_action = ((labels==0)&(roles==ACTION)).sum()
    c1_animal = ((labels==1)&(roles==ANIMAL)).sum(); c1_action = ((labels==1)&(roles==ACTION)).sum()
    purity_a = (c0_animal+c1_action)/len(labels); purity_b = (c0_action+c1_animal)/len(labels)
    return max(purity_a, purity_b), (c0_animal,c0_action,c1_animal,c1_action)

print("\n" + "="*72)
print("PHASE 10 RESULTS: emergent-category-context residual clustering\n")
print(f"{'Word':<8}{'N':>6}  {'Residual silhouette':>20}  {'Residual purity':>16}")
print("-"*72)
results = {}
for w in DUAL_WORDS:
    recs = [r for r in occurrence_records if r[0] == w]
    if len(recs) < 20: continue
    roles = np.array([r[1] for r in recs])
    Z = np.array([r[2] for r in recs])
    WDIR = np.array([r[3] for r in recs])
    what = WDIR / (np.linalg.norm(WDIR, axis=1, keepdims=True) + 1e-9)
    proj = np.sum(Z.conj()*what, axis=1, keepdims=True)
    R = Z - proj*what
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    labels_res, _ = spherical_kmeans_k2(R, seed=1)
    sil_res = silhouette_complex(R, labels_res)
    purity_res, breakdown = confusion(labels_res, roles)
    results[w] = dict(sil_res=sil_res, purity_res=purity_res)
    print(f"{w:<8}{len(recs):>6}  {sil_res:>20.3f}  {purity_res:>16.3f}")

print("\n" + "="*72)
print("SUMMARY COMPARISON ACROSS ALL THREE CONDITIONS:\n")
print(f"  {'Condition':<45}{'Silhouette':>12}{'Purity':>10}")
print(f"  {'-'*67}")
print(f"  {'Track A: real xi[prev_k] (specific word)':<45}{0.190:>12.3f}{0.591:>10.3f}")
mean_sil_emergent = np.mean([r['sil_res'] for r in results.values()]) if results else 0
mean_pur_emergent = np.mean([r['purity_res'] for r in results.values()]) if results else 0
print(f"  {'Phase 10: emergent category (unsupervised)':<45}{mean_sil_emergent:>12.3f}{mean_pur_emergent:>10.3f}")
print(f"  {'9B: oracle true category (ceiling, uses labels)':<45}{0.886:>12.3f}{0.907:>10.3f}")

print(f"\nEmergent category discovery quality (vs true label, single-role words only): {purity:.3f}")
print(f"Number of emergent categories found: {n_cat_found} (true count: 3)")

if mean_pur_emergent > 0.8:
    print("\n-> SUCCESS: unsupervised category emergence closes most of the gap to oracle.")
    print("   Categories AND senses co-emerge from the same recruit/update mechanism,")
    print("   with zero labels injected anywhere in the pipeline.")
elif mean_pur_emergent > 0.591 + 0.1:
    print("\n-> PARTIAL: emergent categories improve on raw xi[prev_k] but don't reach")
    print("   oracle quality. Category discovery itself is imperfect -- diagnose")
    print("   whether the gap is in category purity or in the residual-gating step.")
else:
    print("\n-> FAILURE: emergent category discovery did not transfer the Phase 9B gain.")
    print("   The transition-profile clustering likely isn't finding clean categories.")
