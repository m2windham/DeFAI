"""
PHASE 9B -- ORACLE CATEGORY CONTEXT: decisive ablation (Opus's proposed test)

Track A showed the residual r = z_store - proj_word(z_store) is approximately
alpha_ctx * xi[prev_k] -- the SPECIFIC previous word's attractor, not an
abstract category signal. Same-role pairs preceded by different words within
that category are just as far apart as different-role pairs.

This phase isolates exactly one variable: replace xi[prev_k] (specific word
attractor) with the TRUE-CATEGORY-AVERAGED attractor (an oracle -- ground
truth role labels ARE used here, deliberately, only to test capability, not
as a proposed unsupervised solution).

If residual purity snaps to near-1.0 under the oracle: the recruitment/
residual-gating MECHANISM is capable, and the only remaining problem is
*unsupervised category discovery* -- a separate, well-scoped problem.

If residual purity stays low even with oracle category context: the
mechanism itself (residual projection, complex overlap geometry, or the
0.35 alpha_ctx blend) is broken regardless of context quality, and Track B
needs deeper rework, not just better context.

Nothing here is proposed as a final architecture -- this is a diagnostic
to localize the bottleneck, per Opus's recommendation.
"""

import numpy as np
from organism import normalize
from phase8_true_polysemy import (
    embeddings, N, NORM, word_to_idx, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, ALPHA_CTX,
    org_base, slot_word_b, slot_cat_b, n_base,
)

rng = np.random.default_rng(7)
idx_to_word = {v: k for k, v in word_to_idx.items()}
def vocab_word(wi): return idx_to_word[wi]

# ---- build oracle category attractors (averaged xi per TRUE category) --------
# Uses org_base's learned memory slots (already labeled by majority word ->
# true category via slot_cat_b from phase8). This is a one-time oracle
# construction, not used inside any online recruitment decision.
cat_attractor = {}
for c in [ANIMAL, ACTION, OBJECT]:
    slots_in_cat = [k for k in range(n_base) if slot_cat_b.get(k) == c]
    if not slots_in_cat:
        continue
    vecs = org_base.mem[slots_in_cat]   # (n_slots, N) complex
    mean_vec = vecs.mean(0)
    cat_attractor[c] = normalize(mean_vec, NORM)
    print(f"Category {cat_names[c]}: built oracle attractor from {len(slots_in_cat)} slots")

# ---- replay perceive() but bind context = oracle category attractor ----------
class OracleContextOrg:
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pn = np.zeros((K, K))

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive_oracle(self, wseq, roles, cat_attractor, hold=12, g_in=5.0,
                         dt=0.05, eta=0.02, recruit=0.5, alpha_ctx=ALPHA_CTX,
                         record_words=None):
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        record_words = record_words or set()
        records = []  # (word, true_role, z_store, wdir)
        prev_role = None
        step = 0
        for w, role in zip(wseq, roles):
            x = normalize(embeddings[w].astype(complex), NORM)
            for _ in range(hold):
                z = self._settle(z, x, g_in, dt)
                wdir = z

                if prev_role is not None and prev_role in cat_attractor and alpha_ctx > 0:
                    z_store = normalize(z + alpha_ctx*cat_attractor[prev_role], self.norm)
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
                step += 1
            prev_role = role
        return records

print(f"\nReplaying perceive with ORACLE category context (alpha_ctx={ALPHA_CTX})...")
org = OracleContextOrg(N=N, K=70, omega=0.15, beta=10.0, seed=0)
records = org.perceive_oracle(train_seq, train_roles, cat_attractor, hold=12,
                               g_in=5.0, dt=0.05, eta=0.02, recruit=0.5,
                               alpha_ctx=ALPHA_CTX, record_words=set(DUAL_WORDS))

HOLD = 12
occurrence_records = records[HOLD-1::HOLD]
print(f"Occurrence-level samples: {len(occurrence_records)}")

# ---- same clustering / silhouette / purity code as Track A -------------------
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
            labels = new_labels
            break
        labels = new_labels
        for c in range(2):
            members = X[labels == c]
            if len(members) > 0:
                mean = members.mean(0)
                nrm = np.linalg.norm(mean)
                if nrm > 1e-9:
                    centers[c] = mean / nrm
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
    c0_animal = ((labels == 0) & (roles == ANIMAL)).sum()
    c0_action = ((labels == 0) & (roles == ACTION)).sum()
    c1_animal = ((labels == 1) & (roles == ANIMAL)).sum()
    c1_action = ((labels == 1) & (roles == ACTION)).sum()
    purity_a = (c0_animal + c1_action) / len(labels)
    purity_b = (c0_action + c1_animal) / len(labels)
    return max(purity_a, purity_b), (c0_animal, c0_action, c1_animal, c1_action)

print("\n" + "="*72)
print("PHASE 9B RESULTS: oracle-category-context residual clustering\n")
print(f"{'Word':<8}{'N':>6}  {'Residual silhouette':>20}  {'Residual purity':>16}")
print("-"*72)

results = {}
for w in DUAL_WORDS:
    recs = [r for r in occurrence_records if r[0] == w]
    if len(recs) < 20:
        continue
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
    results[w] = dict(sil_res=sil_res, purity_res=purity_res, n=len(recs), breakdown=breakdown)
    print(f"{w:<8}{len(recs):>6}  {sil_res:>20.3f}  {purity_res:>16.3f}")

print("\nSame-role vs different-role residual cosine (oracle category context):")
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
    animal_idx = np.where(roles == ANIMAL)[0]
    action_idx = np.where(roles == ACTION)[0]
    sub = 60
    ai = rng.choice(animal_idx, size=min(sub, len(animal_idx)), replace=False)
    bi = rng.choice(action_idx, size=min(sub, len(action_idx)), replace=False)
    same_animal = np.abs(R[ai].conj() @ R[ai].T)
    same_action = np.abs(R[bi].conj() @ R[bi].T)
    same_role = np.concatenate([same_animal[np.triu_indices(len(ai), 1)],
                                 same_action[np.triu_indices(len(bi), 1)]])
    diff_role = np.abs(R[ai].conj() @ R[bi].T).flatten()
    print(f"  {w}: same-role cos={same_role.mean():.3f}  diff-role cos={diff_role.mean():.3f}  "
          f"separation={'YES' if same_role.mean() - diff_role.mean() > 0.1 else 'NO'}")

print("\n" + "="*72)
print("VERDICT:")
mean_sil = np.mean([r['sil_res'] for r in results.values()])
mean_purity = np.mean([r['purity_res'] for r in results.values()])
print(f"  Mean residual silhouette (oracle context): {mean_sil:.3f}")
print(f"  Mean residual purity vs true role (oracle context): {mean_purity:.3f}")
print(f"\n  For comparison, Phase 9 Track A (real xi[prev_k], no oracle): silhouette=0.190, purity=0.591")
if mean_purity > 0.9:
    print("\n  -> MECHANISM IS CAPABLE. Residual gating + recruitment WORKS when context")
    print("     is category-level rather than word-level. The remaining problem is")
    print("     purely unsupervised category discovery -- a separate, well-scoped task.")
elif mean_purity > mean_purity:
    pass
else:
    print("\n  -> Oracle category context did NOT meaningfully improve purity over real")
    print("     xi[prev_k]. The bottleneck is NOT context granularity alone -- the")
    print("     residual-projection/recruitment mechanism itself needs rework.")
