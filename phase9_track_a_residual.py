"""
PHASE 9 -- TRACK A: residual clustering diagnostic (validate before patching)

Question: once the shared word-direction component is projected out of the
composite z_store = normalize(z + alpha_ctx * xi[prev_k]), does the
remaining residual separate cleanly by TRUE ROLE (ANIMAL vs ACTION) for the
dual-role words fish/duck/bear?

This instruments phase8_true_polysemy's actual ContextOrg.perceive() loop
(not a toy reimplementation) so the residuals are the real composites the
recruitment decision sees.

Per the handoff brief: use wdir = z BEFORE the context blend (the settled
state) as the word-direction reference, not the static embedding table --
this is what's "available for free" at perceive time and is what the
recruit gate actually compares against.
"""

import numpy as np
from organism import normalize
from phase8_true_polysemy import (
    embeddings, N, NORM, word_to_idx, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, ALPHA_CTX,
)

rng = np.random.default_rng(7)

# ---- replay perceive() with instrumentation -----------------------------------
class InstrumentedContextOrg:
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pn = np.zeros((K, K))
        self.prev_k = -1

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive_instrumented(self, wseq, roles, hold=12, g_in=5.0, dt=0.05,
                               eta=0.02, recruit=0.5, alpha_ctx=ALPHA_CTX,
                               record_words=None):
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        record_words = record_words or set()
        records = []  # (word, true_role, z_store, wdir)
        step = 0
        for w, role in zip(wseq, roles):
            x = normalize(embeddings[w].astype(complex), NORM)
            for _ in range(hold):
                z = self._settle(z, x, g_in, dt)

                wdir = z  # word-direction reference: settled z BEFORE context blend

                if self.prev_k >= 0 and self.used[self.prev_k] and alpha_ctx > 0:
                    z_store = normalize(z + alpha_ctx*self.xi[self.prev_k], self.norm)
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
                if self.prev_k >= 0 and k >= 0:
                    self.Pn[self.prev_k, k] += 1
                if step % 1000 == 0:
                    rs = self.Pn.sum(1, keepdims=True)
                    self.Pn = np.where(rs > 0, self.Pn/(rs+1e-9), self.Pn)
                self.prev_k = k
                step += 1
        return records

idx_to_word = {v: k for k, v in word_to_idx.items()}
def vocab_word(wi):
    return idx_to_word[wi]

print(f"Replaying ContextOrg.perceive on real corpus (alpha_ctx={ALPHA_CTX}), "
      f"instrumenting composites for {DUAL_WORDS}...")
org = InstrumentedContextOrg(N=N, K=70, omega=0.15, beta=10.0, seed=0)
records = org.perceive_instrumented(train_seq, train_roles, hold=12,
                                     g_in=5.0, dt=0.05, eta=0.02, recruit=0.5,
                                     alpha_ctx=ALPHA_CTX, record_words=set(DUAL_WORDS))
print(f"Collected {len(records)} composite samples across hold-steps "
      f"(many per occurrence; will subsample last hold-step per occurrence)")

# Keep only the LAST hold-step per occurrence (the settled, stable z_store) --
# this matches what the recruit/update decision actually sees at decision time.
HOLD = 12
occurrence_records = records[HOLD-1::HOLD]
print(f"Occurrence-level samples (1 per word occurrence): {len(occurrence_records)}")

# ---- spherical k-means (k=2) on raw composites vs on residuals ----------------
def spherical_kmeans_k2(X, n_iter=50, seed=0):
    """X: (n, d) complex array, unit-norm rows. Returns labels (0/1)."""
    rng_ = np.random.default_rng(seed)
    n = X.shape[0]
    centers_idx = rng_.choice(n, size=2, replace=False)
    centers = X[centers_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        sims = np.abs(X.conj() @ centers.T)  # (n,2)
        new_labels = sims.argmax(1)
        if np.array_equal(new_labels, labels) and _ > 0:
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
    """Silhouette score using |1 - |cos sim|| as distance (cosine distance on unit complex vecs)."""
    Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # ensure unit norm
    n = Xu.shape[0]
    sims = np.abs(Xu.conj() @ Xu.T)  # cosine similarity matrix
    dist = 1 - sims
    sil = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        other = ~same
        other[i] = False
        a = dist[i, same].mean() if same.sum() > 0 else 0.0
        b_candidates = [dist[i, labels == c].mean() for c in set(labels) if c != labels[i]]
        b = min(b_candidates) if b_candidates else 0.0
        sil[i] = (b - a) / max(a, b, 1e-9)
    return sil.mean()

def confusion(labels, true_roles):
    """true_roles: 0=ANIMAL,1=ACTION. labels: 0/1 cluster id. Report best-match purity."""
    roles = np.array(true_roles)
    c0_animal = ((labels == 0) & (roles == ANIMAL)).sum()
    c0_action = ((labels == 0) & (roles == ACTION)).sum()
    c1_animal = ((labels == 1) & (roles == ANIMAL)).sum()
    c1_action = ((labels == 1) & (roles == ACTION)).sum()
    # best alignment
    purity_a = (c0_animal + c1_action) / len(labels)
    purity_b = (c0_action + c1_animal) / len(labels)
    purity = max(purity_a, purity_b)
    return purity, (c0_animal, c0_action, c1_animal, c1_action)

print("\n" + "="*72)
print("TRACK A RESULTS: residual clustering diagnostic\n")
print(f"{'Word':<8}{'N':>6}  {'Raw-z silhouette':>18}  {'Residual silhouette':>20}  {'Residual purity':>16}")
print("-"*72)

results = {}
for w in DUAL_WORDS:
    recs = [r for r in occurrence_records if r[0] == w]
    if len(recs) < 20:
        print(f"{w}: too few occurrences ({len(recs)}), skipping")
        continue
    roles = np.array([r[1] for r in recs])
    Z = np.array([r[2] for r in recs])           # raw composites z_store
    WDIR = np.array([r[3] for r in recs])         # wdir = settled z before blend

    # raw composite clustering
    labels_raw, _ = spherical_kmeans_k2(Z, seed=1)
    sil_raw = silhouette_complex(Z, labels_raw)
    purity_raw, _ = confusion(labels_raw, roles)

    # residual: project out the word-direction component, per-sample wdir
    what = WDIR / (np.linalg.norm(WDIR, axis=1, keepdims=True) + 1e-9)
    proj = np.sum(Z.conj() * what, axis=1, keepdims=True)  # vdot per row
    R = Z - proj * what
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)

    labels_res, _ = spherical_kmeans_k2(R, seed=1)
    sil_res = silhouette_complex(R, labels_res)
    purity_res, breakdown = confusion(labels_res, roles)

    results[w] = dict(sil_raw=sil_raw, sil_res=sil_res, purity_raw=purity_raw,
                       purity_res=purity_res, n=len(recs), breakdown=breakdown)

    print(f"{w:<8}{len(recs):>6}  {sil_raw:>18.3f}  {sil_res:>20.3f}  {purity_res:>16.3f}"
          f"   (raw purity={purity_raw:.3f})")

print("\nCluster breakdown (residual space): (c0_animal, c0_action, c1_animal, c1_action)")
for w, r in results.items():
    print(f"  {w}: {r['breakdown']}  purity={r['purity_res']:.3f}")

# ---- same-role vs different-role residual cosine similarity (GPT's check) -----
print("\n--- Same-role vs different-role residual cosine similarity ---")
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
mean_sil_res = np.mean([r['sil_res'] for r in results.values()])
mean_purity = np.mean([r['purity_res'] for r in results.values()])
print(f"  Mean residual silhouette: {mean_sil_res:.3f}")
print(f"  Mean residual cluster purity vs true role: {mean_purity:.3f}")
if mean_sil_res > 0.4 and mean_purity > 0.85:
    print("  -> RECOVERABLE: context residual cleanly separates by true role.")
    print("     Proceed to Track B (residual-gated recruitment patch).")
else:
    print("  -> NOT CLEANLY RECOVERABLE at this silhouette/purity level.")
    print("     Track B may still help but expect partial splits.")
