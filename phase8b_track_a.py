"""
PHASE 8b -- TRACK A: is the REAL context signal separable in residual space?

verify_residual_gating.py proves residual gating separates a word's contexts
when the contexts are distinct vectors. But in the organism the "context" is
the PREVIOUS SLOT'S ATTRACTOR (xi[prev_k]), whose bimodality across senses is
an empirical question: fish-as-ANIMAL follows OBJECT words (10 different
attractors), fish-as-ACTION follows ANIMAL words (8+ attractors), plus 12%
grammar noise, plus the settled state z still carries the previous word.

This diagnostic runs the UNMODIFIED phase-8 context mechanism (additive blend,
alpha=0.35), records the composite for every dual-word occurrence, projects
out the word direction, and asks whether the residuals cluster by true
grammatical role -- WITHOUT ever giving the mechanism the role labels
(labels are used only to score the clustering afterwards).

Two composite variants are tested:
  A1: the faithful phase-8 composite at the FIRST frame of each occurrence
      (the only frame where prev_k is the previous word's slot; frames 2..hold
      blend with the word's own slot -- a contamination worth knowing about).
  A2: the composite Track B would form: fully-settled word state at the LAST
      frame + alpha * xi[context slot captured at occurrence start]
      (attractors taken after training; they are stable by then).

Decision rule (from the handoff brief, refined to three-way):
  PROCEED  : role-aligned clusters (purity well above the ~0.52 majority-role
             chance) with positive silhouette -- residual gating has a signal
             to work with, even if the margin is thin.
  MARGINAL : alignment above chance but silhouette near the noise floor --
             run Track B as an empirical test, expect partial splits.
  STOP     : purity near chance -- the prev-attractor context signal does not
             separate the senses; representation problem, report as the
             boundary condition.

RESULT (recorded from the run committed with this script): purity 0.74-0.83,
silhouette ~0.20 on all three dual words, BUT within-role residual overlap is
only ~0.30 vs ~0.22 across roles: residuals point at the specific previous
WORD's attractor, not at its category mean, so the role signal is the small
shared category component. Controls cluster too (by previous word) -- gating
thresholds in Track B must live inside a thin margin, and slot averaging
(centroids drifting toward the category mean) has to do the amplifying.
"""

import numpy as np
from organism import normalize

# ---- corpus: identical to phase8_true_polysemy.py -----------------------------
PURE_ANIMALS = ['cat','dog','bird','horse','cow','pig','sheep','wolf']
PURE_ACTIONS = ['run','jump','swim','eat','sleep','hunt','hide','play']
OBJECTS      = ['food','water','ground','sky','tree','rock','cave','nest','field','river']
DUAL_WORDS   = ['fish','duck','bear']

vocab = PURE_ANIMALS + PURE_ACTIONS + OBJECTS + DUAL_WORDS
word_to_idx = {w: i for i, w in enumerate(vocab)}
N_WORDS = len(vocab)
ANIMAL, ACTION, OBJECT = 0, 1, 2
cat_names = ['ANIMAL', 'ACTION', 'OBJECT']
NEXT_CAT = {0: 1, 1: 2, 2: 0}
ANIMAL_FILLERS = PURE_ANIMALS + DUAL_WORDS
ACTION_FILLERS = PURE_ACTIONS + DUAL_WORDS
OBJECT_FILLERS = OBJECTS

DIM = 30; N = DIM; NORM = np.sqrt(N)
P_CORRECT = 0.88

emb_rng = np.random.default_rng(13)
cat_bases = np.zeros((3, DIM))
cat_bases[ANIMAL, 0:3] = 1.0
cat_bases[ACTION, 3:6] = 1.0
cat_bases[OBJECT, 6:9] = 1.0
embeddings = np.zeros((N_WORDS, DIM))
for w in PURE_ANIMALS:
    embeddings[word_to_idx[w]] = 0.6*cat_bases[ANIMAL] + 0.4*emb_rng.standard_normal(DIM)
for w in PURE_ACTIONS:
    embeddings[word_to_idx[w]] = 0.6*cat_bases[ACTION] + 0.4*emb_rng.standard_normal(DIM)
for w in OBJECTS:
    embeddings[word_to_idx[w]] = 0.6*cat_bases[OBJECT] + 0.4*emb_rng.standard_normal(DIM)
for w in DUAL_WORDS:
    embeddings[word_to_idx[w]] = 0.3*(cat_bases[ANIMAL]+cat_bases[ACTION]) + 0.5*emb_rng.standard_normal(DIM)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9


def sample_stream(n, seed=None, p_dual=0.5):
    local = np.random.default_rng(seed)
    cat = ANIMAL
    seq = []; roles = []

    def fill(c):
        pool = {ANIMAL: ANIMAL_FILLERS, ACTION: ACTION_FILLERS, OBJECT: OBJECT_FILLERS}[c]
        if c in (ANIMAL, ACTION) and local.random() < p_dual:
            w = local.choice(DUAL_WORDS)
        else:
            w = local.choice([x for x in pool if x not in DUAL_WORDS])
        return word_to_idx[w]

    w = fill(cat); seq.append(w); roles.append(cat)
    for _ in range(n - 1):
        nc = NEXT_CAT[cat] if local.random() < P_CORRECT else int(
            local.choice([c for c in [0, 1, 2] if c != NEXT_CAT[cat]]))
        w = fill(nc); seq.append(w); roles.append(nc); cat = nc
    return seq, roles


SEQ_LEN = 12000
train_seq, train_roles = sample_stream(SEQ_LEN, seed=99, p_dual=0.55)
ALPHA_CTX = 0.35

# ---- faithful phase-8 ContextOrg, instrumented per occurrence ------------------
class RecordingContextOrg:
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.prev_k = -1

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive_words(self, seq, hold=12, g_in=5.0, dt=0.05, eta=0.02,
                       recruit=0.5, alpha_ctx=ALPHA_CTX):
        """Identical dynamics + gate to phase8 ContextOrg.perceive on the held
        stream, but iterates word-by-word so occurrence boundaries are known.
        Records, per occurrence: the context slot at occurrence start, the
        faithful first-frame composite, and the fully-settled last-frame state."""
        rec = {'ctx_slot': [], 'z_store1': [], 'z_end': []}
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        for wi in seq:
            x = normalize(embeddings[wi].astype(complex), self.norm)
            rec['ctx_slot'].append(self.prev_k)
            for h in range(hold):
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
                if (1 - best_ov) > recruit and not self.used.all():
                    f = int(np.argmin(self.used.astype(float)))
                    self.xi[f] = z_store; self.used[f] = True; k = f
                elif k >= 0:
                    phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                    self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                self.count[k] += 1
                if h == 0:
                    rec['z_store1'].append(z_store.copy())
                self.prev_k = k
            rec['z_end'].append(z.copy())
        return rec


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def residual_rows(V, w_vec):
    """Project the word direction out of each row and L2-normalize."""
    w_hat = unit(w_vec)
    R = V - np.outer(V @ w_hat.conj(), w_hat)
    return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)


def spherical_kmeans(R, k=2, iters=60, restarts=20, seed=0):
    """k-means with similarity |<c, r>| (phase-invariant, matching overlaps())."""
    rng = np.random.default_rng(seed)
    best_lab, best_score = None, -1.0
    n = len(R)
    for _ in range(restarts):
        C = R[rng.choice(n, k, replace=False)].copy()
        lab = np.zeros(n, int)
        for _ in range(iters):
            S = np.abs(C.conj() @ R.T)
            new_lab = S.argmax(0)
            if (new_lab == lab).all() and _ > 0:
                break
            lab = new_lab
            for j in range(k):
                m = R[lab == j]
                if len(m) == 0:
                    C[j] = R[rng.integers(n)]
                    continue
                coef = m @ C[j].conj()            # align each member's phase
                aligned = m * np.exp(-1j*np.angle(coef))[:, None]
                C[j] = unit(aligned.mean(0))
        score = np.abs(C.conj() @ R.T).max(0).mean()
        if score > best_score:
            best_score, best_lab = score, lab.copy()
    return best_lab


def silhouette(D, lab):
    n = len(lab); s = np.zeros(n)
    for i in range(n):
        same = lab == lab[i]; same[i] = False
        a = D[i][same].mean() if same.any() else 0.0
        b = min(D[i][lab == c].mean() for c in set(lab) if c != lab[i])
        s[i] = (b - a) / max(a, b, 1e-12)
    return s.mean()


def analyze(word, V, roles, label):
    wi = word_to_idx[word]
    R = residual_rows(V, embeddings[wi].astype(complex))
    G = np.abs(R.conj() @ R.T)
    D = 1 - np.clip(G, 0, 1)
    lab = spherical_kmeans(R, k=2)
    sil = silhouette(D, lab)
    # confusion vs true role (labels used ONLY here, for scoring)
    conf = np.zeros((2, 3), int)
    for l, r in zip(lab, roles):
        conf[l, r] += 1
    purity = sum(conf[l].max() for l in range(2)) / max(conf.sum(), 1)
    # direct separation check: within-role vs across-role residual overlap
    roles = np.asarray(roles)
    within = across = wn = an = 0.0
    for ra in (ANIMAL, ACTION):
        m = roles == ra
        if m.sum() > 1:
            within += G[np.ix_(m, m)].sum() - m.sum(); wn += m.sum()*(m.sum()-1)
    m0, m1 = roles == ANIMAL, roles == ACTION
    if m0.any() and m1.any():
        across = G[np.ix_(m0, m1)].mean()
    within = within / max(wn, 1)
    print(f"  '{word}' [{label}] n={len(V)}  silhouette={sil:.3f}  purity={purity:.3f}  "
          f"within-role ov={within:.3f}  across-role ov={across:.3f}")
    print(f"      confusion (cluster x role): "
          f"c0={conf[0].tolist()}  c1={conf[1].tolist()}  (roles={cat_names})")
    return sil, purity


print(f"Corpus: {SEQ_LEN} tokens, alpha_ctx={ALPHA_CTX} (faithful phase-8 mechanism)")
org = RecordingContextOrg(N=N, K=70, omega=0.15, beta=10.0, seed=0)
rec = org.perceive_words(train_seq, hold=12)
ctx_slot = np.array(rec['ctx_slot'])
Z1 = np.array(rec['z_store1'])
Zend = np.array(rec['z_end'])

results = {}
for variant in ('A1', 'A2'):
    print(f"\n=== Variant {variant}: "
          + ("faithful first-frame composite" if variant == 'A1'
             else "occurrence-end composite (Track B's view)") + " ===")
    print("--- dual words (want 2 role-aligned clusters) ---")
    sils, purs = [], []
    for w in DUAL_WORDS:
        wi = word_to_idx[w]
        occ = [t for t in range(1, SEQ_LEN) if train_seq[t] == wi and ctx_slot[t] >= 0]
        if variant == 'A1':
            V = Z1[occ]
        else:
            V = np.array([normalize(Zend[t] + ALPHA_CTX*org.xi[ctx_slot[t]], NORM) for t in occ])
        roles = [train_roles[t] for t in occ]
        s, p = analyze(w, V, roles, variant)
        sils.append(s); purs.append(p)
    results[variant] = (np.mean(sils), np.mean(purs))
    print("--- single-role controls (clusters, if any, are not sense splits) ---")
    for w in ('cat', 'run', 'food'):
        wi = word_to_idx[w]
        occ = [t for t in range(1, SEQ_LEN) if train_seq[t] == wi and ctx_slot[t] >= 0]
        if variant == 'A1':
            V = Z1[occ]
        else:
            V = np.array([normalize(Zend[t] + ALPHA_CTX*org.xi[ctx_slot[t]], NORM) for t in occ])
        analyze(w, V, [train_roles[t] for t in occ], variant)

print("\n" + "="*68)
print("TRACK A VERDICT")
for variant, (s, p) in results.items():
    print(f"  {variant}: mean dual-word silhouette={s:.3f}  mean role purity={p:.3f}")
sil, pur = max(results.values(), key=lambda sp: sp[1])
if pur >= 0.75 and sil >= 0.10:
    print("  -> PROCEED to Track B: clusters are role-aligned (chance ~0.52) with")
    print("     positive silhouette. Caveat: the within-vs-across-role overlap")
    print("     margin is thin, so expect threshold sensitivity, not a free win.")
elif pur >= 0.65 and sil >= 0.03:
    print("  -> MARGINAL: alignment above chance but weak. Run Track B as an")
    print("     empirical test; preserve the result either way.")
else:
    print("  -> STOP: the prev-attractor context signal does not separate the")
    print("     senses in residual space. Representation problem -- this is the")
    print("     boundary condition; report it rather than patching the gate.")
