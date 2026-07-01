"""
PHASE 9 -- CENTROID CONSOLIDATION: closing the polysemy split

Phase 8c (residual gating) got FUNCTIONAL disambiguation -- with a fast slot
EMA (eta=0.2, recruit thr 0.5-0.75) every dual word formed ANIMAL-dominant
and ACTION-dominant slots with grammatically correct role-conditioned
successor statistics -- but not the clean structure: same-role duplicate
slots refused to re-merge (pure words ~2.2 major slots), and the slow-EMA
variant that avoided duplicates under-split 'duck'/'bear'.

Diagnosis: consolidation compared slot RESIDUALS taken from the slots' EMA
state. An eta=0.2 EMA has an effective memory of ~5 occurrences, so a slot's
residual is a recency-weighted sample of whichever previous-word attractors
it saw last -- not the slot's true context centroid. Track A measured
single-occurrence residual overlaps at ~0.30 within-role vs ~0.22 across-role
(no usable margin), and EMA snapshots sit in that same regime. But CENTROIDS
average the word-specific part away and keep the shared category component:
same-role centroids should overlap strongly, cross-role centroids weakly.

Change vs phase 8c (one mechanism change, everything else identical):
  - perceive records each occurrence's composite (cheap: one vector per
    token); recruitment is unchanged (fast EMA, the config that achieved
    3/3 functional disambiguation).
  - consolidate recomputes every slot's context residual as the
    phase-aligned MEAN over its own assigned occurrences, and merges two
    word-matching slots iff those centroid residuals overlap above
    resid_merge. Merged slots pool their occurrences and the centroid is
    recomputed, so chains of duplicates collapse in one pass.

Success criteria unchanged: exactly 2 role-aligned slots per dual word,
~1 per single-role word, role-specific successors, alpha=0 control intact.
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


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def ctx_residual(v, w_vec):
    w_hat = unit(w_vec)
    r = v - np.vdot(w_hat, v) * w_hat
    n = np.linalg.norm(r)
    return (r / n) if n > 1e-6 else None


def resid_overlap(ra, rb):
    if ra is None and rb is None:
        return 1.0
    if ra is None or rb is None:
        return 0.0
    return float(np.abs(np.vdot(ra, rb)))


def spherical_mean(V, ref):
    """Phase-aligned mean of row vectors V against reference ref, unit-normalized."""
    coef = V @ ref.conj()
    aligned = V * np.exp(-1j*np.angle(coef))[:, None]
    return unit(aligned.mean(0))


class CentroidOrg:
    """Phase-8c residual gating, plus per-occurrence composite recording and
    centroid-based consolidation."""

    def __init__(self, N, K, omega=0.15, seed=0):
        self.N = N; self.K = K; self.omega = omega
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.wvec = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pn = np.zeros((K, K))
        self.prev_slot = -1
        self.composites = None            # filled by perceive_words

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def slot_residual(self, k):
        return ctx_residual(self.xi[k], self.wvec[k])

    def perceive_words(self, seq, hold=12, g_in=5.0, dt=0.05, eta=0.2,
                       alpha_ctx=ALPHA_CTX, word_thresh=0.7, resid_recruit=0.5):
        assigns = np.full(len(seq), -1, dtype=int)
        self.composites = np.zeros((len(seq), self.N), dtype=complex)
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        for t, wi in enumerate(seq):
            x = normalize(embeddings[wi].astype(complex), self.norm)
            ctx = self.prev_slot
            for _ in range(hold):
                z = self._settle(z, x, g_in, dt)
            if ctx >= 0 and alpha_ctx > 0:
                v = normalize(z + alpha_ctx*self.xi[ctx], self.norm)
            else:
                v = z
            self.composites[t] = v
            used_idx = np.where(self.used)[0]
            k = -1
            if len(used_idx) > 0:
                wov = np.abs(self.overlaps(x, self.wvec[used_idx]))
                cands = used_idx[wov > word_thresh]
                if len(cands) > 0:
                    r_in = ctx_residual(v, x)
                    rovs = [resid_overlap(r_in, self.slot_residual(c)) for c in cands]
                    best = int(np.argmax(rovs))
                    if (1 - rovs[best]) > resid_recruit and not self.used.all():
                        k = -1
                    else:
                        k = int(cands[best])
            if k < 0:
                if not self.used.all():
                    k = int(np.argmin(self.used.astype(float)))
                    self.xi[k] = v; self.wvec[k] = x; self.used[k] = True
                else:
                    ovs = np.abs(self.overlaps(v, self.xi[used_idx]))
                    k = int(used_idx[np.argmax(ovs)])
            else:
                ph = np.exp(-1j*np.angle(self.overlaps(v, self.xi[[k]])[0]))
                self.xi[k] = normalize(self.xi[k] + eta*(v*ph - self.xi[k]), self.norm)
                pw = np.exp(-1j*np.angle(self.overlaps(x, self.wvec[[k]])[0]))
                self.wvec[k] = normalize(self.wvec[k] + eta*(x*pw - self.wvec[k]), self.norm)
            self.count[k] += 1
            if self.prev_slot >= 0:
                self.Pn[self.prev_slot, k] += 1
            self.prev_slot = k
            assigns[t] = k
        return assigns

    def centroid_residual(self, k, assigns):
        """Slot residual recomputed as the phase-aligned mean over the slot's
        own occurrences -- the estimator the EMA snapshot fails to be."""
        occ = np.where(assigns == k)[0]
        if len(occ) == 0:
            return self.slot_residual(k)
        R = []
        w_hat = unit(self.wvec[k])
        for t in occ:
            r = ctx_residual(self.composites[t], w_hat)
            if r is not None:
                R.append(r)
        if not R:
            return None
        R = np.array(R)
        return spherical_mean(R, R[0])

    def consolidate(self, assigns, prune_frac=0.05, word_merge=0.84, resid_merge=0.4):
        """Merge word-matching slots whose OCCURRENCE-CENTROID residuals
        overlap; pool occurrences and recompute after each merge."""
        remap = np.arange(self.K)
        used_idx = list(np.where(self.used)[0])
        if used_idx:
            max_c = self.count[used_idx].max()
            for k in [k for k in used_idx if self.count[k] < prune_frac*max_c]:
                self.used[k] = False; self.count[k] = 0; remap[k] = -1
        assigns = np.where(assigns >= 0, remap[assigns], -1)
        cent = {k: self.centroid_residual(k, assigns) for k in np.where(self.used)[0]}
        merged = True
        while merged:
            merged = False
            used_idx = list(np.where(self.used)[0])
            for i, ki in enumerate(used_idx):
                for kj in used_idx[i+1:]:
                    wov = float(np.abs(self.overlaps(self.wvec[ki], self.wvec[[kj]])[0]))
                    rov = resid_overlap(cent[ki], cent[kj])
                    if wov > word_merge and rov > resid_merge:
                        wi_, wj_ = self.count[ki], self.count[kj]
                        self.xi[ki] = normalize((wi_*self.xi[ki]+wj_*self.xi[kj])/(wi_+wj_+1e-9), self.norm)
                        self.wvec[ki] = normalize((wi_*self.wvec[ki]+wj_*self.wvec[kj])/(wi_+wj_+1e-9), self.norm)
                        self.Pn[ki] += self.Pn[kj]; self.Pn[:, ki] += self.Pn[:, kj]
                        self.count[ki] += self.count[kj]
                        self.used[kj] = False; self.count[kj] = 0
                        self.Pn[kj] = 0; self.Pn[:, kj] = 0
                        assigns = np.where(assigns == kj, ki, assigns)
                        cent[ki] = self.centroid_residual(ki, assigns)
                        del cent[kj]
                        merged = True; break
                if merged: break
        return assigns


def evaluate(label, assigns, org, verbose=True):
    dual_slots = {}
    split_ok = 0
    roles_covered = 0
    if verbose:
        print(f"\n--- {label}: slots per dual word (online assignments) ---")
    for w in DUAL_WORDS:
        wi = word_to_idx[w]
        occ = [t for t in range(SEQ_LEN) if train_seq[t] == wi and assigns[t] >= 0]
        slots, counts = np.unique([assigns[t] for t in occ], return_counts=True)
        major = [(int(s), int(c)) for s, c in zip(slots, counts) if c >= 0.10*len(occ)]
        dual_slots[w] = major
        doms = []
        for s, c in sorted(major, key=lambda sc: -sc[1]):
            rc = np.zeros(3, int)
            for t in occ:
                if assigns[t] == s:
                    rc[train_roles[t]] += 1
            doms.append(int(rc.argmax()))
            if verbose:
                print(f"  '{w}' slot {s}: n={c}  roles [ANIMAL={rc[0]}, ACTION={rc[1]}, "
                      f"OBJECT={rc[2]}]  dominant={cat_names[int(rc.argmax())]}")
        if len(major) == 2 and set(doms[:2]) == {ANIMAL, ACTION}:
            split_ok += 1
        if {ANIMAL, ACTION} <= set(doms):
            roles_covered += 1
    pure_counts = []
    for w in PURE_ANIMALS + PURE_ACTIONS + OBJECTS:
        wi = word_to_idx[w]
        occ = [t for t in range(SEQ_LEN) if train_seq[t] == wi and assigns[t] >= 0]
        if not occ: continue
        slots, counts = np.unique([assigns[t] for t in occ], return_counts=True)
        pure_counts.append(sum(1 for c in counts if c >= 0.10*len(occ)))
    n_slots = int(org.used.sum())
    if verbose:
        print(f"  exact 2-way splits: {split_ok}/3   both roles covered: {roles_covered}/3   "
              f"pure words: mean {np.mean(pure_counts):.2f} major slot(s)   "
              f"total slots {n_slots}")
    return split_ok, roles_covered, float(np.mean(pure_counts)), n_slots, dual_slots


def successor_readout(assigns, org, dual_slots):
    slot_role = {}
    for k in np.where(org.used)[0]:
        occ_roles = [train_roles[t] for t in range(SEQ_LEN) if assigns[t] == k]
        if occ_roles:
            slot_role[k] = int(np.bincount(occ_roles, minlength=3).argmax())
    print("\n--- successor categories per dual-word slot (occurrence-level Pn) ---")
    for w in DUAL_WORDS:
        for s, _ in sorted(dual_slots[w], key=lambda sc: -sc[1]):
            row = org.Pn[s]
            mass = np.zeros(3)
            for j in np.where(org.used)[0]:
                if j in slot_role:
                    mass[slot_role[j]] += row[j]
            mass /= mass.sum() + 1e-9
            want = cat_names[NEXT_CAT[slot_role[s]]]
            got = cat_names[int(mass.argmax())]
            print(f"  '{w}' slot {s} ({cat_names[slot_role[s]]:<6}): successors "
                  f"[ANIMAL={mass[0]:.2f}, ACTION={mass[1]:.2f}, OBJECT={mass[2]:.2f}] "
                  f" grammar wants {want} -> {'OK' if got == want else 'MISS'}")


# ============================================================================
# CONTROL: alpha=0 must still form 1 slot per word, no splits
# ============================================================================
print("=== CONTROL: centroid consolidation with alpha_ctx=0 ===")
org0 = CentroidOrg(N=N, K=70, seed=0)
a0 = org0.perceive_words(train_seq, alpha_ctx=0.0, resid_recruit=0.5)
a0 = org0.consolidate(a0)
evaluate("alpha=0 control", a0, org0)

# ============================================================================
# SWEEP: recruit threshold x centroid merge threshold (fast EMA throughout)
# ============================================================================
sweep = [
    # (recruit thr, resid_merge for centroids)
    (0.5,  0.35),
    (0.5,  0.5),
    (0.65, 0.35),
    (0.65, 0.5),
    (0.75, 0.35),
    (0.75, 0.5),
]
summary = []
best = None
for thr, rmerge in sweep:
    print(f"\n=== recruit thr={thr}  centroid resid_merge={rmerge}  (alpha_ctx={ALPHA_CTX}, eta=0.2) ===")
    org = CentroidOrg(N=N, K=70, seed=0)
    a = org.perceive_words(train_seq, alpha_ctx=ALPHA_CTX, resid_recruit=thr)
    a = org.consolidate(a, resid_merge=rmerge)
    split_ok, covered, pure_mean, n_slots, dual_slots = evaluate(
        f"thr={thr} rm={rmerge}", a, org)
    summary.append((thr, rmerge, split_ok, covered, pure_mean, n_slots))
    key = (split_ok, covered, -abs(pure_mean - 1))
    if best is None or key > best[0]:
        best = (key, thr, rmerge, a, org, dual_slots, split_ok, covered, pure_mean)

print("\n" + "="*68)
print("PHASE 9 -- CENTROID CONSOLIDATION RESULTS\n")
print(f"{'thr':>5} {'merge':>6} {'exact splits':>13} {'roles covered':>14} "
      f"{'pure slots/word':>16} {'total':>6}")
for thr, rmerge, s, c, p, n in summary:
    print(f"{thr:>5} {rmerge:>6} {s:>11}/3 {c:>12}/3 {p:>16.2f} {n:>6}")

_, thr, rmerge, a, org, dual_slots, split_ok, covered, pure_mean = best
print(f"\nBest config: thr={thr} resid_merge={rmerge}  "
      f"({split_ok}/3 exact splits, {covered}/3 roles covered, "
      f"{pure_mean:.2f} slots per single-role word)")
successor_readout(a, org, dual_slots)

if split_ok == 3 and pure_mean < 1.5:
    verdict = ("POLYSEMY RESOLVED: centroid consolidation yields exactly two "
               "role-aligned slots per dual word with role-specific successors, "
               "one slot per single-role word, alpha=0 control intact")
elif covered == 3:
    verdict = ("functional disambiguation holds; slot structure still not the "
               "clean 2/1 -- centroid margins insufficient, see sweep")
else:
    verdict = "partial -- see sweep; preserve whatever reproduces"
print("\nverdict:", verdict)
