"""
PHASE 8c -- TRACK B: residual-gated recruitment splits dual-role words

Phase 8's negative result had three proven causes (see
verify_residual_gating.py and the phase-8 header):
  1. the additive composite caps stored-vs-new overlap at 1/(1+alpha^2), so
     the 0.5 recruit gate can never fire on context alone;
  2. consolidate(merge_thresh=0.84) re-merges any same-word split (overlap
     ~0.89 at alpha=0.35);
  3. evaluation assigned occurrences by pure word embedding, which cannot
     attribute two senses to two slots.

This experiment patches all three with WORD-CONDITIONAL RESIDUAL GATING.
Each slot stores its composite attractor xi[k] AND the word direction
wvec[k] that built it (the settled input -- observable online; no category
labels anywhere in the mechanism):

  recruit gate  : find slots matching the incoming WORD direction; within
                  them, compare in the residual space orthogonal to the word.
                  Recruit iff residual novelty exceeds a separate threshold
                  (residuals are lower-magnitude; the composite's 0.5 is not
                  reused -- it is swept below).
  merge         : two slots merge only if they match in word direction AND
                  their context residuals overlap. Same-word/different-
                  context slots are exempt from the 0.84 composite merge.
  evaluation    : the organism's own occurrence-level online assignments
                  (recorded during perceive, remapped through consolidation)
                  -- not a post-hoc pure-embedding argmax.

Two incidental phase-8 accounting fixes, kept because they bear on the
successor-statistics readout:
  - transitions are counted once per occurrence (word -> next word), not on
    every held frame (phase 8 counted 11 self-transitions per word);
  - Pn stays a count matrix during perceive and is normalized once at the
    end (phase 8 renormalized every 1000 steps while still adding raw
    counts, making rows a recency-weighted hybrid).

Success criteria: ~2 role-aligned slots per dual word, ~1 slot per
single-role word, and role-specific successor statistics (the ANIMAL-fish
slot should predict ACTION successors; the ACTION-fish slot, OBJECT).
Controls: alpha=0 (no context) must stay at 1 slot per word.

RESULT (recorded from the committed run): PARTIAL, exactly along the thin
margin Track A measured.
  - alpha=0 control: 1 slot per word, 0 splits -- the gate does not misfire.
  - fast EMA (eta=0.2), thr 0.5-0.75: FUNCTIONAL disambiguation 3/3 -- every
    dual word gets ANIMAL-dominant and ACTION-dominant slots, and every such
    slot's successor distribution matches its role's grammar (the phase-8
    goal); but same-role duplicates over-split (pure words ~2.1-2.3 slots).
  - slow EMA (eta=0.05), thr 0.75-0.85: 'fish' splits EXACTLY 2-way,
    role-aligned; 'duck'/'bear' never split (residual novelty stays under
    threshold once slots average toward the category mean).
  - No config reaches 3/3 exact 2-way splits: the usable threshold window is
    word-dependent because within-vs-across-role residual overlap margins
    are word-dependent (Track A). The remaining gap is consolidation --
    same-role centroids stop overlapping enough to re-merge once formed.
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
    """Component of v orthogonal to the word direction, unit-normalized.
    Returns None when v has essentially no context component."""
    w_hat = unit(w_vec)
    r = v - np.vdot(w_hat, v) * w_hat
    n = np.linalg.norm(r)
    return (r / n) if n > 1e-6 else None


def resid_overlap(ra, rb):
    if ra is None and rb is None:
        return 1.0        # both context-free: same context
    if ra is None or rb is None:
        return 0.0
    return float(np.abs(np.vdot(ra, rb)))


class ResidualOrg:
    """Slot memory with word-conditional residual gating. Decisions are made
    once per occurrence (at the end of the hold, when z has settled onto the
    word), with the context slot frozen at occurrence start -- phase 8's
    per-frame gate blended frames 2..hold with the word's OWN slot, burying
    the context signal under self-similarity."""

    def __init__(self, N, K, omega=0.15, seed=0):
        self.N = N; self.K = K; self.omega = omega
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)     # composite attractors
        self.wvec = np.zeros((K, N), dtype=complex)   # word direction per slot
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pn = np.zeros((K, K))                    # occurrence-level counts
        self.prev_slot = -1

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
                        k = -1                        # same word, novel context
                    else:
                        k = int(cands[best])
            if k < 0:
                if not self.used.all():
                    k = int(np.argmin(self.used.astype(float)))
                    self.xi[k] = v; self.wvec[k] = x; self.used[k] = True
                else:                                  # capacity full: nearest wins
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

    def consolidate(self, assigns, prune_frac=0.05, word_merge=0.84, resid_merge=0.5):
        """Merge only same-word AND same-context slots; prune rare slots.
        Returns assigns remapped through the merges/prunes (-1 = pruned)."""
        remap = np.arange(self.K)
        used_idx = list(np.where(self.used)[0])
        if used_idx:
            max_c = self.count[used_idx].max()
            for k in [k for k in used_idx if self.count[k] < prune_frac*max_c]:
                self.used[k] = False; self.count[k] = 0; remap[k] = -1
        merged = True
        while merged:
            merged = False
            used_idx = list(np.where(self.used)[0])
            for i, ki in enumerate(used_idx):
                for kj in used_idx[i+1:]:
                    wov = float(np.abs(self.overlaps(self.wvec[ki], self.wvec[[kj]])[0]))
                    rov = resid_overlap(self.slot_residual(ki), self.slot_residual(kj))
                    if wov > word_merge and rov > resid_merge:
                        wi_, wj_ = self.count[ki], self.count[kj]
                        self.xi[ki] = normalize((wi_*self.xi[ki]+wj_*self.xi[kj])/(wi_+wj_+1e-9), self.norm)
                        self.wvec[ki] = normalize((wi_*self.wvec[ki]+wj_*self.wvec[kj])/(wi_+wj_+1e-9), self.norm)
                        self.Pn[ki] += self.Pn[kj]; self.Pn[:, ki] += self.Pn[:, kj]
                        self.count[ki] += self.count[kj]
                        self.used[kj] = False; self.count[kj] = 0
                        self.Pn[kj] = 0; self.Pn[:, kj] = 0
                        remap[remap == kj] = ki
                        merged = True; break
                if merged: break
        # collapse chains (a->b->c) and apply
        for k in range(self.K):
            while remap[k] >= 0 and remap[remap[k]] != remap[k]:
                remap[k] = remap[remap[k]]
        return np.where(assigns >= 0, remap[assigns], -1)


def evaluate(label, assigns, org, verbose=True):
    """Score the organism's own online assignments. Role labels are used only
    here, for scoring -- never in the mechanism.
    split_ok       : dual words with EXACTLY two major slots, one per role.
    roles_covered  : dual words with at least one ANIMAL-dominant and one
                     ACTION-dominant major slot (functional disambiguation,
                     even if a role is spread over two slots)."""
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
    """Role-specific successor statistics: the payoff polysemy was blocking.
    Slot roles here are majority labels for REPORTING; grammar truth is
    ANIMAL->ACTION->OBJECT."""
    slot_role = {}
    for k in np.where(org.used)[0]:
        occ_roles = [train_roles[t] for t in range(SEQ_LEN) if assigns[t] == k]
        if occ_roles:
            slot_role[k] = int(np.bincount(occ_roles, minlength=3).argmax())
    print("\n--- successor categories per dual-word slot (from occurrence-level Pn) ---")
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
# CONTROL: alpha=0 (no context) must not split anything
# ============================================================================
print("=== CONTROL: residual gate with alpha_ctx=0 (no context signal) ===")
org0 = ResidualOrg(N=N, K=70, seed=0)
a0 = org0.perceive_words(train_seq, alpha_ctx=0.0, resid_recruit=0.5)
a0 = org0.consolidate(a0)
evaluate("alpha=0 control", a0, org0)

# ============================================================================
# SWEEP (alpha=0.35, the value that failed in phase 8)
# ============================================================================
# Track A (phase8b) measured within-role residual novelty ~0.70 and
# across-role ~0.78 for single occurrences: workable recruit thresholds live
# in that window. A fast EMA (eta=0.2) makes slot residuals recency-weighted
# samples rather than centroids, which both over-recruits online and defeats
# the same-role re-merge in consolidate; a slow EMA (eta=0.05) is swept as
# the stabilized alternative, paired with a lower residual merge threshold.
sweep = [
    # (recruit thr, eta, resid_merge)
    (0.5,  0.2,  0.5),
    (0.65, 0.2,  0.5),
    (0.75, 0.2,  0.5),
    (0.85, 0.2,  0.5),
    (0.65, 0.05, 0.3),
    (0.75, 0.05, 0.3),
    (0.85, 0.05, 0.3),
]
summary = []
best = None
for thr, eta, rmerge in sweep:
    print(f"\n=== recruit thr={thr}  eta={eta}  resid_merge={rmerge}  (alpha_ctx={ALPHA_CTX}) ===")
    org = ResidualOrg(N=N, K=70, seed=0)
    a = org.perceive_words(train_seq, alpha_ctx=ALPHA_CTX, resid_recruit=thr, eta=eta)
    a = org.consolidate(a, resid_merge=rmerge)
    split_ok, covered, pure_mean, n_slots, dual_slots = evaluate(
        f"thr={thr} eta={eta} rm={rmerge}", a, org)
    summary.append((thr, eta, rmerge, split_ok, covered, pure_mean, n_slots))
    key = (split_ok, covered, -abs(pure_mean - 1))
    if best is None or key > best[0]:
        best = (key, thr, eta, rmerge, a, org, dual_slots, split_ok, covered, pure_mean)

print("\n" + "="*68)
print("PHASE 8c -- RESIDUAL GATING RESULTS\n")
print(f"{'thr':>5} {'eta':>5} {'merge':>6} {'exact splits':>13} {'roles covered':>14} "
      f"{'pure slots/word':>16} {'total':>6}")
for thr, eta, rmerge, s, c, p, n in summary:
    print(f"{thr:>5} {eta:>5} {rmerge:>6} {s:>11}/3 {c:>12}/3 {p:>16.2f} {n:>6}")

_, thr, eta, rmerge, a, org, dual_slots, split_ok, covered, pure_mean = best
print(f"\nBest config: thr={thr} eta={eta} resid_merge={rmerge}  "
      f"({split_ok}/3 exact splits, {covered}/3 roles covered, "
      f"{pure_mean:.2f} slots per single-role word)")
successor_readout(a, org, dual_slots)

if split_ok == 3 and pure_mean < 1.5:
    verdict = ("residual gating SPLITS dual-role words into role-aligned slots "
               "with role-specific successors, controls intact")
elif covered == 3:
    verdict = ("FUNCTIONAL disambiguation: every dual word has role-specific "
               "slots and successor statistics, but slot counts are not the "
               "clean 2/1 -- consolidation is the remaining gap")
else:
    verdict = "partial -- see sweep; the negative result stands where splits are missing"
print("\nverdict:", verdict)
