"""
PHASE 10 -- CONTEXT-PRIMED SETTLING: context in the dynamics, not the storage rule

Phase 9 localized the remaining polysemy failure to ONLINE MIXING: with
post-hoc composite comparison (store z + alpha*xi[prev], then compare), a
slot that absorbs both senses of a dual word early can never be unmixed --
consolidation can only merge, and replay provably inherits the mixing.

This phase moves context INTO the perception dynamics, which is also the
architecture the organism should have had from the start:

  1. CONTEXT = FIELD CARRY-OVER. No alpha blend, no engineered composite.
     The field state z simply persists between words (it always physically
     did); at the moment a new word arrives, z still lies near the previous
     word's attractor. That carry-over IS the context, at full strength --
     the 1/(1+alpha^2) dilution theorem does not apply because nothing is
     being additively mixed into storage.
  2. RECOGNITION = BASIN CAPTURE. During the first frame of each word, the
     field settles under input drive PLUS the same softmax attractor pull
     the organism already uses in recall (g_mem * (T - z)). If two
     context-variants of the word exist as slots, the carry-over decides
     which basin captures the state -- assignment is winner-take-all in the
     dynamics, not an argmax over stored vectors after the fact.
  3. The slot decision happens at the END OF FRAME 1, where the settled
     state still carries ~13% context (input drive decays it by e^-2 per
     frame): exactly the state Track A (phase 8b, variant A1) verified to be
     role-separable (purity 0.74-0.83). Frames 2..hold refine the word
     direction only; the context-rich frame-1 state is what gets stored.
  4. Recruit gate, occurrence-level transitions, and occurrence-centroid
     consolidation are carried over unchanged from phase 9 (proven parts).
     Role labels appear ONLY in evaluation.

Controls and accuracy checks:
  - carry=False control: z is reset to noise at each word onset, so there is
    no context signal; must stay at 1 slot per word, 0 splits.
  - g_mem=0 ablation: carry-over priming without basin capture -- isolates
    how much work the attractor pull does.
  - The best config is re-run on 3 seeds; a split that does not survive
    seed variation is reported as unstable, not as a result.

RESULT (recorded from the committed runs; STABLE across seeds 0/1/2):
  - Pure carry-over priming (g_mem=0, thr=0.65) reaches full FUNCTIONAL
    disambiguation -- 3/3 dual words form role-covered slots and 10/10
    dual-word slots have grammatically correct role-conditioned successor
    statistics -- with NO alpha parameter and NO engineered composite.
    The organism's own inter-word field carry-over is the context signal.
    This matches phase 9's functional result with a strictly simpler,
    organism-native mechanism, and is the recommended architecture going
    forward.
  - The no-carry control (z reset to the input at onset) forms exactly
    1 slot per word, 0 splits: the context signal, not the gate, drives
    all splitting.
  - NEGATIVE result, twice: attractor pull during perception (g_mem>0)
    suppresses splitting in BOTH orderings. Pull-before-gate drags the
    state into the nearest existing basin and erases the context evidence
    (0/3 coverage). Pull-after-gate still collapses to one slot per dual
    word -- consistent with the pulled state feeding each slot's stale
    historical context back into the carry chain, blurring the signal the
    NEXT word needs. Perception should read the field, not bend it.
  - Still open (unchanged from phase 9): exact 2/1 slot structure. Roles
    split reliably but a role sometimes spreads over 2-3 same-role slots
    that centroid merging at safe thresholds does not pool (0/3 exact,
    stable across seeds).
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
    coef = V @ ref.conj()
    aligned = V * np.exp(-1j*np.angle(coef))[:, None]
    return unit(aligned.mean(0))


class PrimedOrg:
    """Perception with context-primed settling and basin-capture assignment."""

    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.wvec = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.Pn = np.zeros((K, K))
        self.prev_slot = -1
        self.composites = None

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, g_mem, dt, steps=8):
        """One frame of settling. g_mem > 0 adds the SAME softmax attractor
        pull used in recall: existing slots act as basins during perception."""
        for _ in range(steps):
            drive = g_in*(x - z)
            if g_mem > 0 and self.used.any():
                M = self.xi[self.used]
                o = self.overlaps(z, M); m = np.abs(o)
                w = np.exp(self.beta*(m - m.max())); w /= w.sum()
                T = (w*(o/(m+1e-9))) @ M
                drive = drive + g_mem*(T - z)
            z = normalize(z + dt*(1j*self.omega*z + drive), self.norm)
        return z

    def slot_residual(self, k):
        return ctx_residual(self.xi[k], self.wvec[k])

    def perceive_words(self, seq, hold=12, g_in=5.0, g_mem=2.5, dt=0.05, eta=0.2,
                       word_thresh=0.7, resid_recruit=0.5, carry=True):
        assigns = np.full(len(seq), -1, dtype=int)
        self.composites = np.zeros((len(seq), self.N), dtype=complex)
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        for t, wi in enumerate(seq):
            x = normalize(embeddings[wi].astype(complex), self.norm)
            if not carry:   # control: destroy the context signal at word onset
                # reset to the input itself -- removes context without
                # injecting a noise residual for the gate to misfire on
                z = x.copy()
            # frame 1: input-only settling. The carry-over context must be
            # OBSERVED before any attractor pull acts on it -- pulling first
            # drags the state into the nearest existing basin and erases the
            # evidence the recruit gate needs (measured: g_mem during this
            # frame drops role coverage from 3/3 to 0/3).
            z = self._settle(z, x, g_in, 0.0, dt)
            v = z.copy()
            self.composites[t] = v
            used_idx = np.where(self.used)[0]
            k = -1
            if len(used_idx) > 0:
                wov = np.abs(self.overlaps(x, self.wvec[used_idx]))
                cands = used_idx[wov > word_thresh]
                if len(cands) > 0:
                    r_in = ctx_residual(v, x)
                    rovs = [resid_overlap(r_in, self.slot_residual(c)) for c in cands]
                    if (1 - max(rovs)) > resid_recruit and not self.used.all():
                        k = -1                      # same word, novel context
                    else:
                        # basin capture AFTER the gate: a short burst of
                        # attractor pull from the context-rich state lets the
                        # existing variants compete as basins; the winner is
                        # the assignment
                        zb = self._settle(v, x, g_in, g_mem, dt, steps=4) if g_mem > 0 else v
                        cov = np.abs(self.overlaps(zb, self.xi[cands]))
                        k = int(cands[int(np.argmax(cov))])
                        z = zb
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
            # frames 2..hold: input-only refinement toward the pure word;
            # the final state seeds the NEXT word's context carry-over
            for _ in range(hold - 1):
                z = self._settle(z, x, g_in, 0.0, dt)
            pw = np.exp(-1j*np.angle(self.overlaps(z, self.wvec[[k]])[0]))
            self.wvec[k] = normalize(self.wvec[k] + eta*(z*pw - self.wvec[k]), self.norm)
            self.count[k] += 1
            if self.prev_slot >= 0:
                self.Pn[self.prev_slot, k] += 1
            self.prev_slot = k
            assigns[t] = k
        return assigns

    # ---- occurrence-centroid consolidation (unchanged from phase 9) ----------
    def centroid_residual(self, k, assigns):
        occ = np.where(assigns == k)[0]
        if len(occ) == 0:
            return self.slot_residual(k)
        w_hat = unit(self.wvec[k])
        R = []
        for t in occ:
            r = ctx_residual(self.composites[t], w_hat)
            if r is not None:
                R.append(r)
        if not R:
            return None
        R = np.array(R)
        return spherical_mean(R, R[0])

    def consolidate(self, assigns, prune_frac=0.05, word_merge=0.84, resid_merge=0.45):
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
    n_ok = n_tot = 0
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
            ok = got == want; n_ok += ok; n_tot += 1
            print(f"  '{w}' slot {s} ({cat_names[slot_role[s]]:<6}): successors "
                  f"[ANIMAL={mass[0]:.2f}, ACTION={mass[1]:.2f}, OBJECT={mass[2]:.2f}] "
                  f" grammar wants {want} -> {'OK' if ok else 'MISS'}")
    print(f"  successor readout: {n_ok}/{n_tot} slots grammatically correct")


def run_config(seed, g_mem, thr, rmerge, carry=True):
    org = PrimedOrg(N=N, K=70, seed=seed)
    a = org.perceive_words(train_seq, g_mem=g_mem, resid_recruit=thr, carry=carry)
    a = org.consolidate(a, resid_merge=rmerge)
    return org, a


import os
STAGE = os.environ.get('PHASE10_STAGE', 'sweep')

if STAGE == 'sweep':
    print("=== CONTROL: carry=False (context destroyed at word onset) ===")
    org0, a0 = run_config(seed=0, g_mem=2.5, thr=0.5, rmerge=0.45, carry=False)
    evaluate("no-carry control", a0, org0)

    sweep = [
        # (recruit thr, g_mem)
        (0.5,  0.0),    # priming only, no basin pull (ablation)
        (0.65, 0.0),
        (0.5,  2.5),    # priming + basin capture
        (0.65, 2.5),
        (0.8,  2.5),
    ]
    summary = []
    best = None
    for thr, g_mem in sweep:
        print(f"\n=== recruit thr={thr}  g_mem={g_mem}  (carry=True, resid_merge=0.45) ===")
        org, a = run_config(seed=0, g_mem=g_mem, thr=thr, rmerge=0.45)
        s, c, p, n, dual_slots = evaluate(f"thr={thr} g_mem={g_mem}", a, org)
        summary.append((thr, g_mem, s, c, p, n))
        key = (s, c, -abs(p - 1))
        if best is None or key > best[0]:
            best = (key, thr, g_mem, a, org, dual_slots)

    print("\n" + "="*68)
    print("PHASE 10 -- CONTEXT-PRIMED SETTLING RESULTS (seed 0)\n")
    print(f"{'thr':>5} {'g_mem':>6} {'exact splits':>13} {'roles covered':>14} "
          f"{'pure slots/word':>16} {'total':>6}")
    for thr, g_mem, s, c, p, n in summary:
        print(f"{thr:>5} {g_mem:>6} {s:>11}/3 {c:>12}/3 {p:>16.2f} {n:>6}")

    _, thr, g_mem, a, org, dual_slots = best
    print(f"\nBest config: thr={thr} g_mem={g_mem}")
    successor_readout(a, org, dual_slots)
    print(f"\n(next: PHASE10_STAGE=seeds thr={thr} g_mem={g_mem} for stability check)")

elif STAGE == 'seeds':
    thr = float(os.environ['PHASE10_THR'])
    g_mem = float(os.environ['PHASE10_GMEM'])
    print(f"=== SEED-STABILITY CHECK: thr={thr} g_mem={g_mem}, seeds 0/1/2 ===")
    rows = []
    for seed in (0, 1, 2):
        org, a = run_config(seed=seed, g_mem=g_mem, thr=thr, rmerge=0.45)
        s, c, p, n, dual_slots = evaluate(f"seed {seed}", a, org)
        rows.append((seed, s, c, p, n))
        if seed == 0:
            successor_readout(a, org, dual_slots)
    print(f"\n{'seed':>5} {'exact splits':>13} {'roles covered':>14} {'pure slots/word':>16} {'total':>6}")
    for seed, s, c, p, n in rows:
        print(f"{seed:>5} {s:>11}/3 {c:>12}/3 {p:>16.2f} {n:>6}")
    stable = all(s == rows[0][1] and c == rows[0][2] for _, s, c, _, _ in rows)
    print("\nverdict:", ("STABLE across seeds -- report as the phase-10 result"
                         if stable else
                         "seed-dependent -- report the distribution, not the best seed"))
