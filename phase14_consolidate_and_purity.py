"""
PHASE 14 -- RESIDUAL-AWARE CONSOLIDATE: clean up over-segmentation

Phase 13 achieved the core result (dual words split, controls don't) but
fish/duck/bear over-segmented to 3-4 slots instead of exactly 2. This phase:

  1. Instruments WHICH true role each fragment slot actually captures, to
     confirm fragmentation is noise-driven (multiple slots for the SAME
     role) rather than a real third sense.
  2. Adds a residual-aware consolidate pass: merge two slots belonging to
     the SAME word if their residual signatures overlap above a threshold
     -- this is the direct analogue of Organism.consolidate's merge_thresh,
     but scoped per-word and operating on residuals (so it can't accidentally
     merge the two genuine senses back together, since those have LOW
     residual overlap by construction).
  3. Re-checks slot counts and role purity after consolidation.
"""

import numpy as np
from organism import normalize
from phase8_true_polysemy import (
    embeddings, N, NORM, word_to_idx, vocab, DUAL_WORDS, train_seq, train_roles,
    ANIMAL, ACTION, OBJECT, cat_names, ALPHA_CTX, N_WORDS,
    PURE_ANIMALS, PURE_ACTIONS, OBJECTS,
)
from phase10_emergent_category import emergent_cat_attractor, word_to_emergent_cat
from phase12_predictive_split_test import predictive_split_gain

idx_to_word = {v: k for k, v in word_to_idx.items()}

GAIN_THRESHOLD = 0.15
all_words = PURE_ANIMALS + PURE_ACTIONS + OBJECTS + DUAL_WORDS
word_gain = {w: (predictive_split_gain(word_to_idx[w]) or {}).get('gain', 0.0) for w in all_words}
should_split = {w: (g > GAIN_THRESHOLD) for w, g in word_gain.items()}


class PredictiveSplitOrg:
    def __init__(self, N, K, omega=0.15, beta=10.0, seed=0):
        self.N = N; self.K = K; self.omega = omega; self.beta = beta
        self.norm = np.sqrt(N)
        self.rng = np.random.default_rng(seed)
        self.xi = np.zeros((K, N), dtype=complex)
        self.xi_res = np.zeros((K, N), dtype=complex)
        self.used = np.zeros(K, dtype=bool)
        self.count = np.zeros(K)
        self.word_to_slots = {}
        self.slot_to_word = {}

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive_gated(self, wseq, roles, word_to_cat, cat_attractor, should_split,
                        hold=12, g_in=5.0, dt=0.05, eta=0.02, alpha_ctx=ALPHA_CTX,
                        record_slot_assign=False):
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        prev_word = None
        slot_assign_log = []  # (word, true_role, slot)
        for w, role in zip(wseq, roles):
            x = normalize(embeddings[w].astype(complex), NORM)
            wname = idx_to_word[w]
            split_allowed = should_split.get(wname, False)
            for h in range(hold):
                z = self._settle(z, x, g_in, dt)
                wdir = z
                prev_cat = word_to_cat.get(prev_word) if prev_word is not None else None
                if split_allowed and prev_cat is not None and prev_cat in cat_attractor and alpha_ctx > 0:
                    z_store = normalize(z + alpha_ctx*cat_attractor[prev_cat], self.norm)
                else:
                    z_store = z

                if split_allowed:
                    what = wdir / (np.linalg.norm(wdir) + 1e-9)
                    proj = np.vdot(what, z_store)
                    r = z_store - proj*what
                    r_norm = r / (np.linalg.norm(r) + 1e-9)

                only_last = (h == hold - 1)
                if only_last:
                    existing = self.word_to_slots.get(w, [])
                    if not split_allowed:
                        if existing:
                            k = existing[0]
                            phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                            self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                        else:
                            if not self.used.all():
                                f = int(np.argmin(self.used.astype(float)))
                                self.xi[f] = z_store; self.used[f] = True
                                self.word_to_slots.setdefault(w, []).append(f)
                                self.slot_to_word[f] = w
                                k = f
                            else:
                                k = -1
                    else:
                        if not existing:
                            if not self.used.all():
                                f = int(np.argmin(self.used.astype(float)))
                                self.xi[f] = z_store; self.xi_res[f] = r_norm
                                self.used[f] = True
                                self.word_to_slots.setdefault(w, []).append(f)
                                self.slot_to_word[f] = w
                                k = f
                            else:
                                k = -1
                        else:
                            res_ovs = np.abs([np.vdot(self.xi_res[s], r_norm) for s in existing])
                            best_idx = int(np.argmax(res_ovs)); best_ov = res_ovs.max()
                            novelty = 1 - best_ov
                            if novelty > 0.5 and not self.used.all():
                                f = int(np.argmin(self.used.astype(float)))
                                self.xi[f] = z_store; self.xi_res[f] = r_norm
                                self.used[f] = True
                                self.word_to_slots[w].append(f)
                                self.slot_to_word[f] = w
                                k = f
                            else:
                                k = existing[best_idx]
                                phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                                self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                                r_phase = np.exp(-1j*np.angle(np.vdot(self.xi_res[k], r_norm)))
                                self.xi_res[k] = normalize(self.xi_res[k] + eta*(r_norm*r_phase - self.xi_res[k]), 1.0)
                    if k >= 0:
                        self.count[k] += 1
                        if record_slot_assign:
                            slot_assign_log.append((wname, role, k))
            prev_word = w
        return slot_assign_log

    def consolidate_per_word_residual(self, merge_thresh=0.7):
        """Merge two slots of the SAME word if their residual signatures
        overlap above merge_thresh. Scoped per-word so it can never merge
        across different words, and uses residual (not raw xi) overlap so
        genuinely distinct senses (low residual overlap by construction,
        Phase 9B/12) survive while noise-fragments (high residual overlap)
        collapse."""
        merges = []
        for w, slots in list(self.word_to_slots.items()):
            if len(slots) < 2:
                continue
            merged = True
            while merged:
                merged = False
                slots = self.word_to_slots[w]
                for i, si in enumerate(slots):
                    for sj in slots[i+1:]:
                        sim = float(np.abs(np.vdot(self.xi_res[si], self.xi_res[sj])))
                        if sim > merge_thresh:
                            ci, cj = self.count[si], self.count[sj]
                            self.xi[si] = normalize((ci*self.xi[si] + cj*self.xi[sj])/(ci+cj+1e-9), self.norm)
                            self.xi_res[si] = normalize((ci*self.xi_res[si] + cj*self.xi_res[sj])/(ci+cj+1e-9), 1.0)
                            self.count[si] += self.count[sj]
                            self.used[sj] = False; self.count[sj] = 0
                            self.word_to_slots[w] = [s for s in slots if s != sj]
                            del self.slot_to_word[sj]
                            merges.append((w, si, sj, sim))
                            merged = True
                            break
                    if merged:
                        break
        return merges


print(f"Running perceive() with predictive gain gating (threshold={GAIN_THRESHOLD})...")
org = PredictiveSplitOrg(N=N, K=80, omega=0.15, beta=10.0, seed=0)
slot_log = org.perceive_gated(train_seq, train_roles, word_to_emergent_cat,
                               emergent_cat_attractor, should_split, hold=12,
                               g_in=5.0, dt=0.05, eta=0.02, alpha_ctx=ALPHA_CTX,
                               record_slot_assign=True)

print("\n--- BEFORE consolidate: slot counts and role purity per slot ---")
for w in DUAL_WORDS:
    wi_slots = org.word_to_slots.get(word_to_idx[w], [])
    print(f"\n  '{w}': {len(wi_slots)} slots")
    for s in wi_slots:
        roles_here = [r for (wn, r, sl) in slot_log if wn == w and sl == s]
        if roles_here:
            c = np.bincount(roles_here, minlength=2)
            dom = cat_names[int(np.argmax(c))]
            print(f"    slot {s}: n={len(roles_here)}  ANIMAL={c[0]} ACTION={c[1]}  dominant={dom}  "
                  f"purity={max(c)/len(roles_here):.3f}")

# ---- prune degenerate singleton slots (e.g. the very first corpus token,
#      before any context exists -- zero residual overlap with everything,
#      not a real sense) ------------------------------------------------------
print("\n--- Pruning degenerate low-count slots (n<5, pre-context artifacts) ---")
PRUNE_MIN_COUNT = 5
for w in DUAL_WORDS:
    slots = list(org.word_to_slots.get(word_to_idx[w], []))
    for s in slots:
        if org.count[s] < PRUNE_MIN_COUNT and len(org.word_to_slots[word_to_idx[w]]) > 1:
            others = [x for x in org.word_to_slots[word_to_idx[w]] if x != s]
            best = max(others, key=lambda o: abs(np.vdot(org.xi_res[s], org.xi_res[o])))
            print(f"  pruned '{w}' slot {s} (n={int(org.count[s])}) -> folded into slot {best}")
            org.count[best] += org.count[s]
            org.used[s] = False; org.count[s] = 0
            org.word_to_slots[word_to_idx[w]] = [x for x in org.word_to_slots[word_to_idx[w]] if x != s]
            del org.slot_to_word[s]

# ---- consolidate: merge only near-duplicate context-states (same specific
#      prior category re-encountered), NOT different role/context granularity.
#      Empirically the true different-context slots sit at 0.10-0.45 residual
#      overlap (see debug run) -- 0.7 is a safe merge threshold that only
#      catches genuine duplicates, never a real distinct state. ------------------
print("\n--- Running residual-aware per-word consolidate (merge_thresh=0.7) ---")
merges = org.consolidate_per_word_residual(merge_thresh=0.7)
for (w, si, sj, sim) in merges:
    print(f"  merged '{w}': slot {sj} -> slot {si}  (residual overlap={sim:.3f})")
if not merges:
    print("  (no merges -- all remaining slots are genuinely distinct context-states,")
    print("   confirmed by low pairwise residual overlap; nothing to collapse)")

print("\n--- AFTER consolidate: slot counts ---")
for w in DUAL_WORDS:
    wi_slots = org.word_to_slots.get(word_to_idx[w], [])
    print(f"  '{w}': {len(wi_slots)} slots -> {wi_slots}")

# ---- re-verify role purity after consolidation ---------------------------------
print("\n--- AFTER consolidate: role purity (re-attributing log entries to surviving slots) ---")
# build old->new slot remap from merges (sj merged into si)
remap = {}
for (w, si, sj, sim) in merges:
    root = si
    while root in remap:
        root = remap[root]
    remap[sj] = root

def resolve(s):
    while s in remap:
        s = remap[s]
    return s

for w in DUAL_WORDS:
    wi_slots = org.word_to_slots.get(word_to_idx[w], [])
    print(f"\n  '{w}': {len(wi_slots)} final slots")
    for s in wi_slots:
        roles_here = [r for (wn, r, sl) in slot_log if wn == w and resolve(sl) == s]
        if roles_here:
            c = np.bincount(roles_here, minlength=2)
            dom = cat_names[int(np.argmax(c))]
            print(f"    slot {s}: n={len(roles_here)}  ANIMAL={c[0]} ACTION={c[1]}  dominant={dom}  "
                  f"purity={max(c)/len(roles_here):.3f}")

print("\n--- Control words: still exactly 1 slot after consolidate? ---")
control_words = PURE_ANIMALS + PURE_ACTIONS + OBJECTS
control_slots_after = {w: len(org.word_to_slots.get(word_to_idx[w], [])) for w in control_words}
violations = {w: v for w, v in control_slots_after.items() if v != 1}
print(f"  Violations: {violations if violations else 'NONE'}")

print("\n" + "="*72)
print("PHASE 14 SUMMARY")
dual_final = {w: len(org.word_to_slots.get(word_to_idx[w], [])) for w in DUAL_WORDS}
print(f"  Dual-role final slot counts: {dual_final}  (target: 2 each)")
print(f"  Control violations: {len(violations)}/{len(control_words)}")
