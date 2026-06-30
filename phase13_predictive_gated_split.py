"""
PHASE 13 -- END-TO-END SPLIT WITH PREDICTIVE GATING (the actual fix, live)

Replaces Phase 11's residual-overlap recruit gate with Phase 12's validated
predictive-gain criterion. Two passes, same spirit as Phase 9B->10 (build
statistics, then apply them in perceive) -- but now the statistic used to
decide WHETHER to split is fully unsupervised (predictive gain from
emergent categories), not an oracle.

Pass 1: for every word, compute predictive_split_gain (Phase 12) using
        emergent categories (Phase 10) -- entirely observable.
Pass 2: perceive() where a word is ALLOWED to recruit a second context-
        keyed slot only if its precomputed gain exceeds a threshold;
        words below threshold always single-slot (collapse context),
        exactly matching the formal Myhill-Nerode equivalence test:
        if context doesn't change future behavior, contexts are merged.

Success criterion (same as Phase 11): fish/duck/bear recruit >=2 slots,
single-role controls stay at exactly 1 slot -- now via a principled gate
instead of a tuned magic-number threshold.
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

# ---- Pass 1: predictive gain per word (fully unsupervised) -------------------
GAIN_THRESHOLD = 0.15   # well inside the 0.041-0.341 bit gap found in Phase 12
all_words = PURE_ANIMALS + PURE_ACTIONS + OBJECTS + DUAL_WORDS
word_gain = {}
for w in all_words:
    r = predictive_split_gain(word_to_idx[w])
    word_gain[w] = r['gain'] if r else 0.0

should_split = {w: (g > GAIN_THRESHOLD) for w, g in word_gain.items()}
print("Pass 1 -- predictive gain per word (threshold = {:.2f} bits):".format(GAIN_THRESHOLD))
for w in DUAL_WORDS:
    print(f"  {w:<8} gain={word_gain[w]:.3f}  should_split={should_split[w]}")
n_controls_flagged = sum(1 for w in (PURE_ANIMALS+PURE_ACTIONS+OBJECTS) if should_split[w])
print(f"  Controls flagged for splitting (should be 0): {n_controls_flagged}/{len(PURE_ANIMALS+PURE_ACTIONS+OBJECTS)}")

# ---- Pass 2: perceive() gated by precomputed should_split --------------------
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

    def overlaps(self, z, M):
        return (M.conj() @ z) / self.N

    def _settle(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def perceive_gated(self, wseq, roles, word_to_cat, cat_attractor, should_split,
                        hold=12, g_in=5.0, dt=0.05, eta=0.02, alpha_ctx=ALPHA_CTX,
                        record_words=None):
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)
        prev_word = None
        records = []
        record_words = record_words or set()
        for w, role in zip(wseq, roles):
            x = normalize(embeddings[w].astype(complex), NORM)
            wname = idx_to_word[w]
            split_allowed = should_split.get(wname, False)
            for h in range(hold):
                z = self._settle(z, x, g_in, dt)
                wdir = z  # word-direction reference, settled BEFORE context blend

                prev_cat = word_to_cat.get(prev_word) if prev_word is not None else None
                if split_allowed and prev_cat is not None and prev_cat in cat_attractor and alpha_ctx > 0:
                    z_store = normalize(z + alpha_ctx*cat_attractor[prev_cat], self.norm)
                else:
                    z_store = z   # context collapsed -- word merges all contexts into one slot

                if split_allowed:
                    what = wdir / (np.linalg.norm(wdir) + 1e-9)
                    proj = np.vdot(what, z_store)
                    r = z_store - proj*what
                    r_norm = r / (np.linalg.norm(r) + 1e-9)

                only_last = (h == hold - 1)
                if only_last:
                    if wname in record_words:
                        records.append((wname, role, prev_cat))
                    existing = self.word_to_slots.get(w, [])
                    if not split_allowed:
                        # Force single slot: always match/create the ONE slot for this word
                        if existing:
                            k = existing[0]
                            phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                            self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                        else:
                            if not self.used.all():
                                f = int(np.argmin(self.used.astype(float)))
                                self.xi[f] = z_store; self.used[f] = True
                                self.word_to_slots.setdefault(w, []).append(f)
                                k = f
                            else:
                                k = -1
                    else:
                        # Context-keyed: RESIDUAL-gated recruit among this word's slots
                        # (predictive gain already told us context matters for this word;
                        # residual gating decides which existing sense-slot, if any, matches)
                        if not existing:
                            if not self.used.all():
                                f = int(np.argmin(self.used.astype(float)))
                                self.xi[f] = z_store; self.xi_res[f] = r_norm
                                self.used[f] = True
                                self.word_to_slots.setdefault(w, []).append(f)
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
                                k = f
                            else:
                                k = existing[best_idx]
                                phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                                self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                                r_phase = np.exp(-1j*np.angle(np.vdot(self.xi_res[k], r_norm)))
                                self.xi_res[k] = normalize(self.xi_res[k] + eta*(r_norm*r_phase - self.xi_res[k]), 1.0)
                    if k >= 0:
                        self.count[k] += 1
            prev_word = w
        return records

print(f"\nRunning perceive() gated by predictive-split decision...")
org = PredictiveSplitOrg(N=N, K=80, omega=0.15, beta=10.0, seed=0)
records = org.perceive_gated(train_seq, train_roles, word_to_emergent_cat,
                              emergent_cat_attractor, should_split, hold=12,
                              g_in=5.0, dt=0.05, eta=0.02, alpha_ctx=ALPHA_CTX,
                              record_words=set(DUAL_WORDS))

print("\n" + "="*72)
print("PHASE 13 FINAL RESULT\n")
dual_slots = {w: len(org.word_to_slots.get(word_to_idx[w], [])) for w in DUAL_WORDS}
control_words = PURE_ANIMALS + PURE_ACTIONS + OBJECTS
control_slots = {w: len(org.word_to_slots.get(word_to_idx[w], [])) for w in control_words}

print(f"Dual-role words (target >=2 slots): {dual_slots}")
print(f"\nControl words (target ==1 slot, n={len(control_words)}):")
n_control_violations = sum(1 for v in control_slots.values() if v != 1)
print(f"  {control_slots}")
print(f"\nControl violations (slots != 1): {n_control_violations}/{len(control_words)}")

dual_ok = all(v >= 2 for v in dual_slots.values())
control_ok = (n_control_violations == 0)

print(f"\nDual-role words all split (>=2 slots): {dual_ok}")
print(f"All controls single-slot: {control_ok}")

if dual_ok and control_ok:
    print("\n*** SUCCESS: predictive-gain gating achieves clean polysemy split")
    print("    with ZERO false positives on single-role controls. ***")
else:
    print("\nNot fully clean -- see breakdown above.")

# ---- role purity check for the dual-word slots --------------------------------
print("\n--- Role purity of recruited slots (using true role, evaluation only) ---")
for w in DUAL_WORDS:
    wi = word_to_idx[w]
    occ = [(role, pc) for (wname, role, pc) in records if wname == w]
    # reconstruct slot assignment isn't directly tracked per-occurrence here;
    # report context-conditioned role distribution as a proxy
    from collections import Counter
    c = Counter(occ)
    print(f"  {w}: (true_role, prev_emergent_cat) distribution: {dict(c)}")
