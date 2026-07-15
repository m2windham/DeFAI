"""
PolysemyOrganism -- first-class, reusable version of the Phase 9B-14 pipeline.

Folds the validated mechanism into a subclass of Organism instead of
standalone per-phase scripts:

  1. discover_categories()  -- unsupervised category emergence (Phase 10):
     greedy recruit/update over each memory slot's Hebbian transition
     profile (Pn row + column), not over embeddings. Categories emerge
     from observed successor/predecessor statistics alone.

  2. perceive_polysemy()    -- single ONLINE streaming pass (Phase 16 --
     replaces Phase 12/13's two-pass precompute-then-apply):
       - maintains running (prev_category -> successor_category) counts
         per word, incrementally, with a one-step lag (need to see t+1's
         category before the gain update for word at t is well-defined)
       - recomputes each word's predictive SPLIT GAIN (entropy reduction
         from conditioning on previous category) from those running counts
         at every occurrence -- no full-corpus pre-pass required
       - gates recruitment with that live gain estimate: below threshold
         (or not enough data yet -- min_count warmup) the word always
         collapses context into ONE shared slot; above threshold, new
         sense-slots are recruited via RESIDUAL-gated comparison (the
         mechanism validated capable in Phase 9B), keyed per-word so
         different words never compete for the same slot pool.

Both stages use only observable corpus statistics. No ground-truth role
labels are used anywhere in this file.
"""

import numpy as np
from collections import Counter
from organism import Organism, normalize


def _entropy(counts):
    counts = np.asarray(list(counts), dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _greedy_recruit_cluster(X, recruit_thresh, eta=0.15, seed=0):
    """Same control-flow as Organism.perceive's WTA recruitment, applied to
    arbitrary unit-norm real feature vectors. No k specified in advance --
    clusters emerge purely from the recruit threshold, exactly like word
    slots emerge from embedding novelty."""
    n, d = X.shape
    proto = np.zeros((n, d))
    used = np.zeros(n, dtype=bool)
    counts = np.zeros(n)
    assign = np.full(n, -1)
    order = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    for i in order:
        x = X[i]
        used_idx = np.where(used)[0]
        if len(used_idx) > 0:
            overlaps = x @ proto[used_idx].T
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
            if nrm > 1e-9:
                proto[best] /= nrm
            assign[i] = best; counts[best] += 1
        else:
            f = 0; proto[f] = x; used[f] = True; assign[i] = f; counts[f] += 1
    cat_slots = sorted(set(assign.tolist()))
    return assign, proto, cat_slots, counts


def _ppmi_transform(counts):
    """Same transform used for word embeddings (Phase 19/20): removes
    frequency-MAGNITUDE bias from a raw co-occurrence-like count matrix,
    leaving only how much MORE (or less) than chance two things co-occur.
    This is the fix for category discovery's real-text failure: raw
    transition-profile clustering (_greedy_recruit_cluster on Pn, which is
    already row-normalized but still reflects each row's absolute
    magnitude structure) let high-frequency words dominate cluster
    formation by sheer count, causing either fragmentation (many
    frequency-isolated singleton categories) or collapse (one giant blob
    absorbing everything of "ordinary" magnitude) depending on scale."""
    total = counts.sum()
    row_sum = counts.sum(1, keepdims=True)
    col_sum = counts.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((counts * total) / (row_sum @ col_sum + 1e-12) + 1e-12)
    return np.maximum(pmi, 0.0)


def _kmeans_real(X, k, n_iter=100, seed=0, n_restarts=6):
    """Standard real-valued k-means with multiple restarts (best inertia).
    Used instead of the old threshold-gated greedy recruitment, which had
    no mechanism to keep cluster sizes balanced -- a single global
    threshold either merges everything or splits everything, with no
    middle ground tunable independent of scale."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia, best_centers = None, np.inf, None
    n = X.shape[0]
    k = min(k, n)
    for r in range(n_restarts):
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()
        labels = np.full(n, -1)
        for _ in range(n_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = d.argmin(1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    centers[c] = members.mean(0)
        d = ((X - centers[labels]) ** 2).sum(-1)
        inertia = d.sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def _silhouette_real(X, labels):
    """Silhouette score on real-valued features (Euclidean)."""
    n = X.shape[0]
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    uniq = sorted(set(labels.tolist()))
    if len(uniq) < 2:
        return -1.0
    sil = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i]).copy()
        same[i] = False
        a = D[i, same].mean() if same.sum() > 0 else 0.0
        others = [c for c in uniq if c != labels[i]]
        b_candidates = [D[i, labels == c].mean() for c in others if (labels == c).any()]
        b = min(b_candidates) if b_candidates else 0.0
        sil[i] = (b - a) / max(a, b, 1e-9)
    return float(sil.mean())


class PolysemyOrganism(Organism):
    def __init__(self, N=128, K=8, omega=0.25, beta=12.0, seed=0, backend=None):
        super().__init__(N=N, K=K, omega=omega, beta=beta, seed=seed, backend=backend)
        # populated by discover_categories()
        self.cat_attractor = {}          # emergent_category_id -> attractor vector (complex, norm=sqrt(N))
        self.word_slot_to_cat = {}       # baseline word-slot index -> emergent_category_id
        # populated by perceive_polysemy()
        self.xi_res = np.zeros((K, N), dtype=complex)   # residual signature per polysemy slot
        self.poly_used = np.zeros(K, dtype=bool)
        self.poly_count = np.zeros(K)
        self.word_to_slots = {}          # word_id -> [slot indices] (polysemy slot pool)
        self.slot_to_word = {}
        # online gain-tracking state (Phase 16)
        self._gain_cond_counts = {}      # word_id -> {prev_cat: Counter(succ_cat)}
        self._gain_uncond_counts = {}    # word_id -> Counter(succ_cat)

    # ------------------------------------------------------------------ #
    # Stage 1: unsupervised category emergence over transition profiles
    # ------------------------------------------------------------------ #
    def discover_categories(self, thresh_sweep=(0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7),
                             target_k=None, eta=0.15, seed=3, verbose=False):
        """Requires self.mem / self.Pn (i.e. consolidate() already called on
        a standard, non-polysemy perceive() pass -- one slot per word)."""
        n = self.Pn.shape[0]
        profiles = np.concatenate([self.Pn, self.Pn.T], axis=1)
        profiles = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-9)

        best = None
        for thresh in thresh_sweep:
            assign, proto, cat_slots, counts = _greedy_recruit_cluster(profiles, thresh, eta=eta, seed=seed)
            n_cats = len(cat_slots)
            if verbose:
                print(f"  thresh={thresh:.2f}: n_emergent_categories={n_cats}")
            if target_k is not None:
                if best is None or abs(n_cats - target_k) < abs(best[0] - target_k):
                    best = (n_cats, thresh, assign, proto, cat_slots)
            else:
                if 2 <= n_cats <= 8:
                    best = (n_cats, thresh, assign, proto, cat_slots)
                    break
        if best is None:
            raise RuntimeError("discover_categories: no threshold produced a usable category count")

        n_cats, thresh, assign, proto, cat_slots = best
        self.word_slot_to_cat = {k: int(assign[k]) for k in range(n)}
        self.cat_attractor = {}
        for cs in cat_slots:
            members = [k for k in range(n) if assign[k] == cs]
            if members:
                self.cat_attractor[cs] = normalize(self.mem[members].mean(0), self.norm)
        return dict(n_categories=n_cats, threshold=thresh, word_slot_to_cat=self.word_slot_to_cat,
                    cat_attractor=self.cat_attractor)

    def discover_categories_v2(self, k_range=range(3, 16), eta=None, seed=3,
                                verbose=False, raw_counts=None):
        """Fixed category discovery: PPMI-transformed transition profiles
        (removes frequency-magnitude bias) + real k-means with k chosen by
        silhouette score (replaces the single-threshold greedy recruitment
        that fragmented on small corpora and collapsed on large ones).

        raw_counts: optional (n,n) raw transition-count matrix restricted
        to the kept memory slots, in the SAME order as self.mem. If not
        given, uses self.P restricted via self.kept_idx (set by
        Organism.consolidate()) -- the normal path when this organism did
        its own perceive()+consolidate(). Pass raw_counts explicitly when
        reconstructing state from a saved/pickled run (e.g. after loading
        org.P from disk with a different index space).
        """
        n = self.mem.shape[0]
        if raw_counts is None:
            if not hasattr(self, 'kept_idx'):
                raise RuntimeError("discover_categories_v2: no raw_counts given and "
                                    "self.kept_idx not set -- call consolidate() first "
                                    "or pass raw_counts explicitly.")
            raw_counts = self.P[np.ix_(self.kept_idx, self.kept_idx)]

        ppmi = _ppmi_transform(raw_counts)
        profiles = np.concatenate([ppmi, ppmi.T], axis=1)
        profiles = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-9)

        best = None
        for k in k_range:
            if k >= n:
                continue
            labels, centers = _kmeans_real(profiles, k, seed=seed)
            sil = _silhouette_real(profiles, labels)
            sizes = np.bincount(labels, minlength=k)
            balance = sizes.min() / max(sizes.max(), 1)   # 1.0 = perfectly balanced
            if verbose:
                print(f"  k={k:2d}: silhouette={sil:.3f}  sizes={sorted(sizes.tolist())}  "
                      f"balance={balance:.3f}")
            if best is None or sil > best[0]:
                best = (sil, k, labels, centers)

        if best is None:
            raise RuntimeError("discover_categories_v2: no k in k_range produced clusters")

        sil, k, labels, centers = best
        self.word_slot_to_cat = {i: int(labels[i]) for i in range(n)}
        self.cat_attractor = {}
        for c in sorted(set(labels.tolist())):
            members = [i for i in range(n) if labels[i] == c]
            if members:
                self.cat_attractor[c] = normalize(self.mem[members].mean(0), self.norm)
        return dict(n_categories=k, silhouette=sil, word_slot_to_cat=self.word_slot_to_cat,
                    cat_attractor=self.cat_attractor)

    def category_of_word(self, word_id, slot_word_map):
        """slot_word_map: {baseline_slot_idx: word_id} from the caller's own
        slot->word attribution (built the same way Phase 8 did, via overlap
        argmax against pure embeddings -- outside this class's scope since
        it needs the training stream, not just organism state)."""
        for k, w in slot_word_map.items():
            if w == word_id:
                return self.word_slot_to_cat.get(k)
        return None

    # ------------------------------------------------------------------ #
    # Stage 2: single online streaming pass -- predictive-gain-gated,
    # residual-gated recruitment. No full-corpus pre-pass.
    # ------------------------------------------------------------------ #
    def _update_gain_stats(self, word, prev_cat, succ_cat):
        if prev_cat is None or succ_cat is None:
            return
        self._gain_uncond_counts.setdefault(word, Counter())[succ_cat] += 1
        self._gain_cond_counts.setdefault(word, {}).setdefault(prev_cat, Counter())[succ_cat] += 1

    def _current_gain(self, word, min_count=60):
        uncond = self._gain_uncond_counts.get(word)
        if not uncond:
            return 0.0
        total = sum(uncond.values())
        if total < min_count:
            return 0.0   # warmup: not enough data yet, default to "no split"
        H_uncond = _entropy(uncond.values())
        cond = self._gain_cond_counts.get(word, {})
        H_cond = 0.0
        for pc, counter in cond.items():
            n_pc = sum(counter.values())
            H_cond += (n_pc/total) * _entropy(counter.values())
        return max(H_uncond - H_cond, 0.0)

    def perceive_polysemy(self, word_id_stream, embeddings, word_to_emergent_cat,
                           hold=12, g_in=5.0, dt=0.05, eta=0.02, alpha_ctx=0.35,
                           gain_threshold=0.15, min_count_for_gain=60,
                           residual_recruit_thresh=0.5):
        """word_id_stream: sequence of integer word ids (one token per
        occurrence, NOT pre-expanded by hold -- this method expands
        internally, matching Organism.perceive's stream convention).
        embeddings: (n_words, N) real or complex array indexable by word id.
        word_to_emergent_cat: {word_id: emergent_category_id}, built from
        discover_categories() + the caller's slot->word attribution.

        Single streaming pass. Predictive gain is recomputed from RUNNING
        counts at every step -- no pre-pass over the full corpus. Gain
        updates lag by one token (need to see t+1's category to finalize
        the gain-relevant successor-category label for token t)."""
        z = normalize(self.rng.standard_normal(self.N).astype(complex), self.norm)

        # sliding window of (word, emergent_cat) for the lagged gain update
        win_prev2 = None   # (word, cat) at t-2
        win_prev1 = None   # (word, cat) at t-1  -- gain update target once cur arrives
        prev_word = None   # word at t-1, used as CONTEXT for the current token

        for w in word_id_stream:
            x = normalize(np.asarray(embeddings[w]).astype(complex), np.sqrt(self.N))
            cur_cat = word_to_emergent_cat.get(w)

            # finalize the lagged gain update for win_prev1 using (cat(prev2), cat(cur))
            if win_prev1 is not None and win_prev2 is not None:
                self._update_gain_stats(win_prev1[0], win_prev2[1], cur_cat)

            for h in range(hold):
                z = self._settle_field(z, x, g_in, dt)
                wdir = z

                prev_cat = word_to_emergent_cat.get(prev_word) if prev_word is not None else None
                gain = self._current_gain(w, min_count=min_count_for_gain)
                split_allowed = gain > gain_threshold

                if split_allowed and prev_cat is not None and prev_cat in self.cat_attractor and alpha_ctx > 0:
                    z_store = normalize(z + alpha_ctx*self.cat_attractor[prev_cat], self.norm)
                else:
                    z_store = z

                if split_allowed:
                    what = wdir / (np.linalg.norm(wdir) + 1e-9)
                    proj = np.vdot(what, z_store)
                    r = z_store - proj*what
                    r_norm = r / (np.linalg.norm(r) + 1e-9)

                if h == hold - 1:
                    existing = self.word_to_slots.get(w, [])
                    if not split_allowed:
                        if existing:
                            k = existing[0]
                            phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                            self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                        else:
                            k = self._recruit_slot(w, z_store, None)
                    else:
                        if not existing:
                            k = self._recruit_slot(w, z_store, r_norm)
                        else:
                            res_ovs = np.abs([np.vdot(self.xi_res[s], r_norm) for s in existing])
                            best_idx = int(np.argmax(res_ovs)); best_ov = res_ovs.max()
                            novelty = 1 - best_ov
                            if novelty > residual_recruit_thresh and not self.poly_used.all():
                                k = self._recruit_slot(w, z_store, r_norm)
                            else:
                                k = existing[best_idx]
                                phase = np.exp(-1j*np.angle(self.overlaps(z_store, self.xi[[k]])[0]))
                                self.xi[k] = normalize(self.xi[k] + eta*(z_store*phase - self.xi[k]), self.norm)
                                r_phase = np.exp(-1j*np.angle(np.vdot(self.xi_res[k], r_norm)))
                                self.xi_res[k] = normalize(self.xi_res[k] + eta*(r_norm*r_phase - self.xi_res[k]), 1.0)
                    if k is not None and k >= 0:
                        self.poly_count[k] += 1

            win_prev2 = win_prev1
            win_prev1 = (w, cur_cat)
            prev_word = w

    def _settle_field(self, z, x, g_in, dt, steps=8):
        for _ in range(steps):
            z = normalize(z + dt*(1j*self.omega*z + g_in*(x - z)), self.norm)
        return z

    def _recruit_slot(self, word_id, z_store, r_norm):
        if self.poly_used.all():
            return -1
        f = int(np.argmin(self.poly_used.astype(float)))
        self.xi[f] = z_store
        if r_norm is not None:
            self.xi_res[f] = r_norm
        self.poly_used[f] = True
        self.word_to_slots.setdefault(word_id, []).append(f)
        self.slot_to_word[f] = word_id
        return f

    def consolidate_polysemy(self, merge_thresh=0.7, prune_min_count=5):
        """Per-word residual-aware cleanup (Phase 14): merge near-duplicate
        sense-slots of the SAME word (high residual overlap => redundant
        fragmentation, not a real distinct sense), and prune degenerate
        low-count slots (e.g. the very first token before any context
        existed) into their nearest residual neighbor."""
        # prune first
        for w, slots in list(self.word_to_slots.items()):
            for s in list(slots):
                if self.poly_count[s] < prune_min_count and len(self.word_to_slots[w]) > 1:
                    others = [x for x in self.word_to_slots[w] if x != s]
                    best = max(others, key=lambda o: abs(np.vdot(self.xi_res[s], self.xi_res[o])))
                    self.poly_count[best] += self.poly_count[s]
                    self.poly_used[s] = False; self.poly_count[s] = 0
                    self.word_to_slots[w] = [x for x in self.word_to_slots[w] if x != s]
                    del self.slot_to_word[s]

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
                            ci, cj = self.poly_count[si], self.poly_count[sj]
                            self.xi[si] = normalize((ci*self.xi[si]+cj*self.xi[sj])/(ci+cj+1e-9), self.norm)
                            self.xi_res[si] = normalize((ci*self.xi_res[si]+cj*self.xi_res[sj])/(ci+cj+1e-9), 1.0)
                            self.poly_count[si] += self.poly_count[sj]
                            self.poly_used[sj] = False; self.poly_count[sj] = 0
                            self.word_to_slots[w] = [s for s in slots if s != sj]
                            del self.slot_to_word[sj]
                            merges.append((w, si, sj, sim))
                            merged = True
                            break
                    if merged:
                        break
        return merges
